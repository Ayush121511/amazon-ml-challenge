import argparse
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posting_index import build, CountryIndex, load_manifest
import gzip
from submission_pipeline import (FEATURE_NAMES, TargetTable, candidates, decide, f05, features,
                                 macro_score, merge_reverse, stage_merge, stage_predict, stage_train)
from test_posting_index import FILLER, ref, write_source

try:
    import lightgbm
except Exception:  # Missing, or blocked by OS application control.
    lightgbm = None


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls.tmp.name)
        s2 = [('S2-typo', 'Payne Enterpires', '3315 Fremont St, Peoria, IL', 'US'),
              ('S2-addr', 'Drexkor', '85 Wayne Avenue, Ticonderoga, NY', 'US'),
              ('S2-fr', 'Boulangerie Martin', '14 Rue de Rivoli, Paris', 'France')]
        s3 = [('S3-same', 'Payne Enterprises', '3315 Fremont Street, Peoria, IL', 'US')]
        s3 += [(f'S3-f{i}', n, a, 'US') for i, (n, a) in enumerate(FILLER)]
        write_source(root / 'test_source2.tsv.gz', s2)
        write_source(root / 'test_source3.tsv.gz', s3)
        write_source(root / 'test_source1.tsv.gz', [
            ('S1-1', 'Payne Enterprises', '3315 Fremont St, Peoria, IL', 'US'),
            ('S1-2', 'Nothing Alike Here', '1 Nowhere Road, Nome, AK', 'US'),
            ('S1-3', 'Boulangerie Martin', '14 Rue de Rivoli, Paris', 'France'),
            ('S1-4', 'Kyoto Tea House', '3 Gion, Kyoto', 'Japan'),
            ('S1-5', 'Payne Enterprises', '3315 Fremont St, Peoria, IL', 'US')])
        cls.index_root = root / 'index'
        build([root / 'test_source2.tsv.gz', root / 'test_source3.tsv.gz'], cls.index_root,
              workers=1, chunk_rows=3, max_df_fraction=1.0, min_max_df=1000)
        cls.manifest = load_manifest(cls.index_root)
        cls.index = CountryIndex(cls.index_root, 'US', cls.manifest)
        cls.table = TargetTable(root, 'test', 'US', cls.index.ids)
        cls.rows = [ref('Payne Enterprises', '3315 Fremont St, Peoria, IL', entity_id='S1-1'),
                    ref('Nothing Alike Here', '1 Nowhere Road, Nome, AK', entity_id='S1-2')]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_candidates_are_grouped_capped_and_ranked(self):
        c = candidates(self.index, self.rows, top_k=10, top_terms=64, keep=3, threads=1)
        self.assertTrue(np.all(np.diff(c['ref']) >= 0))
        for i in set(c['ref'].tolist()):
            ranks = c['fused_rank'][c['ref'] == i]
            self.assertEqual(ranks.tolist(), list(range(1, len(ranks) + 1)))
            self.assertLessEqual(len(ranks), 3)
        top = self.table.fields['entity_id'][c['ord'][(c['ref'] == 0) & (c['fused_rank'] <= 2)]]
        self.assertTrue({'S3-same', 'S2-typo'} & set(top))

    def test_features_are_finite_and_consistent(self):
        c = candidates(self.index, self.rows, top_k=10, top_terms=64, keep=5, threads=1)
        x = features(c, self.rows, self.table, threads=1)
        self.assertEqual(x.shape, (len(c['ref']), len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(x).all())
        exact = x[:, FEATURE_NAMES.index('name_exact')]
        same = self.table.fields['entity_id'][c['ord']] == 'S3-same'
        self.assertEqual(exact[same & (c['ref'] == 0)].tolist(), [1.0])
        # A route's top-1 candidate has zero gap to itself.
        top1 = c['rank_name'] == 1
        self.assertTrue(np.allclose(x[top1, FEATURE_NAMES.index('gap_name')], 0))

    def test_merge_reverse_unions_and_marks_sources(self):
        c = candidates(self.index, self.rows, top_k=10, top_terms=64, keep=2, threads=1)
        rev = {'ref': np.array([1, 0]), 't_ord': np.array([0, int(c['ord'][c['ref'] == 0][0])]),
               'rank': np.array([1, 3]), 'rrf': np.array([.5, .1], np.float32),
               'score_name': np.array([.9, .2], np.float16), 'score_address': np.zeros(2, np.float16),
               'score_anchor': np.zeros(2, np.float16)}
        m = merge_reverse(c, rev)
        keys = list(zip(m['ref'].tolist(), m['ord'].tolist()))
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(keys), len(set(zip(c['ref'].tolist(), c['ord'].tolist())) | {(1, 0)}))
        both = (m['ref'] == 0) & (m['ord'] == rev['t_ord'][1])
        self.assertEqual((m['fused_rank'][both] > 0).tolist(), [True])
        self.assertEqual(m['rev_rank'][both].tolist(), [3.0])
        rev_only = (m['ref'] == 1) & (m['ord'] == 0)
        if not ((c['ref'] == 1) & (c['ord'] == 0)).any():
            self.assertEqual(m['fused_rank'][rev_only].tolist(), [0])
        x = features(m, self.rows, self.table, threads=1)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x[rev_only, FEATURE_NAMES.index('rev_top1')].tolist(), [1.0])

    def test_group_features_pick_best_candidate_as_anchor(self):
        c = candidates(self.index, self.rows, top_k=10, top_terms=64, keep=6, threads=1)
        x = features(c, self.rows, self.table, threads=1)
        col = lambda name: x[:, FEATURE_NAMES.index(name)]
        ids = self.table.fields['entity_id'][c['ord']]
        ref0 = c['ref'] == 0
        anchor = ids[ref0 & (col('is_anchor1') == 1)]
        self.assertEqual(len(anchor), 1)
        self.assertIn(anchor[0], {'S3-same', 'S2-typo'})
        # The other copy of the same business resembles the anchor closely.
        twin = ref0 & np.isin(ids, ['S3-same', 'S2-typo']) & (col('is_anchor1') == 0)
        self.assertGreater(col('anchor1_address_folded')[twin].max(), 0.6)
        self.assertEqual(sorted(col('proxy_rank')[ref0].tolist()), list(range(1, int(ref0.sum()) + 1)))
        self.assertTrue(np.all(col('proxy_gap') <= 1e-6))

    def test_dense_candidates_and_embedding_features(self):
        from posting_index import country_dir
        from submission_pipeline import Dense, merge_dense
        root = self.root / 'dense' / 'test'
        root.mkdir(parents=True, exist_ok=True)
        name = country_dir('US')
        ids = self.index.ids
        rng = np.random.default_rng(0)
        targets = rng.normal(size=(len(ids), 8)).astype(np.float32)
        targets /= np.linalg.norm(targets, axis=1, keepdims=True)
        same = int(np.flatnonzero(ids == b'S3-same')[0])
        refs = np.stack([targets[same], -targets[same]]).astype(np.float16)  # S1-1 ~ S3-same
        np.save(root / f'targets_{name}.npy', targets.astype(np.float16))
        np.save(root / f'targets_{name}_ids.npy', ids)
        np.save(root / 's1_vectors.npy', refs)
        np.save(root / 's1_ids.npy', np.array([b'S1-1', b'S1-2']))
        np.save(root / 's1_country.npy', np.array([b'US', b'US']))
        k = 3
        fwd_ord = np.argsort(-(refs.astype(np.float32) @ targets.T), axis=1)[:, :k].astype(np.int32)
        rev_ref = np.zeros((len(ids), 1), dtype=np.int32)
        np.savez(root / f'dense_{name}.npz', ref_ids=np.array([b'S1-1', b'S1-2']),
                 fwd_score=np.ones((2, k), np.float16), fwd_ord=fwd_ord,
                 rev_score=np.ones((len(ids), 1), np.float16), rev_ref=rev_ref)
        dense = Dense(self.root / 'dense', 'test', 'US', ids)
        c = candidates(self.index, self.rows, top_k=10, top_terms=64, keep=2, threads=1)
        fwd, rev, vecs = dense.lookup(self.rows)
        m = merge_dense(c, fwd, rev)
        keys = set(zip(m['ref'].tolist(), m['ord'].tolist()))
        self.assertIn((0, same), keys)
        self.assertEqual(len(rev['ref']), len(ids))  # every target listed S1-1 as its top reference
        x = features(m, self.rows, self.table, threads=1, ref_vecs=vecs, target_vecs=dense.targets)
        col = lambda n: x[:, FEATURE_NAMES.index(n)]
        pick = (m['ref'] == 0) & (m['ord'] == same)
        self.assertAlmostEqual(float(col('emb_cos')[pick][0]), 1.0, places=2)
        self.assertEqual(float(col('emb_rank')[pick][0]), 1.0)
        self.assertTrue(np.isfinite(x).all())

    def test_f05_edge_cases(self):
        self.assertEqual(f05(set(), set()), 1.0)
        self.assertEqual(f05(set(), {'a'}), 0.0)
        self.assertEqual(f05({'a'}, set()), 0.0)
        self.assertAlmostEqual(f05({'a', 'b'}, {'a'}), 1.25 * 0.5 / (0.25 + 0.5))

    def test_decision_rules(self):
        ref_idx = np.array([0, 0, 1, 1, 2])
        prob = np.array([0.9, 0.2, 0.1, 0.05, 0.6])
        self.assertEqual(decide(ref_idx, prob, 'threshold', 0.5, 3).tolist(),
                         [True, False, False, False, True])
        expected = decide(ref_idx, prob, 'expected', 0.05, 3)
        self.assertTrue(expected[0] and not expected[2] and not expected[3])

    def test_macro_score_uses_reference_ids(self):
        gold = {'S1-a': {'S2-1'}, 'S1-b': set()}
        score = macro_score(['S1-a', 'S1-b'], np.array([0, 1]), np.array(['S2-1', 'S2-9'], dtype=object),
                            np.array([True, False]), gold)
        self.assertEqual(score, 1.0)

    @unittest.skipIf(lightgbm is None, 'lightgbm unavailable')
    def test_train_stage_end_to_end(self):
        rng = np.random.default_rng(0)
        pairs = self.root / 'pairs_train'
        pairs.mkdir(exist_ok=True)
        names = [f'f{i}' for i in range(3)]
        refs = [f'S1-{i}' for i in range(300)]
        rows, labels, gold = [], [], {}
        for i, sid in enumerate(refs):
            gold[sid] = [f'S2-{i}'] if i % 10 else []
            for rank, t in enumerate([f'S2-{i}', f'S3-{i}'], 1):
                y = int(t in gold[sid])
                rows.append((sid, t, [y + rng.normal(0, .3), rng.normal(), rank], y))
        np.save(pairs / 'x.npy', np.array([r[2] for r in rows], dtype=np.float32))
        np.save(pairs / 'y.npy', np.array([r[3] for r in rows], dtype=np.int8))
        with gzip.open(pairs / 'pairs.tsv.gz', 'wt', encoding='utf-8') as f:
            f.writelines(f'{r[0]}\t{r[1]}\n' for r in rows)
        (pairs / 'gold.json').write_text(json.dumps(gold))
        (pairs / 'report.json').write_text(json.dumps(
            {'features': ['score', 'noise', 'fused_rank'], 'candidate_recall': {}, 'parameters': {'keep': 2}}))
        out = self.root / 'model_train'
        stage_train(argparse.Namespace(pairs=pairs, output=out, rounds=20, threads=1, max_rank=2, train_refs=0,
                                       num_leaves=63, learning_rate=0.05, min_data_in_leaf=100,
                                       early_stopping=50))
        # A smaller training subset leaves dev/holdout untouched.
        small = self.root / 'model_train_small'
        stage_train(argparse.Namespace(pairs=pairs, output=small, rounds=20, threads=1, max_rank=2,
                                       train_refs=50, num_leaves=15, learning_rate=0.1,
                                       min_data_in_leaf=10, early_stopping=10))
        # XGBoost backend (GPU on Padum, CPU here): same interface, model reloads for scoring.
        xgb_out = self.root / 'model_train_xgb'
        stage_train(argparse.Namespace(pairs=pairs, output=xgb_out, rounds=30, threads=1, max_rank=2,
                                       train_refs=0, num_leaves=15, learning_rate=0.1,
                                       min_data_in_leaf=10, early_stopping=10, backend='xgboost'))
        xgb_report = json.loads((xgb_out / 'report.json').read_text())
        self.assertEqual(xgb_report['backend'], 'xgboost')
        self.assertGreater(xgb_report['macro_f05_holdout'], 0.8)
        from submission_pipeline import load_scorer
        probs = load_scorer(xgb_out, 1).predict(np.load(pairs / 'x.npy')[:5])
        self.assertEqual(probs.shape, (5,))
        small_report = json.loads((small / 'report.json').read_text())
        self.assertEqual(small_report['train_refs_used'], 50)
        self.assertEqual(small_report['references']['holdout'],
                         json.loads((out / 'report.json').read_text())['references']['holdout'])
        report = json.loads((out / 'report.json').read_text())
        self.assertGreater(report['macro_f05_holdout'], 0.8)
        self.assertEqual(report['keep'], 2)
        self.assertTrue((out / 'model.txt').exists())

    @unittest.skipIf(lightgbm is None, 'lightgbm unavailable')
    def test_predict_writes_valid_tsvs_for_every_reference(self):
        c = candidates(self.index, self.rows, top_k=10, top_terms=64, keep=5, threads=1)
        x = features(c, self.rows, self.table, threads=1)
        y = (self.table.fields['entity_id'][c['ord']] == 'S3-same').astype(int)
        x, y = np.vstack([x] * 20), np.concatenate([y] * 20)
        booster = lightgbm.train({'objective': 'binary', 'min_data_in_leaf': 1, 'verbose': -1},
                                 lightgbm.Dataset(x, y), num_boost_round=5)
        model = self.root / 'model'
        model.mkdir(exist_ok=True)
        booster.save_model(str(model / 'model.txt'))
        (model / 'report.json').write_text(json.dumps(
            {'decision': {'rule': 'threshold', 'threshold': 0.5}, 'features': list(FEATURE_NAMES),
             'keep': 5}))
        out = self.root / 'out'
        stage_predict(argparse.Namespace(data=self.root, index=self.index_root, output=out,
                                         split='test', model=model, limit=0, keep=0, top_k=10,
                                         top_terms=64, batch=2, threads=1, sample=0, shard='',
                                         sample_seed='s', exclusive=True, reverse=None, ref_index=None,
                                         country_threshold=''))
        tables = {}
        for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
            with open(out / name, encoding='utf-8', newline='') as f:
                rows = list(csv.reader(f, delimiter='\t'))
            self.assertEqual(rows[0][0], 'source1_entity_id')
            tables[name] = {r[0]: set(filter(None, r[1].split(','))) for r in rows[1:]}
            self.assertEqual(sorted(tables[name]), ['S1-1', 'S1-2', 'S1-3', 'S1-4', 'S1-5'])
        for sid, matched in tables['matching_results.tsv'].items():
            self.assertLessEqual(matched, tables['candidate_pairs.tsv'][sid])
        self.assertIn('S2-fr', tables['candidate_pairs.tsv']['S1-3'])  # France, unseen in training
        self.assertEqual(tables['candidate_pairs.tsv']['S1-4'], set())  # No index for Japan
        # S1-1 and S1-5 are identical; exclusivity lets at most one of them keep each target.
        matched = [t for ts in tables['matching_results.tsv'].values() for t in ts]
        self.assertEqual(len(matched), len(set(matched)))
        self.assertIn('S3-same', matched)
        # Two shards + merge must reproduce the single-run files exactly.
        shards = []
        for i in range(2):
            shard = self.root / f'shard{i}'
            stage_predict(argparse.Namespace(data=self.root, index=self.index_root, output=shard,
                                             split='test', model=model, limit=0, keep=0, top_k=10,
                                             top_terms=64, batch=2, threads=1, sample=0, shard=f'{i}/2',
                                             sample_seed='s', exclusive=True, reverse=None, ref_index=None,
                                         country_threshold=''))
            shards.append(shard)
        merged = self.root / 'merged'
        stage_merge(argparse.Namespace(data=self.root, split='test', model=model, shards=shards,
                                       output=merged, exclusive=True, country_threshold=''))
        for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
            self.assertEqual((merged / name).read_text(encoding='utf-8'), (out / name).read_text(encoding='utf-8'))
        # Variants: one pass, stage-1 pruning (top-2 and adaptive), one output folder each.
        import xgboost as xgb
        from prune_analysis import STAGE1_FEATURES
        cols = [FEATURE_NAMES.index(f) for f in STAGE1_FEATURES]
        stage1 = xgb.train({'objective': 'binary:logistic', 'max_depth': 2, 'verbosity': 0},
                           xgb.DMatrix(x[:, cols], y), num_boost_round=5)
        stage1.save_model(str(self.root / 'stage1.json'))
        spec = self.root / 'variants.json'
        spec.write_text(json.dumps({'stage1': str(self.root / 'stage1.json'), 'variants': [
            {'name': 'top2', 'rule': 'top', 'k': 2, 'model': str(model)},
            {'name': 'adaptive', 'rule': 'adaptive', 't': 0.5, 'min': 1, 'max': 3, 'model': str(model)}]}))
        var_out = self.root / 'out_variants'
        stage_predict(argparse.Namespace(data=self.root, index=self.index_root, output=var_out,
                                         split='test', model=model, limit=0, keep=0, top_k=10,
                                         top_terms=64, batch=2, threads=1, sample=0, shard='',
                                         sample_seed='s', exclusive=True, reverse=None, ref_index=None,
                                         country_threshold='', variants=spec))
        for name, cap in (('top2', 2), ('adaptive', 3)):
            with open(var_out / name / 'candidate_pairs.tsv', encoding='utf-8', newline='') as f:
                cands = {r[0]: set(filter(None, r[1].split(','))) for r in list(csv.reader(f, delimiter='	'))[1:]}
            with open(var_out / name / 'matching_results.tsv', encoding='utf-8', newline='') as f:
                found = {r[0]: set(filter(None, r[1].split(','))) for r in list(csv.reader(f, delimiter='	'))[1:]}
            self.assertEqual(sorted(cands), ['S1-1', 'S1-2', 'S1-3', 'S1-4', 'S1-5'])
            self.assertTrue(all(len(v) <= cap for v in cands.values()))
            self.assertTrue(all(found[k] <= cands[k] for k in found))
        # A per-country cutoff above any probability removes that country's matches only.
        strict = self.root / 'merged_strict'
        stage_merge(argparse.Namespace(data=self.root, split='test', model=model, shards=shards,
                                       output=strict, exclusive=True, country_threshold='US=1.01'))
        with open(strict / 'matching_results.tsv', encoding='utf-8') as f:
            rows = {r[0]: r[1] for r in csv.reader(f, delimiter='	')}
        self.assertEqual(rows['S1-1'] + rows['S1-5'], '')
        self.assertEqual(rows['S1-3'], tables['matching_results.tsv'] and ','.join(sorted(tables['matching_results.tsv']['S1-3'])))
        report = json.loads((strict / 'report.json').read_text())
        self.assertEqual(report['stats']['country_thresholds']['US'], 1.01)


if __name__ == '__main__':
    unittest.main()
