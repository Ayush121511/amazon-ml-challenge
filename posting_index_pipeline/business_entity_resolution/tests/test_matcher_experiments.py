import argparse
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from matcher_experiments import decision_grid, reference_weights
from preflight_v4_caches import inspect_dense, inspect_reverse
from submission_pipeline import Scorer, decide, macro_score, merge_reverse, stage_train, load_scorer


class MatcherExperimentTests(unittest.TestCase):
    def test_reference_weight_mass_and_partial_weighting(self):
        refs = np.array(['a', 'b', 'b', 'b'])
        full = reference_weights(refs, 1)
        self.assertAlmostEqual(float(full.mean()), 1)
        self.assertAlmostEqual(float(full[refs == 'a'].sum()), float(full[refs == 'b'].sum()))
        np.testing.assert_equal(reference_weights(refs, 0), np.ones(4))
        partial = reference_weights(refs, 0.5)
        self.assertGreater(partial[0], partial[1])
        self.assertLess(partial[0] / partial[1], full[0] / full[1])

    def test_fast_grid_matches_original_including_ties_and_missing_gold(self):
        rng = np.random.default_rng(42)
        refs = [f'r{i}' for i in range(20)]
        ri = np.repeat(np.arange(19), rng.integers(1, 20, size=19))
        targets = np.array([f't{i}' for i in range(len(ri))], dtype=object)
        prob = np.round(rng.random(len(ri)), 2).astype(np.float32)
        gold = {r: set() for r in refs}
        for i, t, yes in zip(ri, targets, rng.random(len(ri)) < .3):
            if yes:
                gold[refs[i]].add(t)
        gold['r2'].add('unretrieved')
        gold['r19'].add('no_candidates')
        rules = [(r, t) for r in ('threshold', 'expected') for t in (0., .05, .1, .25, .5, .7, .75, 1.)]
        fast = decision_grid(refs, ri, targets, prob, gold, rules)
        for r, t in rules:
            original = macro_score(refs, ri, targets, decide(ri, prob, r, t, len(refs)), gold)
            self.assertAlmostEqual(fast[(r, t)], original, places=12)

    def test_reverse_optional_route_scores_preserve_candidate_set(self):
        c = {'ref': np.array([0]), 'ord': np.array([3]), 'top': {}}
        rev = {'ref': np.array([0, 1]), 't_ord': np.array([3, 9]),
               'rank': np.array([1, 2]), 'rrf': np.array([.1, .05]),
               'score_name': np.array([.9, .2]), 'score_address': np.array([.8, .3]),
               'score_anchor': np.array([.5, .6])}
        old = merge_reverse(c, rev)
        updated = merge_reverse(c, {**rev, 'score_combined': np.array([.95, .4])})
        np.testing.assert_equal(old['ref'], updated['ref'])
        np.testing.assert_equal(old['ord'], updated['ord'])
        np.testing.assert_allclose(updated['rev_score_combined'], [.95, .4])
        np.testing.assert_equal(old['rev_score_combined'], [0, 0])
        self.assertNotIn('rev_rank_combined', updated)

    def test_cache_preflight_distinguishes_missing_zero_and_useful_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.npz'
            np.savez(path, rank=[1, 2])
            self.assertFalse(inspect_reverse(path)['combined_score_present'])
            np.savez(path, score_combined=np.array([0, 0], np.float16))
            self.assertEqual(inspect_reverse(path)['combined_score_nonzero'], 0)
            np.savez(path, score_combined=np.array([0, .8], np.float16))
            self.assertEqual(inspect_reverse(path)['combined_score_nonzero'], 1)
            np.savez(path, rev_score=np.array([[.9, .8], [.7, .2]], np.float32))
            dense = inspect_dense(path)
            self.assertTrue(dense['target_margin_available_without_embedding_search'])
            self.assertAlmostEqual(dense['top12_margin_mean'], .3, places=6)

    def test_weighted_xgboost_dev_only_trains_reloads_and_never_scores_holdout(self):
        try:
            import xgboost
        except ImportError:
            self.skipTest('XGBoost not installed')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = root / 'pairs'
            pairs.mkdir()
            gold, countries, lines, x, y = {}, {}, [], [], []
            rng = np.random.default_rng(11)
            for i in range(300):
                sid = f'S1-{i}'
                gold[sid] = [f'S2-{i}-0'] if i % 5 else []
                countries[sid] = 'US' if i % 2 else 'India'
                for rank in range(2 + i % 4):
                    tid = f'S2-{i}-{rank}'
                    label = int(tid in gold[sid])
                    x.append([label + rng.normal(0, .2), rank + 1])
                    y.append(label)
                    lines.append(f'{sid}\t{tid}\n')
            np.save(pairs / 'x.npy', np.array(x, np.float32))
            np.save(pairs / 'y.npy', np.array(y, np.int8))
            with gzip.open(pairs / 'pairs.tsv.gz', 'wt') as f:
                f.writelines(lines)
            for name, value in [('gold', gold), ('countries', countries), ('report', {
                    'features': ['signal', 'fused_rank'], 'candidate_recall': {}, 'parameters': {'keep': 5}})]:
                (pairs / f'{name}.json').write_text(json.dumps(value))
            out = root / 'model'
            sizes = []
            original_predict = Scorer.predict
            def record_predict(scorer, matrix):
                sizes.append(len(matrix))
                return original_predict(scorer, matrix)
            with patch.object(Scorer, 'predict', record_predict):
                stage_train(argparse.Namespace(pairs=pairs, output=out, rounds=30, threads=1,
                    max_rank=5, train_refs=0, num_leaves=15, learning_rate=.1,
                    early_stopping=5, backend='xgboost', reference_weight_power=1.,
                    threshold_step=.01, dev_only=True))
            report = json.loads((out / 'report.json').read_text())
            self.assertIsNone(report['macro_f05_holdout'])
            self.assertFalse(report['holdout_evaluated'])
            self.assertEqual(report['holdout_by_country'], {})
            self.assertEqual(report['reference_weight_power'], 1.)
            self.assertEqual(sizes, [report['pairs']['dev']])
            self.assertGreater(report['macro_f05_dev'], .9)
            self.assertTrue(np.isfinite(load_scorer(out, 1).predict(np.asarray(x[:3], np.float32))).all())


if __name__ == '__main__':
    unittest.main()
