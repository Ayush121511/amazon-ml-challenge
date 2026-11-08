"""Pair-scoring and decision-model experiments.

Each command is one analysis script. Run `python model_experiments.py <command> [args]`; `python model_experiments.py` lists the commands.
"""
import sys


def cmd_analyze_matcher_loss():
    """Where does macro F0.5 go? Splits holdout loss into blocking (gold never a candidate)
and matcher (wrong decisions among candidates), for a trained model."""
    import argparse
    import gzip
    import hashlib
    import json
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np

    from submission_pipeline import decide, f05


    def main():
        p = argparse.ArgumentParser(description=cmd_analyze_matcher_loss.__doc__)
        p.add_argument('--pairs', type=Path, required=True)
        p.add_argument('--model', type=Path, required=True)
        p.add_argument('--partition', default='holdout')
        p.add_argument('--output', type=Path, required=True)
        a = p.parse_args()
        names = json.loads((a.pairs / 'report.json').read_text())['features']
        model_report = json.loads((a.model / 'report.json').read_text())
        x, y = np.load(a.pairs / 'x.npy'), np.load(a.pairs / 'y.npy')
        pairs = [line.rstrip('\n').split('\t') for line in gzip.open(a.pairs / 'pairs.tsv.gz', 'rt')]
        pair_ref = np.array([q[0] for q in pairs], dtype=object)
        pair_target = np.array([q[1] for q in pairs], dtype=object)
        gold = {k: set(v) for k, v in json.loads((a.pairs / 'gold.json').read_text()).items()}
        owners = Counter(t for v in gold.values() for t in v)
        if max(owners.values(), default=1) > 1:
            raise ValueError('Gold targets shared between references: component split needed')

        def part(s):  # Same assignment as stage_train when every component is a single reference.
            h = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 100
            return 'train' if h < 70 else 'dev' if h < 85 else 'holdout'
        refs = sorted(s for s in gold if part(s) == a.partition)
        position = {s: i for i, s in enumerate(refs)}
        keep = model_report['keep']
        rank = x[:, names.index('fused_rank')]
        idx = np.flatnonzero(np.array([part(s) == a.partition for s in pair_ref]) & (rank <= keep))
        from submission_pipeline import load_scorer
        booster = load_scorer(a.model, 8)
        columns = [names.index(n) for n in model_report['features']]
        prob = booster.predict(x[idx][:, columns])
        ref_idx = np.array([position[s] for s in pair_ref[idx]])
        chosen = decide(ref_idx, prob, model_report['decision']['rule'], model_report['decision']['threshold'], len(refs))
        pred, cand = defaultdict(set), defaultdict(set)
        for i, t, c in zip(ref_idx, pair_target[idx], chosen):
            cand[i].add(t)
            if c:
                pred[i].add(t)
        per_ref = []
        for i, s in enumerate(refs):
            g, pr, cs = gold[s], pred.get(i, set()), cand.get(i, set())
            actual, oracle = f05(g, pr), f05(g, g & cs)
            fp, fn_block, fn_match = len(pr - g), len(g - cs), len((g & cs) - pr)
            kind = ('perfect' if actual == 1 else
                    'singleton_false_match' if not g else
                    'blocking_only' if fn_block and not fp and not fn_match else
                    'false_positive_only' if fp and not fn_block and not fn_match else
                    'missed_candidate_only' if fn_match and not fp and not fn_block else 'mixed')
            per_ref.append((s, actual, oracle, kind, len(g), fp, fn_block, fn_match))
        total = len(refs)
        summary = {'references': total, 'keep': keep, 'decision': model_report['decision'],
                   'macro_f05': sum(r[1] for r in per_ref) / total,
                   'oracle_matcher_macro_f05': sum(r[2] for r in per_ref) / total,
                   'loss_by_kind': {}, 'false_positive_links': sum(r[5] for r in per_ref),
                   'blocking_missed_links': sum(r[6] for r in per_ref),
                   'matcher_missed_links': sum(r[7] for r in per_ref),
                   'gold_links': sum(r[4] for r in per_ref),
                   'loss_by_gold_count': {}}
        by_kind, by_count = defaultdict(float), defaultdict(float)
        for s, actual, oracle, kind, n, *_ in per_ref:
            by_kind[kind] += (1 - actual) / total
            by_count[min(n, 8)] += (1 - actual) / total
        summary['loss_by_kind'] = dict(sorted(by_kind.items(), key=lambda kv: -kv[1]))
        summary['loss_by_gold_count'] = {str(k): v for k, v in sorted(by_count.items())}
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / 'loss_summary.json').write_text(json.dumps(summary, indent=2))
        worst = sorted(per_ref, key=lambda r: r[1])[:300]
        (a.output / 'worst_refs.json').write_text(json.dumps(
            [{'ref': s, 'f05': f, 'kind': k, 'gold': sorted(gold[s]), 'pred': sorted(pred.get(position[s], set())),
              'fp': fp, 'fn_block': fb, 'fn_match': fm} for s, f, o, k, n, fp, fb, fm in worst], indent=1))
        print(json.dumps(summary, indent=2), flush=True)


    if __name__ == '__main__':
        main()



def cmd_calibrate_probs():
    """Probability calibration of the feature matcher, with the decision rule re-chosen on dev."""
    import gzip
    import hashlib
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    sys.path.insert(0, str(Path(__file__).parent))
    from submission_pipeline import decide, f05, macro_score


    def load_pairs(d):
        pairs = []
        with gzip.open(d / 'pairs.tsv.gz', 'rt') as f:
            for line in f:
                ref, target = line.rstrip('\n').split('\t')
                pairs.append((ref, target))
        return pairs


    def partition_of(gold):
        parent = {s: s for s in gold}

        def find(s):
            while parent[s] != s:
                parent[s] = parent[parent[s]]
                s = parent[s]
            return s
        owner = {}
        for s, targets in gold.items():
            for t in targets:
                if t in owner:
                    parent[find(s)] = find(owner[t])
                owner[t] = s

        def part(s):
            h = int(hashlib.sha256(find(s).encode()).hexdigest()[:8], 16) % 100
            return 'train' if h < 70 else 'dev' if h < 85 else 'holdout'
        return {s: part(s) for s in gold}


    def main():
        root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'scratch/vishal_v4_artifacts'
        pairs_dir = root / 'pairs'
        model_dir = root / 'model_xgb'

        x = np.load(pairs_dir / 'x.npy')
        y = np.load(pairs_dir / 'y.npy')
        pairs = load_pairs(pairs_dir)
        pair_ref = np.array([p[0] for p in pairs], dtype=object)
        pair_target = np.array([p[1] for p in pairs], dtype=object)
        gold = {k: set(v) for k, v in json.loads((pairs_dir / 'gold.json').read_text()).items()}
        countries = json.loads((pairs_dir / 'countries.json').read_text())
        report = json.loads((model_dir / 'report.json').read_text())
        names = report['features']
        best_rule, best_t = report['decision']['rule'], report['decision']['threshold']

        nthread = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        booster = xgb.Booster()
        booster.load_model(str(model_dir / 'model.json'))
        booster.set_param({'device': 'cpu', 'nthread': nthread})
        print(f'predicting {len(x)} rows, nthread={nthread}...', flush=True)
        prob = np.asarray(booster.inplace_predict(x, iteration_range=(0, report['best_iteration'] + 1)),
                           dtype=np.float64)
        print('predict done', flush=True)

        partition = partition_of(gold)
        pair_part = np.array([partition[s] for s in pair_ref], dtype=object)
        max_rank = report.get('keep', 0)
        in_range = (x[:, names.index('fused_rank')] <= max_rank) if max_rank else np.ones(len(y), bool)
        m = {p: (pair_part == p) & in_range for p in ('dev', 'holdout')}
        pair_country = np.array([countries.get(s, 'UNK') for s in pair_ref], dtype=object)

        def run(prob_arr, label):
            idx_dev = np.flatnonzero(m['dev'])
            idx_hold = np.flatnonzero(m['holdout'])
            refs_dev = sorted(s for s in gold if partition[s] == 'dev')
            pos_dev = {s: i for i, s in enumerate(refs_dev)}
            ref_idx_dev = np.array([pos_dev[s] for s in pair_ref[idx_dev]])
            refs_hold = sorted(s for s in gold if partition[s] == 'holdout')
            pos_hold = {s: i for i, s in enumerate(refs_hold)}
            ref_idx_hold = np.array([pos_hold[s] for s in pair_ref[idx_hold]])

            grid = [(rule, round(float(t), 3)) for rule in ('threshold', 'expected') for t in np.arange(0.05, 0.96, 0.05)]
            dev_scores = {}
            for rule, t in grid:
                chosen = decide(ref_idx_dev, prob_arr[idx_dev], rule, t, len(refs_dev))
                dev_scores[(rule, t)] = macro_score(refs_dev, ref_idx_dev, pair_target[idx_dev], chosen, gold)
            best = max(dev_scores, key=lambda k: (dev_scores[k], k[1]))
            chosen_hold = decide(ref_idx_hold, prob_arr[idx_hold], best[0], best[1], len(refs_hold))
            holdout_f05 = macro_score(refs_hold, ref_idx_hold, pair_target[idx_hold], chosen_hold, gold)
            n_matched = int(chosen_hold.sum())
            avg_per_entity = n_matched / len(refs_hold)
            print(f'[{label}] best_dev={best} dev_f05={dev_scores[best]:.6f} '
                  f'holdout_f05={holdout_f05:.6f} avg_matches_per_entity={avg_per_entity:.3f}')
            return holdout_f05, avg_per_entity

        print('=== baseline (uncalibrated) ===')
        run(prob, 'uncalibrated')

        print('=== isotonic calibration (global, fit on dev) ===')
        idx_dev = np.flatnonzero(m['dev'])
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(prob[idx_dev], y[idx_dev])
        prob_cal_global = iso.transform(prob)
        run(prob_cal_global, 'isotonic-global')

        print('=== isotonic calibration (per-country, fit on dev) ===')
        prob_cal_country = prob.copy()
        for c in sorted(set(pair_country[idx_dev])):
            mask_dev_c = idx_dev[pair_country[idx_dev] == c]
            if len(mask_dev_c) < 200:
                continue
            iso_c = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            iso_c.fit(prob[mask_dev_c], y[mask_dev_c])
            mask_all_c = pair_country == c
            prob_cal_country[mask_all_c] = iso_c.transform(prob[mask_all_c])
            print(f'  fit isotonic for country={c} on {len(mask_dev_c)} dev pairs')
        run(prob_cal_country, 'isotonic-per-country')


    if __name__ == '__main__':
        main()



def cmd_ce_report():
    """Holdout report for the cross-encoders: scores and F0.5 per model."""
    import gzip
    import json
    import sys
    import time
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import torch
    import xgboost as xgb
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).parent))
    from submission_pipeline import decide, f05
    import importlib.util

    HOME = Path.home()
    V4 = HOME / 'scratch/vishal_v4_artifacts'
    VISHAL = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
    CE = HOME / 'scratch/amazon_ml_2026/artifacts/cross_encoder_v1'
    EXP = CE / 'export'


    def main():
        EXP.mkdir(parents=True, exist_ok=True)
        spec = importlib.util.spec_from_file_location('cem', Path(__file__).parent / 'cross_encoder_exp.py')
        cem = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cem)
        gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
        partition = cem.partition_of(gold)
        mask = np.load(VISHAL / 'prune_v1/mask_top10.npy')
        row_index = np.flatnonzero(mask)
        refs, tgts = [], []
        with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t')
                    refs.append(r); tgts.append(t)
        refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
        part = np.array([partition[r] for r in refs], dtype=object)
        y = np.load(V4 / 'pairs/y.npy')[row_index].astype(np.float32)
        ce = np.load(CE / 'ce_logit.npy')
        R = {'row_alignment': 'arrays have one entry per True position of prune_v1/mask_top10.npy, in pairs.tsv.gz order; '
                              'row_index.npy gives the 0-based line index into pairs.tsv.gz / x.npy / y.npy'}
        R['folds'] = {}
        for p in ('trainA', 'trainB', 'dev', 'holdout'):
            m = part == p
            R['folds'][p] = {'references': int(len({r for r in refs[m]})), 'refs_in_gold': int(sum(v == p for v in partition.values())),
                             'pairs': int(m.sum()), 'positive': int(y[m].sum()), 'negative': int((y[m] == 0).sum()),
                             'ce_scored': int((~np.isnan(ce[m])).sum())}
        R['leakage_check'] = {'ce_trained_only_on': 'trainA',
                              'trainA_rows_with_ce_score (must be 0)': int((~np.isnan(ce[part == 'trainA'])).sum()),
                              'nonA_rows_without_ce_score (must be 0)': int(np.isnan(ce[part != 'trainA']).sum())}
        np.save(EXP / 'row_index.npy', row_index)
        np.save(EXP / 'fold.npy', part.astype('U8'))
        np.save(EXP / 'ce_e5small_foldA_logit.npy', ce)

        x = np.asarray(np.load(V4 / 'pairs/x.npy', mmap_mode='r')[row_index])
        params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                      grow_policy='lossguide', max_depth=0, max_leaves=255, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=1.0, max_bin=256, seed=2026)
        tb, dv = part == 'trainB', part == 'dev'
        probs = {'cross_encoder_alone': 1 / (1 + np.exp(-np.nan_to_num(ce, nan=-20).astype(np.float64)))}
        for name, F in (('xgb_baseline_foldB', x), ('xgb_plus_ce_foldB', np.column_stack([x, ce]))):
            dtr = xgb.QuantileDMatrix(F[tb], y[tb])
            bst = xgb.train(params, dtr, 4000, evals=[(xgb.QuantileDMatrix(F[dv], y[dv], ref=dtr), 'dev')],
                            early_stopping_rounds=100, verbose_eval=False)
            probs[name] = np.asarray(bst.inplace_predict(F, iteration_range=(0, bst.best_iteration + 1)), dtype=np.float64)
        np.save(EXP / 'xgb_plus_ce_foldB_prob.npy', probs['xgb_plus_ce_foldB'].astype(np.float32))

        prefs = {p: sorted(s for s in gold if partition[s] == p) for p in ('dev', 'holdout')}
        idx = {p: np.flatnonzero(part == p) for p in prefs}
        rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in prefs}

        def evaluate(p, prob, ru, t):
            c = decide(rix[p], prob[idx[p]], ru, t, len(prefs[p]))
            pred = defaultdict(set)
            for i, tg in zip(rix[p][c], tgts[idx[p]][c]):
                pred[i].add(tg)
            cand = defaultdict(set)
            for i, tg in zip(rix[p], tgts[idx[p]]):
                cand[i].add(tg)
            sc, fp, fn_cand, fn_block, single = [], 0, 0, 0, 0
            for i, s in enumerate(prefs[p]):
                g, pr = gold[s], pred.get(i, set())
                sc.append(f05(g, pr))
                fp += len(pr - g)
                fn_cand += len((g - pr) & cand.get(i, set()))
                fn_block += len(g - cand.get(i, set()))
                single += int(not g and bool(pr))
            return {'macro_f05': float(np.mean(sc)), 'false_links': fp, 'missed_true_in_candidates': fn_cand,
                    'missed_true_by_blocking': fn_block, 'singleton_errors': single, 'references': len(prefs[p])}

        grid = [(ru, round(float(t), 2)) for ru in ('expected', 'threshold') for t in np.arange(0.3, 0.96, 0.05)]
        R['results'] = {}
        for name, prob in probs.items():
            dev = {g: evaluate('dev', prob, *g)['macro_f05'] for g in grid}
            best = max(dev, key=lambda k: (dev[k], k[1]))
            R['results'][name] = {'rule': best, 'dev': evaluate('dev', prob, *best), 'holdout': evaluate('holdout', prob, *best)}
            print(name, R['results'][name], flush=True)

        tok = AutoTokenizer.from_pretrained(CE / 'model')
        model = AutoModelForSequenceClassification.from_pretrained(CE / 'model').cuda().eval()
        sample = [('Sharma Traders Pvt Ltd | 12, MG Road, Pune', 'Sharma Traders Private Limited | 12 M G Road Pune 411001')] * 60000
        with torch.no_grad():
            for warm in range(2):
                t0 = time.time()
                for b in range(0, len(sample), 1024):
                    L, Rr = zip(*sample[b:b + 1024])
                    e = tok(list(L), list(Rr), truncation=True, max_length=128, padding=True, return_tensors='pt').to('cuda')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        model(**e).logits
                torch.cuda.synchronize()
                el = time.time() - t0
        R['throughput_e5small'] = {'pairs_per_second_synthetic': round(len(sample) / el), 'batch': 1024, 'max_len': 128,
                                   'dtype': 'bf16 autocast', 'cpus': 3, 'gpu': torch.cuda.get_device_name(0)}
        (EXP / 'report.json').write_text(json.dumps(R, indent=1, default=str))
        print(json.dumps(R, default=str), flush=True)


    if __name__ == '__main__':
        main()



def cmd_cut_and_infer():
    """Production candidate cut for upload 2: the reduced candidate set is exactly what the final models decide on.

keep = (raw v5 probability >= T) OR (first-listed candidate of the reference)   [a mask given via KEEP
must equal this one]. On the kept pairs only: T6's decision rule + global exclusivity for US/India, FRtrained_w4's
rule + exclusivity for France, then (in the job script) census v2 on the kept French pairs. Writes
artifacts/submission_UP2_T6_FRw4_cut/pre_census/{late_scores.npz, <rule>/matching_results.tsv + candidate_pairs.tsv}.
"""
    import csv
    import json
    import os
    import sys
    import time
    from collections import Counter
    from pathlib import Path

    import numpy as np

    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    A = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
    sys.path[:0] = [str(ME / 'business_entity_resolution/src'), str(A / 'business_entity_resolution/src')]
    from submission_pipeline import finalize  # noqa: E402

    T = float(os.environ.get('CUT', 0.001))
    OUT = ME / 'artifacts/submission_UP2_T6_FRw4_cut'


    def main():
        (OUT / 'pre_census').mkdir(parents=True, exist_ok=True)
        refs_t, cty = [], {}
        with open(ME / 'data/dataset/test/test_source1.tsv', encoding='utf-8', newline='') as fh:
            for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                refs_t.append({'entity_id': row['entity_id'], 'country': row['country']}); cty[row['entity_id']] = row['country']
        pos = {r['entity_id']: i for i, r in enumerate(refs_t)}
        refs, tgts, first = [], [], []
        with open(ME / 'artifacts/submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
            next(f)
            for line in f:
                s, c = line.rstrip('\n').split('\t')
                for k, t in enumerate(c.split(',') if c else []):
                    refs.append(s); tgts.append(t); first.append(k == 0)
        tgts = np.array(tgts, dtype=object)
        first = np.array(first)
        v5 = np.load(ME / 'artifacts/v5raw/v5_test_raw.npz')['v5_prob']
        keep = (v5 >= T) | first
        R = {'cut': T, 'pairs': int(len(keep)), 'kept': int(keep.sum()), 'kept_share': round(float(keep.mean()), 4),
             'candidates_per_ref': round(float(keep.sum()) / len(refs_t), 3)}
        if os.environ.get('KEEP'):
            k2 = np.load(os.environ['KEEP']).astype(bool)
            R['tapan_mask_equal'] = bool(k2.shape == keep.shape and (k2 == keep).all())
            if not R['tapan_mask_equal']:
                R['tapan_mask_diff_rows'] = int((k2 != keep).sum()) if k2.shape == keep.shape else 'shape mismatch'
                (OUT / 'cut_report.json').write_text(json.dumps(R, indent=1)); raise SystemExit(f'ABORT: masks differ {R}')
        np.save(OUT / f'keep_t{T}_vishal.npy', keep)
        ref_idx = np.array([pos[r] for r in refs], dtype=np.int64)
        Z6 = np.load(ME / 'artifacts/submission_T6_own/late_scores.npz', allow_pickle=True)
        ZF = np.load(ME / 'artifacts/submission_FRtrained_w4/late_scores.npz', allow_pickle=True)
        assert (Z6['target'].astype(str) == tgts.astype(str)).all() and (ZF['target'].astype(str) == tgts.astype(str)).all()
        rule6 = json.load(open(ME / 'artifacts/fusion/boundary_models_t6/stage3b_meta.json'))['rule']
        rulef = json.load(open(ME / 'artifacts/submission_FRtrained_w4/stage3b_frtrained_meta.json'))['rule']
        # reduced candidate lists, original order
        cands = {r['entity_id']: [] for r in refs_t}
        for r, t, kp in zip(refs, tgts, keep):
            if kp:
                cands[r].append(t)
        cands_out = {r: ','.join(v) for r, v in cands.items()}
        kj = np.flatnonzero(keep)
        out_m = {}
        for name, Z, rule in (('T6', Z6, rule6), ('FRw4', ZF, rulef)):
            tmp = OUT / f'_tmp_{name}'
            tmp.mkdir(exist_ok=True)
            finalize(refs_t, cands_out, ref_idx[kj], tgts[kj], Z['prob'][kj].astype(np.float32),
                     {'decision': {'rule': rule[0], 'threshold': rule[1]}}, True, tmp, Counter(),
                     {'model': name, 'started': time.monotonic()}, {})
            with open(tmp / 'matching_results.tsv', encoding='utf-8') as f:
                next(f); out_m[name] = {l.split('\t')[0]: l.rstrip('\n').split('\t')[1] for l in f}
        dest = OUT / 'pre_census' / f'{rulef[0]}_{rulef[1]}'
        dest.mkdir(parents=True, exist_ok=True)
        with open(dest / 'matching_results.tsv', 'w', encoding='utf-8') as f:
            f.write('source1_entity_id\tmatched_entity_ids\n')
            for r in refs_t:
                e = r['entity_id']
                f.write(e + '\t' + (out_m['FRw4'][e] if r['country'] == 'France' else out_m['T6'][e]) + '\n')
        (dest / 'candidate_pairs.tsv').write_text((OUT / '_tmp_T6' / 'candidate_pairs.tsv').read_text(encoding='utf-8'), encoding='utf-8')
        # census input: kept rows only; France = FRw4 probabilities, others T6
        fr = np.array([cty[refs[j]] == 'France' for j in kj])
        prob = np.where(fr, ZF['prob'][kj], Z6['prob'][kj]).astype(np.float32)
        np.savez(OUT / 'pre_census' / 'late_scores.npz', ref_idx=ref_idx[kj], target=tgts[kj].astype(str), prob=prob,
                 stage1=Z6['stage1'][kj], ce=Z6['ce'][kj])
        import shutil
        for name in ('T6', 'FRw4'):
            shutil.rmtree(OUT / f'_tmp_{name}')
        R['rules'] = {'T6_US_India': rule6, 'FRtrained_w4_France': rulef}
        (OUT / 'cut_report.json').write_text(json.dumps(R, indent=1))
        print(json.dumps(R, indent=1), flush=True)


    if __name__ == '__main__':
        main()



def cmd_decision_tweaks():
    """Two cheap tweaks on the S4 setup (big-data mdeberta fold-A logits + v5 probability, decoy w=3, L2 on fold B):

1. Exact expected-F0.5 top-k choice per reference under independent candidate probabilities (Poisson-binomial DP),
   versus the current ratio-of-expectations 'expected' rule. Same probability floor gate, chosen on dev.
2. L2 + label-free flags computable at test: address empty (ref / target), exact folded name, target source,
   candidate count, v5 rank, cross-encoder rank / gap-to-best / count-positive within the reference.
Rule and floor chosen on dev; holdout scored once per variant.
"""
    import gzip
    import json
    import os
    import re
    import sys
    import time
    import unicodedata
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import xgboost as xgb

    sys.path.insert(0, str(Path(__file__).parent))
    from distractor_check import decide, f05, partition_of

    HOME = Path.home()
    V4 = HOME / 'scratch/vishal_v4_artifacts'
    PRUNE = HOME / 'scratch/amazon_ml_2026/artifacts/prune_v1_copy'
    FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
    RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
    CE = Path(os.environ.get('CE_FILE', str(HOME / 'scratch/amazon_ml_2026/artifacts/cross_encoder_v1/mdeberta_big/ce_logit_foldA.npy')))
    W = float(os.environ.get('DECOY_W', 3))


    def log(*a):
        print(time.strftime('%H:%M:%S'), *a, flush=True)


    def norm(s):
        s = unicodedata.normalize('NFKD', s or '')
        s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
        return re.sub(r'[^\w]+', ' ', s).strip()


    def pb(p):
        """Poisson-binomial pmf of a sum of independent Bernoullis."""
        d = np.zeros(len(p) + 1); d[0] = 1.0
        for q in p:
            d[1:] = d[1:] * (1 - q) + d[:-1] * q
            d[0] *= (1 - q)
        return d


    _CACHE = {}


    def decide_exact(ref_idx, prob, floor, n_refs):
        key = (id(prob), len(prob))
        if key not in _CACHE:
            order = np.lexsort((-prob, ref_idx))
            bounds = np.searchsorted(ref_idx[order], np.arange(n_refs + 1))
            segs = []
            for i in range(n_refs):
                idx = order[bounds[i]:bounds[i + 1]]
                if not len(idx):
                    continue
                p = np.clip(prob[idx].astype(np.float64), 1e-9, 1 - 1e-9)
                n = len(p)
                best_k, best_v = 0, float(np.prod(1 - p))
                for k in range(1, n + 1):
                    a = pb(p[:k]); b = pb(p[k:])
                    A = np.arange(k + 1)[:, None]; B = np.arange(n - k + 1)[None, :]
                    v = float((a[:, None] * b[None, :] * (1.25 * A / (0.25 * (A + B) + k))).sum())
                    if v > best_v:
                        best_k, best_v = k, v
                segs.append((idx, best_k, p[0]))
            _CACHE[key] = segs
        res = np.zeros(len(prob), dtype=bool)
        for idx, k, p0 in _CACHE[key]:
            if p0 >= floor and k:
                res[idx[:k]] = True
        return res


    def main():
        gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
        cp = partition_of(gold)
        mask = np.load(PRUNE / 'mask_top10.npy')
        refs, tgts = [], []
        with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t')
                    refs.append(r); tgts.append(t)
        refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
        part = np.array([cp[r] for r in refs], dtype=object)
        S = np.load(FUS / 'stage1.npz')
        y = np.load(V4 / 'pairs/y.npy')[S['rows']].astype(np.float32)
        s1 = S['stage1'].astype(np.float64)
        s1f = np.where(s1 < 0.02, 0.0, s1)
        ce = np.nan_to_num(np.load(CE).astype(np.float64), nan=0.0)

        need = set(refs) | set(tgts)
        text, owned = {}, set()
        for k in (1, 2, 3):
            with open(RAW / f'train_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if len(q) > iaddr and q[ie] in need:
                        text[q[ie]] = (q[iname], q[iaddr])
        with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) > 1 and q[1]:
                    owned.update(t.strip() for t in q[1].split(','))
        by_ref = defaultdict(list)
        for j, r in enumerate(refs):
            by_ref[r].append(j)
        flags = np.zeros((len(refs), 9), dtype=np.float32)
        for r, js in by_ref.items():
            rn, ra = text.get(r, ('', ''))
            o1 = np.argsort(-s1f[js], kind='stable'); rk1 = np.empty(len(js)); rk1[o1] = np.arange(1, len(js) + 1)
            c = ce[js]; oc = np.argsort(-c, kind='stable'); rkc = np.empty(len(js)); rkc[oc] = np.arange(1, len(js) + 1)
            for q, j in enumerate(js):
                tn, ta = text.get(tgts[j], ('', ''))
                flags[j] = [float(not ra.strip()), float(not ta.strip()), float(norm(rn) == norm(tn) and bool(norm(rn))),
                            float(tgts[j].startswith('S3')), len(js), rk1[q], rkc[q], c.max() - c[q], float((c > 0).sum())]
        log('flags built')
        wt = np.where((y == 0) & np.array([t not in owned for t in tgts]), W, 1.0).astype(np.float32)
        tb, dv = part == 'trainB', part == 'dev'
        idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
        prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
        rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}
        params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                      max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)

        def score(p, chosen):
            pred = defaultdict(set)
            for i, tt in zip(rix[p][chosen], tgts[idx[p]][chosen]):
                pred[i].add(tt)
            return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

        R = {'baseline_ref': 'S4 setup: dev 0.98728 / holdout 0.98571'}
        for fname, X in (('L2', np.column_stack([s1f, ce])), ('L2_flags', np.column_stack([s1f, ce, flags]))):
            X = X.astype(np.float32)
            dtr = xgb.QuantileDMatrix(X[tb], y[tb], weight=wt[tb])
            bst = xgb.train(params, dtr, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=dtr), 'dev')],
                            early_stopping_rounds=100, verbose_eval=False)
            it = (0, bst.best_iteration + 1)
            P = {p: np.asarray(bst.inplace_predict(X[idx[p]], iteration_range=it), dtype=np.float64) for p in idx}
            for rule in ('expected', 'threshold', 'exact'):
                grid = np.arange(0.4, 0.961, 0.02)
                def ch(p, t):
                    if rule == 'exact':
                        return decide_exact(rix[p], P[p], t, len(prefs[p]))
                    return decide(rix[p], P[p], rule, t, len(prefs[p]))
                dev = {round(float(t), 2): score('dev', ch('dev', t)) for t in grid}
                bt = max(dev, key=lambda k: (dev[k], k))
                R[f'{fname}/{rule}'] = {'floor': bt, 'dev': round(dev[bt], 6), 'holdout': round(score('holdout', ch('holdout', bt)), 6)}
                log(f'{fname}/{rule}', R[f'{fname}/{rule}'])
                (FUS / 'decision_tweaks.json').write_text(json.dumps(R, indent=1))
            if fname == 'L2_flags':
                g = bst.get_score(importance_type='total_gain')
                R['flag_gain'] = {k: round(v) for k, v in sorted(g.items(), key=lambda kv: -kv[1])}
        (FUS / 'decision_tweaks.json').write_text(json.dumps(R, indent=1))
        log('done')


    if __name__ == '__main__':
        main()



def cmd_decoy_weight_check():
    """Decoy reweighting for the late-fusion matcher (L2: XGBoost on [v5 top-10 probability, mdeberta logit]).

Test has far more decoy records (S2/S3 records that belong to no S1 business) than train. This trains L2 on fold B
with decoy negatives weighted by w, picks the rule on dev, scores holdout once, and saves each model in the format
late_predict.py reads. False links are split into decoy and non-decoy, since only decoy false links scale with the
test decoy share. Inputs are read-only from the shared scratch; outputs go to OUT_DIR.
"""
    import gzip
    import json
    import os
    import sys
    import time
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import xgboost as xgb

    sys.path.insert(0, str(Path(__file__).parent))
    from distractor_check import decide, f05, partition_of

    SCR = Path(os.environ.get('ER_SCRATCH', str(Path.home() / 'scratch')))
    V4 = SCR / 'vishal_v4_artifacts'
    PRUNE = SCR / 'amazon_ml_2026/artifacts/prune_v1_copy'
    FUS = SCR / 'amazon_ml_2026/artifacts/fusion'
    CE = Path(os.environ.get('CE_FILE', str(SCR / 'amazon_ml_2026/artifacts/cross_encoder_v1/mdeberta_drop0/ce_logit.npy')))
    RAW = SCR / 'amazon_ml_2026/data/dataset/train'
    OUT = Path(os.environ.get('OUT_DIR', str(FUS / 'decoy')))
    WEIGHTS = [float(w) for w in os.environ.get('WEIGHTS', '1,1.5,2,3,4').split(',')]
    TEST_S23_IDS = 9_969_589      # organizer validator, test S2+S3 records
    TEST_PRED_LINKS = 5_790_570   # v5 top-10 test output, sum of match counts (report.json)


    def log(*a):
        print(time.strftime('%H:%M:%S'), *a, flush=True)


    def main():
        OUT.mkdir(parents=True, exist_ok=True)
        gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
        cp = partition_of(gold)
        mask = np.load(PRUNE / 'mask_top10.npy')
        refs, tgts = [], []
        with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t')
                    refs.append(r); tgts.append(t)
        refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
        part = np.array([cp[r] for r in refs], dtype=object)
        S = np.load(FUS / 'stage1.npz')
        rows = S['rows']
        assert len(rows) == len(refs)
        y = np.load(V4 / 'pairs/y.npy')[rows].astype(np.float32)
        s1 = S['stage1'].astype(np.float64)
        s1f = np.where(s1 < 0.02, 0.0, s1)
        ce = np.nan_to_num(np.load(CE).astype(np.float64), nan=0.0)
        X = np.column_stack([s1f, ce]).astype(np.float32)

        owned = set()
        with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) > 1 and p[1]:
                    owned.update(t.strip() for t in p[1].split(','))
        n_train_s23 = 0
        for src in ('train_source2.tsv', 'train_source3.tsv'):
            with open(RAW / src, encoding='utf-8') as f:
                n_train_s23 += sum(1 for _ in f) - 1
        train_share = 1 - len(owned) / n_train_s23
        test_share = 1 - TEST_PRED_LINKS / TEST_S23_IDS
        odds = (test_share / (1 - test_share)) / (train_share / (1 - train_share))
        decoy = (y == 0) & np.array([t not in owned for t in tgts])
        R = {'decoy_share_records': {'train': round(train_share, 4), 'test_estimated': round(test_share, 4),
                                     'implied_weight_odds_ratio': round(odds, 3)},
             'decoy_share_of_negative_pairs': {p: round(float(decoy[(part == p) & (y == 0)].mean()), 4)
                                               for p in ('trainB', 'dev', 'holdout')}}
        log(R)

        tb, dv = part == 'trainB', part == 'dev'
        idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
        prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
        rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}
        grid = [(ru, round(float(t), 3)) for ru in ('expected', 'threshold') for t in np.arange(0.5, 0.981, 0.02)]
        params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                      max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)

        def evaluate(p, prob, ru, t):
            c = decide(rix[p], prob, ru, t, len(prefs[p]))
            pred = defaultdict(set)
            for i, tg in zip(rix[p][c], tgts[idx[p]][c]):
                pred[i].add(tg)
            yy, dd = y[idx[p]], decoy[idx[p]]
            single = sum(1 for i, s in enumerate(prefs[p]) if not gold[s] and pred.get(i))
            return {'f05': float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])])),
                    'false_links': int((c & (yy == 0)).sum()), 'false_links_decoy': int((c & dd).sum()),
                    'missed_true': int((~c & (yy == 1)).sum()), 'singleton_errors': single,
                    'links_per_ref': round(float(c.sum()) / len(prefs[p]), 4)}

        R['runs'] = {}
        for w in WEIGHTS:
            wt = np.where(decoy, w, 1.0).astype(np.float32)
            dtr = xgb.QuantileDMatrix(X[tb], y[tb], weight=wt[tb])
            ddv = xgb.QuantileDMatrix(X[dv], y[dv], ref=dtr)
            bst = xgb.train(params, dtr, 3000, evals=[(ddv, 'dev')], early_stopping_rounds=100, verbose_eval=False)
            it = (0, bst.best_iteration + 1)
            pd_ = np.asarray(bst.inplace_predict(X[idx['dev']], iteration_range=it), dtype=np.float64)
            ph = np.asarray(bst.inplace_predict(X[idx['holdout']], iteration_range=it), dtype=np.float64)
            dev = {g: evaluate('dev', pd_, *g)['f05'] for g in grid}
            best = max(dev, key=lambda k: (dev[k], k[1]))
            run = {'rule': best, 'best_iteration': int(bst.best_iteration), 'dev': evaluate('dev', pd_, *best),
                   'holdout': evaluate('holdout', ph, *best)}
            R['runs'][str(w)] = run
            md = OUT / f'L2_decoy_w{w:g}'
            md.mkdir(parents=True, exist_ok=True)
            bst.save_model(str(md / 'model.json'))
            (md / 'config.json').write_text(json.dumps({'name': f'L2_decoy_w{w:g}', 'tag': 'mdeberta_drop0',
                                                       'rule': list(best), 'best_iteration': int(bst.best_iteration),
                                                       'features': ['v5_top10_prob_floored_0.02', 'mdeberta_foldA_logit'],
                                                       'decoy_weight': w, 'dev_f05': run['dev']['f05'],
                                                       'holdout_f05': run['holdout']['f05']}))
            log('w', w, run)
        (OUT / 'decoy_weight_check.json').write_text(json.dumps(R, indent=1, default=str))
        log('done', OUT / 'decoy_weight_check.json')


    if __name__ == '__main__':
        main()



def cmd_exclusivity_norm():
    """Exclusivity-consistent re-decision: every S2/S3 record belongs to at most one S1, so its probabilities over the S1
lists it appears in must sum to at most 1. France is over-committed 12x more often than US/India (overcommit_check.py:
1.1% of accepted France links sit on records with sum > 1.2, against 0.1%), typically templated names such as
'Tourcoing Club SAS' at several addresses, where the model is sure of two owners at once. For pairs of the listed
countries, p' = p / max(1, sum over S1 of p for that record); then the same expected-F0.5 rule and exclusivity.
Env BASE (folder with late_scores.npz + candidates.tsv), OUT, RULE (e.g. expected@0.64), COUNTRIES (default France).
"""
    import os
    import shutil
    import sys
    from argparse import Namespace
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).parent))
    from late_predict import decide_and_write, load_refs_text


    def main():
        B, O = Path(os.environ['BASE']), Path(os.environ['OUT'])
        O.mkdir(parents=True, exist_ok=True)
        C = set(os.environ.get('COUNTRIES', 'France').split(','))
        Z = dict(np.load(B / 'late_scores.npz', allow_pickle=True))
        p = Z['prob'].astype(np.float64)
        _, inv = np.unique(Z['target'].astype(str), return_inverse=True)
        S = np.bincount(inv, weights=p)[inv]
        raw = Path.home() / 'scratch/amazon_ml_2026/data/dataset/test/test_source1.tsv'
        with open(raw, encoding='utf-8') as fh:
            ic = fh.readline().rstrip('\n').split('\t').index('country')
            cty = np.array([line.rstrip('\n').split('\t')[ic] for line in fh], dtype=object)
        m = np.isin(cty[Z['ref_idx']], list(C))
        pn = np.where(m, p / np.maximum(1.0, S), p)
        print('pairs renormalised', int((m & (S > 1)).sum()), 'mean p change on them',
              float((p - pn)[m & (S > 1)].mean()), flush=True)
        Z['prob'] = pn.astype(Z['prob'].dtype)
        np.savez(O / 'late_scores.npz', **Z)
        shutil.copy(B / 'candidates.tsv', O / 'candidates.tsv')
        rule, thr = os.environ['RULE'].split('@')
        decide_and_write(Namespace(output=str(O), rule=os.environ['RULE'], tag='', late_model=str(B), country_threshold=''),
                         {'rule': [rule, float(thr)]})


    if __name__ == '__main__':
        main()



def cmd_fusion_predict():
    """Test inference for the cross-encoder fusion matcher.

Input dir (--pairs): pairs.tsv.gz (ref \\t target, one line per candidate), x.npy (N x 103 float32, the v4 feature
order in prune_v1/model_top10/report.json). Text comes from the organizer test TSVs.
Stages: stage-1 top-10 matcher -> anchors -> sibling features -> cross-encoder scores (fold A/B mean) -> fusion
XGBoost -> finalize() (global exclusivity, decision rule, TSVs). Probabilities are saved so `--rerule` can re-run
the decision with a different rule/floor in minutes.
"""
    import argparse
    import gzip
    import json
    import re
    import sys
    import time
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    import xgboost as xgb
    from rapidfuzz import fuzz
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).parent))
    from submission_pipeline import decide, finalize, parse_country_thresholds
    from matcher_eda import norm
    from sibling_check import prefix_match, num_jacc

    HOME = Path.home()
    VISHAL = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
    CE = HOME / 'scratch/amazon_ml_2026/artifacts/cross_encoder_v1'
    FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
    RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/test'
    MAX_ANCHORS, MIN_SIB_PROB, MAX_LEN = 3, 0.005, 128


    def log(*a):
        print(time.strftime('%H:%M:%S'), *a, flush=True)


    def ce_score(model_dir, L, R):
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
        out = np.zeros(len(L), dtype=np.float32)
        order = np.argsort([len(a) + len(b) for a, b in zip(L, R)], kind='stable')
        t0 = time.time()
        with torch.no_grad():
            for b in range(0, len(order), 1024):
                ids = order[b:b + 1024]
                e = tok([L[i] for i in ids], [R[i] for i in ids], truncation=True, max_length=MAX_LEN, padding=True,
                        return_tensors='pt').to('cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out[ids] = model(**e).logits.squeeze(-1).float().cpu().numpy()
                if b % (1024 * 2000) == 0:
                    log(f'  {model_dir.name}: {b}/{len(order)} {time.time() - t0:.0f}s')
        del model
        torch.cuda.empty_cache()
        return out


    def refs_and_text():
        text, refs = {}, []
        for src in ('test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv'):
            df = pd.read_csv(RAW / src, sep='\t', dtype=str, keep_default_na=False, quoting=3)
            for e, n, a in zip(df.entity_id, df.business_name, df.business_address):
                text[e] = (n, a)
            if src == 'test_source1.tsv':
                refs = [{'entity_id': e, 'country': c} for e, c in zip(df.entity_id, df.country)]
            log('text', src, len(df))
        return refs, text


    def score_all(a):
        out = Path(a.output)
        out.mkdir(parents=True, exist_ok=True)
        cfg = json.loads((FUS / 'models' / a.config / 'config.json').read_text())
        log('config', a.config, cfg)
        pr, tg = [], []
        with gzip.open(Path(a.pairs) / 'pairs.tsv.gz', 'rt') as f:
            for line in f:
                r, t = line.rstrip('\n').split('\t')
                pr.append(r); tg.append(t)
        refs_arr, tgts = np.array(pr, dtype=object), np.array(tg, dtype=object)
        x = np.load(Path(a.pairs) / 'x.npy', mmap_mode='r')
        assert x.shape[0] == len(refs_arr), (x.shape, len(refs_arr))
        x = np.asarray(x, dtype=np.float32)
        log('pairs', len(refs_arr))
        refs, text = refs_and_text()
        position = {r['entity_id']: i for i, r in enumerate(refs)}
        ref_idx = np.array([position[r] for r in refs_arr], dtype=np.int64)

        rep = json.loads((VISHAL / 'prune_v1/model_top10/report.json').read_text())
        s1m = xgb.Booster()
        s1m.load_model(str(VISHAL / 'prune_v1/model_top10/model.json'))
        s1m.set_param({'device': 'cuda'})
        stage1 = np.asarray(s1m.inplace_predict(x, iteration_range=(0, rep['best_iteration'] + 1)), dtype=np.float64)
        chosen = decide(ref_idx, stage1, rep['decision']['rule'], rep['decision']['threshold'], len(refs))
        anchor = chosen & (stage1 >= 0.9)
        log('stage1 done; anchors', int(anchor.sum()))

        blocks = cfg['feature_blocks']
        feats = {'base': x}
        if 'ctx_sibrf' in blocks or any(b.startswith('sibce_') for b in blocks):
            ntext = {}
            for t in set(tgts):
                n_, ad = text.get(t, ('', ''))
                ntext[t] = (norm(n_), norm(ad))
            emb = None
            if 'ctx_sibrf' in blocks:
                tok = AutoTokenizer.from_pretrained(VISHAL / 'dense_v1/model')
                enc = AutoModel.from_pretrained(VISHAL / 'dense_v1/model').cuda().eval()
                uniq = sorted(set(tgts))
                tid = {t: i for i, t in enumerate(uniq)}
                emb = np.zeros((len(uniq), enc.config.hidden_size), dtype=np.float32)
                with torch.no_grad():
                    for b in range(0, len(uniq), 2048):
                        bt = [f'passage: {text.get(t, ("", ""))[0]} {text.get(t, ("", ""))[1]}' for t in uniq[b:b + 2048]]
                        e = tok(bt, truncation=True, max_length=96, padding=True, return_tensors='pt').to('cuda')
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            h = enc(**e).last_hidden_state
                        m = e['attention_mask'].unsqueeze(-1).float()
                        emb[b:b + 2048] = torch.nn.functional.normalize((h.float() * m).sum(1) / m.sum(1), dim=-1).cpu().numpy()
                del enc
                log('embedded targets', len(uniq))
            by_ref = defaultdict(list)
            for j, r in enumerate(ref_idx):
                by_ref[r].append(j)
            ctx = np.zeros((len(tgts), 7), dtype=np.float32)
            sib = np.full((len(tgts), 8), -1.0, dtype=np.float32)
            pairpre = np.zeros(len(tgts), dtype=np.float32)
            sib_pairs = []
            for r, js in by_ref.items():
                js = sorted(js, key=lambda j: -stage1[j])
                p = stage1[js]
                top, s, n9, n5 = p[0], p.sum(), (p >= 0.9).sum(), (p >= 0.5).sum()
                anc = [j for j in js if anchor[j]]
                ra = text.get(refs[r]['entity_id'], ('', ''))[1]
                for rk, j in enumerate(js):
                    nxt = p[rk + 1] if rk + 1 < len(p) else 0.0
                    ctx[j] = [stage1[j], rk + 1, top - stage1[j], stage1[j] - nxt, s, n9, n5]
                    ta = text.get(tgts[j], ('', ''))[1]
                    pairpre[j] = prefix_match(ra, ta)
                    an = [q for q in anc if q != j][:MAX_ANCHORS]
                    if not an:
                        continue
                    if stage1[j] >= MIN_SIB_PROB:
                        sib_pairs.extend((j, q) for q in an)
                    if emb is None:
                        continue
                    tn, tna = ntext[tgts[j]]
                    F = []
                    for q in an:
                        an_n, an_a = ntext[tgts[q]]
                        raw_a = text.get(tgts[q], ('', ''))[1]
                        F.append([fuzz.token_sort_ratio(tn, an_n) / 100,
                                  fuzz.token_set_ratio(tna, an_a) / 100 if tna and an_a else -1.0,
                                  num_jacc(ta, raw_a), prefix_match(ta, raw_a),
                                  float(emb[tid[tgts[j]]] @ emb[tid[tgts[q]]]), float(tgts[j][:2] != tgts[q][:2])])
                    F = np.array(F)
                    sib[j] = [F[:, 0].max(), F[:, 1].max(), F[:, 2].max(), F[:, 3].max(), F[:, 4].max(),
                              F[:, 4].mean(), F[:, 5].mean(), len(an)]
            sib_pairs = np.array(sib_pairs, dtype=np.int64).reshape(-1, 2)
            feats['ctx_sibrf'] = np.column_stack([ctx, sib, pairpre])
            log('set/sibling features; sibling pairs', len(sib_pairs))

        fmt = lambda na: f'{na[0]} | {na[1]}'
        L = [fmt(text.get(refs[r]['entity_id'], ('', ''))) for r in ref_idx]
        Rt = [fmt(text.get(t, ('', ''))) for t in tgts]
        for tag in cfg['tags']:
            if f'ce_{tag}' not in blocks and f'sibce_{tag}' not in blocks:
                continue
            folds = [CE / tag / f'model_fold{k}' for k in ('A', 'B')]
            feats[f'ce_{tag}'] = np.mean([ce_score(d, L, Rt) for d in folds], axis=0)[:, None]
            np.save(out / f'ce_{tag}.npy', feats[f'ce_{tag}'][:, 0])
            if f'sibce_{tag}' in blocks and len(sib_pairs):
                SL = [Rt[j] for j in sib_pairs[:, 0]]
                SR = [Rt[q] for q in sib_pairs[:, 1]]
                sl = np.mean([ce_score(d, SL, SR) for d in folds], axis=0)
                mx = np.full(len(tgts), -20.0, dtype=np.float32)
                sm, cnt = np.zeros(len(tgts), np.float32), np.zeros(len(tgts), np.float32)
                np.maximum.at(mx, sib_pairs[:, 0], sl)
                np.add.at(sm, sib_pairs[:, 0], sl)
                np.add.at(cnt, sib_pairs[:, 0], 1)
                feats[f'sibce_{tag}'] = np.column_stack([mx, np.where(cnt > 0, sm / np.maximum(cnt, 1), -20.0)])
            log('cross-encoder done', tag)

        F = np.column_stack([feats[b] for b in blocks]).astype(np.float32)
        fm = xgb.Booster()
        fm.load_model(str(FUS / 'models' / a.config / 'model.json'))
        fm.set_param({'device': 'cuda'})
        prob = np.asarray(fm.inplace_predict(F, iteration_range=(0, cfg['best_iteration'] + 1)), dtype=np.float32)
        cands = defaultdict(list)
        for r, t in zip(ref_idx, tgts):
            cands[r].append(t)
        np.savez(out / 'scores.npz', ref_idx=ref_idx, target=tgts.astype(str), prob=prob, stage1=stage1.astype(np.float32))
        with open(out / 'candidates.tsv', 'w', encoding='utf-8') as f:
            for i, r in enumerate(refs):
                f.write(r['entity_id'] + '\t' + ','.join(cands.get(i, [])) + '\n')
        log('fusion scored', len(prob))
        return cfg


    def run_finalize(a, cfg):
        out = Path(a.output)
        refs, _ = refs_and_text()
        S = np.load(out / 'scores.npz', allow_pickle=True)
        cands_out = {}
        with open(out / 'candidates.tsv', encoding='utf-8') as f:
            for line in f:
                sid, c = line.rstrip('\n').split('\t')
                cands_out[sid] = c
        rule, thr = cfg['rule']
        if a.rule:
            rule, thr = a.rule.split('@')[0], float(a.rule.split('@')[1])
        dest = out / (a.tag or f'{rule}_{thr}')
        dest.mkdir(parents=True, exist_ok=True)
        stats = Counter()
        finalize(refs, cands_out, S['ref_idx'], S['target'].astype(object), S['prob'],
                 {'decision': {'rule': rule, 'threshold': thr}}, True, dest, stats,
                 {'config': a.config, 'rule': [rule, thr], 'started': time.monotonic()}, parse_country_thresholds(a.country_threshold))
        log('wrote', dest, dict(matches=stats.get('matches')))


    def main():
        p = argparse.ArgumentParser()
        p.add_argument('--pairs', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--config', default='all')
        p.add_argument('--rerule', action='store_true', help='skip scoring; re-run the decision from scores.npz')
        p.add_argument('--rule', default='', help='override, e.g. expected@0.9')
        p.add_argument('--country-threshold', default='')
        p.add_argument('--tag', default='')
        a = p.parse_args()
        cfg = json.loads((FUS / 'models' / a.config / 'config.json').read_text()) if a.rerule else score_all(a)
        run_finalize(a, cfg)


    if __name__ == '__main__':
        main()



def cmd_gap_diag():
    """Where do the remaining points go? Per-country diagnostics for the final push (27 Sep afternoon).

1. Blocking, sample (US/India, labels): gold links per reference by stage-1 bucket (top 10, ranks 11-20, beyond 20,
   never retrieved). Test (label-free): rank 11-20 candidates with a high v5 top-20-model probability per reference.
2. Cardinality: links per source (S2, S3) per reference, train ground truth vs the submitted predictions, per country.
3. Row-order / ID check on train: do file positions or ID values of a reference and its gold records correlate, and do
   decoys sit elsewhere in the files than linked records?
Usage: gap_diag.py <matching_results.tsv>  -> prints tables, writes artifacts/gap_diag/report.json
"""
    import csv
    import gzip
    import json
    import sys
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np
    from scipy.stats import spearmanr

    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    ART = ME / 'artifacts'
    OUT = ART / 'gap_diag'
    csv.field_size_limit(10 ** 9)


    def read_ids(path):
        ids, cty = [], {}
        with open(path, encoding='utf-8', newline='') as fh:
            for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                ids.append(row['entity_id']); cty[row['entity_id']] = row['country']
        return ids, cty


    def load_lists(path):
        with open(path, encoding='utf-8') as f:
            next(f)
            return {l.split('\t')[0]: [x for x in l.rstrip('\n').split('\t')[1].split(',') if x] for l in f}


    def main():
        OUT.mkdir(parents=True, exist_ok=True)
        R = {}
        # ---- 1a. blocking on the sample
        refs = [l.split('\t')[0] for l in gzip.open(ART / 'matcher_v4/pairs/pairs.tsv.gz', 'rt', encoding='utf-8')]
        refs = np.array(refs)
        y = np.load(ART / 'matcher_v4/pairs/y.npy').astype(bool)
        m10 = np.load(ART / 'prune_v1/mask_top10.npy').astype(bool)
        m20 = np.load(ART / 'prune_v1/mask_top20.npy').astype(bool)
        assert len(refs) == len(y) == len(m10) == len(m20)
        sc = json.load(open(ART / 'matcher_v4/pairs/countries.json'))
        gold = json.load(open(ART / 'matcher_v4/pairs/gold.json'))
        ngold = {r: len(v) for r, v in gold.items()} if isinstance(next(iter(gold.values())), list) else gold
        u, inv = np.unique(refs, return_inverse=True)
        c10 = np.bincount(inv, weights=y & m10, minlength=len(u))
        c20 = np.bincount(inv, weights=y & m20 & ~m10, minlength=len(u))
        cbe = np.bincount(inv, weights=y & ~m20, minlength=len(u))
        blk = {}
        for c in ('US', 'India'):
            k = np.array([sc[r] == c for r in u])
            g = np.array([ngold.get(r, 0) for r in u])[k]
            tot = g.sum()
            blk[c] = {'refs': int(k.sum()), 'gold_links': int(tot),
                      'share_top10': round(c10[k].sum() / tot, 5), 'share_11_20': round(c20[k].sum() / tot, 5),
                      'share_beyond20': round(cbe[k].sum() / tot, 5),
                      'share_not_retrieved': round((tot - c10[k].sum() - c20[k].sum() - cbe[k].sum()) / tot, 5),
                      'refs_with_gold_at_11_20': round(float((c20[k] > 0).mean()), 5)}
        R['sample_blocking'] = blk
        print('== sample blocking (gold link share by stage-1 bucket)', json.dumps(blk, indent=1), flush=True)
        del refs, y, m10, m20, inv
        # ---- 1b. blocking on test, label-free
        t_ids, tcty = read_ids(ME / 'data/dataset/test/test_source1.tsv')
        top10 = load_lists(ART / 'submission_v5_top10/candidate_pairs.tsv')
        top20 = load_lists(ART / 'submission_v5_top20/candidate_pairs.tsv')
        same_prefix = sum(top20[r][:10] == top10[r] for r in t_ids)
        prob = {}
        for s in (0, 1):
            z = np.load(ART / f'full_v5_s{s}/top20/scores.npz', allow_pickle=True)
            for r, t, p in zip(z['ref'].astype(str), z['target'].astype(str), z['prob']):
                prob[(r, t)] = float(p)
        tb = {}
        for c in ('US', 'India', 'France'):
            st = Counter(); n = 0
            for r in t_ids:
                if tcty[r] != c:
                    continue
                n += 1
                s10 = set(top10[r])
                for t in top20[r]:
                    p = prob.get((r, t), 0.0)
                    key = 'top10' if t in s10 else 'r11_20'
                    st[key + '_p05'] += p >= 0.5; st[key + '_p09'] += p >= 0.9; st[key + '_p02'] += p >= 0.2
                st['refs_r11_20_p05'] += any(prob.get((r, t), 0) >= 0.5 for t in top20[r] if t not in s10)
            tb[c] = {k: round(v / n, 5) for k, v in st.items()}
            tb[c]['refs'] = n
        R['test_blocking_label_free'] = tb
        R['test_top20_prefix_equals_top10'] = same_prefix
        print('== test: candidates per reference with v5(top-20 model) p >= x, top 10 vs ranks 11-20',
              json.dumps(tb, indent=1), 'prefix equal', same_prefix, flush=True)
        del prob
        # ---- 2. cardinality per source
        tr_ids, trcty = read_ids(ME / 'data/dataset/train/train_source1.tsv')
        gt = load_lists(ME / 'data/dataset/train/train_ground_truth.tsv')
        sub = load_lists(sys.argv[1])

        def dist(lists, ids, cty, c):
            h2, h3, hb, n, links = Counter(), Counter(), Counter(), 0, 0
            for r in ids:
                if cty[r] != c:
                    continue
                L = lists.get(r, [])
                n2 = sum(x.startswith('S2-') for x in L); n3 = len(L) - n2
                h2[min(n2, 5)] += 1; h3[min(n3, 5)] += 1; hb[(min(n2, 4), min(n3, 4))] += 1; n += 1; links += len(L)
            return {'refs': n, 'links_per_ref': round(links / n, 4),
                    'S2': [round(h2[k] / n, 4) for k in range(6)], 'S3': [round(h3[k] / n, 4) for k in range(6)],
                    'empty': round(hb[(0, 0)] / n, 4),
                    'top_pairs': [(f'{a}/{b}', round(v / n, 4)) for (a, b), v in hb.most_common(8)]}
        card = {'train_gt': {c: dist(gt, tr_ids, trcty, c) for c in ('US', 'India')},
                'test_pred': {c: dist(sub, t_ids, tcty, c) for c in ('US', 'India', 'France')}}
        R['cardinality'] = card
        print('== links per source (S2, S3 histograms for 0..5+)', json.dumps(card, indent=1), flush=True)
        # ---- 3. row order / ID values on train
        pos = {}
        for s in (1, 2, 3):
            ids, _ = read_ids(ME / f'data/dataset/train/train_source{s}.tsv')
            n = len(ids)
            for i, e in enumerate(ids):
                pos[e] = i / n
        a, b, ia, ib = [], [], [], []
        linked = set()
        for r, L in gt.items():
            for t in L:
                linked.add(t)
                if r in pos and t in pos:
                    a.append(pos[r]); b.append(pos[t]); ia.append(int(r.split('-')[1])); ib.append(int(t.split('-')[1]))
        rng = np.random.default_rng(0)
        k = rng.choice(len(a), size=min(len(a), 2_000_000), replace=False)
        a, b, ia, ib = (np.array(v)[k] for v in (a, b, ia, ib))
        leak = {'spearman_rowpos_S1_vs_target': round(float(spearmanr(a, b)[0]), 5),
                'spearman_id_S1_vs_target': round(float(spearmanr(ia, ib)[0]), 5)}
        for s in ('S2', 'S3'):
            dp, lp, di, li = [], [], [], []
            for e, p in pos.items():
                if not e.startswith(s + '-'):
                    continue
                (lp if e in linked else dp).append(p); (li if e in linked else di).append(int(e.split('-')[1]))
            leak[s] = {'decoy_share': round(len(dp) / (len(dp) + len(lp)), 4),
                       'mean_rowpos_decoy': round(float(np.mean(dp)), 4), 'mean_rowpos_linked': round(float(np.mean(lp)), 4),
                       'mean_id_decoy': round(float(np.mean(di)), 1), 'mean_id_linked': round(float(np.mean(li)), 1),
                       'decoy_share_by_rowpos_decile': [round(float(v), 4) for v in np.histogram(dp, bins=10, range=(0, 1))[0]
                                                        / np.maximum(1, np.histogram(dp + lp, bins=10, range=(0, 1))[0])]}
        R['train_row_order'] = leak
        print('== train row order / ID values', json.dumps(leak, indent=1), flush=True)
        (OUT / 'report.json').write_text(json.dumps(R, indent=1))


    if __name__ == '__main__':
        main()



def cmd_late_fusion_check():
    """Late fusion of a cross-encoder with the feature-matcher probability, checked on dev and holdout."""
    import gzip
    import importlib.util
    import json
    import sys
    import time
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression

    sys.path.insert(0, str(Path(__file__).parent))
    from submission_pipeline import decide, f05

    HOME = Path.home()
    V4 = HOME / 'scratch/vishal_v4_artifacts'
    VISHAL = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
    FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
    CE = HOME / 'scratch/amazon_ml_2026/artifacts/cross_encoder_v1'
    TAG = sys.argv[1] if len(sys.argv) > 1 else 'mdeberta_drop0'


    def log(*a):
        print(time.strftime('%H:%M:%S'), *a, flush=True)


    def main():
        spec = importlib.util.spec_from_file_location('cem', Path(__file__).parent / 'cross_encoder_exp.py')
        cem = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cem)
        gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
        cp = cem.partition_of(gold)
        mask = np.load(VISHAL / 'prune_v1/mask_top10.npy')
        refs, tgts = [], []
        with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t')
                    refs.append(r); tgts.append(t)
        refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
        part = np.array([cp[r] for r in refs], dtype=object)
        S = np.load(FUS / 'stage1.npz')
        rows = S['rows']
        y = np.load(V4 / 'pairs/y.npy')[rows].astype(np.float32)
        x = np.asarray(np.load(V4 / 'pairs/x.npy', mmap_mode='r')[rows])
        s1 = S['stage1'].astype(np.float64)
        s1f = np.where(s1 < 0.02, 0.0, s1)
        lg = np.log(np.clip(s1f, 1e-4, 1 - 1e-6) / (1 - np.clip(s1f, 1e-4, 1 - 1e-6)))
        ce = np.nan_to_num(np.load(CE / TAG / 'ce_logit.npy').astype(np.float64), nan=0.0)
        by_ref = defaultdict(list)
        for j, r in enumerate(refs):
            by_ref[r].append(j)
        ctx = np.zeros((len(rows), 7), dtype=np.float32)
        for js in by_ref.values():
            js = sorted(js, key=lambda j: -s1f[j])
            p = s1f[js]
            for rk, j in enumerate(js):
                nxt = p[rk + 1] if rk + 1 < len(p) else 0.0
                ctx[j] = [s1f[j], rk + 1, p[0] - s1f[j], s1f[j] - nxt, p.sum(), (p >= 0.9).sum(), (p >= 0.5).sum()]
        ctx_floor = np.column_stack([ctx, S['sib'], S['pairpre']])
        ce_ctx = np.zeros((len(rows), 4), dtype=np.float32)
        for js in by_ref.values():
            c = ce[js]
            o = np.argsort(-c)
            rank = np.empty(len(js)); rank[o] = np.arange(1, len(js) + 1)
            ce_ctx[js] = np.column_stack([rank, c.max() - c, np.full(len(js), (c > 0).sum()), np.full(len(js), c.max())])

        tb = part == 'trainB'
        idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
        prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
        rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}
        grid = [(ru, round(float(t), 3)) for ru in ('expected', 'threshold') for t in np.arange(0.3, 0.96, 0.025)]

        def macro(p, prob, ru, t):
            c = decide(rix[p], prob, ru, t, len(prefs[p]))
            pred = defaultdict(set)
            for i, tg in zip(rix[p][c], tgts[idx[p]][c]):
                pred[i].add(tg)
            return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

        def report(name, prob):
            dev = {g: macro('dev', prob[idx['dev']], *g) for g in grid}
            best = max(dev, key=lambda k: (dev[k], k[1]))
            R[name] = {'rule': best, 'dev_f05': dev[best], 'holdout_f05': macro('holdout', prob[idx['holdout']], *best)}
            log(name, R[name])

        R = {'tag': TAG, 'note': 'Trained on fold B (cross-encoder trained on fold A). Stage-1 probabilities are 5-fold '
                                'out-of-fold on train and floored at 0.02 like the saved test scores.'}
        report('stage1_only_floored', s1f)
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.column_stack([lg, ce])[tb], y[tb])
        report('L1_logreg_stage1_ce', lr.predict_proba(np.column_stack([lg, ce]))[:, 1])
        small = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                     max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)
        big = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                   grow_policy='lossguide', max_depth=0, max_leaves=255, learning_rate=0.05, subsample=0.8,
                   colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=1.0, max_bin=256, seed=2026)
        dv = part == 'dev'
        variants = {'L2_xgb_stage1_ce': (np.column_stack([s1f, ce]), small),
                    'L3_xgb_stage1_ce_setctx_sibling': (np.column_stack([ctx_floor, ce, ce_ctx]), big),
                    'F_feature_level_103_ce': (np.column_stack([x, ce]), big),
                    'F2_feature_level_103_ce_setctx_sibling': (np.column_stack([x, ctx_floor, ce, ce_ctx]), big)}
        for name, (F, prm) in variants.items():
            F = F.astype(np.float32)
            dtr = xgb.QuantileDMatrix(F[tb], y[tb])
            bst = xgb.train(prm, dtr, 3000, evals=[(xgb.QuantileDMatrix(F[dv], y[dv], ref=dtr), 'dev')],
                            early_stopping_rounds=100, verbose_eval=False)
            report(name, np.asarray(bst.inplace_predict(F, iteration_range=(0, bst.best_iteration + 1)), dtype=np.float64))
            md = FUS / 'late_models' / f'{name}_{TAG}'
            md.mkdir(parents=True, exist_ok=True)
            bst.save_model(str(md / 'model.json'))
            (md / 'config.json').write_text(json.dumps({'name': name, 'tag': TAG, 'rule': R[name]['rule'],
                                                       'best_iteration': int(bst.best_iteration),
                                                       'dev_f05': R[name]['dev_f05'], 'holdout_f05': R[name]['holdout_f05']}))
        (FUS / f'late_fusion_{TAG}.json').write_text(json.dumps(R, indent=1, default=str))
        log('done')


    if __name__ == '__main__':
        main()



def cmd_multi_ce_eval():
    """Which cross-encoders add information on top of each other? Late-fusion comparison on one fixed protocol.

Every member's logits are aligned to prune_v1 mask_top10 rows. Fold-A models are out-of-sample on fold B, dev and
holdout; full-train/listwise models are out-of-sample on all sample rows. So all configs train L2-style XGBoost on
fold B (decoy w=3), pick the rule on dev and score holdout once. Members whose file is missing are skipped.
Env MEMBERS: 'name=path;name=path'. Writes fusion/multi_ce_eval.json.
"""
    import gzip
    import itertools
    import json
    import os
    import sys
    import time
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import xgboost as xgb

    sys.path.insert(0, str(Path(__file__).parent))
    from distractor_check import decide, f05, partition_of

    HOME = Path.home()
    V4 = HOME / 'scratch/vishal_v4_artifacts'
    PRUNE = HOME / 'scratch/amazon_ml_2026/artifacts/prune_v1_copy'
    FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
    RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
    W = float(os.environ.get('DECOY_W', 3))


    def log(*a):
        print(time.strftime('%H:%M:%S'), *a, flush=True)


    def main():
        members = {}
        for item in os.environ['MEMBERS'].split(';'):
            n, p = item.split('=', 1)
            if Path(p).exists():
                members[n] = np.load(p).astype(np.float64)
            else:
                log('skip missing', n, p)
        gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
        cp = partition_of(gold)
        mask = np.load(PRUNE / 'mask_top10.npy')
        refs, tgts = [], []
        with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t')
                    refs.append(r); tgts.append(t)
        refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
        part = np.array([cp[r] for r in refs], dtype=object)
        S = np.load(FUS / 'stage1.npz')
        y = np.load(V4 / 'pairs/y.npy')[S['rows']].astype(np.float32)
        s1 = S['stage1'].astype(np.float64)
        s1f = np.where(s1 < 0.02, 0.0, s1)
        owned = set()
        with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) > 1 and q[1]:
                    owned.update(t.strip() for t in q[1].split(','))
        wt = np.where((y == 0) & np.array([t not in owned for t in tgts]), W, 1.0).astype(np.float32)
        tb, dv = part == 'trainB', part == 'dev'
        idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
        prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
        rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}
        grid = [(ru, round(float(t), 3)) for ru in ('expected', 'threshold') for t in np.arange(0.5, 0.981, 0.02)]
        params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                      max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)

        def macro(p, prob, ru, t):
            c = decide(rix[p], prob, ru, t, len(prefs[p]))
            pred = defaultdict(set)
            for i, tt in zip(rix[p][c], tgts[idx[p]][c]):
                pred[i].add(tt)
            return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

        names = sorted(members)
        configs = [(n,) for n in names] + [tuple(c) for c in itertools.combinations(names, 2)]
        if len(names) > 2:
            configs.append(tuple(names))
        R = {'members': names, 'decoy_weight': W, 'protocol': 'L2 on fold B, rule on dev, holdout once'}
        for cfg in configs:
            X = np.column_stack([s1f] + [np.nan_to_num(members[n], nan=0.0) for n in cfg]).astype(np.float32)
            dtr = xgb.QuantileDMatrix(X[tb], y[tb], weight=wt[tb])
            bst = xgb.train(params, dtr, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=dtr), 'dev')],
                            early_stopping_rounds=100, verbose_eval=False)
            it = (0, bst.best_iteration + 1)
            pdv = np.asarray(bst.inplace_predict(X[idx['dev']], iteration_range=it), dtype=np.float64)
            pho = np.asarray(bst.inplace_predict(X[idx['holdout']], iteration_range=it), dtype=np.float64)
            dev = {g: macro('dev', pdv, *g) for g in grid}
            best = max(dev, key=lambda k: (dev[k], k[1]))
            R['+'.join(cfg)] = {'rule': best, 'dev': round(dev[best], 6), 'holdout': round(macro('holdout', pho, *best), 6)}
            log('+'.join(cfg), R['+'.join(cfg)])
            (FUS / 'multi_ce_eval.json').write_text(json.dumps(R, indent=1))


    if __name__ == '__main__':
        main()



def cmd_overcommit_check():
    """Exclusivity-consistent probabilities: a record belongs to at most one S1, so its probabilities over S1s sum to <= 1.

For every candidate record, S = sum of stage-3b p over the S1 lists it appears in. Records with S > 1 are
over-committed: the model is sure of two owners at once (typically blank same-name ties). Per country: records with
S > 1.2, accepted links on them, blank share, and how many accepted links would fall below the F0.5 break-even (0.77)
after p' = p / max(1, S). Env LATE, SUB, OUT.
"""
    import json
    import os
    from collections import Counter

    import numpy as np

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    from france_census import RAW, load


    def main():
        Z = np.load(os.environ['LATE'], allow_pickle=True)
        ri, tg, p = Z['ref_idx'], Z['target'].astype(str), Z['prob'].astype(np.float64)
        refs, cty = [], []
        with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t'); ie, ic = head.index('entity_id'), head.index('country')
            for line in fh:
                q = line.rstrip('\n').split('\t'); refs.append(q[ie]); cty.append(q[ic])
        blank = set()
        for k in (2, 3):
            with open(RAW / f'test_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t'); ie, ia = head.index('entity_id'), head.index('business_address')
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if len(q) <= ia or not q[ia].strip():
                        blank.add(q[ie])
        uniq, inv = np.unique(tg, return_inverse=True)
        S = np.bincount(inv, weights=p)[inv]
        match = load(os.environ['SUB'])
        acc = np.array([t in set(match.get(refs[i], ())) for i, t in zip(ri, tg)])
        pc = np.array(cty, dtype=object)[ri]
        isb = np.array([t in blank for t in tg])
        pn = p / np.maximum(1.0, S)
        R = {}
        for c in ('US', 'India', 'France'):
            mc = pc == c
            oc = mc & (S > 1.2)
            R[c] = {'pairs': int(mc.sum()), 'accepted': int((mc & acc).sum()),
                    'overcommitted_pairs_S>1.2': int(oc.sum()), 'accepted_on_overcommitted': int((oc & acc).sum()),
                    'accepted_overcommitted_blank': int((oc & acc & isb).sum()),
                    'accepted_drop_below_0.77_after_norm': int((mc & acc & (p >= 0.77) & (pn < 0.77)).sum()),
                    'of_which_blank': int((mc & acc & (p >= 0.77) & (pn < 0.77) & isb).sum()),
                    'S_of_accepted_q50_90_99': np.quantile(S[mc & acc], [0.5, 0.9, 0.99]).round(3).tolist()}
        out = json.dumps(R, indent=1)
        open(os.environ['OUT'], 'w').write(out)
        print(out, flush=True)


    if __name__ == '__main__':
        main()



def cmd_pilot_matcher():
    """Pilot: entity-grouped train/dev/holdout, fixed LightGBM, dev-only threshold."""
    import argparse
    import gzip
    import json
    from pathlib import Path
    from metrics import macro_f05


    def features(pair):
        from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio
        a, b = pair['reference'], pair['target']
        values = []
        for field in ['name_norm', 'name_folded', 'address_norm', 'address_folded']:
            x, y = a[field], b[field]
            valid = bool(x and y)
            values.extend([ratio(x,y)/100 if valid else 0,
                           token_sort_ratio(x,y)/100 if valid else 0,
                           token_set_ratio(x,y)/100 if valid else 0,
                           int(x == y and valid), min(len(x),len(y))/max(1,len(x),len(y))])
        for field in ['address_numbers', 'postal_candidates']:
            x, y = set(a[field].split()), set(b[field].split())
            values.extend([len(x & y)/max(1,len(x | y)), int(bool(x and y) and not x & y),
                           int(not x), int(not y)])
        values.extend([int(a['address_missing']), int(b['address_missing'])])
        for channel in ['name_words', 'address_words', 'name_trigrams']:
            rank = pair['retrieval_ranks'][channel]
            values.append(0 if rank is None else 1/rank)
        return values


    def main():
        import numpy as np
        from sklearn.model_selection import GroupShuffleSplit
        from lightgbm import LGBMClassifier
        p = argparse.ArgumentParser(description=cmd_pilot_matcher.__doc__)
        p.add_argument('--data', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        args = p.parse_args()
        if args.output.exists():
            raise FileExistsError('Use a new output directory; do not overwrite a pilot')
        truth_rows = json.loads((args.data/'reference_truth.json').read_text())
        truth = {r['source1_entity_id']: set(r['gold_ids']) for r in truth_rows}
        parent = {sid: sid for sid in truth}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a,b):
            parent[find(a)] = find(b)
        owners = {}
        for sid, gold in truth.items():
            for tid in gold:
                if tid in owners:
                    union(sid, owners[tid])
                owners[tid] = sid
        ids, targets, labels, rows = [], [], [], []
        # Connect references sharing candidate targets too: no target leaks across splits.
        candidate_owners = {}
        with gzip.open(args.data/'pairs.jsonl.gz', 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                pair = json.loads(line)
                sid, tid = pair['source1_entity_id'], pair['target_entity_id']
                if tid in candidate_owners:
                    union(sid, candidate_owners[tid])
                candidate_owners[tid] = sid
                if tid in owners:
                    union(sid, owners[tid])
                if pair['label'] != int(tid in truth[sid]):
                    raise ValueError('Pair label disagrees with truth')
                ids.append(sid); targets.append(tid); labels.append(pair['label'])
                rows.append(features(pair))
                if i % 20000 == 0:
                    print(f'Features: {i:,} pairs', flush=True)
        anchors = sorted(truth)
        groups = [find(sid) for sid in anchors]
        if len(set(groups)) < 10:
            raise ValueError('Too few independent components; need more data, not a leaky split')
        pool, holdout = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=2026).split(anchors,groups=groups))
        train_rel, dev_rel = next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=2027).split(
            pool, groups=[groups[i] for i in pool]))
        partitions = {'train': {anchors[pool[i]] for i in train_rel},
                      'dev': {anchors[pool[i]] for i in dev_rel},
                      'holdout': {anchors[i] for i in holdout}}
        component_sets = {k: {find(sid) for sid in members} for k,members in partitions.items()}
        assert not (component_sets['train'] & component_sets['dev'])
        assert not (component_sets['train'] & component_sets['holdout'])
        assert not (component_sets['dev'] & component_sets['holdout'])
        for part, members in partitions.items():
            if len(members) < 30:
                raise ValueError(f'{part} has too few references: {len(members)}')
        x, y = np.asarray(rows,dtype=np.float32), np.asarray(labels)
        masks = {k: np.asarray([sid in members for sid in ids]) for k,members in partitions.items()}
        if len(set(y[masks['train']])) != 2:
            raise ValueError('Training needs both classes')
        model = LGBMClassifier(n_estimators=300, num_leaves=15, max_depth=-1,
            learning_rate=.05, min_child_samples=50, reg_lambda=5,
            n_jobs=4, random_state=2026, verbosity=-1)
        model.fit(x[masks['train']], y[masks['train']])
        def predictions(part):
            idx = np.flatnonzero(masks[part])
            return idx, model.predict_proba(x[idx])[:,1]
        def score(part, idx, probabilities, threshold):
            pred = {sid: set() for sid in partitions[part]}
            for j, probability in zip(idx,probabilities):
                if probability >= threshold:
                    pred[ids[j]].add(targets[j])
            return macro_f05({sid: truth[sid] for sid in pred}, pred)
        idx, probabilities = predictions('dev')
        grid = [(float(t), score('dev',idx,probabilities,t)) for t in np.linspace(.05,.95,91)]
        threshold, dev_score = max(grid, key=lambda r: (r[1],r[0]))
        idx, probabilities = predictions('holdout')
        holdout_score = score('holdout',idx,probabilities,threshold)
        args.output.mkdir(parents=True)
        model.booster_.save_model(str(args.output/'model.txt'))
        import importlib.metadata
        versions = {name: importlib.metadata.version(name) for name in ['numpy', 'scikit-learn', 'lightgbm', 'rapidfuzz']}
        report = {'macro_f05_dev': dev_score, 'macro_f05_holdout': holdout_score,
           'threshold_selected_on_dev': threshold, 'references': {k:len(v) for k,v in partitions.items()},
           'pairs': {k:int(v.sum()) for k,v in masks.items()}, 'components': len(set(groups)),
           'feature_count': x.shape[1], 'versions': versions,
           'warning': 'Pilot on retrieval-selected 850-reference prefix; not a leaderboard estimate. Model frozen before holdout. Unretrieved gold and singletons included.'}
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        (args.output/'split.json').write_text(json.dumps({k:sorted(v) for k,v in partitions.items()},indent=2))
        (args.output/'threshold_grid.json').write_text(json.dumps(grid))
        print(json.dumps(report,indent=2),flush=True)


    if __name__ == '__main__':
        main()



def cmd_score_decomp():
    """Where is the leaderboard score lost: France vs US+India, without labels for test.

Expected macro F0.5 of a decision set from the model's own probabilities (Monte Carlo: each candidate true with its
probability, plus gold links outside the top 10 ~ Poisson(lambda) per reference, lambda measured on the holdout).
1. Holdout check (labels): T5's estimate vs T5's actual holdout F0.5, per country. The method is trusted only if they agree.
2. Leaderboard file (0.989635 = T5 US/India + census v2 France): US and India estimated on test with T5's probabilities
   and decisions (+ the holdout correction), then France implied: F_FR = (LB - w_US F_US - w_IN F_IN) / w_FR.
3. Also the models' own France estimate (what the France model believes) and U5's estimates.
"""
    import csv
    import gzip
    import json
    import sys
    from collections import defaultdict
    from pathlib import Path

    import numpy as np

    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    AY = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
    ART = ME / 'artifacts'
    LB = 0.989635
    sys.path[:0] = [str(AY / 'business_entity_resolution/src')]


    def mc(P, C, lam, S=64, seed=0):
        rng = np.random.default_rng(seed)
        pred = C.sum(1)
        Fs = np.zeros(len(P)); wrong = np.zeros(len(P))
        for _ in range(S):
            Z = rng.random(P.shape) < P
            g = Z.sum(1) + (rng.poisson(lam, len(P)) if np.any(lam) else 0)
            tp = (Z & C).sum(1)
            with np.errstate(divide='ignore', invalid='ignore'):
                pr = tp / np.maximum(pred, 1); rc = tp / np.maximum(g, 1)
                F = np.where((g == 0) & (pred == 0), 1.0, np.where(tp == 0, 0.0, 1.25 * pr * rc / (0.25 * pr + rc)))
            Fs += F; wrong += F < 1
        return Fs / S, wrong / S


    def pad(groups, n, *arrs):
        out = [np.zeros((n, 10), dtype=a.dtype) for a in arrs]
        for i, js in enumerate(groups):
            js = js[:10]
            for o, a in zip(out, arrs):
                o[i, :len(js)] = a[js]
        return out


    def main():
        from distractor_check import decide, f05, partition_of
        R = {}
        # ---------- 1. holdout
        gold = {k: set(v) for k, v in json.loads((ART / 'matcher_v4/pairs/gold.json').read_text()).items()}
        cp = partition_of(gold)
        cty_s = json.loads((ART / 'matcher_v4/pairs/countries.json').read_text())
        m10 = np.load(ART / 'prune_v1/mask_top10.npy').astype(bool)
        pr, pt = [], []
        with gzip.open(ART / 'matcher_v4/pairs/pairs.tsv.gz', 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if m10[i]:
                    a, b = line.rstrip('\n').split('\t'); pr.append(a); pt.append(b)
        pr, pt = np.array(pr, dtype=object), np.array(pt, dtype=object)
        S1 = np.load(ART / 'fusion/stage1.npz')
        y = np.load(ART / 'matcher_v4/pairs/y.npy')[S1['rows']] > 0
        for tag, mdir in (('T5', 'boundary_models_t5'), ('T6', 'boundary_models_t6')):
            ph = np.load(ART / f'fusion/{mdir}/stage3b_prob.npy')
            rule = json.load(open(ART / f'fusion/{mdir}/stage3b_meta.json'))['rule']
            prefs = sorted(s for s in gold if cp[s] == 'holdout')
            pos = {s: i for i, s in enumerate(prefs)}
            idx = np.flatnonzero(np.array([cp[r] == 'holdout' for r in pr]))
            rix = np.array([pos[r] for r in pr[idx]])
            ch = decide(rix, ph[idx], rule[0], rule[1], len(prefs))
            groups = [[] for _ in prefs]
            for k, i in enumerate(rix):
                groups[i].append(k)
            P, C, Y = pad(groups, len(prefs), ph[idx], ch, y[idx])
            pred = defaultdict(set)
            for i, t in zip(rix[ch], pt[idx][ch]):
                pred[i].add(t)
            actual = np.array([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs)])
            outside = np.array([len(gold[s]) - int(Y[i].sum()) for i, s in enumerate(prefs)], dtype=np.float64)
            cty = np.array([cty_s[s] for s in prefs])
            est, wrong = mc(P, C.astype(bool), outside.mean())
            est0, _ = mc(P, C.astype(bool), 0.0)
            R[f'holdout_{tag}'] = {'rule': rule}
            for c in ('US', 'India', 'all'):
                k = np.ones(len(prefs), bool) if c == 'all' else cty == c
                R[f'holdout_{tag}'][c] = {'refs': int(k.sum()), 'actual_F05': round(float(actual[k].mean()), 6),
                                          'estimated_F05': round(float(est[k].mean()), 6),
                                          'estimated_without_blocking_term': round(float(est0[k].mean()), 6),
                                          'lambda_outside_top10': round(float(outside[k].mean()), 4),
                                          'actual_refs_wrong_share': round(float((actual[k] < 1).mean()), 4),
                                          'estimated_refs_wrong_share': round(float(wrong[k].mean()), 4)}
            print(tag, json.dumps(R[f'holdout_{tag}'], indent=1), flush=True)
        lam = {c: R['holdout_T5'][c]['lambda_outside_top10'] for c in ('US', 'India')}
        corr = {c: R['holdout_T5'][c]['actual_F05'] - R['holdout_T5'][c]['estimated_F05'] for c in ('US', 'India')}
        # ---------- 2. test
        refs, tgts = [], []
        with open(ART / 'submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
            next(f)
            for line in f:
                s, c = line.rstrip('\n').split('\t')
                refs.append(s); tgts.extend(c.split(','))
        n = len(refs)
        assert len(tgts) == 10 * n
        T = np.array(tgts, dtype=object).reshape(n, 10)
        cty = {}
        with open(ME / 'data/dataset/test/test_source1.tsv', encoding='utf-8', newline='') as fh:
            for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                cty[row['entity_id']] = row['country']
        ct = np.array([cty[r] for r in refs])
        w = {c: float((ct == c).mean()) for c in ('US', 'India', 'France')}

        def chosen(path):
            sub = {}
            with open(path, encoding='utf-8') as f:
                next(f)
                for l in f:
                    r, m = l.rstrip('\n').split('\t'); sub[r] = set(x for x in m.split(',') if x)
            C = np.array([[t in sub[r] for t in T[i]] for i, r in enumerate(refs)])
            extra = np.array([len(sub[r] - set(T[i])) for i, r in enumerate(refs)])  # rescue links outside top 10
            return C, extra

        def est_country(P, C, c, lamc):
            k = ct == c
            e, wr = mc(P[k], C[k], lamc)
            return float(e.mean()), float(wr.mean()), int(k.sum())
        p5 = np.load(ART / 'submission_T5_own/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)
        C5, _ = chosen(ART / 'submission_T5_own/expected_0.66/matching_results.tsv')
        L = {}
        for c in ('US', 'India'):
            e, wr, m = est_country(p5, C5, c, lam[c])
            L[c] = {'refs': m, 'weight': round(w[c], 5), 'estimated_F05': round(e, 6), 'corrected_F05': round(e + corr[c], 6),
                    'estimated_refs_wrong': int(round(wr * m))}
        e, wr, m = est_country(p5, C5, 'France', np.mean(list(lam.values())))
        L['France_T5_self_estimate'] = {'refs': m, 'estimated_F05': round(e, 6), 'estimated_refs_wrong': int(round(wr * m))}
        for kind in ('estimated_F05', 'corrected_F05'):
            f_ui = w['US'] * L['US'][kind] + w['India'] * L['India'][kind]
            L[f'France_implied_from_LB_using_{kind}'] = round((LB - f_ui) / w['France'], 5)
            L[f'US+India_{kind}'] = round(f_ui / (w['US'] + w['India']), 6)
        for kind in ('estimated_F05', 'corrected_F05'):
            ff = L[f'France_implied_from_LB_using_{kind}']
            L[f'points_lost_{kind}'] = {'US+India': round((w['US'] + w['India']) * (1 - L[f'US+India_{kind}']), 5),
                                        'France': round(w['France'] * (1 - ff), 5)}
        # sensitivity: US/India off by +-0.001 moves implied France by 0.001*(1-w_FR)/w_FR
        L['implied_France_sensitivity_per_0.001_US+India_error'] = round(0.001 * (1 - w['France']) / w['France'], 5)
        R['leaderboard_file'] = L
        print('leaderboard file', json.dumps(L, indent=1), flush=True)
        # ---------- 3. U5: US/India from the bag, France self-estimate with the France-trained model
        try:
            pb = np.load('/scratch/scai/mtech/aib262467/t6bag/test/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)
            pF = np.load(ART / 'submission_FRtrained_w4/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)
            CU, extra = chosen('/scratch/scai/mtech/aib262467/final_u5/out/matching_results.tsv')
            U = {}
            for c in ('US', 'India'):
                e, wr, m = est_country(pb, CU, c, lam[c])
                U[c] = {'estimated_F05': round(e, 6), 'corrected_F05': round(e + corr[c], 6)}
            e, wr, m = est_country(pF, CU, 'France', np.mean(list(lam.values())))
            U['France_FRw4_self_estimate'] = {'estimated_F05': round(e, 6), 'estimated_refs_wrong': int(round(wr * m)),
                                              'note': 'rescue links outside the top 10 ignored'}
            R['U5'] = U
            print('U5', json.dumps(U, indent=1), flush=True)
        except Exception as ex:  # noqa: BLE001
            print('U5 part skipped:', ex, flush=True)
        Path('/scratch/scai/mtech/aib262467/score_decomp.json').write_text(json.dumps(R, indent=1))


    if __name__ == '__main__':
        main()



def cmd_seg_calib():
    """Calibration of T6 by error segment on dev and holdout: is stage-3b under-confident where it misses real copies?

Segments from usin_errors.rel (name relation x address relation, blank separate). Per segment and p bin: pairs, mean
p, true rate, per split. A segment whose true rate sits well above mean p near the F0.5 margin (0.3-0.8) is a
recalibration candidate. Env OUT, PROB.
"""
    import gzip
    import json
    import os
    import sys
    from collections import defaultdict
    from pathlib import Path

    import numpy as np

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    sys.path.insert(0, str(Path(__file__).parent))
    from distractor_check import partition_of
    from usin_errors import rel

    H = Path('/home/scai/mtech/aib262015/scratch')
    V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
    BINS = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.01]


    def main():
        S = np.load(H / 'amazon_ml_2026/artifacts/fusion/stage1.npz')
        y = np.load(H / 'vishal_v4_artifacts/pairs/y.npy')[S['rows']]
        gold = {k: set(v) for k, v in json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text()).items()}
        cp = partition_of(gold)
        mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
        refs, tgts = [], []
        with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t'); refs.append(r); tgts.append(t)
        need = set(refs) | set(tgts)
        text = {}
        for k in (1, 2, 3):
            with open(H / f'amazon_ml_2026/data/dataset/train/train_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if q[ie] in need:
                        text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
        P = np.load(os.environ.get('PROB', str(V / 'fusion/boundary_models_t6/stage3b_prob.npy')))
        acc = defaultdict(lambda: np.zeros(3))
        for j, (r, t) in enumerate(zip(refs, tgts)):
            sp = cp[r]
            if sp == 'train' or not (0.05 <= P[j] < 0.95):
                continue
            rn, ra, cty = text.get(r, ('', '', ''))
            tn, ta, _ = text.get(t, ('', '', ''))
            nm, ad = rel(rn, ra, tn, ta)
            seg = f'{nm}|{ad if ad == "blank" else ad.split("_")[0] + "_" + ad.split("_")[-1]}'
            b = int(np.searchsorted(BINS, P[j], side='right') - 1)
            k = (sp, cty, seg, f'{BINS[b]}-{BINS[b + 1]}')
            acc[k] += [1, P[j], y[j]]
        rows = {}
        for (sp, cty, seg, b), (n, sp_, sy) in acc.items():
            rows.setdefault(f'{cty}|{seg}|{b}', {})[sp] = [int(n), round(sp_ / n, 3), round(sy / n, 3)]
        keep = {k: v for k, v in rows.items() if sum(x[0] for x in v.values()) >= 40}
        out = json.dumps(dict(sorted(keep.items())), indent=0)
        open(os.environ['OUT'], 'w').write(out)
        print(out, flush=True)


    if __name__ == '__main__':
        main()



def cmd_shap_misclassified():
    """SHAP feature attributions for the feature matcher's misclassified holdout pairs."""
    import gzip
    import hashlib
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import xgboost as xgb

    sys.path.insert(0, str(Path(__file__).parent))
    from submission_pipeline import decide


    def load_pairs(d):
        pairs = []
        with gzip.open(d / 'pairs.tsv.gz', 'rt') as f:
            for line in f:
                ref, target = line.rstrip('\n').split('\t')
                pairs.append((ref, target))
        return pairs


    def partition_of(gold):
        parent = {s: s for s in gold}

        def find(s):
            while parent[s] != s:
                parent[s] = parent[parent[s]]
                s = parent[s]
            return s
        owner = {}
        for s, targets in gold.items():
            for t in targets:
                if t in owner:
                    parent[find(s)] = find(owner[t])
                owner[t] = s

        def part(s):
            h = int(hashlib.sha256(find(s).encode()).hexdigest()[:8], 16) % 100
            return 'train' if h < 70 else 'dev' if h < 85 else 'holdout'
        return {s: part(s) for s in gold}


    def top_features(mean_abs_shap, names, k=20):
        order = np.argsort(-mean_abs_shap)[:k]
        return [(names[i], round(float(mean_abs_shap[i]), 5)) for i in order]


    def main():
        root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'scratch/vishal_v4_artifacts'
        nthread = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        pairs_dir = root / 'pairs'
        model_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else root / 'model_xgb'
        row_mask = np.load(sys.argv[4]) if len(sys.argv) > 4 else None

        x = np.load(pairs_dir / 'x.npy')
        y = np.load(pairs_dir / 'y.npy')
        pairs = load_pairs(pairs_dir)
        pair_ref = np.array([p[0] for p in pairs], dtype=object)
        gold = {k: set(v) for k, v in json.loads((pairs_dir / 'gold.json').read_text()).items()}
        report = json.loads((model_dir / 'report.json').read_text())
        names = report['features']
        rule, threshold = report['decision']['rule'], report['decision']['threshold']

        booster = xgb.Booster()
        booster.load_model(str(model_dir / 'model.json'))
        booster.set_param({'device': 'cuda', 'nthread': nthread})

        partition = partition_of(gold)
        pair_part = np.array([partition[s] for s in pair_ref], dtype=object)
        max_rank = report.get('keep', 0)
        in_range = (x[:, names.index('fused_rank')] <= max_rank) if max_rank else np.ones(len(y), bool)
        mask_hold = (pair_part == 'holdout') & in_range
        if row_mask is not None:
            mask_hold &= row_mask
        idx = np.flatnonzero(mask_hold)
        print(f'holdout pairs: {len(idx)}', flush=True)

        refs = sorted(s for s in gold if partition[s] == 'holdout')
        position = {s: i for i, s in enumerate(refs)}
        ref_idx = np.array([position[s] for s in pair_ref[idx]])

        print('predicting probabilities...', flush=True)
        prob = np.asarray(booster.inplace_predict(x[idx], iteration_range=(0, report['best_iteration'] + 1)),
                           dtype=np.float64)
        chosen = decide(ref_idx, prob, rule, threshold, len(refs))
        y_hold = y[idx]

        false_positive = chosen & (y_hold == 0)
        missed = (~chosen) & (y_hold == 1)
        correct_positive = chosen & (y_hold == 1)
        correct_negative = (~chosen) & (y_hold == 0)
        print(f'false_positive={false_positive.sum()} missed={missed.sum()} '
              f'correct_positive={correct_positive.sum()} correct_negative={correct_negative.sum()}', flush=True)

        rng = np.random.default_rng(2026)
        buckets = {}
        for label, mask in [('false_positive', false_positive), ('missed_candidate', missed),
                             ('correct_positive', correct_positive), ('correct_negative', correct_negative)]:
            rows = np.flatnonzero(mask)
            if len(rows) > 60000:
                rows = rng.choice(rows, 60000, replace=False)
            buckets[label] = rows
        all_rows = np.concatenate(list(buckets.values()))
        print(f'computing SHAP on {len(all_rows)} sampled rows...', flush=True)
        contribs = booster.predict(xgb.DMatrix(x[idx][all_rows], feature_names=names), pred_contribs=True,
                                    iteration_range=(0, report['best_iteration'] + 1))
        feat_contribs = contribs[:, :-1]
        offsets = np.cumsum([0] + [len(r) for r in buckets.values()])

        for (label, rows), start, end in zip(buckets.items(), offsets[:-1], offsets[1:]):
            if len(rows) == 0:
                print(f'{label}: no samples')
                continue
            c = feat_contribs[start:end]
            mean_abs = np.abs(c).mean(axis=0)
            mean_signed = c.mean(axis=0)
            print(f'\n=== {label} (n={len(rows)}) top-15 by mean |SHAP| ===')
            order = np.argsort(-mean_abs)[:15]
            for i in order:
                print(f'  {names[i]:35s} mean|shap|={mean_abs[i]:.5f}  mean_signed={mean_signed[i]:+.5f}')


    if __name__ == '__main__':
        main()



def cmd_shift_audit():
    """Train-sample -> test transfer audit of every stage-3b input (the listwise lesson: holdout does not see test shift).

On pairs every base decider agrees on, the inputs should look alike in the sample and on test:
  sure-match     v5 >= 0.99 and chunk2 logit >= 9
  sure-nonmatch  v5 <= 0.02 and chunk2 logit <= -5
For each feature and for the stage-3b probability: sample vs test quantiles (p1/p10/p50/p90) and the KS distance, per
group and for test countries separately. Features ranked by the worst KS on sure-match pairs. A large shift there is
the listwise failure signature.
Env CACHE (boundary_models_*/exp2_cache.npz), META (stage3b_meta.json), P3_SAMPLE (stage3b_prob.npy),
TEST_DIR (stage3b_predict output with features.npy + late_scores.npz). Writes TEST_DIR/shift_audit.json.
"""
    import json
    import os
    from pathlib import Path

    import numpy as np

    HOME = Path.home()
    FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
    RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/test'
    C2S = '/home/scai/mtech/aib262140/scratch/amazon_ml_2026/ftchunk2/ce_logit_sample.npy'


    def ks(a, b):
        a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
        if not len(a) or not len(b):
            return float('nan')
        g = np.unique(np.concatenate([a, b]))
        if len(g) > 4000:
            g = np.quantile(g, np.linspace(0, 1, 4000))
        return float(np.max(np.abs(np.searchsorted(a, g, side='right') / len(a) - np.searchsorted(b, g, side='right') / len(b))))


    def qs(a):
        a = a[np.isfinite(a)]
        return [round(float(np.quantile(a, q)), 3) for q in (0.01, 0.1, 0.5, 0.9)] if len(a) else []


    def main():
        C = np.load(os.environ['CACHE'])
        names = json.load(open(os.environ['META']))['features']
        Xs = np.column_stack([C['F'], C['Frev'], C['Fnum']]).astype(np.float32)
        names = names[:Xs.shape[1]]
        S = np.load(FUS / 'stage1.npz')
        v5s = np.where(S['stage1'] < 0.02, 0.0, S['stage1'])
        c2s = np.nan_to_num(np.load(C2S))
        p3s = np.load(os.environ['P3_SAMPLE'])
        T = Path(os.environ['TEST_DIR'])
        Z = np.load(T / 'late_scores.npz', allow_pickle=True)
        Xt = np.load(T / 'features.npy', mmap_mode='r')
        v5t, c2t, p3t, ri = Z['stage1'], Z['ce'], Z['prob'], Z['ref_idx']
        cty = []
        with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ic = head.index('country')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                cty.append(q[ic] if len(q) > ic else '')
        pc = np.array(cty, dtype=object)[ri]
        rng = np.random.default_rng(0)
        groups = {'sure_match': ((v5s >= 0.99) & (c2s >= 9), (v5t >= 0.99) & (c2t >= 9)),
                  'sure_nonmatch': ((v5s <= 0.02) & (c2s <= -5), (v5t <= 0.02) & (c2t <= -5))}
        R = {'n_features': len(names), 'groups': {}}
        for g, (ms, mt) in groups.items():
            js = np.flatnonzero(ms); js = rng.choice(js, min(200000, len(js)), replace=False)
            jt = np.flatnonzero(mt); jt = np.sort(rng.choice(jt, min(400000, len(jt)), replace=False))
            Xtg = np.asarray(Xt[jt, :Xs.shape[1]])
            rows = []
            for k, nm in enumerate(names + ['stage3b_p']):
                a = Xs[js, k] if nm != 'stage3b_p' else p3s[js]
                b = Xtg[:, k] if nm != 'stage3b_p' else p3t[jt]
                by = {c: round(ks(a, b[pc[jt] == c]), 4) for c in ('US', 'India', 'France')}
                rows.append({'feature': nm, 'ks_all': round(ks(a, b), 4), 'ks_by_country': by,
                             'sample_q': qs(a), 'test_q': qs(b)})
            rows.sort(key=lambda r: -r['ks_all'])
            R['groups'][g] = {'n_sample': int(len(js)), 'n_test': int(len(jt)), 'features': rows}
            print(g, [(r['feature'], r['ks_all'], r['ks_by_country']) for r in rows[:12]], flush=True)
        (T / 'shift_audit.json').write_text(json.dumps(R, indent=1))


    if __name__ == '__main__':
        main()



def cmd_shift_ens_check():
    """Two last-day checks on the stage-3b decision.

A. Label shift. Test has ~5.8 S2/S3 records per S1 against 4.67 in train, so about twice the distractors per S1. If the
   extra records land in the ambiguous region, test posteriors are too high. EM prior re-estimation (Saerens et al.
   2002) on T5's test probabilities per country, with the labelled holdout as the control (EM there should return the
   true rate), plus the share of candidates in the ambiguous band.
B. Stage-3b ensemble. Every stage-3b variant scores the same 1M-pair train sample; average them (probability and logit
   means), choose the expected-F0.5 floor on dev, report holdout, against T5 alone.
"""
    import gzip
    import json
    import sys
    from collections import defaultdict
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).parent))
    from distractor_check import decide, f05, partition_of

    H = Path('/home/scai/mtech/aib262015/scratch')
    V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
    MODELS = {'t1b': H / 'amazon_ml_2026/artifacts/fusion/boundary_models_c2_w1/stage3b_prob.npy',
              't4': H / 'amazon_ml_2026/artifacts/fusion/boundary_models_t4/stage3b_prob.npy',
              't5': V / 'fusion/boundary_models_t5/stage3b_prob.npy',
              't6': V / 'fusion/boundary_models_t6/stage3b_prob.npy',
              'frrobust': V / 'submission_FRrobust/robust_sample_prob.npy'}


    def em_prior(p, pi_tr, it=200):
        q = float(p.mean())
        for _ in range(it):
            w, v = q / pi_tr, (1 - q) / (1 - pi_tr)
            pp = w * p / (w * p + v * (1 - p))
            q = float(pp.mean())
        return q


    def main():
        S = np.load(H / 'amazon_ml_2026/artifacts/fusion/stage1.npz')
        y = np.load(H / 'vishal_v4_artifacts/pairs/y.npy')[S['rows']].astype(np.int8)
        gold = {k: set(v) for k, v in json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text()).items()}
        cp = partition_of(gold)
        mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
        refs, tgts = [], []
        with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t'); refs.append(r); tgts.append(t)
        refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
        part = np.array([cp[r] for r in refs], dtype=object)
        idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
        prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
        rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}

        def macro(p, chosen):
            pred = defaultdict(set)
            for i, tt in zip(rix[p][chosen], tgts[idx[p]][chosen]):
                pred[i].add(tt)
            return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

        P = {}
        for k, f in MODELS.items():
            try:
                a = np.load(f).astype(np.float64)
                if len(a) == len(y):
                    P[k] = a
                print('model', k, len(a), flush=True)
            except Exception as e:
                print('model', k, 'missing', e, flush=True)

        # A. label shift
        h = idx['holdout']
        pi_tr = float(y[h].mean())
        band = lambda p: round(float(((p > 0.1) & (p < 0.9)).mean()), 4)
        print('A holdout: true rate', round(pi_tr, 4), 'mean p', round(float(P['t5'][h].mean()), 4), 'EM prior',
              round(em_prior(P['t5'][h], pi_tr), 4), 'band 0.1-0.9', band(P['t5'][h]), flush=True)
        Z = np.load(V / 'submission_T5_own/late_scores.npz', allow_pickle=True)
        cty = []
        with open(H / 'amazon_ml_2026/data/dataset/test/test_source1.tsv', encoding='utf-8') as fh:
            ic = fh.readline().rstrip('\n').split('\t').index('country')
            for line in fh:
                q = line.rstrip('\n').split('\t'); cty.append(q[ic] if len(q) > ic else '')
        pc = np.array(cty, dtype=object)[Z['ref_idx']]
        pt = Z['prob'].astype(np.float64)
        for c in ('US', 'India', 'France'):
            m = pc == c
            print('A test', c, 'mean p', round(float(pt[m].mean()), 4), 'EM prior', round(em_prior(pt[m], pi_tr), 4),
                  'band 0.1-0.9', band(pt[m]), flush=True)

        # B. ensemble
        grid = [round(float(t), 2) for t in np.arange(0.5, 0.801, 0.02)]

        def evaluate(name, prob):
            dev = {t: macro('dev', decide(rix['dev'], prob[idx['dev']], 'expected', t, len(prefs['dev']))) for t in grid}
            t = max(dev, key=lambda k: (dev[k], k))
            ho = macro('holdout', decide(rix['holdout'], prob[idx['holdout']], 'expected', t, len(prefs['holdout'])))
            print('B', name, 'rule', t, 'dev', round(dev[t], 6), 'holdout', round(ho, 6), flush=True)

        for k in P:
            evaluate(k, P[k])
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        sig = lambda z: 1 / (1 + np.exp(-z))
        for combo in (('t5', 't6'), ('t5', 't6', 't4'), ('t5', 't6', 't4', 't1b'), tuple(P)):
            if all(c in P for c in combo):
                evaluate('mean:' + '+'.join(combo), np.mean([P[c] for c in combo], axis=0))
                evaluate('logit:' + '+'.join(combo), sig(np.mean([lg(P[c]) for c in combo], axis=0)))


    if __name__ == '__main__':
        main()



def cmd_sibling_noise_check():
    """Do true aliases share their address noise with the S1's other copies? Train check (dev + holdout).

For a candidate with no name word in common with its S1 and a non-blank address: noise = address tokens of the S1's
sure copies (T6 p >= 0.9, sharing a name word) that are absent from the S1's own address. Shares = the candidate's
address has at least one such noise token (tokens of 2+ characters). True rate by shares yes/no, overall and in the
F0.5 margin (0.2 <= p < 0.8), by country. Env OUT, PROB.
"""
    import gzip
    import json
    import os
    import re
    import sys
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    sys.path.insert(0, str(Path(__file__).parent))
    from distractor_check import partition_of
    from france_census import core, fold

    H = Path('/home/scai/mtech/aib262015/scratch')
    V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
    toks = lambda a: {t for t in re.sub(r'[^a-z0-9/-]+', ' ', fold(a)).split() if len(t) >= 2}


    def main():
        S = np.load(H / 'amazon_ml_2026/artifacts/fusion/stage1.npz')
        y = np.load(H / 'vishal_v4_artifacts/pairs/y.npy')[S['rows']]
        gold = {k: set(v) for k, v in json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text()).items()}
        cp = partition_of(gold)
        mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
        refs, tgts = [], []
        with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t'); refs.append(r); tgts.append(t)
        need = set(refs) | set(tgts)
        text = {}
        for k in (1, 2, 3):
            with open(H / f'amazon_ml_2026/data/dataset/train/train_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if q[ie] in need:
                        text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
        P = np.load(os.environ.get('PROB', str(V / 'fusion/boundary_models_t6/stage3b_prob.npy')))
        by = defaultdict(list)
        for j, r in enumerate(refs):
            if cp[r] != 'train':
                by[r].append(j)
        C = Counter(); T = Counter()
        for r, js in by.items():
            rn, ra, cty = text.get(r, ('', '', ''))
            rc, A1 = set(core(rn)), toks(ra)
            sure = [j for j in js if P[j] >= 0.9 and set(core(text.get(tgts[j], ('', ''))[0])) & rc]
            for j in js:
                tn, ta, _ = text.get(tgts[j], ('', '', ''))
                if not ta.strip() or set(core(tn)) & rc:
                    continue
                noise = set().union(*[toks(text.get(tgts[s], ('', ''))[1]) for s in sure if s != j]) - A1 if sure else set()
                sh = 'shares' if toks(ta) & noise else ('no_share' if noise else 'no_sibling_noise')
                for band in ('all', 'margin') if 0.2 <= P[j] < 0.8 else ('all',):
                    k = (cty, band, sh); C[k] += 1; T[k] += int(y[j])
        R = {str(k): [C[k], round(T[k] / C[k], 3)] for k in sorted(C)}
        open(os.environ['OUT'], 'w').write(json.dumps(R, indent=0))
        print(json.dumps(R, indent=0), flush=True)


    if __name__ == '__main__':
        main()



def cmd_t1_empty_check():
    """Why does T1 leave 16.9k references empty that S5 matched with near-certainty?

For each such reference: S5's match, and T1's view of every candidate (stage-3b p3, L2, v5, chunk2, listwise), with
text. Summary: listwise logit distribution on these pairs vs pairs every decider agrees on, and by candidate position
in the list (a position or length effect would show here). Writes fusion/t1_empty_check.txt.
"""
    import re
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np

    HOME = Path.home()
    A = HOME / 'scratch/amazon_ml_2026/artifacts'
    RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/test'
    VCAND = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/submission_v5_top10/candidate_pairs.tsv')
    out = []


    def say(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True); out.append(s)


    def load(f):
        d = {}
        with open(f, encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                s, m = (line.rstrip('\n').split('\t') + [''])[:2]
                d[s] = [x for x in m.split(',') if x]
        return d


    def main():
        t1 = load(A / 'submission_T1_s3b_c2lw/expected_0.6/matching_results.tsv')
        s5 = load(A / 'submission_S5/threshold_0.68/matching_results.tsv')
        Z = np.load(A / 'submission_T1_s3b_c2lw/late_scores.npz', allow_pickle=True)
        ri, tg = Z['ref_idx'], Z['target'].astype(str)
        p3, l2, c2, v5 = Z['prob'], Z['l2'], Z['ce'], Z['stage1']
        refs, cty = [], {}
        with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, ic = head.index('entity_id'), head.index('country')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                refs.append(q[ie]); cty[q[ie]] = q[ic]
        pos = {r: i for i, r in enumerate(refs)}
        lw = {}
        for k in (0, 1):
            z = np.load(A / f'listwise/ce_listwise_shard{k}of2.npz', allow_pickle=True)
            for r, t, v in zip(z['ref'], z['target'], z['logit']):
                lw[(str(r), str(t))] = float(v)
        say('listwise test pairs', len(lw))
        cand = {}
        with open(VCAND, encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                s, c = line.rstrip('\n').split('\t')
                cand[s] = c.split(',') if c else []
        newly = [r for r in refs if not t1.get(r) and s5.get(r)]
        say('newly empty', len(newly))
        idx = defaultdict(dict)
        nset = {pos[r] for r in newly}
        for j in np.flatnonzero(np.isin(ri, list(nset))):
            idx[ri[j]][tg[j]] = j
        L_new, P_new = [], Counter()
        for r in newly:
            for t in s5[r]:
                L_new.append(lw.get((r, t), np.nan))
                P_new[cand[r].index(t) + 1 if t in cand[r] else -1] += 1
        L_new = np.array(L_new)
        ok = (v5 >= 0.99) & (c2 >= 9)
        samp = np.random.default_rng(0).choice(np.flatnonzero(ok), 200000, replace=False)
        L_ok = np.array([lw.get((refs[ri[j]], tg[j]), np.nan) for j in samp])
        q = lambda a: {k: round(float(np.nanquantile(a, k)), 2) for k in (0.1, 0.25, 0.5, 0.75, 0.9)}
        say('listwise logit on the dropped S5 matches:', q(L_new), 'nan', int(np.isnan(L_new).sum()))
        say('listwise logit on pairs with v5>=0.99 & chunk2>=9:', q(L_ok))
        say('candidate-list position of the dropped S5 match:', dict(sorted(P_new.items())))
        pos_ok = Counter()
        for j in samp[:50000]:
            r = refs[ri[j]]
            pos_ok[cand[r].index(tg[j]) + 1 if tg[j] in cand[r] else -1] += 1
        say('candidate-list position of agreed pairs:', dict(sorted(pos_ok.items())))
        say('by country:', Counter(cty[r] for r in newly))
        need = set(newly[:4000]) | {t for r in newly[:4000] for t in cand[r]}
        text = {}
        for k in (1, 2, 3):
            with open(RAW / f'test_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
                for line in fh:
                    q_ = line.rstrip('\n').split('\t')
                    if len(q_) > iaddr and q_[ie] in need:
                        text[q_[ie]] = (q_[iname], q_[iaddr])
        rng = np.random.default_rng(1)
        for r in rng.choice(newly[:4000], 30, replace=False):
            rn, ra = text.get(r, ('', ''))
            say(f'\n**{r}** [{cty[r]}] {rn} | {ra}')
            for n, t in enumerate(cand[r], 1):
                j = idx[pos[r]].get(t)
                tn, ta = text.get(t, ('', ''))
                flag = 'S5' if t in s5[r] else '  '
                vals = f'p3={p3[j]:.2f} l2={l2[j]:.2f} v5={v5[j]:.2f} c2={c2[j]:5.1f} lw={lw.get((r, t), float("nan")):5.1f}' if j is not None else ''
                say(f'  {n:2d} {flag} {vals} | {tn} | {ta}')
        (A / 'fusion/t1_empty_check.txt').write_text('\n'.join(out))


    if __name__ == '__main__':
        main()



def cmd_t5_shift_audit():
    """Transfer audit for T5 (the shift_audit method, extended to T5): on pairs v5 and chunk2 agree on, compare
sample vs test distributions (KS distance, quantiles; per test country) of every stage-3b input (T4 cache vs T4 test
features), the 26 ownership columns, and the T5 probability. Large KS on sure-match pairs = the listwise failure mode.
Writes artifacts/submission_T5_census/t5_shift_audit.json."""
    import json
    from pathlib import Path
    import numpy as np
    A = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    def ks(a, b):
        a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
        if not len(a) or not len(b):
            return float('nan')
        g = np.unique(np.concatenate([a, b]))
        if len(g) > 4000:
            g = np.quantile(g, np.linspace(0, 1, 4000))
        return float(np.max(np.abs(np.searchsorted(a, g, side='right') / len(a) - np.searchsorted(b, g, side='right') / len(b))))
    def qs(a):
        a = a[np.isfinite(a)]
        return [round(float(np.quantile(a, q)), 3) for q in (0.01, 0.1, 0.5, 0.9)] if len(a) else []
    C = np.load(A / 'artifacts/fusion/boundary_models_t4/exp2_cache.npz')
    names = json.load(open(ME / 'artifacts/fusion/boundary_models_t5/stage3b_meta.json'))['features']
    Xs = np.column_stack([C['F'], C['Frev'], C['Fnum']]).astype(np.float32)
    n53 = Xs.shape[1]
    own_s = np.column_stack([np.load(ME / f'artifacts/ownership_v1/{f}_sample.npy') for f in ('own', 'own_addr', 'own_raw')])
    own_t = [np.load(ME / f'artifacts/ownership_v1/{f}_test.npy', mmap_mode='r') for f in ('own', 'own_addr', 'own_raw')]
    on = json.load(open(ME / 'artifacts/ownership_v1/own_sample.json'))
    own_names = on['features'] + on['addr_features'] + on['raw_features']
    S = np.load(A / 'artifacts/fusion/stage1.npz')
    v5s = np.where(S['stage1'] < 0.02, 0.0, S['stage1'])
    c2s = np.nan_to_num(np.load('/scratch/scai/mtech/aib262140/amazon_ml_2026/ftchunk2/ce_logit_sample.npy'))
    p5s = np.load(ME / 'artifacts/fusion/boundary_models_t5/stage3b_prob.npy')
    Z = np.load(ME / 'artifacts/submission_T5_own/late_scores.npz', allow_pickle=True)
    Xt = np.load(A / 'artifacts/submission_T4_s3b/features.npy', mmap_mode='r')
    v5t, c2t, p5t, ri = Z['stage1'], Z['ce'], Z['prob'], Z['ref_idx']
    cty = []
    with open(ME / 'data/dataset/test/test_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t'); ic = head.index('country')
        for line in fh:
            q = line.rstrip('\n').split('\t'); cty.append(q[ic] if len(q) > ic else '')
    pc = np.array(cty, dtype=object)[ri]
    rng = np.random.default_rng(0)
    groups = {'sure_match': ((v5s >= 0.99) & (c2s >= 9), (v5t >= 0.99) & (c2t >= 9)),
              'sure_nonmatch': ((v5s <= 0.02) & (c2s <= -5), (v5t <= 0.02) & (c2t <= -5))}
    R = {'method': 'KS distance sample vs test on agreed pairs (v5 + chunk2); T4 stage-3b inputs + ownership + T5 probability',
         'groups': {}}
    for g, (ms, mt) in groups.items():
        js = np.flatnonzero(ms); js = rng.choice(js, min(200000, len(js)), replace=False)
        jt = np.flatnonzero(mt); jt = np.sort(rng.choice(jt, min(400000, len(jt)), replace=False))
        Xtg = np.asarray(Xt[jt, :n53])
        Otg = np.column_stack([np.asarray(o[jt]) for o in own_t])
        rows = []
        for k, nm in enumerate(names[:n53] + own_names + ['T5_probability']):
            if k < n53:
                a, b = Xs[js, k], Xtg[:, k]
            elif nm != 'T5_probability':
                a, b = own_s[js, k - n53], Otg[:, k - n53]
            else:
                a, b = p5s[js], p5t[jt]
            by = {c: round(ks(a, b[pc[jt] == c]), 4) for c in ('US', 'India', 'France')}
            rows.append({'feature': nm, 'ks_all': round(ks(a, b), 4), 'ks_by_country': by, 'sample_q': qs(a), 'test_q': qs(b)})
        rows.sort(key=lambda r: -r['ks_all'])
        R['groups'][g] = {'n_sample': int(len(js)), 'n_test': int(len(jt)), 'features': rows}
        print(g, 'n', len(js), len(jt), flush=True)
        for r in rows[:15]:
            print('  ', r['feature'], r['ks_all'], r['ks_by_country'], r['sample_q'], r['test_q'], flush=True)
        p = next(r for r in rows if r['feature'] == 'T5_probability')
        print('  T5_probability', p['ks_all'], p['ks_by_country'], p['sample_q'], p['test_q'], flush=True)
    (ME / 'artifacts/submission_T5_census/t5_shift_audit.json').write_text(json.dumps(R, indent=1))



def cmd_train_listwise():
    """Listwise cross-encoder: one reference + its whole candidate list in one sequence, one logit per candidate.

Sequence: [CLS] reference [SEP] [CAND] cand_1 [CAND] cand_2 ... [SEP]; a linear head reads the hidden state at each
[CAND] marker. Seeing all candidates together lets the model use siblings and look-alikes as context (the
set-encoder idea). Trained data-parallel on the full-train lists (references outside the 100k sample), time-budgeted.
After training it scores (a) the 100k-sample v5 top-10 lists -> ce_logit_sample.npy aligned to mask_top10 rows,
all out-of-sample, and (b) the test v5 top-10 lists -> test_listwise.npz (ref, target, logit) for late_predict.py.
Launch: python -m torch.distributed.run --nproc_per_node=N train_listwise.py
"""
    import gzip
    import os
    import time
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import AutoModel, AutoTokenizer

    HOME = Path.home()
    SCR = Path(os.environ.get('ER_SCRATCH', str(HOME / 'scratch')))
    PAIRS = SCR / 'amazon_ml_2026/artifacts/fulltrain_pairs'
    RAW = SCR / 'amazon_ml_2026/data/dataset'
    V4 = SCR / 'vishal_v4_artifacts'
    MASK = SCR / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy'
    BASE = Path(os.environ.get('BASE', str(SCR / 'amazon_ml_2026/hf_models/mdeberta-v3-base')))
    OUT = Path(os.environ.get('OUT_DIR', str(SCR / 'amazon_ml_2026/artifacts/listwise')))
    TEST_CAND = os.environ.get('TEST_CAND', '')
    MAX_MIN = float(os.environ.get('MAX_MIN', 180))
    LR = float(os.environ.get('LR', 2e-5))
    LISTS = int(os.environ.get('LISTS', 32))
    INIT = os.environ.get('INIT', '')  # continue from a trained listwise model dir (model/ + listwise_head.pt)
    PSEUDO_LISTS = os.environ.get('PSEUDO_LISTS', '')  # France self-training lists (france_pseudo.py), test split
    PSEUDO_FRAC = float(os.environ.get('PSEUDO_FRAC', 0.4))
    TEST_COUNTRY = os.environ.get('TEST_COUNTRY', '')
    MAX_CANDS, REF_LEN, CAND_LEN = 12, 48, 34


    def log(*a):
        if int(os.environ.get('RANK', 0)) == 0:
            print(time.strftime('%H:%M:%S'), *a, flush=True)


    def load_text(split, need):
        text = {}
        for k in (1, 2, 3):
            with open(RAW / split / f'{split}_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if len(q) > iaddr and q[ie] in need:
                        text[q[ie]] = f'{q[iname]} | {q[iaddr]}'
        return text


    class Listwise(torch.nn.Module):
        def __init__(self, base, vocab):
            super().__init__()
            self.enc = AutoModel.from_pretrained(base)
            self.enc.resize_token_embeddings(vocab)
            self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

        def forward(self, ids, att, pos):
            h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
            return self.head(h[pos[:, 0], pos[:, 1]]).squeeze(-1)


    def build(tok, cand_id, ref_texts, cand_lists):
        """ref_texts: list[str]; cand_lists: list[list[str]] -> tensors + marker positions (row, col)."""
        r_ids = tok(ref_texts, add_special_tokens=False, truncation=True, max_length=REF_LEN)['input_ids']
        flat = [c for cl in cand_lists for c in cl]
        c_ids = tok(flat, add_special_tokens=False, truncation=True, max_length=CAND_LEN)['input_ids'] if flat else []
        seqs, pos, k = [], [], 0
        for i, cl in enumerate(cand_lists):
            s = [tok.cls_token_id] + r_ids[i] + [tok.sep_token_id]
            for _ in cl:
                pos.append((i, len(s)))
                s += [cand_id] + c_ids[k]
                k += 1
            seqs.append(s + [tok.sep_token_id])
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        return ids, att, torch.tensor(pos, dtype=torch.long)


    def score_lists(model, tok, cand_id, dev, refs, lists, text, bs=64):
        out = []
        order = np.argsort([len(l) for l in lists], kind='stable')
        res = [None] * len(lists)
        with torch.no_grad():
            for b in range(0, len(order), bs):
                o = order[b:b + bs]
                ids, att, pos = build(tok, cand_id, [text.get(refs[i], ' | ') for i in o],
                                      [[text.get(t, ' | ') for t in lists[i]] for i in o])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    lg = model(ids.to(dev), att.to(dev), pos.to(dev)).float().cpu().numpy()
                k = 0
                for i in o:
                    res[i] = lg[k:k + len(lists[i])]
                    k += len(lists[i])
        return res


    def main():
        dist.init_process_group('nccl')
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local)
        dev = torch.device('cuda', local)
        OUT.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        groups = defaultdict(list)
        for f in sorted(PAIRS.glob('chunk*.npz')):
            z = np.load(f)
            for r, t, l in zip(z['ref'], z['target'], z['label']):
                groups[r].append((t, float(l)))
        keys = sorted(groups)
        rng = np.random.default_rng(2026 + rank)
        mine = [keys[i] for i in np.random.default_rng(2026).permutation(len(keys))[rank::world]]
        need = set(mine) | {t for r in mine for t, _ in groups[r]}
        text = load_text('train', need)
        log('world', world, 'train lists', len(keys), 'per rank', len(mine), f'{time.time() - t0:.0f}s')

        if INIT:
            tok = AutoTokenizer.from_pretrained(Path(INIT) / 'model')
            cand_id = tok.convert_tokens_to_ids('[CAND]')
            net = Listwise(Path(INIT) / 'model', len(tok))
            net.head.load_state_dict(torch.load(Path(INIT) / 'model' / 'listwise_head.pt', map_location='cpu'))
            net = net.to(dev)
        else:
            tok = AutoTokenizer.from_pretrained(BASE)
            tok.add_special_tokens({'additional_special_tokens': ['[CAND]']})
            cand_id = tok.convert_tokens_to_ids('[CAND]')
            net = Listwise(BASE, len(tok)).to(dev)
        pgroups, pkeys, n_ps = defaultdict(list), [], 0
        if PSEUDO_LISTS:
            zp = np.load(PSEUDO_LISTS)
            for r, t, l in zip(zp['ref'], zp['target'], zp['label']):
                pgroups[r].append((t, float(l)))
            pkeys = sorted(pgroups)
            ptext = load_text('test', set(pkeys) | {t for r in pkeys for t, _ in pgroups[r]})
            n_ps = int(LISTS * PSEUDO_FRAC)
            log('pseudo lists', len(pkeys), 'per batch', n_ps)
        model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local])
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        lossf = torch.nn.BCEWithLogitsLoss()
        planned = (len(keys) // world) // (LISTS - n_ps)
        warm = max(1, int(0.03 * min(planned, 20000)))
        model.train()
        t1 = time.time()
        step = 0
        while step < planned:
            frac = min(1.0, (step + 1) / warm) * max(0.0, (planned - step) / planned)
            for g in opt.param_groups:
                g['lr'] = LR * frac
            nb = LISTS - n_ps
            batch = mine[step * nb:(step + 1) * nb]
            lists, labels, rtexts = [], [], []
            for r in [('p', pkeys[k]) for k in (rng.integers(0, len(pkeys), n_ps) if n_ps else [])] + [('t', r) for r in batch]:
                src, r = r
                G_, T_ = (pgroups, ptext) if src == 'p' else (groups, text)
                rtexts.append(T_.get(r, ' | '))
                c = G_[r][:]
                rng.shuffle(c)
                pos_c = [x for x in c if x[1] > 0][:MAX_CANDS]
                neg_c = [x for x in c if x[1] == 0][:MAX_CANDS - len(pos_c)]
                c = pos_c + neg_c
                rng.shuffle(c)
                lists.append([T_.get(t, ' | ') for t, _ in c])
                labels += [l for _, l in c]
            ids, att, pos = build(tok, cand_id, rtexts, lists)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(ids.to(dev), att.to(dev), pos.to(dev))
            loss = lossf(lg.float(), torch.tensor(labels, device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step == 200:
                sps = torch.tensor([(time.time() - t1) / 200], device=dev)
                dist.all_reduce(sps, op=dist.ReduceOp.MAX)
                left = MAX_MIN * 60 - (time.time() - t1)
                plan_t = torch.tensor([int(min(planned, 200 + left / (float(sps) * 1.03)))], device=dev)
                dist.broadcast(plan_t, src=0)
                planned = int(plan_t.item())
                log(f'{float(sps):.3f} s/step -> planned {planned} steps ({planned * LISTS * world:,} lists)')
            if step % 2000 == 0:
                log(f'step {step}/{planned} loss {loss.item():.4f} {time.time() - t1:.0f}s')
        net.eval()
        dist.barrier()
        if rank == 0:
            (OUT / 'model').mkdir(exist_ok=True)
            net.enc.save_pretrained(OUT / 'model')
            tok.save_pretrained(OUT / 'model')
            torch.save(net.head.state_dict(), OUT / 'model' / 'listwise_head.pt')
        log('trained', step, 'steps', f'{time.time() - t1:.0f}s')

        mask = np.load(MASK)
        s_ref, s_tgt = [], []
        with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t')
                    s_ref.append(r); s_tgt.append(t)
        sg = defaultdict(list)
        for j, r in enumerate(s_ref):
            sg[r].append(j)
        skeys = sorted(sg)[rank::world]
        stext = load_text('train', set(skeys) | {s_tgt[j] for r in skeys for j in sg[r]})
        res = score_lists(net, tok, cand_id, dev, skeys, [[s_tgt[j] for j in sg[r]] for r in skeys], stext)
        part = np.full(len(s_ref), np.nan, dtype=np.float32)
        for r, lg in zip(skeys, res):
            part[sg[r]] = lg
        np.save(OUT / f'sample_part{rank}.npy', part)
        dist.barrier()
        if rank == 0:
            ce = np.full(len(s_ref), np.nan, dtype=np.float32)
            for k in range(world):
                p = np.load(OUT / f'sample_part{k}.npy')
                ce = np.where(np.isnan(ce), p, ce)
            np.save(OUT / 'ce_logit_sample.npy', ce)
            log('sample rows scored', int((~np.isnan(ce)).sum()), 'of', len(ce))

        if TEST_CAND:
            trefs, tlists = [], []
            keep = None
            if TEST_COUNTRY:
                keep = set()
                with open(RAW / 'test' / 'test_source1.tsv', encoding='utf-8') as fh:
                    hd = fh.readline().rstrip('\n').split('\t')
                    ie_, ic_ = hd.index('entity_id'), hd.index('country')
                    for line in fh:
                        q = line.rstrip('\n').split('\t')
                        if len(q) > ic_ and q[ic_] == TEST_COUNTRY:
                            keep.add(q[ie_])
            with open(TEST_CAND, encoding='utf-8') as f:
                next(f)
                for k, line in enumerate(f):
                    if k % world != rank:
                        continue
                    s, c = line.rstrip('\n').split('\t')
                    if c and (keep is None or s in keep):
                        trefs.append(s); tlists.append(c.split(','))
            ttext = load_text('test', set(trefs) | {t for l in tlists for t in l})
            res = score_lists(net, tok, cand_id, dev, trefs, tlists, ttext)
            R = np.array([r for r, l in zip(trefs, tlists) for _ in l])
            T = np.array([t for l in tlists for t in l])
            np.savez(OUT / f'ce_listwise_shard{rank}of{world}.npz', ref=R, target=T,
                     logit=np.concatenate(res).astype(np.float32))
            log('test lists scored', len(trefs), f'{time.time() - t0:.0f}s')
        dist.barrier()
        dist.destroy_process_group()


    if __name__ == '__main__':
        main()



COMMANDS = {'analyze_matcher_loss': cmd_analyze_matcher_loss, 'calibrate_probs': cmd_calibrate_probs, 'ce_report': cmd_ce_report, 'cut_and_infer': cmd_cut_and_infer, 'decision_tweaks': cmd_decision_tweaks, 'decoy_weight_check': cmd_decoy_weight_check, 'exclusivity_norm': cmd_exclusivity_norm, 'fusion_predict': cmd_fusion_predict, 'gap_diag': cmd_gap_diag, 'late_fusion_check': cmd_late_fusion_check, 'multi_ce_eval': cmd_multi_ce_eval, 'overcommit_check': cmd_overcommit_check, 'pilot_matcher': cmd_pilot_matcher, 'score_decomp': cmd_score_decomp, 'seg_calib': cmd_seg_calib, 'shap_misclassified': cmd_shap_misclassified, 'shift_audit': cmd_shift_audit, 'shift_ens_check': cmd_shift_ens_check, 'sibling_noise_check': cmd_sibling_noise_check, 't1_empty_check': cmd_t1_empty_check, 't5_shift_audit': cmd_t5_shift_audit, 'train_listwise': cmd_train_listwise}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('commands:', ', '.join(COMMANDS))
        sys.exit(0 if len(sys.argv) < 2 else 1)
    name = sys.argv.pop(1)
    sys.argv[0] = f'{sys.argv[0]} {name}'
    COMMANDS[name]()
