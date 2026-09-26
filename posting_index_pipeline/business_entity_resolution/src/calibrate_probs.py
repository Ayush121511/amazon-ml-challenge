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
