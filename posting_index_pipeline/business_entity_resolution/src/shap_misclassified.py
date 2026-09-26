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
