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
