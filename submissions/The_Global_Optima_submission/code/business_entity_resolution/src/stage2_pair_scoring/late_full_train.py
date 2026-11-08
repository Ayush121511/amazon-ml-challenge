"""L2 late-fusion model trained on all train rows with 2-fold out-of-fold cross-encoder logits.

Train rows: fold-A rows use the fold-B model's logit and vice versa (out-of-fold). Dev/holdout use the mean of both
fold models, which is also what test inference uses (mean of fold-A and fold-B test scores). Stage-1 probabilities
are the 5-fold out-of-fold ones from fusion/stage1.npz, floored at 0.02 like the saved test scores.
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

HOME = Path.home()
V4 = HOME / 'scratch/vishal_v4_artifacts'
PRUNE = HOME / 'scratch/amazon_ml_2026/artifacts/prune_v1_copy'
FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
CE = HOME / 'scratch/amazon_ml_2026/artifacts/cross_encoder_v1'
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
TAG = os.environ.get('TAG', 'mdeberta_drop0')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


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
    if os.environ.get('CE_SAMPLE'):
        ce = np.load(os.environ['CE_SAMPLE']).astype(np.float64)
    else:
        a = np.load(CE / TAG / 'ce_logit_foldA.npy').astype(np.float64)
        b = np.load(CE / TAG / 'ce_logit_foldB.npy').astype(np.float64)
        ce = np.where(part == 'trainA', b, np.where(part == 'trainB', a, (a + b) / 2))
    assert not np.isnan(ce).any(), 'missing cross-encoder scores'
    X = np.column_stack([s1f, ce]).astype(np.float32)
    owned = set()
    with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if len(q) > 1 and q[1]:
                owned.update(t.strip() for t in q[1].split(','))
    decoy = (y == 0) & np.array([t not in owned for t in tgts])
    tr = np.isin(part, ['trainA', 'trainB'])
    dv = part == 'dev'
    idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
    prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
    rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}
    grid = [(ru, round(float(t), 3)) for ru in ('expected', 'threshold') for t in np.arange(0.5, 0.981, 0.02)]
    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                  max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)
    params['device'] = os.environ.get('DEVICE', 'cuda')

    def macro(p, prob, ru, t):
        c = decide(rix[p], prob, ru, t, len(prefs[p]))
        pred = defaultdict(set)
        for i, tt in zip(rix[p][c], tgts[idx[p]][c]):
            pred[i].add(tt)
        return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

    R = {'tag': TAG, 'train_rows': int(tr.sum()),
         'compare': 'L2 fold-B-only (fold-A logits): dev 0.98624 / holdout 0.98459; decoy w2: 0.98607 / 0.98475'}
    for w in [float(v) for v in os.environ.get('WEIGHTS', '1,2,3').split(',')]:
        wt = np.where(decoy, w, 1.0).astype(np.float32)
        dtr = xgb.QuantileDMatrix(X[tr], y[tr], weight=wt[tr])
        bst = xgb.train(params, dtr, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=dtr), 'dev')],
                        early_stopping_rounds=100, verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        pdv = np.asarray(bst.inplace_predict(X[idx['dev']], iteration_range=it), dtype=np.float64)
        pho = np.asarray(bst.inplace_predict(X[idx['holdout']], iteration_range=it), dtype=np.float64)
        dev = {g: macro('dev', pdv, *g) for g in grid}
        best = max(dev, key=lambda k: (dev[k], k[1]))
        R[f'w{w:g}'] = {'rule': best, 'dev_f05': dev[best], 'holdout_f05': macro('holdout', pho, *best)}
        md = FUS / 'late_models' / f'L2full_{TAG}_w{w:g}'
        md.mkdir(parents=True, exist_ok=True)
        bst.save_model(str(md / 'model.json'))
        (md / 'config.json').write_text(json.dumps({'name': f'L2full_{TAG}_w{w:g}', 'tag': TAG, 'rule': list(best),
                                                   'best_iteration': int(bst.best_iteration), 'decoy_weight': w,
                                                   **R[f'w{w:g}']}))
        log(f'w{w:g}', R[f'w{w:g}'])
    (FUS / f'late_full_{TAG}.json').write_text(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
