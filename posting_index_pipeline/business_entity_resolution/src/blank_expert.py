"""Fixes for blank-address misses on top of stage-3b (cached features from boundary_exp2.py).

  A  per-segment cutoffs: separate dev-chosen thresholds for blank / non-blank rows on the stage-3b probability
  B  stage-3b + disagreement/identity features on all rows: decoy word added (train: 0% positives), benign noise
     word added, exact name = reference, exact name = an accepted non-blank sibling, same-name siblings in the list,
     v5 - sigmoid(ce), |.|, cross-encoder rank and v5 rank within the list
  C  blank expert: model B's features, trained on blank rows only (non-blank rows keep model B)
Rule chosen on dev, holdout once. Writes fusion/blank_expert.json.
"""
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from blank_disagree import DECOY, toks
from boundary_exp import CE, FUS, OUT, PRUNE, RAW, V4, W
from distractor_check import decide, f05, partition_of

BENIGN = {'center', 'services', 'service', 'partners', 'trust', 'the', 'and', 'com'}


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
    v5 = np.where(S['stage1'] < 0.02, 0.0, S['stage1']).astype(np.float64)
    ce = np.nan_to_num(np.load(CE).astype(np.float64), nan=0.0)
    C = np.load(OUT / 'exp2_cache.npz')
    base = np.column_stack([C['F'], C['Frev'], C['Fnum']]).astype(np.float32)
    P, blank = C['P'], C['blank']
    owned = set()
    with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if len(q) > 1 and q[1]:
                owned.update(t.strip() for t in q[1].split(','))
    wt = np.where((y == 0) & np.array([t not in owned for t in tgts]), W, 1.0).astype(np.float32)
    need = set(refs) | set(tgts)
    name = {}
    for k in (1, 2, 3):
        with open(RAW / f'train_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname = head.index('entity_id'), head.index('business_name')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if q[ie] in need:
                    name[q[ie]] = frozenset(toks(q[iname]))
    log('names')
    by_ref = defaultdict(list)
    for j, r in enumerate(refs):
        by_ref[r].append(j)
    X = np.zeros((len(y), 9), dtype=np.float32)
    sig = 1 / (1 + np.exp(-ce))
    for r, js in by_ref.items():
        rt = name.get(r, frozenset())
        tn = [name.get(tgts[j], frozenset()) for j in js]
        acc = [P[j] >= 0.5 and not blank[j] for j in js]
        rc = np.empty(len(js)); rc[np.argsort(-ce[js], kind='stable')] = np.arange(1, len(js) + 1)
        rv = np.empty(len(js)); rv[np.argsort(-v5[js], kind='stable')] = np.arange(1, len(js) + 1)
        for q, j in enumerate(js):
            t = tn[q]
            X[j] = [bool((t - rt) & DECOY), bool((t - rt) & BENIGN), bool(t) and t == rt,
                    any(acc[z] and z != q and tn[z] == t for z in range(len(js))) and bool(t),
                    sum(tn[z] == t and not blank[js[z]] for z in range(len(js)) if z != q),
                    v5[j] - sig[j], abs(v5[j] - sig[j]), rc[q], rv[q]]
    log('extra features')
    XB = np.column_stack([base, X])
    tr, dv = np.isin(part, ['trainA', 'trainB']), part == 'dev'
    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                  max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)

    def fit(Xm, rows, dev_rows):
        d = xgb.QuantileDMatrix(Xm[rows], y[rows], weight=wt[rows])
        b = xgb.train(params, d, 3000, evals=[(xgb.QuantileDMatrix(Xm[dev_rows], y[dev_rows], ref=d), 'dev')],
                      early_stopping_rounds=100, verbose_eval=False)
        return np.asarray(b.inplace_predict(Xm, iteration_range=(0, b.best_iteration + 1)), dtype=np.float64)

    idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
    prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
    rix = {}
    for p in idx:
        pos = {s: i for i, s in enumerate(prefs[p])}
        rix[p] = np.array([pos[r] for r in refs[idx[p]]])

    def macro(p, chosen):
        pred = defaultdict(set)
        for i, tt in zip(rix[p][chosen], tgts[idx[p]][chosen]):
            pred[i].add(tt)
        return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

    grid = [(ru, round(float(t), 2)) for ru in ('expected', 'threshold') for t in np.arange(0.4, 0.961, 0.02)]
    R = {'ref': 'stage3b expected@0.60 dev 0.99124 / holdout 0.98985'}

    def by_rule(nm, prob):
        dev = {g: macro('dev', decide(rix['dev'], prob[idx['dev']], *g, len(prefs['dev']))) for g in grid}
        g = max(dev, key=lambda k: (dev[k], k[1]))
        R[nm] = {'rule': g, 'dev': round(dev[g], 6),
                 'holdout': round(macro('holdout', decide(rix['holdout'], prob[idx['holdout']], *g, len(prefs['holdout']))), 6)}
        log(nm, R[nm])
        (FUS / 'blank_expert.json').write_text(json.dumps(R, indent=1))

    P3 = np.load(OUT / 'stage3b_prob.npy')
    by_rule('stage3b_recheck', P3)
    # A: per-segment thresholds on stage-3b
    ts = np.arange(0.4, 0.961, 0.04)
    bl = {p: blank[idx[p]] for p in idx}
    dev = {(a, b): macro('dev', np.where(bl['dev'], P3[idx['dev']] >= a, P3[idx['dev']] >= b)) for a in ts for b in ts}
    a, b = max(dev, key=dev.get)
    R['A_segment_thresholds'] = {'blank_t': round(float(a), 2), 'nonblank_t': round(float(b), 2), 'dev': round(dev[(a, b)], 6),
                                 'holdout': round(macro('holdout', np.where(bl['holdout'], P3[idx['holdout']] >= a, P3[idx['holdout']] >= b)), 6)}
    log('A', R['A_segment_thresholds'])
    PB = fit(XB, tr, dv)
    by_rule('B_stage3b+identity', PB)
    PC = PB.copy()
    PC[blank] = fit(XB, tr & blank, dv & blank)[blank]
    by_rule('C_blank_expert', PC)
    np.save(OUT / 'blank_expert_prob.npy', PC)
    log('done')


if __name__ == '__main__':
    main()
