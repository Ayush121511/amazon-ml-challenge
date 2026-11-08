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
