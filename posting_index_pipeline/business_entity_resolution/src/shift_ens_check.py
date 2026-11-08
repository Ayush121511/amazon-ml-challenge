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
