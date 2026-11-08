"""Label-free per-country look at the S5 test scores: is France (no training data) less certain than US/India?

On late_scores.npz (pre-exclusivity L2 probability, v5 probability, cross-encoder logit per test pair) and the
threshold@0.68 decision, per country: pair-probability bands, uncertain candidates per reference, best-probability
quantiles, expected matches (sum of p) vs chosen, v5 / cross-encoder disagreement, and the expected F0.5 of the
decision if p were calibrated (ratio-of-expectations approximation). Writes fusion/france_check.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
S5 = HOME / 'scratch/amazon_ml_2026/artifacts/submission_S5/late_scores.npz'
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/test/test_source1.tsv'
OUT = HOME / 'scratch/amazon_ml_2026/artifacts/fusion/france_check.json'
THR = 0.68


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    Z = np.load(S5, allow_pickle=True)
    ri, p, s1, ce = Z['ref_idx'], Z['prob'].astype(np.float64), Z['stage1'].astype(np.float64), Z['ce'].astype(np.float64)
    countries = []
    with open(RAW, encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ic = head.index('country')
        for line in fh:
            q = line.rstrip('\n').split('\t')
            countries.append(q[ic] if len(q) > ic else '')
    countries = np.array(countries, dtype=object)
    n = len(countries)
    log('pairs', len(p), 'refs', n)
    ch = p >= THR
    sp = np.bincount(ri, p, n)
    k = np.bincount(ri, ch.astype(float), n)
    tp = np.bincount(ri, p * ch, n)
    none = np.exp(np.bincount(ri, np.log(np.clip(1 - p, 1e-12, 1)), n))
    ef = np.where(k > 0, 1.25 * tp / np.maximum(0.25 * sp + k, 1e-12), none)
    best = np.zeros(n); np.maximum.at(best, ri, p)
    unc = np.bincount(ri, ((p >= 0.1) & (p < 0.9)).astype(float), n)
    pc = countries[ri]
    bands = [0, .02, .1, .3, .5, .68, .9, .98, 1.0001]
    R = {'threshold': THR}
    for c in sorted(set(countries)):
        m, mp = countries == c, pc == c
        top = np.full(n, -np.inf); np.maximum.at(top, ri[mp], ce[mp])
        top1 = np.zeros(n); np.maximum.at(top1, ri[mp], s1[mp])
        R[c] = {
            'refs': int(m.sum()),
            'expected_f05_if_calibrated': round(float(ef[m].mean()), 5),
            'mean_sum_p': round(float(sp[m].mean()), 4), 'mean_chosen': round(float(k[m].mean()), 4),
            'refs_no_match': round(float((k[m] == 0).mean()), 4),
            'refs_with_uncertain_cand': round(float((unc[m] > 0).mean()), 4),
            'mean_uncertain_cands': round(float(unc[m].mean()), 4),
            'pair_band_share': {f'{bands[i]}-{min(bands[i + 1], 1)}': round(float(((p[mp] >= bands[i]) & (p[mp] < bands[i + 1])).mean()), 5)
                                for i in range(len(bands) - 1)},
            'best_p_quantiles': {q: round(float(np.quantile(best[m], q)), 4) for q in (0.01, 0.03, 0.05, 0.07, 0.1, 0.15)},
            'top_ce_logit_quantiles': {q: round(float(np.quantile(top[m], q)), 3) for q in (0.05, 0.1, 0.25, 0.5)},
            'top_v5_quantiles': {q: round(float(np.quantile(top1[m], q)), 4) for q in (0.05, 0.1, 0.25, 0.5)},
            'v5_hi_ce_lo': round(float(((s1[mp] >= 0.9) & (ce[mp] < 0)).mean()), 5),
            'v5_lo_ce_hi': round(float(((s1[mp] < 0.1) & (ce[mp] > 2)).mean()), 5),
            'ef_loss_share_uncertain_refs': round(float((1 - ef[m & (unc > 0)]).sum() / max((1 - ef[m]).sum(), 1e-9)), 4),
        }
        log(c, R[c])
    OUT.write_text(json.dumps(R, indent=1))
    log('done')


if __name__ == '__main__':
    main()
