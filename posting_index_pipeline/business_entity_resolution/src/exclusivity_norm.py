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
