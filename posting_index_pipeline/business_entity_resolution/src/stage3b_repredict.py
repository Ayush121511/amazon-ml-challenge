"""Re-predict stage-3b on test from a finished test run's saved features, with extra column blocks appended.

Skips the 75-minute feature build: the base run's features.npy already holds [F, Frev, Fnum, ce2...] in model order,
so a model retrained on the same inputs plus extra blocks (e.g. Vishal's ownership features, candidate_pairs.tsv
expanded order = late_scores order) only needs those columns appended. Then the dev-chosen rule, global exclusivity
(finalize) and the usual output folder, exactly like stage3b_predict.py.
Env BASE_TEST (stage3b_predict output folder), MODEL_DIR, OWN_TEST (';'-separated .npy), OUT, RULE (optional).
"""
import json
import os
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from late_predict import decide_and_write


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    B, M, OUT = Path(os.environ['BASE_TEST']), Path(os.environ['MODEL_DIR']), Path(os.environ['OUT'])
    OUT.mkdir(parents=True, exist_ok=True)
    Z = np.load(B / 'late_scores.npz', allow_pickle=True)
    n = len(Z['prob'])
    extra = [np.load(f).astype(np.float32).reshape(n, -1) for f in os.environ.get('OWN_TEST', '').split(';') if f]
    X = np.column_stack([np.load(B / 'features.npy')] + extra).astype(np.float32)
    meta = json.load(open(M / 'stage3b_meta.json'))
    assert X.shape[1] == len(meta['features']), (X.shape, len(meta['features']))
    b3 = xgb.Booster(); b3.load_model(str(M / 'stage3b.json')); b3.set_param({'device': 'cuda'})
    P3 = np.asarray(b3.inplace_predict(X, iteration_range=(0, meta['best_iteration'] + 1)), dtype=np.float32)
    log('predicted', n, 'mean', float(P3.mean()), 'base mean', float(Z['prob'].mean()))
    np.savez(OUT / 'late_scores.npz', ref_idx=Z['ref_idx'], target=Z['target'], prob=P3, stage1=Z['stage1'], ce=Z['ce'],
             l2=Z['l2'])
    shutil.copy(B / 'candidates.tsv', OUT / 'candidates.tsv')
    rule = os.environ.get('RULE') or f"{meta['rule'][0]}@{meta['rule'][1]}"
    decide_and_write(Namespace(output=str(OUT), rule=rule, tag='', late_model=str(M), country_threshold=''),
                     {'rule': meta['rule']})
    log('done', rule)


if __name__ == '__main__':
    main()
