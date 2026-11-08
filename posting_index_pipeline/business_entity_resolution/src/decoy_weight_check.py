"""Decoy reweighting for the late-fusion matcher (L2: XGBoost on [v5 top-10 probability, mdeberta logit]).

Test has far more decoy records (S2/S3 records that belong to no S1 business) than train. This trains L2 on fold B
with decoy negatives weighted by w, picks the rule on dev, scores holdout once, and saves each model in the format
late_predict.py reads. False links are split into decoy and non-decoy, since only decoy false links scale with the
test decoy share. Inputs are read-only from Ayush's scratch; outputs go to OUT_DIR.
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

SCR = Path(os.environ.get('ER_SCRATCH', str(Path.home() / 'scratch')))
V4 = SCR / 'vishal_v4_artifacts'
PRUNE = SCR / 'amazon_ml_2026/artifacts/prune_v1_copy'
FUS = SCR / 'amazon_ml_2026/artifacts/fusion'
CE = Path(os.environ.get('CE_FILE', str(SCR / 'amazon_ml_2026/artifacts/cross_encoder_v1/mdeberta_drop0/ce_logit.npy')))
RAW = SCR / 'amazon_ml_2026/data/dataset/train'
OUT = Path(os.environ.get('OUT_DIR', str(FUS / 'decoy')))
WEIGHTS = [float(w) for w in os.environ.get('WEIGHTS', '1,1.5,2,3,4').split(',')]
TEST_S23_IDS = 9_969_589      # organizer validator, test S2+S3 records
TEST_PRED_LINKS = 5_790_570   # v5 top-10 test output, sum of match counts (report.json)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
    rows = S['rows']
    assert len(rows) == len(refs)
    y = np.load(V4 / 'pairs/y.npy')[rows].astype(np.float32)
    s1 = S['stage1'].astype(np.float64)
    s1f = np.where(s1 < 0.02, 0.0, s1)
    ce = np.nan_to_num(np.load(CE).astype(np.float64), nan=0.0)
    X = np.column_stack([s1f, ce]).astype(np.float32)

    owned = set()
    with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) > 1 and p[1]:
                owned.update(t.strip() for t in p[1].split(','))
    n_train_s23 = 0
    for src in ('train_source2.tsv', 'train_source3.tsv'):
        with open(RAW / src, encoding='utf-8') as f:
            n_train_s23 += sum(1 for _ in f) - 1
    train_share = 1 - len(owned) / n_train_s23
    test_share = 1 - TEST_PRED_LINKS / TEST_S23_IDS
    odds = (test_share / (1 - test_share)) / (train_share / (1 - train_share))
    decoy = (y == 0) & np.array([t not in owned for t in tgts])
    R = {'decoy_share_records': {'train': round(train_share, 4), 'test_estimated': round(test_share, 4),
                                 'implied_weight_odds_ratio': round(odds, 3)},
         'decoy_share_of_negative_pairs': {p: round(float(decoy[(part == p) & (y == 0)].mean()), 4)
                                           for p in ('trainB', 'dev', 'holdout')}}
    log(R)

    tb, dv = part == 'trainB', part == 'dev'
    idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
    prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
    rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}
    grid = [(ru, round(float(t), 3)) for ru in ('expected', 'threshold') for t in np.arange(0.5, 0.981, 0.02)]
    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                  max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)

    def evaluate(p, prob, ru, t):
        c = decide(rix[p], prob, ru, t, len(prefs[p]))
        pred = defaultdict(set)
        for i, tg in zip(rix[p][c], tgts[idx[p]][c]):
            pred[i].add(tg)
        yy, dd = y[idx[p]], decoy[idx[p]]
        single = sum(1 for i, s in enumerate(prefs[p]) if not gold[s] and pred.get(i))
        return {'f05': float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])])),
                'false_links': int((c & (yy == 0)).sum()), 'false_links_decoy': int((c & dd).sum()),
                'missed_true': int((~c & (yy == 1)).sum()), 'singleton_errors': single,
                'links_per_ref': round(float(c.sum()) / len(prefs[p]), 4)}

    R['runs'] = {}
    for w in WEIGHTS:
        wt = np.where(decoy, w, 1.0).astype(np.float32)
        dtr = xgb.QuantileDMatrix(X[tb], y[tb], weight=wt[tb])
        ddv = xgb.QuantileDMatrix(X[dv], y[dv], ref=dtr)
        bst = xgb.train(params, dtr, 3000, evals=[(ddv, 'dev')], early_stopping_rounds=100, verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        pd_ = np.asarray(bst.inplace_predict(X[idx['dev']], iteration_range=it), dtype=np.float64)
        ph = np.asarray(bst.inplace_predict(X[idx['holdout']], iteration_range=it), dtype=np.float64)
        dev = {g: evaluate('dev', pd_, *g)['f05'] for g in grid}
        best = max(dev, key=lambda k: (dev[k], k[1]))
        run = {'rule': best, 'best_iteration': int(bst.best_iteration), 'dev': evaluate('dev', pd_, *best),
               'holdout': evaluate('holdout', ph, *best)}
        R['runs'][str(w)] = run
        md = OUT / f'L2_decoy_w{w:g}'
        md.mkdir(parents=True, exist_ok=True)
        bst.save_model(str(md / 'model.json'))
        (md / 'config.json').write_text(json.dumps({'name': f'L2_decoy_w{w:g}', 'tag': 'mdeberta_drop0',
                                                   'rule': list(best), 'best_iteration': int(bst.best_iteration),
                                                   'features': ['v5_top10_prob_floored_0.02', 'mdeberta_foldA_logit'],
                                                   'decoy_weight': w, 'dev_f05': run['dev']['f05'],
                                                   'holdout_f05': run['holdout']['f05']}))
        log('w', w, run)
    (OUT / 'decoy_weight_check.json').write_text(json.dumps(R, indent=1, default=str))
    log('done', OUT / 'decoy_weight_check.json')


if __name__ == '__main__':
    main()
