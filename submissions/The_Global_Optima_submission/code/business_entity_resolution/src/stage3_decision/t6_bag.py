"""T6 bagging: the T6 stage-3b retrained with 4 more seeds on the same inputs, probabilities averaged with T6's.

Inputs are exactly T6's: T4 cache (F, Frev, Fnum), chunk2 L2, full21 + decoyw2 logits, ownership (26) +
W2 (8) columns. Check first: T6's saved model on the rebuilt matrix must reproduce stage3b_prob.npy. Rule chosen on
dev for the bag, holdout scored once; the bag is used only if it beats T6 on dev AND holdout. Also reports links per
reference vs gold, and F0.5 with decoy false matches counted x2 (test has about twice train's decoy odds).
  eval             -> artifacts: /scratch/.../t6bag/{stage3b_s*.json, bag_meta.json, eval.json}
  test <out dir>   -> T4 saved test features + ownership + W2 -> averaged probability -> rule + exclusivity (finalize)
"""
import gzip
import json
import os
import sys
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
AY = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
W6 = ME / 'artifacts/fusion/boundary_models_t6'
BAG = Path(os.environ.get('BAG_DIR', '/scratch/scai/mtech/aib262467/t6bag'))
SEEDS = [11, 12, 13, 14]
os.environ.setdefault('BOUNDARY_OUT', 'boundary_models_t6bag_unused')
sys.path[:0] = [str(AY / 'business_entity_resolution/src')]
from boundary_exp import CE, FUS, PRUNE, RAW, V4, W  # noqa: E402
from distractor_check import decide, f05, partition_of  # noqa: E402


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def build():
    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    cp = partition_of(gold)
    mask = np.load(PRUNE / 'mask_top10.npy')
    refs, tgts = [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t'); refs.append(r); tgts.append(t)
    refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
    part = np.array([cp[r] for r in refs], dtype=object)
    S = np.load(FUS / 'stage1.npz')
    y = np.load(V4 / 'pairs/y.npy')[S['rows']].astype(np.float32)
    ce2 = [np.nan_to_num(np.load(f).astype(np.float64), nan=0.0) for f in os.environ['CE2_FILE'].split(';') if f]
    owned = set()
    with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if len(q) > 1 and q[1]:
                owned.update(t.strip() for t in q[1].split(','))
    decoy = (y == 0) & np.array([t not in owned for t in tgts])
    wt = np.where(decoy, W, 1.0).astype(np.float32)
    C = np.load(Path(os.environ['CACHE_FROM']) / 'exp2_cache.npz')
    own = np.column_stack([np.load(f).astype(np.float32).reshape(len(y), -1) for f in os.environ['OWN_FILE'].split(';') if f])
    X = np.column_stack([C['F'], C['Frev'], C['Fnum'], np.column_stack(ce2), own]).astype(np.float32)
    return gold, cp, refs, tgts, part, y, wt, decoy, X


def evaluate():
    BAG.mkdir(parents=True, exist_ok=True)
    gold, cp, refs, tgts, part, y, wt, decoy, X = build()
    meta6 = json.load(open(W6 / 'stage3b_meta.json'))
    assert X.shape[1] == len(meta6['features']), (X.shape, len(meta6['features']))
    b6 = xgb.Booster(); b6.load_model(str(W6 / 'stage3b.json')); b6.set_param({'device': 'cuda'})
    p6 = np.asarray(b6.inplace_predict(X, iteration_range=(0, meta6['best_iteration'] + 1)), dtype=np.float64)
    saved = np.load(W6 / 'stage3b_prob.npy')
    log('T6 reproduction max abs diff', float(np.abs(p6 - saved).max()))
    assert np.abs(p6 - saved).max() < 1e-4
    tr, dv = np.isin(part, ['trainA', 'trainB']), part == 'dev'
    probs, its = [p6], {3: meta6['best_iteration']}
    for s in SEEDS:
        params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda', max_depth=6,
                      learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=s)
        d = xgb.QuantileDMatrix(X[tr], y[tr], weight=wt[tr])
        b = xgb.train(params, d, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=d), 'dev')],
                      early_stopping_rounds=100, verbose_eval=False)
        b.save_model(str(BAG / f'stage3b_s{s}.json')); its[s] = b.best_iteration
        probs.append(np.asarray(b.inplace_predict(X, iteration_range=(0, b.best_iteration + 1)), dtype=np.float64))
        log('seed', s, 'best iteration', b.best_iteration)
    pbag = np.mean(probs, axis=0)
    idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
    prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
    pos = {p: {s: i for i, s in enumerate(prefs[p])} for p in idx}
    rix = {p: np.array([pos[p][r] for r in refs[idx[p]]]) for p in idx}
    grid = [(ru, round(float(t), 2)) for ru in ('expected', 'threshold') for t in np.arange(0.4, 0.961, 0.02)]

    def score(p, prob, g, dw=1.0):
        ch = decide(rix[p], prob[idx[p]], *g, len(prefs[p]))
        pred = defaultdict(set)
        for i, t in zip(rix[p][ch], tgts[idx[p]][ch]):
            pred[i].add(t)
        dset = set(tgts[idx[p]][decoy[idx[p]]])
        out, links = [], 0
        for i, s in enumerate(prefs[p]):
            P = pred.get(i, set()); links += len(P)
            if dw == 1.0:
                out.append(f05(gold[s], P)); continue
            tp = len(gold[s] & P); fp = sum(dw if t in dset else 1.0 for t in P - gold[s])
            if not gold[s] and not P:
                out.append(1.0)
            elif tp == 0:
                out.append(0.0)
            else:
                pr, rc = tp / (tp + fp), tp / len(gold[s])
                out.append(1.25 * pr * rc / (0.25 * pr + rc))
        return float(np.mean(out)), links / len(prefs[p])
    R = {'seeds': its}
    for name, prob in (('t6', p6), ('bag', pbag)):
        dev = {g: score('dev', prob, g)[0] for g in grid}
        g = max(dev, key=lambda k: (dev[k], k[1]))
        h, lk = score('holdout', prob, g)
        R[name] = {'rule': g, 'dev': round(dev[g], 6), 'holdout': round(h, 6), 'holdout_links_per_ref': round(lk, 4)}
        devw = {gg: score('dev', prob, gg, 2.06)[0] for gg in grid}
        gw = max(devw, key=lambda k: (devw[k], k[1]))
        R[name]['decoy_x2'] = {'rule_chosen_on_dev': gw, 'holdout_at_own_rule_x2': round(score('holdout', prob, g, 2.06)[0], 6),
                               'holdout_at_x2_rule_x2': round(score('holdout', prob, gw, 2.06)[0], 6),
                               'holdout_at_x2_rule_plain': round(score('holdout', prob, gw)[0], 6)}
        log(name, R[name])
    R['gold_links_per_ref_holdout'] = round(sum(len(gold[s]) for s in prefs['holdout']) / len(prefs['holdout']), 4)
    R['use_bag'] = bool(R['bag']['dev'] > R['t6']['dev'] and R['bag']['holdout'] > R['t6']['holdout'])
    json.dump({'features': meta6['features'], 'seeds': its, 'rule': R['bag']['rule'], 'use_bag': R['use_bag']},
              open(BAG / 'bag_meta.json', 'w'), default=str)
    (BAG / 'eval.json').write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps(R, indent=1, default=str), flush=True)


def test(out):
    from late_predict import decide_and_write
    import shutil
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    M = json.load(open(BAG / 'bag_meta.json'))
    B = Path(os.environ['BASE_TEST'])
    Z = np.load(B / 'late_scores.npz', allow_pickle=True)
    n = len(Z['prob'])
    extra = [np.load(f).astype(np.float32).reshape(n, -1) for f in os.environ['OWN_TEST'].split(';') if f]
    X = np.column_stack([np.load(B / 'features.npy')] + extra).astype(np.float32)
    assert X.shape[1] == len(M['features'])
    probs = []
    for s, it in M['seeds'].items():
        b = xgb.Booster(); b.load_model(str(W6 / 'stage3b.json' if int(s) == 3 else BAG / f'stage3b_s{s}.json'))
        b.set_param({'device': 'cuda'})
        probs.append(np.asarray(b.inplace_predict(X, iteration_range=(0, int(it) + 1)), dtype=np.float32))
    P = np.mean(probs, axis=0).astype(np.float32)
    ref6 = np.load(ME / 'artifacts/submission_T6_own/late_scores.npz', allow_pickle=True)['prob']
    log('test predicted', n, 'mean', float(P.mean()), 'T6 mean', float(ref6.mean()), 'corr', float(np.corrcoef(P, ref6)[0, 1]),
        'T6 seed reproduces', float(np.abs(probs[0] - ref6).max()))
    np.savez(out / 'late_scores.npz', ref_idx=Z['ref_idx'], target=Z['target'], prob=P, stage1=Z['stage1'], ce=Z['ce'], l2=Z['l2'])
    shutil.copy(B / 'candidates.tsv', out / 'candidates.tsv')
    rule = f"{M['rule'][0]}@{M['rule'][1]}"
    decide_and_write(Namespace(output=str(out), rule=rule, tag='', late_model=str(W6), country_threshold=''),
                     {'rule': M['rule']})
    log('done', rule)


if __name__ == '__main__':
    evaluate() if sys.argv[1] == 'eval' else test(sys.argv[2])
