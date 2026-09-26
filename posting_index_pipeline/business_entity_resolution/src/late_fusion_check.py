import gzip
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from submission_pipeline import decide, f05

HOME = Path.home()
V4 = HOME / 'scratch/vishal_v4_artifacts'
VISHAL = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
CE = HOME / 'scratch/amazon_ml_2026/artifacts/cross_encoder_v1'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'mdeberta_drop0'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    spec = importlib.util.spec_from_file_location('cem', Path(__file__).parent / 'cross_encoder_exp.py')
    cem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cem)
    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    cp = cem.partition_of(gold)
    mask = np.load(VISHAL / 'prune_v1/mask_top10.npy')
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
    y = np.load(V4 / 'pairs/y.npy')[rows].astype(np.float32)
    x = np.asarray(np.load(V4 / 'pairs/x.npy', mmap_mode='r')[rows])
    s1 = S['stage1'].astype(np.float64)
    s1f = np.where(s1 < 0.02, 0.0, s1)
    lg = np.log(np.clip(s1f, 1e-4, 1 - 1e-6) / (1 - np.clip(s1f, 1e-4, 1 - 1e-6)))
    ce = np.nan_to_num(np.load(CE / TAG / 'ce_logit.npy').astype(np.float64), nan=0.0)
    by_ref = defaultdict(list)
    for j, r in enumerate(refs):
        by_ref[r].append(j)
    ctx = np.zeros((len(rows), 7), dtype=np.float32)
    for js in by_ref.values():
        js = sorted(js, key=lambda j: -s1f[j])
        p = s1f[js]
        for rk, j in enumerate(js):
            nxt = p[rk + 1] if rk + 1 < len(p) else 0.0
            ctx[j] = [s1f[j], rk + 1, p[0] - s1f[j], s1f[j] - nxt, p.sum(), (p >= 0.9).sum(), (p >= 0.5).sum()]
    ctx_floor = np.column_stack([ctx, S['sib'], S['pairpre']])
    ce_ctx = np.zeros((len(rows), 4), dtype=np.float32)
    for js in by_ref.values():
        c = ce[js]
        o = np.argsort(-c)
        rank = np.empty(len(js)); rank[o] = np.arange(1, len(js) + 1)
        ce_ctx[js] = np.column_stack([rank, c.max() - c, np.full(len(js), (c > 0).sum()), np.full(len(js), c.max())])

    tb = part == 'trainB'
    idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
    prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
    rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}
    grid = [(ru, round(float(t), 3)) for ru in ('expected', 'threshold') for t in np.arange(0.3, 0.96, 0.025)]

    def macro(p, prob, ru, t):
        c = decide(rix[p], prob, ru, t, len(prefs[p]))
        pred = defaultdict(set)
        for i, tg in zip(rix[p][c], tgts[idx[p]][c]):
            pred[i].add(tg)
        return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

    def report(name, prob):
        dev = {g: macro('dev', prob[idx['dev']], *g) for g in grid}
        best = max(dev, key=lambda k: (dev[k], k[1]))
        R[name] = {'rule': best, 'dev_f05': dev[best], 'holdout_f05': macro('holdout', prob[idx['holdout']], *best)}
        log(name, R[name])

    R = {'tag': TAG, 'note': 'Trained on fold B (cross-encoder trained on fold A). Stage-1 probabilities are 5-fold '
                            'out-of-fold on train and floored at 0.02 like the saved test scores.'}
    report('stage1_only_floored', s1f)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.column_stack([lg, ce])[tb], y[tb])
    report('L1_logreg_stage1_ce', lr.predict_proba(np.column_stack([lg, ce]))[:, 1])
    small = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                 max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)
    big = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
               grow_policy='lossguide', max_depth=0, max_leaves=255, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=1.0, max_bin=256, seed=2026)
    dv = part == 'dev'
    variants = {'L2_xgb_stage1_ce': (np.column_stack([s1f, ce]), small),
                'L3_xgb_stage1_ce_setctx_sibling': (np.column_stack([ctx_floor, ce, ce_ctx]), big),
                'F_feature_level_103_ce': (np.column_stack([x, ce]), big),
                'F2_feature_level_103_ce_setctx_sibling': (np.column_stack([x, ctx_floor, ce, ce_ctx]), big)}
    for name, (F, prm) in variants.items():
        F = F.astype(np.float32)
        dtr = xgb.QuantileDMatrix(F[tb], y[tb])
        bst = xgb.train(prm, dtr, 3000, evals=[(xgb.QuantileDMatrix(F[dv], y[dv], ref=dtr), 'dev')],
                        early_stopping_rounds=100, verbose_eval=False)
        report(name, np.asarray(bst.inplace_predict(F, iteration_range=(0, bst.best_iteration + 1)), dtype=np.float64))
        md = FUS / 'late_models' / f'{name}_{TAG}'
        md.mkdir(parents=True, exist_ok=True)
        bst.save_model(str(md / 'model.json'))
        (md / 'config.json').write_text(json.dumps({'name': name, 'tag': TAG, 'rule': R[name]['rule'],
                                                   'best_iteration': int(bst.best_iteration),
                                                   'dev_f05': R[name]['dev_f05'], 'holdout_f05': R[name]['holdout_f05']}))
    (FUS / f'late_fusion_{TAG}.json').write_text(json.dumps(R, indent=1, default=str))
    log('done')


if __name__ == '__main__':
    main()
