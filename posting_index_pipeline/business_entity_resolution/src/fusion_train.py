import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from submission_pipeline import decide, f05
from matcher_eda import partition_of

HOME = Path.home()
V4 = HOME / 'scratch/vishal_v4_artifacts'
VISHAL = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
CE = HOME / 'scratch/amazon_ml_2026/artifacts/cross_encoder_v1'
TAGS = [t for t in os.environ.get('TAGS', 'e5small_f').split(',') if t]
OOF_CONFIGS = os.environ.get('OOF_CONFIGS', 'all,base+all_ce,base+ce_mdeberta_drop0').split(',')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)



def main():
    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    partition = partition_of(gold)
    S = np.load(FUS / 'stage1.npz')
    rows = S['rows']
    mask = np.load(VISHAL / 'prune_v1/mask_top10.npy')
    refs, tgts = [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t')
                refs.append(r); tgts.append(t)
    refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
    part = np.array([partition[r] for r in refs], dtype=object)
    x = np.asarray(np.load(V4 / 'pairs/x.npy', mmap_mode='r')[rows])
    y = np.load(V4 / 'pairs/y.npy')[rows].astype(np.float32)

    import importlib.util
    spec = importlib.util.spec_from_file_location('ce', Path(__file__).parent / 'cross_encoder_exp.py')
    cem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cem)
    cp = cem.partition_of(gold)
    cepart = np.array([cp[r] for r in refs], dtype=object)

    sp = S['sib_pairs']
    feats = {'base': x, 'ctx_sibrf': np.column_stack([S['ctx'], S['sib'], S['pairpre']])}
    for tag in TAGS:
        a = np.load(CE / tag / 'ce_logit_foldA.npy')
        b = np.load(CE / tag / 'ce_logit_foldB.npy')
        ce = np.where(cepart == 'trainA', b, np.where(cepart == 'trainB', a, (a + b) / 2))
        feats[f'ce_{tag}'] = ce[:, None]
        sa, sb = CE / tag / 'sib_logit_foldA.npy', CE / tag / 'sib_logit_foldB.npy'
        if sa.exists() and sb.exists():
            la, lb = np.load(sa), np.load(sb)
            cpart = cepart[sp[:, 0]]
            sl = np.where(cpart == 'trainA', lb, np.where(cpart == 'trainB', la, (la + lb) / 2))
            mx = np.full(len(rows), -20.0, dtype=np.float32)
            sm = np.zeros(len(rows), dtype=np.float32)
            cnt = np.zeros(len(rows), dtype=np.float32)
            np.maximum.at(mx, sp[:, 0], sl)
            np.add.at(sm, sp[:, 0], sl)
            np.add.at(cnt, sp[:, 0], 1)
            mean = np.where(cnt > 0, sm / np.maximum(cnt, 1), -20.0)
            feats[f'sibce_{tag}'] = np.column_stack([mx, mean])
        log('loaded', tag, 'nan', int(np.isnan(ce).sum()))

    ce_keys = [k for k in feats if k.startswith('ce_')]
    sib_keys = [k for k in feats if k.startswith('sibce_')]
    configs = {'base': ['base'], 'base+ctx_sibrf': ['base', 'ctx_sibrf']}
    for k in ce_keys:
        configs[f'base+{k}'] = ['base', k]
    if len(ce_keys) > 1:
        configs['base+all_ce'] = ['base'] + ce_keys
    configs['base+all_ce+sibce'] = ['base'] + ce_keys + sib_keys
    configs['all'] = ['base', 'ctx_sibrf'] + ce_keys + sib_keys

    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                  grow_policy='lossguide', max_depth=0, max_leaves=255, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=1.0, max_bin=256, seed=2026)
    grid = [(ru, round(float(t), 2)) for ru in ('expected', 'threshold') for t in np.arange(0.3, 0.96, 0.025)]
    idx = {p: np.flatnonzero(part == p) for p in ('train', 'dev', 'holdout')}
    prefs = {p: sorted(s for s in gold if partition[s] == p) for p in ('dev', 'holdout')}
    rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in prefs}

    def macro(p, prob, ru, t):
        c = decide(rix[p], prob, ru, t, len(prefs[p]))
        pred = defaultdict(set)
        for i, tg in zip(rix[p][c], tgts[idx[p]][c]):
            pred[i].add(tg)
        return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

    results = {'tags': TAGS}
    for name, keys in configs.items():
        F = np.column_stack([feats[k] for k in keys]).astype(np.float32)
        dtr = xgb.QuantileDMatrix(F[idx['train']], y[idx['train']])
        ddv = xgb.QuantileDMatrix(F[idx['dev']], y[idx['dev']], ref=dtr)
        bst = xgb.train(params, dtr, 4000, evals=[(ddv, 'dev')], early_stopping_rounds=100, verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        pd_, ph = (np.asarray(bst.inplace_predict(F[idx[p]], iteration_range=it), dtype=np.float64)
                   for p in ('dev', 'holdout'))
        dev = {g: macro('dev', pd_, *g) for g in grid}
        best = max(dev, key=lambda k: (dev[k], k[1]))
        results[name] = {'features': int(F.shape[1]), 'best_iteration': int(bst.best_iteration), 'rule': best,
                         'dev_f05': dev[best], 'holdout_f05': macro('holdout', ph, *best)}
        log(name, results[name])
        md = FUS / 'models' / name
        md.mkdir(parents=True, exist_ok=True)
        bst.save_model(str(md / 'model.json'))
        (md / 'config.json').write_text(json.dumps({'feature_blocks': keys, 'tags': TAGS, 'rule': best,
                                                   'best_iteration': int(bst.best_iteration),
                                                   'dev_f05': dev[best], 'holdout_f05': results[name]['holdout_f05']}))
        np.save(md / 'holdout_prob.npy', ph)
        np.save(md / 'dev_prob.npy', pd_)
        if name in OOF_CONFIGS or (OOF_CONFIGS == ['*']):
            oof = np.full(len(rows), np.nan, dtype=np.float32)
            tr = idx['train']
            fk = np.array([int(hashlib.sha256((r + '#oof').encode()).hexdigest()[:8], 16) % 5 for r in refs[tr]])
            for k in range(5):
                fit, prd = tr[fk != k], tr[fk == k]
                b = xgb.train(params, xgb.QuantileDMatrix(F[fit], y[fit]), bst.best_iteration + 1)
                oof[prd] = b.inplace_predict(F[prd])
            full = np.full(len(rows), np.nan, dtype=np.float32)
            full[idx['train']] = oof[idx['train']]
            full[idx['dev']] = pd_
            full[idx['holdout']] = ph
            np.save(md / 'prob_all_rows.npy', full)
            (md / 'README.txt').write_text(
                'prob_all_rows.npy: one value per True position of prune_v1/mask_top10.npy in pairs.tsv.gz order '
                '(row_index = np.flatnonzero(mask)). Train rows are 5-fold out-of-fold (folds by reference, '
                "sha256(ref+'#oof')%5, rounds fixed to the full model's best iteration); dev and holdout come from the "
                'model trained on all of train. partition: see cross_encoder_v1/export/fold.npy.\n')
            log('oof saved', name)
    (FUS / f'results_{"_".join(TAGS)}.json').write_text(json.dumps(results, indent=1, default=str))
    log('done')


if __name__ == '__main__':
    main()
