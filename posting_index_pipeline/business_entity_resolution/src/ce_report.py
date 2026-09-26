import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from submission_pipeline import decide, f05
import importlib.util

HOME = Path.home()
V4 = HOME / 'scratch/vishal_v4_artifacts'
VISHAL = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
CE = HOME / 'scratch/amazon_ml_2026/artifacts/cross_encoder_v1'
EXP = CE / 'export'


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('cem', Path(__file__).parent / 'cross_encoder_exp.py')
    cem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cem)
    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    partition = cem.partition_of(gold)
    mask = np.load(VISHAL / 'prune_v1/mask_top10.npy')
    row_index = np.flatnonzero(mask)
    refs, tgts = [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t')
                refs.append(r); tgts.append(t)
    refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
    part = np.array([partition[r] for r in refs], dtype=object)
    y = np.load(V4 / 'pairs/y.npy')[row_index].astype(np.float32)
    ce = np.load(CE / 'ce_logit.npy')
    R = {'row_alignment': 'arrays have one entry per True position of prune_v1/mask_top10.npy, in pairs.tsv.gz order; '
                          'row_index.npy gives the 0-based line index into pairs.tsv.gz / x.npy / y.npy'}
    R['folds'] = {}
    for p in ('trainA', 'trainB', 'dev', 'holdout'):
        m = part == p
        R['folds'][p] = {'references': int(len({r for r in refs[m]})), 'refs_in_gold': int(sum(v == p for v in partition.values())),
                         'pairs': int(m.sum()), 'positive': int(y[m].sum()), 'negative': int((y[m] == 0).sum()),
                         'ce_scored': int((~np.isnan(ce[m])).sum())}
    R['leakage_check'] = {'ce_trained_only_on': 'trainA',
                          'trainA_rows_with_ce_score (must be 0)': int((~np.isnan(ce[part == 'trainA'])).sum()),
                          'nonA_rows_without_ce_score (must be 0)': int(np.isnan(ce[part != 'trainA']).sum())}
    np.save(EXP / 'row_index.npy', row_index)
    np.save(EXP / 'fold.npy', part.astype('U8'))
    np.save(EXP / 'ce_e5small_foldA_logit.npy', ce)

    x = np.asarray(np.load(V4 / 'pairs/x.npy', mmap_mode='r')[row_index])
    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                  grow_policy='lossguide', max_depth=0, max_leaves=255, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=1.0, max_bin=256, seed=2026)
    tb, dv = part == 'trainB', part == 'dev'
    probs = {'cross_encoder_alone': 1 / (1 + np.exp(-np.nan_to_num(ce, nan=-20).astype(np.float64)))}
    for name, F in (('xgb_baseline_foldB', x), ('xgb_plus_ce_foldB', np.column_stack([x, ce]))):
        dtr = xgb.QuantileDMatrix(F[tb], y[tb])
        bst = xgb.train(params, dtr, 4000, evals=[(xgb.QuantileDMatrix(F[dv], y[dv], ref=dtr), 'dev')],
                        early_stopping_rounds=100, verbose_eval=False)
        probs[name] = np.asarray(bst.inplace_predict(F, iteration_range=(0, bst.best_iteration + 1)), dtype=np.float64)
    np.save(EXP / 'xgb_plus_ce_foldB_prob.npy', probs['xgb_plus_ce_foldB'].astype(np.float32))

    prefs = {p: sorted(s for s in gold if partition[s] == p) for p in ('dev', 'holdout')}
    idx = {p: np.flatnonzero(part == p) for p in prefs}
    rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in prefs}

    def evaluate(p, prob, ru, t):
        c = decide(rix[p], prob[idx[p]], ru, t, len(prefs[p]))
        pred = defaultdict(set)
        for i, tg in zip(rix[p][c], tgts[idx[p]][c]):
            pred[i].add(tg)
        cand = defaultdict(set)
        for i, tg in zip(rix[p], tgts[idx[p]]):
            cand[i].add(tg)
        sc, fp, fn_cand, fn_block, single = [], 0, 0, 0, 0
        for i, s in enumerate(prefs[p]):
            g, pr = gold[s], pred.get(i, set())
            sc.append(f05(g, pr))
            fp += len(pr - g)
            fn_cand += len((g - pr) & cand.get(i, set()))
            fn_block += len(g - cand.get(i, set()))
            single += int(not g and bool(pr))
        return {'macro_f05': float(np.mean(sc)), 'false_links': fp, 'missed_true_in_candidates': fn_cand,
                'missed_true_by_blocking': fn_block, 'singleton_errors': single, 'references': len(prefs[p])}

    grid = [(ru, round(float(t), 2)) for ru in ('expected', 'threshold') for t in np.arange(0.3, 0.96, 0.05)]
    R['results'] = {}
    for name, prob in probs.items():
        dev = {g: evaluate('dev', prob, *g)['macro_f05'] for g in grid}
        best = max(dev, key=lambda k: (dev[k], k[1]))
        R['results'][name] = {'rule': best, 'dev': evaluate('dev', prob, *best), 'holdout': evaluate('holdout', prob, *best)}
        print(name, R['results'][name], flush=True)

    tok = AutoTokenizer.from_pretrained(CE / 'model')
    model = AutoModelForSequenceClassification.from_pretrained(CE / 'model').cuda().eval()
    sample = [('Sharma Traders Pvt Ltd | 12, MG Road, Pune', 'Sharma Traders Private Limited | 12 M G Road Pune 411001')] * 60000
    with torch.no_grad():
        for warm in range(2):
            t0 = time.time()
            for b in range(0, len(sample), 1024):
                L, Rr = zip(*sample[b:b + 1024])
                e = tok(list(L), list(Rr), truncation=True, max_length=128, padding=True, return_tensors='pt').to('cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    model(**e).logits
            torch.cuda.synchronize()
            el = time.time() - t0
    R['throughput_e5small'] = {'pairs_per_second_synthetic': round(len(sample) / el), 'batch': 1024, 'max_len': 128,
                               'dtype': 'bf16 autocast', 'cpus': 3, 'gpu': torch.cuda.get_device_name(0)}
    (EXP / 'report.json').write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps(R, default=str), flush=True)


if __name__ == '__main__':
    main()
