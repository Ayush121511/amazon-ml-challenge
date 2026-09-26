"""Candidate pruning (second blocking stage) on saved pairs: how small can the candidate set get?

Stage 1 is an XGBoost ranker over RETRIEVAL signals only (route scores/ranks, reverse ranks,
embedding cosine and ranks; no string-similarity features), so it is a genuine, cheap blocking
filter. It is trained out-of-fold on the train partition (2 folds by reference) and on the whole
train partition for dev/holdout/test. Candidate rules are compared on dev by END-TO-END recall
(gold links never retrieved count as misses) and mean candidates per reference. Labels are
used only to train stage 1 and to score recall.
"""
import argparse
import gzip
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

STAGE1_FEATURES = [
    'score_name', 'inv_rank_name', 'gap_name', 'score_address', 'inv_rank_address', 'gap_address',
    'score_anchor', 'inv_rank_anchor', 'gap_anchor', 'score_combined', 'inv_rank_combined',
    'gap_combined', 'rrf', 'rrf_ratio', 'fused_rank', 'route_hits', 'in_forward', 'rev_rank',
    'rev_rrf', 'rev_score_name', 'rev_score_address', 'rev_score_anchor', 'rev_inv_rank', 'rev_top1',
    'emb_cos', 'emb_gap', 'emb_rank', 'dense_rank', 'dense_score', 'dense_rev_rank',
    'dense_rev_score', 'dense_inv_rank', 'dense_rev_inv_rank', 'dense_rev_top1']


def partition_of(ref_id):
    h = int(hashlib.sha256(ref_id.encode()).hexdigest()[:8], 16) % 100
    return 'train' if h < 70 else 'dev' if h < 85 else 'holdout'


def rank_within(ref_idx, score):
    """1-based rank of each row within its reference, by descending score (ties by row)."""
    n = len(score)
    order = np.lexsort((np.arange(n), -score, ref_idx))
    sorted_ref = ref_idx[order]
    rank = np.empty(n, dtype=np.int32)
    rank[order] = np.arange(n) - np.searchsorted(sorted_ref, sorted_ref, side='left') + 1
    return rank


def fit_stage1(x, y, device, threads, rounds=600):
    import xgboost as xgb
    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device=device,
                  max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=5.0, max_bin=256, seed=2026, nthread=threads)
    return xgb.train(params, xgb.QuantileDMatrix(x, y), num_boost_round=rounds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--masks', default='5,10,20', help='Stage-1 top-K values to save as row masks')
    a = p.parse_args()
    started = time.monotonic()
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        device = 'cpu'
    names = json.loads((a.pairs / 'report.json').read_text())['features']
    import pandas as pd
    pairs = pd.read_csv(a.pairs / 'pairs.tsv.gz', sep='\t', header=None, names=['ref', 'target'],
                        dtype=str, usecols=[0])
    ref_codes, ref_names = pd.factorize(pairs['ref'])
    ref_idx = ref_codes.astype(np.int64)
    del pairs
    part_of_ref = np.array([partition_of(s) for s in ref_names], dtype=object)
    row_part = part_of_ref[ref_idx]
    y = np.load(a.pairs / 'y.npy').astype(np.int8)
    xm = np.load(a.pairs / 'x.npy', mmap_mode='r')
    cols = [names.index(f) for f in STAGE1_FEATURES]
    x1 = np.ascontiguousarray(xm[:, cols], dtype=np.float32)
    gold = json.loads((a.pairs / 'gold.json').read_text())
    countries = json.loads((a.pairs / 'countries.json').read_text())
    print(f'loaded {len(y):,} rows, {len(ref_names):,} refs in {time.monotonic() - started:.0f}s', flush=True)

    # Stage 1: out-of-fold on train (2 folds by reference), full-train model for dev/holdout.
    s1 = np.zeros(len(y), dtype=np.float32)
    train_rows = row_part == 'train'
    fold = (np.array([int(hashlib.sha256((s + ':fold').encode()).hexdigest()[:8], 16) % 2
                      for s in ref_names]))[ref_idx]
    for f in (0, 1):
        fit_rows = train_rows & (fold != f)
        booster = fit_stage1(x1[fit_rows], y[fit_rows], device, a.threads)
        pred_rows = train_rows & (fold == f)
        s1[pred_rows] = booster.inplace_predict(x1[pred_rows])
    full = fit_stage1(x1[train_rows], y[train_rows], device, a.threads)
    rest = ~train_rows
    s1[rest] = full.inplace_predict(x1[rest])
    full.save_model(str(a.output / 'stage1.json'))
    np.save(a.output / 'stage1_scores.npy', s1)
    print(f'stage 1 trained in {time.monotonic() - started:.0f}s', flush=True)

    rank_s1 = rank_within(ref_idx, s1)
    emb_rank = x1[:, STAGE1_FEATURES.index('emb_rank')]
    ref_country = np.array([countries[s] for s in ref_names], dtype=object)

    def evaluate(mask, part):
        """End-to-end recall + mean candidates per reference for one partition."""
        refs = np.flatnonzero(part_of_ref == part)
        total = sum(len(gold[ref_names[i]]) for i in refs)
        rows = (row_part == part) & mask
        kept = np.bincount(ref_idx[rows], minlength=len(ref_names))[refs]
        out = {'recall': float(y[rows].sum() / total), 'mean_candidates': float(kept.mean()),
               'max_candidates': int(kept.max()), 'refs_with_zero': int((kept == 0).sum())}
        for c in ('India', 'US'):
            rc = refs[ref_country[refs] == c]
            tot_c = sum(len(gold[ref_names[i]]) for i in rc)
            sel = np.isin(ref_idx, rc) & rows
            out[f'recall_{c}'] = float(y[sel].sum() / tot_c) if tot_c else None
        return out

    everything = np.ones(len(y), dtype=bool)
    results = {'all_candidates': {p_: evaluate(everything, p_) for p_ in ('dev', 'holdout')}}
    for k in (1, 2, 3, 5, 8, 10, 15, 20, 30, 50):
        results[f'emb_rank<={k}'] = {'dev': evaluate(emb_rank <= k, 'dev')}
        results[f'stage1_top{k}'] = {'dev': evaluate(rank_s1 <= k, 'dev')}
    for t in (0.001, 0.005, 0.01, 0.02, 0.05):
        for n in (10, 15, 20, 30):
            mask = (rank_s1 <= 3) | ((s1 >= t) & (rank_s1 <= n))
            results[f'adaptive_t{t}_min3_max{n}'] = {'dev': evaluate(mask, 'dev')}
    for k in (int(v) for v in a.masks.split(',')):
        mask = rank_s1 <= k
        np.save(a.output / f'mask_top{k}.npy', mask)
        results[f'stage1_top{k}']['holdout'] = evaluate(mask, 'holdout')
    report = {'rows': int(len(y)), 'references': int(len(ref_names)), 'stage1_features': STAGE1_FEATURES,
              'device': device, 'results': results, 'seconds': time.monotonic() - started}
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    for name, r in results.items():
        d = r['dev']
        print(f"{name:32s} dev recall {d['recall']:.4f}  cand/ref {d['mean_candidates']:.1f}  "
              f"IN {d['recall_India']:.4f} US {d['recall_US']:.4f}"
              + (f"  | holdout recall {r['holdout']['recall']:.4f} cand/ref {r['holdout']['mean_candidates']:.1f}"
                 if 'holdout' in r else ''), flush=True)


if __name__ == '__main__':
    main()
