import gzip
import json
import sys
from pathlib import Path

import numpy as np


def load_pairs(d):
    refs, targets = [], []
    with gzip.open(d / 'pairs.tsv.gz', 'rt') as f:
        for line in f:
            r, t = line.rstrip('\n').split('\t')
            refs.append(r)
            targets.append(t)
    return refs, targets


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'scratch/vishal_v4_artifacts'
    pairs_dir = root / 'pairs'
    model_dir = root / 'model_xgb'

    x = np.load(pairs_dir / 'x.npy', mmap_mode='r')
    y = np.load(pairs_dir / 'y.npy')
    report = json.loads((model_dir / 'report.json').read_text())
    names = report['features']
    fr_idx = names.index('fused_rank')
    fused_rank = np.asarray(x[:, fr_idx])
    del x

    print(f'total pairs: {len(y)}, total true (y=1): {int(y.sum())}', flush=True)

    refs, targets = load_pairs(pairs_dir)
    ref_arr = np.array(refs, dtype=object)
    del refs, targets

    print('encoding references...', flush=True)
    uniq_refs, ref_codes = np.unique(ref_arr, return_inverse=True)
    n_refs = len(uniq_refs)
    print(f'distinct references: {n_refs}', flush=True)

    total_true = int(y.sum())

    zero_mask = fused_rank == 0
    print(f'fused_rank==0 (reverse-only sentinel) pairs: {int(zero_mask.sum())} '
          f'({int(zero_mask.sum())/len(y):.4%}), true among them: {int((zero_mask & (y==1)).sum())}',
          flush=True)

    rev_idx = names.index('rev_rank') if 'rev_rank' in names else None
    rev_rank = np.asarray(np.load(pairs_dir / 'x.npy', mmap_mode='r')[:, rev_idx]) if rev_idx is not None else None

    # fused_rank is 1-indexed best-first within each reference (lower = better); 0 = not in
    # forward top-K (reverse-only). A real forward-blocking cutoff must exclude the 0 sentinel.
    thresholds = [3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]
    print(f'{"K":>5} {"kept_pairs":>12} {"avg_cand/entity":>16} {"true_kept":>10} {"recall":>8}')
    for k in thresholds:
        keep = (fused_rank >= 1) & (fused_rank <= k)
        if rev_rank is not None:
            keep = keep | ((rev_rank >= 1) & (rev_rank <= k))
        kept_pairs = int(keep.sum())
        true_kept = int((keep & (y == 1)).sum())
        recall = true_kept / total_true if total_true else 0.0
        avg_per_entity = kept_pairs / n_refs
        print(f'{k:>5} {kept_pairs:>12} {avg_per_entity:>16.3f} {true_kept:>10} {recall:>8.5f}', flush=True)


if __name__ == '__main__':
    main()
