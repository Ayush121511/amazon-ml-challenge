"""Inspect real v4 cache contents before deciding whether feature reuse is possible.

Run on a compute node: compressed NPZ members must be decompressed to count values.
No cache is modified and no record IDs or text are included in the aggregate report.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def inspect_reverse(path):
    with np.load(path) as data:
        out = {'fields': sorted(data.files), 'combined_score_present': 'score_combined' in data.files,
               'combined_rank_present': 'rank_combined' in data.files}
        if out['combined_score_present']:
            score = data['score_combined']
            out.update(combined_score_rows=int(score.size),
                       combined_score_nonzero=int(np.count_nonzero(score)),
                       combined_score_finite=bool(np.isfinite(score).all()))
        return out


def inspect_dense(path):
    with np.load(path) as data:
        score = data['rev_score']
        if score.ndim != 2:
            raise ValueError(f'Expected target-by-rank reverse scores: {path}')
        out = {'targets': len(score), 'reverse_k': score.shape[1],
               'target_margin_available_without_embedding_search': score.shape[1] >= 2}
        if len(score) and score.shape[1] >= 2:
            margin = score[:, 0].astype(np.float32) - score[:, 1].astype(np.float32)
            out.update(top12_margin_mean=float(margin.mean()),
                       top12_margin_min=float(margin.min()),
                       top12_margin_max=float(margin.max()))
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = {}
    for split in ('train', 'test'):
        reverse_root = a.root / f'artifacts/posting_index_v1/{split}_reverse_t'
        manifest = a.root / f'artifacts/posting_index_v1/{split}_ref_index/manifest.json'
        dense_root = a.root / f'artifacts/dense_v1/{split}'
        item = {'reference_index_routes': None,
                'reverse_files': {}, 'dense_files': {}}
        if manifest.exists():
            item['reference_index_routes'] = json.loads(manifest.read_text())['routes']
        for path in sorted(reverse_root.glob('*/reverse.npz')):
            item['reverse_files'][path.parent.name] = inspect_reverse(path)
        for path in sorted(dense_root.glob('dense_*.npz')):
            item['dense_files'][path.name] = inspect_dense(path)
        report[split] = item
    report['interpretation'] = (
        'A missing or all-zero combined score offers no new signal by wiring alone. '
        'Per-route ranks cannot be recovered from fused rank. Dense reverse top-2 scores '
        'suffice for target confidence margins; stored pairs still need aligned new feature columns.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
