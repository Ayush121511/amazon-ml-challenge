"""Reverse blocking: every target record searches an index of Source 1 references.

Ground truth links each target to at most one reference, and a target's best reference
is usually clear even when a reference's own target list is crowded by look-alikes (chain
stores, similar names). Build a posting index over Source 1 with posting_index.build,
then run `query` here: each target keeps its fused top-M references. Label-free.

Output per country: reverse.npz with t_ord (target ordinal in the forward target index),
r_ord (reference ordinal in the reference index), rank, rrf and per-route scores, sorted by
r_ord so a reference's reverse candidates are one contiguous slice.
"""
import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np

from posting_index import CountryIndex, load_manifest, ROUTES


def target_chunks(path, rows):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        batch = []
        for row in csv.DictReader(f, delimiter='\t'):
            batch.append(row)
            if len(batch) >= rows:
                yield batch
                batch = []
        if batch:
            yield batch


def query(ref_index_root, target_index_root, output, top_k, top_terms, keep, batch, threads):
    from submission_pipeline import candidates
    ref_manifest = load_manifest(ref_index_root)
    target_manifest = load_manifest(target_index_root)
    output.mkdir(parents=True)
    report = {'countries': {}, 'parameters': {'top_k': top_k, 'top_terms': top_terms, 'keep': keep}}
    for country, entry in sorted(target_manifest['countries'].items()):
        started = time.monotonic()
        directory = output / entry['dir']
        directory.mkdir()
        if country not in ref_manifest['countries']:
            report['countries'][country] = {'targets': entry['targets'], 'pairs': 0, 'no_reference_index': True}
            continue
        index = CountryIndex(ref_index_root, country, ref_manifest)
        parts, offset = [], 0
        for rows in target_chunks(Path(target_index_root) / entry['dir'] / 'targets.tsv.gz', batch):
            c = candidates(index, rows, top_k, top_terms, keep, threads)
            parts.append({'t_ord': (c['ref'] + offset).astype(np.int32), 'r_ord': c['ord'].astype(np.int32),
                          'rank': c['fused_rank'].astype(np.int16), 'rrf': c['rrf'].astype(np.float32),
                          **{f'score_{r}': c[f'score_{r}'].astype(np.float16) for r in ROUTES}})
            offset += len(rows)
            print(f'{country}: {offset:,}/{entry["targets"]:,} targets, '
                  f'{time.monotonic() - started:.0f}s', flush=True)
        if offset != entry['targets']:
            raise ValueError(f'{country}: read {offset} targets, index has {entry["targets"]}')
        merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        order = np.argsort(merged['r_ord'], kind='stable')
        np.savez(directory / 'reverse.npz', **{k: v[order] for k, v in merged.items()})
        report['countries'][country] = {'targets': offset, 'pairs': int(len(order)),
                                        'seconds': time.monotonic() - started}
        del index
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


class Reverse:
    """Reverse candidates of one country, looked up by reference entity_id."""

    def __init__(self, reverse_root, ref_index_root, target_dir, country):
        ref_manifest = load_manifest(ref_index_root)
        self.arrays = None
        if country not in ref_manifest['countries']:
            return
        ids = np.load(Path(ref_index_root) / ref_manifest['countries'][country]['dir'] / 'ids.npy')
        self.position = {i.decode('utf-8'): n for n, i in enumerate(ids)}
        data = np.load(Path(reverse_root) / target_dir / 'reverse.npz')
        self.arrays = {k: data[k] for k in data.files}
        self.bounds = np.searchsorted(self.arrays['r_ord'], np.arange(len(ids) + 1))

    def lookup(self, rows):
        """Arrays (ref = row position, t_ord, rank, rrf, scores) for these reference rows."""
        empty = {'ref': np.zeros(0, np.int64), 't_ord': np.zeros(0, np.int64)}
        if self.arrays is None:
            return empty
        slices, refs = [], []
        for i, row in enumerate(rows):
            n = self.position.get(row['entity_id'])
            if n is None:
                continue
            lo, hi = self.bounds[n], self.bounds[n + 1]
            if hi > lo:
                slices.append(np.arange(lo, hi))
                refs.append(np.full(hi - lo, i))
        if not slices:
            return empty
        take = np.concatenate(slices)
        out = {k: v[take] for k, v in self.arrays.items() if k != 'r_ord'}
        out['ref'] = np.concatenate(refs)
        out['t_ord'] = out['t_ord'].astype(np.int64)
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ref-index', type=Path, required=True, help='posting_index built over Source 1')
    p.add_argument('--target-index', type=Path, required=True, help='posting_index built over Sources 2+3')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--top-terms', type=int, default=64)
    p.add_argument('--keep', type=int, default=10, help='References kept per target')
    p.add_argument('--batch', type=int, default=8192)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f'{a.output} exists; choose a new output directory')
    query(a.ref_index, a.target_index, a.output, a.top_k, a.top_terms, a.keep, a.batch, a.threads)


if __name__ == '__main__':
    main()
