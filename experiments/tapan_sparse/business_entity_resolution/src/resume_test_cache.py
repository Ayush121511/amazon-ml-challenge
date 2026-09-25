"""Finish an interrupted test cache without repeating completed target chunks.

This is for a cache killed before its manifest was written. A chunk is reused
only when its ID array and both sparse matrices are present and readable.
The original SQLite commit precedes matrix writes, so resumed inserts replace
any rows left behind by an interrupted chunk.
"""
import argparse
import json
import os
import pickle
import re
import sqlite3
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse

from compact_topk import decode_target_id, encode_target_id
from retrieval_experiment import records
from test_pipeline import COUNTRIES, FIELDS, atomic_json


CHUNK_RE = re.compile(r'^target-(\d{5})-(?:ids\.npy|name_folded\.npz|address_folded\.npz)$')


def complete_chunks(folder, models, chunk_rows):
    """Return the intact prefix; at most the final chunk may be interrupted."""
    numbers = {int(match.group(1)) for path in folder.iterdir()
               if (match := CHUNK_RE.fullmatch(path.name))}
    if not numbers or numbers != set(range(max(numbers) + 1)):
        raise ValueError(f'Noncontiguous or missing cache chunks in {folder}')
    chunks = []
    for number in range(max(numbers) + 1):
        prefix = f'target-{number:05d}'
        paths = [folder / f'{prefix}-ids.npy'] + [folder / f'{prefix}-{field}.npz' for field in FIELDS]
        if not all(path.is_file() and path.stat().st_size for path in paths):
            if number != max(numbers):
                raise ValueError(f'Incomplete cache chunk before final chunk: {prefix}')
            break
        try:
            ids = np.load(paths[0], allow_pickle=False)
            if ids.ndim != 1 or not 0 < len(ids) <= chunk_rows:
                raise ValueError('Bad ID array length')
            if number != max(numbers) and len(ids) != chunk_rows:
                raise ValueError('Short cache chunk before the final chunk')
            # A walltime kill can leave the final .npz present but truncated.
            # Earlier chunks were fully written before the final flush began.
            if number == max(numbers):
                for field, path in zip(FIELDS, paths[1:]):
                    matrix = sparse.load_npz(path)
                    if matrix.shape != (len(models[field].vocabulary_), len(ids)):
                        raise ValueError('Sparse matrix shape mismatch')
        except (OSError, ValueError, EOFError, KeyError, TypeError, zipfile.BadZipFile) as error:
            if number != max(numbers):
                raise ValueError(f'Corrupt cache chunk before final chunk: {prefix}') from error
            print(f'Rebuilding interrupted {folder.name} chunk {prefix}: {error}', flush=True)
            break
        chunks.append({'prefix': prefix, 'rows': len(ids)})
    return chunks


def resume(args):
    start = time.monotonic()
    cache = args.cache
    if (cache / 'manifest.json').exists():
        print('Completed test cache already exists.', flush=True)
        return
    if not cache.is_dir():
        raise FileNotFoundError(f'Missing partial cache: {cache}')
    paths = [args.data / f'test_source{i}.tsv.gz' for i in (1, 2, 3)]
    signatures = {}
    for path in paths:
        source_manifest = json.loads(Path(str(path) + '.json').read_text())
        if source_manifest['input']['version'] != 2 or source_manifest['input']['limit'] != 0:
            raise ValueError(f'Not a full version-2 input: {path}')
        if path.stat().st_size != source_manifest['output_bytes']:
            raise ValueError(f'Input size mismatch: {path}')
        signatures[path.name] = source_manifest

    models, databases, details = {}, {}, {}
    try:
        for country in COUNTRIES:
            folder = cache / country
            with (folder / 'vectorizers.pkl').open('rb') as stream:
                models[country] = pickle.load(stream)  # Cache created by this project.
            chunks = complete_chunks(folder, models[country], args.chunk_rows)
            with (folder / 'references.jsonl').open(encoding='utf-8') as stream:
                references = sum(1 for _ in stream)
            completed = sum(chunk['rows'] for chunk in chunks)
            details[country] = {'references': references, 'targets': completed, 'chunks': chunks}
            databases[country] = sqlite3.connect(folder / 'targets.sqlite')
            print(f'Reusing {country}: {completed:,} targets in {len(chunks)} chunks', flush=True)
        reused = {country: details[country]['targets'] for country in COUNTRIES}
        seen = dict.fromkeys(COUNTRIES, 0)
        buffers = {country: [] for country in COUNTRIES}

        def flush(country):
            batch = buffers[country]
            if not batch:
                return
            folder = cache / country
            prefix = f'target-{len(details[country]["chunks"]):05d}'
            ids = np.asarray([encode_target_id(row['entity_id']) for row in batch], dtype=np.uint64)
            for row, identifier in zip(batch, ids):
                if decode_target_id(identifier) != row['entity_id'] or int(identifier) >= 2**63:
                    raise ValueError('Target ID cannot be round-tripped through the target store')
            databases[country].executemany('INSERT OR REPLACE INTO targets VALUES (?, ?)',
                ((int(identifier), json.dumps(row, ensure_ascii=False)) for identifier, row in zip(ids, batch)))
            databases[country].commit()
            temporary = folder / f'{prefix}-ids.partial.npy'
            np.save(temporary, ids)
            os.replace(temporary, folder / f'{prefix}-ids.npy')
            for field in FIELDS:
                matrix = models[country][field].transform([row[field] for row in batch])
                temporary = folder / f'{prefix}-{field}.partial.npz'
                sparse.save_npz(temporary, matrix.T.tocsr(), compressed=False)
                os.replace(temporary, folder / f'{prefix}-{field}.npz')
            details[country]['chunks'].append({'prefix': prefix, 'rows': len(batch)})
            details[country]['targets'] += len(batch)
            buffers[country] = []
            print(f'Resumed {country}: {details[country]["targets"]:,} targets; '
                  f'{time.monotonic()-start:.1f}s', flush=True)

        for path in paths[1:]:
            for row in records(path):
                country = row['country']
                if country not in seen:
                    raise ValueError(f'Unexpected country: {country}')
                seen[country] += 1
                if seen[country] <= reused[country]:
                    continue
                buffers[country].append(row)
                if len(buffers[country]) >= args.chunk_rows:
                    flush(country)
        for country in COUNTRIES:
            if seen[country] < reused[country]:
                raise ValueError(f'Input is shorter than existing {country} cache')
            flush(country)
            if details[country]['targets'] != seen[country]:
                raise ValueError(f'Target count mismatch for {country}')
            stored = databases[country].execute('SELECT COUNT(*) FROM targets').fetchone()[0]
            if stored != seen[country]:
                raise ValueError(f'SQLite target count mismatch for {country}: {stored} != {seen[country]}')
    finally:
        for database in databases.values():
            database.close()
    atomic_json(cache / 'manifest.json', {
        'version': 1, 'inputs': signatures, 'fit_rows': args.fit_rows,
        'chunk_rows': args.chunk_rows, 'countries': details,
        'seconds': time.monotonic() - start, 'resume_from_targets': reused,
        'fit_policy': 'Unlabeled test-target prefix per country. No labels or external data. France included.'})
    print('Cache manifest complete.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--fit-rows', type=int, default=20000)
    parser.add_argument('--chunk-rows', type=int, default=50000)
    args = parser.parse_args()
    if args.fit_rows < 1 or args.chunk_rows < 1:
        parser.error('fit-rows and chunk-rows must be positive')
    resume(args)


if __name__ == '__main__':
    main()
