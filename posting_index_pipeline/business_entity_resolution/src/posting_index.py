"""Persistent hashed TF-IDF posting index over target records, built once per split.

Routes: name char 3-5 grams, address char 3-5 grams, and address-number+word
anchors. Terms are hashed (no fitted vocabulary), so unseen countries such as
France need no special handling. IDF comes from the target pool itself, never
from labels. Terms whose document frequency exceeds a cap are left out of the
postings; queries use only their highest-weighted remaining terms, so query
cost is bounded by posting lengths instead of the pool size.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from transliterate import phonetic, transliterate

FORMAT = 1
N_FEATURES = 2 ** 22
ROUTES = ('name', 'address', 'anchor', 'phonetic', 'combined')
FIELD = {'name': 'name_folded', 'address': 'address_folded', 'anchor': 'address_folded',
         'phonetic': 'name_folded', 'combined': None}


def route_text(route, row):
    """Text a route indexes. 'combined' scores name and address in ONE vector, so a chain
    branch whose name matches thousands of targets still ranks first when its address
    matches too; the separate name/address routes each get crowded out in that case."""
    if route == 'combined':
        return row['name_folded'] + ' ' + row['address_folded']
    return row[FIELD[route]]
TEXT_COLUMNS = ['entity_id', 'business_name', 'business_address',
                'name_folded', 'address_folded']


def anchor_keys(address):
    """Number+neighbouring-word keys. A bare number is too common to block on."""
    tokens = address.split()
    keys = set()
    for i, token in enumerate(tokens):
        numbers = {n.lstrip('0') or '0' for n in re.findall(r'\d+', token)}
        if not numbers:
            continue
        neighbours = []
        for step in (-1, 1):
            for j in range(i + step, i + 3 * step, step):
                if 0 <= j < len(tokens):
                    word = tokens[j]
                    if len(word) >= 3 and not any(c.isdigit() for c in word):
                        neighbours.append(word)
                        break
        for number in numbers:
            for word in neighbours:
                keys.add(number + '|' + word)
    return sorted(keys)


def phonetic_grams(text):
    """Word-bounded 2-4 character grams of the name's sound skeleton (see transliterate.phonetic)."""
    grams = []
    for word in phonetic(text).split():
        w = f' {word} '
        grams.extend(w[i:i + n] for n in (2, 3, 4) for i in range(len(w) - n + 1))
    return grams


def vectorizer(route):
    if route == 'phonetic':
        return HashingVectorizer(analyzer=phonetic_grams, n_features=N_FEATURES,
                                 alternate_sign=False, norm=None, dtype=np.float32)
    if route == 'anchor':
        return HashingVectorizer(analyzer=anchor_keys, n_features=N_FEATURES,
                                 alternate_sign=False, norm=None, dtype=np.float32)
    return HashingVectorizer(analyzer='char_wb', ngram_range=(3, 5), lowercase=False,
                             n_features=N_FEATURES, alternate_sign=False, norm=None,
                             dtype=np.float32)


_VECTORIZERS = {route: vectorizer(route) for route in ROUTES}


def raw_counts(route, texts):
    # Indic-script text is transliterated to Latin at build and query time alike, so a
    # Devanagari/Tamil/... target can meet a Latin reference. ASCII text is unchanged.
    matrix = _VECTORIZERS[route].transform([transliterate(t) for t in texts]).tocsr()
    matrix.sum_duplicates()
    return matrix


def weigh(counts, idf, keep, top_terms=0):
    """TF-IDF, L2-normalised over ALL terms, then restricted to indexed terms."""
    matrix = counts.copy()
    matrix.data *= idf[matrix.indices]
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    matrix = sp.diags((1.0 / norms).astype(np.float32)) @ matrix
    matrix = matrix.tocsr()
    matrix.data *= keep[matrix.indices]
    matrix.eliminate_zeros()
    if top_terms:
        matrix = keep_top_terms(matrix, top_terms)
    return matrix.astype(np.float32)


def keep_top_terms(matrix, limit):
    indptr, data = matrix.indptr, matrix.data
    mask = np.ones(len(data), dtype=bool)
    for row in range(matrix.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        if end - start > limit:
            order = np.argpartition(-data[start:end], limit)[limit:]
            mask[start + order] = False
    trimmed = matrix.copy()
    trimmed.data = np.where(mask, data, 0).astype(np.float32)
    trimmed.eliminate_zeros()
    return trimmed


def records(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        yield from csv.DictReader(f, delimiter='\t')


def country_dir(country):
    slug = re.sub(r'[^A-Za-z0-9]+', '_', country).strip('_') or 'x'
    return slug + '_' + hashlib.sha1(country.encode('utf-8')).hexdigest()[:8]


# ---------------------------------------------------------------- build ----

def _df_chunk(texts_by_route):
    result = {}
    for route, texts in texts_by_route.items():
        counts = raw_counts(route, texts)
        result[route] = np.unique(counts.indices, return_counts=True)
    return result


_WEIGHTS = {}


def _init_weights(weights):
    _WEIGHTS.update(weights)


def _weighted_chunk(country, texts_by_route):
    out = {}
    for route, texts in texts_by_route.items():
        idf, keep = _WEIGHTS[(country, route)]
        out[route] = weigh(raw_counts(route, texts), idf, keep)
    return out


def _stream_chunks(paths, chunk_rows, on_row=None):
    """Yields (country, rows) in a deterministic order; same order on every pass."""
    buffers = defaultdict(list)
    for path in paths:
        for row in records(path):
            if on_row:
                on_row(row)
            buffers[row['country']].append(row)
            if len(buffers[row['country']]) >= chunk_rows:
                yield row['country'], buffers.pop(row['country'])
    for country in sorted(buffers):
        yield country, buffers[country]


def _texts(rows):
    return {route: [route_text(route, r) for r in rows] for route in ROUTES}


def _bounded_map(pool, fn, jobs, window):
    pending = []
    for job in jobs:
        pending.append(pool.submit(fn, *job))
        if len(pending) >= window:
            yield pending.pop(0).result()
    for future in pending:
        yield future.result()


def build(paths, output, workers=4, chunk_rows=50000, max_df_fraction=0.005,
          min_max_df=1000):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new output directory')
    started = time.monotonic()
    staging = output.with_name(output.name + '.partial')
    staging.mkdir(parents=True)
    df = defaultdict(lambda: np.zeros(N_FEATURES, dtype=np.int64))
    sizes = defaultdict(int)
    text_files = {}

    def write_text(row):
        country = row['country']
        if country not in text_files:
            directory = staging / country_dir(country)
            directory.mkdir()
            handle = gzip.open(directory / 'targets.tsv.gz', 'wt', encoding='utf-8',
                               newline='', compresslevel=1)
            writer = csv.writer(handle, delimiter='\t', lineterminator='\n',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(TEXT_COLUMNS)
            text_files[country] = (handle, writer)
        text_files[country][1].writerow([row[c] for c in TEXT_COLUMNS])
        sizes[country] += 1

    with ProcessPoolExecutor(max_workers=workers) as pool:
        countries_in_order = []

        def tagged():
            for country, rows in _stream_chunks(paths, chunk_rows, write_text):
                countries_in_order.append(country)
                yield (_texts(rows),)

        for i, result in enumerate(_bounded_map(pool, _df_chunk, tagged(), workers * 2)):
            country = countries_in_order[i]
            for route, (terms, counts) in result.items():
                df[(country, route)][terms] += counts
    for handle, _ in text_files.values():
        handle.close()
    pass1 = time.monotonic()
    print(f'Pass 1 (document frequencies) done in {pass1 - started:.1f}s: '
          f'{dict(sizes)}', flush=True)

    weights, route_meta = {}, {}
    for (country, route), counts in df.items():
        n = sizes[country]
        cap = max(min_max_df, int(max_df_fraction * n))
        idf = (np.log((1 + n) / (1 + counts)) + 1).astype(np.float32)
        keep = ((counts > 0) & (counts <= cap)).astype(np.float32)
        weights[(country, route)] = (idf, keep)
        route_meta[(country, route)] = {
            'max_df': cap, 'terms_seen': int((counts > 0).sum()),
            'terms_indexed': int(keep.sum()), 'terms_dropped_common': int((counts > cap).sum())}
    del df

    chunk_counter = defaultdict(int)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_weights,
                             initargs=(weights,)) as pool:
        order = []

        def jobs2():
            for country, rows in _stream_chunks(paths, chunk_rows):
                order.append(country)
                yield (country, _texts(rows))

        for i, result in enumerate(_bounded_map(pool, _weighted_chunk, jobs2(), workers * 2)):
            country = order[i]
            seq = chunk_counter[country]
            chunk_counter[country] += 1
            for route, matrix in result.items():
                directory = staging / country_dir(country) / 'chunks' / route
                directory.mkdir(parents=True, exist_ok=True)
                sp.save_npz(directory / f'{seq:06d}.npz', matrix, compressed=False)
    pass2 = time.monotonic()
    print(f'Pass 2 (weighted chunks) done in {pass2 - pass1:.1f}s', flush=True)

    manifest = {'format': FORMAT, 'n_features': N_FEATURES, 'routes': list(ROUTES),
                'inputs': [{'path': str(p.resolve()), 'bytes': p.stat().st_size} for p in paths],
                'max_df_fraction': max_df_fraction, 'min_max_df': min_max_df,
                'countries': {}}
    for country in sorted(sizes):
        base = staging / country_dir(country)
        entry = {'dir': country_dir(country), 'targets': sizes[country], 'routes': {}}
        for route in ROUTES:
            parts = sorted((base / 'chunks' / route).glob('*.npz'))
            matrix = sp.vstack([sp.load_npz(p) for p in parts], format='csr')
            if matrix.shape[0] != sizes[country]:
                raise ValueError(f'{country} {route}: {matrix.shape[0]} rows != {sizes[country]}')
            postings = matrix.T.tocsr()
            del matrix
            index_dtype = np.int32 if postings.nnz < 2 ** 31 - 1 else np.int64
            np.save(base / f'{route}.indptr.npy', postings.indptr.astype(index_dtype))
            np.save(base / f'{route}.indices.npy', postings.indices.astype(index_dtype))
            np.save(base / f'{route}.data.npy', postings.data.astype(np.float32))
            idf, keep = weights[(country, route)]
            np.save(base / f'{route}.idf.npy', idf)
            np.save(base / f'{route}.keep.npy', keep.astype(bool))
            entry['routes'][route] = {**route_meta[(country, route)], 'nnz': int(postings.nnz)}
            del postings
            for p in parts:
                p.unlink()
            (base / 'chunks' / route).rmdir()
        (base / 'chunks').rmdir()
        ids = []
        with gzip.open(base / 'targets.tsv.gz', 'rt', encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                ids.append(row['entity_id'])
        np.save(base / 'ids.npy', np.array(ids, dtype='S'))
        manifest['countries'][country] = entry
    manifest['build_seconds'] = {'pass1_df': pass1 - started, 'pass2_weights': pass2 - pass1,
                                 'finalize': time.monotonic() - pass2,
                                 'total': time.monotonic() - started}
    manifest['index_bytes'] = sum(p.stat().st_size for p in staging.rglob('*') if p.is_file())
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    os.replace(staging, output)
    return manifest


# ---------------------------------------------------------------- query ----

class CountryIndex:
    def __init__(self, root, country, manifest):
        entry = manifest['countries'][country]
        base = Path(root) / entry['dir']
        self.country = country
        self.size = entry['targets']
        self.ids = np.load(base / 'ids.npy')
        self.routes = {}
        for route in manifest['routes']:
            postings = sp.csr_matrix((np.load(base / f'{route}.data.npy'),
                                      np.load(base / f'{route}.indices.npy'),
                                      np.load(base / f'{route}.indptr.npy')),
                                     shape=(manifest['n_features'], self.size), copy=False)
            self.routes[route] = (postings, np.load(base / f'{route}.idf.npy'),
                                  np.load(base / f'{route}.keep.npy').astype(np.float32))


def load_manifest(root):
    manifest = json.loads((Path(root) / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['format'] != FORMAT:
        raise ValueError('Unsupported index format')
    return manifest


def search(index, rows, top_k=100, top_terms=32, threads=4):
    """Returns per-row {route: [(target_id, score), ...]} sorted by score."""
    from sparse_dot_topn import sp_matmul_topn
    results = [dict() for _ in rows]
    for route, (postings, idf, keep) in index.routes.items():
        queries = weigh(raw_counts(route, [route_text(route, r) for r in rows]), idf, keep, top_terms)
        if queries.nnz == 0 or postings.nnz == 0:
            for result in results:
                result[route] = []
            continue
        scores = sp_matmul_topn(queries, postings, top_n=top_k, threshold=0.0,
                                sort=True, n_threads=threads)
        for i, result in enumerate(results):
            start, end = scores.indptr[i], scores.indptr[i + 1]
            result[route] = [(index.ids[j].decode('utf-8'), float(s))
                             for j, s in zip(scores.indices[start:end], scores.data[start:end])]
    return results


def fuse(channels, budget=50, k=60):
    fusion = defaultdict(float)
    for hits in channels.values():
        for rank, (identifier, _) in enumerate(hits, 1):
            fusion[identifier] += 1.0 / (k + rank)
    return sorted(fusion, key=lambda x: (-fusion[x], x))[:budget]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='Directory with preprocessed v2 <split>_source{2,3}.tsv.gz')
    parser.add_argument('--split', choices=['train', 'test'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--chunk-rows', type=int, default=50000)
    parser.add_argument('--max-df-fraction', type=float, default=0.005)
    parser.add_argument('--min-max-df', type=int, default=1000)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--sources', default='2,3',
                        help='Source numbers to index; "1" builds a reference index for reverse search')
    args = parser.parse_args()
    paths = [args.data / f'{args.split}_source{i}.tsv.gz' for i in args.sources.split(',')]
    for path in paths:
        manifest = json.loads(Path(str(path) + '.json').read_text())
        if manifest['input']['version'] != 2 or (manifest['input']['limit'] and not args.smoke):
            raise ValueError('Full v2 preprocessing inputs required')
        if path.stat().st_size != manifest['output_bytes']:
            raise ValueError('Input size mismatch')
    result = build(paths, args.output, args.workers, args.chunk_rows,
                   args.max_df_fraction, args.min_max_df)
    try:
        import resource
        result['peak_rss_kib_linux'] = {
            'self': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'largest_worker': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
        (args.output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    except ImportError:
        pass
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
