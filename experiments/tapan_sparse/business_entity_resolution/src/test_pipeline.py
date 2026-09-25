"""Cached, resumable test retrieval, batched matching, and TSV assembly.

This module reads no ground truth. Cache building is a one-time streaming pass;
country retrieval resumes at target-chunk checkpoints within reference blocks.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
import pickle
import sqlite3
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from compact_topk import CompactTopK, encode_target_id, decode_target_id
from pair_features import feature_spec
from retrieval_experiment import records

FIELDS = ('name_folded', 'address_folded')
COUNTRIES = ('India', 'US', 'France')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def json_rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def batched(stream, size):
    while True:
        batch = list(itertools.islice(stream, size))
        if not batch:
            return
        yield batch


def cache_build(args):
    if args.output.exists():
        raise FileExistsError('Cache directory exists; use its completed manifest or a new directory')
    paths = [args.data / f'test_source{i}.tsv.gz' for i in (1, 2, 3)]
    signatures = {}
    for path in paths:
        manifest = json.loads(Path(str(path) + '.json').read_text())
        if manifest['input']['version'] != 2 or manifest['input']['limit'] != 0:
            raise ValueError('Full version-2 test inputs are required')
        if path.stat().st_size != manifest['output_bytes']:
            raise ValueError(f'Input size mismatch: {path}')
        signatures[path.name] = manifest
    start = time.monotonic()
    corpus = {country: [] for country in COUNTRIES}
    for path in paths[1:]:
        for row in records(path):
            country = row['country']
            if country not in corpus:
                raise ValueError(f'Unexpected country: {country}')
            if len(corpus[country]) < args.fit_rows:
                corpus[country].append(row)
            if all(len(pool) >= args.fit_rows for pool in corpus.values()):
                break
        if all(len(pool) >= args.fit_rows for pool in corpus.values()):
            break
    if any(len(pool) < args.fit_rows for pool in corpus.values()):
        raise ValueError('Insufficient target fitting rows for a country')
    args.output.mkdir(parents=True)
    models, databases, reference_streams, buffers = {}, {}, {}, {}
    details = {country: {'references': 0, 'targets': 0, 'chunks': []} for country in COUNTRIES}
    try:
        for country in COUNTRIES:
            folder = args.output / country
            folder.mkdir()
            models[country] = {}
            for field in FIELDS:
                model = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2,
                                        max_features=300000, dtype=np.float32)
                model.fit([row[field] for row in corpus[country]])
                models[country][field] = model
            with (folder / 'vectorizers.pkl').open('wb') as stream:
                pickle.dump(models[country], stream)
            databases[country] = sqlite3.connect(folder / 'targets.sqlite')
            databases[country].execute('PRAGMA cache_size=-65536')
            databases[country].execute('CREATE TABLE targets (id INTEGER PRIMARY KEY, record TEXT NOT NULL)')
            reference_streams[country] = (folder / 'references.jsonl').open('w', encoding='utf-8')
            buffers[country] = []
        del corpus
        for row in records(paths[0]):
            country = row['country']
            reference_streams[country].write(json.dumps(row, ensure_ascii=False) + '\n')
            details[country]['references'] += 1
        for stream in reference_streams.values():
            stream.close()

        def flush(country):
            batch = buffers[country]
            if not batch:
                return
            folder = args.output / country
            number = len(details[country]['chunks'])
            prefix = f'target-{number:05d}'
            ids = np.asarray([encode_target_id(row['entity_id']) for row in batch], dtype=np.uint64)
            for row, identifier in zip(batch, ids):
                if decode_target_id(identifier) != row['entity_id'] or int(identifier) >= 2**63:
                    raise ValueError('Target ID cannot be round-tripped through the target store')
            databases[country].executemany('INSERT INTO targets VALUES (?, ?)',
                ((int(identifier), json.dumps(row, ensure_ascii=False)) for identifier, row in zip(ids, batch)))
            databases[country].commit()
            np.save(folder / f'{prefix}-ids.npy', ids)
            for field in FIELDS:
                matrix = models[country][field].transform([row[field] for row in batch])
                sparse.save_npz(folder / f'{prefix}-{field}.npz', matrix.T.tocsr(), compressed=False)
            details[country]['chunks'].append({'prefix': prefix, 'rows': len(batch)})
            details[country]['targets'] += len(batch)
            buffers[country] = []
            print(f'Cached {country}: {details[country]["targets"]:,} targets; {time.monotonic()-start:.1f}s', flush=True)

        for path in paths[1:]:
            for row in records(path):
                country = row['country']
                buffers[country].append(row)
                if len(buffers[country]) >= args.chunk_rows:
                    flush(country)
        for country in COUNTRIES:
            flush(country)
    finally:
        for stream in reference_streams.values():
            stream.close()
        for database in databases.values():
            database.close()
    atomic_json(args.output / 'manifest.json', {
        'version': 1, 'inputs': signatures, 'fit_rows': args.fit_rows,
        'chunk_rows': args.chunk_rows, 'countries': details,
        'seconds': time.monotonic() - start,
        'fit_policy': 'Unlabeled test-target prefix per country. No labels or external data. France included.'})


def save_arrays(path, stores, next_chunk):
    temporary = path.with_suffix('.partial.npz')
    np.savez(temporary, next_chunk=np.asarray(next_chunk),
             ids0=stores[0].ids, scores0=stores[0].scores,
             ids1=stores[1].ids, scores1=stores[1].scores)
    os.replace(temporary, path)


def retrieve(args):
    manifest_path = args.cache / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    signature = {'cache_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                 'country': args.country, 'top_k': args.top_k, 'reference_block': args.reference_block}
    args.output.mkdir(parents=True, exist_ok=True)
    run_file = args.output / 'run.json'
    if run_file.exists():
        if json.loads(run_file.read_text()) != signature:
            raise ValueError('Resume parameters/cache differ from the saved run')
    else:
        atomic_json(run_file, signature)
    if (args.output / 'complete.json').exists():
        print('Country retrieval already complete', flush=True)
        return
    folder = args.cache / args.country
    with (folder / 'vectorizers.pkl').open('rb') as stream:
        models = pickle.load(stream)  # Only the cache created by this program.
    chunks = manifest['countries'][args.country]['chunks']
    start = time.monotonic()
    completed = 0
    chunk_work = 0
    for number, rows in enumerate(batched(json_rows(folder / 'references.jsonl'), args.reference_block)):
        block = args.output / f'block-{number:05d}'
        block.mkdir(exist_ok=True)
        if (block / 'complete.json').exists():
            completed += len(rows)
            continue
        queries = [models[field].transform([row[field] for row in rows]) for field in FIELDS]
        stores = [CompactTopK(len(rows), args.top_k) for _ in FIELDS]
        checkpoint = block / 'checkpoint.npz'
        next_chunk = 0
        if checkpoint.exists():
            with np.load(checkpoint) as state:
                next_chunk = int(state['next_chunk'])
                for index, store in enumerate(stores):
                    if state[f'ids{index}'].shape != store.ids.shape:
                        raise ValueError('Invalid checkpoint shape')
                    store.ids[:] = state[f'ids{index}']
                    store.scores[:] = state[f'scores{index}']
        for chunk_index in range(next_chunk, len(chunks)):
            prefix = chunks[chunk_index]['prefix']
            ids = np.load(folder / f'{prefix}-ids.npy')
            for index, field in enumerate(FIELDS):
                matrix = sparse.load_npz(folder / f'{prefix}-{field}.npz')
                for lo in range(0, len(rows), args.query_batch):
                    scores = sp_matmul_topn(queries[index][lo:lo+args.query_batch], matrix,
                                            top_n=args.top_k, threshold=0.0, sort=False, n_threads=args.threads)
                    stores[index].update(lo, scores, ids)
            chunk_work += 1
            if (chunk_index + 1) % args.checkpoint_chunks == 0:
                save_arrays(checkpoint, stores, chunk_index + 1)
            print(f'{args.country} block {number}: target chunk {chunk_index+1}/{len(chunks)}; '
                  f'{completed:,} references finished; {time.monotonic()-start:.1f}s', flush=True)
            if args.max_chunks and chunk_work >= args.max_chunks:
                save_arrays(checkpoint, stores, chunk_index + 1)
                print('Checkpoint saved at requested work limit', flush=True)
                return
        save_arrays(block / 'candidates.npz', stores, len(chunks))
        temporary = block / 'references.jsonl.partial'
        with temporary.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        os.replace(temporary, block / 'references.jsonl')
        atomic_json(block / 'complete.json', {'references': len(rows)})
        if checkpoint.exists():
            checkpoint.unlink()
        completed += len(rows)
        print(f'{args.country}: retrieved {completed:,} references', flush=True)
    if completed != manifest['countries'][args.country]['references']:
        raise ValueError('Country reference count mismatch')
    atomic_json(args.output / 'complete.json', {'references': completed, 'seconds_this_invocation': time.monotonic()-start})


def ranked_candidates(stores, index):
    routes = {label: [tid for score, tid in store.ranked(index)]
              for store, label in zip(stores, ('name_words', 'address_words'))}
    routes['name_trigrams'] = []
    union = sorted(set().union(*map(set, routes.values())))
    ranks = {label: {tid: rank for rank, tid in enumerate(ids, 1)} for label, ids in routes.items()}
    return union, ranks


def fetch_targets(database, identifiers):
    result = {}
    for batch in batched(iter(identifiers), 900):
        placeholders = ','.join('?' for _ in batch)
        for identifier, record in database.execute(f'SELECT id, record FROM targets WHERE id IN ({placeholders})', batch):
            result[decode_target_id(identifier)] = json.loads(record)
    if len(result) != len(identifiers):
        raise ValueError('Candidate targets missing from the cache')
    return result


def score_country(args):
    from lightgbm import Booster
    if not (args.retrieval / 'complete.json').exists():
        raise ValueError('Country retrieval is incomplete')
    run = json.loads((args.retrieval / 'run.json').read_text())
    manifest_path = args.cache / 'manifest.json'
    if run['cache_manifest_sha256'] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError('Retrieval/cache mismatch')
    report = json.loads((args.model / 'report.json').read_text())
    threshold = report['threshold_selected_on_dev']
    model = Booster(model_file=str(args.model / 'model.txt'))
    extract_features, feature_names = feature_spec(report.get('feature_version', 'v1'))
    if model.num_feature() != len(feature_names) or not 0 <= threshold <= 1:
        raise ValueError('Unexpected matcher features/threshold')
    signature = {'retrieval': run, 'model_sha256': hashlib.sha256((args.model / 'model.txt').read_bytes()).hexdigest(),
                 'threshold': threshold}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'run.json').exists():
        if json.loads((args.output / 'run.json').read_text()) != signature:
            raise ValueError('Use a new scoring output for a different model or threshold')
    else:
        atomic_json(args.output / 'run.json', signature)
    database = sqlite3.connect((args.cache / run['country'] / 'targets.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    database.execute('PRAGMA cache_size=-131072')
    count = 0
    try:
        for block in sorted(args.retrieval.glob('block-*')):
            expected = json.loads((block / 'complete.json').read_text())['references']
            destination = args.output / block.name
            destination.mkdir(exist_ok=True)
            if (destination / 'complete.json').exists():
                count += expected
                continue
            with np.load(block / 'candidates.npz') as saved:
                stores = [CompactTopK(expected, run['top_k']) for _ in FIELDS]
                for index, store in enumerate(stores):
                    store.ids[:] = saved[f'ids{index}']
                    store.scores[:] = saved[f'scores{index}']
            written = 0
            candidate_path = destination / 'candidate_pairs.tsv.partial'
            matching_path = destination / 'matching_results.tsv.partial'
            with candidate_path.open('w', encoding='utf-8', newline='') as cf, matching_path.open('w', encoding='utf-8', newline='') as mf:
                cw, mw = csv.writer(cf, delimiter='\t'), csv.writer(mf, delimiter='\t')
                cw.writerow(['source1_entity_id', 'candidate_entity_ids'])
                mw.writerow(['source1_entity_id', 'matched_entity_ids'])
                for batch in batched(json_rows(block / 'references.jsonl'), args.score_batch):
                    candidates = [ranked_candidates(stores, written + i) for i in range(len(batch))]
                    identifiers = {encode_target_id(tid) for union, ranks in candidates for tid in union}
                    target_rows = fetch_targets(database, identifiers)
                    pair_count = sum(len(union) for union, ranks in candidates)
                    x = np.empty((pair_count, len(feature_names)), dtype=np.float32)
                    position = 0
                    for row, (union, ranks) in zip(batch, candidates):
                        for tid in union:
                            pair = {'reference': row, 'target': target_rows[tid],
                                    'retrieval_ranks': {label: rank.get(tid) for label, rank in ranks.items()}}
                            x[position] = extract_features(pair)
                            position += 1
                    probabilities = model.predict(x, num_threads=args.threads) if pair_count else []
                    position = 0
                    for row, (union, ranks) in zip(batch, candidates):
                        predicted = [tid for tid, probability in zip(union, probabilities[position:position+len(union)])
                                     if probability >= threshold]
                        cw.writerow([row['entity_id'], ','.join(union)])
                        mw.writerow([row['entity_id'], ','.join(predicted)])
                        position += len(union)
                    written += len(batch)
                    if written % 1000 == 0:
                        print(f'Scoring {run["country"]} {block.name}: {written:,}/{expected:,}', flush=True)
            if written != expected:
                raise ValueError('Scored reference count mismatch')
            os.replace(candidate_path, destination / 'candidate_pairs.tsv')
            os.replace(matching_path, destination / 'matching_results.tsv')
            atomic_json(destination / 'complete.json', {'references': written})
            count += written
            print(f'Scored {run["country"]}: {count:,} references', flush=True)
    finally:
        database.close()
    if count != json.loads((args.retrieval / 'complete.json').read_text())['references']:
        raise ValueError('Incomplete country scoring')
    atomic_json(args.output / 'complete.json', {'references': count})


def assemble(args):
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    if args.output.exists():
        raise FileExistsError('Use a fresh final output directory')
    args.output.mkdir(parents=True)
    expected = sqlite3.connect(args.output / 'validation.sqlite')
    expected.execute('CREATE TABLE refs (id TEXT PRIMARY KEY, seen INTEGER NOT NULL DEFAULT 0)')
    for country in COUNTRIES:
        expected.executemany('INSERT INTO refs(id) VALUES (?)',
                            ((row['entity_id'],) for row in json_rows(args.cache / country / 'references.jsonl')))
    expected.commit()
    count = 0
    try:
        with (args.output / 'matching_results.tsv.partial').open('w', encoding='utf-8', newline='') as mf, \
             (args.output / 'candidate_pairs.tsv.partial').open('w', encoding='utf-8', newline='') as cf:
            mw, cw = csv.writer(mf, delimiter='\t'), csv.writer(cf, delimiter='\t')
            mw.writerow(['source1_entity_id', 'matched_entity_ids'])
            cw.writerow(['source1_entity_id', 'candidate_entity_ids'])
            for country in COUNTRIES:
                folder = args.scored / country
                completed = json.loads((folder / 'complete.json').read_text())
                if completed['references'] != manifest['countries'][country]['references']:
                    raise ValueError('Scored country count mismatch')
                for block in sorted(folder.glob('block-*')):
                    with (block / 'matching_results.tsv').open(encoding='utf-8', newline='') as m, \
                         (block / 'candidate_pairs.tsv').open(encoding='utf-8', newline='') as c:
                        matches, candidates = csv.DictReader(m, delimiter='\t'), csv.DictReader(c, delimiter='\t')
                        for mr, cr in itertools.zip_longest(matches, candidates):
                            if mr is None or cr is None or mr['source1_entity_id'] != cr['source1_entity_id']:
                                raise ValueError('Submission rows are misaligned')
                            sid = mr['source1_entity_id']
                            pred = list(filter(None, mr['matched_entity_ids'].split(',')))
                            cand = list(filter(None, cr['candidate_entity_ids'].split(',')))
                            if len(set(pred)) != len(pred) or len(set(cand)) != len(cand) or not set(pred) <= set(cand):
                                raise ValueError('Duplicate IDs or matches outside candidate set')
                            changed = expected.execute('UPDATE refs SET seen=1 WHERE id=? AND seen=0', (sid,)).rowcount
                            if changed != 1:
                                raise ValueError('Unknown or duplicate reference ID')
                            mw.writerow([sid, mr['matched_entity_ids']])
                            cw.writerow([sid, cr['candidate_entity_ids']])
                            count += 1
                    expected.commit()
        if expected.execute('SELECT COUNT(*) FROM refs WHERE seen=0').fetchone()[0]:
            raise ValueError('Missing test references')
    finally:
        expected.close()
    for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
        os.replace(args.output / (name + '.partial'), args.output / name)
    atomic_json(args.output / 'report.json', {'references': count, 'status': 'assembled; run organizer validator before upload'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('cache')
    build.add_argument('--data', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--fit-rows', type=int, default=20000)
    build.add_argument('--chunk-rows', type=int, default=50000)
    query = commands.add_parser('retrieve')
    query.add_argument('--cache', type=Path, required=True)
    query.add_argument('--country', choices=COUNTRIES, required=True)
    query.add_argument('--output', type=Path, required=True)
    query.add_argument('--reference-block', type=int, default=50000)
    query.add_argument('--query-batch', type=int, default=5000)
    query.add_argument('--top-k', type=int, default=100)
    query.add_argument('--threads', type=int, default=4)
    query.add_argument('--checkpoint-chunks', type=int, default=10)
    query.add_argument('--max-chunks', type=int, default=0, help='Stop at a checkpoint; for planned partial work or tests')
    score = commands.add_parser('score')
    score.add_argument('--cache', type=Path, required=True)
    score.add_argument('--retrieval', type=Path, required=True)
    score.add_argument('--model', type=Path, required=True)
    score.add_argument('--output', type=Path, required=True)
    score.add_argument('--score-batch', type=int, default=250)
    score.add_argument('--threads', type=int, default=4)
    merge = commands.add_parser('assemble')
    merge.add_argument('--cache', type=Path, required=True)
    merge.add_argument('--scored', type=Path, required=True)
    merge.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ('fit_rows', 'chunk_rows', 'reference_block', 'query_batch', 'top_k', 'threads', 'checkpoint_chunks', 'score_batch'):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    {'cache': cache_build, 'retrieve': retrieve, 'score': score_country, 'assemble': assemble}[args.command](args)


if __name__ == '__main__':
    main()
