"""Disk-backed lexical retrieval diagnostic; ground truth is evaluation-only."""
import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import sqlite3
import time
from collections import defaultdict
from pathlib import Path


def records(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        yield from csv.DictReader(f, delimiter='\t')


def sample_references(path, per_country=1000, seed=2026):
    pools, counts, rngs = {}, defaultdict(int), {}
    for row in records(path):
        country = row['country']
        if country not in pools:
            pools[country] = []
            rngs[country] = random.Random(f'{seed}:{country}')
        counts[country] += 1
        pool = pools[country]
        if len(pool) < per_country:
            pool.append(row)
        else:
            position = rngs[country].randrange(counts[country])
            if position < per_country:
                pool[position] = row
    return sorted((r for p in pools.values() for r in p), key=lambda r: r['entity_id'])


def connect(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA cache_size=-131072')  # 128 MiB page cache.
    db.execute('PRAGMA temp_store=FILE')
    return db


def build_index(paths, destination):
    signature = json.dumps([(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                            for p in paths])
    db = connect(destination)
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
    previous = db.execute("SELECT value FROM metadata WHERE key='inputs'").fetchone()
    if previous and previous[0] != signature:
        raise ValueError('Index inputs changed; choose a new output directory')
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('inputs', ?)", (signature,))
    db.execute('CREATE TABLE IF NOT EXISTS entities (id INTEGER PRIMARY KEY, entity_id TEXT UNIQUE, country TEXT)')
    # Both tables are on disk. Trigrams recover spelling/spacing variants.
    db.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS words USING fts5(name, address,
        tokenize="unicode61 remove_diacritics 0 categories 'L* N* M*'")''')
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chars USING fts5(name, tokenize='trigram')")
    db.commit()
    for path in paths:
        key = 'done:' + path.name
        if db.execute('SELECT 1 FROM metadata WHERE key=?', (key,)).fetchone():
            print('Index already complete:', path.name, flush=True)
            continue
        # A source is atomic: interrupted source inserts roll back; previous source stays.
        with db:
            count = 0
            for count, row in enumerate(records(path), 1):
                cursor = db.execute('INSERT INTO entities(entity_id,country) VALUES (?,?)',
                                    (row['entity_id'], row['country']))
                rid = cursor.lastrowid
                db.execute('INSERT INTO words(rowid,name,address) VALUES (?,?,?)',
                           (rid, row['name_folded'], row['address_folded']))
                db.execute('INSERT INTO chars(rowid,name) VALUES (?,?)', (rid, row['name_folded']))
                if count % 250000 == 0:
                    print(f'Index {path.name}: {count:,}', flush=True)
            db.execute('INSERT INTO metadata VALUES (?,?)', (key, str(count)))
        print('Indexed', path.name, count, flush=True)
    return db


def quote(term):
    return '"' + term.replace('"', '""') + '"'


def queries(row):
    name = row['name_folded']
    # Long tokens first, bounded query length. This is heuristic, not learned IDF.
    names = sorted(set(name.split()), key=lambda x: (-len(x), x))[:12]
    address = sorted(set(row['address_folded'].split()),
                     key=lambda x: (-len(x), x))[:16]
    grams = sorted(set(name[i:i+3] for i in range(max(0, len(name)-2))
                       if ' ' not in name[i:i+3]))
    # Spread the bounded selection over the complete name.
    if len(grams) > 16:
        grams = [grams[i * (len(grams)-1)//15] for i in range(16)]
    return [('name_words', 'words', 'name : (' + ' OR '.join(map(quote, names)) + ')', bool(names)),
            ('address_words', 'words', 'address : (' + ' OR '.join(map(quote, address)) + ')', bool(address)),
            ('name_trigrams', 'chars', ' OR '.join(map(quote, grams)), bool(grams))]


def retrieve(db, row, budget=50):
    fusion, channels = defaultdict(float), {}
    for label, table, query, enabled in queries(row):
        if not enabled:
            channels[label] = []
            continue
        # Hard country blocking is an explicit experiment assumption, evaluated below.
        hits = db.execute(f'''SELECT e.entity_id FROM {table} JOIN entities e ON e.id={table}.rowid
            WHERE {table} MATCH ? AND e.country=? ORDER BY {table}.rank LIMIT ?''',
                          (query, row['country'], budget)).fetchall()
        ids = [r[0] for r in hits]
        channels[label] = ids
        for rank, identifier in enumerate(ids, 1):
            fusion[identifier] += 1.0 / (60 + rank)
    ranked = sorted(fusion, key=lambda x: (-fusion[x], x))[:budget]
    return {'entity_id': row['entity_id'], 'country': row['country'],
            'non_ascii_name': any(ord(c) > 127 for c in row['name_norm']),
            'ranked': ranked, 'channels': channels}


def evaluate(results, ground_truth):
    wanted = {r['entity_id'] for r in results}
    gold = {}
    with ground_truth.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['source1_entity_id'] in wanted:
                gold[row['source1_entity_id']] = set(filter(None, row['matched_entity_ids'].split(',')))
    if set(gold) != wanted:
        raise ValueError('Missing ground truth for sampled references')
    report = {}
    for group in ['all'] + sorted({r['country'] for r in results}) + ['non_ascii_name']:
        subset = [r for r in results if group == 'all' or r['country'] == group or
                  (group == 'non_ascii_name' and r['non_ascii_name'])]
        positive = [r for r in subset if gold[r['entity_id']]]
        total = sum(len(gold[r['entity_id']]) for r in positive)
        metrics = {'references': len(subset), 'positive_references': len(positive),
                   'singletons': len(subset)-len(positive), 'gold_links': total}
        for k in [10, 20, 50]:
            recovered = [len(set(r['ranked'][:k]) & gold[r['entity_id']]) for r in positive]
            metrics[f'recall_at_{k}'] = sum(recovered)/total if total else None
            metrics[f'macro_recall_at_{k}'] = (sum(n/len(gold[r['entity_id']])
                for n, r in zip(recovered, positive))/len(positive)) if positive else None
        for channel in ['name_words', 'address_words', 'name_trigrams']:
            metrics[channel + '_recall_at_50'] = (sum(len(set(r['channels'][channel]) & gold[r['entity_id']])
                for r in positive)/total) if total else None
        metrics['union_recall'] = (sum(len(set().union(*map(set, r['channels'].values())) & gold[r['entity_id']])
            for r in positive)/total) if total else None
        report[group] = metrics
    return report


def load_checkpoint(path, refs):
    """Validate a saved ordered prefix; discard only an interrupted final line."""
    if not path.exists():
        return []
    results = []
    end = 0
    with path.open('rb') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                # Preserve interrupted bytes for inspection before repairing.
                path.with_suffix('.interrupted_tail').write_bytes(line)
                break
            result = json.loads(line)
            i = len(results)
            if i >= len(refs) or result['entity_id'] != refs[i]['entity_id'] or result['country'] != refs[i]['country']:
                raise ValueError('Candidate checkpoint does not match sampled references')
            if not isinstance(result['ranked'], list) or set(result['channels']) != {'name_words', 'address_words', 'name_trigrams'}:
                raise ValueError('Invalid checkpoint record')
            if 'non_ascii_name' not in result:
                result['non_ascii_name'] = result.pop('non_latin_name', False)
            results.append(result)
            end = f.tell()
    with path.open('r+b') as f:
        f.truncate(end)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preprocessed', type=Path, required=True)
    p.add_argument('--ground-truth', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-country', type=int, default=1000)
    p.add_argument('--smoke', action='store_true', help='Allow truncated inputs; NOT a valid full-pool recall estimate')
    args = p.parse_args()
    if args.per_country < 1:
        p.error('per-country must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    paths = [args.preprocessed / f'train_source{i}.tsv.gz' for i in [1, 2, 3]]
    for path in paths:
        manifest = json.loads(Path(str(path)+'.json').read_text())
        if manifest['input']['version'] != 2 or (manifest['input']['limit'] != 0 and not args.smoke):
            raise ValueError('Full version-2 inputs required')
        if path.stat().st_size != manifest['output_bytes']:
            raise ValueError('Output size disagrees with manifest')
    refs = sample_references(paths[0], args.per_country)
    reference_path = args.output/'references.json'
    if reference_path.exists() and json.loads(reference_path.read_text(encoding='utf-8')) != refs:
        raise ValueError('Saved reference sample differs; choose a new output directory')
    reference_path.write_text(json.dumps(refs, ensure_ascii=False), encoding='utf-8')
    db = build_index(paths[1:], args.output/'targets.sqlite')
    indexed = time.monotonic()
    candidate_path = args.output/'candidates.jsonl'
    results = load_checkpoint(candidate_path, refs)
    resumed = len(results)
    print(f'Resuming from {resumed}/{len(refs)} saved references', flush=True)
    with candidate_path.open('a', encoding='utf-8') as f:
        for i, row in enumerate(refs[resumed:], resumed + 1):
            result = retrieve(db, row)
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
            f.flush()
            if i % 25 == 0:
                os.fsync(f.fileno())
                print(f'Retrieved {i}/{len(refs)}', flush=True)
    # Labels are first opened here, after all candidates have been saved.
    report = evaluate(results, args.ground_truth)
    report['smoke_only'] = args.smoke
    report['resumed_references'] = resumed
    report['timing'] = {'prepare_and_index_seconds': indexed-start,
                        'retrieve_and_evaluate_seconds': time.monotonic()-indexed}
    report['index_bytes'] = (args.output/'targets.sqlite').stat().st_size
    try:
        import resource
        report['peak_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        pass
    report['sample_ids_sha256'] = hashlib.sha256('\n'.join(r['entity_id'] for r in refs).encode()).hexdigest()
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    db.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
