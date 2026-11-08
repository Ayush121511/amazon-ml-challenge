"""Benchmark selective searches against the existing persistent FTS target index."""
import argparse
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from blocking_v2 import select
from retrieval_experiment import evaluate, quote, records


LEGAL = {'private', 'limited', 'pvt', 'ltd', 'llc', 'inc', 'corporation',
         'company', 'co', 'the', 'and', 'services', 'service', 'center'}
ADDRESS_GENERIC = {'road', 'street', 'avenue', 'floor', 'unit', 'building',
                   'house', 'shop', 'plot', 'sector', 'near', 'opp', 'block',
                   'state', 'india', 'usa', 'north', 'south', 'east', 'west'}


def query_plan(row):
    """Return bounded FTS5 MATCH expressions; no labels or target IDs used."""
    name = sorted({word for word in row['name_folded'].split()
                   if len(word) >= 3 and word not in LEGAL},
                  key=lambda word: (-len(word), word))
    numbers = sorted({word for word in row['address_numbers'].split()
                      if len(word) >= 2}, key=lambda word: (-len(word), word))
    address = sorted({word for word in row['address_folded'].split()
                      if len(word) >= 4 and word not in ADDRESS_GENERIC and
                      not word.isdecimal()}, key=lambda word: (-len(word), word))
    plans = []
    if name:
        # Conjunction avoids the enormous OR result sets of the first FTS run.
        plans.append(('name_words', 'name : (' + ' AND '.join(map(quote, name[:2])) + ')'))
    if numbers and address:
        # Requiring both a number and an address word is more selective than
        # searching every business with the same common house/floor number.
        plans.append(('address_words', 'address : ((' + ' OR '.join(map(quote, numbers[:3])) +
                      ') AND (' + ' OR '.join(map(quote, address[:12])) + '))'))
    return plans


def connect_readonly(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA cache_size=-131072')
    return db


def retrieve(db, row, limit=100, timeout_seconds=2.0):
    channels = {'name_words': [], 'address_words': [], 'name_trigrams': []}
    timings = []
    for label, expression in query_plan(row):
        start = time.monotonic()
        deadline = start + timeout_seconds
        db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 2000)
        try:
            found = db.execute('''SELECT e.entity_id FROM words
                JOIN entities e ON e.id=words.rowid
                WHERE words MATCH ? AND e.country=?
                ORDER BY words.rank LIMIT ?''',
                (expression, row['country'], limit)).fetchall()
            status = 'complete'
            channels[label] = [identifier for (identifier,) in found]
        except sqlite3.OperationalError as error:
            if 'interrupted' not in str(error).lower():
                raise
            status = 'timeout'
        finally:
            db.set_progress_handler(None, 0)
        timings.append({'route': label, 'seconds': time.monotonic()-start,
                        'status': status, 'hits': len(channels[label])})
    fusion = defaultdict(float)
    for ids in channels.values():
        for rank, identifier in enumerate(ids, 1):
            fusion[identifier] += 1/(60+rank)
    ranked = sorted(fusion, key=lambda identifier: (-fusion[identifier], identifier))[:50]
    result = {'entity_id': row['entity_id'], 'country': row['country'],
              'non_ascii_name': any(ord(char)>127 for char in row['name_norm']),
              'ranked': ranked, 'channels': channels}
    return result, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--old-sample', type=Path, required=True)
    parser.add_argument('--new-sample', type=Path, required=True)
    parser.add_argument('--gold', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-country', type=int, default=200)
    parser.add_argument('--timeout', type=float, default=2.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    if args.per_country < 1 or args.timeout <= 0:
        parser.error('per-country and timeout must be positive')
    excluded = {row['entity_id'] for row in json.loads(args.old_sample.read_text(encoding='utf-8'))}
    excluded.update(row['entity_id'] for row in json.loads(args.new_sample.read_text(encoding='utf-8')))
    refs = select(args.data/'train_source1.tsv.gz', excluded, args.per_country)
    if len(refs) != 2*args.per_country:
        raise ValueError(f'Expected {2*args.per_country} references, got {len(refs)}')
    db = connect_readonly(args.index)
    # The index is reused. No target scan or index build occurs here.
    count = db.execute('SELECT COUNT(*) FROM entities').fetchone()[0]
    if count != 10320219:
        raise ValueError(f'Expected full target index, got {count} records')
    args.output.mkdir(parents=True)
    (args.output/'references.json').write_text(json.dumps(refs, ensure_ascii=False), encoding='utf-8')
    results, timings = [], []
    start = time.monotonic()
    with (args.output/'candidates.jsonl').open('w', encoding='utf-8') as handle:
        for i, row in enumerate(refs, 1):
            result, one = retrieve(db, row, timeout_seconds=args.timeout)
            results.append(result)
            timings.extend(one)
            handle.write(json.dumps(result, ensure_ascii=False)+'\n')
            if i % 25 == 0:
                print(f'Retrieved {i}/{len(refs)}; seconds {time.monotonic()-start:.1f}', flush=True)
    db.close()
    queried = time.monotonic()
    report = evaluate(results, args.gold)  # Labels only after candidates are saved.
    report['benchmark'] = {'index': str(args.index), 'index_reused': True,
                           'target_rows': count, 'reference_rows': len(refs),
                           'retrieve_seconds': queried-start,
                           'references_per_second': len(refs)/(queried-start),
                           'route_queries': len(timings),
                           'timeouts': dict(Counter(t['route'] for t in timings
                                                    if t['status']=='timeout')),
                           'route_seconds': {route: sum(t['seconds'] for t in timings
                                          if t['route']==route)
                                          for route in ('name_words','address_words')},
                           'note': 'Fresh sample excludes both previous reference sets. Existing full target index reused. No label access until after retrieval.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
