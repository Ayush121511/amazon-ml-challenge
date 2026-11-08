"""Read saved candidates, score partial progress, and time bounded search probes."""
import argparse
import json
import random
import sqlite3
import time
from pathlib import Path

from retrieval_experiment import evaluate, queries


def read_saved(path, refs):
    results = []
    with path.open('rb') as f:
        for line in f:
            if not line.endswith(b'\n'):
                break  # Interrupted final write; leave the source file untouched.
            result = json.loads(line)
            i = len(results)
            if i >= len(refs) or result['entity_id'] != refs[i]['entity_id']:
                raise ValueError('Saved candidates disagree with reference order')
            result.setdefault('non_ascii_name', result.get('non_latin_name', False))
            results.append(result)
    if not results:
        raise ValueError('No complete saved candidate records')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--ground-truth', type=Path, required=True)
    p.add_argument('--query-timeout', type=float, default=5)
    args = p.parse_args()
    refs = json.loads((args.run/'references.json').read_text(encoding='utf-8'))
    results = read_saved(args.run/'candidates.jsonl', refs)
    report = evaluate(results, args.ground_truth)
    report['status'] = 'partial_diagnostic_not_final_validation'
    report['completed'] = len(results)
    report['planned'] = len(refs)
    report['caveat'] = 'Completed prefix is ordered by entity ID, not a fresh random sample; selection may be biased.'
    (args.run/'partial_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PARTIAL RECALL REPORT', flush=True)
    print(json.dumps(report, indent=2), flush=True)
    # Read-only probes reuse the index. No rebuild, modification, or GPU model.
    db = sqlite3.connect((args.run/'targets.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
    db.execute('PRAGMA cache_size=-131072')
    rng = random.Random(2026)
    probes = []
    for country in sorted({r['country'] for r in refs}):
        available = [r for r in refs if r['country'] == country]
        probes.extend(rng.sample(available, min(3, len(available))))
    timings = []
    for row in probes:
        for channel, table, query, enabled in queries(row):
            if not enabled:
                continue
            sql = f'''SELECT e.entity_id FROM {table} JOIN entities e ON e.id={table}.rowid
                WHERE {table} MATCH ? AND e.country=? ORDER BY {table}.rank LIMIT ?'''
            parameters = (query, row['country'], 50)
            plan = db.execute('EXPLAIN QUERY PLAN '+sql, parameters).fetchall()
            start = time.monotonic()
            db.set_progress_handler(lambda: int(time.monotonic()-start > args.query_timeout), 1000)
            try:
                hits = db.execute(sql, parameters).fetchall()
                status, count = 'complete', len(hits)
            except sqlite3.OperationalError as exc:
                if 'interrupted' not in str(exc):
                    raise
                status, count = 'timeout', None
            finally:
                db.set_progress_handler(None, 0)
            item = {'id': row['entity_id'], 'country': row['country'], 'channel': channel,
                    'seconds': time.monotonic()-start, 'status': status, 'hits': count, 'plan': plan}
            timings.append(item)
            print(json.dumps(item), flush=True)
            (args.run/'query_profile.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
    db.close()
    print('Diagnostics complete. Original index and candidates unchanged.', flush=True)


if __name__ == '__main__':
    main()
