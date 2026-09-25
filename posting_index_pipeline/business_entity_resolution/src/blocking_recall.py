"""Forward candidate recall for several route combinations of one index, on fixed references.

Candidates are generated before labels are read; labels only score recall. Used to decide
whether new routes (phonetic, combined) are worth a full retrain.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from posting_index import CountryIndex, load_manifest, records
from submission_pipeline import candidates, read_gold

CONFIGS = {'v2_routes': ('name', 'address', 'anchor'),
           'plus_phonetic': ('name', 'address', 'anchor', 'phonetic'),
           'plus_combined': ('name', 'address', 'anchor', 'combined'),
           'all_routes': ('name', 'address', 'anchor', 'phonetic', 'combined')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--gold', type=Path, required=True)
    p.add_argument('--reference-ids', type=Path, required=True)
    p.add_argument('--sample', type=int, default=20000)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--top-k', type=int, default=200)
    p.add_argument('--top-terms', type=int, default=64)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    started = time.monotonic()
    ids = sorted(a.reference_ids.read_text().split())
    ids = set(random.Random('blocking-recall').sample(ids, min(a.sample, len(ids))))
    refs = [r for r in records(a.data / 'train_source1.tsv.gz') if r['entity_id'] in ids]
    manifest = load_manifest(a.index)
    found = {c: defaultdict(list) for c in CONFIGS}  # config -> ref -> [(fused_rank, target)]
    timing = defaultdict(float)
    by_country = defaultdict(list)
    for r in refs:
        by_country[r['country']].append(r)
    for country, rows in sorted(by_country.items()):
        index = CountryIndex(a.index, country, manifest)
        for start in range(0, len(rows), a.batch):
            batch = rows[start:start + a.batch]
            for config, routes in CONFIGS.items():
                t = time.monotonic()
                c = candidates(index, batch, a.top_k, a.top_terms, 200, a.threads, routes=routes)
                timing[config] += time.monotonic() - t
                ids_ = index.ids[c['ord']]
                for i, rank, tid in zip(c['ref'], c['fused_rank'], ids_):
                    found[config][batch[i]['entity_id']].append((int(rank), tid.decode()))
            print(f'{country}: {start + len(batch):,}/{len(rows):,}', flush=True)
        del index
    gold = read_gold(a.gold, {r['entity_id'] for r in refs})  # Labels read only now.
    country_of = {r['entity_id']: r['country'] for r in refs}
    report = {'references': len(refs), 'gold_links': sum(map(len, gold.values())), 'configs': {},
              'seconds': time.monotonic() - started, 'timing': dict(timing)}
    for config in CONFIGS:
        out = {}
        for group in ['all'] + sorted(by_country):
            subset = [s for s in gold if group == 'all' or country_of[s] == group]
            total = sum(len(gold[s]) for s in subset)
            row = {}
            for k in (50, 100, 200):
                hit = sum(len({t for rank, t in found[config][s] if rank <= k} & gold[s]) for s in subset)
                row[f'recall_at_{k}'] = hit / total if total else None
            out[group] = row
        report['configs'][config] = out
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
