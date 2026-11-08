"""Audit missed gold links in a completed sparse retrieval benchmark."""
import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path


def load_gold(path, wanted):
    gold = {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            ref_id = row['source1_entity_id']
            if ref_id in wanted:
                gold[ref_id] = set(filter(None, row['matched_entity_ids'].split(',')))
    if set(gold) != wanted:
        raise ValueError(f'Missing labels for {len(wanted - set(gold))} references')
    return gold


def audit(references, candidates, gold):
    refs = {row['entity_id']: row for row in references}
    if set(refs) != set(candidates) or set(refs) != set(gold):
        raise ValueError('Reference, candidate, and label ID sets differ')
    missed = []
    counts = Counter()
    for ref_id, result in candidates.items():
        ranked = set(result['ranked'][:50])
        routes = {name: set(ids) for name, ids in result['channels'].items()}
        union = set().union(*routes.values())
        for target_id in gold[ref_id]:
            if target_id in ranked:
                counts['found_top50'] += 1
                continue
            kind = 'lost_at_50' if target_id in union else 'absent_from_routes'
            counts[kind] += 1
            counts[f'{refs[ref_id]["country"]}:{kind}'] += 1
            missed.append({'reference_id': ref_id, 'target_id': target_id,
                           'country': refs[ref_id]['country'], 'kind': kind,
                           'routes': sorted(name for name, ids in routes.items()
                                            if target_id in ids),
                           'reference': refs[ref_id]})
    return missed, counts


def fetch_targets(paths, wanted):
    found = {}
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                identifier = row['entity_id']
                if identifier in wanted:
                    found[identifier] = row
        print(f'Scanned {path.name}; found {len(found)}/{len(wanted)} missed targets', flush=True)
    if set(found) != wanted:
        raise ValueError(f'Missing {len(wanted - set(found))} target records')
    return found


def token_jaccard(left, right):
    a, b = set(left.split()), set(right.split())
    return len(a & b) / len(a | b) if a or b else None


def enrich(missed, targets):
    for row in missed:
        ref, target = row['reference'], targets[row['target_id']]
        row['target'] = target
        row['name_token_jaccard'] = token_jaccard(ref['name_folded'], target['name_folded'])
        row['address_token_jaccard'] = token_jaccard(ref['address_folded'], target['address_folded'])
        row['same_country'] = ref['country'] == target['country']
        row['exact_folded_name'] = bool(ref['name_folded']) and ref['name_folded'] == target['name_folded']
        row['exact_folded_address'] = bool(ref['address_folded']) and ref['address_folded'] == target['address_folded']
        row['either_non_ascii'] = any(ord(char) > 127 for char in
            ref['business_name'] + target['business_name'])
    return missed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--gold', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new audit output directory')
    references = json.loads((args.benchmark/'references.json').read_text(encoding='utf-8'))
    with (args.benchmark/'candidates.jsonl').open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    candidates = {row['entity_id']: row for row in rows}
    if len(candidates) != len(rows):
        raise ValueError('Duplicate candidate reference ID')
    gold = load_gold(args.gold, set(candidates))
    missed, counts = audit(references, candidates, gold)
    targets = fetch_targets([args.data/f'train_source{i}.tsv.gz' for i in (2, 3)],
                            {row['target_id'] for row in missed})
    enrich(missed, targets)
    counts['missed_total'] = len(missed)
    counts['missed_wrong_country'] = sum(not row['same_country'] for row in missed)
    counts['missed_either_non_ascii'] = sum(row['either_non_ascii'] for row in missed)
    counts['missed_exact_name'] = sum(row['exact_folded_name'] for row in missed)
    counts['missed_exact_address'] = sum(row['exact_folded_address'] for row in missed)
    args.output.mkdir(parents=True)
    with (args.output/'misses.jsonl').open('w', encoding='utf-8') as handle:
        for row in sorted(missed, key=lambda item: (item['kind'], item['country'], item['reference_id'], item['target_id'])):
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    summary = {'benchmark': str(args.benchmark), 'references': len(references),
               'gold_links': sum(map(len, gold.values())), 'counts': dict(counts)}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
