"""Prepare labeled development pairs from saved retrieval; never inject missing gold."""
import argparse
import csv
import gzip
import json
import os
import time
from pathlib import Path
from diagnose_retrieval import read_saved
from retrieval_experiment import records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--ground-truth', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    start = time.monotonic()
    refs = json.loads((args.run/'references.json').read_text(encoding='utf-8'))
    saved = read_saved(args.run/'candidates.jsonl', refs)
    lookup = {r['entity_id']: r for r in refs}
    selected = {r['entity_id'] for r in saved}
    wanted = set()
    for result in saved:
        wanted.update(set().union(*map(set, result['channels'].values())))
    gold = {}
    owners = {}
    shared = 0
    with args.ground_truth.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            sid = row['source1_entity_id']
            if sid in selected:
                gold[sid] = set(filter(None, row['matched_entity_ids'].split(',')))
                for tid in gold[sid]:
                    if tid in owners and owners[tid] != sid:
                        shared += 1
                    owners[tid] = sid
    if set(gold) != selected:
        raise ValueError('Missing labels for selected references')
    targets = {}
    for source in [2, 3]:
        path = args.targets/f'train_source{source}.tsv.gz'
        for count, row in enumerate(records(path), 1):
            if row['entity_id'] in wanted:
                if row['entity_id'] in targets:
                    raise ValueError('Duplicate target ID')
                targets[row['entity_id']] = row
            if count % 1000000 == 0:
                print(f'Scanned source {source}: {count:,}; retained {len(targets):,}', flush=True)
    if set(targets) != wanted:
        raise ValueError(f'{len(wanted-set(targets))} candidate IDs missing from targets')
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output/'pairs.jsonl.gz'
    if destination.exists():
        raise FileExistsError('Pair output already exists; inspect it instead of overwriting')
    partial = args.output/'pairs.jsonl.gz.partial'
    total = positives = 0
    missing = []
    with gzip.open(partial, 'wt', encoding='utf-8', compresslevel=1) as f:
        for result in saved:
            sid = result['entity_id']
            union = set().union(*map(set, result['channels'].values()))
            ranks = {channel: {tid: i for i, tid in enumerate(ids, 1)}
                     for channel, ids in result['channels'].items()}
            final_ranks = {tid: i for i, tid in enumerate(result['ranked'], 1)}
            for tid in sorted(union):
                label = int(tid in gold[sid])
                item = {'source1_entity_id': sid, 'target_entity_id': tid,
                    'reference': lookup[sid], 'target': targets[tid], 'label': label,
                    'retrieval_ranks': {k: v.get(tid) for k, v in ranks.items()},
                    'fused_top50_rank': final_ranks.get(tid)}
                f.write(json.dumps(item, ensure_ascii=False)+'\n')
                total += 1
                positives += label
            missing.append({'source1_entity_id': sid, 'country': lookup[sid]['country'], 'gold_ids': sorted(gold[sid]),
                            'unretrieved_gold_ids': sorted(gold[sid]-union)})
    os.replace(partial, destination)
    (args.output/'reference_truth.json').write_text(json.dumps(missing, indent=2), encoding='utf-8')
    report = {'references': len(saved), 'unique_candidate_targets': len(targets),
              'pairs': total, 'positives': positives, 'negatives': total-positives,
              'gold_links': sum(map(len, gold.values())),
              'missed_gold_links': sum(len(r['unretrieved_gold_ids']) for r in missing),
              'shared_gold_target_occurrences_within_sample': shared,
              'seconds': time.monotonic()-start,
              'purpose': 'Development data only. No train/validation split or model trained. Keep reference/entity groups separated; do not randomly split pairs.'}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
