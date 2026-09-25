"""Measure per-reference oracle F0.5 from saved candidates, without retrieval.

Oracle predictions keep only known true candidates. This is an upper bound,
not an achievable model score or a prediction of leaderboard performance.
Ground truth is used only for evaluation; candidates are never modified.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from metrics import entity_f05


def candidate_sets(row):
    return {
        'top50': set(row['ranked'][:50]),
        'route_union': set().union(*(set(ids) for ids in row['channels'].values())),
    }


def audit(candidate_path, ground_truth):
    selected = set()
    with candidate_path.open(encoding='utf-8') as stream:
        for line in stream:
            sid = json.loads(line)['entity_id']
            if sid in selected:
                raise ValueError(f'Duplicate reference: {sid}')
            selected.add(sid)
    if not selected:
        raise ValueError('No references to audit')
    truth = {}
    with ground_truth.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            sid = row['source1_entity_id']
            if sid in selected:
                if sid in truth:
                    raise ValueError(f'Duplicate ground truth: {sid}')
                truth[sid] = set(filter(None, row['matched_entity_ids'].split(',')))
    if set(truth) != selected:
        raise ValueError(f'Missing labels for {len(selected - set(truth))} references')
    totals = defaultdict(lambda: defaultdict(float))
    with candidate_path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            gold = truth[row['entity_id']]
            sets = candidate_sets(row)
            for group in ('all', 'country:' + row['country']):
                item = totals[group]
                item['references'] += 1
                item['gold_links'] += len(gold)
                item['singletons'] += not gold
                for name, candidates in sets.items():
                    recovered = gold & candidates
                    item[name + '_oracle_f05_sum'] += entity_f05(gold, recovered)
                    item[name + '_recovered_links'] += len(recovered)
                    item[name + '_candidate_count'] += len(candidates)
                    item[name + '_positive_refs_with_zero_recovery'] += bool(gold) and not recovered
                    item[name + '_positive_refs_with_all_links'] += bool(gold) and recovered == gold
                    item[name + '_empty_candidate_refs'] += not candidates
    report = {}
    for group, item in totals.items():
        result = {key: int(value) for key, value in item.items()
                  if not key.endswith('_oracle_f05_sum')}
        for name in ('top50', 'route_union'):
            result[name + '_oracle_macro_f05'] = item[name + '_oracle_f05_sum'] / item['references']
            result[name + '_link_recall'] = (item[name + '_recovered_links'] / item['gold_links']
                                            if item['gold_links'] else None)
            result[name + '_mean_candidates'] = item[name + '_candidate_count'] / item['references']
        report[group] = result
    return {'source': str(candidate_path), 'groups': report,
            'warning': 'Perfect classifier upper bound on these references only. Singletons assume perfect abstention. Not a validation model score or leaderboard estimate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gold', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new audit output')
    report = audit(args.run / 'candidates.jsonl', args.gold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
