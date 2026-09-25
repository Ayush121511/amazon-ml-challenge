"""Pilot: entity-grouped train/dev/holdout, fixed LightGBM, dev-only threshold."""
import argparse
import gzip
import json
from pathlib import Path
from metrics import macro_f05


def features(pair):
    from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio
    a, b = pair['reference'], pair['target']
    values = []
    for field in ['name_norm', 'name_folded', 'address_norm', 'address_folded']:
        x, y = a[field], b[field]
        valid = bool(x and y)
        values.extend([ratio(x,y)/100 if valid else 0,
                       token_sort_ratio(x,y)/100 if valid else 0,
                       token_set_ratio(x,y)/100 if valid else 0,
                       int(x == y and valid), min(len(x),len(y))/max(1,len(x),len(y))])
    for field in ['address_numbers', 'postal_candidates']:
        x, y = set(a[field].split()), set(b[field].split())
        values.extend([len(x & y)/max(1,len(x | y)), int(bool(x and y) and not x & y),
                       int(not x), int(not y)])
    values.extend([int(a['address_missing']), int(b['address_missing'])])
    for channel in ['name_words', 'address_words', 'name_trigrams']:
        rank = pair['retrieval_ranks'][channel]
        values.append(0 if rank is None else 1/rank)
    return values


def main():
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit
    from lightgbm import LGBMClassifier
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; do not overwrite a pilot')
    truth_rows = json.loads((args.data/'reference_truth.json').read_text())
    truth = {r['source1_entity_id']: set(r['gold_ids']) for r in truth_rows}
    parent = {sid: sid for sid in truth}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        parent[find(a)] = find(b)
    owners = {}
    for sid, gold in truth.items():
        for tid in gold:
            if tid in owners:
                union(sid, owners[tid])
            owners[tid] = sid
    ids, targets, labels, rows = [], [], [], []
    # Connect references sharing candidate targets too: no target leaks across splits.
    candidate_owners = {}
    with gzip.open(args.data/'pairs.jsonl.gz', 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            pair = json.loads(line)
            sid, tid = pair['source1_entity_id'], pair['target_entity_id']
            if tid in candidate_owners:
                union(sid, candidate_owners[tid])
            candidate_owners[tid] = sid
            if tid in owners:
                union(sid, owners[tid])
            if pair['label'] != int(tid in truth[sid]):
                raise ValueError('Pair label disagrees with truth')
            ids.append(sid); targets.append(tid); labels.append(pair['label'])
            rows.append(features(pair))
            if i % 20000 == 0:
                print(f'Features: {i:,} pairs', flush=True)
    anchors = sorted(truth)
    groups = [find(sid) for sid in anchors]
    if len(set(groups)) < 10:
        raise ValueError('Too few independent components; need more data, not a leaky split')
    pool, holdout = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=2026).split(anchors,groups=groups))
    train_rel, dev_rel = next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=2027).split(
        pool, groups=[groups[i] for i in pool]))
    partitions = {'train': {anchors[pool[i]] for i in train_rel},
                  'dev': {anchors[pool[i]] for i in dev_rel},
                  'holdout': {anchors[i] for i in holdout}}
    component_sets = {k: {find(sid) for sid in members} for k,members in partitions.items()}
    assert not (component_sets['train'] & component_sets['dev'])
    assert not (component_sets['train'] & component_sets['holdout'])
    assert not (component_sets['dev'] & component_sets['holdout'])
    for part, members in partitions.items():
        if len(members) < 30:
            raise ValueError(f'{part} has too few references: {len(members)}')
    x, y = np.asarray(rows,dtype=np.float32), np.asarray(labels)
    masks = {k: np.asarray([sid in members for sid in ids]) for k,members in partitions.items()}
    if len(set(y[masks['train']])) != 2:
        raise ValueError('Training needs both classes')
    model = LGBMClassifier(n_estimators=300, num_leaves=15, max_depth=-1,
        learning_rate=.05, min_child_samples=50, reg_lambda=5,
        n_jobs=4, random_state=2026, verbosity=-1)
    model.fit(x[masks['train']], y[masks['train']])
    def predictions(part):
        idx = np.flatnonzero(masks[part])
        return idx, model.predict_proba(x[idx])[:,1]
    def score(part, idx, probabilities, threshold):
        pred = {sid: set() for sid in partitions[part]}
        for j, probability in zip(idx,probabilities):
            if probability >= threshold:
                pred[ids[j]].add(targets[j])
        return macro_f05({sid: truth[sid] for sid in pred}, pred)
    idx, probabilities = predictions('dev')
    grid = [(float(t), score('dev',idx,probabilities,t)) for t in np.linspace(.05,.95,91)]
    threshold, dev_score = max(grid, key=lambda r: (r[1],r[0]))
    idx, probabilities = predictions('holdout')
    holdout_score = score('holdout',idx,probabilities,threshold)
    args.output.mkdir(parents=True)
    model.booster_.save_model(str(args.output/'model.txt'))
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'scikit-learn', 'lightgbm', 'rapidfuzz']}
    report = {'macro_f05_dev': dev_score, 'macro_f05_holdout': holdout_score,
       'threshold_selected_on_dev': threshold, 'references': {k:len(v) for k,v in partitions.items()},
       'pairs': {k:int(v.sum()) for k,v in masks.items()}, 'components': len(set(groups)),
       'feature_count': x.shape[1], 'versions': versions,
       'warning': 'Pilot on retrieval-selected 850-reference prefix; not a leaderboard estimate. Model frozen before holdout. Unretrieved gold and singletons included.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    (args.output/'split.json').write_text(json.dumps({k:sorted(v) for k,v in partitions.items()},indent=2))
    (args.output/'threshold_grid.json').write_text(json.dumps(grid))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
