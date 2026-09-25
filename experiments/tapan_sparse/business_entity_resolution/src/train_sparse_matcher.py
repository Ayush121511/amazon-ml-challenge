"""Train a sparse-candidate matcher with entity splits and purged training targets."""
import argparse
import gzip
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from lightgbm import LGBMClassifier

from metrics import macro_f05
from pair_features import feature_spec


def split_references(truth):
    parent = {sid: sid for sid in truth}

    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    owners = {}
    for sid, targets in truth.items():
        for tid in targets:
            if tid in owners:
                parent[find(sid)] = find(owners[tid])
            owners[tid] = sid
    anchors = sorted(truth)
    groups = [find(sid) for sid in anchors]
    pool, holdout = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=2026).split(anchors, groups=groups))
    train, dev = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=2027).split(
        pool, groups=[groups[i] for i in pool]))
    partitions = {'train': {anchors[pool[i]] for i in train},
                  'dev': {anchors[pool[i]] for i in dev},
                  'holdout': {anchors[i] for i in holdout}}
    return partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-version', choices=('v1', 'v2'), default='v1')
    parser.add_argument('--split-from', type=Path)
    parser.add_argument('--dev-only', action='store_true', help='Do not predict or score holdout while developing a new model')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new model output directory')
    summary = json.loads((args.data / 'summary.json').read_text())
    truth_rows = json.loads((args.data / 'reference_truth.json').read_text())
    truth = {row['source1_entity_id']: set(row['gold_ids']) for row in truth_rows}
    partitions = (split_references(truth) if args.split_from is None else
                  {name: set(members) for name, members in json.loads(args.split_from.read_text()).items()})
    if set(partitions) != {'train', 'dev', 'holdout'} or set.union(*partitions.values()) != set(truth):
        raise ValueError('Split does not cover the reference truth')
    if sum(map(len, partitions.values())) != len(truth):
        raise ValueError('Reference splits overlap')
    entity_parts = {}
    for name, members in partitions.items():
        for sid in members:
            for tid in truth[sid]:
                if tid in entity_parts and entity_parts[tid] != name:
                    raise ValueError('Gold entity crosses partitions')
                entity_parts[tid] = name
    if min(map(len, partitions.values())) < 100:
        raise ValueError('Need at least 100 references per partition')
    print('Reference splits:', {k: len(v) for k, v in partitions.items()}, flush=True)
    part_by_sid = {sid: number for number, name in enumerate(('train', 'dev', 'holdout'))
                   for sid in partitions[name]}
    n = summary['pairs']
    extract_features, feature_names = feature_spec(args.feature_version)
    x = np.empty((n, len(feature_names)), dtype=np.float32)
    y = np.empty(n, dtype=np.uint8)
    part = np.empty(n, dtype=np.uint8)
    ids, targets = [], []
    countries = {row['source1_entity_id']: row['country'] for row in truth_rows if 'country' in row}
    validation_targets = set()
    dev_targets, holdout_targets = set(), set()
    with gzip.open(args.data / 'pairs.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for i, line in enumerate(stream):
            if i >= n:
                raise ValueError('More pairs than summary declares')
            pair = json.loads(line)
            sid, tid = pair['source1_entity_id'], pair['target_entity_id']
            if pair['label'] != int(tid in truth[sid]):
                raise ValueError('Label disagrees with reference truth')
            if not args.dev_only or part_by_sid[sid] != 2:
                x[i] = extract_features(pair)
            y[i] = pair['label']
            part[i] = part_by_sid[sid]
            ids.append(sid)
            targets.append(tid)
            countries[sid] = pair['reference']['country']
            if part[i] != 0:
                validation_targets.add(tid)
                (dev_targets if part[i] == 1 else holdout_targets).add(tid)
            if (i + 1) % 100000 == 0:
                print(f'Features: {i+1:,}/{n:,}', flush=True)
    if len(ids) != n:
        raise ValueError('Fewer pairs than summary declares')
    # Keep evaluation candidate sets intact. Purge shared candidate targets only
    # from training; sharing a false candidate is not an identity relationship.
    train_mask = (part == 0) & np.fromiter((tid not in validation_targets for tid in targets), bool, count=n)
    if y[train_mask].sum() < 1000 or len(set(y[train_mask])) != 2:
        raise ValueError('Too few training positives after target purging')
    assert not ({tid for tid, keep in zip(targets, train_mask) if keep} & validation_targets)
    model = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=.05,
                          min_child_samples=100, reg_lambda=5, n_jobs=4,
                          random_state=2026, verbosity=-1)
    print(f'Training: {train_mask.sum():,} pairs; {y[train_mask].sum():,} positives', flush=True)
    model.fit(x[train_mask], y[train_mask])

    def score(name, index, probabilities, threshold):
        predictions = {sid: set() for sid in partitions[name]}
        for row, probability in zip(index, probabilities):
            if probability >= threshold:
                predictions[ids[row]].add(targets[row])
        return macro_f05({sid: truth[sid] for sid in predictions}, predictions), predictions

    dev_index = np.flatnonzero(part == 1)
    dev_prob = model.predict_proba(x[dev_index])[:, 1]
    grid = [(float(t), score('dev', dev_index, dev_prob, float(t))[0])
            for t in np.linspace(.05, .99, 95)]
    threshold, dev_score = max(grid, key=lambda item: (item[1], item[0]))
    holdout_score = None
    if args.dev_only:
        _, predictions = score('dev', dev_index, dev_prob, threshold)
    else:
        holdout_index = np.flatnonzero(part == 2)
        holdout_prob = model.predict_proba(x[holdout_index])[:, 1]
        holdout_score, predictions = score('holdout', holdout_index, holdout_prob, threshold)
    by_country = {}
    for country in sorted(set(countries.values())):
        members = [sid for sid in predictions if countries.get(sid) == country]
        if members:
            by_country[country] = {'references': len(members),
                                  'macro_f05': macro_f05({sid: truth[sid] for sid in members},
                                                         {sid: predictions[sid] for sid in members})}
    report = {'macro_f05_dev': dev_score, 'macro_f05_holdout': holdout_score,
              'threshold_selected_on_dev': threshold,
              ('dev_by_country' if args.dev_only else 'holdout_by_country'): by_country,
              'references': {k: len(v) for k, v in partitions.items()},
              'pairs': {k: int((part == number).sum()) for number, k in enumerate(('train', 'dev', 'holdout'))},
              'training_pairs_after_purge': int(train_mask.sum()),
              'training_positives_after_purge': int(y[train_mask].sum()),
              'training_positives_before_purge': int(y[part == 0].sum()),
              'dev_holdout_shared_candidate_targets': len(dev_targets & holdout_targets),
              'feature_count': len(feature_names), 'feature_version': args.feature_version,
              'feature_names': feature_names, 'dev_only': args.dev_only,
              'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scikit-learn', 'lightgbm', 'rapidfuzz')},
              'validation': 'Gold-connected reference entities are disjoint. All dev/holdout candidate targets are purged from training pairs. Evaluation candidates remain intact. Dev/holdout can share negative candidate targets, but not gold entities. Threshold chosen on dev only. Includes singletons and unretrieved gold. No France validation.'}
    args.output.mkdir(parents=True)
    model.booster_.save_model(str(args.output / 'model.txt'))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output / 'threshold_grid.json').write_text(json.dumps(grid), encoding='utf-8')
    (args.output / 'split.json').write_text(json.dumps({k: sorted(v) for k, v in partitions.items()}), encoding='utf-8')
    prediction_name = 'dev_predictions.jsonl' if args.dev_only else 'holdout_predictions.jsonl'
    with (args.output / prediction_name).open('w', encoding='utf-8') as stream:
        for sid, predicted in predictions.items():
            stream.write(json.dumps({'entity_id': sid, 'gold_ids': sorted(truth[sid]),
                                     'predicted_ids': sorted(predicted)}) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
