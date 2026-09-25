"""Decompose development-set F0.5 loss without retraining or inspecting holdout."""
import argparse
import gzip
import heapq
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from lightgbm import Booster

from metrics import entity_f05
from pair_features import feature_spec, baseline_names


def loss_components(gold, candidates, predicted):
    if not predicted <= candidates:
        raise ValueError('Predictions outside candidate set')
    actual = entity_f05(gold, predicted)
    precision_fixed = entity_f05(gold, predicted & gold)
    oracle = entity_f05(gold, candidates & gold)
    return {'macro_f05': actual, 'oracle_macro_f05': oracle,
            'blocking_loss': 1 - oracle,
            'rejected_true_candidate_loss': oracle - precision_fixed,
            'false_positive_loss': precision_fixed - actual}


def feature_names():
    return baseline_names()


def audit(args):
    if args.output.exists():
        raise FileExistsError('Use a new diagnostic output directory')
    report = json.loads((args.model / 'report.json').read_text())
    split = json.loads((args.model / 'split.json').read_text())
    selected = set(split['dev'])
    truth_rows = json.loads((args.data / 'reference_truth.json').read_text())
    states = {row['source1_entity_id']: {
        'gold': set(row['gold_ids']), 'candidates': set(), 'predicted': set(),
        'country': row.get('country', 'unknown'), 'fp_examples': [], 'fn_examples': []}
        for row in truth_rows if row['source1_entity_id'] in selected}
    if set(states) != selected:
        raise ValueError('Development labels missing')
    threshold = report['threshold_selected_on_dev']
    model = Booster(model_file=str(args.model / 'model.txt'))
    extract_features, names = feature_spec(report.get('feature_version', 'v1'))
    if model.num_feature() != len(names):
        raise ValueError('Feature version differs from model')
    batch = []
    counted = 0

    def process():
        nonlocal counted
        if not batch:
            return
        x = np.asarray([extract_features(pair) for pair in batch], dtype=np.float32)
        probabilities = model.predict(x, num_threads=4)
        for pair, probability in zip(batch, probabilities):
            sid, tid = pair['source1_entity_id'], pair['target_entity_id']
            state = states[sid]
            if tid in state['candidates']:
                raise ValueError('Duplicate pair in development data')
            state['candidates'].add(tid)
            state['country'] = pair['reference']['country']
            actual = tid in state['gold']
            if int(actual) != pair['label']:
                raise ValueError('Label disagrees with reference truth')
            predicted = probability >= threshold
            if predicted:
                state['predicted'].add(tid)
            if actual != predicted:
                example = {key: pair[key] for key in ('source1_entity_id', 'target_entity_id', 'reference', 'target', 'retrieval_ranks')}
                example.update(probability=float(probability), label=int(actual))
                kind = 'fn_examples' if actual else 'fp_examples'
                # Keep the three most confidently wrong examples per reference.
                priority = 1 - float(probability) if actual else float(probability)
                heapq.heappush(state[kind], (priority, tid, example))
                if len(state[kind]) > 3:
                    heapq.heappop(state[kind])
        counted += len(batch)
        batch.clear()
        if counted % 100000 == 0:
            print(f'Audited {counted:,} development pairs', flush=True)

    with gzip.open(args.data / 'pairs.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            pair = json.loads(line)
            if pair['source1_entity_id'] in selected:
                batch.append(pair)
                if len(batch) >= 5000:
                    process()
    process()
    groups = defaultdict(lambda: defaultdict(float))
    reference_errors = []
    for sid, state in states.items():
        gold, candidates, predicted = state['gold'], state['candidates'], state['predicted']
        loss = loss_components(gold, candidates, predicted)
        counts = {'references': 1, 'gold_links': len(gold), 'candidate_pairs': len(candidates),
                  'true_positives': len(gold & predicted), 'false_positives': len(predicted - gold),
                  'unretrieved_true_links': len(gold - candidates),
                  'rejected_true_candidates': len((gold & candidates) - predicted),
                  'references_with_false_positives': bool(predicted - gold),
                  'positive_references_predicted_empty': bool(gold) and not predicted}
        for group in ('all', 'country:' + state['country'], 'positive_references' if gold else 'singletons'):
            for name, value in {**loss, **counts}.items():
                groups[group][name] += value
        reference_errors.append({'entity_id': sid, 'country': state['country'], **loss,
                                 'gold_ids': sorted(gold), 'predicted_ids': sorted(predicted),
                                 'unretrieved_gold_ids': sorted(gold - candidates),
                                 'false_positive_examples': [item[2] for item in sorted(state['fp_examples'], reverse=True)],
                                 'false_negative_examples': [item[2] for item in sorted(state['fn_examples'], reverse=True)]})
    for group in groups.values():
        for key in ('macro_f05', 'oracle_macro_f05', 'blocking_loss', 'rejected_true_candidate_loss', 'false_positive_loss'):
            group[key] /= group['references']
    if abs(groups['all']['macro_f05'] - report['macro_f05_dev']) > 1e-8:
        raise ValueError('Recomputed dev score does not match saved report; investigate before interpreting errors')
    gain = model.feature_importance(importance_type='gain')
    importance = sorted(zip(names, map(float, gain)), key=lambda item: item[1], reverse=True)
    output = {'partition': 'dev_only', 'threshold': threshold, 'groups': dict(groups),
              'feature_importance_gain': importance,
              'loss_order': 'Remove false positives first, then accept all retrieved true candidates. Components add to 1 - macro F0.5; attribution depends on this order.',
              'warning': 'Development diagnostic only. Holdout has not been rescored or used to choose changes.'}
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    worst = sorted(reference_errors, key=lambda row: (row['macro_f05'], row['entity_id']))[:args.examples]
    (args.output / 'worst_dev_references.json').write_text(json.dumps(worst, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(output, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--examples', type=int, default=100)
    audit(parser.parse_args())


if __name__ == '__main__':
    main()
