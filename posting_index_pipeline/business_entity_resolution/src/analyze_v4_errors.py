"""Aggregate v4 dev/holdout errors from saved pairs and a trained model.

This reads existing artifacts and writes only aggregate JSON. It does not train,
change candidates, inspect test labels, or emit business IDs or dataset text.
Run on Padum compute nodes, not the login node.
"""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from submission_pipeline import decide, f05, load_scorer


PROB_BOUNDS = (0.0, 0.02, 0.2, 0.5, 0.75, 0.9, 0.99, 1.01)


def partition(ref_id):
    h = int(hashlib.sha256(ref_id.encode()).hexdigest()[:8], 16) % 100
    return 'train' if h < 70 else 'dev' if h < 85 else 'holdout'


def exclusive_prob(ref_idx, targets, prob):
    """Mirror the final test merge: highest probability owns a shared target."""
    if not len(prob):
        return prob.copy(), 0
    _, t_idx = np.unique(np.asarray(targets, dtype=str), return_inverse=True)
    order = np.lexsort((ref_idx, -prob, t_idx))
    first = np.ones(len(order), dtype=bool)
    first[1:] = t_idx[order][1:] != t_idx[order][:-1]
    loser = np.ones(len(prob), dtype=bool)
    loser[order[first]] = False
    return np.where(loser, 0.0, prob), int(loser.sum())


def score_and_errors(refs, ref_idx, targets, labels, prob, gold, rule, threshold):
    """Compute end-to-end F0.5 and probability bins for selected pair errors."""
    chosen = decide(ref_idx, prob, rule, threshold, len(refs))
    pred = defaultdict(set)
    for i, t in zip(ref_idx[chosen], targets[chosen]):
        pred[int(i)].add(t)
    score = float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(refs)]))
    missed = labels.astype(bool) & ~chosen
    false = ~labels.astype(bool) & chosen

    def bins(mask):
        counts = np.histogram(prob[mask], bins=PROB_BOUNDS)[0]
        return {f'{lo:g}-{hi:g}': int(n) for lo, hi, n in zip(PROB_BOUNDS[:-1], PROB_BOUNDS[1:], counts)}

    gold_links = sum(len(gold[s]) for s in refs)
    candidate_true_links = int(labels.sum())
    return {'macro_f05': score, 'gold_links': gold_links,
            'candidate_true_links': candidate_true_links,
            'blocking_missed_links': gold_links - candidate_true_links,
            'predicted_links': int(chosen.sum()),
            'missed_true_links': int(missed.sum()), 'false_positive_links': int(false.sum()),
            'all_pair_probability_bins': bins(np.ones(len(prob), dtype=bool)),
            'missed_true_probability_bins': bins(missed),
            'false_positive_probability_bins': bins(false)}


def analyze(pairs_dir, model_dir, output, which, batch, threads):
    pair_report = json.loads((pairs_dir / 'report.json').read_text())
    model_report = json.loads((model_dir / 'report.json').read_text())
    names = pair_report['features']
    columns = [names.index(name) for name in model_report['features']]
    x = np.load(pairs_dir / 'x.npy', mmap_mode='r')
    y = np.load(pairs_dir / 'y.npy', mmap_mode='r')
    gold = {k: set(v) for k, v in json.loads((pairs_dir / 'gold.json').read_text()).items()}
    countries = json.loads((pairs_dir / 'countries.json').read_text())
    refs = sorted(s for s in gold if partition(s) == which)
    position = {s: i for i, s in enumerate(refs)}
    rank_col = names.index('fused_rank')
    keep = model_report['keep']
    generated_keep = int(pair_report['parameters']['keep'])
    filter_rank = bool(keep and keep < generated_keep)

    selected_rows, ref_idx, targets = [], [], []
    rows_seen = 0
    with gzip.open(pairs_dir / 'pairs.tsv.gz', 'rt', encoding='utf-8') as f:
        for row, line in enumerate(f):
            rows_seen += 1
            s, t = line.rstrip('\n').split('\t')
            if s in position and (not filter_rank or x[row, rank_col] <= keep):
                selected_rows.append(row)
                ref_idx.append(position[s])
                targets.append(t)
    if rows_seen != len(x):
        raise ValueError('pairs.tsv.gz and x.npy have different row counts')
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    ref_idx = np.asarray(ref_idx, dtype=np.int32)
    targets = np.asarray(targets, dtype=object)
    labels = np.asarray(y[selected_rows], dtype=bool)
    scorer = load_scorer(model_dir, threads)
    prob = np.empty(len(selected_rows), dtype=np.float32)
    for lo in range(0, len(prob), batch):
        hi = min(lo + batch, len(prob))
        prob[lo:hi] = scorer.predict(np.asarray(x[selected_rows[lo:hi]][:, columns]))
    rule = model_report['decision']['rule']
    threshold = model_report['decision']['threshold']
    original = score_and_errors(refs, ref_idx, targets, labels, prob, gold, rule, threshold)
    exclusive, removed = exclusive_prob(ref_idx, targets, prob)
    zeroed = (prob > 0) & (exclusive == 0)
    after = score_and_errors(refs, ref_idx, targets, labels, exclusive, gold, rule, threshold)
    by_country = {}
    ref_country = np.array([countries[s] for s in refs], dtype=object)
    pair_country = ref_country[ref_idx]
    for country in sorted(set(ref_country)):
        local_refs = np.flatnonzero(ref_country == country)
        local_pairs = np.flatnonzero(pair_country == country)
        local_ref_idx = np.searchsorted(local_refs, ref_idx[local_pairs])
        country_ids = [refs[i] for i in local_refs]
        by_country[country] = score_and_errors(
            country_ids, local_ref_idx, targets[local_pairs], labels[local_pairs],
            exclusive[local_pairs], gold, rule, threshold)
        by_country[country]['references'] = len(country_ids)
    report = {'partition': which, 'references': len(refs), 'pairs': len(prob),
              'decision': model_report['decision'], 'before_exclusivity': original,
              'after_exclusivity': after, 'exclusivity_zeroed_pairs': removed,
              'exclusivity_zeroed_true_links': int((zeroed & labels).sum()),
              'exclusivity_zeroed_pairs_above_threshold': int((zeroed & (prob >= threshold)).sum()),
              'by_country_after_exclusivity': by_country,
              'caveat': 'Exclusivity is within this partition only; test runs it over all references.'}
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{which}_diagnostic.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--partition', choices=('dev', 'holdout'), default='dev')
    p.add_argument('--batch', type=int, default=200000)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    analyze(a.pairs, a.model, a.output, a.partition, a.batch, a.threads)


if __name__ == '__main__':
    main()
