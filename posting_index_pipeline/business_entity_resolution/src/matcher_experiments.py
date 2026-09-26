"""Small, testable helpers for dev-only matcher experiments."""
import numpy as np


def reference_weights(ref_ids, power):
    """Mean-one weights. power=1 gives every reference equal total training weight."""
    if power not in (0.0, 0.5, 1.0):
        raise ValueError('weight power must be 0, 0.5 or 1')
    if not len(ref_ids):
        raise ValueError('cannot weight an empty training set')
    _, inverse, counts = np.unique(ref_ids, return_inverse=True, return_counts=True)
    weights = np.power(counts[inverse].astype(np.float64), -power)
    return (weights / weights.mean()).astype(np.float32)


def decision_grid(ref_names, ref_idx, targets, prob, gold, rules):
    """Same decisions as submission_pipeline.decide; group/sort once for all cutoffs.

    Assumes unique (reference,target) candidates, as emitted by the pipeline unions.
    Missing gold links and references with no candidates are included in the score.
    """
    if not ref_names:
        raise ValueError('evaluation partition is empty')
    if any(rule not in ('threshold', 'expected') for rule, _ in rules):
        raise ValueError('unknown decision rule')
    thresholds = np.array([t for _, t in rules], dtype=np.float64)
    expected = np.array([rule == 'expected' for rule, _ in rules])
    total = np.zeros(len(rules), dtype=np.float64)
    order = np.lexsort((-prob, ref_idx))
    bounds = np.searchsorted(ref_idx[order], np.arange(len(ref_names) + 1))
    for i, name in enumerate(ref_names):
        idx = order[bounds[i]:bounds[i + 1]]
        n_gold = len(gold[name])
        empty_score = float(n_gold == 0)
        if not len(idx):
            total += empty_score
            continue
        p = prob[idx].astype(np.float64)
        tp = np.concatenate(([0], np.cumsum([t in gold[name] for t in targets[idx]])))
        # NumPy casts the Python cutoff to prob.dtype in decide's threshold rule.
        # Preserve that boundary behavior for float32 probabilities (e.g. p == .7).
        cutoffs = thresholds.astype(prob.dtype).astype(np.float64)
        counts = np.searchsorted(-p, -cutoffs, side='right')
        values = np.divide(1.25 * tp[counts], counts + 0.25 * n_gold,
                           out=np.zeros(len(rules)), where=counts + 0.25 * n_gold > 0)
        values[counts == 0] = empty_score
        k = np.arange(1, len(p) + 1)
        approximate = 1.25 * np.cumsum(p) / (0.25 * p.sum() + k)
        best = int(np.argmax(approximate))
        empty_probability = float(np.prod(1 - np.clip(p, 0, 1 - 1e-9)))
        accept = (approximate[best] > empty_probability) & (p[0] >= thresholds)
        selected_score = 1.25 * tp[best + 1] / (best + 1 + 0.25 * n_gold)
        values[expected] = np.where(accept[expected], selected_score, empty_score)
        total += values
    return {rule: float(score / len(ref_names)) for rule, score in zip(rules, total)}
