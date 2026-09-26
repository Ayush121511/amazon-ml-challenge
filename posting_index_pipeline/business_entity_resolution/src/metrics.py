"""Challenge macro F0.5, including the official singleton convention."""
from collections.abc import Mapping, Set


def entity_f05(truth: Set[str], predicted: Set[str]) -> float:
    if not truth:
        return 1.0 if not predicted else 0.0
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    return 5.0 * tp / (5 * tp + 4 * fp + fn)


def macro_f05(truth: Mapping[str, Set[str]], predicted: Mapping[str, Set[str]]) -> float:
    if truth.keys() != predicted.keys():
        missing = len(truth.keys() - predicted.keys())
        extra = len(predicted.keys() - truth.keys())
        raise ValueError(f"Reference ID mismatch: {missing} missing and {extra} extra predictions")
    if not truth:
        raise ValueError("Cannot score an empty evaluation set")
    return sum(entity_f05(gold, predicted[anchor]) for anchor, gold in truth.items()) / len(truth)
