"""Where does macro F0.5 go? Splits holdout loss into blocking (gold never a candidate)
and matcher (wrong decisions among candidates), for a trained model."""
import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from submission_pipeline import decide, f05


def main():
    import lightgbm as lgb
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--partition', default='holdout')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    names = json.loads((a.pairs / 'report.json').read_text())['features']
    model_report = json.loads((a.model / 'report.json').read_text())
    x, y = np.load(a.pairs / 'x.npy'), np.load(a.pairs / 'y.npy')
    pairs = [line.rstrip('\n').split('\t') for line in gzip.open(a.pairs / 'pairs.tsv.gz', 'rt')]
    pair_ref = np.array([q[0] for q in pairs], dtype=object)
    pair_target = np.array([q[1] for q in pairs], dtype=object)
    gold = {k: set(v) for k, v in json.loads((a.pairs / 'gold.json').read_text()).items()}
    owners = Counter(t for v in gold.values() for t in v)
    if max(owners.values(), default=1) > 1:
        raise ValueError('Gold targets shared between references: component split needed')

    def part(s):  # Same assignment as stage_train when every component is a single reference.
        h = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 100
        return 'train' if h < 70 else 'dev' if h < 85 else 'holdout'
    refs = sorted(s for s in gold if part(s) == a.partition)
    position = {s: i for i, s in enumerate(refs)}
    keep = model_report['keep']
    rank = x[:, names.index('fused_rank')]
    idx = np.flatnonzero(np.array([part(s) == a.partition for s in pair_ref]) & (rank <= keep))
    booster = lgb.Booster(model_file=str(a.model / 'model.txt'))
    columns = [names.index(n) for n in model_report['features']]
    prob = booster.predict(x[idx][:, columns])
    ref_idx = np.array([position[s] for s in pair_ref[idx]])
    chosen = decide(ref_idx, prob, model_report['decision']['rule'], model_report['decision']['threshold'], len(refs))
    pred, cand = defaultdict(set), defaultdict(set)
    for i, t, c in zip(ref_idx, pair_target[idx], chosen):
        cand[i].add(t)
        if c:
            pred[i].add(t)
    per_ref = []
    for i, s in enumerate(refs):
        g, pr, cs = gold[s], pred.get(i, set()), cand.get(i, set())
        actual, oracle = f05(g, pr), f05(g, g & cs)
        fp, fn_block, fn_match = len(pr - g), len(g - cs), len((g & cs) - pr)
        kind = ('perfect' if actual == 1 else
                'singleton_false_match' if not g else
                'blocking_only' if fn_block and not fp and not fn_match else
                'false_positive_only' if fp and not fn_block and not fn_match else
                'missed_candidate_only' if fn_match and not fp and not fn_block else 'mixed')
        per_ref.append((s, actual, oracle, kind, len(g), fp, fn_block, fn_match))
    total = len(refs)
    summary = {'references': total, 'keep': keep, 'decision': model_report['decision'],
               'macro_f05': sum(r[1] for r in per_ref) / total,
               'oracle_matcher_macro_f05': sum(r[2] for r in per_ref) / total,
               'loss_by_kind': {}, 'false_positive_links': sum(r[5] for r in per_ref),
               'blocking_missed_links': sum(r[6] for r in per_ref),
               'matcher_missed_links': sum(r[7] for r in per_ref),
               'gold_links': sum(r[4] for r in per_ref),
               'loss_by_gold_count': {}}
    by_kind, by_count = defaultdict(float), defaultdict(float)
    for s, actual, oracle, kind, n, *_ in per_ref:
        by_kind[kind] += (1 - actual) / total
        by_count[min(n, 8)] += (1 - actual) / total
    summary['loss_by_kind'] = dict(sorted(by_kind.items(), key=lambda kv: -kv[1]))
    summary['loss_by_gold_count'] = {str(k): v for k, v in sorted(by_count.items())}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'loss_summary.json').write_text(json.dumps(summary, indent=2))
    worst = sorted(per_ref, key=lambda r: r[1])[:300]
    (a.output / 'worst_refs.json').write_text(json.dumps(
        [{'ref': s, 'f05': f, 'kind': k, 'gold': sorted(gold[s]), 'pred': sorted(pred.get(position[s], set())),
          'fp': fp, 'fn_block': fb, 'fn_match': fm} for s, f, o, k, n, fp, fb, fm in worst], indent=1))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
