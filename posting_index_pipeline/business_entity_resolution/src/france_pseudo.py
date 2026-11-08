"""France self-training set from T1's test predictions: only pairs that every decider is very sure about.

France has no labels; the rules allow self-training on the provided test records. To limit confirmation bias, a pair
enters only when four independent signals agree at extreme values, and uncertain pairs (the ones we want to fix)
are never used:
  positive  stage-3b p >= 0.995, v5 >= 0.9, chunk2 logit >= 4, listwise logit >= 4, and the pair is in T1's final
            matches (it won global exclusivity)
  negative  stage-3b p <= 0.001, v5 <= 0.02, chunk2 logit <= -4, listwise logit <= -4
  lists     France references whose every candidate is a very-confident positive or negative (for the listwise model)
Only the cross-encoders are fine-tuned on this; fusion and stage-3b stay trained on train labels only.
Env T1_DIR. Writes artifacts/fr_pseudo/{pairs.npz, lists.npz, stats.json}.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HOME = Path.home()
T1 = Path(os.environ['T1_DIR'])
RAW = Path(os.environ.get('TEST_RAW', str(HOME / 'scratch/amazon_ml_2026/data/dataset/test')))
OUT = Path(os.environ.get('OUT_DIR', str(HOME / 'scratch/amazon_ml_2026/artifacts/fr_pseudo')))
NO_LW = os.environ.get('NO_LW', '') == '1'  # base without a listwise column (e.g. T1b): use the three other deciders


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Z = np.load(T1 / 'late_scores.npz', allow_pickle=True)
    ri, tg = Z['ref_idx'], Z['target'].astype(str)
    p3, v5, c2 = Z['prob'].astype(np.float64), Z['stage1'].astype(np.float64), Z['ce'].astype(np.float64)
    lw = None if NO_LW else np.load(T1 / 'features.npy', mmap_mode='r')[:, -1].astype(np.float64)
    refs, cty = [], []
    with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, ic = head.index('entity_id'), head.index('country')
        for line in fh:
            q = line.rstrip('\n').split('\t')
            refs.append(q[ie]); cty.append(q[ic] if len(q) > ic else '')
    refs, cty = np.array(refs, dtype=object), np.array(cty, dtype=object)
    sub = next(d for d in T1.iterdir() if (d / 'matching_results.tsv').exists())
    final = set()
    with open(sub / 'matching_results.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            s, m = line.rstrip('\n').split('\t') if '\t' in line else (line.strip(), '')
            final.update((s, t) for t in m.split(',') if t)
    fr = cty[ri] == 'France'
    rid = refs[ri]
    in_final = np.array([(r, t) in final for r, t in zip(rid, tg)]) & fr
    pos = fr & in_final & (p3 >= 0.995) & (v5 >= 0.9) & (c2 >= 4)
    neg = fr & (p3 <= 0.001) & (v5 <= 0.02) & (c2 <= -4)
    if lw is not None:
        pos &= lw >= 4
        neg &= lw <= -4
    sure = pos | neg
    by_ref = defaultdict(list)
    for j in np.flatnonzero(fr):
        by_ref[ri[j]].append(j)
    full = [r for r, js in by_ref.items() if all(sure[j] for j in js) and any(pos[j] for j in js)]
    lj = np.array([j for r in full for j in by_ref[r]], dtype=np.int64)
    pj = np.flatnonzero(sure)
    np.savez(OUT / 'pairs.npz', ref=rid[pj].astype(str), target=tg[pj], label=pos[pj].astype(np.int8))
    np.savez(OUT / 'lists.npz', ref=rid[lj].astype(str), target=tg[lj], label=pos[lj].astype(np.int8))
    R = {'france_refs': int((cty == 'France').sum()), 'france_pairs': int(fr.sum()),
         'france_pairs_in_T1_matches': int(in_final.sum()), 'pseudo_pos': int(pos.sum()), 'pseudo_neg': int(neg.sum()),
         'pseudo_share_of_france_pairs': round(float(sure.sum() / fr.sum()), 4),
         'pos_share_of_T1_france_matches': round(float(pos.sum() / max(in_final.sum(), 1)), 4),
         'fully_confident_lists': len(full), 'list_rows': int(len(lj)),
         'excluded_uncertain_pairs': int((fr & ~sure).sum())}
    (OUT / 'stats.json').write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1), flush=True)


if __name__ == '__main__':
    main()
