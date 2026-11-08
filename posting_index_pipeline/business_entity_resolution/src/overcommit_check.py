"""Exclusivity-consistent probabilities: a record belongs to at most one S1, so its probabilities over S1s sum to <= 1.

For every candidate record, S = sum of stage-3b p over the S1 lists it appears in. Records with S > 1 are
over-committed: the model is sure of two owners at once (typically blank same-name ties). Per country: records with
S > 1.2, accepted links on them, blank share, and how many accepted links would fall below the F0.5 break-even (0.77)
after p' = p / max(1, S). Env LATE, SUB, OUT.
"""
import json
import os
from collections import Counter

import numpy as np

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
from france_census import RAW, load


def main():
    Z = np.load(os.environ['LATE'], allow_pickle=True)
    ri, tg, p = Z['ref_idx'], Z['target'].astype(str), Z['prob'].astype(np.float64)
    refs, cty = [], []
    with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t'); ie, ic = head.index('entity_id'), head.index('country')
        for line in fh:
            q = line.rstrip('\n').split('\t'); refs.append(q[ie]); cty.append(q[ic])
    blank = set()
    for k in (2, 3):
        with open(RAW / f'test_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t'); ie, ia = head.index('entity_id'), head.index('business_address')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) <= ia or not q[ia].strip():
                    blank.add(q[ie])
    uniq, inv = np.unique(tg, return_inverse=True)
    S = np.bincount(inv, weights=p)[inv]
    match = load(os.environ['SUB'])
    acc = np.array([t in set(match.get(refs[i], ())) for i, t in zip(ri, tg)])
    pc = np.array(cty, dtype=object)[ri]
    isb = np.array([t in blank for t in tg])
    pn = p / np.maximum(1.0, S)
    R = {}
    for c in ('US', 'India', 'France'):
        mc = pc == c
        oc = mc & (S > 1.2)
        R[c] = {'pairs': int(mc.sum()), 'accepted': int((mc & acc).sum()),
                'overcommitted_pairs_S>1.2': int(oc.sum()), 'accepted_on_overcommitted': int((oc & acc).sum()),
                'accepted_overcommitted_blank': int((oc & acc & isb).sum()),
                'accepted_drop_below_0.77_after_norm': int((mc & acc & (p >= 0.77) & (pn < 0.77)).sum()),
                'of_which_blank': int((mc & acc & (p >= 0.77) & (pn < 0.77) & isb).sum()),
                'S_of_accepted_q50_90_99': np.quantile(S[mc & acc], [0.5, 0.9, 0.99]).round(3).tolist()}
    out = json.dumps(R, indent=1)
    open(os.environ['OUT'], 'w').write(out)
    print(out, flush=True)


if __name__ == '__main__':
    main()
