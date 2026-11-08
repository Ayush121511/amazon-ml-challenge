"""Read what a new France submission changed: random France links added / removed vs a base, with the S1's accepted
siblings for context, plus accepted links whose house number differs from the S1's (the shifted-decoy family).
Env LATE (new run's late_scores.npz), NEW, BASE (matching_results.tsv), OUT, N (per group, default 40).
"""
import os
import random
from collections import defaultdict

import numpy as np

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
from fr_labels import read_france
from france_census import house, load


def main():
    Z = np.load(os.environ['LATE'], allow_pickle=True)
    ri, tg, p3 = Z['ref_idx'], Z['target'].astype(str), Z['prob']
    new, base = load(os.environ['NEW']), load(os.environ['BASE'])
    refs, text, m, _ = read_france(ri, tg)
    pj = {(refs[ri[j]], tg[j]): j for j in np.flatnonzero(m)}
    fr = {refs[ri[j]] for j in np.flatnonzero(m)}
    add = [(r, t) for r in fr for t in set(new.get(r, [])) - set(base.get(r, []))]
    rem = [(r, t) for r in fr for t in set(base.get(r, [])) - set(new.get(r, []))]
    shifted = []
    for r in fr:
        h1 = house(text[r][1])[0]
        for t in new.get(r, []):
            a = text.get(t, ('', ''))[1]
            h2 = house(a)[0] if a.strip() else ''
            if h1 and h2 and h1 != h2:
                shifted.append((r, t))
    n = int(os.environ.get('N', 40))
    random.seed(11)
    with open(os.environ['OUT'], 'w', encoding='utf-8') as f:
        for g, pairs in (('ADDED', add), ('REMOVED', rem), ('ACCEPTED_NUMBER_DIFFERS', shifted)):
            f.write(f'#### {g}: {len(pairs)}\n')
            for r, t in random.sample(pairs, min(n, len(pairs))):
                j = pj.get((r, t))
                f.write(f"\n{text[r][0]} | {text[r][1]}\n  {'+' if g != 'REMOVED' else '-'} {p3[j] if j is not None else -1:.2f} "
                        f"{text.get(t, ('?', ''))[0]} | {text.get(t, ('', ''))[1]}\n")
                for s in new.get(r, []):
                    if s != t:
                        f.write(f"    = {text.get(s, ('?', ''))[0]} | {text.get(s, ('', ''))[1]}\n")
            f.write('\n')


if __name__ == '__main__':
    main()
