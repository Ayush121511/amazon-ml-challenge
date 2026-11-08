"""France case file for manual reading: businesses where the decision is shakiest, one block per S1.

Groups (30 S1 each, fixed seed): U uncertain (a candidate with 0.15 < p < 0.85), L low-link (<= 1 accepted but >= 3
candidates at the S1's house number), B blank copy with 0.3 < p < 0.85, E emptied (no match), R random.
Per candidate: accepted mark, stage-3b p, v5, chunk2 logit, house-number relation (= same, ~ other, 0 blank/none),
census/extreme label source, name | address. Candidates shown: accepted, p >= 0.01, or at the S1's number.
Env LATE (late_scores.npz), SUB (matching_results.tsv), LABELS (fr_labels.npz), OUT (text file).
"""
import os
from collections import defaultdict

import numpy as np

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
from fr_labels import read_france
from france_census import house, load

TAG = {0: '', 1: 'cP', 2: 'cN', 3: 'xP', 4: 'xN'}


def main():
    Z = np.load(os.environ['LATE'], allow_pickle=True)
    ri, tg = Z['ref_idx'], Z['target'].astype(str)
    p3, v5, c2 = Z['prob'], Z['stage1'], np.nan_to_num(Z['ce'])
    match = load(os.environ['SUB'])
    L = np.load(os.environ['LABELS'], allow_pickle=True)
    src = np.zeros(len(tg), np.int8); src[L['idx']] = L['src']
    refs, text, m, _ = read_france(ri, tg)
    by = defaultdict(list)
    for j in np.flatnonzero(m):
        by[ri[j]].append(j)
    rel = {}
    for i, js in by.items():
        h1 = house(text[refs[i]][1])[0]
        for j in js:
            ta = text.get(tg[j], ('', ''))[1]
            h2 = house(ta)[0] if ta.strip() else ''
            rel[j] = '=' if h1 and h2 == h1 else ('~' if h1 and h2 else '0')
    groups = {'U': [], 'L': [], 'B': [], 'E': [], 'R': list(by)}
    for i, js in by.items():
        acc = set(match.get(refs[i], []))
        if any(0.15 < p3[j] < 0.85 for j in js):
            groups['U'].append(i)
        if len(acc) <= 1 and sum(rel[j] == '=' for j in js) >= 3:
            groups['L'].append(i)
        if any(0.3 < p3[j] < 0.85 and not text.get(tg[j], ('', ''))[1].strip() for j in js):
            groups['B'].append(i)
        if not acc:
            groups['E'].append(i)
    rng = np.random.default_rng(7)
    with open(os.environ['OUT'], 'w', encoding='utf-8') as f:
        for g, ids in groups.items():
            f.write(f'#### group {g}: {len(ids)} S1 in France\n')
            for i in rng.choice(ids, min(30, len(ids)), replace=False):
                r = refs[i]; acc = set(match.get(r, []))
                f.write(f'\n{r} | {text[r][0]} | {text[r][1]}\n')
                for j in sorted(by[i], key=lambda j: -p3[j]):
                    t = tg[j]
                    if not (t in acc or p3[j] >= 0.01 or rel[j] == '='):
                        continue
                    n, a = text.get(t, ('', ''))
                    f.write(f"  {'A' if t in acc else '.'} {p3[j]:.2f} v{v5[j]:.2f} c{c2[j]:+.0f} {rel[j]} {TAG[src[j]]:2s} {n} | {a}\n")
            f.write('\n')


if __name__ == '__main__':
    main()
