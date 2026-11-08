"""France pairs by (name relation, house-number/street relation), accepted vs rejected in a submission.

Train says a record with the S1's core name (legal forms ignored) on the S1's street at a different house number is
95-96% a true match (alias_check.py), so rejected France pairs of that kind are restore candidates. Prints counts,
accepted share and p quantiles per cell, plus 30 rejected same-name / other-number / same-street examples with the
number pair. Env LATE, SUB, OUT.
"""
import os
import random
from collections import Counter, defaultdict

import numpy as np

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
from fr_labels import read_france
from france_census import core, house, load, street_words, tokens_addr

SHIFT = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def main():
    Z = np.load(os.environ['LATE'], allow_pickle=True)
    ri, tg, p3 = Z['ref_idx'], Z['target'].astype(str), Z['prob']
    match = load(os.environ['SUB'])
    owner = {t: r for r, ts in match.items() for t in ts}
    refs, text, m, s1_core = read_france(ri, tg)
    cell, acc, ps = Counter(), Counter(), defaultdict(list)
    cand = []
    acc_diff = defaultdict(Counter)
    shifted = []
    for j in np.flatnonzero(m):
        r, t = refs[ri[j]], tg[j]
        ra = text[r][1]; tn, ta = text.get(t, ('', ''))
        if not ta.strip():
            continue
        h1, h2 = house(ra)[0], house(ta)[0]
        sw = street_words(ra)
        if not h1 or not h2 or not sw:
            continue
        st = ('same_no' if h1 == h2 else 'diff_no') + ('_street' if sw <= tokens_addr(ta) else '_other')
        rc, tc = set(s1_core[r]), set(core(tn))
        nm = 'same' if rc == tc else ('none' if not rc & tc else 'edit')
        k = (nm, st); a = t in match.get(r, ())
        cell[k] += 1; acc[k] += a; ps[k].append(p3[j])
        if k == ('same', 'diff_no_street') and not a:
            cand.append((j, r, t, h1, h2))
        if nm in ('same', 'edit', 'none') and st == 'diff_no_street' and a:
            try:
                d = int(''.join(ch for ch in h2 if ch.isdigit())) - int(''.join(ch for ch in h1 if ch.isdigit()))
            except ValueError:
                d = None
            acc_diff[nm][d if d is None or abs(d) < 30 else ('big+' if d > 0 else 'big-')] += 1
            if nm == 'same' and d in SHIFT:
                own_same = sum(1 for x in match.get(r, ()) if house(text.get(x, ('', ''))[1])[0] == h1)
                shifted.append((j, r, t, h1, h2, own_same))
    lines = []
    for k in sorted(cell):
        q = np.quantile(ps[k], [0.1, 0.5, 0.9]).round(3).tolist()
        lines.append(f'{k} pairs {cell[k]} accepted {acc[k] / cell[k]:.3f} p_q10/50/90 {q}')
    free = [c for c in cand if c[2] not in owner]
    lines.append(f'rejected same-name other-number same-street: {len(cand)}, of which record unowned: {len(free)}')
    diffs = Counter()
    for _, _, _, h1, h2 in free:
        try:
            diffs[min(abs(int(''.join(ch for ch in h2 if ch.isdigit())) - int(''.join(ch for ch in h1 if ch.isdigit()))), 99)] += 1
        except ValueError:
            diffs['suffix'] += 1
    lines.append('abs number difference (unowned): ' + str(sorted(diffs.items(), key=lambda kv: -kv[1])[:20]))
    random.seed(2)
    for j, r, t, h1, h2 in random.sample(free, min(30, len(free))):
        lines.append(f'  {p3[j]:.2f} [{h1} -> {h2}] {text[r][0]} | {text[r][1]}  =>  {text[t][0]} | {text[t][1]}')
    for nm, c in acc_diff.items():
        lines.append(f'ACCEPTED {nm} diff_no_street signed diff: ' + str(sorted(c.items(), key=lambda kv: -kv[1])[:25]))
    lines.append(f'ACCEPTED same-name same-street shift-up in SHIFT: {len(shifted)}; S1 also has a link at its own number: '
                 f'{sum(1 for x in shifted if x[5] > 0)}; p q10/50/90 '
                 f'{np.quantile([p3[x[0]] for x in shifted], [0.1, 0.5, 0.9]).round(3).tolist() if shifted else []}')
    for j, r, t, h1, h2, o in random.sample(shifted, min(25, len(shifted))):
        lines.append(f'  S {p3[j]:.2f} own={o} [{h1} -> {h2}] {text[r][0]} | {text[r][1]}  =>  {text[t][0]} | {text[t][1]}')
    out = '\n'.join(lines)
    open(os.environ['OUT'], 'w', encoding='utf-8').write(out)
    print(out, flush=True)


if __name__ == '__main__':
    main()
