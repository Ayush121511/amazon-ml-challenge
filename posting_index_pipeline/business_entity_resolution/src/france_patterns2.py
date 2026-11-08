"""Size four France patterns found by reading case files, on every France pair (label-free counts + p quantiles).

  typo    census restore test passes except a street word, which has a close typo in the record (similarity >= 0.8)
  drop    only a business-type word dropped and only filler words added, everything else as the strict restore test
          (town + type names, where the rarest-word guard rejects the copy)
  init    record name = initials of the S1's core name, same house number and street words
  blank   blank-address record, p >= 0.8, not accepted: how many are owned by another S1 in the submission
Per pattern: pairs, not accepted now, p quantiles, 12 examples. Env LATE, SUB, OUT.
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
from rapidfuzz.fuzz import ratio

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
from fr_labels import read_france
from france_census import FILLER, TYPES, core, house, load, street_words, tokens_addr


def near(w, toks):
    return any(ratio(w, t) >= 80 for t in toks if abs(len(t) - len(w)) <= 2)


def main():
    Z = np.load(os.environ['LATE'], allow_pickle=True)
    ri, tg, p3 = Z['ref_idx'], Z['target'].astype(str), Z['prob'].astype(np.float64)
    match = load(os.environ['SUB'])
    owner = {t: r for r, ts in match.items() for t in ts}
    refs, text, m, s1_core = read_france(ri, tg)
    freq = Counter(t for c in s1_core.values() for t in set(c))
    hits = defaultdict(list)
    for j in np.flatnonzero(m):
        r, t = refs[ri[j]], tg[j]
        rn, ra = text[r]; tn, ta = text.get(t, ('', ''))
        acc = t in match.get(r, ())
        if not ta.strip():
            if p3[j] >= 0.8 and not acc:
                hits['blank_owned_elsewhere' if t in owner else 'blank_unowned'].append((j, r, t))
            continue
        (h1, s1n), (h2, s2n) = house(ra), house(ta)
        if not h1 or h1 != h2 or (s1n and s2n and s1n != s2n):
            continue
        rc, tc = s1_core[r], core(tn)
        sw, ta_t = street_words(ra), tokens_addr(ta)
        if not sw:
            continue
        exact_st = sw <= ta_t
        fuzzy_st = all(w in ta_t or near(w, ta_t) for w in sw)
        added, dropped = set(tc) - set(rc), set(rc) - set(tc)
        if exact_st and rc and tc and dropped and not added & TYPES:
            unmatched = {w for w in added if not any(ratio(w, d) >= 75 for d in dropped)}
            if all(any(ratio(d, w) >= 75 for w in added) for d in dropped) and unmatched <= FILLER:
                hits['nametypo'].append((j, r, t))
                continue
        if rc and tc and len(tc) == 1 and len(rc) >= 2 and tc[0] == ''.join(w[0] for w in rc) and exact_st:
            hits['init'].append((j, r, t))
            continue
        if not rc or not tc or not added <= FILLER:
            continue
        rare_ok = min(rc, key=lambda x: (freq[x], x)) in tc
        first_ok = rc[0] == tc[0]
        if first_ok and rare_ok and fuzzy_st and not exact_st:
            hits['typo'].append((j, r, t))
        elif exact_st and dropped and dropped <= TYPES and first_ok and not rare_ok:
            hits['drop'].append((j, r, t))
    R, ex = {}, {}
    rng = np.random.default_rng(3)
    for k, v in hits.items():
        js = np.array([x[0] for x in v])
        na = [x for x in v if x[2] not in match.get(x[1], ())]
        R[k] = {'pairs': len(v), 'not_accepted': len(na),
                'p_q10_50_90': [round(float(x), 3) for x in np.quantile(p3[js], [0.1, 0.5, 0.9])],
                'owned_by_other_s1': sum(x[2] in owner and owner[x[2]] != x[1] for x in na)}
        pick = [na[i] for i in rng.choice(len(na), min(12, len(na)), replace=False)] if na else []
        ex[k] = [f"{p3[j]:.2f} {text[r][0]} | {text[r][1]}  =>  {text[t][0]} | {text[t][1]}"
                 + (f"  [owner {text.get(owner[t], ('?',))[0]} | {text.get(owner[t], ('', '?'))[1]}]" if t in owner else '')
                 for j, r, t in pick]
    out = json.dumps({'stats': R, 'examples': ex}, indent=1, ensure_ascii=False)
    open(os.environ['OUT'], 'w', encoding='utf-8').write(out)
    print(out, flush=True)


if __name__ == '__main__':
    main()
