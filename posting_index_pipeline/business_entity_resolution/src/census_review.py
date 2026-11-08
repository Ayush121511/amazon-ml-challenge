"""Manual-review sheet for the France census fixes: sampled restored / rejected pairs with each S1's full candidate list.

Strata (the census logic says where the risk is):
  restore, S1 has records at other house numbers (its decoy cluster is accounted for)      -> expected safe
  restore, S1 has none elsewhere (a same-address record may be the decoy)                  -> risky
  reject, S1 keeps other matches                                                           -> expected safe
  reject, rejection leaves the S1 with no match                                            -> risky
Marks: * accepted by the base, + restored, - rejected; house number, p, name, address.
Env BASE_DIR (base submission folder), FIX_DIR (census output folder). Writes FIX_DIR/census_review.md.
"""
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

HOME = Path.home()
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/test'
BASE, FIX = Path(os.environ['BASE_DIR']), Path(os.environ['FIX_DIR'])


def load(f):
    d = {}
    with open(f, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            s, m = (line.rstrip('\n').split('\t') + [''])[:2]
            d[s] = set(x for x in m.split(',') if x)
    return d


def num(a):
    m = re.search(r'(?<![\d/])0*(\d+)', unicodedata.normalize('NFKD', a or '').lower())
    return m.group(1) if m else ''


def main():
    rd = next(d for d in BASE.iterdir() if (d / 'matching_results.tsv').exists())
    base = load(rd / 'matching_results.tsv')
    fix = load(FIX / rd.name / 'matching_results.tsv')
    Z = np.load(BASE / 'late_scores.npz', allow_pickle=True)
    ri, tg, p = Z['ref_idx'], Z['target'].astype(str), Z['prob']
    refs, cty, text = [], {}, {}
    with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, iname, iaddr, ic = (head.index(c) for c in ('entity_id', 'business_name', 'business_address', 'country'))
        for line in fh:
            q = line.rstrip('\n').split('\t')
            refs.append(q[ie]); cty[q[ie]] = q[ic]
            if q[ic] == 'France':
                text[q[ie]] = (q[iname], q[iaddr])
    fr = [i for i, r in enumerate(refs) if cty[r] == 'France']
    frs = set(fr)
    cand = defaultdict(list)
    for j in np.flatnonzero(np.isin(ri, fr)):
        cand[refs[ri[j]]].append((tg[j], float(p[j])))
    need = {t for r in cand for t, _ in cand[r]}
    for k in (2, 3):
        with open(RAW / f'test_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if q[ie] in need:
                    text[q[ie]] = (q[iname], q[iaddr] if len(q) > iaddr else '')
    strata = defaultdict(list)
    for r in (refs[i] for i in fr):
        added, removed = fix[r] - base[r], base[r] - fix[r]
        n1 = num(text[r][1])
        elsewhere = sum(1 for t, _ in cand[r] if num(text.get(t, ('', ''))[1]) not in ('', n1))
        for t in added:
            strata['restore | S1 has records at other numbers' if elsewhere else 'restore | S1 has NO records elsewhere'].append((r, t))
        for t in removed:
            strata['reject | S1 keeps other matches' if fix[r] else 'reject | leaves S1 EMPTY'].append((r, t))
    rng = np.random.default_rng(4)
    out = ['# France census fixes: manual review sheet\n',
           'Marks: `*` accepted by base, `+` restored by census, `-` rejected by census. num = first house number.\n']
    for k in sorted(strata):
        v = strata[k]
        out.append(f'\n## {k}: {len(v)} pairs (showing 12)\n')
        for i in rng.choice(len(v), min(12, len(v)), replace=False):
            r, t = v[i]
            rn, ra = text[r]
            out.append(f'\n**{r}** {rn} | {ra}  (num {num(ra)})  -> pair {t}')
            for c, pv in sorted(cand[r], key=lambda x: -x[1]):
                mk = '+' if c in fix[r] - base[r] else '-' if c in base[r] - fix[r] else '*' if c in base[r] else ' '
                cn, ca = text.get(c, ('', ''))
                out.append(f'- `{mk}` p={pv:.2f} num={num(ca) or "-":>5s} {c[:2]} | {cn} | {ca}')
    (FIX / 'census_review.md').write_text('\n'.join(out))
    print('\n'.join(f'{k}: {len(v)}' for k, v in sorted(strata.items())))


if __name__ == '__main__':
    main()
