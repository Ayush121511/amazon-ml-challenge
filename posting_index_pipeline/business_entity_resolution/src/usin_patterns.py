"""Two generator patterns behind T6's non-blank US/India errors, measured on labelled train pairs (dev + holdout).

alias   the S1 has a candidate 'ALIAS aka/a.k.a./formerly/fka/dba <S1 name>'; another candidate whose core name is the
        alias part (the bare alias) is a sibling of the same alias. True rate, and how many T6 rejects.
digit   house number differs from the S1's: one-digit deletion/insertion (2118 -> 118, 27 -> 278), one-digit
        substitution, small upward shift (+1..+21, the decoy family), other. True rate per type and name relation,
        and T6 decisions per type.
Env OUT, PROB (T6 sample probabilities), RULE_T.
"""
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
sys.path.insert(0, str(Path(__file__).parent))
from distractor_check import decide, partition_of
from france_census import core, house, street_words, tokens_addr
from usin_errors import EN_FILLER

H = Path('/home/scai/mtech/aib262015/scratch')
V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
AKA = re.compile(r'\b(?:a\s*/\s*k\s*/\s*a|a\.?k\.?a\.?|aka|f\s*/\s*k\s*/\s*a|f\.?k\.?a\.?|fka|formerly(?:\s+known\s+as)?|'
                 r'd\s*/\s*b\s*/\s*a|d\.?b\.?a\.?|dba|doing business as|née|nee)\b[:\s]*', re.I)


def num_rel(a, b):
    a, b = re.sub(r'\D', '', a), re.sub(r'\D', '', b)
    if not a or not b or a == b:
        return 'same_or_none'
    if len(a) == len(b) + 1 and any(a[:i] + a[i + 1:] == b for i in range(len(a))):
        return 'digit_drop'
    if len(b) == len(a) + 1 and any(b[:i] + b[i + 1:] == a for i in range(len(b))):
        return 'digit_insert'
    if 1 <= int(b) - int(a) <= 21:
        return 'shift_up_1_21'
    if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1:
        return 'digit_sub'
    return 'other'


def main():
    S = np.load(H / 'amazon_ml_2026/artifacts/fusion/stage1.npz')
    y = np.load(H / 'vishal_v4_artifacts/pairs/y.npy')[S['rows']]
    gold = {k: set(v) for k, v in json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text()).items()}
    cp = partition_of(gold)
    mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
    refs, tgts = [], []
    with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t'); refs.append(r); tgts.append(t)
    refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
    need = set(refs) | set(tgts)
    text = {}
    for k in (1, 2, 3):
        with open(H / f'amazon_ml_2026/data/dataset/train/train_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if q[ie] in need:
                    text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
    P = np.load(os.environ.get('PROB', str(V / 'fusion/boundary_models_t6/stage3b_prob.npy')))
    thr = float(os.environ.get('RULE_T', '0.74'))
    part = np.array([cp[r] for r in refs], dtype=object)
    chosen = np.zeros(len(refs), bool)
    for p in ('dev', 'holdout'):
        ix = np.flatnonzero(part == p)
        prefs = sorted(s for s in gold if cp[s] == p)
        pos = {s: i for i, s in enumerate(prefs)}
        chosen[ix] = decide(np.array([pos[r] for r in refs[ix]]), P[ix], 'expected', thr, len(prefs))
    by = defaultdict(list)
    for j in np.flatnonzero(part != 'train'):
        by[refs[j]].append(j)
    A = Counter(); D = Counter(); ex = defaultdict(list)
    for s, js in by.items():
        rn, ra, cty = text[s]
        rc = set(core(rn))
        aliases = set()
        for j in js:
            tn = text.get(tgts[j], ('', ''))[0]
            parts = AKA.split(tn)
            if len(parts) >= 2:
                left, right = set(core(parts[0])), set(core(' '.join(parts[1:])))
                if left and right and len(right & rc) >= max(1, len(rc) // 2) and not left & rc:
                    aliases.add(frozenset(left))
        for j in js:
            tn, ta, _ = text.get(tgts[j], ('', '', ''))
            tc = set(core(tn))
            if aliases and tc and not tc & rc and len(AKA.split(tn)) == 1 and any(tc <= al or al <= tc for al in aliases):
                k = (cty, 'bare_alias_sibling')
                A[k + ('n',)] += 1; A[k + ('true',)] += int(y[j]); A[k + ('t6_acc',)] += int(chosen[j])
                A[k + ('t6_missed',)] += int(y[j] and not chosen[j]); A[k + ('t6_false',)] += int(chosen[j] and not y[j])
                if len(ex['alias']) < 12:
                    ex['alias'].append(f'{int(y[j])} {P[j]:.2f} {rn} => {tn} | {ta}')
            if not ta.strip():
                continue
            h1, h2 = house(ra)[0], house(ta)[0]
            if not h1 or not h2 or h1 == h2:
                continue
            nr = num_rel(h1, h2)
            nm = 'same' if rc == tc else ('filler' if (tc - rc) <= EN_FILLER and (rc - tc) <= EN_FILLER else ('none' if not rc & tc else 'edit'))
            sw = street_words(ra)
            st = 'street' if sw and sw <= tokens_addr(ta) else 'other'
            k = (cty, nm, st, nr)
            D[k + ('n',)] += 1; D[k + ('true',)] += int(y[j]); D[k + ('missed',)] += int(y[j] and not chosen[j])
            D[k + ('false_acc',)] += int(chosen[j] and not y[j])
    def table(C):
        keys = sorted({k[:-1] for k in C})
        return {str(k): {m: C[k + (m,)] for m in ('n', 'true', 't6_acc', 't6_missed', 't6_false', 'missed', 'false_acc') if k + (m,) in C}
                for k in keys}
    R = {'alias': table(A), 'digit': {k: v for k, v in table(D).items() if v['n'] >= 30}, 'examples': ex}
    out = json.dumps(R, indent=1, ensure_ascii=False)
    open(os.environ['OUT'], 'w', encoding='utf-8').write(out)
    print(out, flush=True)


if __name__ == '__main__':
    main()
