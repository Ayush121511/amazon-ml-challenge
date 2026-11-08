"""Every-row check of U5 + the textual-decidability ceiling for any reader (LLM or human).

1. Exact-copy rule. Key = (country, normalized full name, normalized address); records whose key matches exactly one
   S1 (non-blank address). Train (FULL ground truth, 2.2M S1): how often is the record that S1's? Also a looser key
   (core-name token set + address token set). Test: U5 status of every such record (linked to that S1 / to another S1 /
   unlinked but in candidates / unlinked and not in candidates).
2. Twins. An S1 whose normalized full name is shared by another S1 of the same country cannot be told apart from it for
   a blank-address record by any reader. Share of the uncertain band and of holdout error loss (error_sweep.py frames)
   on such pairs = the part no LLM can fix.
Env A (sweep dir), U5_DIR.
"""
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
sys.path.insert(0, str(Path(__file__).parent))
from france_census import core, fold, load

H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')


def nrm(s):
    return re.sub(r'[^a-z0-9]+', ' ', fold(s or '')).strip()


def keys(name, addr, cty):
    return ((cty, nrm(name), nrm(addr)),
            (cty, ' '.join(sorted(set(core(name)))), ' '.join(sorted(set(nrm(addr).split())))))


def read(split):
    s1, rec = {}, {}
    for k in (1, 2, 3):
        with open(H / f'data/dataset/{split}/{split}_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                v = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
                (s1 if k == 1 else rec)[q[ie]] = v
    return s1, rec


def exact_matches(s1, rec):
    idx = [defaultdict(list), defaultdict(list)]
    for e, (n, a, c) in s1.items():
        if a.strip():
            for i, k in enumerate(keys(n, a, c)):
                idx[i][k].append(e)
    out = [[], []]
    for r, (n, a, c) in rec.items():
        if not a.strip():
            continue
        for i, k in enumerate(keys(n, a, c)):
            hit = idx[i].get(k)
            if hit and len(hit) == 1:
                out[i].append((r, hit[0]))
    return out


def twins(s1):
    c = Counter((v[2], nrm(v[0])) for v in s1.values())
    return {e for e, v in s1.items() if c[(v[2], nrm(v[0]))] > 1}


def main():
    A = os.environ['A']
    s1, rec = read('train')
    gold = {}
    with open(H / 'data/dataset/train/train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            q = line.rstrip('\n').split('\t')
            gold[q[0]] = set(x.strip() for x in (q[1] if len(q) > 1 else '').split(',') if x.strip())
    owner = {t: s for s, ts in gold.items() for t in ts}
    print('TRAIN records', len(rec), 'S1', len(s1))
    for i, pairs in enumerate(exact_matches(s1, rec)):
        y = np.array([owner.get(r) == s for r, s in pairs])
        other = np.array([owner.get(r) not in (None, s) for r, s in pairs])
        print(f'  key{i} ({"strict" if i == 0 else "loose"}): matches {len(pairs)}, true {y.mean():.5f}, '
              f'owned by another S1 {other.mean():.5f}, owned by nobody {1 - y.mean() - other.mean():.5f}')
    tw = twins(s1)
    print('  train S1 with an exact-name twin:', len(tw), f'({len(tw) / len(s1):.3%})')
    df = pd.read_pickle(f'{A}/train.json.train_pairs.pkl')
    df['twin'] = df.s1.isin(tw)
    band = df[(df.p > 0.02) & (df.p < 0.98)]
    err = df[df.kind.isin(['FN', 'FP']) & (df.part == 'holdout')]
    for nm, d in (('uncertain band pairs', band), ('holdout error loss', err)):
        w = d.loss if nm.startswith('holdout') else pd.Series(1.0, index=d.index)
        cell = d.assign(w=w).groupby(['blank', 'twin']).w.sum()
        print(f'  {nm}: share by (blank, S1 has exact-name twin):', (cell / cell.sum()).round(4).to_dict())
    del s1, rec, gold, owner
    s1, rec = read('test')
    u5, cand = load(Path(os.environ['U5_DIR']) / 'matching_results.tsv'), load(Path(os.environ['U5_DIR']) / 'candidate_pairs.tsv')
    link = {t: s for s, ts in u5.items() for t in ts}
    candset = {(s, t) for s, ts in cand.items() for t in ts}
    print('TEST records', len(rec), 'S1', len(s1), 'linked records', len(link))
    for i, pairs in enumerate(exact_matches(s1, rec)):
        R = Counter(); ex = defaultdict(list)
        for r, s in pairs:
            c = s1[s][2]
            if link.get(r) == s:
                k = 'agree'
            elif r in link:
                k = 'linked_to_other_S1'
            elif (s, r) in candset:
                k = 'unlinked_in_candidates'
            else:
                k = 'unlinked_not_candidate'
            R[(c, k)] += 1
            if k != 'agree' and len(ex[(c, k)]) < 6:
                o = link.get(r)
                ex[(c, k)].append(f'{s1[s][0]} | {s1[s][1][:45]}  <=  {rec[r][0]} | {rec[r][1][:45]}' +
                                  (f'   [U5 gave it to: {s1[o][0]} | {s1[o][1][:40]}]' if o else ''))
        print(f'  key{i}:', dict(sorted(R.items())))
        for k, e in sorted(ex.items()):
            print('   ', k); [print('      ', x) for x in e]


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
