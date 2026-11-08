"""Alias link-up: a sibling record "X dba/fka/aka <S1 name>" names X as an alias of that S1, so another candidate record
of the same S1 whose name is X belongs to it (blank address or not). The pair models never see sibling alias text.

Train (US/India dev + holdout, T6 decisions from error_sweep.py frames): flagged pairs (candidate name == an alias
announced by an ACCEPTED sibling of the same S1), their true rate, how many T6 rejects, F0.5 change when the rejected
flagged pairs are added. Test: the same flags on U5 (candidates + accepted links), count of links it would add per
country, and records already held by another S1. Env A (sweep dir), U5_DIR.
"""
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
from france_census import core, load

H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
A = os.environ['A']
SEP = re.compile(r'\s+(?:d\s*/\s*b\s*/\s*a|dba|f\s*/\s*k\s*/\s*a|fka|a\s*/\s*k\s*/\s*a|aka|formerly(?:\s+known\s+as)?|'
                 r'doing\s+business\s+as|trading\s+as|t\s*/\s*a)\.?\s+', re.I)


def ckey(name):
    return ' '.join(sorted(set(core(name))))


def compact(name):
    return re.sub(r'[^a-z0-9]', '', ' '.join(core(name)))


def alias_of(rec_name, s1_name):
    """Alias side of 'X dba Y' when the other side is the S1's name, else None."""
    parts = SEP.split(rec_name, maxsplit=1)
    if len(parts) != 2:
        return None
    s = set(core(s1_name))
    if not s:
        return None
    j = [len(set(core(p)) & s) / max(1, len(set(core(p)) | s)) for p in parts]
    k = int(np.argmax(j))
    if j[k] < 0.5:
        return None
    al = parts[1 - k]
    return compact(al) or None


def flags(s1_name, cands, accepted, text):
    """cands: list of record ids; accepted: set. Returns set of candidate ids named like an alias that an accepted
    sibling announces."""
    als = {}
    for t in cands:
        if t in accepted:
            a = alias_of(text.get(t, ('',))[0], s1_name)
            if a and len(a) >= 3:
                als[a] = t
    if not als:
        return set()
    out = set()
    for t in cands:
        c = compact(text.get(t, ('',))[0])
        if c and c in als and als[c] != t:
            out.add(t)
    return out


def read_text(split, need):
    text, cty = {}, {}
    for k in (1, 2, 3):
        with open(H / f'data/dataset/{split}/{split}_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if k == 1:
                    cty[q[ie]] = q[c]
                if k == 1 or q[ie] in need:
                    text[q[ie]] = (q[n], q[a] if len(q) > a else '')
    return text, cty


def F(tp, npred, ngold):
    tp, npred, ngold = map(np.asarray, (tp, npred, ngold))
    pr = np.where(npred > 0, tp / np.maximum(npred, 1), 0); rc = np.where(ngold > 0, tp / np.maximum(ngold, 1), 0)
    f = np.where(tp > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
    return np.where((npred == 0) & (ngold == 0), 1.0, f)


def train():
    df = pd.read_pickle(f'{A}/train.json.train_pairs.pkl')
    bf = pd.read_pickle(f'{A}/train.json.train_blocking.pkl')
    text, _ = read_text('train', set(df.t))
    miss = bf.groupby('s1').size()
    rows = []
    for s, g in df.groupby('s1'):
        acc = set(g.t[g.D == 1])
        fl = flags(text[s][0], list(g.t), acc, text)
        for t, y, d, p, blank in zip(g.t, g.y, g.D, g.p, g.blank):
            if t in fl:
                rows.append(dict(s1=s, t=t, y=y, D=d, p=p, blank=blank, part=g.part.iloc[0], country=g.country.iloc[0]))
    fl = pd.DataFrame(rows)
    print('TRAIN flagged pairs', len(fl))
    if not len(fl):
        return
    print(fl.groupby(['part', 'D']).agg(n=('y', 'size'), true_rate=('y', 'mean'), mean_p=('p', 'mean')).to_string())
    print('flagged & rejected by blank:', fl[fl.D == 0].groupby('blank').agg(n=('y', 'size'), true_rate=('y', 'mean')).to_string())
    g = df.groupby('s1')
    S = pd.DataFrame({'part': g.part.first(), 'ytrue': g.y.sum(), 'npred': g.D.sum(), 'tp': (df.y * df.D).groupby(df.s1).sum()})
    S['ngold'] = S.ytrue + miss.reindex(S.index).fillna(0).astype(int)
    add = fl[fl.D == 0].groupby('s1').agg(a=('y', 'size'), b=('y', 'sum'))
    S = S.join(add).fillna({'a': 0, 'b': 0})
    for p in ('dev', 'holdout'):
        s = S[S.part == p]
        d = F(s.tp + s.b, s.npred + s.a, s.ngold).mean() - F(s.tp, s.npred, s.ngold).mean()
        print(f'  {p}: add {int(s.a.sum())} links ({int(s.b.sum())} true) -> F0.5 {d:+.6f}  (public-scale x0.85 {0.85 * d:+.6f})')
    print('examples:')
    for _, r in fl[fl.D == 0].head(15).iterrows():
        print('   y', r.y, f'p={r.p:.2f}', text[r.s1][0], '|', text[r.s1][1][:40], ' => ', text.get(r.t, ('', ''))[0], '|', text.get(r.t, ('', ''))[1][:40])


def test():
    d = Path(os.environ['U5_DIR'])
    u5, cand = load(d / 'matching_results.tsv'), load(d / 'candidate_pairs.tsv')
    need = {t for ts in cand.values() for t in ts}
    text, cty = read_text('test', need)
    owner = {t: s for s, ts in u5.items() for t in ts}
    R = Counter(); ex = defaultdict(list)
    for s, cs in cand.items():
        acc = set(u5.get(s, []))
        for t in flags(text[s][0], cs, acc, text):
            c = cty[s]
            if t in acc:
                R[(c, 'already_accepted')] += 1
            elif t in owner:
                R[(c, 'held_by_other_S1')] += 1
            else:
                R[(c, 'would_add')] += 1
                if len(ex[c]) < 12:
                    ex[c].append(f"{text[s][0]} | {text[s][1][:40]}  =>  {text.get(t, ('', ''))[0]} | {text.get(t, ('', ''))[1][:40]}")
    print('TEST', dict(R))
    for c, e in ex.items():
        print(' ', c); [print('   ', x) for x in e]


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    (train if os.environ.get('MODE') == 'train' else test)()
