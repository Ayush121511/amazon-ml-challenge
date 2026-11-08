"""Alias rule, conditioned on the model: among US/India dev + holdout top-10 pairs (error_sweep frames, T6 decisions),
records whose name is one Latin word (after dropping non-Latin script) sharing no word with the S1 name, at the S1's
exact address key (first house number + street words), with that S1 the only S1 at the key. True rate split by T6's
decision D and by country, and the F0.5 change on dev / holdout if every rejected flagged pair is added."""
import os
import re
import unicodedata
from collections import defaultdict
import numpy as np
import pandas as pd

H = '/home/scai/mtech/aib262015/scratch/amazon_ml_2026/data/dataset/train/'
fold = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s or '') if not unicodedata.combining(c)).lower()
toks = lambda s: re.sub(r'[^a-z0-9]+', ' ', fold(s)).split()
STOP = {'inc', 'llc', 'ltd', 'corp', 'co', 'pvt', 'private', 'limited', 'the', 'and', 'of', 'company', 'corporation', 'lp', 'llp',
        'pc', 'group', 'services', 'service', 'center', 'partners', 'holdings', 'enterprises', 'india'}


def akey(a):
    t = toks(a); nums = [x for x in t if x.isdigit()]
    if not nums:
        return None
    return (str(int(nums[0])), ' '.join(sorted(set(x for x in t if not any(ch.isdigit() for ch in x) and len(x) > 2))[:6]))


df = pd.read_pickle(f"{os.environ['A']}/train.json.train_pairs.pkl")
bf = pd.read_pickle(f"{os.environ['A']}/train.json.train_blocking.pkl")
need = set(df.t)
s1, rec, idx = {}, {}, defaultdict(int)
for k in (1, 2, 3):
    f = open(H + f'train_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
    for l in f:
        q = l.rstrip('\n').split('\t'); ad = q[a] if len(q) > a else ''
        if k == 1:
            s1[q[ie]] = (q[n], ad, q[c]); kk = akey(ad)
            if kk:
                idx[(q[c],) + kk] += 1
        elif q[ie] in need:
            rec[q[ie]] = (q[n], ad)


def flag(s, t):
    n, a, c = s1[s]; tn, ta = rec.get(t, ('', ''))
    if not ta.strip():
        return False
    k1, k2 = akey(a), akey(ta)
    if not k1 or k1 != k2 or idx[(c,) + k1] != 1:
        return False
    tk = toks(tn)
    return len(tk) == 1 and tk[0].isalpha() and not (set(tk) & (set(toks(n)) - STOP))


df['flag'] = [flag(s, t) for s, t in zip(df.s1, df.t)]
F = df[df.flag]
print('flagged', len(F))
print(F.groupby(['country', 'D']).agg(n=('y', 'size'), true_rate=('y', 'mean'), mean_p=('p', 'mean')).round(3).to_string())
miss = bf.groupby('s1').size(); g = df.groupby('s1')
S = pd.DataFrame({'part': g.part.first(), 'ytrue': g.y.sum(), 'npred': g.D.sum(), 'tp': (df.y * df.D).groupby(df.s1).sum()})
S['ngold'] = S.ytrue + miss.reindex(S.index).fillna(0).astype(int)
add = F[F.D == 0].groupby('s1').agg(a=('y', 'size'), b=('y', 'sum')); S = S.join(add).fillna({'a': 0, 'b': 0})


def Fs(tp, npred, ngold):
    tp, npred, ngold = map(np.asarray, (tp, npred, ngold))
    pr = np.where(npred > 0, tp / np.maximum(npred, 1), 0); rc = np.where(ngold > 0, tp / np.maximum(ngold, 1), 0)
    f = np.where(tp > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
    return np.where((npred == 0) & (ngold == 0), 1.0, f)


for p in ('dev', 'holdout'):
    s = S[S.part == p]
    print(p, 'add', int(s.a.sum()), 'true', int(s.b.sum()), 'F0.5 delta', round(float(Fs(s.tp + s.b, s.npred + s.a, s.ngold).mean() - Fs(s.tp, s.npred, s.ngold).mean()), 6))
