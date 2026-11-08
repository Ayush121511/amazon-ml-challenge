"""Pattern cells (pattern_audit.py definitions) on the US/India dev + holdout top-10 pairs, split by T6's decision D:
true rate of the pairs T6 rejects / accepts per cell, and the F0.5 change on dev and holdout if a cell's rejected pairs
were all added (or accepted pairs all dropped)."""
import os, sys
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import pattern_audit as pa
pa.D = '/home/scai/mtech/aib262015/scratch/amazon_ml_2026/data/dataset/'
s1, own, street, names = pa.build_index('train')
df = pd.read_pickle(f"{os.environ['A']}/train.json.train_pairs.pkl"); bf = pd.read_pickle(f"{os.environ['A']}/train.json.train_blocking.pkl")
need = set(df.t); rec = {}
for k in (2, 3):
    f = open(pa.D + f'train/train_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    ie, n, a = (h.index(x) for x in ('entity_id', 'business_name', 'business_address'))
    for l in f:
        q = l.rstrip('\n').split('\t')
        if q[ie] in need: rec[q[ie]] = (q[n], q[a] if len(q) > a else '')
def cell(s, t):
    rn, ra = rec.get(t, ('', '')); sn, sa, cc = s1[s]
    if ra.strip():
        num, words = pa.akey(ra); hit = own.get((cc, num, words)) if num and words else None
        if hit and len(hit) == 1:
            return ('A_own_address', pa.rel(sn, rn)[0]) if hit[0] == s else None
        hit = street.get((cc, words)) if words else None
        if hit and len(hit) == 1 and hit[0] == s and num:
            snum = pa.akey(sa)[0]
            if snum:
                return ('B_street_' + pa.shift(snum, num), pa.rel(sn, rn)[0])
        return None
    hit = names.get((cc, pa.nrm(rn)))
    return ('C_blank_unique_name', 'exact') if hit and len(hit) == 1 and hit[0] == s else None
df['cell'] = [cell(s, t) for s, t in zip(df.s1, df.t)]
d = df[df.cell.notna()].copy(); d['cell'] = d.cell.map(lambda c: c[0] + ':' + c[1])
T = d.groupby(['cell', 'D']).agg(n=('y', 'size'), true_rate=('y', 'mean')).unstack('D')
T = T[(T[('n', 0)].fillna(0) + T[('n', 1)].fillna(0)) >= 50]
pd.set_option('display.width', 200); print(T.round(3).to_string())
miss = bf.groupby('s1').size(); g = df.groupby('s1')
S = pd.DataFrame({'part': g.part.first(), 'ytrue': g.y.sum(), 'npred': g.D.sum(), 'tp': (df.y * df.D).groupby(df.s1).sum()})
S['ngold'] = S.ytrue + miss.reindex(S.index).fillna(0).astype(int)
def Fs(tp, npred, ngold):
    tp, npred, ngold = map(np.asarray, (tp, npred, ngold))
    pr = np.where(npred > 0, tp / np.maximum(npred, 1), 0); rc = np.where(ngold > 0, tp / np.maximum(ngold, 1), 0)
    f = np.where(tp > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
    return np.where((npred == 0) & (ngold == 0), 1.0, f)
print('\nper-cell action on dev / holdout (ADD = add all T6-rejected pairs of the cell; DROP = drop all accepted):')
for c in sorted(d.cell.unique()):
    for act, dv in (('ADD', 0), ('DROP', 1)):
        sel = d[(d.cell == c) & (d.D == dv)]
        if len(sel) < 15: continue
        a = sel.groupby('s1').agg(k=('y', 'size'), b=('y', 'sum'))
        out = []
        for p in ('dev', 'holdout'):
            s = S[S.part == p].join(a).fillna({'k': 0, 'b': 0})
            if act == 'ADD': new = Fs(s.tp + s.b, s.npred + s.k, s.ngold)
            else: new = Fs(s.tp - s.b, s.npred - s.k, s.ngold)
            out.append(f'{p} {new.mean() - Fs(s.tp, s.npred, s.ngold).mean():+.6f}')
        print(f'  {act:4s} {c:40s} n={len(sel):5d} true={sel.y.mean():.3f}  ' + '  '.join(out))
