"""Empty-guard check on T6 dev/holdout: when the expected-F0.5 rule leaves an S1 empty, add its top candidate if p >= x.

Holdout loss has 0.225 points on S1s predicted empty that have matches vs 0.007 on true singletons given a link, so
the empty choice may be too eager (prod(1 - p) treats candidates as independent, ignoring that 94% of S1s have a match).
Uses error_sweep.py frames. Also profiles the wrongly-empty S1s. Env A (sweep dir).
"""
import os
from collections import Counter

import numpy as np
import pandas as pd

A = os.environ['A']
df = pd.read_pickle(f'{A}/train.json.train_pairs.pkl')
bf = pd.read_pickle(f'{A}/train.json.train_blocking.pkl')
miss = bf.groupby('s1').size()
g = df.groupby('s1')
S = pd.DataFrame({'part': g.part.first(), 'country': g.country.first(), 'ytrue': g.y.sum(), 'npred': g.D.sum(),
                  'tp': (df.y * df.D).groupby(df.s1).sum(), 'top_p': g.p.max(), 'n': g.size()})
S['ngold'] = S.ytrue + miss.reindex(S.index).fillna(0).astype(int)
top = df.sort_values('p', ascending=False).drop_duplicates('s1').set_index('s1')
S['top_y'] = top.y.reindex(S.index); S['top_blank'] = top.blank.reindex(S.index); S['top_name'] = top.name.reindex(S.index)
S['sum_p'] = g.p.sum()


def F(tp, npred, ngold):
    tp, npred, ngold = map(np.asarray, (tp, npred, ngold))
    pr = np.where(npred > 0, tp / np.maximum(npred, 1), 0); rc = np.where(ngold > 0, tp / np.maximum(ngold, 1), 0)
    f = np.where(tp > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
    return np.where((npred == 0) & (ngold == 0), 1.0, f)


S['F'] = F(S.tp, S.npred, S.ngold)
for p in ('dev', 'holdout'):
    s = S[S.part == p]
    e = s[s.npred == 0]
    print(p, 'F', round(s.F.mean(), 6), 'pred-empty', len(e), 'of', len(s), 'wrong-empty', int((e.ngold > 0).sum()),
          'true singletons', int((s.ngold == 0).sum()), 'singleton FP', int(((s.ngold == 0) & (s.npred > 0)).sum()))
    print('  wrong-empty: top_y', Counter(e[e.ngold > 0].top_y), 'top blank', Counter(e[e.ngold > 0].top_blank),
          'ngold', Counter(e[e.ngold > 0].ngold.clip(upper=6)), 'ytrue in list', Counter(e[e.ngold > 0].ytrue.clip(upper=4)))
    for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .74), (.74, 1.01)):
        b = e[(e.top_p >= lo) & (e.top_p < hi)]
        print(f'  empty, top_p {lo:.2f}-{hi:.2f}: n {len(b)} has match {int((b.ngold > 0).sum())} top true {int((b.top_y == 1).sum())}'
              f' sum_p mean {b.sum_p.mean():.2f}')
for x in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
    out = []
    for p in ('dev', 'holdout'):
        s = S[S.part == p]
        add = (s.npred == 0) & (s.top_p >= x)
        f2 = np.where(add, F(s.tp + s.top_y, s.npred + 1, s.ngold), s.F)
        out.append(f'{p} {f2.mean() - s.F.mean():+.6f} (n {int(add.sum())})')
    print('guard top_p >=', x, ' | '.join(out))
for x in (0.6, 0.8, 1.0, 1.2, 1.5):
    out = []
    for p in ('dev', 'holdout'):
        s = S[S.part == p]
        add = (s.npred == 0) & (s.sum_p >= x)
        f2 = np.where(add, F(s.tp + s.top_y, s.npred + 1, s.ngold), s.F)
        out.append(f'{p} {f2.mean() - s.F.mean():+.6f} (n {int(add.sum())})')
    print('guard sum_p >=', x, ' | '.join(out))
