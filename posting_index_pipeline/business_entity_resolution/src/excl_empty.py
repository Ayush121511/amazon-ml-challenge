"""Exclusivity finalize on test: when a record is claimed by several S1s, the highest p keeps it and the rest re-decide.
How often does the loser S1 end with NO link (full F0.5 = 0 if it has matches), who won the record, with what p, and
how many links the winner keeps. Uses error_sweep.py test frames + U2 final + the T6 / FRtrained_w4 probabilities.
Env A (sweep dir).
"""
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

A = os.environ['A']
V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
tf = pd.read_pickle(f'{A}/test.json.test_pairs.pkl')
final = {}
with open(H / 'artifacts/submission_U2_final/matching_results.tsv', encoding='utf-8') as fh:
    next(fh)
    for line in fh:
        s, m = (line.rstrip('\n').split('\t') + [''])[:2]
        final[s] = [x for x in m.split(',') if x]
owner = {t: s for s, ts in final.items() for t in ts}
X = tf[tf.step == 'exclusivity_drop'].copy()
need = set(X.t)
s1 = []
with open(H / 'data/dataset/test/test_source1.tsv', encoding='utf-8') as fh:
    next(fh)
    for line in fh:
        s1.append(line.split('\t', 1)[0])
P = {}
for f, fr in ((V / 'submission_T6_own/late_scores.npz', False), (V / 'submission_FRtrained_w4/late_scores.npz', True)):
    Z = np.load(f, allow_pickle=True)
    tg = Z['target'].astype(str); ri = Z['ref_idx']; p = Z['prob']
    for j in np.flatnonzero(np.isin(tg, list(need))):
        P[(s1[ri[j]], tg[j], fr)] = float(p[j])
X['winner'] = X.t.map(owner)
X['L_final'] = X.s1.map(lambda s: len(final.get(s, [])))
X['W_final'] = X.winner.map(lambda s: len(final.get(s, [])) if isinstance(s, str) else -1)
X['pW'] = [P.get((w, t, c == 'France'), np.nan) if isinstance(w, str) else np.nan for w, t, c in zip(X.winner, X.t, X.country)]
X['nclaim'] = X.t.map(tf[tf.p > 0.5].groupby('t').size()).fillna(0)
for c, g in X.groupby('country'):
    e = g[g.L_final == 0]
    print(f'{c}: dropped {len(g)} | winner none {int(g.winner.isna().sum())} | loser emptied {len(e)} ({e.s1.nunique()} S1) '
          f'| emptied & winner keeps >=2 links {int((e.W_final >= 2).sum())} | pL mean {e.p.mean():.3f} pW mean {e.pW.mean():.3f}')
    print('   emptied by blank/name:', dict(Counter(zip(e.blank, e.name)).most_common(6)))
    print('   emptied pL bands:', dict(Counter(pd.cut(e.p, [0, .5, .74, .9, .98, 1.01]).astype(str))))
    print('   emptied winner link count:', dict(Counter(e.W_final.clip(upper=6))))
    print('   emptied mult:', dict(Counter(e.mult)), ' claimants(p>0.5):', dict(Counter(e.nclaim.clip(upper=5))))
    print('   public points if every emptied S1 were a true non-singleton owning t (ceiling):',
          round(100 * e.s1.nunique() / 1732544, 4))
