"""Share of India holdout loss (T6) on records whose name is (mostly) in a native Indian script, split by stage."""
import os, re
import pandas as pd
A = os.environ['A']
df = pd.read_pickle(f'{A}/train.json.train_pairs.pkl'); bf = pd.read_pickle(f'{A}/train.json.train_blocking.pkl')
need = set(df.t) | set(bf.t); name = {}
for k in (2, 3):
    f = open(f'/home/scai/mtech/aib262015/scratch/amazon_ml_2026/data/dataset/train/train_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    ie, n = h.index('entity_id'), h.index('business_name')
    for l in f:
        q = l.rstrip('\n').split('\t')
        if q[ie] in need: name[q[ie]] = q[n]
def native(s):
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(ord(c) > 0x0900 for c in letters) / len(letters) > 0.5
df['native'] = df.t.map(lambda t: native(name.get(t, ''))); bf['native'] = bf.t.map(lambda t: native(name.get(t, '')))
for p in ('dev', 'holdout'):
    d = df[(df.part == p) & (df.country == 'India')]; b = bf[(bf.part == p) & (bf.country == 'India')] if 'country' in bf else bf[bf.part == p]
    nS = d.s1.nunique()
    print(f'{p} India: pairs {len(d)}, native-script share of pairs {d.native.mean():.3f}')
    for nat in (True, False):
        e = d[d.native == nat]
        print(f'   native={nat}: FN loss pts {100 * e[e.kind == "FN"].loss.sum() / nS:.4f}  FP loss pts {100 * e[e.kind == "FP"].loss.sum() / nS:.4f}  '
              f'blocking loss pts {100 * b[b.native == nat].loss.sum() / nS:.4f}  | accepted true-rate {e[e.D == 1].y.mean():.4f}  rejected true-rate {e[e.D == 0].y.mean():.4f}')
