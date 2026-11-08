"""Blind judge set: 80 holdout pairs from T6's decision band (0.15 < p < 0.85), with the context an LLM prompt would
get (S1 record, candidate record, the S1's other candidates, how many other S1s share the S1's exact normalized name).
Questions file has no label and no p; the key file (id, y, p, blank) is written separately. Env A (sweep dir), OUT_Q,
OUT_K.
"""
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
sys.path.insert(0, str(Path(__file__).parent))
from france_census import fold

H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
nrm = lambda s: re.sub(r'[^a-z0-9]+', ' ', fold(s or '')).strip()
df = pd.read_pickle(f"{os.environ['A']}/train.json.train_pairs.pkl")
band = df[(df.part == 'holdout') & (df.p > 0.15) & (df.p < 0.85)]
smp = pd.concat([band[band.blank == 'blank'].sample(45, random_state=7), band[band.blank != 'blank'].sample(35, random_state=7)])
smp = smp.sample(frac=1, random_state=3).reset_index(drop=True)
need = set(df[df.s1.isin(smp.s1)].t) | set(smp.s1)
text, s1names = {}, Counter()
for k in (1, 2, 3):
    with open(H / f'data/dataset/train/train_source{k}.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if k == 1:
                s1names[(q[c], nrm(q[n]))] += 1
            if q[ie] in need:
                text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
with open(os.environ['OUT_Q'], 'w', encoding='utf-8') as fq, open(os.environ['OUT_K'], 'w') as fk:
    fk.write('id\ty\tp\tblank\n')
    for i, r in smp.iterrows():
        n, a, c = text[r.s1]
        tw = s1names[(c, nrm(n))] - 1
        others = [t for t in df[df.s1 == r.s1].t if t != r.t]
        fq.write(f'Q{i:02d} | S1: {n} | {a} | {c} | other S1s with this exact name: {tw}\n')
        fq.write(f'     CANDIDATE [{r.t[:2]}]: {text[r.t][0]} | {text[r.t][1] or "(blank)"}\n')
        for t in others:
            fq.write(f'       other cand [{t[:2]}]: {text[t][0]} | {text[t][1] or "(blank)"}\n')
        fk.write(f'Q{i:02d}\t{r.y}\t{r.p:.4f}\t{r.blank}\n')
print('written', len(smp))
