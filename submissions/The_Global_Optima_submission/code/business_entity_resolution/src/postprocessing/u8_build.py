"""U8 = U7 + France invented-alias records at the S1's exact address (unique S1 at the address key, single Latin word
sharing no name word, unlinked in U7, in candidates) that the US/India-trained T6 bag accepts (p >= T6_T, its rule
0.68). Train labels: this rule is 98.7% true overall and 99.8-100% where T6 accepts. US/India rows untouched.
Env ADDS (s1, record tsv from the test alias scan), OUT_DIR, T6_T."""
import os, shutil
from collections import Counter
from pathlib import Path
import numpy as np

os.environ.setdefault('SUB_DIR', '.')
from france_census import load

U7 = Path('/scratch/scai/mtech/aib262467/final_u7')
BAG = Path('/scratch/scai/mtech/aib262467/t6bag/test/late_scores.npz')
H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
s1, cty = [], {}
with open(H / 'data/dataset/test/test_source1.tsv', encoding='utf-8') as fh:
    head = fh.readline().rstrip('\n').split('\t'); ie, ic = head.index('entity_id'), head.index('country')
    for line in fh:
        q = line.rstrip('\n').split('\t'); s1.append(q[ie]); cty[q[ie]] = q[ic]
adds = [l.rstrip('\n').split('\t') for l in open(os.environ['ADDS'])]
adds = [(s, t) for s, t in adds if cty[s] == 'France']
Z = np.load(BAG, allow_pickle=True)
ri, tg, p = Z['ref_idx'], Z['target'].astype(str), Z['prob']
want = set(adds); pos = {r: i for i, r in enumerate(s1)}
key = {(pos[s], t): (s, t) for s, t in adds}
pb = {}
for j in np.flatnonzero(np.isin(tg, [t for _, t in adds])):
    k = (int(ri[j]), tg[j])
    if k in key:
        pb[key[k]] = float(p[j])
thr = float(os.environ.get('T6_T', 0.68))
bands = Counter(('missing' if a not in pb else ('>=%.2f' % thr if pb[a] >= thr else ('0.5-%.2f' % thr if pb[a] >= 0.5 else ('0.2-0.5' if pb[a] >= 0.2 else '<0.2')))) for a in adds)
print('France alias candidates', len(adds), 'by T6-bag p:', dict(bands))
u7 = load(U7 / 'matching_results.tsv')
owner = Counter(t for ts in u7.values() for t in ts)
new = {s: list(ts) for s, ts in u7.items()}; R = Counter()
for (s, t) in sorted(adds, key=lambda a: -pb.get(a, 0)):
    if pb.get((s, t), 0) < thr:
        continue
    if owner[t]:
        R['held'] += 1; continue
    new[s].append(t); owner[t] += 1; R['added'] += 1; R['revived'] += int(len(new[s]) == 1)
out = Path(os.environ['OUT_DIR']); out.mkdir(parents=True, exist_ok=True)
order = [l.split('\t', 1)[0] for l in open(U7 / 'matching_results.tsv', encoding='utf-8')][1:]
with open(out / 'matching_results.tsv', 'w', encoding='utf-8') as f:
    f.write('source1_entity_id\tmatched_entity_ids\n')
    for s in order:
        f.write(s + '\t' + ','.join(new[s]) + '\n')
shutil.copy(U7 / 'candidate_pairs.tsv', out / 'candidate_pairs.tsv')
R['records_in_2_rows'] = sum(v > 1 for v in owner.values())
print(dict(R), flush=True)
