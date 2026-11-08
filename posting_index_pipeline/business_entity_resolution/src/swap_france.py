"""Ablation build: U5 with its France rows replaced by another France version, keeping U5's France rescue links.

U5 France = U2 France (FRtrained_w4 -> exclusivity prune -> census v2) + rescue links (ranks 11-20). U6 = U5 US/India
rows + FR_TSV's France rows (default: U1's France, census v2 without the prune) + the same rescue links, each dropped if
its record is already held by another S1 in the new rows (a record belongs to one S1). Candidates = U5's.
Env U5_DIR, U2_TSV, FR_TSV, OUT_DIR.
"""
import os
import shutil
from collections import Counter
from pathlib import Path

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
from france_census import load

H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
u5d = Path(os.environ['U5_DIR'])
u5, u2, fr = load(u5d / 'matching_results.tsv'), load(os.environ['U2_TSV']), load(os.environ['FR_TSV'])
cand = load(u5d / 'candidate_pairs.tsv')
rows, cty = [], {}
with open(H / 'data/dataset/test/test_source1.tsv', encoding='utf-8') as fh:
    head = fh.readline().rstrip('\n').split('\t')
    ie, ic = head.index('entity_id'), head.index('country')
    for line in fh:
        q = line.rstrip('\n').split('\t'); cty[q[ie]] = q[ic]
order = []
with open(u5d / 'matching_results.tsv', encoding='utf-8') as fh:
    next(fh)
    order = [line.split('\t', 1)[0].rstrip('\n') for line in fh]
new = {r: (list(fr.get(r, [])) if cty[r] == 'France' else list(u5[r])) for r in order}
owner = Counter(t for ts in new.values() for t in ts)
R = Counter()
for r in order:
    if cty[r] != 'France':
        continue
    for t in set(u5[r]) - set(u2[r]):
        if owner[t]:
            R['rescue_dropped_record_held'] += 1
            continue
        new[r].append(t); owner[t] += 1; R['rescue_kept'] += 1
R['records_in_2_rows'] = sum(v > 1 for v in owner.values())
R['matches_outside_candidates'] = sum(len(set(new[r]) - set(cand.get(r, []))) for r in order)
for c in ('France', 'India', 'US'):
    a = sum(len(set(new[r]) - set(u5[r])) for r in order if cty[r] == c)
    d = sum(len(set(u5[r]) - set(new[r])) for r in order if cty[r] == c)
    R[f'{c}_vs_U5_added'] = a; R[f'{c}_vs_U5_removed'] = d
    R[f'{c}_empty'] = sum(not new[r] for r in order if cty[r] == c)
out = Path(os.environ['OUT_DIR']); out.mkdir(parents=True, exist_ok=True)
with open(out / 'matching_results.tsv', 'w', encoding='utf-8') as f:
    f.write('source1_entity_id\tmatched_entity_ids\n')
    for r in order:
        f.write(r + '\t' + ','.join(new[r]) + '\n')
shutil.copy(u5d / 'candidate_pairs.tsv', out / 'candidate_pairs.tsv')
print(dict(R), flush=True)
