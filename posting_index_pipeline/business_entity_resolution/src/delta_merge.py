"""Apply the LLM-judge changes to the best file as a delta, keeping every other rule's links.

US/India delta = (LLM-blended decisions) - (T6 bag decisions reproduced by the same finalize), from llm_apply.py's
OUT_DIR/decided and OUT_DIR/repro_check. France delta = the ADD/DROP list written by llm_france.py (optional).
Adds only when the record is unowned; removes only when the S1 keeps another link. Env BASE_DIR, USIN_DIR, FR_DIR,
OUT_DIR.
"""
import os, shutil
from collections import Counter
from pathlib import Path


def load_rows(f):
    rows, d = [], {}
    for i, l in enumerate(open(f, encoding='utf-8')):
        if i:
            s, m = (l.rstrip('\n').split('\t') + [''])[:2]; rows.append(s); d[s] = [x for x in m.split(',') if x]
    return rows, d


base = Path(os.environ['BASE_DIR'])
order, u = load_rows(base / 'matching_results.tsv')
owner = {t: s for s, ts in u.items() for t in ts}
R = Counter(); adds, drops = [], []
if os.environ.get('USIN_DIR'):
    U = Path(os.environ['USIN_DIR'])
    _, rep = load_rows(U / 'repro_check/matching_results.tsv'); _, dec = load_rows(U / 'decided/matching_results.tsv')
    cty = {}
    for i, l in enumerate(open('/home/scai/mtech/aib262015/scratch/amazon_ml_2026/data/dataset/test/test_source1.tsv', encoding='utf-8')):
        if i: q = l.rstrip('\n').split('\t'); cty[q[0]] = q[3]
    for s in rep:
        if cty.get(s) == 'France': continue
        a, b = set(dec[s]), set(rep[s])
        adds += [(s, t, 'usin') for t in a - b]; drops += [(s, t, 'usin') for t in b - a]
if os.environ.get('FR_DIR'):
    F = Path(os.environ['FR_DIR'])
    _, fr_new = load_rows(F / 'matching_results.tsv'); _, fr_old = load_rows(Path(os.environ['FR_BASE']) / 'matching_results.tsv')
    for s in fr_new:
        a, b = set(fr_new[s]), set(fr_old[s])
        adds += [(s, t, 'fr') for t in a - b]; drops += [(s, t, 'fr') for t in b - a]
for s, t, tag in drops:
    if t in u[s] and len(u[s]) >= 2:
        u[s].remove(t); owner.pop(t, None); R[tag + '_drop'] += 1
    else:
        R[tag + '_drop_skipped'] += 1
for s, t, tag in adds:
    if t not in owner:
        u[s].append(t); owner[t] = s; R[tag + '_add'] += 1
    else:
        R[tag + '_add_skipped'] += 1
out = Path(os.environ['OUT_DIR']); out.mkdir(parents=True, exist_ok=True)
with open(out / 'matching_results.tsv', 'w', encoding='utf-8') as fo:
    fo.write('source1_entity_id\tmatched_entity_ids\n')
    for s in order:
        fo.write(s + '\t' + ','.join(u[s]) + '\n')
shutil.copy(base / 'candidate_pairs.tsv', out / 'candidate_pairs.tsv')
print(dict(R), 'records in >1 row', sum(v > 1 for v in Counter(t for ts in u.values() for t in ts).values()), flush=True)
