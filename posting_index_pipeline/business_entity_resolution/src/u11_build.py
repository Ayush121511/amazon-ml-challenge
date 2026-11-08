"""U11 = U10 + France S2/S3 records at an S1's exact address key (unique S1 there) named <invented word> + <alias suffix>
('sys', 'labs', 'one', 'com', 'c0m', 'and'): train 98.6-99.7% true at a unique own address; 'services'/'center' suffixes
are decoys (12-16%) and excluded. Record must be unowned in U10; pairs missing from candidates are appended (rule channel)."""
import os, sys, random
from collections import Counter
sys.path.insert(0, '.')
import pattern_audit as pa
from inv2 import vocab_of, classify
GOOD = {'sys', 'labs', 'one', 'com', 'c0m', 'and'}
def load_rows(f):
    rows, d = [], {}
    for i, l in enumerate(open(f, encoding='utf-8')):
        if i:
            s, m = (l.rstrip('\n').split('\t') + [''])[:2]; rows.append(s); d[s] = [x for x in m.split(',') if x]
    return rows, d
order, u = load_rows('U10/matching_results.tsv'); _, cand = load_rows('U10/candidate_pairs.tsv')
vte = vocab_of('test')
s1, own, street, names = pa.build_index('test')
owner = {t: s for s, ts in u.items() for t in ts}
R = Counter(); adds = []
for k in (2, 3):
    f = open(pa.D + f'test/test_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
    for l in f:
        q = l.rstrip('\n').split('\t'); rid, rn, ra, cc = q[ie], q[n], (q[a] if len(q) > a else ''), q[c]
        if not ra.strip(): continue
        num, words = pa.akey(ra); hit = own.get((cc, num, words)) if num and words else None
        if not hit or len(hit) != 1: continue
        suf = classify(s1[hit[0]][0], rn, cc, vte)
        if not suf or suf not in GOOD: continue
        st = 'linked' if owner.get(rid) == hit[0] else ('other_S1' if rid in owner else ('unl_cand' if rid in set(cand[hit[0]]) else 'unl_new'))
        R[(cc, suf, st)] += 1
        if cc == 'France' and st in ('unl_cand', 'unl_new'):
            adds.append((hit[0], rid, st, rn, ra))
for k_ in sorted(R): print(k_, R[k_])
for s, rid, st, rn, ra in adds:
    u[s].append(rid)
    if st == 'unl_new': cand[s].append(rid)
os.makedirs('U11', exist_ok=True)
for fn, d, col in (('matching_results.tsv', u, 'matched_entity_ids'), ('candidate_pairs.tsv', cand, 'candidate_entity_ids')):
    with open('U11/' + fn, 'w', encoding='utf-8') as fo:
        fo.write(f'source1_entity_id\t{col}\n')
        for s in order: fo.write(s + '\t' + ','.join(d[s]) + '\n')
with open('U11/added_vs_U10.tsv', 'w') as fo:
    fo.write('source1_entity_id\tentity_id\tstatus\n'); fo.write(''.join(f'{s}\t{r}\t{st}\n' for s, r, st, _, _ in adds))
print('France added', len(adds), Counter(a[2] for a in adds), '| records in >1 row', sum(v > 1 for v in Counter(t for ts in u.values() for t in ts).values()))
random.seed(2)
for s, rid, st, rn, ra in random.sample(adds, min(15, len(adds))):
    print(f'   [{st}] {s1[s][0]} | {s1[s][1][:38]}  =>  {rn} | {ra[:38]}')
