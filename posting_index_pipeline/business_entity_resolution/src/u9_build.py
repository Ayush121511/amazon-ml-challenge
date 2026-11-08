"""U9 = U8 + rule channel: S2/S3 records at an S1's exact address key (unique S1 there) whose name is an invented single
word or the S1's initials, not linked to anyone in U8. Train (full GT): invented1 97.5%, initials 100% true at a unique own
address. Pairs not yet in candidate_pairs.tsv are appended to it (rule-based retrieval channel)."""
import os, sys
from collections import Counter
sys.path.insert(0, '.')
import pattern_audit as pa
def load_rows(f):
    rows, d = [], {}
    for i, l in enumerate(open(f, encoding='utf-8')):
        if i:
            s, m = (l.rstrip('\n').split('\t') + [''])[:2]; rows.append(s); d[s] = [x for x in m.split(',') if x]
    return rows, d
order, u8 = load_rows('u8_matching.tsv')
_, cand = load_rows('/Users/ayushsrivastava/Documents/AmazonMLChallenge/submissions/final_u5/candidate_pairs.tsv')
s1, own, street, names = pa.build_index('test')
owner = {t: s for s, ts in u8.items() for t in ts}
R = Counter(); adds = []
for k in (2, 3):
    f = open(pa.D + f'test/test_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
    for l in f:
        q = l.rstrip('\n').split('\t'); rid, rn, ra, cc = q[ie], q[n], (q[a] if len(q) > a else ''), q[c]
        if not ra.strip() or rid in owner:
            continue
        num, words = pa.akey(ra); hit = own.get((cc, num, words)) if num and words else None
        if not hit or len(hit) != 1:
            continue
        r, _ = pa.rel(s1[hit[0]][0], rn)
        if r in ('invented1', 'initials'):
            s = hit[0]; inc = rid in set(cand.get(s, []))
            R[(cc, r, 'in_cand' if inc else 'new_cand')] += 1
            adds.append((s, rid, inc, cc, r))
print(dict(sorted(R.items())))
keep = [x for x in adds if x[3] == 'France' and not (x[4] == 'invented1' and x[2])]   # France only; in-candidate invented1 were already judged by T6/U8
print('adding', len(keep), Counter((x[3], x[4], x[2]) for x in keep))
for s, rid, inc, cc, r in keep:
    u8[s].append(rid)
    if not inc:
        cand[s].append(rid)
os.makedirs('U9', exist_ok=True)
with open('U9/matching_results.tsv', 'w', encoding='utf-8') as fo:
    fo.write('source1_entity_id\tmatched_entity_ids\n')
    for s in order:
        fo.write(s + '\t' + ','.join(u8[s]) + '\n')
with open('U9/candidate_pairs.tsv', 'w', encoding='utf-8') as fo:
    fo.write('source1_entity_id\tcandidate_entity_ids\n')
    for s in order:
        fo.write(s + '\t' + ','.join(cand[s]) + '\n')
with open('U9/rule_channel_links.tsv', 'w', encoding='utf-8') as fo:
    fo.write('source1_entity_id\tentity_id\tcountry\tpattern\twas_in_candidates\n')
    for s, rid, inc, cc, r in keep:
        fo.write(f'{s}\t{rid}\t{cc}\t{r}\t{int(inc)}\n')
own_ct = Counter(t for ts in u8.values() for t in ts)
print('records in >1 row', sum(v > 1 for v in own_ct.values()))
import random; random.seed(3)
rec = {}
need = {x[1] for x in keep}
for k in (2, 3):
    f = open(pa.D + f'test/test_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    ie, n, a = (h.index(x) for x in ('entity_id', 'business_name', 'business_address'))
    for l in f:
        q = l.rstrip('\n').split('\t')
        if q[ie] in need: rec[q[ie]] = (q[n], q[a] if len(q) > a else '')
for s, rid, inc, cc, r in random.sample(keep, min(20, len(keep))):
    print(f'  [{cc} {r} {"cand" if inc else "NEW"}] {s1[s][0]} | {s1[s][1][:40]}  =>  {rec[rid][0]} | {rec[rid][1][:40]}')
