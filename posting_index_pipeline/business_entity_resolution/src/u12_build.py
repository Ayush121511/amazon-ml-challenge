"""U12 = U11 + France invented single-word records at an S1's exact address key (only S1 there), unlinked in U11, in
candidates, with at most 2 invented-name records at that address. Train (full GT): 1 invented record at the unique own
address 99.3% true (38,826), 2 records 97.1-98.0%, 3+ 26-71% (excluded)."""
import os, sys, random
from collections import Counter, defaultdict
sys.path.insert(0, '.')
import pattern_audit as pa
from inv2 import vocab_of
from inv_struct import inv_word
def load_rows(f):
    rows, d = [], {}
    for i, l in enumerate(open(f, encoding='utf-8')):
        if i:
            s, m = (l.rstrip('\n').split('\t') + [''])[:2]; rows.append(s); d[s] = [x for x in m.split(',') if x]
    return rows, d
order, u = load_rows('U11/matching_results.tsv'); _, cand = load_rows('U11/candidate_pairs.tsv')
vte = vocab_of('test')
s1, own, street, names = pa.build_index('test')
owner = {t: s for s, ts in u.items() for t in ts}
at = defaultdict(list); text = {}
for k in (2, 3):
    f = open(pa.D + f'test/test_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
    for l in f:
        q = l.rstrip('\n').split('\t'); ra = q[a] if len(q) > a else ''
        if not ra.strip() or q[c] != 'France': continue
        num, words = pa.akey(ra); hit = own.get((q[c], num, words)) if num and words else None
        if hit and len(hit) == 1 and inv_word(s1[hit[0]][0], q[n], q[c], vte):
            at[hit[0]].append(q[ie]); text[q[ie]] = (q[n], ra)
R = Counter(); adds = []
for s, lst in at.items():
    for r in lst:
        st = 'linked' if owner.get(r) == s else ('other_S1' if r in owner else ('unl_cand' if r in set(cand[s]) else 'unl_new'))
        R[(min(len(lst), 3), st)] += 1
        if st == 'unl_cand' and len(lst) <= 2:
            adds.append((s, r))
for k in sorted(R): print(k, R[k])
for s, r in adds: u[s].append(r)
os.makedirs('U12', exist_ok=True)
with open('U12/matching_results.tsv', 'w', encoding='utf-8') as fo:
    fo.write('source1_entity_id\tmatched_entity_ids\n')
    for s in order: fo.write(s + '\t' + ','.join(u[s]) + '\n')
with open('U12/added_vs_U11.tsv', 'w') as fo:
    fo.write('source1_entity_id\tentity_id\n'); fo.write(''.join(f'{s}\t{r}\n' for s, r in adds))
print('added', len(adds), 'revived', sum(1 for s in set(a for a, _ in adds) if len(u[s]) == sum(1 for x, _ in adds if x == s)),
      'records in >1 row', sum(v > 1 for v in Counter(t for ts in u.values() for t in ts).values()))
random.seed(11)
for s, r in random.sample(adds, min(30, len(adds))):
    others = [x for x in u[s] if x != r][:3]
    print(f'   {s1[s][0]} | {s1[s][1][:36]}  =>  {text[r][0]} | {text[r][1][:36]}   [n inv at addr {len(at[s])}]')
