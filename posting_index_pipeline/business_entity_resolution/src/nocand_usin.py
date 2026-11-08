import sys, os
from collections import Counter
sys.path.insert(0, '.')
import pattern_audit as pa
def load_rows(f):
    rows, d = [], {}
    for i, l in enumerate(open(f, encoding='utf-8')):
        if i:
            s, m = (l.rstrip('\n').split('\t') + [''])[:2]; rows.append(s); d[s] = [x for x in m.split(',') if x]
    return rows, d
order, u = load_rows('U15b/matching_results.tsv'); _, cand = load_rows('U15b/candidate_pairs.tsv')
owner = {t: s for s, ts in u.items() for t in ts}; cs = {s: set(v) for s, v in cand.items()}
OWN_OK = {'exact', 'same_core', 'dropped_word', 'initials', 'handle'}
NONUM_OK = {'exact', 'same_core', 'dropped_word', 'initials'}
s1, own, street, names = pa.build_index('test')
R = Counter(); adds = []
for k in (2, 3):
    f = open(pa.D + f'test/test_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
    for l in f:
        q = l.rstrip('\n').split('\t'); rid, rn, ra, cc = q[ie], q[n], (q[a] if len(q) > a else ''), q[c]
        if cc == 'France' or rid in owner: continue
        tag = None
        if ra.strip():
            num, words = pa.akey(ra)
            hit = own.get((cc, num, words)) if num and words else None
            if hit and len(hit) == 1:
                r = pa.rel(s1[hit[0]][0], rn)[0]
                if r in OWN_OK and rid not in cs[hit[0]]: tag = ('own_' + r, hit[0])
            elif not num and words:
                h2 = street.get((cc, words))
                if h2 and len(h2) == 1:
                    r = pa.rel(s1[h2[0]][0], rn)[0]
                    if r in NONUM_OK and rid not in cs[h2[0]]: tag = ('nonum_' + r, h2[0])
        else:
            hit = names.get((cc, pa.nrm(rn)))
            if hit and len(hit) == 1 and rid not in cs[hit[0]]: tag = ('blank_unique_exact', hit[0])
        if tag:
            R[(cc, tag[0])] += 1; adds.append((tag[1], rid, cc + ':' + tag[0])); owner[rid] = tag[1]
for k_ in sorted(R): print(k_, R[k_])
for s, rid, tg in adds:
    u[s].append(rid); cand[s].append(rid)
os.makedirs('U15c', exist_ok=True)
for fn, d, col in (('matching_results.tsv', u, 'matched_entity_ids'), ('candidate_pairs.tsv', cand, 'candidate_entity_ids')):
    with open('U15c/' + fn, 'w', encoding='utf-8') as fo:
        fo.write(f'source1_entity_id\t{col}\n')
        for s in order: fo.write(s + '\t' + ','.join(d[s]) + '\n')
open('U15c/added_vs_U15b.tsv', 'w').write('source1_entity_id\tentity_id\trule\n' + ''.join(f'{a}\t{b}\t{c}\n' for a, b, c in adds))
print('added', len(adds), 'dups', sum(v > 1 for v in Counter(t for ts in u.values() for t in ts).values()))
