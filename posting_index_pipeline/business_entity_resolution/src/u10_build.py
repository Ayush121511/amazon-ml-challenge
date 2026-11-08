"""U10 = U9 + small label-backed additions (adds only):
 1. all countries: records whose normalized name equals the S1's exactly, at the S1's own address key (unique S1) or on its
    street (unique S1) at any other number, unlinked, already in candidates. Train: 96.5-100% true (decoys change the name).
 2. France: blank-address record whose exact name belongs to one France S1 only, accepted by the T6 bag (US/India: blank
    unique-name pairs T6 accepts are 98.1% true).
 3. France: own-address copy recipes (invented1, handle, same_core, dropped_word, initials) accepted by the T6 bag
    (US/India: 100% true where T6 accepts).
 4. France: invented1 at the own address present only in the rank 11-20 rescue pool (no stage-3b score; unconditional
    97.5% true).
Record must be unowned in U9."""
import os, sys, random
from collections import Counter
sys.path.insert(0, '.')
import pattern_audit as pa, re as _re
_src = open('/Users/ayushsrivastava/Documents/AmazonMLChallenge/posting_index_pipeline/business_entity_resolution/src/france_census.py').read()
TYPES = set(_re.findall(r"'([^']+)'", _re.search(r"TYPES = \{(.*?)\}", _src, _re.S).group(1)))
FILLER = set(_re.findall(r"'([^']+)'", _re.search(r"FILLER = \{(.*?)\}", _src, _re.S).group(1)))
def load_rows(f):
    rows, d = [], {}
    for i, l in enumerate(open(f, encoding='utf-8')):
        if i:
            s, m = (l.rstrip('\n').split('\t') + [''])[:2]; rows.append(s); d[s] = [x for x in m.split(',') if x]
    return rows, d
order, u9 = load_rows('U9/matching_results.tsv'); _, c9 = load_rows('U9/candidate_pairs.tsv')
_, bag = load_rows('t6bag_matching.tsv')
_, top10 = load_rows('/Users/ayushsrivastava/Documents/AmazonMLChallenge/submissions/U2_T6_FRw4_xprune/candidate_pairs.tsv')
bagset = {(s, t) for s, ts in bag.items() for t in ts}
s1, own, street, names = pa.build_index('test')
owner = {t: s for s, ts in u9.items() for t in ts}
vocab = Counter()
for e, (nm, ad, cc0) in s1.items():
    for t in set(pa.toks(nm)): vocab[(cc0, t)] += 1
c9s = {s: set(v) for s, v in c9.items()}; t10 = {s: set(v) for s, v in top10.items()}
R = Counter(); adds = []; ex = {}
for k in (2, 3):
    f = open(pa.D + f'test/test_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
    for l in f:
        q = l.rstrip('\n').split('\t'); rid, rn, ra, cc = q[ie], q[n], (q[a] if len(q) > a else ''), q[c]
        if rid in owner:
            continue
        rule = None
        if ra.strip():
            num, words = pa.akey(ra); hit = own.get((cc, num, words)) if num and words else None
            if hit and len(hit) == 1:
                s = hit[0]; r, _ = pa.rel(s1[s][0], rn)
                tk = [x for x in pa.toks(rn) if x not in pa.LEGAL]
                inv_ok = r == 'invented1' and len(tk) == 1 and len(tk[0]) >= 3 and vocab[('France', tk[0])] == 0
                if rid in c9s.get(s, ()) and cc == 'France':
                    if (inv_ok or r in ('same_core', 'dropped_word')) and (s, rid) in bagset:
                        rule = '3_fr_own_bag_' + r
                    elif inv_ok and rid not in t10.get(s, ()):
                        rule = '4_fr_alias_rescue_pool'
                    elif r == 'swapped_word':
                        a_, b_ = sorted(set(pa.core(s1[s][0])) - set(pa.core(rn)))[0], sorted(set(pa.core(rn)) - set(pa.core(s1[s][0])))[0]
                        if (a_ in TYPES or a_ in FILLER) and b_ in FILLER:
                            rule = '5_fr_type_to_filler'
        else:
            hit = names.get((cc, pa.nrm(rn)))
            if hit and len(hit) == 1 and cc == 'France':
                s = hit[0]
                if rid in c9s.get(s, ()) and (s, rid) in bagset:
                    rule = '2_fr_blank_unique_bag'
        if rule:
            R[(cc, rule)] += 1; adds.append((s, rid, cc, rule)); owner[rid] = s
            ex.setdefault(rule, []).append((s, rid, rn, ra))
for k, v in sorted(R.items()):
    print(k, v)
for s, rid, cc, rule in adds:
    u9[s].append(rid)
os.makedirs('U10', exist_ok=True)
with open('U10/matching_results.tsv', 'w', encoding='utf-8') as fo:
    fo.write('source1_entity_id\tmatched_entity_ids\n')
    for s in order:
        fo.write(s + '\t' + ','.join(u9[s]) + '\n')
with open('U10/added_vs_U9.tsv', 'w') as fo:
    fo.write('source1_entity_id\tentity_id\tcountry\trule\n'); fo.write(''.join(f'{a}\t{b}\t{c}\t{d}\n' for a, b, c, d in adds))
print('total added', len(adds), 'records in >1 row', sum(v > 1 for v in Counter(t for ts in u9.values() for t in ts).values()))
random.seed(4)
for rule, lst in ex.items():
    print('==', rule)
    for s, rid, rn, ra in random.sample(lst, min(5, len(lst))):
        print(f'    {s1[s][0]} | {s1[s][1][:40]}  =>  {rn} | {ra[:40] or "(blank)"}')
