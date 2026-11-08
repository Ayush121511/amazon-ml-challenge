"""Pre-upload audit of Vishal's U5 against U2: per-country link diff, exclusivity (a record in at most one S1 row),
candidate superset, France rows = U2 France + additions only, and what the rescue links look like (name / address
relation to the S1, house-number shift class, S1 name multiplicity), with examples. Env U5 (dir with the two TSVs).
"""
import os
import random
from collections import Counter

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
from error_sweep import mkey, shift_class
from france_census import load
from usin_errors import rel

H = '/home/scai/mtech/aib262015/scratch/amazon_ml_2026'
U5 = os.environ['U5']
BASE = os.environ.get('BASE', f'{H}/artifacts/submission_U2_final')
u5, u2 = load(f'{U5}/matching_results.tsv'), load(f'{BASE}/matching_results.tsv')
c5, c2 = load(f'{U5}/candidate_pairs.tsv'), load(f'{BASE}/candidate_pairs.tsv')
text, cty, s1 = {}, {}, []
for k in (1, 2, 3):
    with open(f'{H}/data/dataset/test/test_source{k}.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
        for line in fh:
            q = line.rstrip('\n').split('\t')
            text[q[ie]] = (q[n], q[a] if len(q) > a else '')
            if k == 1:
                cty[q[ie]] = q[c]; s1.append(q[ie])
mult = Counter((cty[r], mkey(text[r][0])) for r in s1)
print('rows', len(u5), len(u2), 'same S1 set', set(u5) == set(u2))
own = Counter(t for ts in u5.values() for t in ts)
print('records in >1 S1 row (U5):', sum(v > 1 for v in own.values()), '| U2:', sum(v > 1 for v in Counter(t for ts in u2.values() for t in ts).values()))
print('matches outside own candidates:', sum(len(set(u5[r]) - set(c5.get(r, []))) for r in u5))
print('U2 candidates not in U5 candidates:', sum(len(set(c2[r]) - set(c5.get(r, []))) for r in c2),
      '| extra U5 candidates:', sum(len(set(c5[r]) - set(c2.get(r, []))) for r in c5))
diff = Counter(); rescue = []; emptied = Counter()
for r in u5:
    a, b = set(u5[r]), set(u2[r])
    diff[(cty[r], 'added')] += len(a - b); diff[(cty[r], 'removed')] += len(b - a)
    if b and not a:
        emptied[cty[r]] += 1
    for t in a - b:
        if t not in set(c2.get(r, [])):
            rescue.append((r, t))
print('diff', dict(diff), 'emptied', dict(emptied))
print('France: U2 France rows subset of U5:', all(set(u2[r]) <= set(u5[r]) for r in u5 if cty[r] == 'France'))
rng = random.Random(0)
for C in ('France', 'India', 'US'):
    R = [(r, t) for r, t in rescue if cty[r] == C]
    A = Counter(); S = Counter(); M = Counter(); acc_same = Counter()
    ex = []
    for r, t in R:
        rn, ra = text[r]; tn, ta = text.get(t, ('', ''))
        nm, ad = rel(rn, ra, tn, ta)
        A[(nm, ad)] += 1; S[shift_class(ra, ta) if ta.strip() else 'blank'] += 1
        m = mult[(cty[r], mkey(rn))]; M['1' if m <= 1 else ('2-5' if m <= 5 else '6+')] += 1
        others = [x for x in u5[r] if x != t]
        acc_same[len(others)] += 1
        ex.append(f"{rn} | {ra}  =>  {tn} | {ta}   [other links {len(others)}: " + '; '.join(f'{text.get(x, ("", ""))[0]} | {text.get(x, ("", ""))[1]}' for x in others[:3]) + ']')
    print(f'\n== {C} rescue links {len(R)}')
    print('  name x addr:', dict(A.most_common(12)))
    print('  number shift:', dict(S)); print('  S1 name multiplicity:', dict(M)); print('  other links of the S1:', dict(sorted(Counter(min(k, 6) for k in acc_same.elements()).items())))
    for e in rng.sample(ex, min(25 if C == 'France' else 10, len(ex))):
        print('   ', e[:300])
