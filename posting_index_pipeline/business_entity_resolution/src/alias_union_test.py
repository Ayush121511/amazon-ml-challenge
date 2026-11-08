import sys, random
from collections import Counter
sys.path.insert(0, '.')
from alias_union import run
def load(f):
    d = {}
    for i, l in enumerate(open(f, encoding='utf-8')):
        if i: s, m = (l.rstrip('\n').split('\t') + [''])[:2]; d[s] = [x for x in m.split(',') if x]
    return d
u12 = load('U12/matching_results.tsv'); cand = load('U12/candidate_pairs.tsv')
R, new, s1, rec = run('test', u12)
print('TEST new alias links beyond U12:', dict(R))
cs = Counter(); 
for s, t in new: cs[(s1[s][2], t in set(cand[s]))] += 1
print('  by (country, already in candidates):', dict(cs))
random.seed(3)
fr = [x for x in new if s1[x[0]][2] == 'France']
for s, t in random.sample(fr, min(12, len(fr))):
    print(f'   {s1[s][0]} | {s1[s][1][:34]}  =>  {rec[t][0]} | {rec[t][1][:34]}')
import pickle; pickle.dump(new, open('alias_union_new.pkl', 'wb'))
