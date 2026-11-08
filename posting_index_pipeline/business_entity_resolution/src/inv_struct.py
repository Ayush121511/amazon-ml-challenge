import sys
from collections import Counter, defaultdict
sys.path.insert(0, '.')
import pattern_audit as pa
from inv2 import vocab_of
def inv_word(sn, rn, cc, vocab):
    tk = [x for x in pa.toks(rn) if x not in pa.LEGAL]
    return len(tk) == 1 and tk[0].isalpha() and len(tk[0]) >= 3 and vocab[(cc, tk[0])] == 0 and not (set(tk) & set(pa.core(sn)))
def scan(split, vocab):
    s1, own, street, names = pa.build_index(split)
    at = defaultdict(list)   # S1 -> invented records at its unique own address
    for k in (2, 3):
        f = open(pa.D + f'{split}/{split}_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
        ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
        for l in f:
            q = l.rstrip('\n').split('\t'); ra = q[a] if len(q) > a else ''
            if not ra.strip(): continue
            num, words = pa.akey(ra); hit = own.get((q[c], num, words)) if num and words else None
            if hit and len(hit) == 1 and inv_word(s1[hit[0]][0], q[n], q[c], vocab):
                at[hit[0]].append((q[ie], k))
    return s1, at
# ---- train
owner = {}
f = open(pa.D + 'train/train_ground_truth.tsv'); next(f)
gold_n = Counter()
for l in f:
    q = l.rstrip('\n').split('\t')
    for t in (q[1].split(',') if len(q) > 1 else []):
        if t.strip(): owner[t.strip()] = q[0]; gold_n[q[0]] += 1
s1, at = scan('train', vocab_of('train'))
per = Counter(); R = Counter()
for s, lst in at.items():
    tr = [r for r, k in lst if owner.get(r) == s]
    per[(len(lst), len(tr))] += 1
    for r, k in lst:
        R[('n_invented_at_address=%d' % min(len(lst), 3), 'same_source_twin' if sum(1 for _, kk in lst if kk == k) > 1 else 'one_per_source', owner.get(r) == s)] += 1
print('TRAIN: S1s with invented records at their unique own address: (n records, n true) ->', dict(sorted(per.items())[:12]))
for key in sorted({k[:2] for k in R}):
    t, f_ = R[key + (True,)], R[key + (False,)]
    print('   ', key, f'n={t + f_} true={t / max(1, t + f_):.3f}')
