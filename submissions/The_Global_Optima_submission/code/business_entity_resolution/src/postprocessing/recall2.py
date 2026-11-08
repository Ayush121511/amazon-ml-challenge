"""Two recall rules measured on full train labels, then counted on test France (unlinked in U13):
 N: record without a house number whose street-word key matches exactly one S1 of the country; name relation.
 K: blank-address record whose CORE name (legal forms / stop words removed, sorted) matches exactly one S1 core name."""
import sys, pickle
from collections import Counter, defaultdict
sys.path.insert(0, '.')
import pattern_audit as pa
def ckey(n): return ' '.join(sorted(set(pa.core(n))))
def scan(split, emit):
    s1 = {}; street = defaultdict(list); cores = defaultdict(list)
    f = open(pa.D + f'{split}/{split}_source1.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
    for l in f:
        q = l.rstrip('\n').split('\t'); ad = q[a] if len(q) > a else ''; s1[q[ie]] = (q[n], ad, q[c])
        num, words = pa.akey(ad)
        if words: street[(q[c], words)].append(q[ie])
        k = ckey(q[n])
        if k: cores[(q[c], k)].append(q[ie])
    for k in (2, 3):
        f = open(pa.D + f'{split}/{split}_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
        ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
        for l in f:
            q = l.rstrip('\n').split('\t'); rid, rn, ra, cc = q[ie], q[n], (q[a] if len(q) > a else ''), q[c]
            if ra.strip():
                num, words = pa.akey(ra)
                if num or not words: continue
                hit = street.get((cc, words))
                if hit and len(hit) == 1:
                    emit('N_nonum_street', pa.rel(s1[hit[0]][0], rn)[0], cc, hit[0], rid)
            else:
                k_ = ckey(rn); hit = cores.get((cc, k_)) if k_ else None
                if hit and len(hit) == 1:
                    emit('K_blank_unique_core', 'exact' if pa.nrm(rn) == pa.nrm(s1[hit[0]][0]) else 'core_only', cc, hit[0], rid)
if sys.argv[1] == 'train':
    owner = {}
    f = open(pa.D + 'train/train_ground_truth.tsv'); next(f)
    for l in f:
        q = l.rstrip('\n').split('\t')
        for t in (q[1].split(',') if len(q) > 1 else []):
            if t.strip(): owner[t.strip()] = q[0]
    C = Counter()
    scan('train', lambda fam, r, cc, s, rid: C.__setitem__((fam, r, owner.get(rid) == s), C[(fam, r, owner.get(rid) == s)] + 1))
    for fam, r in sorted({k[:2] for k in C}):
        t, fl = C[(fam, r, True)], C[(fam, r, False)]
        if t + fl >= 50: print(f'TRAIN {fam:22s} {r:18s} n={t + fl:7d} true={t / (t + fl):.3f}')
else:
    def load(fp):
        d = {}
        for i, l in enumerate(open(fp, encoding='utf-8')):
            if i: s, m = (l.rstrip('\n').split('\t') + [''])[:2]; d[s] = [x for x in m.split(',') if x]
        return d
    u = load('U13/matching_results.tsv'); cand = load('U13/candidate_pairs.tsv'); bag = load('t6bag_matching.tsv')
    owner = {t: s for s, ts in u.items() for t in ts}; bagset = {(s, t) for s, ts in bag.items() for t in ts}
    C = Counter(); pairs = defaultdict(list)
    def emit(fam, r, cc, s, rid):
        if cc != 'France': return
        st = 'linked' if owner.get(rid) == s else ('other' if rid in owner else ('unl_cand' if rid in set(cand[s]) else 'unl_new'))
        C[(fam, r, st, (s, rid) in bagset)] += 1
        if st.startswith('unl'): pairs[(fam, r)].append((s, rid, st, (s, rid) in bagset))
    scan('test', emit)
    for k in sorted(C): print('TEST France', k, C[k])
    pickle.dump(dict(pairs), open('recall2_pairs.pkl', 'wb'))
