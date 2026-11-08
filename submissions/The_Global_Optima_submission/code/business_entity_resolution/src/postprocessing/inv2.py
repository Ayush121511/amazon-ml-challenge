import sys, pickle
from collections import Counter, defaultdict
sys.path.insert(0, '.')
import pattern_audit as pa
def vocab_of(split):
    v = Counter()
    f = open(pa.D + f'{split}/{split}_source1.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    n, c = h.index('business_name'), h.index('country')
    for l in f:
        q = l.rstrip('\n').split('\t')
        for t in set(pa.toks(q[n])): v[(q[c], t)] += 1
    return v
def classify(sn, rn, cc, vocab):
    tk = [x for x in pa.toks(rn) if x not in pa.LEGAL]
    if len(tk) != 2 or set(tk) & set(pa.core(sn)):
        return None
    inv = [x for x in tk if x.isalpha() and len(x) >= 5 and vocab[(cc, x)] == 0]
    if len(inv) != 1:
        return None
    other = [x for x in tk if x != inv[0]]
    return other[0] if other else None
if __name__ == '__main__':
    # ---- train
    owner = {}
    f = open(pa.D + 'train/train_ground_truth.tsv'); next(f)
    for l in f:
        q = l.rstrip('\n').split('\t')
        for t in (q[1].split(',') if len(q) > 1 else []):
            if t.strip(): owner[t.strip()] = q[0]
    vtr = vocab_of('train')
    s1, own, street, names = pa.build_index('train')
    TR = defaultdict(Counter)
    for k in (2, 3):
        f = open(pa.D + f'train/train_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
        ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
        for l in f:
            q = l.rstrip('\n').split('\t'); ra = q[a] if len(q) > a else ''
            if not ra.strip(): continue
            num, words = pa.akey(ra); hit = own.get((q[c], num, words)) if num and words else None
            if not hit or len(hit) != 1: continue
            suf = classify(s1[hit[0]][0], q[n], q[c], vtr)
            if suf: TR[suf]['true' if owner.get(q[ie]) == hit[0] else 'false'] += 1
    print('TRAIN invented+suffix at unique own address, by suffix (n>=30):')
    good = []
    for suf, c in sorted(TR.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        if n >= 30:
            r = c['true'] / n; print(f'   {suf:14s} n={n:6d} true={r:.3f}')
            if r >= 0.95: good.append(suf)
    tot = sum(sum(c.values()) for c in TR.values()); tt = sum(c['true'] for c in TR.values())
    print('all suffixes', tot, 'true', round(tt / max(tot, 1), 3), '| accepted suffixes', good)
    pickle.dump(good, open('inv2_good.pkl', 'wb'))
