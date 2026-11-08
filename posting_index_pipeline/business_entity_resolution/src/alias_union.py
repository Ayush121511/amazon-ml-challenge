"""Tapan-style alias union: an invented-name record (one invented word, optionally + sys/labs/one/com suffix) is linked to
the S1 whose address set (S1 address + addresses of its accepted copies) is the ONLY one containing the record's full
normalized address. Train: validated with ground-truth copies. Test: accepted copies = U12 links; new links vs U12."""
import re, sys, random
from collections import Counter, defaultdict
sys.path.insert(0, '.')
import pattern_audit as pa
from inv2 import vocab_of
SUF = {'sys', 'labs', 'one', 'com', 'c0m'}
def fa(a): return ' '.join(pa.toks(a))
def is_alias(rn, sn, cc, vocab):
    tk = [x for x in pa.toks(rn) if x not in pa.LEGAL]
    if not tk or len(tk) > 2 or set(tk) & set(pa.core(sn)): return False
    inv = [x for x in tk if x.isalpha() and len(x) >= 5 and vocab[(cc, x)] == 0]
    return len(inv) == 1 and (len(tk) == 1 or [x for x in tk if x != inv[0]][0] in SUF)
def run(split, links, truth=None):
    vocab = vocab_of(split)
    s1 = {}
    f = open(pa.D + f'{split}/{split}_source1.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
    for l in f:
        q = l.rstrip('\n').split('\t'); s1[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c])
    rec = {}
    for k in (2, 3):
        f = open(pa.D + f'{split}/{split}_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
        ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
        for l in f:
            q = l.rstrip('\n').split('\t'); rec[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c])
    addr = defaultdict(set)
    for s, (n_, a_, c_) in s1.items():
        if a_.strip(): addr[(c_, fa(a_))].add(s)
    for s, ts in links.items():
        for t in ts:
            ra = rec.get(t, ('', '', ''))[1]
            if ra.strip(): addr[(s1[s][2], fa(ra))].add(s)
    owner = {t: s for s, ts in links.items() for t in ts}
    R = Counter(); new = []
    for t, (rn, ra, cc) in rec.items():
        if not ra.strip() or t in owner: continue
        hit = addr.get((cc, fa(ra)))
        if not hit or len(hit) != 1: continue
        s = next(iter(hit))
        if not is_alias(rn, s1[s][0], cc, vocab): continue
        if truth is not None:
            R[(cc, truth.get(t) == s)] += 1
        else:
            R[cc] += 1; new.append((s, t))
    return R, new, s1, rec
if __name__ == '__main__':
    gt = defaultdict(list); tr_owner = {}
    f = open(pa.D + 'train/train_ground_truth.tsv'); next(f)
    for l in f:
        q = l.rstrip('\n').split('\t')
        for t in (q[1].split(',') if len(q) > 1 else []):
            if t.strip(): gt[q[0]].append(t.strip()); tr_owner[t.strip()] = q[0]
    # train: pretend 60% of true copies are "accepted", test the rule on the unaccepted aliases
    random.seed(0)
    acc = {s: [t for t in ts if random.random() < 0.95] for s, ts in gt.items()}
    R, _, _, _ = run('train', acc, truth=tr_owner)
    for cc in ('US', 'India'):
        t, fl = R[(cc, True)], R[(cc, False)]; print(f'TRAIN {cc}: flagged {t + fl}, true {t / max(1, t + fl):.4f}')
