import sys, os, difflib, re, pickle
from collections import Counter, defaultdict
os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
sys.path.insert(0, '.'); sys.path.insert(0, '/Users/ayushsrivastava/Documents/AmazonMLChallenge/posting_index_pipeline/business_entity_resolution/src')
import pattern_audit as pa
from france_census import street_words, house
LEG = {'sa', 'sarl', 'sas', 'sasu', 'eurl', 'ei', 'snc', 'sci', 'inc', 'llc', 'ltd', 'corp', 'co', 'lp', 'llp', 'pllc', 'pc', 'pvt', 'private', 'limited', 'incorporated', 'corporation', 'company'}
def near(a, b): return difflib.SequenceMatcher(None, a, b).ratio() >= 0.75
def num(a):
    m = re.match(r'\d+', house(a)[0]); return m.group(0) if m else ''
def classify(sn, sa, rn, ra):
    st, rt = set(pa.toks(sn)), set(pa.toks(rn))
    if set(pa.core(sn)) != set(pa.core(rn)): return None
    ls, lr = st & LEG, rt & LEG
    if not ls or not lr or ls == lr: return None
    w1, w2 = street_words(sa), street_words(ra); n1, n2 = num(sa), num(ra)
    if not w1 or not w2 or not n1 or not n2: return None
    ds = not any(near(x, y) for x in w1 for y in w2); dn = n1 != n2
    return ('diff_street' if ds else 'same_street') + '+' + ('diff_num' if dn else 'same_num')
split = sys.argv[1]
if split == 'train':
    owner = {}
    f = open(pa.D + 'train/train_ground_truth.tsv'); next(f)
    for l in f:
        q = l.rstrip('\n').split('\t')
        for t in (q[1].split(',') if len(q) > 1 else []):
            if t.strip(): owner[t.strip()] = q[0]
    idx = defaultdict(list); s1 = {}
    f = open(pa.D + 'train/train_source1.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
    for l in f:
        q = l.rstrip('\n').split('\t'); ad = q[a] if len(q) > a else ''; s1[q[ie]] = (q[n], ad)
        k = ' '.join(sorted(set(pa.core(q[n])))); c_ = q[c]
        if k: idx[(c_, k)].append(q[ie])
    R = Counter()
    for kk in (2, 3):
        f = open(pa.D + f'train/train_source{kk}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
        ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
        for l in f:
            q = l.rstrip('\n').split('\t'); ra = q[a] if len(q) > a else ''
            if not ra.strip(): continue
            hit = idx.get((q[c], ' '.join(sorted(set(pa.core(q[n]))))))
            if not hit or len(hit) != 1: continue
            s = hit[0]; cl = classify(s1[s][0], s1[s][1], q[n], ra)
            if cl: R[(cl, owner.get(q[ie]) == s)] += 1
    for cl in sorted({k[0] for k in R}):
        t, fl = R[(cl, True)], R[(cl, False)]; print(f'TRAIN legal swap, {cl}: n={t + fl} true={t / max(1, t + fl):.3f}')
else:
    def load(fp):
        d = {}
        for i, l in enumerate(open(fp, encoding='utf-8')):
            if i: s, m = (l.rstrip('\n').split('\t') + [''])[:2]; d[s] = [x for x in m.split(',') if x]
        return d
    u = load('U18/matching_results.tsv'); text = {}; cty = {}
    for k in (1, 2, 3):
        f = open(pa.D + f'test/test_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
        ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
        for l in f:
            q = l.rstrip('\n').split('\t'); text[q[ie]] = (q[n], q[a] if len(q) > a else ''); cty[q[ie]] = q[c]
    R = Counter(); L = []
    for s, ts in u.items():
        for t in ts:
            if not text[t][1].strip(): continue
            cl = classify(text[s][0], text[s][1], text[t][0], text[t][1])
            if cl: R[(cty[s], cl)] += 1; L.append((s, t, cty[s], cl))
    print('TEST accepted links with a legal-form swap:', dict(R))
    pickle.dump(L, open('legalswap_test.pkl', 'wb'))
