"""Pattern audit: P(record belongs to the S1 | model-free pattern) on FULL train labels vs how the current submission
decides the same pattern on test, per country. Patterns:
  A. own address: record's (first house number + street words) key matches exactly one S1 of the country; name relation.
  B. same street, other number: (street words) key matches exactly one S1; number relation x name relation.
  C. blank address: record's normalized full name matches exactly one S1 name of the country (no twin).
Writes pattern_train.tsv / pattern_test.tsv and the test pair list per cell (for building fixes).
Usage: python3 pattern_audit.py SUB_MATCHING CAND_TSV
"""
import difflib
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict

D = '/Users/ayushsrivastava/Documents/AmazonMLChallenge/data/student_resource/dataset/'
fold = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s or '') if not unicodedata.combining(c)).lower()
toks = lambda s: re.sub(r'[^a-z0-9]+', ' ', fold(s)).split()
LEGAL = {'inc', 'llc', 'ltd', 'corp', 'co', 'pvt', 'private', 'limited', 'lp', 'llp', 'pc', 'plc', 'pllc', 'incorporated',
         'corporation', 'company', 'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'sa', 'ei', 'scop', 'selarl', 'l', 'c', 's', 'a', 'r', 'u', 'e', 'i'}
STOPW = {'the', 'and', 'of', 'de', 'la', 'le', 'les', 'des', 'du', 'd', 'et', 'en', 'au', 'aux', 'a', 'm', 'mr', 'mrs', 'dr', 'smt', 'shri', 'sri', 'ms'}
FILL = {'services', 'service', 'center', 'centre', 'partners', 'associates', 'associes', 'enterprises', 'cie', 'compagnie',
        'solutions', 'com', 'www', 'dba', 'aka', 'fka'}


def core(n):
    return [t for t in toks(n) if t not in LEGAL and t not in STOPW]


def nrm(n):
    return ' '.join(toks(n))


def akey(a):
    t = toks(a); nums = [x for x in t if x.isdigit()]
    words = ' '.join(sorted(set(x for x in t if not any(ch.isdigit() for ch in x) and len(x) > 2))[:6])
    return (str(int(nums[0])) if nums else None), words


def shift(a, b):
    if a == b:
        return 'same'
    if len(a) != len(b) and (a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a)):
        return 'digit_drop'
    d = int(b) - int(a)
    return ('up_%d' % d) if 0 < d <= 21 else ('down_small' if -21 <= d < 0 else 'far')


def rel(sn, rn):
    if nrm(sn) == nrm(rn):
        return 'exact', ''
    sc, rc = core(sn), core(rn)
    S, R = set(sc), set(rc)
    raw = rn.strip()
    if S and S == R:
        return 'same_core', ''
    if not R:
        return 'only_legal_or_stop', ''
    if raw.startswith('#') or raw.startswith('@') or '.com' in raw.lower() or 'www.' in raw.lower():
        return 'handle', ''
    comp = ''.join(sc)
    if len(R) == 1 and len(sc) >= 2 and list(R)[0] == ''.join(w[0] for w in sc):
        return 'initials', ''
    if len(R) == 1 and comp and list(R)[0] == comp:
        return 'concat', ''
    if not S & R:
        if len(rc) == 1 and rc[0].isalpha():
            return 'invented1', ''
        return 'none_multi', ''
    if R < S:
        return 'dropped_word', ''
    if S < R:
        add = R - S
        if add <= FILL:
            return 'added_filler', '+'.join(sorted(add))
        return 'added_word', '+'.join(sorted(add))[:30]
    r = difflib.SequenceMatcher(None, ' '.join(sorted(S)), ' '.join(sorted(R))).ratio()
    if r >= 0.85:
        return 'typo', ''
    if len(S) == len(R) and len(S - R) == 1:
        return 'swapped_word', f'{sorted(S - R)[0]}>{sorted(R - S)[0]}'[:30]
    return 'partial', ''


def build_index(split):
    s1, own, street, names = {}, defaultdict(list), defaultdict(list), defaultdict(list)
    f = open(D + f'{split}/{split}_source1.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
    ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
    for l in f:
        q = l.rstrip('\n').split('\t'); ad = q[a] if len(q) > a else ''
        s1[q[ie]] = (q[n], ad, q[c])
        num, words = akey(ad)
        if num and words:
            own[(q[c], num, words)].append(q[ie])
        if words:
            street[(q[c], words)].append(q[ie])
        names[(q[c], nrm(q[n]))].append(q[ie])
    return s1, own, street, names


def scan(split, emit):
    s1, own, street, names = build_index(split)
    for k in (2, 3):
        f = open(D + f'{split}/{split}_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
        ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
        for l in f:
            q = l.rstrip('\n').split('\t'); rid, rn, ra, cc = q[ie], q[n], (q[a] if len(q) > a else ''), q[c]
            if ra.strip():
                num, words = akey(ra)
                hit = own.get((cc, num, words)) if num and words else None
                if hit and len(hit) == 1:
                    r, sub = rel(s1[hit[0]][0], rn)
                    emit(cc, 'A_own_address', r, sub, hit[0], rid)
                    continue
                hit = street.get((cc, words)) if words else None
                if hit and len(hit) == 1 and num:
                    sn_, sa_, _ = s1[hit[0]]
                    snum = akey(sa_)[0]
                    if snum:
                        r, sub = rel(sn_, rn)
                        if r in ('exact', 'same_core', 'typo', 'added_filler', 'added_word', 'dropped_word', 'invented1'):
                            emit(cc, 'B_street_' + shift(snum, num), r, '', hit[0], rid)
            else:
                hit = names.get((cc, nrm(rn)))
                if hit and len(hit) == 1:
                    emit(cc, 'C_blank_unique_name', 'exact', '', hit[0], rid)


def main():
    sub, cand_f = sys.argv[1], sys.argv[2]
    owner = {}
    f = open(D + 'train/train_ground_truth.tsv'); next(f)
    for l in f:
        q = l.rstrip('\n').split('\t')
        for t in (q[1].split(',') if len(q) > 1 else []):
            if t.strip():
                owner[t.strip()] = q[0]
    TR = Counter(); TRsub = Counter()

    def emit_tr(cc, fam, r, sub_, s, rid):
        o = owner.get(rid)
        y = 'true' if o == s else ('other' if o else 'nobody')
        TR[(fam, r, cc, y)] += 1
        if sub_:
            TRsub[(fam, r, sub_, y)] += 1
    scan('train', emit_tr)
    del owner
    print('train scanned', flush=True)
    link, cand = {}, defaultdict(set)
    for i, l in enumerate(open(sub, encoding='utf-8')):
        if i:
            s, m = (l.rstrip('\n').split('\t') + [''])[:2]
            for t in m.split(','):
                if t:
                    link[t] = s
    for i, l in enumerate(open(cand_f, encoding='utf-8')):
        if i:
            s, m = (l.rstrip('\n').split('\t') + [''])[:2]
            cand[s] = set(x for x in m.split(',') if x)
    TE = Counter(); TEsub = Counter(); pairs = defaultdict(list)

    def emit_te(cc, fam, r, sub_, s, rid):
        st = 'linked' if link.get(rid) == s else ('other_S1' if rid in link else ('unlinked_cand' if rid in cand.get(s, ()) else 'unlinked_nocand'))
        TE[(fam, r, cc, st)] += 1
        if sub_:
            TEsub[(fam, r, sub_, cc, st)] += 1
        if st in ('linked', 'unlinked_cand'):
            pairs[(fam, r, sub_, cc, st)].append((s, rid))
    scan('test', emit_te)
    pickle.dump(dict(TR=TR, TRsub=TRsub, TE=TE, TEsub=TEsub, pairs=dict(pairs)), open('pattern_audit.pkl', 'wb'))
    print('done', flush=True)


if __name__ == '__main__':
    main()
