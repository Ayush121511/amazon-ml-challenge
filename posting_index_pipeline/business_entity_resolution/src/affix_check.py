"""Is 'S1 name + one extra word' a true match or a generated decoy? Measured with train labels, by added word.

France test lists are full of 'X Holding / International / Groupe / Développement / Et Fils' records near the S1
address that the cross-encoder rejects and v5 accepts. If the generator builds affiliate decoys the same way in
train (English: Group, International, Development, Holding, & Sons), their train positive rate is ~0 and the
cross-encoder is right on France. For every train S2/S3 record whose name minus one word equals an S1 name key
(keys shared by <= 20 S1), label = the record is that S1's gold match; split by same / different house number.
Legal forms (removed from keys) are measured on exact-key pairs: extra legal token on the record vs none.
Writes fusion/affix_check.json.
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HOME = Path.home()
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
OUT = HOME / 'scratch/amazon_ml_2026/artifacts/fusion/affix_check.json'
LEG = {'llc', 'inc', 'corp', 'co', 'ltd', 'lp', 'llp', 'pllc', 'pc', 'pa', 'pvt', 'private', 'limited', 'corporation',
       'incorporated', 'company', 'l', 'c', 'p'}
AFFIX = {'group', 'international', 'intl', 'development', 'developments', 'holding', 'holdings', 'sons', 'distribution',
         'participations', 'services', 'service', 'partners', 'associates', 'enterprises', 'ventures', 'global',
         'solutions', 'industries', 'trading', 'center', 'centre', 'the', 'and', 'of', 'india', 'usa', 'america',
         'com', 'dba', 'aka', 'systems', 'technologies', 'consulting', 'management', 'foundation', 'trust'}


def toks(s):
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r'[^\w]+', ' ', s).split()


def num(a):
    m = re.search(r'\d+', a or '')
    return m.group(0) if m else ''


def main():
    owner = {}
    with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if len(q) > 1 and q[1]:
                for t in q[1].split(','):
                    owner[t.strip()] = q[0]
    idx = defaultdict(list)
    with open(RAW / 'train_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if len(q) > iaddr:
                t = toks(q[iname])
                idx[' '.join(sorted(set(t) - LEG))].append((q[ie], num(q[iaddr]), set(t)))
    idx = {k: v for k, v in idx.items() if k and len(v) <= 20}
    print('S1 keys', len(idx), flush=True)
    stat = defaultdict(lambda: [0, 0])
    for k in (2, 3):
        with open(RAW / f'train_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) <= iaddr:
                    continue
                e, n = q[ie], num(q[iaddr])
                blank = not q[iaddr].strip()
                T = set(toks(q[iname]))
                core = T - LEG
                own = owner.get(e)
                for s, sn, st in idx.get(' '.join(sorted(core)), []):
                    where = 'blank' if blank else 'same_num' if n and n == sn else 'diff_num'
                    extra = (T & LEG) - st
                    tag = 'exact_key+legal_added' if extra else 'exact_key'
                    stat[(tag, where)][0] += 1; stat[(tag, where)][1] += own == s
                for w in core & AFFIX:
                    for s, sn, st in idx.get(' '.join(sorted(core - {w})), []):
                        if w in st:
                            continue
                        where = 'blank' if blank else 'same_num' if n and n == sn else 'diff_num'
                        stat[(w, where)][0] += 1; stat[(w, where)][1] += own == s
        print('source', k, 'done', flush=True)
    R = {}
    for (w, where), (n, pos) in sorted(stat.items()):
        R.setdefault(w, {})[where] = {'n': n, 'pos_rate': round(pos / n, 4)}
    OUT.write_text(json.dumps(R, indent=1))
    for w in sorted(R, key=lambda w: -sum(v['n'] for v in R[w].values())):
        print(w, R[w], flush=True)


if __name__ == '__main__':
    main()
