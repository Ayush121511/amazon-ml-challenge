"""W2 rival-completeness features: does a same-name rival already have its own copy in this record's source?

Blank-address ties are the largest remaining loss: a record with only a name whose name fits several Source 1 businesses.
Records belong to at most one business, and a business usually has a clear copy (same name, its own house number) in
each source where it appears. So if every same-name rival already holds a clear copy in the record's source and the
reference does not, the blank record is more likely the reference's, and vice versa.

Label-free and computed identically on each split over its full tables (train: 2.2M S1 / 10.3M S2+S3; test: 1.73M / 10M).
Name key = sorted set of folded name words without legal/stop words; a "clear copy" of S1 business R in source S = a
non-blank S record with R's name key and R's first house number (or, when R has no number, R's first address word).
Features per (reference, record) row:
  w2_n_s1_key          S1 businesses in the country with the record's name key
  w2_ref_same_key      the reference's own name key equals the record's
  w2_ref_clear_src     clear copies the reference has in the record's source (the record itself counts if non-blank)
  w2_n_rivals          same-key S1 businesses other than the reference
  w2_rivals_clear      of those, rivals holding at least one clear copy in the record's source
  w2_rivals_lacking    rivals with no clear copy in that source (possible owners of a blank record)
  w2_blank_key_src     blank records in the record's source with the record's name key
  w2_slot_ratio        blank_key_src / (possible owners lacking a copy, reference included), 0 if none lack one

  sample: w2_rival_copies.py sample OUT   (mask_top10 rows)   test: w2_rival_copies.py test OUT   (candidate_pairs rows)
"""
import csv
import gzip
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
V4 = Path('/scratch/scai/mtech/aib262015/vishal_v4_artifacts')
PRUNE = Path('/scratch/scai/mtech/aib262015/amazon_ml_2026/artifacts/prune_v1_copy')
CANDS = ME / 'artifacts/submission_v5_top10/candidate_pairs.tsv'
STOP = {'inc', 'llc', 'ltd', 'pvt', 'private', 'limited', 'co', 'corp', 'corporation', 'company', 'lp', 'llp', 'pllc',
        'plc', 'sa', 'sarl', 'sas', 'sasu', 'eurl', 'sci', 'the', 'and', 'of', 'group', 'de', 'la', 'le', 'les', 'des',
        'du', 'et'}
NAMES = ['w2_n_s1_key', 'w2_ref_same_key', 'w2_ref_clear_src', 'w2_n_rivals', 'w2_rivals_clear', 'w2_rivals_lacking',
         'w2_blank_key_src', 'w2_slot_ratio']


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def fold(s):
    s = unicodedata.normalize('NFKD', s or '')
    return re.sub(r'[^\w]+', ' ', ''.join(c for c in s if not unicodedata.combining(c)).lower()).split()


def key(name):
    return ' '.join(sorted({t for t in fold(name) if t not in STOP}))


def anchor(addr):
    """First house number (leading zeros dropped), else first address word; '' for a blank address."""
    m = re.search(r'\d+', addr or '')
    if m:
        return 'n' + str(int(m.group(0)))
    w = fold(addr)
    return ('w' + w[0]) if w else ''


def main():
    stage, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    split = 'train' if stage == 'sample' else 'test'
    raw = ME / f'data/dataset/{split}'
    s1_key, s1_anchor, s1_cty = {}, {}, {}
    by_key = defaultdict(list)                       # (country, key) -> S1 ids
    rec_key, rec_blank, rec_src = {}, {}, {}
    clear = Counter()                                # (country, key, anchor, src) -> non-blank records
    blank_cnt = Counter()                            # (country, key, src) -> blank records
    for k in (1, 2, 3):
        with open(raw / f'{split}_source{k}.tsv', encoding='utf-8', newline='') as fh:
            for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                e, c = row['entity_id'], row['country']
                kk, an = key(row['business_name']), anchor(row['business_address'])
                if k == 1:
                    s1_key[e], s1_anchor[e], s1_cty[e] = kk, an, c
                    by_key[(c, kk)].append(e)
                else:
                    src = f'S{k}'
                    rec_key[e], rec_blank[e], rec_src[e] = kk, not an, src
                    if an:
                        clear[(c, kk, an, src)] += 1
                    else:
                        blank_cnt[(c, kk, src)] += 1
        log('read source', k)

    def n_clear(r, src):
        return clear.get((s1_cty[r], s1_key[r], s1_anchor[r], src), 0) if s1_anchor[r] else 0

    if stage == 'sample':
        mask = np.load(PRUNE / 'mask_top10.npy')
        rows = []
        with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    rows.append(line.rstrip('\n').split('\t'))
    else:
        rows = []
        with open(CANDS, encoding='utf-8') as f:
            next(f)
            for line in f:
                s, c = line.rstrip('\n').split('\t')
                rows.extend((s, t) for t in (c.split(',') if c else []))
    log('rows', len(rows))
    F = np.full((len(rows), len(NAMES)), np.nan, dtype=np.float32)
    for i, (r, t) in enumerate(rows):
        if r not in s1_key or t not in rec_key:
            continue
        c, kt, src = s1_cty[r], rec_key[t], rec_src[t]
        group = by_key.get((c, kt), [])
        n_key = len(group)
        rivals = [g for g in group if g != r][:200]
        rc = sum(1 for g in rivals if n_clear(g, src) > 0)
        ref_clear = n_clear(r, src) if s1_key[r] == kt else 0
        lacking = len(rivals) - rc + (1 if s1_key[r] == kt and ref_clear == 0 else 0)
        nb = blank_cnt.get((c, kt, src), 0)
        F[i] = [n_key, float(s1_key[r] == kt), ref_clear, len(rivals), rc, len(rivals) - rc, nb,
                nb / lacking if lacking else 0.0]
        if i % 2_000_000 == 0:
            log('row', i)
    np.save(out / f'w2_{stage}.npy', F)
    (out / f'w2_{stage}.json').write_text(json.dumps({'rows': len(rows), 'features': NAMES,
                                                      'missing': int(np.isnan(F[:, 0]).sum())}, indent=1))
    log('saved', out / f'w2_{stage}.npy', 'missing', int(np.isnan(F[:, 0]).sum()))


if __name__ == '__main__':
    main()
