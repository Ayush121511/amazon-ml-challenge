"""Train check for census v2's typo path: is a street-word typo copy noise or a decoy signal in US/India labels?

On the labelled train sample (v5 top-10 pairs), pairs at the S1's first house number with a non-blank address, split by
name relation (identical core name / only filler or legal words added / other edit) and street relation (exact street
words / every street word matched up to a typo / no match). Prints the true-match rate and count per cell.
"""
import gzip
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import os
os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
from france_census import FILLER, core, house, near, street_words, tokens_addr

H = Path.home() / 'scratch'
EN_FILLER = FILLER | {'group', 'services', 'and', 'sons', 'co', 'company', 'associates', 'enterprises', 'llc', 'inc',
                      'corp', 'ltd', 'pvt', 'private', 'limited', 'the'}


def main():
    S = np.load(H / 'amazon_ml_2026/artifacts/fusion/stage1.npz')
    y = np.load(H / 'vishal_v4_artifacts/pairs/y.npy')[S['rows']]
    mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
    pr = []
    with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                pr.append(line.rstrip('\n').split('\t'))
    need = {x for p in pr for x in p}
    text = {}
    for k in (1, 2, 3):
        with open(H / f'amazon_ml_2026/data/dataset/train/train_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if q[ie] in need:
                    text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
    cell = Counter(); pos = Counter()
    for (r, t), lab in zip(pr, y):
        rn, ra, cty = text[r]; tn, ta, _ = text.get(t, ('', '', ''))
        if not ta.strip():
            continue
        h1, h2 = house(ra)[0], house(ta)[0]
        if not h1 or h1 != h2:
            continue
        sw, tt = street_words(ra), tokens_addr(ta)
        if not sw:
            continue
        st = 'exact' if sw <= tt else ('typo' if all(w in tt or near(w, tt) for w in sw) else 'other')
        rc, tc = set(core(rn)), set(core(tn))
        nm = 'same' if rc == tc else ('filler' if rc <= tc and (tc - rc) <= EN_FILLER else 'edit')
        k = (cty, nm, st)
        cell[k] += 1; pos[k] += int(lab)
    for k in sorted(cell):
        print(k, cell[k], 'true', round(pos[k] / cell[k], 4), flush=True)


if __name__ == '__main__':
    main()
