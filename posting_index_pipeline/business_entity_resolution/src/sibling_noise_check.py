"""Do true aliases share their address noise with the S1's other copies? Train check (dev + holdout).

For a candidate with no name word in common with its S1 and a non-blank address: noise = address tokens of the S1's
sure copies (T6 p >= 0.9, sharing a name word) that are absent from the S1's own address. Shares = the candidate's
address has at least one such noise token (tokens of 2+ characters). True rate by shares yes/no, overall and in the
F0.5 margin (0.2 <= p < 0.8), by country. Env OUT, PROB.
"""
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
sys.path.insert(0, str(Path(__file__).parent))
from distractor_check import partition_of
from france_census import core, fold

H = Path('/home/scai/mtech/aib262015/scratch')
V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
toks = lambda a: {t for t in re.sub(r'[^a-z0-9/-]+', ' ', fold(a)).split() if len(t) >= 2}


def main():
    S = np.load(H / 'amazon_ml_2026/artifacts/fusion/stage1.npz')
    y = np.load(H / 'vishal_v4_artifacts/pairs/y.npy')[S['rows']]
    gold = {k: set(v) for k, v in json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text()).items()}
    cp = partition_of(gold)
    mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
    refs, tgts = [], []
    with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t'); refs.append(r); tgts.append(t)
    need = set(refs) | set(tgts)
    text = {}
    for k in (1, 2, 3):
        with open(H / f'amazon_ml_2026/data/dataset/train/train_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if q[ie] in need:
                    text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
    P = np.load(os.environ.get('PROB', str(V / 'fusion/boundary_models_t6/stage3b_prob.npy')))
    by = defaultdict(list)
    for j, r in enumerate(refs):
        if cp[r] != 'train':
            by[r].append(j)
    C = Counter(); T = Counter()
    for r, js in by.items():
        rn, ra, cty = text.get(r, ('', '', ''))
        rc, A1 = set(core(rn)), toks(ra)
        sure = [j for j in js if P[j] >= 0.9 and set(core(text.get(tgts[j], ('', ''))[0])) & rc]
        for j in js:
            tn, ta, _ = text.get(tgts[j], ('', '', ''))
            if not ta.strip() or set(core(tn)) & rc:
                continue
            noise = set().union(*[toks(text.get(tgts[s], ('', ''))[1]) for s in sure if s != j]) - A1 if sure else set()
            sh = 'shares' if toks(ta) & noise else ('no_share' if noise else 'no_sibling_noise')
            for band in ('all', 'margin') if 0.2 <= P[j] < 0.8 else ('all',):
                k = (cty, band, sh); C[k] += 1; T[k] += int(y[j])
    R = {str(k): [C[k], round(T[k] / C[k], 3)] for k in sorted(C)}
    open(os.environ['OUT'], 'w').write(json.dumps(R, indent=0))
    print(json.dumps(R, indent=0), flush=True)


if __name__ == '__main__':
    main()
