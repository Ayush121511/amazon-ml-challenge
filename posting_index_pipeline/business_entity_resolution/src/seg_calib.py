"""Calibration of T6 by error segment on dev and holdout: is stage-3b under-confident where it misses real copies?

Segments from usin_errors.rel (name relation x address relation, blank separate). Per segment and p bin: pairs, mean
p, true rate, per split. A segment whose true rate sits well above mean p near the F0.5 margin (0.3-0.8) is a
recalibration candidate. Env OUT, PROB.
"""
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
sys.path.insert(0, str(Path(__file__).parent))
from distractor_check import partition_of
from usin_errors import rel

H = Path('/home/scai/mtech/aib262015/scratch')
V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
BINS = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.01]


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
    acc = defaultdict(lambda: np.zeros(3))
    for j, (r, t) in enumerate(zip(refs, tgts)):
        sp = cp[r]
        if sp == 'train' or not (0.05 <= P[j] < 0.95):
            continue
        rn, ra, cty = text.get(r, ('', '', ''))
        tn, ta, _ = text.get(t, ('', '', ''))
        nm, ad = rel(rn, ra, tn, ta)
        seg = f'{nm}|{ad if ad == "blank" else ad.split("_")[0] + "_" + ad.split("_")[-1]}'
        b = int(np.searchsorted(BINS, P[j], side='right') - 1)
        k = (sp, cty, seg, f'{BINS[b]}-{BINS[b + 1]}')
        acc[k] += [1, P[j], y[j]]
    rows = {}
    for (sp, cty, seg, b), (n, sp_, sy) in acc.items():
        rows.setdefault(f'{cty}|{seg}|{b}', {})[sp] = [int(n), round(sp_ / n, 3), round(sy / n, 3)]
    keep = {k: v for k, v in rows.items() if sum(x[0] for x in v.values()) >= 40}
    out = json.dumps(dict(sorted(keep.items())), indent=0)
    open(os.environ['OUT'], 'w').write(out)
    print(out, flush=True)


if __name__ == '__main__':
    main()
