"""Co-located unrelated names: alias or same-building distractor? True rate by how many such records share the S1's address.

A record whose core name shares no word with the S1's, at the S1's house number and street, is an alias in some cases
and a different business in the same building in others. Train (labelled sample): true rate of these pairs by k, the
number of such records in the S1's candidate list, and by country. France (a submission): pairs and accepted links by
the same k, with p quantiles, so the France decision can be set against the train rate. Env LATE, SUB, OUT.
"""
import gzip
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
from fr_labels import read_france
from france_census import core, house, load, street_words, tokens_addr

H = Path('/home/scai/mtech/aib262015/scratch')


def colocated(ra, tn, ta, rc):
    if not ta.strip():
        return False
    h1, h2 = house(ra)[0], house(ta)[0]
    sw = street_words(ra)
    return bool(h1 and h1 == h2 and sw and sw <= tokens_addr(ta) and rc and not set(rc) & set(core(tn)))


def main():
    R = {}
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
    by = defaultdict(list)
    V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
    PM = {'t5': np.load(V / 'fusion/boundary_models_t5/stage3b_prob.npy'), 'frrobust': np.load(V / 'submission_FRrobust/robust_sample_prob.npy')}
    from distractor_check import partition_of
    gold = json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text())
    cp = partition_of({k: set(v) for k, v in gold.items()})
    hi = Counter(); hip = Counter()
    rows = defaultdict(list)
    for i, ((r, t), lab) in enumerate(zip(pr, y)):
        rn, ra, cty = text[r]; tn, ta, _ = text.get(t, ('', '', ''))
        if colocated(ra, tn, ta, core(rn)):
            by[r].append(int(lab)); rows[r].append((i, int(lab)))
    for r, lst in rows.items():
        if cp[r] == 'train':
            continue
        k = min(len(lst), 4)
        for mdl, P in PM.items():
            for i, lab in lst:
                if P[i] >= 0.9:
                    hi[(text[r][2], mdl, k)] += 1; hip[(text[r][2], mdl, k)] += lab
    R['train_p_ge_0.9_devholdout'] = {str(k): [hi[k], round(hip[k] / hi[k], 3)] for k in sorted(hi, key=str)}
    cell, pos = Counter(), Counter()
    for r, labs in by.items():
        k = min(len(labs), 6); cty = text[r][2]
        cell[(cty, k)] += len(labs); pos[(cty, k)] += sum(labs)
        cell[(cty, 'refs', k)] += 1; pos[(cty, 'refs', k)] += sum(labs)
    R['train'] = {str(k): [cell[k], round(pos[k] / cell[k], 4)] for k in sorted(cell, key=str) if 'refs' not in k}
    R['train_true_per_ref'] = {str(k): round(pos[k] / cell[k], 3) for k in sorted(cell, key=str) if 'refs' in k}
    Z = np.load(os.environ['LATE'], allow_pickle=True)
    ri, tg, p3 = Z['ref_idx'], Z['target'].astype(str), Z['prob']
    match = load(os.environ['SUB'])
    refs, ftext, m, s1_core = read_france(ri, tg)
    fb = defaultdict(list)
    for j in np.flatnonzero(m):
        r, t = refs[ri[j]], tg[j]
        tn, ta = ftext.get(t, ('', ''))
        if colocated(ftext[r][1], tn, ta, s1_core[r]):
            fb[r].append((j, t in match.get(r, ())))
    fc, fa, fp = Counter(), Counter(), defaultdict(list)
    for r, js in fb.items():
        k = min(len(js), 6)
        fc[k] += len(js); fa[k] += sum(a for _, a in js); fp[k] += [p3[j] for j, _ in js]
    R['france'] = {str(k): {'pairs': fc[k], 'accepted': fa[k], 'acc_rate': round(fa[k] / fc[k], 3),
                            'p_q10_50_90': np.quantile(fp[k], [0.1, 0.5, 0.9]).round(3).tolist()} for k in sorted(fc)}
    out = json.dumps(R, indent=1)
    open(os.environ['OUT'], 'w').write(out)
    print(out, flush=True)


if __name__ == '__main__':
    main()
