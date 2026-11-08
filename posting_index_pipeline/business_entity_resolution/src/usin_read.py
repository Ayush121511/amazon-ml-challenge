"""Every non-blank US/India holdout error of T6, for reading one by one, plus the blocking misses.

For each holdout S1 with F0.5 < 1: its false links (with the S1 that truly owns the record, if any), its missed links
inside the candidates, and its true records outside the candidate list (blocking), each with p, name/address relation
and the S1's accepted records. Blank-address records are skipped. Writes a text file sorted by the S1's F0.5 loss,
and counts of blocking misses by relation. Env OUT, PROB, RULE_T.
"""
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
sys.path.insert(0, str(Path(__file__).parent))
from distractor_check import decide, f05, partition_of
from usin_errors import rel

H = Path('/home/scai/mtech/aib262015/scratch')
V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')


def main():
    S = np.load(H / 'amazon_ml_2026/artifacts/fusion/stage1.npz')
    y = np.load(H / 'vishal_v4_artifacts/pairs/y.npy')[S['rows']]
    gold = {k: set(v) for k, v in json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text()).items()}
    owner = {t: s for s, ts in gold.items() for t in ts}
    cp = partition_of(gold)
    mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
    refs, tgts = [], []
    with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t'); refs.append(r); tgts.append(t)
    refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
    need = set(refs) | set(tgts) | {t for s in gold if cp[s] == 'holdout' for t in gold[s]}
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
    thr = float(os.environ.get('RULE_T', '0.74'))
    ix = np.flatnonzero(np.array([cp[r] for r in refs], dtype=object) == 'holdout')
    prefs = sorted(s for s in gold if cp[s] == 'holdout')
    pos = {s: i for i, s in enumerate(prefs)}
    ch = decide(np.array([pos[r] for r in refs[ix]]), P[ix], 'expected', thr, len(prefs))
    pred, cand, pj = defaultdict(set), defaultdict(set), {}
    for j, c in zip(ix, ch):
        cand[refs[j]].add(tgts[j]); pj[(refs[j], tgts[j])] = j
        if c:
            pred[refs[j]].add(tgts[j])
    blocks, lines = Counter(), []
    fmt = lambda t: f"{text.get(t, ('?', ''))[0]} | {text.get(t, ('', ''))[1]}"
    for s in sorted(prefs, key=lambda s: f05(gold[s], pred.get(s, set()))):
        f = f05(gold[s], pred.get(s, set()))
        if f == 1:
            break
        rn, ra, cty = text.get(s, ('', '', ''))
        items = []
        for t in sorted(pred.get(s, set()) - gold[s]):
            if text.get(t, ('', ''))[1].strip():
                o = owner.get(t)
                items.append(f"  FP {P[pj[(s, t)]]:.2f} {rel(rn, ra, *text.get(t, ('', ''))[:2])} {fmt(t)}   <true owner: {fmt(o) if o else 'none (distractor)'}>")
        for t in sorted(gold[s] - pred.get(s, set())):
            if not text.get(t, ('', ''))[1].strip():
                continue
            r2 = rel(rn, ra, *text.get(t, ('', ''))[:2])
            if t in cand[s]:
                items.append(f"  FN {P[pj[(s, t)]]:.2f} {r2} {fmt(t)}")
            else:
                items.append(f"  BLOCK {r2} {fmt(t)}"); blocks[(cty,) + r2] += 1
        if items:
            acc = '; '.join(fmt(t) for t in sorted(pred.get(s, set()) & gold[s])[:4])
            lines.append(f"\n[{cty} F={f:.2f}] {rn} | {ra}\n    accepted ok: {acc}\n" + '\n'.join(items))
    open(os.environ['OUT'], 'w', encoding='utf-8').write(''.join(lines))
    print('S1 blocks written', len(lines), flush=True)
    print(json.dumps(sorted(((str(k), v) for k, v in blocks.items()), key=lambda x: -x[1])[:25], indent=0), flush=True)


if __name__ == '__main__':
    main()
