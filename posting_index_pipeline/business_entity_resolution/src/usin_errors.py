"""US/India error anatomy of the final stage-3b (T6) on the labelled sample: where the non-blocking loss sits.

T6's decision (expected@0.74 + per-S1 selection) on dev and holdout. Every error inside the candidate lists is a false
link (FP) or a missed link (FN). An S1's F0.5 loss is shared equally over its error pairs, so each pattern cell gets
loss points (100 x mean over S1 of 1 - F). Cells: error type x target blank x name relation (same / filler / edit /
none) x address relation (same number + street, same number other street, other number same street, other, no number).
Writes counts, loss points and 40 examples per large non-blank cell with the S1's accepted list. Env OUT, PROB, RULE_T.
"""
import gzip
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
sys.path.insert(0, str(Path(__file__).parent))
from distractor_check import decide, f05, partition_of
from france_census import core, house, street_words, tokens_addr

H = Path('/home/scai/mtech/aib262015/scratch')
V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
EN_FILLER = {'group', 'services', 'service', 'and', 'sons', 'co', 'company', 'associates', 'enterprises', 'llc', 'inc',
             'corp', 'ltd', 'pvt', 'private', 'limited', 'the', 'holdings', 'international', 'solutions', 'india'}


def rel(rn, ra, tn, ta):
    rc, tc = set(core(rn)), set(core(tn))
    nm = 'same' if rc == tc else ('filler' if (tc - rc) <= EN_FILLER and (rc - tc) <= EN_FILLER else
                                  ('none' if not rc & tc else 'edit'))
    if not ta.strip():
        return nm, 'blank'
    h1, h2 = house(ra)[0], house(ta)[0]
    sw = street_words(ra)
    st = bool(sw) and sw <= tokens_addr(ta)
    if not h1 or not h2:
        ad = 'nonum_street' if st else 'nonum_other'
    else:
        ad = ('same_no' if h1 == h2 else 'diff_no') + ('_street' if st else '_other')
    return nm, ad


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
    refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
    need = set(refs) | set(tgts) | {t for s in gold for t in gold[s]}
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
    part = np.array([cp[r] for r in refs], dtype=object)
    R = {}
    rng = random.Random(1)
    for p in ('dev', 'holdout'):
        ix = np.flatnonzero(part == p)
        prefs = sorted(s for s in gold if cp[s] == p)
        pos = {s: i for i, s in enumerate(prefs)}
        ch = decide(np.array([pos[r] for r in refs[ix]]), P[ix], 'expected', thr, len(prefs))
        pred = defaultdict(set); cand = defaultdict(list)
        for j, c in zip(ix, ch):
            cand[refs[j]].append(j)
            if c:
                pred[refs[j]].add(tgts[j])
        loss = Counter(); cnt = Counter(); ex = defaultdict(list); total = 0.0; blocking = 0.0
        for s in prefs:
            f = f05(gold[s], pred.get(s, set()))
            total += 1 - f
            if f == 1:
                continue
            cs = {tgts[j] for j in cand.get(s, [])}
            errs = []
            for j in cand.get(s, []):
                t = tgts[j]; yt = t in gold[s]; pt = t in pred.get(s, set())
                if yt != pt:
                    errs.append(('FP' if pt else 'FN', j))
            miss_out = [t for t in gold[s] if t not in cs]
            n_err = len(errs) + len(miss_out)
            for t in miss_out:
                blocking += (1 - f) / n_err
            rn, ra, cty = text.get(s, ('', '', ''))
            for kind, j in errs:
                tn, ta, _ = text.get(tgts[j], ('', '', ''))
                nm, ad = rel(rn, ra, tn, ta)
                k = (kind, cty, nm, ad)
                loss[k] += (1 - f) / n_err; cnt[k] += 1
                if p == 'holdout' and ad != 'blank':
                    ex[(kind, nm, ad)].append(
                        f"{P[j]:.2f} {s} {rn} | {ra}  =>  {tn} | {ta}   [accepted: "
                        + '; '.join(f"{text.get(x, ('?', ''))[0]} | {text.get(x, ('', ''))[1]}" for x in list(pred.get(s, set()))[:4] if x != tgts[j]) + ']')
        n = len(prefs)
        R[p] = {'loss_points_total': round(100 * total / n, 4), 'loss_points_blocking': round(100 * blocking / n, 4),
                'cells': sorted(([str(k), cnt[k], round(100 * v / n, 4)] for k, v in loss.items()), key=lambda x: -x[2])[:45]}
        if p == 'holdout':
            big = sorted(ex, key=lambda k: -sum(loss[(k[0], c, k[1], k[2])] for c in ('US', 'India')))[:6]
            R['examples'] = {str(k): rng.sample(ex[k], min(40, len(ex[k]))) for k in big}
        print(p, 'loss', R[p]['loss_points_total'], 'blocking', R[p]['loss_points_blocking'], flush=True)
    out = json.dumps(R, indent=1, ensure_ascii=False)
    open(os.environ['OUT'], 'w', encoding='utf-8').write(out)


if __name__ == '__main__':
    main()
