"""Cross-source link completion, sized before building: rejected candidates that duplicate an accepted sibling.

For an S1, a rejected candidate whose (core name, house number, street words) key equals the key of a record already
accepted for that S1 from the other source (S2 vs S3), or from the same source, may be a missed copy. Train (T5's own
dev/holdout decisions, labelled): pool size and true rate of such rejected candidates. France (a submission): pool size,
owned-by-another-S1 share, p quantiles, examples. Env LATE, SUB, OUT.
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
from distractor_check import decide, partition_of
from fr_labels import read_france
from france_census import core, house, load, street_words

H = Path('/home/scai/mtech/aib262015/scratch')
V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')


def key(name, addr):
    c = core(name)
    if not c or not addr.strip():
        return None
    return (' '.join(sorted(set(c))), house(addr)[0], ' '.join(sorted(street_words(addr))))


def pool(refs_of, cands, accepted, text):
    """(ref, target, same_source) for rejected candidates whose key equals an accepted sibling's key."""
    out = []
    for r in refs_of:
        acc = accepted.get(r, set())
        if not acc:
            continue
        keys = defaultdict(set)
        for a in acc:
            k = key(*text.get(a, ('', ''))[:2])
            if k:
                keys[k].add(a[:2])
        for t in cands[r]:
            if t in acc:
                continue
            k = key(*text.get(t, ('', ''))[:2])
            if k and k in keys:
                out.append((r, t, t[:2] in keys[k]))
    return out


def main():
    R = {}
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
    P = np.load(V / 'fusion/boundary_models_t5/stage3b_prob.npy')
    lab = {(r, t): int(v) for r, t, v in zip(refs, tgts, y)}
    part = np.array([cp[r] for r in refs], dtype=object)
    for p in ('dev', 'holdout'):
        ix = np.flatnonzero(part == p)
        prefs = sorted(s for s in gold if cp[s] == p)
        pos = {s: i for i, s in enumerate(prefs)}
        ch = decide(np.array([pos[r] for r in refs[ix]]), P[ix], 'expected', 0.66, len(prefs))
        cands, accepted = defaultdict(list), defaultdict(set)
        for j, c in zip(ix, ch):
            cands[refs[j]].append(tgts[j])
            if c:
                accepted[refs[j]].add(tgts[j])
        pl = pool(cands.keys(), cands, accepted, text)
        cnt = Counter(); tru = Counter()
        for r, t, same in pl:
            k = (text[r][2], 'same_src' if same else 'other_src')
            cnt[k] += 1; tru[k] += lab[(r, t)]
        R['train_' + p] = {str(k): [cnt[k], round(tru[k] / cnt[k], 3)] for k in sorted(cnt)}
        print('train', p, R['train_' + p], flush=True)
    Z = np.load(os.environ['LATE'], allow_pickle=True)
    ri, tg, p3 = Z['ref_idx'], Z['target'].astype(str), Z['prob']
    match = load(os.environ['SUB'])
    owner = {t: r for r, ts in match.items() for t in ts}
    frefs, ftext, m, _ = read_france(ri, tg)
    cands = defaultdict(list); pj = {}
    for j in np.flatnonzero(m):
        cands[frefs[ri[j]]].append(tg[j]); pj[(frefs[ri[j]], tg[j])] = j
    accepted = {r: set(match.get(r, [])) for r in cands}
    pl = pool(cands.keys(), cands, accepted, ftext)
    c = Counter((('same_src' if s else 'other_src'), 'owned_elsewhere' if t in owner else 'free') for _, t, s in pl)
    R['france'] = {str(k): v for k, v in c.items()}
    free = [x for x in pl if x[1] not in owner]
    R['france_free_p_q10_50_90'] = np.quantile([p3[pj[(r, t)]] for r, t, _ in free], [0.1, 0.5, 0.9]).round(3).tolist() if free else []
    random.seed(4)
    R['france_examples'] = [f"{p3[pj[(r, t)]]:.2f} {'S' if s else 'O'} {ftext[r][0]} | {ftext[r][1]}  =>  {ftext[t][0]} | {ftext[t][1]}"
                            for r, t, s in random.sample(free, min(20, len(free)))]
    out = json.dumps(R, indent=1, ensure_ascii=False)
    open(os.environ['OUT'], 'w', encoding='utf-8').write(out)
    print(out, flush=True)


if __name__ == '__main__':
    main()
