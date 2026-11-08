"""Build cross-encoder training pairs for every train reference outside the 100k sample.

Each reference gets its ground-truth links (positives) plus its top-K e5 dense neighbours in the same country
(negatives where not gold). Pairs are written as N_CHUNKS shards (split by reference), so chunk models can train in
parallel on different accounts. The 100k sample stays untouched, so every sample row (fold A, fold B, dev, holdout)
is out-of-sample for models trained on these chunks.
Output: artifacts/fulltrain_pairs/chunk{k}.npz with ref, target, label (strings/uint8).
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from bigdata_ce import read_tsv, embed

HOME = Path.home()
V4 = HOME / 'scratch/vishal_v4_artifacts'
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
E5 = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/dense_v1/model')
OUT = HOME / 'scratch/amazon_ml_2026/artifacts/fulltrain_pairs'
N_CHUNKS = int(os.environ.get('N_CHUNKS', 4))
TOPK = int(os.environ.get('TOPK', 10))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sample = set(json.loads((V4 / 'pairs/gold.json').read_text()))
    gold = {}
    with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            q = line.rstrip('\n').split('\t')
            gold[q[0]] = [t.strip() for t in q[1].split(',')] if len(q) > 1 and q[1] else []
    s1 = {e: (n, a, c) for e, n, a, c in read_tsv(RAW / 'train_source1.tsv')}
    tg = {}
    for src in ('train_source2.tsv', 'train_source3.tsv'):
        for e, n, a, c in read_tsv(RAW / src):
            tg[e] = (n, a, c)
    refs = sorted(e for e in s1 if e not in sample and e in gold)
    random.Random(7).shuffle(refs)
    log('references', len(refs), f'{time.time() - t0:.0f}s')

    tok = AutoTokenizer.from_pretrained(E5)
    mod = AutoModel.from_pretrained(E5).cuda().eval()
    by_c, tg_by_c = defaultdict(list), defaultdict(list)
    for e in refs:
        by_c[s1[e][2]].append(e)
    for e, (n, a, c) in tg.items():
        tg_by_c[c].append(e)
    chunk_of = {r: i % N_CHUNKS for i, r in enumerate(refs)}
    out = [([], [], []) for _ in range(N_CHUNKS)]
    for c, rc in by_c.items():
        tids = tg_by_c[c]
        T = embed(mod, tok, [f'passage: {tg[t][0]} {tg[t][1]}' for t in tids])
        log('embedded targets', c, len(tids), f'{time.time() - t0:.0f}s')
        for b0 in range(0, len(rc), 65536):
            block = rc[b0:b0 + 65536]
            Q = embed(mod, tok, [f'query: {s1[r][0]} {s1[r][1]}' for r in block])
            for b in range(0, len(block), 512):
                top = torch.topk(Q[b:b + 512] @ T.T, TOPK, dim=1).indices.cpu().numpy()
                for i, r in enumerate(block[b:b + 512]):
                    g = set(gold[r])
                    R, Tt, L = out[chunk_of[r]]
                    for t in {tids[j] for j in top[i]} | g:
                        if t in tg:
                            R.append(r); Tt.append(t); L.append(t in g)
            del Q
        log('country', c, 'refs', len(rc), f'{time.time() - t0:.0f}s')
        del T
        torch.cuda.empty_cache()
    for k, (R, Tt, L) in enumerate(out):
        np.savez(OUT / f'chunk{k}.npz', ref=np.array(R), target=np.array(Tt), label=np.array(L, dtype=np.uint8))
        log('chunk', k, 'pairs', len(L), 'pos rate', round(float(np.mean(L)), 4))
    (OUT / 'README.txt').write_text(f'{len(refs)} train references outside the 100k sample, top-{TOPK} e5 dense '
                                    f'neighbours + gold links, split into {N_CHUNKS} chunks by reference.\n')
    log('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
