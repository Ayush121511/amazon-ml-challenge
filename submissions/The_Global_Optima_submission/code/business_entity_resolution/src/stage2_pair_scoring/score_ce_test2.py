"""Score test (reference, candidate) pairs from a candidate_pairs.tsv with one cross-encoder model.

--shard i/n scores every n-th reference starting at i, so shards can run as parallel jobs.
Output: <out>/ce_<name>_shard<i>of<n>.npz with ref, target, logit.
"""
import argparse
import sys
import time
from pathlib import Path

import os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROMPT = os.environ.get('PROMPT', '')
QWEN_PRE = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a business record '
            '(name | address), judge whether the Document is a record of the same business\n<Query>: ')
QWEN_SUF = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def encode(tok, L, R, max_len):
    if PROMPT == 'qwen_short':
        return tok([f'<|im_start|>user\nSame business?\nA: {a}\nB: {b}<|im_end|>\n<|im_start|>assistant\n'
                    for a, b in zip(L, R)], truncation=True, max_length=max_len, padding=True, return_tensors='pt')
    if PROMPT == 'qwen':
        return tok([f'{QWEN_PRE}{a}\n<Document>: {b}{QWEN_SUF}' for a, b in zip(L, R)], truncation=True,
                   max_length=max_len, padding=True, return_tensors='pt')
    return tok(list(L), list(R), truncation=True, max_length=max_len, padding=True, return_tensors='pt')

RAW = Path(os.environ.get('ER_DATA', str(Path(os.environ.get('ER_HOME', str(Path.home()))) / 'scratch/amazon_ml_2026/data/dataset/test')))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def read_pairs(cand_file, shard=None):
    refs, tgts = [], []
    i0, n = (map(int, shard.split('/')) if shard else (0, 1))
    with open(cand_file, encoding='utf-8') as f:
        next(f)
        for k, line in enumerate(f):
            if k % n != i0:
                continue
            s, c = line.rstrip('\n').split('\t')
            for t in (c.split(',') if c else []):
                refs.append(s); tgts.append(t)
    return np.array(refs, dtype=object), np.array(tgts, dtype=object)


def load_text():
    text = {}
    for src in ('test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv'):
        with open(RAW / src, encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) > iaddr:
                    text[q[ie]] = f'{q[iname]} | {q[iaddr]}'
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidates', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--max-len', type=int, default=128)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--country', default='', help='score only references of this country')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    refs, tgts = read_pairs(a.candidates, a.shard)
    if a.country:
        keep = set()
        with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, ic = head.index('entity_id'), head.index('country')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) > ic and q[ic] == a.country:
                    keep.add(q[ie])
        m = np.array([r in keep for r in refs], dtype=bool)
        refs, tgts = refs[m], tgts[m]
    text = load_text()
    L = [text.get(r, ' | ') for r in refs]
    R = [text.get(t, ' | ') for t in tgts]
    log('pairs', len(refs), 'shard', a.shard)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).cuda().eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    logit = np.zeros(len(L), dtype=np.float32)
    order = np.argsort([len(x) + len(y) for x, y in zip(L, R)], kind='stable')
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), a.batch):
            ids = order[b:b + a.batch]
            e = encode(tok, [L[i] for i in ids], [R[i] for i in ids], a.max_len).to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logit[ids] = model(**e).logits.squeeze(-1).float().cpu().numpy()
            if (b // a.batch) % 1000 == 0:
                done = b + len(ids)
                log(f'{done}/{len(order)} {done / max(1e-9, time.time() - t0):.0f} pairs/s')
    i0, n = a.shard.split('/')
    np.savez(out / f'ce_{a.name}_shard{i0}of{n}.npz', ref=refs.astype(str), target=tgts.astype(str), logit=logit)
    log('done', len(logit), f'{len(logit) / (time.time() - t0):.0f} pairs/s')


if __name__ == '__main__':
    main()
