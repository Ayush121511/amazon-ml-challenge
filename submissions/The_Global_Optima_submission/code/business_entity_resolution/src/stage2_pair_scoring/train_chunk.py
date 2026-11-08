"""Train one mdeberta cross-encoder on one full-train chunk, then score all 100k-sample top-10 rows.

Runs from any team account: inputs are read from ER_SCRATCH (the shared scratch), outputs go to OUT_DIR.
Output: OUT_DIR/model/ (HF), OUT_DIR/ce_logit_sample.npy (one logit per True position of mask_top10.npy, in
pairs.tsv.gz order; every sample row is out-of-sample for this model).
"""
import gzip
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SCR = Path(os.environ.get('ER_SCRATCH', str(Path.home() / 'scratch')))
CHUNK = int(os.environ['CHUNK'])
OUT = Path(os.environ['OUT_DIR'])
BASE = SCR / 'amazon_ml_2026/hf_models/mdeberta-v3-base'
PAIRS = SCR / 'amazon_ml_2026/artifacts/fulltrain_pairs'
RAW = SCR / 'amazon_ml_2026/data/dataset/train'
V4 = SCR / 'vishal_v4_artifacts'
MASK = SCR / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy'
EPOCHS = float(os.environ.get('EPOCHS', 1))
LR, BATCH, MAX_LEN = 2e-5, 128, 128


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_text(need):
    text = {}
    for src in ('train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv'):
        with open(RAW / src, encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) > iaddr and q[ie] in need:
                    text[q[ie]] = f'{q[iname]} | {q[iaddr]}'
    return text


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    z = np.load(PAIRS / f'chunk{CHUNK}.npz')
    cr, ct, cy = z['ref'], z['target'], z['label'].astype(np.float32)
    mask = np.load(MASK)
    sr, st = [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t')
                sr.append(r); st.append(t)
    text = load_text(set(cr) | set(ct) | set(sr) | set(st))
    L = [text.get(r, ' | ') for r in cr]
    R = [text.get(t, ' | ') for t in ct]
    log('chunk', CHUNK, 'pairs', len(cy), 'pos rate', round(float(cy.mean()), 4), f'{time.time() - t0:.0f}s')

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).cuda()
    steps = int(math.ceil(len(cy) / BATCH) * EPOCHS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, int(0.03 * steps))) * max(0.0, (steps - s) / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(100 + CHUNK)
    model.train()
    step, t1 = 0, time.time()
    while step < steps:
        order = rng.permutation(len(cy))
        for b in range(0, len(order), BATCH):
            if step >= steps:
                break
            ids = order[b:b + BATCH]
            e = tok([L[i] for i in ids], [R[i] for i in ids], truncation=True, max_length=MAX_LEN, padding=True,
                    return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logit = model(**e).logits.squeeze(-1)
            loss = lossf(logit.float(), torch.from_numpy(cy[ids]).cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 2000 == 0:
                log(f'step {step}/{steps} loss {loss.item():.4f} {time.time() - t1:.0f}s')
            if step % 20000 == 0:
                model.save_pretrained(OUT / 'model'); tok.save_pretrained(OUT / 'model')
    model.eval()
    model.save_pretrained(OUT / 'model')
    tok.save_pretrained(OUT / 'model')
    log('trained', f'{time.time() - t1:.0f}s')

    SL = [text.get(r, ' | ') for r in sr]
    SR = [text.get(t, ' | ') for t in st]
    ce = np.zeros(len(SL), dtype=np.float32)
    order = np.argsort([len(a) + len(b) for a, b in zip(SL, SR)], kind='stable')
    with torch.no_grad():
        for b in range(0, len(order), 1024):
            ids = order[b:b + 1024]
            e = tok([SL[i] for i in ids], [SR[i] for i in ids], truncation=True, max_length=MAX_LEN, padding=True,
                    return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ce[ids] = model(**e).logits.squeeze(-1).float().cpu().numpy()
    np.save(OUT / 'ce_logit_sample.npy', ce)
    log('done; sample rows scored', len(ce), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
