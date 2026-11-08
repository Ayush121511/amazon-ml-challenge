"""Data-parallel (torchrun) cross-encoder on the full-train pairs, time-budgeted, then sample scoring.

Launch: python -m torch.distributed.run --nproc_per_node=N train_ddp.py
Env: MAX_MIN (training minutes; the LR schedule is re-planned after 200 steps so it completes inside the budget),
TRAIN_PAIRS (cap on pairs drawn), BASE (HF model dir), PROMPT ('' pair input | 'qwen_short'), LR, BATCH, MAX_LEN,
OUT_DIR, ER_SCRATCH. Outputs in OUT_DIR: model/, ce_logit_sample.npy (one logit per True position of
mask_top10.npy; every 100k-sample row is out-of-sample because the full-train pairs exclude that sample).
"""
import gzip
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HOME = Path.home()
SCR = Path(os.environ.get('ER_SCRATCH', str(HOME / 'scratch')))
PAIRS = SCR / 'amazon_ml_2026/artifacts/fulltrain_pairs'
RAW = SCR / 'amazon_ml_2026/data/dataset/train'
V4 = SCR / 'vishal_v4_artifacts'
MASK = SCR / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy'
BASE = Path(os.environ.get('BASE', str(SCR / 'amazon_ml_2026/hf_models/mdeberta-v3-base')))
OUT = Path(os.environ.get('OUT_DIR', str(SCR / 'amazon_ml_2026/artifacts/cross_encoder_v1/mdeberta_full')))
MAX_MIN = float(os.environ.get('MAX_MIN', 75))
TRAIN_PAIRS = int(os.environ.get('TRAIN_PAIRS', 12_000_000))
PROMPT = os.environ.get('PROMPT', '')
LR = float(os.environ.get('LR', 2e-5))
BATCH = int(os.environ.get('BATCH', 128))
MAX_LEN = int(os.environ.get('MAX_LEN', 128))


def log(*a):
    if int(os.environ.get('RANK', 0)) == 0:
        print(time.strftime('%H:%M:%S'), *a, flush=True)


def encode(tok, L, R):
    if PROMPT == 'qwen_short':
        texts = [f'<|im_start|>user\nSame business?\nA: {a}\nB: {b}<|im_end|>\n<|im_start|>assistant\n'
                 for a, b in zip(L, R)]
        return tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt')
    return tok(list(L), list(R), truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt')


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
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dev = torch.device('cuda', local)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    refs, tgts, labels = [], [], []
    for f in sorted(PAIRS.glob('chunk*.npz')):
        z = np.load(f)
        refs.append(z['ref']); tgts.append(z['target']); labels.append(z['label'])
    refs, tgts = np.concatenate(refs), np.concatenate(tgts)
    y = np.concatenate(labels).astype(np.float32)
    perm = np.random.default_rng(2026).permutation(len(y))[:min(TRAIN_PAIRS, len(y))]
    mine = perm[rank::world]
    mask = np.load(MASK)
    sr, st = [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t')
                sr.append(r); st.append(t)
    text = load_text(set(refs[mine]) | set(tgts[mine]) | set(sr) | set(st))
    L = [text.get(r, ' | ') for r in refs[mine]]
    R = [text.get(t, ' | ') for t in tgts[mine]]
    ym = y[mine]
    log('world', world, 'base', BASE.name, 'prompt', PROMPT or 'pair', 'pairs total', len(y), 'drawn', len(perm),
        f'{time.time() - t0:.0f}s')

    tok = AutoTokenizer.from_pretrained(BASE)
    net = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1)
    if net.config.pad_token_id is None:
        net.config.pad_token_id = tok.pad_token_id
    model = torch.nn.parallel.DistributedDataParallel(net.to(dev), device_ids=[local])
    planned = (len(perm) // world) // BATCH
    warm = max(1, int(0.03 * min(planned, 20000)))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t1 = time.time()
    step = 0
    while step < planned:
        frac = min(1.0, (step + 1) / warm) * max(0.0, (planned - step) / planned)
        for g in opt.param_groups:
            g['lr'] = LR * frac
        ids = np.arange(step * BATCH, min((step + 1) * BATCH, len(ym)))
        e = encode(tok, [L[i] for i in ids], [R[i] for i in ids]).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logit = model(**e).logits.squeeze(-1)
        loss = lossf(logit.float(), torch.from_numpy(ym[ids]).to(dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step == 200:
            sps = torch.tensor([(time.time() - t1) / 200], device=dev)
            dist.all_reduce(sps, op=dist.ReduceOp.MAX)
            left = MAX_MIN * 60 - (time.time() - t1)
            plan_t = torch.tensor([int(min(planned, 200 + left / (float(sps) * 1.03)))], device=dev)
            dist.broadcast(plan_t, src=0)
            planned = int(plan_t.item())
            log(f'{float(sps):.3f} s/step -> planned {planned} steps ({planned * BATCH * world:,} pairs)')
        if step % 2000 == 0:
            log(f'step {step}/{planned} loss {loss.item():.4f} {time.time() - t1:.0f}s')
        if step % 20000 == 0 and rank == 0:
            net.save_pretrained(OUT / 'model_ckpt'); tok.save_pretrained(OUT / 'model_ckpt')
    net.eval()
    dist.barrier()
    if rank == 0:
        net.save_pretrained(OUT / 'model')
        tok.save_pretrained(OUT / 'model')
    log('trained', step, 'steps', f'{time.time() - t1:.0f}s')

    idx = np.arange(rank, len(sr), world)
    SL = [text.get(sr[i], ' | ') for i in idx]
    SR = [text.get(st[i], ' | ') for i in idx]
    part = np.zeros(len(idx), dtype=np.float32)
    order = np.argsort([len(a) + len(b) for a, b in zip(SL, SR)], kind='stable')
    with torch.no_grad():
        for b in range(0, len(order), 512):
            o = order[b:b + 512]
            e = encode(tok, [SL[i] for i in o], [SR[i] for i in o]).to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                part[o] = net(**e).logits.squeeze(-1).float().cpu().numpy()
    np.save(OUT / f'sample_part{rank}.npy', part)
    dist.barrier()
    if rank == 0:
        ce = np.zeros(len(sr), dtype=np.float32)
        for r in range(world):
            ce[np.arange(r, len(sr), world)] = np.load(OUT / f'sample_part{r}.npy')
        np.save(OUT / 'ce_logit_sample.npy', ce)
        log('sample rows scored', len(ce), f'{time.time() - t0:.0f}s')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
