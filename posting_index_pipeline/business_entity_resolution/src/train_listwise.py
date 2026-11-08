"""Listwise cross-encoder: one reference + its whole candidate list in one sequence, one logit per candidate.

Sequence: [CLS] reference [SEP] [CAND] cand_1 [CAND] cand_2 ... [SEP]; a linear head reads the hidden state at each
[CAND] marker. Seeing all candidates together lets the model use siblings and look-alikes as context (the
set-encoder idea). Trained data-parallel on the full-train lists (references outside the 100k sample), time-budgeted.
After training it scores (a) the 100k-sample v5 top-10 lists -> ce_logit_sample.npy aligned to mask_top10 rows,
all out-of-sample, and (b) the test v5 top-10 lists -> test_listwise.npz (ref, target, logit) for late_predict.py.
Launch: python -m torch.distributed.run --nproc_per_node=N train_listwise.py
"""
import gzip
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer

HOME = Path.home()
SCR = Path(os.environ.get('ER_SCRATCH', str(HOME / 'scratch')))
PAIRS = SCR / 'amazon_ml_2026/artifacts/fulltrain_pairs'
RAW = SCR / 'amazon_ml_2026/data/dataset'
V4 = SCR / 'vishal_v4_artifacts'
MASK = SCR / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy'
BASE = Path(os.environ.get('BASE', str(SCR / 'amazon_ml_2026/hf_models/mdeberta-v3-base')))
OUT = Path(os.environ.get('OUT_DIR', str(SCR / 'amazon_ml_2026/artifacts/listwise')))
TEST_CAND = os.environ.get('TEST_CAND', '')
MAX_MIN = float(os.environ.get('MAX_MIN', 180))
LR = float(os.environ.get('LR', 2e-5))
LISTS = int(os.environ.get('LISTS', 32))
INIT = os.environ.get('INIT', '')  # continue from a trained listwise model dir (model/ + listwise_head.pt)
PSEUDO_LISTS = os.environ.get('PSEUDO_LISTS', '')  # France self-training lists (france_pseudo.py), test split
PSEUDO_FRAC = float(os.environ.get('PSEUDO_FRAC', 0.4))
TEST_COUNTRY = os.environ.get('TEST_COUNTRY', '')
MAX_CANDS, REF_LEN, CAND_LEN = 12, 48, 34


def log(*a):
    if int(os.environ.get('RANK', 0)) == 0:
        print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_text(split, need):
    text = {}
    for k in (1, 2, 3):
        with open(RAW / split / f'{split}_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) > iaddr and q[ie] in need:
                    text[q[ie]] = f'{q[iname]} | {q[iaddr]}'
    return text


class Listwise(torch.nn.Module):
    def __init__(self, base, vocab):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        self.enc.resize_token_embeddings(vocab)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, ids, att, pos):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        return self.head(h[pos[:, 0], pos[:, 1]]).squeeze(-1)


def build(tok, cand_id, ref_texts, cand_lists):
    """ref_texts: list[str]; cand_lists: list[list[str]] -> tensors + marker positions (row, col)."""
    r_ids = tok(ref_texts, add_special_tokens=False, truncation=True, max_length=REF_LEN)['input_ids']
    flat = [c for cl in cand_lists for c in cl]
    c_ids = tok(flat, add_special_tokens=False, truncation=True, max_length=CAND_LEN)['input_ids'] if flat else []
    seqs, pos, k = [], [], 0
    for i, cl in enumerate(cand_lists):
        s = [tok.cls_token_id] + r_ids[i] + [tok.sep_token_id]
        for _ in cl:
            pos.append((i, len(s)))
            s += [cand_id] + c_ids[k]
            k += 1
        seqs.append(s + [tok.sep_token_id])
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return ids, att, torch.tensor(pos, dtype=torch.long)


def score_lists(model, tok, cand_id, dev, refs, lists, text, bs=64):
    out = []
    order = np.argsort([len(l) for l in lists], kind='stable')
    res = [None] * len(lists)
    with torch.no_grad():
        for b in range(0, len(order), bs):
            o = order[b:b + bs]
            ids, att, pos = build(tok, cand_id, [text.get(refs[i], ' | ') for i in o],
                                  [[text.get(t, ' | ') for t in lists[i]] for i in o])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(ids.to(dev), att.to(dev), pos.to(dev)).float().cpu().numpy()
            k = 0
            for i in o:
                res[i] = lg[k:k + len(lists[i])]
                k += len(lists[i])
    return res


def main():
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dev = torch.device('cuda', local)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    groups = defaultdict(list)
    for f in sorted(PAIRS.glob('chunk*.npz')):
        z = np.load(f)
        for r, t, l in zip(z['ref'], z['target'], z['label']):
            groups[r].append((t, float(l)))
    keys = sorted(groups)
    rng = np.random.default_rng(2026 + rank)
    mine = [keys[i] for i in np.random.default_rng(2026).permutation(len(keys))[rank::world]]
    need = set(mine) | {t for r in mine for t, _ in groups[r]}
    text = load_text('train', need)
    log('world', world, 'train lists', len(keys), 'per rank', len(mine), f'{time.time() - t0:.0f}s')

    if INIT:
        tok = AutoTokenizer.from_pretrained(Path(INIT) / 'model')
        cand_id = tok.convert_tokens_to_ids('[CAND]')
        net = Listwise(Path(INIT) / 'model', len(tok))
        net.head.load_state_dict(torch.load(Path(INIT) / 'model' / 'listwise_head.pt', map_location='cpu'))
        net = net.to(dev)
    else:
        tok = AutoTokenizer.from_pretrained(BASE)
        tok.add_special_tokens({'additional_special_tokens': ['[CAND]']})
        cand_id = tok.convert_tokens_to_ids('[CAND]')
        net = Listwise(BASE, len(tok)).to(dev)
    pgroups, pkeys, n_ps = defaultdict(list), [], 0
    if PSEUDO_LISTS:
        zp = np.load(PSEUDO_LISTS)
        for r, t, l in zip(zp['ref'], zp['target'], zp['label']):
            pgroups[r].append((t, float(l)))
        pkeys = sorted(pgroups)
        ptext = load_text('test', set(pkeys) | {t for r in pkeys for t, _ in pgroups[r]})
        n_ps = int(LISTS * PSEUDO_FRAC)
        log('pseudo lists', len(pkeys), 'per batch', n_ps)
    model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    lossf = torch.nn.BCEWithLogitsLoss()
    planned = (len(keys) // world) // (LISTS - n_ps)
    warm = max(1, int(0.03 * min(planned, 20000)))
    model.train()
    t1 = time.time()
    step = 0
    while step < planned:
        frac = min(1.0, (step + 1) / warm) * max(0.0, (planned - step) / planned)
        for g in opt.param_groups:
            g['lr'] = LR * frac
        nb = LISTS - n_ps
        batch = mine[step * nb:(step + 1) * nb]
        lists, labels, rtexts = [], [], []
        for r in [('p', pkeys[k]) for k in (rng.integers(0, len(pkeys), n_ps) if n_ps else [])] + [('t', r) for r in batch]:
            src, r = r
            G_, T_ = (pgroups, ptext) if src == 'p' else (groups, text)
            rtexts.append(T_.get(r, ' | '))
            c = G_[r][:]
            rng.shuffle(c)
            pos_c = [x for x in c if x[1] > 0][:MAX_CANDS]
            neg_c = [x for x in c if x[1] == 0][:MAX_CANDS - len(pos_c)]
            c = pos_c + neg_c
            rng.shuffle(c)
            lists.append([T_.get(t, ' | ') for t, _ in c])
            labels += [l for _, l in c]
        ids, att, pos = build(tok, cand_id, rtexts, lists)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = model(ids.to(dev), att.to(dev), pos.to(dev))
        loss = lossf(lg.float(), torch.tensor(labels, device=dev))
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
            log(f'{float(sps):.3f} s/step -> planned {planned} steps ({planned * LISTS * world:,} lists)')
        if step % 2000 == 0:
            log(f'step {step}/{planned} loss {loss.item():.4f} {time.time() - t1:.0f}s')
    net.eval()
    dist.barrier()
    if rank == 0:
        (OUT / 'model').mkdir(exist_ok=True)
        net.enc.save_pretrained(OUT / 'model')
        tok.save_pretrained(OUT / 'model')
        torch.save(net.head.state_dict(), OUT / 'model' / 'listwise_head.pt')
    log('trained', step, 'steps', f'{time.time() - t1:.0f}s')

    mask = np.load(MASK)
    s_ref, s_tgt = [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t')
                s_ref.append(r); s_tgt.append(t)
    sg = defaultdict(list)
    for j, r in enumerate(s_ref):
        sg[r].append(j)
    skeys = sorted(sg)[rank::world]
    stext = load_text('train', set(skeys) | {s_tgt[j] for r in skeys for j in sg[r]})
    res = score_lists(net, tok, cand_id, dev, skeys, [[s_tgt[j] for j in sg[r]] for r in skeys], stext)
    part = np.full(len(s_ref), np.nan, dtype=np.float32)
    for r, lg in zip(skeys, res):
        part[sg[r]] = lg
    np.save(OUT / f'sample_part{rank}.npy', part)
    dist.barrier()
    if rank == 0:
        ce = np.full(len(s_ref), np.nan, dtype=np.float32)
        for k in range(world):
            p = np.load(OUT / f'sample_part{k}.npy')
            ce = np.where(np.isnan(ce), p, ce)
        np.save(OUT / 'ce_logit_sample.npy', ce)
        log('sample rows scored', int((~np.isnan(ce)).sum()), 'of', len(ce))

    if TEST_CAND:
        trefs, tlists = [], []
        keep = None
        if TEST_COUNTRY:
            keep = set()
            with open(RAW / 'test' / 'test_source1.tsv', encoding='utf-8') as fh:
                hd = fh.readline().rstrip('\n').split('\t')
                ie_, ic_ = hd.index('entity_id'), hd.index('country')
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if len(q) > ic_ and q[ic_] == TEST_COUNTRY:
                        keep.add(q[ie_])
        with open(TEST_CAND, encoding='utf-8') as f:
            next(f)
            for k, line in enumerate(f):
                if k % world != rank:
                    continue
                s, c = line.rstrip('\n').split('\t')
                if c and (keep is None or s in keep):
                    trefs.append(s); tlists.append(c.split(','))
        ttext = load_text('test', set(trefs) | {t for l in tlists for t in l})
        res = score_lists(net, tok, cand_id, dev, trefs, tlists, ttext)
        R = np.array([r for r, l in zip(trefs, tlists) for _ in l])
        T = np.array([t for l in tlists for t in l])
        np.savez(OUT / f'ce_listwise_shard{rank}of{world}.npz', ref=R, target=T,
                 logit=np.concatenate(res).astype(np.float32))
        log('test lists scored', len(trefs), f'{time.time() - t0:.0f}s')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
