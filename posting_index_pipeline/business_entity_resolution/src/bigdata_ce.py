"""Train the mdeberta cross-encoder on ~10x more data, then build its late-fusion model.

Extra data: N_EXTRA train references outside Vishal's 100k sample (so dev/holdout stay clean), each paired with its
ground-truth links (positives) and its top-K e5 dense neighbours within the same country (negatives where not gold).
Fold A = original top-10 fold-A pairs + extra pairs. Fold B, dev and holdout are never trained on, so the resulting
logits keep the same out-of-fold role as cross_encoder_v1/<tag>/ce_logit.npy. Then L2 (XGBoost on [v5 prob floored,
this logit]) is trained on fold B, with and without decoy weighting, rule chosen on dev, holdout scored once.
"""
import gzip
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from distractor_check import decide, f05, partition_of

HOME = Path.home()
V4 = HOME / 'scratch/vishal_v4_artifacts'
PRUNE = HOME / 'scratch/amazon_ml_2026/artifacts/prune_v1_copy'
FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
E5 = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/dense_v1/model')
BASE = HOME / 'scratch/amazon_ml_2026/hf_models/mdeberta-v3-base'
TAG = os.environ.get('TAG', 'mdeberta_big')
OUT = HOME / 'scratch/amazon_ml_2026/artifacts/cross_encoder_v1' / TAG
N_EXTRA = int(os.environ.get('N_EXTRA', 150000))
TOPK = int(os.environ.get('TOPK', 10))
EPOCHS = float(os.environ.get('EPOCHS', 1))
LR, BATCH, MAX_LEN = 2e-5, 128, 128


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def read_tsv(path):
    with open(path, encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, iname, iaddr, ic = (head.index(k) for k in ('entity_id', 'business_name', 'business_address', 'country'))
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if len(q) > ic:
                yield q[ie], q[iname], q[iaddr], q[ic]


def embed(model, tok, texts, bs=2048):
    out = torch.empty((len(texts), model.config.hidden_size), dtype=torch.float16, device='cuda')
    order = np.argsort([len(t) for t in texts], kind='stable')
    with torch.no_grad():
        for b in range(0, len(order), bs):
            ids = order[b:b + bs]
            e = tok([texts[i] for i in ids], truncation=True, max_length=96, padding=True, return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = model(**e).last_hidden_state
            m = e['attention_mask'].unsqueeze(-1).float()
            v = torch.nn.functional.normalize((h.float() * m).sum(1) / m.sum(1), dim=-1)
            out[torch.as_tensor(ids, device='cuda')] = v.half()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gold_s = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    cp = partition_of(gold_s)
    full_gold = {}
    with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            q = line.rstrip('\n').split('\t')
            full_gold[q[0]] = {t.strip() for t in q[1].split(',')} if len(q) > 1 and q[1] else set()
    s1 = {e: (n, a, c) for e, n, a, c in read_tsv(RAW / 'train_source1.tsv')}
    tg = {}
    for src in ('train_source2.tsv', 'train_source3.tsv'):
        for e, n, a, c in read_tsv(RAW / src):
            tg[e] = (n, a, c)
    pool = sorted(e for e in s1 if e not in gold_s and e in full_gold)
    extra = random.Random(2026).sample(pool, min(N_EXTRA, len(pool)))
    log('extra refs', len(extra), 'targets', len(tg), f'{time.time() - t0:.0f}s')

    etok = AutoTokenizer.from_pretrained(E5)
    emod = AutoModel.from_pretrained(E5).cuda().eval()
    ex_pairs = []
    by_c = defaultdict(list)
    for e in extra:
        by_c[s1[e][2]].append(e)
    tg_by_c = defaultdict(list)
    for e, (n, a, c) in tg.items():
        tg_by_c[c].append(e)
    for c, refs_c in by_c.items():
        tids = tg_by_c[c]
        T = embed(emod, etok, [f'passage: {tg[t][0]} {tg[t][1]}' for t in tids])
        Q = embed(emod, etok, [f'query: {s1[r][0]} {s1[r][1]}' for r in refs_c])
        for b in range(0, len(refs_c), 512):
            top = torch.topk(Q[b:b + 512] @ T.T, TOPK, dim=1).indices.cpu().numpy()
            for i, r in enumerate(refs_c[b:b + 512]):
                cand = {tids[j] for j in top[i]} | full_gold[r]
                ex_pairs.extend((r, t, float(t in full_gold[r])) for t in cand if t in tg)
        del T, Q
        torch.cuda.empty_cache()
        log('country', c, 'refs', len(refs_c), 'pairs so far', len(ex_pairs))
    del emod
    torch.cuda.empty_cache()

    mask = np.load(PRUNE / 'mask_top10.npy')
    rows, refs, tgts = [], [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t')
                rows.append(i); refs.append(r); tgts.append(t)
    rows, refs, tgts = np.array(rows), np.array(refs, dtype=object), np.array(tgts, dtype=object)
    part = np.array([cp[r] for r in refs], dtype=object)
    y = np.load(V4 / 'pairs/y.npy')[rows].astype(np.float32)
    fmt = lambda d, e: f'{d[e][0]} | {d[e][1]}' if e in d else ' | '
    base_L = [fmt(s1, r) for r in refs]
    base_R = [fmt(tg, t) for t in tgts]
    trA = np.flatnonzero(part == 'trainA')
    TL = [base_L[i] for i in trA] + [fmt(s1, r) for r, _, _ in ex_pairs]
    TR = [base_R[i] for i in trA] + [fmt(tg, t) for _, t, _ in ex_pairs]
    TY = np.concatenate([y[trA], np.array([p for _, _, p in ex_pairs], dtype=np.float32)])
    log('train pairs', len(TY), 'pos rate', round(float(TY.mean()), 4), f'{time.time() - t0:.0f}s')

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).cuda()
    steps = int(math.ceil(len(TY) / BATCH) * EPOCHS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, int(0.05 * steps))) * max(0.0, (steps - s) / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(2026)
    model.train()
    step, t1 = 0, time.time()
    while step < steps:
        order = rng.permutation(len(TY))
        for b in range(0, len(order), BATCH):
            if step >= steps:
                break
            ids = order[b:b + BATCH]
            e = tok([TL[i] for i in ids], [TR[i] for i in ids], truncation=True, max_length=MAX_LEN, padding=True,
                    return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logit = model(**e).logits.squeeze(-1)
            loss = lossf(logit.float(), torch.from_numpy(TY[ids]).cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 1000 == 0:
                log(f'step {step}/{steps} loss {loss.item():.4f} {time.time() - t1:.0f}s')
    model.eval()
    model.save_pretrained(OUT / 'model_foldA')
    tok.save_pretrained(OUT / 'model_foldA')

    sc = np.flatnonzero(part != 'trainA')
    ce = np.full(len(rows), np.nan, dtype=np.float32)
    order = sorted(sc, key=lambda i: len(base_L[i]) + len(base_R[i]))
    with torch.no_grad():
        for b in range(0, len(order), 1024):
            ids = order[b:b + 1024]
            e = tok([base_L[i] for i in ids], [base_R[i] for i in ids], truncation=True, max_length=MAX_LEN,
                    padding=True, return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ce[ids] = model(**e).logits.squeeze(-1).float().cpu().numpy()
    np.save(OUT / 'ce_logit_foldA.npy', ce)
    log('scored', len(sc))

    S = np.load(FUS / 'stage1.npz')
    s1p = S['stage1'].astype(np.float64)
    s1f = np.where(s1p < 0.02, 0.0, s1p)
    X = np.column_stack([s1f, np.nan_to_num(ce, nan=0.0)]).astype(np.float32)
    owned = {t for v in full_gold.values() for t in v}
    decoy = (y == 0) & np.array([t not in owned for t in tgts])
    tb, dv = part == 'trainB', part == 'dev'
    idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
    prefs = {p: sorted(s for s in gold_s if cp[s] == p) for p in idx}
    rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}
    grid = [(ru, round(float(t), 3)) for ru in ('expected', 'threshold') for t in np.arange(0.5, 0.981, 0.02)]
    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                  max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)

    def macro(p, prob, ru, t):
        c = decide(rix[p], prob, ru, t, len(prefs[p]))
        pred = defaultdict(set)
        for i, tt in zip(rix[p][c], tgts[idx[p]][c]):
            pred[i].add(tt)
        return float(np.mean([f05(gold_s[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

    R = {'tag': TAG, 'extra_refs': len(extra), 'extra_pairs': len(ex_pairs), 'train_pairs': int(len(TY)),
         'reference_to_beat': 'L2 with mdeberta_drop0: dev 0.98624 / holdout 0.98459; w=2: dev 0.98607 / holdout 0.98475'}
    for w in (1.0, 2.0):
        wt = np.where(decoy, w, 1.0).astype(np.float32)
        dtr = xgb.QuantileDMatrix(X[tb], y[tb], weight=wt[tb])
        bst = xgb.train(params, dtr, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=dtr), 'dev')],
                        early_stopping_rounds=100, verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        pdv = np.asarray(bst.inplace_predict(X[idx['dev']], iteration_range=it), dtype=np.float64)
        pho = np.asarray(bst.inplace_predict(X[idx['holdout']], iteration_range=it), dtype=np.float64)
        dev = {g: macro('dev', pdv, *g) for g in grid}
        best = max(dev, key=lambda k: (dev[k], k[1]))
        R[f'L2_w{w:g}'] = {'rule': best, 'dev_f05': dev[best], 'holdout_f05': macro('holdout', pho, *best)}
        md = FUS / 'late_models' / f'L2_{TAG}_w{w:g}'
        md.mkdir(parents=True, exist_ok=True)
        bst.save_model(str(md / 'model.json'))
        (md / 'config.json').write_text(json.dumps({'name': f'L2_{TAG}_w{w:g}', 'tag': TAG, 'rule': list(best),
                                                   'best_iteration': int(bst.best_iteration), 'decoy_weight': w,
                                                   **R[f'L2_w{w:g}']}))
        log(f'L2 w{w:g}', R[f'L2_w{w:g}'])
    (OUT / 'results.json').write_text(json.dumps(R, indent=1))
    log('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
