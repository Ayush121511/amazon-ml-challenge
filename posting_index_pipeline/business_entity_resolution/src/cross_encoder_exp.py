import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from submission_pipeline import decide, f05

HOME = Path.home()
V4 = HOME / 'scratch/vishal_v4_artifacts'
VISHAL = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
BASE = sys.argv[1] if len(sys.argv) > 1 else str(VISHAL / 'dense_v1/model')
MASK = sys.argv[2] if len(sys.argv) > 2 else 'mask_top10.npy'
EPOCHS = float(sys.argv[3]) if len(sys.argv) > 3 else 2
MAX_LEN = int(os.environ.get('MAX_LEN', 128))
BATCH = int(os.environ.get('BATCH', 128))
LR = float(os.environ.get('LR', 3e-5))
ADDR_DROP = float(os.environ.get('ADDR_DROP', 0))
TAG = os.environ.get('TAG', 'e5small')
FOLD = os.environ.get('FOLD', 'A')
TRAIN_PART = 'train' + FOLD
PROMPT = os.environ.get('PROMPT', '')
QWEN_PRE = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a business record '
            '(name | address), judge whether the Document is a record of the same business\n<Query>: ')
QWEN_SUF = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def encode(tok, L, R, max_len):
    if PROMPT == 'qwen':
        return tok([f'{QWEN_PRE}{a}\n<Document>: {b}{QWEN_SUF}' for a, b in zip(L, R)], truncation=True,
                   max_length=max_len, padding=True, return_tensors='pt')
    return tok(list(L), list(R), truncation=True, max_length=max_len, padding=True, return_tensors='pt')


SIB_NPZ = HOME / 'scratch/amazon_ml_2026/artifacts/fusion/stage1.npz'
OUT = HOME / 'scratch/amazon_ml_2026/artifacts/cross_encoder_v1' / TAG


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def partition_of(gold):
    parent = {s: s for s in gold}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    owner = {}
    for s, targets in gold.items():
        for t in targets:
            if t in owner:
                parent[find(s)] = find(owner[t])
            owner[t] = s

    def part(s):
        root = find(s)
        h = int(hashlib.sha256(root.encode()).hexdigest()[:8], 16) % 100
        if h >= 70:
            return 'dev' if h < 85 else 'holdout'
        return 'trainA' if int(hashlib.sha256((root + '#fold').encode()).hexdigest()[:8], 16) % 2 else 'trainB'
    return {s: part(s) for s in gold}


def macro(part, refs, tgts, prob, rule, thr, gold, partition):
    prefs = sorted(s for s in gold if partition[s] == part)
    pos = {s: i for i, s in enumerate(prefs)}
    m = np.array([partition[r] == part for r in refs])
    ri = np.array([pos[r] for r in refs[m]])
    chosen = decide(ri, prob[m], rule, thr, len(prefs))
    pred = {}
    for i, t in zip(ri[chosen], tgts[m][chosen]):
        pred.setdefault(i, set()).add(t)
    return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs)]))


def evaluate(name, prob, refs, tgts, gold, partition, results):
    grid = [(r, round(float(t), 2)) for r in ('expected', 'threshold') for t in np.arange(0.3, 0.96, 0.05)]
    dev = {g: macro('dev', refs, tgts, prob, g[0], g[1], gold, partition) for g in grid}
    best = max(dev, key=lambda k: (dev[k], k[1]))
    hold = macro('holdout', refs, tgts, prob, best[0], best[1], gold, partition)
    results[name] = {'rule': best, 'dev_f05': dev[best], 'holdout_f05': hold}
    log(name, results[name])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    partition = partition_of(gold)
    mask = np.load(VISHAL / 'prune_v1' / MASK)
    rows, refs, tgts = [], [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t')
                rows.append(i)
                refs.append(r)
                tgts.append(t)
    rows, refs, tgts = np.array(rows), np.array(refs, dtype=object), np.array(tgts, dtype=object)
    part = np.array([partition[r] for r in refs], dtype=object)
    y = np.load(V4 / 'pairs/y.npy')[rows].astype(np.float32)
    log('pairs', len(rows), {p: int((part == p).sum()) for p in ('trainA', 'trainB', 'dev', 'holdout')},
        'pos_rate', float(y.mean()))

    need = set(refs) | set(tgts)
    text = {}
    for src in ('train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv'):
        df = pd.read_csv(RAW / src, sep='\t', dtype=str, keep_default_na=False, quoting=3)
        df = df[df.entity_id.isin(need)]
        for e, n, a in zip(df.entity_id, df.business_name, df.business_address):
            text[e] = (n, a)
        log('text', src, len(df))
    fmt = lambda na: f'{na[0]} | {na[1]}'
    left = [fmt(text.get(r, ('', ''))) for r in refs]
    right = [fmt(text.get(t, ('', ''))) for t in tgts]
    left_na = [f'{text.get(r, ("", ""))[0]} | ' for r in refs]
    right_na = [f'{text.get(t, ("", ""))[0]} | ' for t in tgts]
    empty_addr = np.array([not text.get(r, ('', ''))[1].strip() or not text.get(t, ('', ''))[1].strip()
                           for r, t in zip(refs, tgts)])

    dev_ = torch.device('cuda')
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(dev_)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    train_idx = np.flatnonzero(part == TRAIN_PART)
    steps = int(math.ceil(len(train_idx) / BATCH) * EPOCHS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, int(0.05 * steps))) * max(0.0, (steps - s) / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(2026)
    model.train()
    step = 0
    t0 = time.time()
    while step < steps:
        order = rng.permutation(train_idx)
        for b in range(0, len(order), BATCH):
            if step >= steps:
                break
            ids = order[b:b + BATCH]
            drop = rng.random(len(ids)) < ADDR_DROP
            side = rng.random(len(ids)) < 0.75
            L = [left_na[i] if d and not s_ else left[i] for i, d, s_ in zip(ids, drop, side)]
            Rt = [right_na[i] if d and s_ else right[i] for i, d, s_ in zip(ids, drop, side)]
            enc = encode(tok, L, Rt, MAX_LEN).to(dev_)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logit = model(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), torch.from_numpy(y[ids]).to(dev_))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                log(f'step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s')
    model.eval()
    model.save_pretrained(OUT / f'model_fold{FOLD}')
    tok.save_pretrained(OUT / f'model_fold{FOLD}')

    def score(L, Rt):
        outv = np.zeros(len(L), dtype=np.float32)
        order = sorted(range(len(L)), key=lambda i: len(L[i]) + len(Rt[i]))
        with torch.no_grad():
            for b in range(0, len(order), 1024):
                ids = order[b:b + 1024]
                enc = encode(tok, [L[i] for i in ids], [Rt[i] for i in ids], MAX_LEN).to(dev_)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outv[ids] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        return outv

    score_idx = np.flatnonzero(part != TRAIN_PART)
    ce = np.full(len(rows), np.nan, dtype=np.float32)
    ce[score_idx] = score([left[i] for i in score_idx], [right[i] for i in score_idx])
    np.save(OUT / f'ce_logit_fold{FOLD}.npy', ce)
    log('scored', len(score_idx))
    if SIB_NPZ.exists():
        sp = np.load(SIB_NPZ)['sib_pairs']
        keep = np.flatnonzero(part[sp[:, 0]] != TRAIN_PART)
        sib = np.full(len(sp), np.nan, dtype=np.float32)
        sib[keep] = score([right[sp[k, 0]] for k in keep], [right[sp[k, 1]] for k in keep])
        np.save(OUT / f'sib_logit_fold{FOLD}.npy', sib)
        log('scored sibling pairs', len(keep))
    if FOLD != 'A' or os.environ.get('EVAL', '1') == '0':
        log('done fold', FOLD)
        return

    results = {'base': BASE, 'tag': TAG, 'addr_drop': ADDR_DROP, 'lr': LR, 'batch': BATCH, 'mask': MASK,
               'epochs': EPOCHS, 'n_train_ce': int(len(train_idx)), 'train_seconds': round(time.time() - t0)}
    ce_prob = 1 / (1 + np.exp(-ce.astype(np.float64)))
    evaluate('cross_encoder_alone', np.nan_to_num(ce_prob), refs, tgts, gold, partition, results)

    report = json.loads((VISHAL / 'prune_v1/model_top10/report.json').read_text())
    x = np.asarray(np.load(V4 / 'pairs/x.npy', mmap_mode='r')[rows])
    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                  grow_policy='lossguide', max_depth=0, max_leaves=255, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=1.0, max_bin=256, seed=2026)
    tb, dv = part == 'trainB', part == 'dev'
    for name, feats in (('xgb_baseline_foldB', x), ('xgb_plus_ce_foldB', np.column_stack([x, ce]))):
        dtr = xgb.QuantileDMatrix(feats[tb], y[tb])
        ddv = xgb.QuantileDMatrix(feats[dv], y[dv], ref=dtr)
        bst = xgb.train(params, dtr, 4000, evals=[(ddv, 'dev')], early_stopping_rounds=100, verbose_eval=False)
        p = bst.inplace_predict(feats, iteration_range=(0, bst.best_iteration + 1))
        p = np.asarray(p, dtype=np.float64)
        evaluate(name, p, refs, tgts, gold, partition, results)
        ru, th = results[name]['rule']
        hm = part == 'holdout'
        prefs = sorted(s for s in gold if partition[s] == 'holdout')
        pos = {s: i for i, s in enumerate(prefs)}
        ri = np.array([pos[r] for r in refs[hm]])
        ch = decide(ri, p[hm], ru, th, len(prefs))
        ea, yh = empty_addr[hm], y[hm]
        results[name]['empty_addr_holdout'] = {
            'true_pairs': int((ea & (yh == 1)).sum()), 'recall': float((ch & ea & (yh == 1)).sum() / max(1, (ea & (yh == 1)).sum())),
            'false_accepted': int((ch & ea & (yh == 0)).sum()),
            'recall_with_address': float((ch & ~ea & (yh == 1)).sum() / max(1, (~ea & (yh == 1)).sum())),
            'false_accepted_with_address': int((ch & ~ea & (yh == 0)).sum())}
        results[name]['best_iteration'] = int(bst.best_iteration)
        if name == 'xgb_plus_ce_foldB':
            g = bst.get_score(importance_type='total_gain')
            results['ce_gain_rank'] = sorted(g, key=lambda k: -g[k]).index(f'f{feats.shape[1] - 1}') + 1 \
                if f'f{feats.shape[1] - 1}' in g else None
    (OUT / f'results_{Path(MASK).stem}.json').write_text(json.dumps(results, indent=1, default=str))
    log('done', results)


if __name__ == '__main__':
    main()
