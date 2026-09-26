"""Distractor-aware validation: does the tonight model (L2 late fusion) hold up on denser candidate pools?

Scores dev/holdout on the top-20 set with the same models used on top-10 (stage-1 = prune_v1/model_top10 floored
at 0.02, cross-encoder = mdeberta fold A, fusion = L2). Compares F0.5, false links and the best rule between top-10
and top-20 pools, and measures how visible target competition is in the sampled train pairs.
"""
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SCR = Path(os.environ.get('ER_SCRATCH', str(Path.home() / 'scratch')))
V4 = SCR / 'vishal_v4_artifacts'
PRUNE = SCR / 'amazon_ml_2026/artifacts/prune_v1_copy'
FUS = SCR / 'amazon_ml_2026/artifacts/fusion'
CE = SCR / 'amazon_ml_2026/artifacts/cross_encoder_v1'
RAW = SCR / 'amazon_ml_2026/data/dataset/train'
OUT = Path(os.environ.get('OUT_DIR', str(FUS)))


def decide(ref_idx, prob, rule, threshold, n_refs):
    """Selected mask. 'threshold': p >= t. 'expected': per-reference top-k maximising
    approximate expected F0.5, and only if it beats the no-match expectation; t is a floor."""
    selected = prob >= threshold
    if rule == 'threshold':
        return selected
    order = np.lexsort((-prob, ref_idx))
    result = np.zeros(len(prob), dtype=bool)
    bounds = np.searchsorted(ref_idx[order], np.arange(n_refs + 1))
    for i in range(n_refs):
        idx = order[bounds[i]:bounds[i + 1]]
        if not len(idx):
            continue
        p = prob[idx].astype(np.float64)
        expected_size = p.sum()
        k = np.arange(1, len(p) + 1)
        value = 1.25 * np.cumsum(p) / (0.25 * expected_size + k)
        best = int(np.argmax(value))
        empty = float(np.prod(1 - np.clip(p, 0, 1 - 1e-9)))
        if value[best] > empty and p[0] >= threshold:
            result[idx[:best + 1]] = True
    return result


def f05(truth, pred):
    if not truth and not pred:
        return 1.0
    tp = len(truth & pred)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)



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


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    cp = partition_of(gold)
    m10 = np.load(PRUNE / 'mask_top10.npy')
    m20 = np.load(PRUNE / 'mask_top20.npy')
    rows, refs, tgts = [], [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if m20[i]:
                r, t = line.rstrip('\n').split('\t')
                rows.append(i); refs.append(r); tgts.append(t)
    rows, refs, tgts = np.array(rows), np.array(refs, dtype=object), np.array(tgts, dtype=object)
    part = np.array([cp[r] for r in refs], dtype=object)
    in10 = m10[rows]
    y = np.load(V4 / 'pairs/y.npy')[rows].astype(np.float32)
    R = {}

    owned = set()
    with open(RAW / 'train_ground_truth.tsv') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) > 1 and p[1]:
                owned.update(t.strip() for t in p[1].split(','))
    n_ref_of_target = Counter(tgts[in10])
    R['density'] = {}
    for name, m in (('top10', in10), ('top20', np.ones(len(rows), bool))):
        mm = m & np.isin(part, ['dev', 'holdout'])
        nref = len(set(refs[mm]))
        neg = mm & (y == 0)
        dis = neg & np.array([t not in owned for t in tgts])
        R['density'][name] = {'candidates_per_ref': round(mm.sum() / nref, 3),
                              'negatives_per_ref': round(neg.sum() / nref, 3),
                              'unowned_distractors_per_ref': round(dis.sum() / nref, 3)}
    R['target_competition_in_sample'] = {
        'targets_in_top10': len(n_ref_of_target),
        'share_targets_seen_for_1_ref': round(sum(v == 1 for v in n_ref_of_target.values()) / len(n_ref_of_target), 4),
        'share_seen_for_2plus_refs': round(sum(v >= 2 for v in n_ref_of_target.values()) / len(n_ref_of_target), 4),
        'note': 'pairs cover 100k sampled S1 references of 2.2M, so competition between references is mostly unobserved'}
    log('density', R['density'], R['target_competition_in_sample'])

    ev = np.flatnonzero(np.isin(part, ['dev', 'holdout']))
    x = np.asarray(np.load(V4 / 'pairs/x.npy', mmap_mode='r')[rows[ev]])
    rep = json.loads((PRUNE / 'model_top10/report.json').read_text())
    s1m = xgb.Booster()
    s1m.load_model(str(PRUNE / 'model_top10/model.json'))
    s1m.set_param({'device': 'cuda'})
    s1 = np.asarray(s1m.inplace_predict(x, iteration_range=(0, rep['best_iteration'] + 1)), dtype=np.float64)
    s1 = np.where(s1 < 0.02, 0.0, s1)

    need = set(refs[ev]) | set(tgts[ev])
    text = {}
    for src in ('train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv'):
        df = pd.read_csv(RAW / src, sep='\t', dtype=str, keep_default_na=False, quoting=3)
        df = df[df.entity_id.isin(need)]
        for e, n, a in zip(df.entity_id, df.business_name, df.business_address):
            text[e] = f'{n} | {a}'
    L = [text.get(r, ' | ') for r in refs[ev]]
    Rt = [text.get(t, ' | ') for t in tgts[ev]]
    md = CE / 'mdeberta_drop0/model'
    tok = AutoTokenizer.from_pretrained(md)
    model = AutoModelForSequenceClassification.from_pretrained(md).cuda().eval()
    ce = np.zeros(len(ev), dtype=np.float32)
    order = np.argsort([len(a) + len(b) for a, b in zip(L, Rt)], kind='stable')
    with torch.no_grad():
        for b in range(0, len(order), 1024):
            ids = order[b:b + 1024]
            e = tok([L[i] for i in ids], [Rt[i] for i in ids], truncation=True, max_length=128, padding=True,
                    return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ce[ids] = model(**e).logits.squeeze(-1).float().cpu().numpy()
    log('scored', len(ev))

    late = FUS / 'late_models/L2_xgb_stage1_ce_mdeberta_drop0'
    cfg = json.loads((late / 'config.json').read_text())
    lb = xgb.Booster()
    lb.load_model(str(late / 'model.json'))
    lb.set_param({'device': 'cuda'})
    prob = np.asarray(lb.inplace_predict(np.column_stack([s1, ce]).astype(np.float32),
                                         iteration_range=(0, cfg['best_iteration'] + 1)), dtype=np.float64)

    grid = [(ru, round(float(t), 3)) for ru in ('expected', 'threshold') for t in np.arange(0.5, 0.981, 0.02)]
    grid.append(tuple(cfg['rule']))

    def run(p_name, pool_mask):
        m = (part[ev] == p_name) & pool_mask
        prefs = sorted(s for s in gold if cp[s] == p_name)
        pos = {s: i for i, s in enumerate(prefs)}
        ri = np.array([pos[r] for r in refs[ev][m]])
        pr, tg, yy = prob[m], tgts[ev][m], y[ev][m]

        def score(ru, t):
            c = decide(ri, pr, ru, t, len(prefs))
            pred = defaultdict(set)
            for i, tt in zip(ri[c], tg[c]):
                pred[i].add(tt)
            f = float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs)]))
            return f, int((c & (yy == 0)).sum()), int(((~c) & (yy == 1)).sum())
        return score

    out = {}
    for pool, pm in (('top10', in10[ev]), ('top20', np.ones(len(ev), bool))):
        sdev, shold = run('dev', pm), run('holdout', pm)
        dev = {g: sdev(*g) for g in grid}
        best = max(dev, key=lambda k: (dev[k][0], k[1]))
        fixed = tuple(cfg['rule'])
        out[pool] = {'rule_from_top10_dev': fixed,
                     'dev_at_top10_rule': dev[fixed], 'holdout_at_top10_rule': shold(*fixed),
                     'best_rule_on_this_pool_dev': best, 'dev_at_best': dev[best], 'holdout_at_best': shold(*best),
                     'dev_curve': {f'{r}@{t}': v[0] for (r, t), v in sorted(dev.items())}}
        log(pool, {k: v for k, v in out[pool].items() if k != 'dev_curve'})
    R['l2_top10_vs_top20'] = out
    R['note'] = ('(f05, false_links, missed_true_in_pool) per entry. The same models trained on top-10 are applied to the '
                 'denser top-20 pools. A drop at the top-10 rule that a stricter rule recovers means test (denser) needs '
                 'a stricter rule than dev picks.')
    (OUT / 'distractor_check.json').write_text(json.dumps(R, indent=1, default=str))
    log('done')


if __name__ == '__main__':
    main()
