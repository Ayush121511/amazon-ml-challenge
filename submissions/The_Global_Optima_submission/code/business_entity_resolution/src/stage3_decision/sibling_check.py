import gzip
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from rapidfuzz import fuzz
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from submission_pipeline import decide, f05
from matcher_eda import partition_of, norm

HOME = Path.home()
V4 = HOME / 'scratch/vishal_v4_artifacts'
VISHAL = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
OUT = HOME / 'scratch/amazon_ml_2026/artifacts/matcher_eda'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def nums(s):
    return re.findall(r'\d+', s or '')


def prefix_match(a, b):
    a, b = nums(a), nums(b)
    if not a or not b:
        return -1.0
    return float(any(x == y or (len(x) != len(y) and (x.startswith(y) or y.startswith(x)))
                     for x in a for y in b))


def num_jacc(a, b):
    a, b = set(nums(a)), set(nums(b))
    if not a or not b:
        return -1.0
    return len(a & b) / len(a | b)


def main():
    model_dir = VISHAL / 'prune_v1/model_top10'
    report = json.loads((model_dir / 'report.json').read_text())
    rule, thr = report['decision']['rule'], report['decision']['threshold']
    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    partition = partition_of(gold)
    mask = np.load(VISHAL / 'prune_v1/mask_top10.npy')
    rows, refs, tgts = [], [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t')
                if partition[r] == 'holdout':
                    rows.append(i); refs.append(r); tgts.append(t)
    rows, refs, tgts = np.array(rows), np.array(refs, dtype=object), np.array(tgts, dtype=object)
    x = np.asarray(np.load(V4 / 'pairs/x.npy', mmap_mode='r')[rows])
    y = np.load(V4 / 'pairs/y.npy')[rows].astype(np.float32)
    bst = xgb.Booster()
    bst.load_model(str(model_dir / 'model.json'))
    bst.set_param({'device': 'cuda'})
    prob = np.asarray(bst.inplace_predict(x, iteration_range=(0, report['best_iteration'] + 1)), dtype=np.float64)
    hold_refs = sorted(s for s in gold if partition[s] == 'holdout')
    pos = {s: i for i, s in enumerate(hold_refs)}
    ref_idx = np.array([pos[s] for s in refs])
    chosen = decide(ref_idx, prob, rule, thr, len(hold_refs))
    log('rows', len(rows))

    need = set(tgts) | set(refs)
    text = {}
    for src in ('train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv'):
        df = pd.read_csv(RAW / src, sep='\t', dtype=str, keep_default_na=False, quoting=3)
        df = df[df.entity_id.isin(need)]
        for e, n, a in zip(df.entity_id, df.business_name, df.business_address):
            text[e] = (n, a)
    log('text loaded')

    tok = AutoTokenizer.from_pretrained(VISHAL / 'dense_v1/model')
    enc_model = AutoModel.from_pretrained(VISHAL / 'dense_v1/model').cuda().eval()
    uniq = sorted(set(tgts))
    tid = {t: i for i, t in enumerate(uniq)}
    emb = np.zeros((len(uniq), enc_model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(uniq), 1024):
            batch = [f'passage: {text.get(t, ("", ""))[0]} {text.get(t, ("", ""))[1]}' for t in uniq[b:b + 1024]]
            e = tok(batch, truncation=True, max_length=96, padding=True, return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = enc_model(**e).last_hidden_state
            m = e['attention_mask'].unsqueeze(-1).float()
            v = (h.float() * m).sum(1) / m.sum(1)
            emb[b:b + 1024] = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
    log('embedded', len(uniq))

    by_ref = defaultdict(list)
    for j in range(len(rows)):
        by_ref[refs[j]].append(j)
    ctx = np.zeros((len(rows), 7), dtype=np.float32)
    sib = np.full((len(rows), 8), -1.0, dtype=np.float32)
    pairpre = np.zeros(len(rows), dtype=np.float32)
    for r, js in by_ref.items():
        js = sorted(js, key=lambda j: -prob[j])
        p = prob[js]
        top, s = p[0], p.sum()
        n9, n5 = (p >= 0.9).sum(), (p >= 0.5).sum()
        anchors = [j for j in js if chosen[j] and prob[j] >= 0.9]
        rn, ra = text.get(r, ('', ''))
        for rk, j in enumerate(js):
            nxt = p[rk + 1] if rk + 1 < len(p) else 0.0
            ctx[j] = [prob[j], rk + 1, top - prob[j], prob[j] - nxt, s, n9, n5]
            tn, ta = text.get(tgts[j], ('', ''))
            pairpre[j] = prefix_match(ra, ta)
            an = [a for a in anchors if a != j]
            if not an:
                continue
            feats = []
            for a in an:
                an_n, an_a = text.get(tgts[a], ('', ''))
                feats.append([fuzz.token_sort_ratio(norm(tn), norm(an_n)) / 100,
                              fuzz.token_set_ratio(norm(ta), norm(an_a)) / 100 if ta and an_a else -1.0,
                              num_jacc(ta, an_a), prefix_match(ta, an_a),
                              float(emb[tid[tgts[j]]] @ emb[tid[tgts[a]]]),
                              float(tgts[j][:2] != tgts[a][:2])])
            F = np.array(feats)
            sib[j] = [F[:, 0].max(), F[:, 1].max(), F[:, 2].max(), F[:, 3].max(), F[:, 4].max(), F[:, 4].mean(),
                      F[:, 5].mean(), len(an)]
    log('features built')

    bucket = np.full(len(rows), 'TN', dtype=object)
    bucket[chosen & (y == 1)] = 'TP'
    bucket[chosen & (y == 0)] = 'FP'
    bucket[~chosen & (y == 1)] = 'FN'
    bucket[(~chosen) & (y == 0) & (prob >= 0.1)] = 'HARD_NEG'
    sib_names = ['sib_name_max', 'sib_addr_max', 'sib_numjacc_max', 'sib_prefix_max', 'sib_emb_max', 'sib_emb_mean',
                 'sib_cross_source', 'n_anchors']
    R = {'anchor_rule': 'accepted and p>=0.9, excluding the candidate itself'}
    med = {}
    for b in ('TP', 'FN', 'FP', 'HARD_NEG'):
        m = bucket == b
        has = m & (sib[:, 7] > 0)
        med[b] = {'n': int(m.sum()), 'with_anchor': int(has.sum())}
        for k, nm in enumerate(sib_names[:7]):
            v = sib[has, k]
            v = v[v >= 0]
            med[b][nm] = round(float(np.median(v)), 4) if len(v) else None
        v = pairpre[m]
        v = v[v >= 0]
        med[b]['pair_number_prefix_match'] = round(float(v.mean()), 4) if len(v) else None
        v2 = x[m, report['features'].index('address_numbers_token_set')]
        med[b]['pair_numbers_token_set_mean'] = round(float(v2[v2 >= 0].mean()), 4) if (v2 >= 0).any() else None
    R['medians_by_bucket'] = med

    unsel = (~chosen) & (sib[:, 7] > 0) & (prob >= 0.01)
    lab = y[unsel]
    R['rescue_pool'] = {'pairs': int(unsel.sum()), 'true': int(lab.sum())}
    auc = {}
    for k, nm in enumerate(sib_names[:6]):
        auc[nm] = round(float(roc_auc_score(lab, sib[unsel, k])), 4)
    auc['own_prob'] = round(float(roc_auc_score(lab, prob[unsel])), 4)
    auc['pair_number_prefix_match'] = round(float(roc_auc_score(lab, pairpre[unsel])), 4)
    R['auc_missed_vs_negative_in_rescue_pool'] = auc

    folds = np.array([int(pos[r]) % 5 for r in refs])
    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                  max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=1)
    variants = {'A_set_context': ctx,
                'B_set_context_plus_sibling': np.column_stack([ctx, sib, pairpre]),
                'C_B_plus_original_features': np.column_stack([ctx, sib, pairpre, x])}
    grid = [(ru, round(float(t), 2)) for ru in ('expected', 'threshold') for t in np.arange(0.3, 0.96, 0.05)]

    def score(p):
        best = None
        for ru, t in grid:
            c = decide(ref_idx, p, ru, t, len(hold_refs))
            pred = defaultdict(set)
            for i, tg in zip(ref_idx[c], tgts[c]):
                pred[i].add(tg)
            sc = float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(hold_refs)]))
            if best is None or sc > best[0]:
                best = (sc, ru, t)
        return best
    R['baseline_best_grid'] = score(prob)
    for name, F in variants.items():
        oof = np.zeros(len(rows))
        for k in range(5):
            tr, te = folds != k, folds == k
            d = xgb.DMatrix(F[tr], y[tr])
            b = xgb.train(params, d, 400)
            oof[te] = b.inplace_predict(F[te])
        R[name] = score(oof)
        log(name, R[name])
    R['note'] = ('Stage-2 models are 5-fold cross-validated within holdout (folds by reference); all rows use the '
                 'same decision grid, so baseline and variants are compared on equal footing. The grid choice itself '
                 'is made on holdout, so absolute values are slightly optimistic; the differences are what matter.')
    (OUT / 'sibling_check.json').write_text(json.dumps(R, indent=1, default=str))
    log('done', json.dumps(R, default=str))


if __name__ == '__main__':
    main()
