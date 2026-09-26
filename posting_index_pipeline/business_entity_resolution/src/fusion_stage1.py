import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from rapidfuzz import fuzz
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from submission_pipeline import decide
from matcher_eda import partition_of, norm
from sibling_check import prefix_match, num_jacc

HOME = Path.home()
V4 = HOME / 'scratch/vishal_v4_artifacts'
VISHAL = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
OUT = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
MAX_ANCHORS = 3
MIN_SIB_PROB = 0.005


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
                rows.append(i); refs.append(r); tgts.append(t)
    rows, refs, tgts = np.array(rows), np.array(refs, dtype=object), np.array(tgts, dtype=object)
    part = np.array([partition[r] for r in refs], dtype=object)
    x = np.asarray(np.load(V4 / 'pairs/x.npy', mmap_mode='r')[rows])
    y = np.load(V4 / 'pairs/y.npy')[rows].astype(np.float32)
    log('rows', len(rows))

    full = xgb.Booster()
    full.load_model(str(model_dir / 'model.json'))
    full.set_param({'device': 'cuda'})
    n_iter = report['best_iteration'] + 1
    stage1 = np.asarray(full.inplace_predict(x, iteration_range=(0, n_iter)), dtype=np.float64)
    params = dict(report['lightgbm_params'])
    params['device'] = 'cuda'
    tr = np.flatnonzero(part == 'train')
    fold = np.array([int(hashlib.sha256((r + '#s1').encode()).hexdigest()[:8], 16) % 5 for r in refs[tr]])
    for k in range(5):
        fit, pred = tr[fold != k], tr[fold == k]
        b = xgb.train(params, xgb.QuantileDMatrix(x[fit], y[fit]), n_iter)
        stage1[pred] = b.inplace_predict(x[pred])
        log('oof fold', k)

    chosen = np.zeros(len(rows), dtype=bool)
    for p in ('train', 'dev', 'holdout'):
        prefs = sorted(s for s in gold if partition[s] == p)
        pos = {s: i for i, s in enumerate(prefs)}
        m = np.flatnonzero(part == p)
        chosen[m] = decide(np.array([pos[r] for r in refs[m]]), stage1[m], rule, thr, len(prefs))
    anchor = chosen & (stage1 >= 0.9)

    need = set(refs) | set(tgts)
    text = {}
    for src in ('train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv'):
        df = pd.read_csv(RAW / src, sep='\t', dtype=str, keep_default_na=False, quoting=3)
        df = df[df.entity_id.isin(need)]
        for e, n, a in zip(df.entity_id, df.business_name, df.business_address):
            text[e] = (n, a)
    log('text loaded')

    tok = AutoTokenizer.from_pretrained(VISHAL / 'dense_v1/model')
    enc = AutoModel.from_pretrained(VISHAL / 'dense_v1/model').cuda().eval()
    uniq = sorted(set(tgts))
    tid = {t: i for i, t in enumerate(uniq)}
    emb = np.zeros((len(uniq), enc.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(uniq), 2048):
            batch = [f'passage: {text.get(t, ("", ""))[0]} {text.get(t, ("", ""))[1]}' for t in uniq[b:b + 2048]]
            e = tok(batch, truncation=True, max_length=96, padding=True, return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = enc(**e).last_hidden_state
            m = e['attention_mask'].unsqueeze(-1).float()
            emb[b:b + 2048] = torch.nn.functional.normalize((h.float() * m).sum(1) / m.sum(1), dim=-1).cpu().numpy()
    log('embedded', len(uniq))

    by_ref = defaultdict(list)
    for j in range(len(rows)):
        by_ref[refs[j]].append(j)
    ctx = np.zeros((len(rows), 7), dtype=np.float32)
    sib = np.full((len(rows), 8), -1.0, dtype=np.float32)
    pairpre = np.zeros(len(rows), dtype=np.float32)
    sib_pairs = []
    ntext = {t: (norm(text.get(t, ('', ''))[0]), norm(text.get(t, ('', ''))[1])) for t in need}
    for r, js in by_ref.items():
        js = sorted(js, key=lambda j: -stage1[j])
        p = stage1[js]
        top, s, n9, n5 = p[0], p.sum(), (p >= 0.9).sum(), (p >= 0.5).sum()
        anc = [j for j in js if anchor[j]]
        ra = text.get(r, ('', ''))[1]
        for rk, j in enumerate(js):
            nxt = p[rk + 1] if rk + 1 < len(p) else 0.0
            ctx[j] = [stage1[j], rk + 1, top - stage1[j], stage1[j] - nxt, s, n9, n5]
            ta = text.get(tgts[j], ('', ''))[1]
            pairpre[j] = prefix_match(ra, ta)
            an = [a for a in anc if a != j][:MAX_ANCHORS]
            if not an:
                continue
            if stage1[j] >= MIN_SIB_PROB:
                sib_pairs.extend((j, a) for a in an)
            tn, tna = ntext[tgts[j]]
            F = []
            for a in an:
                an_n, an_a = ntext[tgts[a]]
                raw_a = text.get(tgts[a], ('', ''))[1]
                F.append([fuzz.token_sort_ratio(tn, an_n) / 100,
                          fuzz.token_set_ratio(tna, an_a) / 100 if tna and an_a else -1.0,
                          num_jacc(ta, raw_a), prefix_match(ta, raw_a),
                          float(emb[tid[tgts[j]]] @ emb[tid[tgts[a]]]), float(tgts[j][:2] != tgts[a][:2])])
            F = np.array(F)
            sib[j] = [F[:, 0].max(), F[:, 1].max(), F[:, 2].max(), F[:, 3].max(), F[:, 4].max(), F[:, 4].mean(),
                      F[:, 5].mean(), len(an)]
    sib_pairs = np.array(sib_pairs, dtype=np.int64).reshape(-1, 2)
    log('features built; sibling pairs', len(sib_pairs))
    np.savez(OUT / 'stage1.npz', rows=rows, stage1=stage1, chosen=chosen, anchor=anchor, ctx=ctx, sib=sib,
             pairpre=pairpre, sib_pairs=sib_pairs)
    log('saved', OUT / 'stage1.npz')


if __name__ == '__main__':
    main()
