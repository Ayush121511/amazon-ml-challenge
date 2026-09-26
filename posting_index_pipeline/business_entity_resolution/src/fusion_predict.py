"""Test inference for the cross-encoder fusion matcher.

Input dir (--pairs): pairs.tsv.gz (ref \\t target, one line per candidate), x.npy (N x 103 float32, the v4 feature
order in prune_v1/model_top10/report.json). Text comes from the organizer test TSVs.
Stages: stage-1 top-10 matcher -> anchors -> sibling features -> cross-encoder scores (fold A/B mean) -> fusion
XGBoost -> finalize() (global exclusivity, decision rule, TSVs). Probabilities are saved so `--rerule` can re-run
the decision with a different rule/floor in minutes.
"""
import argparse
import gzip
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from rapidfuzz import fuzz
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from submission_pipeline import decide, finalize, parse_country_thresholds
from matcher_eda import norm
from sibling_check import prefix_match, num_jacc

HOME = Path.home()
VISHAL = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
CE = HOME / 'scratch/amazon_ml_2026/artifacts/cross_encoder_v1'
FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/test'
MAX_ANCHORS, MIN_SIB_PROB, MAX_LEN = 3, 0.005, 128


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def ce_score(model_dir, L, R):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
    out = np.zeros(len(L), dtype=np.float32)
    order = np.argsort([len(a) + len(b) for a, b in zip(L, R)], kind='stable')
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), 1024):
            ids = order[b:b + 1024]
            e = tok([L[i] for i in ids], [R[i] for i in ids], truncation=True, max_length=MAX_LEN, padding=True,
                    return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out[ids] = model(**e).logits.squeeze(-1).float().cpu().numpy()
            if b % (1024 * 2000) == 0:
                log(f'  {model_dir.name}: {b}/{len(order)} {time.time() - t0:.0f}s')
    del model
    torch.cuda.empty_cache()
    return out


def refs_and_text():
    text, refs = {}, []
    for src in ('test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv'):
        df = pd.read_csv(RAW / src, sep='\t', dtype=str, keep_default_na=False, quoting=3)
        for e, n, a in zip(df.entity_id, df.business_name, df.business_address):
            text[e] = (n, a)
        if src == 'test_source1.tsv':
            refs = [{'entity_id': e, 'country': c} for e, c in zip(df.entity_id, df.country)]
        log('text', src, len(df))
    return refs, text


def score_all(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((FUS / 'models' / a.config / 'config.json').read_text())
    log('config', a.config, cfg)
    pr, tg = [], []
    with gzip.open(Path(a.pairs) / 'pairs.tsv.gz', 'rt') as f:
        for line in f:
            r, t = line.rstrip('\n').split('\t')
            pr.append(r); tg.append(t)
    refs_arr, tgts = np.array(pr, dtype=object), np.array(tg, dtype=object)
    x = np.load(Path(a.pairs) / 'x.npy', mmap_mode='r')
    assert x.shape[0] == len(refs_arr), (x.shape, len(refs_arr))
    x = np.asarray(x, dtype=np.float32)
    log('pairs', len(refs_arr))
    refs, text = refs_and_text()
    position = {r['entity_id']: i for i, r in enumerate(refs)}
    ref_idx = np.array([position[r] for r in refs_arr], dtype=np.int64)

    rep = json.loads((VISHAL / 'prune_v1/model_top10/report.json').read_text())
    s1m = xgb.Booster()
    s1m.load_model(str(VISHAL / 'prune_v1/model_top10/model.json'))
    s1m.set_param({'device': 'cuda'})
    stage1 = np.asarray(s1m.inplace_predict(x, iteration_range=(0, rep['best_iteration'] + 1)), dtype=np.float64)
    chosen = decide(ref_idx, stage1, rep['decision']['rule'], rep['decision']['threshold'], len(refs))
    anchor = chosen & (stage1 >= 0.9)
    log('stage1 done; anchors', int(anchor.sum()))

    blocks = cfg['feature_blocks']
    feats = {'base': x}
    if 'ctx_sibrf' in blocks or any(b.startswith('sibce_') for b in blocks):
        ntext = {}
        for t in set(tgts):
            n_, ad = text.get(t, ('', ''))
            ntext[t] = (norm(n_), norm(ad))
        emb = None
        if 'ctx_sibrf' in blocks:
            tok = AutoTokenizer.from_pretrained(VISHAL / 'dense_v1/model')
            enc = AutoModel.from_pretrained(VISHAL / 'dense_v1/model').cuda().eval()
            uniq = sorted(set(tgts))
            tid = {t: i for i, t in enumerate(uniq)}
            emb = np.zeros((len(uniq), enc.config.hidden_size), dtype=np.float32)
            with torch.no_grad():
                for b in range(0, len(uniq), 2048):
                    bt = [f'passage: {text.get(t, ("", ""))[0]} {text.get(t, ("", ""))[1]}' for t in uniq[b:b + 2048]]
                    e = tok(bt, truncation=True, max_length=96, padding=True, return_tensors='pt').to('cuda')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        h = enc(**e).last_hidden_state
                    m = e['attention_mask'].unsqueeze(-1).float()
                    emb[b:b + 2048] = torch.nn.functional.normalize((h.float() * m).sum(1) / m.sum(1), dim=-1).cpu().numpy()
            del enc
            log('embedded targets', len(uniq))
        by_ref = defaultdict(list)
        for j, r in enumerate(ref_idx):
            by_ref[r].append(j)
        ctx = np.zeros((len(tgts), 7), dtype=np.float32)
        sib = np.full((len(tgts), 8), -1.0, dtype=np.float32)
        pairpre = np.zeros(len(tgts), dtype=np.float32)
        sib_pairs = []
        for r, js in by_ref.items():
            js = sorted(js, key=lambda j: -stage1[j])
            p = stage1[js]
            top, s, n9, n5 = p[0], p.sum(), (p >= 0.9).sum(), (p >= 0.5).sum()
            anc = [j for j in js if anchor[j]]
            ra = text.get(refs[r]['entity_id'], ('', ''))[1]
            for rk, j in enumerate(js):
                nxt = p[rk + 1] if rk + 1 < len(p) else 0.0
                ctx[j] = [stage1[j], rk + 1, top - stage1[j], stage1[j] - nxt, s, n9, n5]
                ta = text.get(tgts[j], ('', ''))[1]
                pairpre[j] = prefix_match(ra, ta)
                an = [q for q in anc if q != j][:MAX_ANCHORS]
                if not an:
                    continue
                if stage1[j] >= MIN_SIB_PROB:
                    sib_pairs.extend((j, q) for q in an)
                if emb is None:
                    continue
                tn, tna = ntext[tgts[j]]
                F = []
                for q in an:
                    an_n, an_a = ntext[tgts[q]]
                    raw_a = text.get(tgts[q], ('', ''))[1]
                    F.append([fuzz.token_sort_ratio(tn, an_n) / 100,
                              fuzz.token_set_ratio(tna, an_a) / 100 if tna and an_a else -1.0,
                              num_jacc(ta, raw_a), prefix_match(ta, raw_a),
                              float(emb[tid[tgts[j]]] @ emb[tid[tgts[q]]]), float(tgts[j][:2] != tgts[q][:2])])
                F = np.array(F)
                sib[j] = [F[:, 0].max(), F[:, 1].max(), F[:, 2].max(), F[:, 3].max(), F[:, 4].max(),
                          F[:, 4].mean(), F[:, 5].mean(), len(an)]
        sib_pairs = np.array(sib_pairs, dtype=np.int64).reshape(-1, 2)
        feats['ctx_sibrf'] = np.column_stack([ctx, sib, pairpre])
        log('set/sibling features; sibling pairs', len(sib_pairs))

    fmt = lambda na: f'{na[0]} | {na[1]}'
    L = [fmt(text.get(refs[r]['entity_id'], ('', ''))) for r in ref_idx]
    Rt = [fmt(text.get(t, ('', ''))) for t in tgts]
    for tag in cfg['tags']:
        if f'ce_{tag}' not in blocks and f'sibce_{tag}' not in blocks:
            continue
        folds = [CE / tag / f'model_fold{k}' for k in ('A', 'B')]
        feats[f'ce_{tag}'] = np.mean([ce_score(d, L, Rt) for d in folds], axis=0)[:, None]
        np.save(out / f'ce_{tag}.npy', feats[f'ce_{tag}'][:, 0])
        if f'sibce_{tag}' in blocks and len(sib_pairs):
            SL = [Rt[j] for j in sib_pairs[:, 0]]
            SR = [Rt[q] for q in sib_pairs[:, 1]]
            sl = np.mean([ce_score(d, SL, SR) for d in folds], axis=0)
            mx = np.full(len(tgts), -20.0, dtype=np.float32)
            sm, cnt = np.zeros(len(tgts), np.float32), np.zeros(len(tgts), np.float32)
            np.maximum.at(mx, sib_pairs[:, 0], sl)
            np.add.at(sm, sib_pairs[:, 0], sl)
            np.add.at(cnt, sib_pairs[:, 0], 1)
            feats[f'sibce_{tag}'] = np.column_stack([mx, np.where(cnt > 0, sm / np.maximum(cnt, 1), -20.0)])
        log('cross-encoder done', tag)

    F = np.column_stack([feats[b] for b in blocks]).astype(np.float32)
    fm = xgb.Booster()
    fm.load_model(str(FUS / 'models' / a.config / 'model.json'))
    fm.set_param({'device': 'cuda'})
    prob = np.asarray(fm.inplace_predict(F, iteration_range=(0, cfg['best_iteration'] + 1)), dtype=np.float32)
    cands = defaultdict(list)
    for r, t in zip(ref_idx, tgts):
        cands[r].append(t)
    np.savez(out / 'scores.npz', ref_idx=ref_idx, target=tgts.astype(str), prob=prob, stage1=stage1.astype(np.float32))
    with open(out / 'candidates.tsv', 'w', encoding='utf-8') as f:
        for i, r in enumerate(refs):
            f.write(r['entity_id'] + '\t' + ','.join(cands.get(i, [])) + '\n')
    log('fusion scored', len(prob))
    return cfg


def run_finalize(a, cfg):
    out = Path(a.output)
    refs, _ = refs_and_text()
    S = np.load(out / 'scores.npz', allow_pickle=True)
    cands_out = {}
    with open(out / 'candidates.tsv', encoding='utf-8') as f:
        for line in f:
            sid, c = line.rstrip('\n').split('\t')
            cands_out[sid] = c
    rule, thr = cfg['rule']
    if a.rule:
        rule, thr = a.rule.split('@')[0], float(a.rule.split('@')[1])
    dest = out / (a.tag or f'{rule}_{thr}')
    dest.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    finalize(refs, cands_out, S['ref_idx'], S['target'].astype(object), S['prob'],
             {'decision': {'rule': rule, 'threshold': thr}}, True, dest, stats,
             {'config': a.config, 'rule': [rule, thr], 'started': time.monotonic()}, parse_country_thresholds(a.country_threshold))
    log('wrote', dest, dict(matches=stats.get('matches')))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='all')
    p.add_argument('--rerule', action='store_true', help='skip scoring; re-run the decision from scores.npz')
    p.add_argument('--rule', default='', help='override, e.g. expected@0.9')
    p.add_argument('--country-threshold', default='')
    p.add_argument('--tag', default='')
    a = p.parse_args()
    cfg = json.loads((FUS / 'models' / a.config / 'config.json').read_text()) if a.rerule else score_all(a)
    run_finalize(a, cfg)


if __name__ == '__main__':
    main()
