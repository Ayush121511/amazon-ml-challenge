"""Late-fusion test inference: the saved v5 probabilities + our cross-encoder test scores.

Features match late_fusion_check.py L3: [stage-1 set context (floored), sibling rapidfuzz/embedding features,
number-prefix match, cross-encoder logit, cross-encoder set context]. Decision via the pipeline's finalize()
(global exclusivity + rule). `--rerule` re-runs only the decision from the saved late probabilities.
"""
import argparse
import glob
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from rapidfuzz import fuzz
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from submission_pipeline import decide, finalize, parse_country_thresholds
from matcher_eda import norm
from sibling_check import prefix_match, num_jacc
from score_ce_test import read_pairs

VISHAL = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
FUS = Path.home() / 'scratch/amazon_ml_2026/artifacts/fusion'
RAW = Path.home() / 'scratch/amazon_ml_2026/data/dataset/test'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_refs_text():
    text, refs = {}, []
    for src in ('test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv'):
        df = pd.read_csv(RAW / src, sep='\t', dtype=str, keep_default_na=False, quoting=3)
        for e, n, ad in zip(df.entity_id, df.business_name, df.business_address):
            text[e] = (n, ad)
        if src == 'test_source1.tsv':
            refs = [{'entity_id': e, 'country': c} for e, c in zip(df.entity_id, df.country)]
    return refs, text


def pair_lookup(files, refs_arr, tgts, field):
    key = {}
    for fn in files:
        z = np.load(fn, allow_pickle=True)
        for r, t, v in zip(z['ref'], z['target'], z[field]):
            key[(str(r), str(t))] = float(v)
    return np.array([key.get((r, t), np.nan) for r, t in zip(refs_arr, tgts)], dtype=np.float64)


def score(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((Path(a.late_model) / 'config.json').read_text())
    refs_arr, tgts = read_pairs(a.candidates)
    log('pairs', len(refs_arr))
    refs, text = load_refs_text()
    position = {r['entity_id']: i for i, r in enumerate(refs)}
    ref_idx = np.array([position[r] for r in refs_arr], dtype=np.int64)

    s1_files = sorted({p for pat in a.stage1_scores.split(';') for p in glob.glob(pat)})
    log('stage-1 files', s1_files)
    s1 = pair_lookup(s1_files, refs_arr, tgts, 'prob')
    found = int((~np.isnan(s1)).sum())
    log('stage-1 probabilities found for', found, 'of', len(s1))
    # The saved scores drop pairs below 0.02 (treated as 0), so ~45% coverage is normal; 0 means broken paths.
    if len(s1_files) < 2 or found < 0.25 * len(s1):
        raise SystemExit(f'ABORT: {len(s1_files)} stage-1 files, probabilities found for {found}/{len(s1)} pairs')
    s1f = np.nan_to_num(s1, nan=0.0)
    s1f = np.where(s1f < 0.02, 0.0, s1f)
    ce = np.zeros(len(tgts))
    ce_files = sorted({p for pat in a.ce_scores.split(';') for p in glob.glob(pat)})
    names = sorted({Path(f).name.split('_shard')[0] for f in ce_files})
    for nm in names:
        ce += np.nan_to_num(pair_lookup([f for f in ce_files if Path(f).name.startswith(nm + '_shard')],
                                        refs_arr, tgts, 'logit'), nan=-20.0)
    ce /= max(1, len(names))
    log('cross-encoder models averaged:', names, 'files', ce_files)
    missing = int((ce <= -20 * 0.999).sum())
    if not names or missing > 0.1 * len(ce):
        raise SystemExit(f'ABORT: cross-encoder scores missing for {missing}/{len(ce)} pairs')

    if cfg['name'].startswith('L2'):
        X = np.column_stack([s1f, ce]).astype(np.float32)
        return finish(a, out, cfg, X, ref_idx, tgts, refs, s1f, ce)
    rep = json.loads((VISHAL / 'prune_v1/model_top10/report.json').read_text())
    chosen = decide(ref_idx, s1f, rep['decision']['rule'], rep['decision']['threshold'], len(refs))
    anchor = chosen & (s1f >= 0.9)
    uniq = sorted(set(tgts))
    tid = {t: i for i, t in enumerate(uniq)}
    tok = AutoTokenizer.from_pretrained(VISHAL / 'dense_v1/model')
    enc = AutoModel.from_pretrained(VISHAL / 'dense_v1/model').cuda().eval()
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
    log('embedded', len(uniq))

    ntext = {t: (norm(text.get(t, ('', ''))[0]), norm(text.get(t, ('', ''))[1])) for t in uniq}
    by_ref = defaultdict(list)
    for j, r in enumerate(ref_idx):
        by_ref[r].append(j)
    ctx = np.zeros((len(tgts), 7), dtype=np.float32)
    sib = np.full((len(tgts), 8), -1.0, dtype=np.float32)
    pairpre = np.zeros(len(tgts), dtype=np.float32)
    ce_ctx = np.zeros((len(tgts), 4), dtype=np.float32)
    for r, js in by_ref.items():
        c = ce[js]
        o = np.argsort(-c)
        rk_ce = np.empty(len(js)); rk_ce[o] = np.arange(1, len(js) + 1)
        ce_ctx[js] = np.column_stack([rk_ce, c.max() - c, np.full(len(js), (c > 0).sum()), np.full(len(js), c.max())])
        js = sorted(js, key=lambda j: -s1f[j])
        p = s1f[js]
        top, s, n9, n5 = p[0], p.sum(), (p >= 0.9).sum(), (p >= 0.5).sum()
        anc = [j for j in js if anchor[j]]
        ra = text.get(refs[r]['entity_id'], ('', ''))[1]
        for rk, j in enumerate(js):
            nxt = p[rk + 1] if rk + 1 < len(p) else 0.0
            ctx[j] = [s1f[j], rk + 1, top - s1f[j], s1f[j] - nxt, s, n9, n5]
            ta = text.get(tgts[j], ('', ''))[1]
            pairpre[j] = prefix_match(ra, ta)
            an = [q for q in anc if q != j][:3]
            if not an:
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
            sib[j] = [F[:, 0].max(), F[:, 1].max(), F[:, 2].max(), F[:, 3].max(), F[:, 4].max(), F[:, 4].mean(),
                      F[:, 5].mean(), len(an)]
    log('features built')
    X = np.column_stack([ctx, sib, pairpre, ce, ce_ctx]).astype(np.float32)
    return finish(a, out, cfg, X, ref_idx, tgts, refs, s1f, ce)


def finish(a, out, cfg, X, ref_idx, tgts, refs, s1f, ce):
    bst = xgb.Booster()
    bst.load_model(str(Path(a.late_model) / 'model.json'))
    bst.set_param({'device': 'cuda'})
    prob = np.asarray(bst.inplace_predict(X, iteration_range=(0, cfg['best_iteration'] + 1)), dtype=np.float32)
    cands = defaultdict(list)
    for r, t in zip(ref_idx, tgts):
        cands[r].append(t)
    np.savez(out / 'late_scores.npz', ref_idx=ref_idx, target=tgts.astype(str), prob=prob,
             stage1=s1f.astype(np.float32), ce=ce.astype(np.float32))
    with open(out / 'candidates.tsv', 'w', encoding='utf-8') as f:
        for i, r in enumerate(refs):
            f.write(r['entity_id'] + '\t' + ','.join(cands.get(i, [])) + '\n')
    log('late fusion scored', len(prob))
    return cfg


def decide_and_write(a, cfg):
    out = Path(a.output)
    refs, _ = load_refs_text()
    S = np.load(out / 'late_scores.npz', allow_pickle=True)
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
    finalize(refs, cands_out, S['ref_idx'], S['target'].astype(object), S['prob'],
             {'decision': {'rule': rule, 'threshold': thr}}, True, dest, Counter(),
             {'late_model': a.late_model, 'rule': [rule, thr], 'started': time.monotonic()},
             parse_country_thresholds(a.country_threshold))
    log('wrote', dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidates', required=True, help="candidate_pairs.tsv for the set")
    p.add_argument('--stage1-scores', required=True, help="glob of the scores.npz shards for the set")
    p.add_argument('--ce-scores', required=True, help='glob of our ce_*_shard*.npz files')
    p.add_argument('--late-model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--rerule', action='store_true')
    p.add_argument('--rule', default='')
    p.add_argument('--country-threshold', default='')
    p.add_argument('--tag', default='')
    a = p.parse_args()
    cfg = json.loads((Path(a.late_model) / 'config.json').read_text()) if a.rerule else score(a)
    decide_and_write(a, cfg)


if __name__ == '__main__':
    main()
