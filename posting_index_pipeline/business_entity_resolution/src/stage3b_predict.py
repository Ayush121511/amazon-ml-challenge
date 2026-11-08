"""Stage-3b on test: S5 candidates -> L2 probability -> list/global-name/reverse-competition/house-number features ->
stage-3b probability -> dev-chosen rule + global exclusivity (Vishal's finalize) -> submission folder.

Every feature is computed exactly as in boundary_exp.py / boundary_exp2.py, with the test split's own tables: name-key
counts over test S1/S2/S3, reverse competition against the full test S1 table (1.73M).
Env: S5_DIR (late_scores.npz + candidates.tsv of S5), MODEL_DIR (l2.json, stage3b.json + metas from boundary_exp2.py),
CE_TEST (optional ';'-separated npz files with ref/target/logit replacing S5's cross-encoder column, e.g. chunk2),
OUT (output dir), RULE (optional override 'expected@0.6'), NPROC.
"""
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from boundary_exp import NAMES, list_features, read_sources
from boundary_exp2 import DENSE, embed, first_num, rev_features
from late_predict import decide_and_write, load_refs_text, pair_lookup

HOME = Path.home()
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/test'
S5_DIR = Path(os.environ.get('S5_DIR', str(HOME / 'scratch/amazon_ml_2026/artifacts/submission_S5')))
MODEL_DIR = Path(os.environ['MODEL_DIR'])
OUT = Path(os.environ['OUT'])
G = {}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def feat_chunk(items):
    out = []
    for r, o in items:
        out.append((o, list_features(G['P'][o], G['s1f'][o], G['ce'][o], list(G['tg'][o]), G['refs'][r],
                                     G['text'], G['c1'], G['c23'])))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Z = np.load(S5_DIR / 'late_scores.npz', allow_pickle=True)
    ri, tg = Z['ref_idx'], Z['target'].astype(str).astype(object)
    s1f, ce = Z['stage1'].astype(np.float64), Z['ce'].astype(np.float64)
    refs_d, raw = load_refs_text()
    refs = [r['entity_id'] for r in refs_d]
    log('pairs', len(tg), 'refs', len(refs))
    if os.environ.get('CE_TEST'):
        ref_arr = np.array([refs[i] for i in ri], dtype=object)
        ce = pair_lookup(os.environ['CE_TEST'].split(';'), ref_arr, tg, 'logit')
        miss = int(np.isnan(ce).sum())
        log('CE_TEST missing', miss)
        if miss > 0.001 * len(ce):
            raise SystemExit(f'ABORT: cross-encoder missing for {miss} pairs')
        ce = np.nan_to_num(ce, nan=-10.0)
    ce2 = []
    ref_arr = np.array([refs[i] for i in ri], dtype=object)
    for member in [m for m in os.environ.get('CE2_TEST', '').split(';') if m]:
        v = pair_lookup(member.split('+'), ref_arr, tg, 'logit')
        miss = int(np.isnan(v).sum())
        log('CE2_TEST', member, 'missing', miss)
        if miss > 0.001 * len(v):
            raise SystemExit(f'ABORT: second cross-encoder missing for {miss} pairs')
        ce2.append(np.nan_to_num(v, nan=0.0))
    l2 = xgb.Booster(); l2.load_model(str(MODEL_DIR / 'l2.json')); l2.set_param({'device': 'cuda'})
    it2 = json.load(open(MODEL_DIR / 'l2_meta.json'))['best_iteration'] + 1
    P = np.asarray(l2.inplace_predict(np.column_stack([s1f, ce] + ce2).astype(np.float32), iteration_range=(0, it2)), dtype=np.float64)
    log('L2 probability', float(P.mean()))

    text, c1, c23 = read_sources(RAW, 'test', set(refs) | set(tg))
    log('test name-key counts', len(c1), len(c23))
    by_ref = defaultdict(list)
    for j, r in enumerate(ri):
        by_ref[r].append(j)
    items = []
    for r, js in by_ref.items():
        js = np.array(js)
        items.append((r, js[np.argsort(-P[js], kind='stable')]))
    G.update(P=P, s1f=s1f, ce=ce, tg=tg, refs=refs, text=text, c1=c1, c23=c23)
    n = int(os.environ.get('NPROC', 8))
    chunks = [items[i::n * 8] for i in range(n * 8)]
    F = np.zeros((len(tg), len(NAMES)), dtype=np.float32)
    with mp.get_context('fork').Pool(n) as pool:
        for res in pool.imap_unordered(feat_chunk, chunks):
            for o, f in res:
                F[o] = f
    log('stage-3 features')

    s1_pos = {e: i for i, e in enumerate(refs)}
    ut = sorted(set(tg))
    upos = {t: i for i, t in enumerate(ut)}
    pt = np.array([upos[t] for t in tg]); pr = ri.astype(np.int64)
    tok = AutoTokenizer.from_pretrained(DENSE)
    enc = AutoModel.from_pretrained(DENSE).cuda().eval()
    R_rev = []
    for nm in ('name', 'full'):
        s_txt = [f'passage: {raw.get(e, ("", ""))[0]}' if nm == 'name' else f'passage: {raw.get(e, ("", ""))[0]} {raw.get(e, ("", ""))[1]}' for e in refs]
        t_txt = [f'passage: {raw.get(t, ("", ""))[0]}' if nm == 'name' else f'passage: {raw.get(t, ("", ""))[0]} {raw.get(t, ("", ""))[1]}' for t in ut]
        semb = embed(s_txt, tok, enc); temb = embed(t_txt, tok, enc)
        R_rev.append(rev_features(temb, semb, pt, pr))
        del semb, temb
        torch.cuda.empty_cache()
        log('reverse competition', nm, len(ut))
    Frev = np.column_stack(R_rev)

    Fnum = np.zeros((len(tg), 5), dtype=np.float32)
    for r, o in items:
        js = list(o)
        rn = first_num(raw.get(refs[r], ('', ''))[1])
        nums = [first_num(raw.get(tg[j], ('', ''))[1]) for j in js]
        cnt = Counter(x for x in nums if x)
        acc = [P[j] >= 0.5 for j in js]
        for q, j in enumerate(js):
            sib = {nums[z] for z in range(len(js)) if acc[z] and z != q and nums[z]}
            x = nums[q]
            Fnum[j] = [float(bool(x) and x == rn), cnt[x] - 1 if x else -1, float(bool(x) and x in sib), len(sib), float(bool(x))]
    log('house-number features')

    meta = json.load(open(MODEL_DIR / 'stage3b_meta.json'))
    extra = list(ce2)
    if os.environ.get('OWN_TEST'):  # Vishal's ownership block in candidate_pairs.tsv expanded order = late_scores order
        own = np.column_stack([np.load(f).astype(np.float32).reshape(len(tg), -1) for f in os.environ['OWN_TEST'].split(';') if f])
        extra.append(own)
    # Feature columns only ever get appended to boundary_exp.NAMES, so a model trained before an addition uses the
    # leading columns: trim to what this model was trained on instead of failing after the whole feature build.
    n_extra = sum(1 if e.ndim == 1 else e.shape[1] for e in extra)
    n_f = len(meta['features']) - Frev.shape[1] - Fnum.shape[1] - n_extra
    if n_f < F.shape[1]:
        log('trimming list features', F.shape[1], '->', n_f, 'for this model')
        F = F[:, :n_f]
    X = np.column_stack([F, Frev, Fnum] + extra).astype(np.float32)
    assert X.shape[1] == len(meta['features']), (X.shape, len(meta['features']))
    b3 = xgb.Booster(); b3.load_model(str(MODEL_DIR / 'stage3b.json')); b3.set_param({'device': 'cuda'})
    P3 = np.asarray(b3.inplace_predict(X, iteration_range=(0, meta['best_iteration'] + 1)), dtype=np.float32)
    np.save(OUT / 'features.npy', X)
    np.savez(OUT / 'late_scores.npz', ref_idx=ri, target=tg.astype(str), prob=P3, stage1=s1f.astype(np.float32),
             ce=ce.astype(np.float32), l2=P.astype(np.float32))
    shutil.copy(S5_DIR / 'candidates.tsv', OUT / 'candidates.tsv')
    cty = np.array([r['country'] for r in refs_d], dtype=object)[ri]
    for c in sorted(set(cty)):
        m = cty == c
        log(c, 'mean P3', round(float(P3[m].mean()), 4), 'uncertain 0.2-0.6', round(float(((P3[m] >= 0.2) & (P3[m] < 0.6)).mean()), 4),
            'decoy-word pairs', int(F[m, NAMES.index('decoy_word')].sum()))
    rule = os.environ.get('RULE') or f"{meta['rule'][0]}@{meta['rule'][1]}"
    a = Namespace(output=str(OUT), rule=rule, tag='', late_model=str(MODEL_DIR), country_threshold='')
    decide_and_write(a, {'rule': meta['rule']})
    log('done', rule)


if __name__ == '__main__':
    main()
