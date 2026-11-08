"""Stage-3b: the stop decision with record-centric competition (each record's best owner over the FULL S1 table).

A record belongs to at most one S1 entity, so 'accept (r, t)?' is really 'is r clearly t's best owner?'. The 100k-ref
sample cannot answer that from its own lists (most same-name rival S1 entities of a sampled reference are not in the
sample), so the competition is computed against the full S1 table of the split (2.2M train / 1.7M test), identically
in train and test, with the fine-tuned e5 retriever:
  name-only and name+address cosine: rank of r among t's top-20 S1 entities, gap to the best rival, rivals within 0.02
Plus house-number context (decoys come in groups at one house number): number equals the reference's, how many list
candidates share it, whether an accepted sibling shares it, distinct numbers among accepted siblings.
Stage-3 (boundary_exp.py features) vs stage-3 + rev vs + num vs + both; rule on dev, holdout once.
Writes fusion/boundary_exp2.json and boundary_models/stage3b*.json.
"""
import gzip
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from boundary_exp import CE, FUS, NAMES, OUT, PRUNE, RAW, V4, W, list_features, read_sources
from distractor_check import decide, f05, partition_of

DENSE = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/dense_v1/model')
K = 20
REV = ['rn_rank', 'rn_gap', 'rn_close', 'rn_cos', 'rf_rank', 'rf_gap', 'rf_close', 'rf_cos']
NUM = ['num_eq_ref', 'num_group', 'num_eq_acc', 'n_num_acc', 'tgt_has_num']


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def embed(texts, tok, enc, bs=2048):
    out = torch.empty((len(texts), enc.config.hidden_size), dtype=torch.float16, device='cuda')
    order = np.argsort([len(t) for t in texts], kind='stable')
    with torch.no_grad():
        for b in range(0, len(texts), bs):
            o = order[b:b + bs]
            e = tok([texts[i] for i in o], truncation=True, max_length=96, padding=True, return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = enc(**e).last_hidden_state
            m = e['attention_mask'].unsqueeze(-1).float()
            out[torch.as_tensor(o, device='cuda')] = torch.nn.functional.normalize((h.float() * m).sum(1) / m.sum(1), dim=-1).half()
    return out


def rev_features(temb, semb, pt, pr):
    """Pair j = (target row pt[j], S1 row pr[j]) -> rank of r in t's top-K S1, gap to best rival, rivals close, cosine."""
    U = len(temb)
    ts = np.zeros((U, K), dtype=np.float32); ti = np.zeros((U, K), dtype=np.int64)
    for b in range(0, U, 1024):
        v, i = (temb[b:b + 1024] @ semb.T).topk(K, dim=1)
        ts[b:b + 1024] = v.float().cpu().numpy(); ti[b:b + 1024] = i.cpu().numpy()
    cos = np.zeros(len(pt), dtype=np.float32)
    for b in range(0, len(pt), 1 << 18):
        a = torch.as_tensor(pt[b:b + (1 << 18)], device='cuda'); c = torch.as_tensor(pr[b:b + (1 << 18)], device='cuda')
        cos[b:b + (1 << 18)] = (temb[a].float() * semb[c].float()).sum(1).cpu().numpy()
    TI, TS = ti[pt], ts[pt]
    hit = TI == pr[:, None]
    rank = np.where(hit.any(1), hit.argmax(1) + 1, K + 1)
    rival = np.where(TI[:, 0] != pr, TS[:, 0], TS[:, 1])
    close = ((TS >= cos[:, None] - 0.02) & ~hit).sum(1)
    return np.column_stack([rank, cos - rival, close, cos]).astype(np.float32)


def first_num(a):
    m = re.search(r'\d+', a or '')
    return str(int(m.group(0))) if m else ''  # leading zeros ('0014') are formatting, not a different number


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    cp = partition_of(gold)
    mask = np.load(PRUNE / 'mask_top10.npy')
    refs, tgts = [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t')
                refs.append(r); tgts.append(t)
    refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
    part = np.array([cp[r] for r in refs], dtype=object)
    S = np.load(FUS / 'stage1.npz')
    y = np.load(V4 / 'pairs/y.npy')[S['rows']].astype(np.float32)
    s1f = np.where(S['stage1'].astype(np.float64) < 0.02, 0.0, S['stage1'].astype(np.float64))
    ce = np.nan_to_num(np.load(CE).astype(np.float64), nan=0.0)
    CE2 = [f for f in os.environ.get('CE2_FILE', '').split(';') if f]
    ce2 = [np.nan_to_num(np.load(f).astype(np.float64), nan=0.0) for f in CE2]
    owned = set()
    with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if len(q) > 1 and q[1]:
                owned.update(t.strip() for t in q[1].split(','))
    wt = np.where((y == 0) & np.array([t not in owned for t in tgts]), W, 1.0).astype(np.float32)
    trA, trB, dv = part == 'trainA', part == 'trainB', part == 'dev'
    tr = trA | trB
    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                  max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)

    def fit(X, rows):
        d = xgb.QuantileDMatrix(X[rows], y[rows], weight=wt[rows])
        b = xgb.train(params, d, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=d), 'dev')],
                      early_stopping_rounds=100, verbose_eval=False)
        return b, (0, b.best_iteration + 1)

    CF = os.environ.get('CACHE_FROM', '')
    if CF:  # same CE inputs as a finished run: reuse its L2 model and features; only the appended block changes
        C = np.load(Path(CF) / 'exp2_cache.npz')
        F, Frev, Fnum, P, blank_all = C['F'], C['Frev'], C['Fnum'], C['P'], C['blank']
        for f in ('l2.json', 'l2_meta.json'):
            shutil.copy(Path(CF) / f, OUT / f)
        log('reused cache', CF)
    else:
        X2 = np.column_stack([s1f, ce] + ce2).astype(np.float32)
        P = np.zeros(len(y))
        for a, b in ((trA, trB), (trB, trA)):
            m, it = fit(X2, a)
            P[b] = m.inplace_predict(X2[b], iteration_range=it)
        m, it = fit(X2, tr)
        P[~tr] = m.inplace_predict(X2[~tr], iteration_range=it)
        m.save_model(str(OUT / 'l2.json'))
        json.dump({'best_iteration': int(m.best_iteration), 'ce_file': CE, 'ce2_files': CE2}, open(OUT / 'l2_meta.json', 'w'))
        log('L2 out-of-fold done')

        text, c1, c23 = read_sources(RAW, 'train', set(refs) | set(tgts))
        by_ref = defaultdict(list)
        for j, r in enumerate(refs):
            by_ref[r].append(j)
        F = np.zeros((len(y), len(NAMES)), dtype=np.float32)
        for r, js in by_ref.items():
            js = np.array(js)
            o = js[np.argsort(-P[js], kind='stable')]
            F[o] = list_features(P[o], s1f[o], ce[o], list(tgts[o]), r, text, c1, c23)
        log('stage-3 features')

        # ---- raw text: full S1 table + sample targets ----
        s1_ids, s1_name, s1_full, raw = [], [], [], {}
        need = set(tgts) | set(refs)
        for k in (1, 2, 3):
            with open(RAW / f'train_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if len(q) <= iaddr:
                        continue
                    if k == 1:
                        s1_ids.append(q[ie]); s1_name.append(f'passage: {q[iname]}'); s1_full.append(f'passage: {q[iname]} {q[iaddr]}')
                    if q[ie] in need:
                        raw[q[ie]] = (q[iname], q[iaddr])
        s1_pos = {e: i for i, e in enumerate(s1_ids)}
        ut = sorted(set(tgts))
        upos = {t: i for i, t in enumerate(ut)}
        pt = np.array([upos[t] for t in tgts]); pr = np.array([s1_pos[r] for r in refs])
        log('S1 table', len(s1_ids), 'unique targets', len(ut))

        tok = AutoTokenizer.from_pretrained(DENSE)
        enc = AutoModel.from_pretrained(DENSE).cuda().eval()
        R_rev = []
        for nm, s1_txt, tfmt in (('name', s1_name, lambda n, a: f'passage: {n}'), ('full', s1_full, lambda n, a: f'passage: {n} {a}')):
            semb = embed(s1_txt, tok, enc)
            temb = embed([tfmt(*raw.get(t, ('', ''))) for t in ut], tok, enc)
            R_rev.append(rev_features(temb, semb, pt, pr))
            del semb, temb
            torch.cuda.empty_cache()
            log('reverse competition', nm)
        Frev = np.column_stack(R_rev)

        Fnum = np.zeros((len(y), len(NUM)), dtype=np.float32)
        for r, js in by_ref.items():
            rn = first_num(raw.get(r, ('', ''))[1])
            nums = [first_num(raw.get(tgts[j], ('', ''))[1]) for j in js]
            cnt = Counter(n for n in nums if n)
            acc = [P[j] >= 0.5 for j in js]
            for q, j in enumerate(js):
                sib = {nums[z] for z in range(len(js)) if acc[z] and z != q and nums[z]}
                n = nums[q]
                Fnum[j] = [float(bool(n) and n == rn), cnt[n] - 1 if n else -1, float(bool(n) and n in sib), len(sib), float(bool(n))]
        log('house-number features')
        blank_all = np.array([not raw.get(t, ('', ''))[1].strip() for t in tgts])
        np.savez(OUT / 'exp2_cache.npz', F=F, Frev=Frev, Fnum=Fnum, P=P, blank=blank_all)

    idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
    prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
    rix = {}
    for p in idx:
        pos = {s: i for i, s in enumerate(prefs[p])}
        rix[p] = np.array([pos[r] for r in refs[idx[p]]])
    grid = [(ru, round(float(t), 2)) for ru in ('expected', 'threshold') for t in np.arange(0.4, 0.961, 0.02)]

    def macro(p, chosen):
        pred = defaultdict(set)
        for i, tt in zip(rix[p][chosen], tgts[idx[p]][chosen]):
            pred[i].add(tt)
        return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

    R = {'ref': 'S5 holdout 0.98622; stage3 (boundary_exp) dev 0.99034 / holdout 0.98839'}
    Fx = np.column_stack(ce2).astype(np.float32) if ce2 else np.zeros((len(y), 0), dtype=np.float32)
    n_own = 0
    if os.environ.get('OWN_FILE'):  # ownership block, same row order (mask_top10 rows)
        own = np.column_stack([np.load(f).astype(np.float32).reshape(len(y), -1) for f in os.environ['OWN_FILE'].split(';') if f])
        Fx, n_own = np.column_stack([Fx, own]), own.shape[1]
    variants = {'stage3': np.column_stack([F, Fx]), 'stage3+rev': np.column_stack([F, Frev, Fx]),
                'stage3+num': np.column_stack([F, Fnum, Fx]), 'stage3+rev+num': np.column_stack([F, Frev, Fnum, Fx])}
    names_all = NAMES + REV + NUM + [f'ce2_{k}' for k in range(len(ce2))] + [f'own_{k}' for k in range(n_own)]
    for nm, X in variants.items():
        X = X.astype(np.float32)
        b, it3 = fit(X, tr)
        prob = np.asarray(b.inplace_predict(X, iteration_range=it3), dtype=np.float64)
        dev = {g: macro('dev', decide(rix['dev'], prob[idx['dev']], *g, len(prefs['dev']))) for g in grid}
        g = max(dev, key=lambda k: (dev[k], k[1]))
        R[nm] = {'rule': g, 'dev': round(dev[g], 6),
                 'holdout': round(macro('holdout', decide(rix['holdout'], prob[idx['holdout']], *g, len(prefs['holdout']))), 6)}
        if nm == 'stage3+rev+num':
            gain = b.get_score(importance_type='total_gain')
            R['gain_top20'] = {names_all[int(k[1:])]: round(v) for k, v in sorted(gain.items(), key=lambda kv: -kv[1])[:20]}
            b.save_model(str(OUT / 'stage3b.json'))
            np.save(OUT / 'stage3b_prob.npy', prob)
            json.dump({'best_iteration': int(b.best_iteration), 'features': names_all, 'rule': g}, open(OUT / 'stage3b_meta.json', 'w'))
            blank = blank_all[idx['holdout']]
            ch = decide(rix['holdout'], prob[idx['holdout']], *g, len(prefs['holdout']))
            yh = y[idx['holdout']] > 0
            R['holdout_errors'] = {'blank_fn': int((blank & yh & ~ch).sum()), 'blank_fp': int((blank & ~yh & ch).sum()),
                                   'nonblank_fn': int((~blank & yh & ~ch).sum()), 'nonblank_fp': int((~blank & ~yh & ch).sum()),
                                   's5_ref': 'blank_fn 1031, blank_fp 141, nonblank_fn 249, nonblank_fp 103'}
        log(nm, R[nm])
        (FUS / f"boundary_exp2{os.environ.get('TAG', '')}.json").write_text(json.dumps(R, indent=1))
    log('done')


if __name__ == '__main__':
    main()
