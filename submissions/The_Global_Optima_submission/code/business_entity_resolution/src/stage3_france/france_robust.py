"""France-robust stage-3b: T5's stage-3b retrained without the inputs whose distribution shifts in France.

The T5 transfer audit (KS distance, sample vs test on pairs v5 and chunk2 agree on) shows US/India inputs transfer, but
France's address-driven inputs (name+address reverse competition, address similarity, address ownership, number groups)
are out of distribution. Drop a feature when its France KS exceeds the larger of its US/India KS by >= t (either the
sure-match or sure-non-match group). For t in THRESHOLDS (most drops first) retrain with boundary_exp2.py's settings
(depth 6, lr 0.05, decoy weight 3, trainA+trainB, early stop on dev), choose the rule on dev, score holdout once, and keep
the first t whose holdout is within GUARD of the same-code T5 reproduction. The kept model is applied to France only:
US/India decisions stay exactly T5's.

Outputs (pre-census, so the veto -> census order holds): artifacts/submission_FRrobust/{late_scores.npz (prob = robust
for France rows, T5 elsewhere), candidates.tsv, <rule>/matching_results.tsv + candidate_pairs.tsv}, report.json.
"""
import csv
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

A = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
sys.path[:0] = [str(ME / 'business_entity_resolution/src'), str(A / 'business_entity_resolution/src')]
from distractor_check import decide, f05, partition_of  # noqa: E402
from submission_pipeline import finalize  # noqa: E402

V4 = Path('/scratch/scai/mtech/aib262015/vishal_v4_artifacts')
B4 = A / 'artifacts/fusion/boundary_models_t4'
T5 = ME / 'artifacts/fusion/boundary_models_t5'
OWN = ME / 'artifacts/ownership_v1'
AUDIT = ME / 'artifacts/submission_T5_census/t5_shift_audit.json'
T5TEST = ME / 'artifacts/submission_T5_own'
T4TEST = A / 'artifacts/submission_T4_s3b'
OUT = ME / 'artifacts/submission_FRrobust'
THRESHOLDS = [float(x) for x in os.environ.get('THRESHOLDS', '0.15,0.25,0.35,0.5').split(',')]
GUARD = float(os.environ.get('GUARD', 0.0003))
PARAMS = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
              max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---------- sample ----------
    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    cp = partition_of(gold)
    mask = np.load(A / 'artifacts/prune_v1_copy/mask_top10.npy')
    refs, tgts = [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t')
                refs.append(r); tgts.append(t)
    refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
    part = np.array([cp[r] for r in refs], dtype=object)
    S = np.load(A / 'artifacts/fusion/stage1.npz')
    y = np.load(V4 / 'pairs/y.npy')[S['rows']].astype(np.float32)
    C = np.load(B4 / 'exp2_cache.npz')
    ce2 = [np.load(ME / f'artifacts/{m}/ce_logit_sample.npy').astype(np.float32) for m in ('mdeberta_full21', 'mdeberta_decoyw2')]
    own = np.column_stack([np.load(OWN / f) for f in ('own_sample.npy', 'own_addr_sample.npy', 'own_raw_sample.npy')])
    X = np.column_stack([C['F'], C['Frev'], C['Fnum']] + ce2 + [own]).astype(np.float32)
    meta5 = json.load(open(T5 / 'stage3b_meta.json'))
    assert X.shape[1] == len(meta5['features']), (X.shape, len(meta5['features']))
    on = json.load(open(OWN / 'own_sample.json'))
    own_names = on['features'] + on['addr_features'] + on['raw_features']
    names = meta5['features'][:55] + own_names          # real names for the 26 own_k columns
    owned = set()
    with open(ME / 'data/dataset/train/train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if len(q) > 1 and q[1]:
                owned.update(t.strip() for t in q[1].split(','))
    wt = np.where((y == 0) & np.array([t not in owned for t in tgts]), 3.0, 1.0).astype(np.float32)
    tr, dv = np.isin(part, ['trainA', 'trainB']), part == 'dev'
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

    def train_eval(cols):
        Xc = X[:, cols]
        d = xgb.QuantileDMatrix(Xc[tr], y[tr], weight=wt[tr])
        b = xgb.train(PARAMS, d, 3000, evals=[(xgb.QuantileDMatrix(Xc[dv], y[dv], ref=d), 'dev')],
                      early_stopping_rounds=100, verbose_eval=False)
        it = (0, b.best_iteration + 1)
        prob = np.asarray(b.inplace_predict(Xc, iteration_range=it), dtype=np.float64)
        dev = {g: macro('dev', decide(rix['dev'], prob[idx['dev']], *g, len(prefs['dev']))) for g in grid}
        g = max(dev, key=lambda k: (dev[k], k[1]))
        ho = macro('holdout', decide(rix['holdout'], prob[idx['holdout']], *g, len(prefs['holdout'])))
        return b, prob, {'rule': list(g), 'dev': round(dev[g], 6), 'holdout': round(ho, 6), 'best_iteration': int(b.best_iteration)}

    # ---------- feature shift from the audit ----------
    aud = json.load(open(AUDIT))['groups']
    excess = defaultdict(float)
    for g in aud.values():
        for r in g['features']:
            k = r['ks_by_country']
            excess[r['feature']] = max(excess[r['feature']], k['France'] - max(k['US'], k['India']))
    R = {'excess_ks_top': dict(sorted(((n, round(v, 3)) for n, v in excess.items()), key=lambda kv: -kv[1])[:25])}
    all_cols = list(range(len(names)))
    b_ref, p_ref, R['T5_reproduction'] = train_eval(all_cols)
    log('T5 reproduction', R['T5_reproduction'], '(boundary_exp2 T5: dev 0.992166 / holdout 0.991093)')
    chosen = None
    for t in THRESHOLDS:
        drop = [n for n in names if excess.get(n, 0.0) >= t]
        cols = [i for i, n in enumerate(names) if n not in drop]
        b, prob, res = train_eval(cols)
        res.update(threshold=t, dropped=drop, n_features=len(cols))
        res['holdout_vs_reproduction'] = round(res['holdout'] - R['T5_reproduction']['holdout'], 6)
        R[f'drop_ge_{t}'] = res
        log(f't={t}', {k: v for k, v in res.items() if k != 'dropped'}, 'dropped', drop)
        if chosen is None and res['holdout_vs_reproduction'] >= -GUARD:
            chosen = (t, cols, b, prob, res)
    if chosen is None:
        R['verdict'] = f'no threshold within {GUARD} of the reproduction; nothing written'
        (OUT / 'report.json').write_text(json.dumps(R, indent=1)); log(R['verdict']); return
    t, cols, b, prob_s, res = chosen
    R['chosen'] = {'threshold': t, **{k: v for k, v in res.items()}}
    b.save_model(str(OUT / 'stage3b_frrobust.json'))
    np.save(OUT / 'robust_sample_prob.npy', prob_s)
    json.dump({'features': [names[i] for i in cols], 'cols': cols, **res}, open(OUT / 'stage3b_frrobust_meta.json', 'w'), indent=1)

    # ---------- test: France only ----------
    Z = np.load(T5TEST / 'late_scores.npz', allow_pickle=True)
    ri, tg = Z['ref_idx'], Z['target'].astype(str)
    n = len(tg)
    Xt = np.load(T4TEST / 'features.npy', mmap_mode='r')
    assert Xt.shape == (n, 55), Xt.shape
    own_t = [np.load(OWN / f, mmap_mode='r') for f in ('own_test.npy', 'own_addr_test.npy', 'own_raw_test.npy')]
    refs_t, cty = [], {}
    with open(ME / 'data/dataset/test/test_source1.tsv', encoding='utf-8', newline='') as fh:
        for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
            refs_t.append({'entity_id': row['entity_id'], 'country': row['country']}); cty[row['entity_id']] = row['country']
    fr = np.array([refs_t[i]['country'] == 'France' for i in ri])
    P = Z['prob'].astype(np.float32).copy()
    frj = np.flatnonzero(fr)
    for s in range(0, len(frj), 2_000_000):
        j = frj[s:s + 2_000_000]
        Xf = np.column_stack([np.asarray(Xt[j])] + [np.asarray(o[j]) for o in own_t]).astype(np.float32)[:, cols]
        P[j] = b.inplace_predict(Xf, iteration_range=(0, res['best_iteration'] + 1))
    log('France rows re-predicted', len(frj), 'mean robust', float(P[frj].mean()), 'mean T5', float(Z['prob'][frj].mean()))
    np.savez(OUT / 'late_scores.npz', ref_idx=ri, target=Z['target'], prob=P, stage1=Z['stage1'], ce=Z['ce'],
             l2=Z['l2'], prob_t5=Z['prob'])
    cands_out = {}
    with open(ME / 'artifacts/submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
        next(f)
        for line in f:
            s_, c_ = line.rstrip('\n').split('\t'); cands_out[s_] = c_
    with open(OUT / 'candidates.tsv', 'w', encoding='utf-8') as f:
        for r in refs_t:
            f.write(r['entity_id'] + '\t' + cands_out[r['entity_id']] + '\n')
    rule = res['rule']
    tmp = OUT.parent / 'submission_FRrobust_all_countries_tmp'   # outside OUT: the census takes OUT's first rule folder
    tmp.mkdir(exist_ok=True)
    finalize(refs_t, cands_out, ri, tg.astype(object), P, {'decision': {'rule': rule[0], 'threshold': rule[1]}}, True, tmp,
             Counter(), {'model': 'france_robust', 'started': time.monotonic()}, {})

    def load(p):
        with open(p, encoding='utf-8') as f:
            next(f); return {l.split('\t')[0]: l.rstrip('\n').split('\t')[1] for l in f}
    rob, t5 = load(tmp / 'matching_results.tsv'), load(next(T5TEST.glob('expected_*')) / 'matching_results.tsv')
    dest = OUT / f'{rule[0]}_{rule[1]}'
    dest.mkdir(exist_ok=True)
    with open(dest / 'matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for r in refs_t:
            e = r['entity_id']
            f.write(e + '\t' + (rob[e] if r['country'] == 'France' else t5[e]) + '\n')
    (dest / 'candidate_pairs.tsv').write_text((tmp / 'candidate_pairs.tsv').read_text(encoding='utf-8'), encoding='utf-8')
    R['output'] = str(dest)
    (OUT / 'report.json').write_text(json.dumps(R, indent=1))
    log('wrote', dest)


if __name__ == '__main__':
    main()
