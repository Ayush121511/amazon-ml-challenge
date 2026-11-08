"""France-trained stage-3b: the France-robust feature set (t = 0.25 drop) trained on all train-sample rows plus
France test rows pseudo-labelled by the census rules and model extremes (artifacts/fr_labels/fr_labels.npz).

Self-training on test France with rule-derived labels (no ground truth, no external data). Setup:
  - France rows: census_pos / census_neg / extreme_pos kept, extreme_neg subsampled to 25%; weight FR_W (0.3) except the
    disagreement rows (census_pos the robust model rejects at p < 0.5, census_neg it accepts at p >= 0.5) at 1.0;
  - France split by reference 80/20 (sha256 of the id); train on 80%; on the held-out 20% report how many disagreement
    rows flipped to the census side (pattern learned, not memorised);
  - guard: US/India holdout within GUARD of T5 (0.991093), rule chosen on US/India dev; else retry with a lower FR_W;
  - an is_france input column (0 on the train sample, 1 on France rows) lets the trees keep France-only rules
    (first run without it: France pattern learned, but US/India holdout -0.001, since French filler words mean
    'copy' while the same words mark decoys in US/India).
The model predicts all France rows (no retrain); US/India stay T5. Pre-census output: artifacts/submission_FRtrained/.
"""
import csv
import gzip
import hashlib
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
OWN = ME / 'artifacts/ownership_v1'
ROB = ME / 'artifacts/submission_FRrobust'
T4TEST = A / 'artifacts/submission_T4_s3b'
T5TEST = ME / 'artifacts/submission_T5_own'
LAB = A / 'artifacts/fr_labels/fr_labels.npz'
OUT = ME / 'artifacts' / os.environ.get('FRT_OUT', 'submission_FRtrained')
NEG_W = float(os.environ.get('EXTREME_NEG_WEIGHT', 1.0))   # 4.0 restores the base rate after the 25% extreme_neg subsample
T5_HOLDOUT = 0.991093
GUARD = float(os.environ.get('GUARD', 0.0003))
FR_WEIGHTS = [float(x) for x in os.environ.get('FR_WEIGHTS', '0.3,0.15').split(',')]
PARAMS = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
              max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cols = json.load(open(ROB / 'stage3b_frrobust_meta.json'))['cols']
    # ---------- train sample (US/India, labels) ----------
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
    X = np.column_stack([C['F'], C['Frev'], C['Fnum']] + ce2 + [own]).astype(np.float32)[:, cols]
    X = np.column_stack([X, np.zeros(len(X), dtype=np.float32)])                       # is_france = 0
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

    # ---------- France pseudo-labels ----------
    L = np.load(LAB, allow_pickle=True)
    Z = np.load(ROB / 'late_scores.npz', allow_pickle=True)
    li, lab, src = L['idx'], L['label'].astype(np.float32), L['src']
    assert (Z['target'][li] == L['target']).all()
    p_rob = Z['prob'][li].astype(np.float64)
    dis_pos, dis_neg = (src == 1) & (p_rob < 0.5), (src == 2) & (p_rob >= 0.5)
    refs_t = []
    with open(ME / 'data/dataset/test/test_source1.tsv', encoding='utf-8', newline='') as fh:
        for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
            refs_t.append({'entity_id': row['entity_id'], 'country': row['country']})
    ri_all = Z['ref_idx']
    ref_of = np.array([refs_t[i]['entity_id'] for i in ri_all[li]], dtype=object)
    in80 = np.array([int(hashlib.sha256(r.encode()).hexdigest()[:8], 16) % 100 < 80 for r in ref_of])
    rng = np.random.default_rng(11)
    keep = (src != 4) | (rng.random(len(src)) < 0.25)
    Xt = np.load(T4TEST / 'features.npy', mmap_mode='r')
    own_t = [np.load(OWN / f, mmap_mode='r') for f in ('own_test.npy', 'own_addr_test.npy', 'own_raw_test.npy')]

    def test_rows(j):
        order = np.argsort(j)
        js = j[order]
        out = np.empty((len(j), len(cols)), dtype=np.float32)
        for s in range(0, len(js), 1_000_000):
            q = js[s:s + 1_000_000]
            out[order[s:s + 1_000_000]] = np.column_stack([np.asarray(Xt[q])] + [np.asarray(o[q]) for o in own_t]).astype(np.float32)[:, cols]
        return np.column_stack([out, np.ones(len(out), dtype=np.float32)])                # is_france = 1
    tr_fr = np.flatnonzero(in80 & keep)
    ev_fr = np.flatnonzero(~in80)
    X_fr_tr, X_fr_ev = test_rows(li[tr_fr]), test_rows(li[ev_fr])
    log('France labelled rows: train', len(tr_fr), 'eval (20% refs)', len(ev_fr), 'disagreement pos/neg', int(dis_pos.sum()), int(dis_neg.sum()))
    R = {'france_labels': {'rows': int(len(li)), 'train_rows_after_subsample': int(len(tr_fr)), 'eval_rows': int(len(ev_fr)),
                           'disagreement_pos': int(dis_pos.sum()), 'disagreement_neg': int(dis_neg.sum())},
         'note': 'self-training on test France with rule-derived pseudo-labels (census + model extremes); no ground truth'}
    chosen = None
    for frw in FR_WEIGHTS:
        w_fr = (np.where(dis_pos[tr_fr] | dis_neg[tr_fr], 1.0, frw) * np.where(src[tr_fr] == 4, NEG_W, 1.0)).astype(np.float32)
        Xall = np.vstack([X[tr], X_fr_tr]); yall = np.concatenate([y[tr], lab[tr_fr]]); wall = np.concatenate([wt[tr], w_fr])
        d = xgb.QuantileDMatrix(Xall, yall, weight=wall)
        b = xgb.train(PARAMS, d, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=d), 'dev')],
                      early_stopping_rounds=100, verbose_eval=False)
        it = (0, b.best_iteration + 1)
        prob = np.asarray(b.inplace_predict(X, iteration_range=it), dtype=np.float64)
        dev = {g: macro('dev', decide(rix['dev'], prob[idx['dev']], *g, len(prefs['dev']))) for g in grid}
        g = max(dev, key=lambda k: (dev[k], k[1]))
        ho = macro('holdout', decide(rix['holdout'], prob[idx['holdout']], *g, len(prefs['holdout'])))
        pe = np.asarray(b.inplace_predict(X_fr_ev, iteration_range=it), dtype=np.float64)
        dp, dn = dis_pos[ev_fr], dis_neg[ev_fr]
        agree = (src[ev_fr] >= 3) | ~(dp | dn)
        res = {'france_weight': frw, 'extreme_neg_weight': NEG_W, 'rule': list(g), 'dev': round(dev[g], 6), 'holdout': round(ho, 6),
               'holdout_vs_T5': round(ho - T5_HOLDOUT, 6), 'best_iteration': int(b.best_iteration),
               'eval20_disagreement_pos_now_p_ge_0.5': round(float((pe[dp] >= 0.5).mean()), 4) if dp.any() else None,
               'eval20_disagreement_neg_now_p_lt_0.5': round(float((pe[dn] < 0.5).mean()), 4) if dn.any() else None,
               'eval20_disagreement_rows': [int(dp.sum()), int(dn.sum())],
               'eval20_label_accuracy_other_rows': round(float(((pe[agree] >= 0.5) == (lab[ev_fr][agree] == 1)).mean()), 5)}
        R[f'fr_w_{frw}'] = res
        log(res)
        if res['holdout_vs_T5'] >= -GUARD:
            chosen = (b, res)
            break
    if chosen is None:
        R['verdict'] = f'guard failed for all France weights {FR_WEIGHTS}; nothing written'
        (OUT / 'report.json').write_text(json.dumps(R, indent=1)); log(R['verdict']); return
    b, res = chosen
    R['chosen'] = res
    b.save_model(str(OUT / 'stage3b_frtrained.json'))
    json.dump({'cols': cols, 'extra_last_column': 'is_france', **res}, open(OUT / 'stage3b_frtrained_meta.json', 'w'), indent=1)

    # ---------- predict all France rows (no retrain) ----------
    fr = np.array([refs_t[i]['country'] == 'France' for i in ri_all])
    frj = np.flatnonzero(fr)
    P = Z['prob'].astype(np.float32).copy()
    for s in range(0, len(frj), 1_000_000):
        j = frj[s:s + 1_000_000]
        P[j] = b.inplace_predict(test_rows(j), iteration_range=(0, res['best_iteration'] + 1))
    log('France rows predicted', len(frj), 'mean new', float(P[frj].mean()), 'mean robust', float(Z['prob'][frj].mean()))
    np.savez(OUT / 'late_scores.npz', ref_idx=ri_all, target=Z['target'], prob=P, stage1=Z['stage1'], ce=Z['ce'],
             l2=Z['l2'], prob_t5=Z['prob_t5'], prob_robust=Z['prob'])
    (OUT / 'candidates.tsv').write_text((ROB / 'candidates.tsv').read_text(encoding='utf-8'), encoding='utf-8')
    cands_out = {}
    with open(ME / 'artifacts/submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
        next(f)
        for line in f:
            s_, c_ = line.rstrip('\n').split('\t'); cands_out[s_] = c_
    rule = res['rule']
    tmp = OUT.parent / (OUT.name + '_all_countries_tmp')
    tmp.mkdir(exist_ok=True)
    tg = Z['target'].astype(str)
    finalize(refs_t, cands_out, ri_all, tg.astype(object), P, {'decision': {'rule': rule[0], 'threshold': rule[1]}}, True, tmp,
             Counter(), {'model': 'france_trained', 'started': time.monotonic()}, {})

    def load(p):
        with open(p, encoding='utf-8') as f:
            next(f); return {l.split('\t')[0]: l.rstrip('\n').split('\t')[1] for l in f}
    new, t5 = load(tmp / 'matching_results.tsv'), load(next(T5TEST.glob('expected_*')) / 'matching_results.tsv')
    dest = OUT / f'{rule[0]}_{rule[1]}'
    dest.mkdir(exist_ok=True)
    with open(dest / 'matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for r in refs_t:
            e = r['entity_id']
            f.write(e + '\t' + (new[e] if r['country'] == 'France' else t5[e]) + '\n')
    (dest / 'candidate_pairs.tsv').write_text((tmp / 'candidate_pairs.tsv').read_text(encoding='utf-8'), encoding='utf-8')
    import shutil
    shutil.rmtree(tmp)
    R['output'] = str(dest)
    (OUT / 'report.json').write_text(json.dumps(R, indent=1))
    log('wrote', dest)


if __name__ == '__main__':
    main()
