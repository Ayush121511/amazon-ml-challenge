"""Matcher EDA for the S5 generation: is the remaining loss in the ranking or in the stop decision?

Same questions as matcher_eda.py (v5 alone), asked of v5 alone ('gen1') and of the S5 late fusion ('gen2': v5 prob +
full-train mdeberta logit, L2 XGBoost, decoy w=3, trained on trainA+trainB like late_full_train.py). Rule chosen on
dev, everything reported on holdout. Loss is split per reference into
  blocking  = 1 - F05(gold, gold & candidates)
  ranking   = blocking ceiling - best top-k F05 under the model's own ordering
  boundary  = best top-k - achieved   (singleton_fp / missed_all / over / under)
Plus: calibration, margins of boundary errors, singleton detection, strata, FP typing, error examples.
Env MEMBERS 'name=path;...' adds cross-encoder logit columns to gen2. Writes fusion/gen2_eda.json + gen2_examples.tsv.
"""
import gzip
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from distractor_check import decide, f05, partition_of

HOME = Path.home()
V4 = HOME / 'scratch/vishal_v4_artifacts'
PRUNE = HOME / 'scratch/amazon_ml_2026/artifacts/prune_v1_copy'
FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
MEMBERS = os.environ.get('MEMBERS', 'mdebfull=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/mdeberta_full/ce_logit_sample.npy')
W = float(os.environ.get('DECOY_W', 3))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def norm(s):
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r'[^\w]+', ' ', s).strip()


def q(a):
    a = np.asarray(a, dtype=np.float64)
    return {k: round(float(np.quantile(a, v)), 4) for k, v in (('p10', .1), ('p25', .25), ('p50', .5), ('p75', .75), ('p90', .9))} | {'n': int(len(a))} if len(a) else {'n': 0}


def best_k(order_t, g):
    """Best top-k F05 and smallest k achieving it; F05 = 1.25 tp / (0.25 |g| + k)."""
    if not g:
        return 1.0, 0
    best, bk, tp = 0.0, 0, 0
    for k, t in enumerate(order_t, 1):
        tp += t in g
        v = 1.25 * tp / (0.25 * len(g) + k)
        if v > best + 1e-12:
            best, bk = v, k
    return best, bk


def main():
    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    countries = json.loads((V4 / 'pairs/countries.json').read_text())
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
    cols, names = [s1f], []
    for item in MEMBERS.split(';'):
        n, p = item.split('=', 1)
        if Path(p).exists():
            cols.append(np.nan_to_num(np.load(p).astype(np.float64), nan=0.0)); names.append(n)
        else:
            log('skip missing', n)
    owned = set()
    with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            r = line.rstrip('\n').split('\t')
            if len(r) > 1 and r[1]:
                owned.update(t.strip() for t in r[1].split(','))
    decoy = np.array([t not in owned for t in tgts])
    wt = np.where((y == 0) & decoy, W, 1.0).astype(np.float32)
    tr, dv = np.isin(part, ['trainA', 'trainB']), part == 'dev'
    idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
    prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
    rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}

    X = np.column_stack(cols).astype(np.float32)
    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                  max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)
    dtr = xgb.QuantileDMatrix(X[tr], y[tr], weight=wt[tr])
    bst = xgb.train(params, dtr, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=dtr), 'dev')],
                    early_stopping_rounds=100, verbose_eval=False)
    it = (0, bst.best_iteration + 1)
    P2 = {p: np.asarray(bst.inplace_predict(X[idx[p]], iteration_range=it), dtype=np.float64) for p in idx}
    P1 = {p: s1f[idx[p]] for p in idx}
    CE = {p: cols[1][idx[p]] for p in idx} if names else None
    log('L2 trained', names, bst.best_iteration)

    ho = idx['holdout']
    need = set(refs[ho]) | set(tgts[ho])
    text = {}
    for k in (1, 2, 3):
        with open(RAW / f'train_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
            for line in fh:
                r = line.rstrip('\n').split('\t')
                if len(r) > iaddr and r[ie] in need:
                    text[r[ie]] = (r[iname], r[iaddr])
    log('text loaded', len(text))

    def macro(p, prob, ru, t):
        c = decide(rix[p], prob, ru, t, len(prefs[p]))
        pred = defaultdict(set)
        for i, tt in zip(rix[p][c], tgts[idx[p]][c]):
            pred[i].add(tt)
        return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])])), c

    grid = [(ru, round(float(t), 2)) for ru in ('expected', 'threshold') for t in np.arange(0.3, 0.981, 0.02)]
    by_ref = defaultdict(list)
    for j, r in enumerate(refs[ho]):
        by_ref[r].append(j)
    H = prefs['holdout']
    n = len(H)
    yh, dh, th = y[ho], decoy[ho], tgts[ho]
    E = {'members': names, 'best_iteration': int(bst.best_iteration), 'n_holdout_refs': n, 'n_holdout_pairs': int(len(ho))}
    examples, nex = [], Counter()

    for gname, P in (('gen1_v5', P1), ('gen2_s5', P2)):
        dev = {g: macro('dev', P['dev'], *g)[0] for g in grid}
        rule = max(dev, key=lambda k: (dev[k], k[1]))
        score, chosen = macro('holdout', P['holdout'], *rule)
        ph = P['holdout']
        G = {'rule': rule, 'dev': round(dev[rule], 6), 'holdout': round(score, 6)}
        loss = Counter()
        cnt = Counter()
        ceil = defaultdict(float)
        margins = defaultdict(list)
        strata = defaultdict(lambda: [0, 0.0])
        top_s, top_ns = [], []
        rank_tp, rank_fn, rank_fp = Counter(), Counter(), Counter()
        fp_type = Counter()
        kdiff = Counter()
        sep = 0
        for s in H:
            g = gold[s]
            js = by_ref.get(s, [])
            order = sorted(js, key=lambda j: -ph[j])
            ot = [th[j] for j in order]
            cand = set(ot)
            pred = {th[j] for j in js if chosen[j]}
            kp = len(pred)
            sc = f05(g, pred)
            perfect = f05(g, g & cand)
            ok, ks = best_k(ot, g)
            ceil['achieved'] += sc; ceil['oracle_topk'] += ok; ceil['blocking'] += perfect
            ceil['oracle_card'] += f05(g, set(ot[:len(g)])) if g else 1.0
            for nm, alt in (('oracle_topk_rank_v5', P1['holdout']), ('oracle_topk_rank_ce', CE['holdout'] if CE else None)):
                if alt is not None:
                    ceil[nm] += best_k([th[j] for j in sorted(js, key=lambda j: -alt[j])], g)[0]
            ceil['singletons_perfect'] += (1.0 if not g else sc)
            ceil['never_empty'] += (f05(g, {ot[0]}) if g and not pred and ot else sc)
            gi = [r for r, t in enumerate(ot) if t in g]
            ngi = [r for r, t in enumerate(ot) if t not in g]
            sep += (not gi or not ngi or max(gi) < min(ngi))
            loss['blocking'] += 1 - perfect
            loss['ranking'] += perfect - ok
            b = ok - sc
            if not g and kp:
                kind = 'singleton_fp'
            elif g and not kp:
                kind = 'missed_all'
            elif kp > ks:
                kind = 'over'
            elif kp < ks:
                kind = 'under'
            else:
                kind = 'none'
            if b > 1e-12:
                loss['boundary_' + kind] += b
                cnt[kind] += 1
                if kind in ('over', 'singleton_fp'):
                    margins[kind].extend(ph[j] for j in order[ks:kp])
                else:
                    margins[kind].extend(ph[j] for j in order[kp:ks] if th[j] in g)
                if nex[gname] < 3000:
                    nex[gname] += len(order)
                    rn, ra = text.get(s, ('', ''))
                    for r, j in enumerate(order):
                        tn, ta = text.get(th[j], ('', ''))
                        examples.append((gname, kind, s, rn, ra, r + 1, th[j], tn, ta, round(ph[j], 3),
                                         int(th[j] in g), int(bool(chosen[j]))))
            (top_ns if g else top_s).append(ph[order[0]] if order else 0.0)
            if g:
                kdiff[max(-3, min(3, kp - len(g & cand)))] += 1
            for r, j in enumerate(order):
                t = th[j]
                if t in g and chosen[j]:
                    rank_tp[r + 1] += 1
                elif t in g:
                    rank_fn[r + 1] += 1
                elif chosen[j]:
                    rank_fp[r + 1] += 1
                    fp_type['decoy' if dh[j] else 'owned_by_other'] += 1
            rn, ra = text.get(s, ('', ''))
            exact = any(norm(rn) and norm(rn) == norm(text.get(t, ('', ''))[0]) for t in ot)
            for key in (f'country={countries.get(s, "UNK")}',
                        f'gold={min(len(g), 6)}' + ('+' if len(g) >= 6 else ''),
                        f'ref_addr_empty={not ra.strip()}', f'exact_name_in_cands={exact}',
                        f'ncand={"<5" if len(js) < 5 else "5-9" if len(js) < 10 else "10"}'):
                strata[key][0] += 1; strata[key][1] += 1 - sc
        G['ceilings'] = {k: round(v / n, 6) for k, v in ceil.items()}
        G['loss_points'] = {k: round(v / n * 100, 4) for k, v in loss.items()}
        G['loss_points']['total'] = round((1 - score) * 100, 4)
        G['boundary_error_refs'] = dict(cnt)
        G['boundary_margins'] = {k: q(v) for k, v in margins.items()}
        G['perfectly_separable_refs'] = round(sep / n, 5)
        G['k_pred_minus_k_reachable'] = {int(k): v for k, v in sorted(kdiff.items())}
        G['singleton_top_prob'] = q(top_s)
        G['nonsingleton_top_prob'] = q(top_ns)
        ts, tn = np.array(top_s), np.array(top_ns)
        G['singleton_auc_top_prob'] = round(float((tn[:, None] > ts[None, :]).mean() + 0.5 * (tn[:, None] == ts[None, :]).mean()), 5) if len(ts) and len(tn) else None
        G['rank_tp_fn_fp'] = {r: [rank_tp[r], rank_fn[r], rank_fp[r]] for r in range(1, 11)}
        G['fp_type'] = dict(fp_type)
        G['strata_loss_points'] = {k: {'refs': v[0], 'f05': round(1 - v[1] / v[0], 5), 'loss_points': round(v[1] / n * 100, 4)}
                                   for k, v in sorted(strata.items())}
        edges = [0, .02, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, 1.0001]
        b = np.digitize(ph, edges) - 1
        G['calibration'] = [{'bin': f'{edges[i]:.2f}-{min(edges[i + 1], 1):.2f}', 'n': int((b == i).sum()),
                             'mean_p': round(float(ph[b == i].mean()), 4), 'pos_rate': round(float(yh[b == i].mean()), 4),
                             'pos_rate_decoy': round(float(yh[(b == i) & dh].mean()), 4) if ((b == i) & dh).any() else None}
                            for i in range(len(edges) - 1) if (b == i).any()]
        E[gname] = G
        log(gname, json.dumps({k: G[k] for k in ('rule', 'dev', 'holdout', 'ceilings', 'loss_points')}))
        (FUS / 'gen2_eda.json').write_text(json.dumps(E, indent=1, default=str))
    with open(FUS / 'gen2_examples.tsv', 'w', encoding='utf-8') as f:
        f.write('gen\tkind\tref\tref_name\tref_addr\trank\ttarget\ttgt_name\ttgt_addr\tp\tgold\tchosen\n')
        for e in examples:
            f.write('\t'.join(str(x).replace('\t', ' ') for x in e) + '\n')
    log('done')


if __name__ == '__main__':
    main()
