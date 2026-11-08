"""Blank-address candidates under the S5 decision: how much loss, and is any of it recoverable?

A blank-address S2/S3 record can only be tied to its S1 entity by name. It is recoverable when the name points to one
S1 entity (name ambiguity 1), or when it agrees with records already accepted for the reference (relational context,
collective ER); it is a coin flip when several S1 entities share the name. Measured on holdout with the S5 L2
probability (trained on trainA+trainB, threshold@0.68):
  ceilings    - S5 decision everywhere except blank-address rows decided by the label (and the same for non-blank)
  by bucket   - positive rate vs mean probability for blank rows, split by name ambiguity over the FULL train S1
                (exact order-free suffix-free key; a looser key), key equality with the reference, sibling agreement
                with accepted candidates, and whether the reference already has an accepted record from that source
  negatives   - blank false candidates in the ambiguous band: owned by another S1 entity vs decoy
Writes fusion/blank_eda.json.
"""
import gzip
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from boundary_exp import CE, FUS, PRUNE, RAW, V4, W, key, norm
from distractor_check import f05, partition_of


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def loose(name):
    return ' '.join(sorted({t for t in re.sub(r'\d+', ' ', key(name)).split() if len(t) >= 3}))


def main():
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
    owner = {}
    with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if len(q) > 1 and q[1]:
                for t in q[1].split(','):
                    owner[t.strip()] = q[0]
    wt = np.where((y == 0) & np.array([t not in owner for t in tgts]), W, 1.0).astype(np.float32)
    tr, dv, ho = np.isin(part, ['trainA', 'trainB']), part == 'dev', np.flatnonzero(part == 'holdout')
    X = np.column_stack([s1f, ce]).astype(np.float32)
    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                  max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)
    d = xgb.QuantileDMatrix(X[tr], y[tr], weight=wt[tr])
    b = xgb.train(params, d, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=d), 'dev')],
                  early_stopping_rounds=100, verbose_eval=False)
    P = np.asarray(b.inplace_predict(X[ho], iteration_range=(0, b.best_iteration + 1)), dtype=np.float64)
    log('L2 done')

    need = set(refs[ho]) | set(tgts[ho]) | {owner[t] for t in tgts[ho] if t in owner}
    text, kc, lc = {}, Counter(), Counter()
    for k in (1, 2, 3):
        with open(RAW / f'train_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) <= iaddr:
                    continue
                if k == 1:
                    kc[key(q[iname])] += 1; lc[loose(q[iname])] += 1
                if q[ie] in need:
                    text[q[ie]] = (q[iname], q[iaddr])
    log('text + S1 name counts', len(text), len(kc))

    th, yh, rh = tgts[ho], y[ho], refs[ho]
    chosen = P >= 0.68
    blank = np.array([not text.get(t, ('', ''))[1].strip() for t in th])
    by_ref = defaultdict(list)
    for j, r in enumerate(rh):
        by_ref[r].append(j)
    H = sorted(s for s in gold if cp[s] == 'holdout')

    def macro(sel):
        return float(np.mean([f05(gold[s], {th[j] for j in by_ref.get(s, []) if sel[j]}) for s in H]))

    E = {'n_holdout_rows': int(len(ho)), 'blank_rows': int(blank.sum()), 'blank_pos': int(yh[blank].sum()),
         'blank_share_rows': round(float(blank.mean()), 4)}
    base = macro(chosen)
    E['ceilings'] = {'s5': round(base, 6),
                     'blank_rows_by_label': round(macro(np.where(blank, yh > 0, chosen)), 6),
                     'nonblank_rows_by_label': round(macro(np.where(blank, chosen, yh > 0)), 6),
                     'all_by_label': round(macro(yh > 0), 6)}
    E['errors'] = {'blank_fn': int((blank & (yh > 0) & ~chosen).sum()), 'blank_fp': int((blank & (yh == 0) & chosen).sum()),
                   'nonblank_fn': int((~blank & (yh > 0) & ~chosen).sum()), 'nonblank_fp': int((~blank & (yh == 0) & chosen).sum())}
    log(E)

    feats = defaultdict(dict)
    for r, js in by_ref.items():
        rn = text.get(r, ('', ''))[0]
        rk = key(rn)
        acc = [j for j in js if chosen[j] and not blank[j]]
        for j in js:
            if not blank[j]:
                continue
            tn = text.get(th[j], ('', ''))[0]
            tk, tl = key(tn), loose(tn)
            sib = max((fuzz.token_sort_ratio(norm(tn), norm(text.get(th[q], ('', ''))[0])) for q in acc if q != j), default=-1)
            same_src = any(th[q][:2] == th[j][:2] for q in acc)
            m = kc.get(tk, 0)
            ml = lc.get(tl, 0)
            feats[j] = {'amb_key': '0' if m == 0 else '1' if m == 1 else '2' if m == 2 else '3-5' if m <= 5 else '6+',
                        'amb_loose': '0' if ml == 0 else '1' if ml == 1 else '2' if ml == 2 else '3-5' if ml <= 5 else '6+',
                        'key_eq_ref': str(bool(rk) and tk == rk),
                        'sibling_sim': 'none' if sib < 0 else '<70' if sib < 70 else '70-89' if sib < 90 else '90+',
                        'ref_has_same_src': str(same_src)}
    band = [j for j in feats if 0.1 <= P[j] < 0.9]
    E['blank_band_rows'] = len(band)
    for f in ('amb_key', 'amb_loose', 'key_eq_ref', 'sibling_sim', 'ref_has_same_src'):
        for nm, rows in (('all', list(feats)), ('band', band)):
            g = defaultdict(list)
            for j in rows:
                g[feats[j][f]].append(j)
            E[f'{f}/{nm}'] = {v: {'n': len(js), 'pos_rate': round(float(yh[js].mean()), 4),
                                  'mean_p': round(float(P[js].mean()), 4), 'fn': int(((yh[js] > 0) & ~chosen[js]).sum()),
                                  'fp': int(((yh[js] == 0) & chosen[js]).sum())} for v, js in sorted(g.items())}
    neg = Counter()
    for j in band:
        if yh[j] == 0:
            o = owner.get(th[j])
            if o is None:
                neg['decoy'] += 1
            else:
                on = key(text.get(o, ('', ''))[0]) if o in text else None
                neg['owned_other_same_key' if on is not None and on == key(text.get(th[j], ('', ''))[0])
                    else 'owned_other'] += 1
    E['band_negatives'] = dict(neg)
    (FUS / 'blank_eda.json').write_text(json.dumps(E, indent=1))
    log('done')


if __name__ == '__main__':
    main()
