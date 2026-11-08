"""Tables from error_sweep.py outputs: loss per stage and pattern cell, train (actual, dev + holdout) vs test (expected),
and the public-score ceiling of fixing each cell.

Public impact, US/India: holdout loss rate per country x test S1 share of that country (score units, 1.0 = perfect).
Cross-check: test expected loss in the cell x the cell's holdout actual/expected ratio. France: test expected only
(no labels; census-changed pairs are counted at the model's own p, i.e. as the model would score them).
Env TRAIN (json of MODE=train), TEST (json of MODE=test), OUT (markdown).
"""
import json
import os

import pandas as pd

TR, TE = os.environ['TRAIN'], os.environ['TEST']
R = json.load(open(TR)); RT = json.load(open(TE))
df = pd.read_pickle(TR + '.train_pairs.pkl'); bf = pd.read_pickle(TR + '.train_blocking.pkl')
sf = pd.read_pickle(TR + '.train_s1.pkl')
tf = pd.read_pickle(TE + '.test_pairs.pkl')
NT = RT['n_s1']; NTOT = sum(NT.values())
NH = {c: int((sf.part.eq('holdout') & sf.country.eq(c)).sum()) for c in ('US', 'India')}
ND = {c: int((sf.part.eq('dev') & sf.country.eq(c)).sum()) for c in ('US', 'India')}
USIN = ('US', 'India')
out = []


def w(s=''):
    out.append(s)


def public_from_holdout(g, part='holdout', col='loss'):
    N = NH if part == 'holdout' else ND
    return sum(g[(g.part == part) & (g.country == c)][col].sum() / N[c] * NT.get(c, 0) / NTOT for c in USIN)


# ---------------- stage summary ----------------
w('# Error sweep: T6 (US/India) + U2 France, train actual vs test expected')
w()
w(f"Test S1: {NT} (total {NTOT}). Holdout S1 US {NH['US']} India {NH['India']}; dev US {ND['US']} India {ND['India']}.")
w()
w('## 1. Stage totals (loss points = 100 x mean(1 - F0.5) over S1)')
w()
for p in ('dev', 'holdout'):
    t = R['totals'][p]
    w(f"- **{p}**: total {t['loss_points']} (in-list expected {t['exp_points_inlist']}); blocking {t['blocking_points']} "
      f"{t['by_stage']}; decision FN {t['decision_FN']}, FP {t['decision_FP']}; by S1 case {t['case_points']}; S1 with >10 gold: {t['gold_over_10']}")
w(f"- **test expected in-list points** by country: {RT['expected_points_inlist']}; public-weighted {RT['expected_points_public_inlist']}")
w(f"- test pred-empty rate {RT['pred_empty_rate']}; S1 with all 10 accepted {RT['all10_accepted']}; post-processing steps {RT['steps']}")
hold_exp = sf[sf.part == 'holdout'].groupby('country').exp_loss.mean() * 100
hold_act = sf[sf.part == 'holdout'].groupby('country').loss.mean() * 100
w(f"- holdout actual vs in-list expected by country: " + ', '.join(f"{c}: {hold_act[c]:.3f} vs {hold_exp[c]:.3f}" for c in USIN))
w()

# ---------------- decision cells ----------------
E = df[df.err == 1]
keys = ['ekind', 'blank', 'name', 'addr']
TU = tf[tf.country.isin(USIN)]
te = TU.groupby(keys).agg(test_n=('p', 'size'), test_exp=('exp_loss', 'sum'))
tr = df.groupby(keys).agg(train_n=('p', 'size'), exp=('exp_loss', 'sum'))
rows = []
for key, ge in E.groupby(keys):
    r = dict(zip(keys, key))
    r['err_hold'] = int((ge.part == 'holdout').sum()); r['err_dev'] = int((ge.part == 'dev').sum())
    r['pub_hold'] = public_from_holdout(ge); r['pub_dev'] = public_from_holdout(ge, 'dev')
    ex = tr.exp.get(key, 0.0)
    r['act/exp'] = ge.loss.sum() / ex if ex > 0 else float('nan')
    r['test_exp_pub'] = te.test_exp.get(key, 0.0) / NTOT
    r['test_cal_pub'] = r['test_exp_pub'] * (r['act/exp'] if ex > 0 else 1)
    r['share_tr'] = tr.train_n.get(key, 0) / len(df); r['share_te'] = te.test_n.get(key, 0) / max(1, len(TU))
    rows.append(r)
C = pd.DataFrame(rows).sort_values('pub_hold', ascending=False)
w('## 2. US/India decision errors by cell (public-score ceiling if the cell were fixed)')
w()
w('pub_* = score units on the public LB (0.001 = +0.1 points). pub_hold/pub_dev from actual labelled loss reweighted to '
  'the test country mix; test_exp_pub = label-free expected loss on test; test_cal_pub = test_exp x holdout act/exp.')
w()
w(C.head(30).to_string(index=False, float_format=lambda x: f'{x:.5f}'))
w()
w(f"Sum pub_hold (decision, US/India): {C.pub_hold.sum():.5f}; sum test_cal_pub: {C.test_cal_pub.sum():.5f}")
w()

# ---------------- one-way breakdowns ----------------
w('## 3. One-way breakdowns of US/India decision loss (pub_hold | pub_dev | test_exp_pub)')
for k in ('mult', 'shift', 'pband', 'case', 'src', 'country'):
    w()
    w(f'### by {k}')
    rr = []
    for key, g in E.groupby([k, 'ekind']):
        tu = tf[(tf.country.isin(USIN)) & (tf[k] == key[0]) & (tf.ekind == key[1])] if k in tf else None
        rr.append({k: key[0], 'ekind': key[1], 'err_hold': int((g.part == 'holdout').sum()), 'pub_hold': public_from_holdout(g),
                   'pub_dev': public_from_holdout(g, 'dev'),
                   'test_exp_pub': tu.exp_loss.sum() / NTOT if tu is not None else float('nan')})
    w(pd.DataFrame(rr).sort_values('pub_hold', ascending=False).to_string(index=False, float_format=lambda x: f'{x:.5f}'))

# ---------------- blank-address ties: multiplicity x name relation ----------------
w()
w('## 4. Blank-address errors by S1 name multiplicity x name relation (pub_hold)')
B = E[E.blank == 'blank']
w(B.groupby(['ekind', 'mult', 'name']).apply(lambda g: pd.Series({'err_hold': int((g.part == 'holdout').sum()), 'pub_hold': public_from_holdout(g), 'pub_dev': public_from_holdout(g, 'dev')}))
  .reset_index().sort_values('pub_hold', ascending=False).head(20).to_string(index=False, float_format=lambda x: f'{x:.5f}'))

# ---------------- blocking ----------------
w()
w('## 5. Blocking misses (true record not in the top-10 list)')
rr = []
for key, g in bf.groupby(['stage', 'blank', 'name', 'addr']):
    rr.append(dict(zip(['stage', 'blank', 'name', 'addr'], key), n_hold=int((g.part == 'holdout').sum()),
                   pub_hold=public_from_holdout(g), pub_dev=public_from_holdout(g, 'dev')))
w(pd.DataFrame(rr).sort_values('pub_hold', ascending=False).head(20).to_string(index=False, float_format=lambda x: f'{x:.5f}'))
w()
w('by case / gold size: ' + str(bf[bf.part == 'holdout'].groupby('case').loss.sum().round(2).to_dict()) +
  ' | gold size of missed: ' + str(bf[bf.part == 'holdout'].ngold.clip(upper=12).value_counts().sort_index().to_dict()))

# ---------------- calibration ----------------
w()
w('## 6. Calibration of uncertain pairs (0.05 < p < 0.95), dev + holdout, US/India')
U = df[(df.p > 0.05) & (df.p < 0.95)]
cal = U.groupby(['blank', 'name', 'addr']).agg(n=('y', 'size'), mean_p=('p', 'mean'), true_rate=('y', 'mean')).reset_index()
cal = cal[cal.n >= 100].sort_values('n', ascending=False)
w(cal.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
w()
rc = R['recalibration']
w(f"Per-cell logit recalibration (dev-fitted, holdout once): base {rc['holdout_base']}, all cells {rc['all_cells_delta']:+.6f}")
w(pd.DataFrame(rc['cells'], columns=['cell', 'shift', 'n', 'holdout_delta']).head(15).to_string(index=False))

# ---------------- France ----------------
w()
w('## 7. France (test, expected loss at FRtrained_w4 p, U2 decisions)')
F = tf[tf.country == 'France']
fr = F.groupby(['ekind', 'blank', 'name', 'addr', 'step']).agg(n=('p', 'size'), mean_p=('p', 'mean'), exp_pub=('exp_loss', 'sum')).reset_index()
fr['exp_pub'] /= NTOT
w(fr.sort_values('exp_pub', ascending=False).head(30).to_string(index=False, float_format=lambda x: f'{x:.5f}'))
w()
w('France by step: ' + str((F.groupby('step').exp_loss.sum() / NTOT).round(5).to_dict()))
w('France by shift: ' + str((F.groupby(['ekind', 'shift']).exp_loss.sum() / NTOT).round(5).sort_values(ascending=False).head(12).to_dict()))
w('France by mult: ' + str((F.groupby(['ekind', 'mult']).exp_loss.sum() / NTOT).round(5).sort_values(ascending=False).to_dict()))
w()
w('US/India test by step: ' + str((tf[tf.country.isin(USIN)].groupby('step').exp_loss.sum() / NTOT).round(5).to_dict()))
open(os.environ['OUT'], 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
