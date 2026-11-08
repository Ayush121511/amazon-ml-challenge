"""Train-sample -> test transfer audit of every stage-3b input (the listwise lesson: holdout does not see test shift).

On pairs every base decider agrees on, the inputs should look alike in the sample and on test:
  sure-match     v5 >= 0.99 and chunk2 logit >= 9
  sure-nonmatch  v5 <= 0.02 and chunk2 logit <= -5
For each feature and for the stage-3b probability: sample vs test quantiles (p1/p10/p50/p90) and the KS distance, per
group and for test countries separately. Features ranked by the worst KS on sure-match pairs. A large shift there is
the listwise failure signature.
Env CACHE (boundary_models_*/exp2_cache.npz), META (stage3b_meta.json), P3_SAMPLE (stage3b_prob.npy),
TEST_DIR (stage3b_predict output with features.npy + late_scores.npz). Writes TEST_DIR/shift_audit.json.
"""
import json
import os
from pathlib import Path

import numpy as np

HOME = Path.home()
FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/test'
C2S = '/home/scai/mtech/aib262140/scratch/amazon_ml_2026/ftchunk2/ce_logit_sample.npy'


def ks(a, b):
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if not len(a) or not len(b):
        return float('nan')
    g = np.unique(np.concatenate([a, b]))
    if len(g) > 4000:
        g = np.quantile(g, np.linspace(0, 1, 4000))
    return float(np.max(np.abs(np.searchsorted(a, g, side='right') / len(a) - np.searchsorted(b, g, side='right') / len(b))))


def qs(a):
    a = a[np.isfinite(a)]
    return [round(float(np.quantile(a, q)), 3) for q in (0.01, 0.1, 0.5, 0.9)] if len(a) else []


def main():
    C = np.load(os.environ['CACHE'])
    names = json.load(open(os.environ['META']))['features']
    Xs = np.column_stack([C['F'], C['Frev'], C['Fnum']]).astype(np.float32)
    names = names[:Xs.shape[1]]
    S = np.load(FUS / 'stage1.npz')
    v5s = np.where(S['stage1'] < 0.02, 0.0, S['stage1'])
    c2s = np.nan_to_num(np.load(C2S))
    p3s = np.load(os.environ['P3_SAMPLE'])
    T = Path(os.environ['TEST_DIR'])
    Z = np.load(T / 'late_scores.npz', allow_pickle=True)
    Xt = np.load(T / 'features.npy', mmap_mode='r')
    v5t, c2t, p3t, ri = Z['stage1'], Z['ce'], Z['prob'], Z['ref_idx']
    cty = []
    with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ic = head.index('country')
        for line in fh:
            q = line.rstrip('\n').split('\t')
            cty.append(q[ic] if len(q) > ic else '')
    pc = np.array(cty, dtype=object)[ri]
    rng = np.random.default_rng(0)
    groups = {'sure_match': ((v5s >= 0.99) & (c2s >= 9), (v5t >= 0.99) & (c2t >= 9)),
              'sure_nonmatch': ((v5s <= 0.02) & (c2s <= -5), (v5t <= 0.02) & (c2t <= -5))}
    R = {'n_features': len(names), 'groups': {}}
    for g, (ms, mt) in groups.items():
        js = np.flatnonzero(ms); js = rng.choice(js, min(200000, len(js)), replace=False)
        jt = np.flatnonzero(mt); jt = np.sort(rng.choice(jt, min(400000, len(jt)), replace=False))
        Xtg = np.asarray(Xt[jt, :Xs.shape[1]])
        rows = []
        for k, nm in enumerate(names + ['stage3b_p']):
            a = Xs[js, k] if nm != 'stage3b_p' else p3s[js]
            b = Xtg[:, k] if nm != 'stage3b_p' else p3t[jt]
            by = {c: round(ks(a, b[pc[jt] == c]), 4) for c in ('US', 'India', 'France')}
            rows.append({'feature': nm, 'ks_all': round(ks(a, b), 4), 'ks_by_country': by,
                         'sample_q': qs(a), 'test_q': qs(b)})
        rows.sort(key=lambda r: -r['ks_all'])
        R['groups'][g] = {'n_sample': int(len(js)), 'n_test': int(len(jt)), 'features': rows}
        print(g, [(r['feature'], r['ks_all'], r['ks_by_country']) for r in rows[:12]], flush=True)
    (T / 'shift_audit.json').write_text(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
