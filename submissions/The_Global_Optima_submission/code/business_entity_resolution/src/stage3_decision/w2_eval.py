"""Does W2 (rival already has a same-source copy) break blank-address ties? Holdout blank rows in T5's 0.1-0.9 band:
true match rate vs mean T5 probability per W2 condition (a gap >= 0.1 is a usable signal). Writes OUT/w2_eval.json.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

A = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
sys.path.insert(0, str(A / 'business_entity_resolution/src'))
from distractor_check import partition_of  # noqa: E402

V4 = Path('/scratch/scai/mtech/aib262015/vishal_v4_artifacts')


def main():
    out = Path(sys.argv[1])
    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    cp = partition_of(gold)
    mask = np.load(A / 'artifacts/prune_v1_copy/mask_top10.npy')
    refs = []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                refs.append(line.split('\t', 1)[0])
    part = np.array([cp[r] for r in refs], dtype=object)
    S = np.load(A / 'artifacts/fusion/stage1.npz')
    y = np.load(V4 / 'pairs/y.npy')[S['rows']].astype(bool)
    blank = np.load(A / 'artifacts/fusion/boundary_models_t4/exp2_cache.npz')['blank']
    p = np.load(ME / 'artifacts/fusion/boundary_models_t5/stage3b_prob.npy')
    W = np.load(out / 'w2_sample.npy')
    names = json.load(open(out / 'w2_sample.json'))['features']
    c = {n: W[:, i] for i, n in enumerate(names)}
    base = (part == 'holdout') & blank & (p > 0.1) & (p < 0.9) & np.isfinite(W[:, 0])
    groups = {
        'no_rival': c['w2_n_rivals'] == 0,
        'ref_lacks_copy_all_rivals_have_one': (c['w2_ref_clear_src'] == 0) & (c['w2_n_rivals'] >= 1) & (c['w2_rivals_lacking'] == 0),
        'ref_lacks_copy_some_rival_lacks_too': (c['w2_ref_clear_src'] == 0) & (c['w2_rivals_lacking'] >= 1),
        'ref_has_copy_all_rivals_have_one': (c['w2_ref_clear_src'] >= 1) & (c['w2_n_rivals'] >= 1) & (c['w2_rivals_lacking'] == 0),
        'ref_has_copy_some_rival_lacks': (c['w2_ref_clear_src'] >= 1) & (c['w2_rivals_lacking'] >= 1),
        'slot_ratio_ge_1': c['w2_slot_ratio'] >= 1, 'slot_ratio_0_1': (c['w2_slot_ratio'] > 0) & (c['w2_slot_ratio'] < 1),
        'slot_ratio_0': c['w2_slot_ratio'] == 0,
        'ref_key_differs': c['w2_ref_same_key'] == 0,
    }
    R = {'band_rows': int(base.sum())}
    for g, m in groups.items():
        mm = base & m
        if mm.sum() >= 15:
            rate, pr = float(y[mm].mean()), float(p[mm].mean())
            R[g] = {'rows': int(mm.sum()), 'match_rate': round(rate, 3), 'mean_p_T5': round(pr, 3),
                    'gap': round(rate - pr, 3), 'signal': abs(rate - pr) >= 0.1}
    (out / 'w2_eval.json').write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
