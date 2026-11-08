"""Transfer check for cross-encoders (adopted after the listwise failure): does a model score test pairs the way it
scores sample pairs, on pairs the other models agree on?

Agreed positives: v5 probability >= 0.99 and the reference cross-encoder (S5's mdeberta_full) logit >= 9.
Agreed negatives: v5 probability < 0.02 and reference logit <= -9.
Prints logit percentiles on sample vs test (all, and test by country) for the reference and each member; FLAG when a
member's agreed-positive p10 drops by more than 2 logits from sample to test beyond the reference model's own drop.

  ce_shift_check.py 'name=sample.npy|test_glob;...'
Submission mode: ce_shift_check.py empties S5_DIR SUB_DIR...  -> references S5 matched at p >= 0.99 that are now empty.
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import late_multi as lm  # noqa: E402

REF_SAMPLE = lm.ME / 'artifacts/mdeberta_full/ce_logit_sample.npy'
REF_TEST = f'{lm.AY}/artifacts/test_ce_full/ce_mdeberta_full_shard*of3.npz|{lm.ME}/artifacts/test_ce_ayush/ce_mdeberta_full_shard2of3.npz'
Q_POS, Q_NEG = (1, 5, 10, 25, 50), (50, 75, 90, 95, 99)


def pct(v, q):
    v = v[~np.isnan(v)]
    return [round(float(x), 2) for x in np.percentile(v, q)] if len(v) else None


def files(pattern):
    return sorted({f for p in pattern.split('|') for f in glob.glob(p)})


def shift(members):
    S = np.load(lm.STAGE1)
    s1 = S['stage1'].astype(np.float64)
    ref_s = np.load(REF_SAMPLE).astype(np.float64)
    pos_s, neg_s = (s1 >= 0.99) & (ref_s >= 9), (s1 < 0.02) & (ref_s <= -9)
    refs, tgts = [], []
    with open(lm.CANDS, encoding='utf-8') as f:
        next(f)
        for line in f:
            s, c = line.rstrip('\n').split('\t')
            for t in (c.split(',') if c else []):
                refs.append(s); tgts.append(t)
    pairs = pd.DataFrame({'r': refs, 't': tgts})
    country = {}
    with open(lm.RAW / 'test/test_source1.tsv', encoding='utf-8', newline='') as fh:
        for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
            country[row['entity_id']] = row['country']
    cty = np.array([country[r] for r in refs], dtype=object)
    s1_t = np.nan_to_num(lm.lookup(lm.S1_TEST, pairs, 'prob'), nan=0.0)
    ref_t = lm.lookup(files(REF_TEST), pairs, 'logit')
    pos_t, neg_t = (s1_t >= 0.99) & (ref_t >= 9), (s1_t < 0.02) & (ref_t <= -9)
    R = {'agreed_pairs': {'sample_pos': int(pos_s.sum()), 'sample_neg': int(neg_s.sum()),
                          'test_pos': int(pos_t.sum()), 'test_neg': int(neg_t.sum())}}
    todo = {'reference_mdeberta_full': (ref_s, ref_t)}
    for item in members.split(';'):
        n, p = item.split('=', 1)
        sp, tp = p.split('|', 1)
        todo[n] = (np.load(sp).astype(np.float64), lm.lookup(files(tp), pairs, 'logit'))
    ref_drop = None
    for n, (vs, vt) in todo.items():
        r = {'pos_p': list(Q_POS), 'sample_pos': pct(vs[pos_s], Q_POS), 'test_pos': pct(vt[pos_t], Q_POS),
             'neg_p': list(Q_NEG), 'sample_neg': pct(vs[neg_s], Q_NEG), 'test_neg': pct(vt[neg_t], Q_NEG),
             'test_pos_by_country': {c: pct(vt[pos_t & (cty == c)], Q_POS) for c in sorted(set(cty))},
             'test_missing': int(np.isnan(vt).sum())}
        drop = r['sample_pos'][2] - r['test_pos'][2]
        if ref_drop is None:
            ref_drop = drop
        r['p10_drop_sample_to_test'] = round(drop, 2)
        r['verdict'] = 'FLAG: does not transfer' if drop - ref_drop > 2 else 'ok'
        R[n] = r
        print(n, json.dumps(r), flush=True)
    return R


def empties(s5_dir, subs):
    z = np.load(Path(s5_dir) / 'late_scores.npz', allow_pickle=True)
    best = pd.Series(z['prob']).groupby(z['ref_idx']).max()
    with open(lm.RAW / 'test/test_source1.tsv', encoding='utf-8', newline='') as fh:
        ids = [row['entity_id'] for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE)]
    sure = {ids[i] for i, p in best.items() if p >= 0.99}

    def load(p):
        with open(p, encoding='utf-8') as f:
            next(f)
            return {l.split('\t')[0]: l.rstrip('\n').split('\t')[1] for l in f}
    s5 = load(Path(s5_dir) / 'threshold_0.68/matching_results.tsv')
    R = {'s5_empty_rate': round(sum(not v for v in s5.values()) / len(s5), 4)}
    print('S5 empty-reference rate', R['s5_empty_rate'], flush=True)
    for sub in subs:
        m = load(Path(sub) / 'matching_results.tsv')
        emptied = [r for r in sure if s5.get(r) and not m.get(r)]
        changed = sum(m.get(r) != v for r, v in s5.items())
        lost = [r for r in s5 if s5[r] and not m.get(r)]
        R[str(sub)] = {'empty_rate': round(sum(not m.get(r) for r in s5) / len(s5), 4),
                       'empty_here_matched_in_s5': len(lost), 'of_which_s5_p_ge_0.99': len(emptied),
                       's5_sure_refs': len(sure), 'rows_changed_vs_s5': changed,
                       'examples': emptied[:10]}
        print(sub, json.dumps(R[str(sub)]), flush=True)
    return R


def main():
    if sys.argv[1] == 'empties':
        R = empties(sys.argv[2], sys.argv[3:])
        out = Path(sys.argv[3]).parent / 'empties_check.json'
    else:
        R = shift(sys.argv[1])
        out = lm.ME / 'artifacts/final_night/ce_shift_check.json'
    out.write_text(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
