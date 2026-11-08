"""Rescue stage: true links that the stage-1 top-10 cut drops at ranks 11-20.

On the 100k sample, 0.34% (US) / 0.31% (India) of gold links sit at stage-1 ranks 11-20 (1.1% of references); on test,
France has ~5x more high-probability rank 11-20 candidates per reference than US/India (gap_diag.py). The final
pipeline never sees them. This stage scores rank 11-20 pairs with the v5 top-20 matcher (prune_v1/model_top20, the
same model that scored the test top-20 run) and the three cross-encoders, and adds a pair to the final matches when a
dev-chosen rule accepts it and no other reference already holds the record.

  prep   sample: dev/holdout rank 11-20 rows with v5 p >= FLOOR -> rescue/sample_cands.tsv + sample_rows.npz
         test:   rank 11-20 pairs (top-20 minus top-10 lists) with v5 p >= FLOOR -> rescue/test_cands.tsv + test_rows.npz
  (then score_ce_test.py on both files with full21 / decoyw2 / chunk2; the sample file through a train-data 'test' dir)
  eval   T6's exact dev/holdout decisions (stage3b_prob.npy, expected@0.74) + rescue rule grid; rule chosen on dev,
         holdout scored once (and the reverse, as a stability check) -> rescue/eval.json
  apply  <base matching_results.tsv> <out dir>: rule from eval.json on test, exclusivity against every accepted record,
         matching_results.tsv + candidate_pairs.tsv (top-10 + the rank 11-20 pairs this stage scored)
"""
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
AY = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
ART = ME / 'artifacts'
OUT = Path(os.environ.get('RESCUE_DIR', '/scratch/scai/mtech/aib262467/rescue'))
FLOOR = float(os.environ.get('FLOOR', 0.02))
CES = ('full21', 'decoyw2', 'chunk2')
sys.path[:0] = [str(ME / 'business_entity_resolution/src'), str(AY / 'business_entity_resolution/src')]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_lists(path):
    with open(path, encoding='utf-8') as f:
        next(f)
        return {l.split('\t')[0]: [x for x in l.rstrip('\n').split('\t')[1].split(',') if x] for l in f}


def write_cands(path, refs, tgts):
    by = defaultdict(list)
    for r, t in zip(refs, tgts):
        by[r].append(t)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for r, v in by.items():
            f.write(r + '\t' + ','.join(v) + '\n')


def sample_frame():
    from distractor_check import partition_of
    gold = {k: set(v) for k, v in json.loads((ART / 'matcher_v4/pairs/gold.json').read_text()).items()}
    cp = partition_of(gold)
    pairs = [l.rstrip('\n').split('\t') for l in gzip.open(ART / 'matcher_v4/pairs/pairs.tsv.gz', 'rt', encoding='utf-8')]
    refs = np.array([p[0] for p in pairs], dtype=object)
    tgts = np.array([p[1] for p in pairs], dtype=object)
    return gold, cp, refs, tgts


def prep():
    OUT.mkdir(parents=True, exist_ok=True)
    from submission_pipeline import load_scorer
    gold, cp, refs, tgts = sample_frame()
    m10 = np.load(ART / 'prune_v1/mask_top10.npy').astype(bool)
    m20 = np.load(ART / 'prune_v1/mask_top20.npy').astype(bool)
    y = np.load(ART / 'matcher_v4/pairs/y.npy')
    part = np.array([cp[r] for r in refs], dtype=object)
    rows = np.flatnonzero(m20 & ~m10 & np.isin(part, ['dev', 'holdout']))
    x = np.load(ART / 'matcher_v4/pairs/x.npy', mmap_mode='r')
    scorer = load_scorer(ART / 'prune_v1/model_top20', 8)
    p = np.asarray(scorer.predict(np.asarray(x[rows])), dtype=np.float64)
    keep = p >= FLOOR
    rows, p = rows[keep], p[keep]
    log('sample rank 11-20 dev/holdout rows', int(keep.size), 'kept', int(keep.sum()), 'gold kept', int(y[rows].sum()),
        'gold at 11-20 in dev/holdout', int(y[np.flatnonzero(m20 & ~m10 & np.isin(part, ['dev', 'holdout']))].sum()))
    np.savez(OUT / 'sample_rows.npz', row=rows, ref=refs[rows].astype(str), target=tgts[rows].astype(str), v5=p,
             y=y[rows].astype(np.int8), part=part[rows].astype(str))
    write_cands(OUT / 'sample_cands.tsv', refs[rows], tgts[rows])
    del x, refs, tgts
    # test
    top10 = load_lists(ART / 'submission_v5_top10/candidate_pairs.tsv')
    top20 = load_lists(ART / 'submission_v5_top20/candidate_pairs.tsv')
    not_sub = sum(not set(top10[r]) <= set(top20[r]) for r in top10)
    prob = {}
    for s in (0, 1):
        z = np.load(ART / f'full_v5_s{s}/top20/scores.npz', allow_pickle=True)
        for r, t, q in zip(z['ref'].astype(str), z['target'].astype(str), z['prob']):
            if q >= FLOOR:
                prob[(r, t)] = float(q)
    R, T, P = [], [], []
    for r, lst in top20.items():
        s10 = set(top10[r])
        for t in lst:
            if t not in s10 and (r, t) in prob:
                R.append(r); T.append(t); P.append(prob[(r, t)])
    log('test rank 11-20 pairs with v5 >=', FLOOR, len(R), '| top-10 not inside top-20 for', not_sub, 'refs')
    np.savez(OUT / 'test_rows.npz', ref=np.array(R), target=np.array(T), v5=np.array(P))
    write_cands(OUT / 'test_cands.tsv', R, T)


def load_ce(kind, refs, tgts):
    """Logits of each cross-encoder aligned to (refs, tgts); NaN where missing."""
    out = {}
    key = {(r, t): i for i, (r, t) in enumerate(zip(refs, tgts))}
    for n in CES:
        v = np.full(len(refs), np.nan)
        for f in sorted((OUT / f'ce_{kind}').glob(f'ce_{n}_shard*.npz')):
            z = np.load(f, allow_pickle=True)
            for r, t, l in zip(z['ref'].astype(str), z['target'].astype(str), z['logit']):
                i = key.get((r, t))
                if i is not None:
                    v[i] = l
        out[n] = v
        log(kind, n, 'scored', int(np.isfinite(v).sum()), 'of', len(v))
    return out


def score_of(v5, ce):
    s = 1 / (1 + np.exp(-np.column_stack([ce[n] for n in CES])))
    return v5, np.nanmean(s, axis=1), np.nanmin(s, axis=1)


GRID = [(a, b, k) for a in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) for b in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
        for k in ('mean', 'min')]


def accept(v5, cmean, cmin, rule):
    a, b, k = rule
    return (v5 >= a) & ((cmean if k == 'mean' else cmin) >= b)


def evaluate():
    from distractor_check import decide, f05
    gold, cp, refs, tgts = sample_frame()
    m10 = np.load(ART / 'prune_v1/mask_top10.npy').astype(bool)
    r10, t10 = refs[m10], tgts[m10]
    prob = np.load(ART / 'fusion/boundary_models_t6/stage3b_prob.npy')
    rule6 = json.load(open(ART / 'fusion/boundary_models_t6/stage3b_meta.json'))['rule']
    assert len(prob) == len(r10)
    part10 = np.array([cp[r] for r in r10], dtype=object)
    Z = np.load(OUT / 'sample_rows.npz', allow_pickle=True)
    ce = load_ce('sample', Z['ref'].astype(str), Z['target'].astype(str))
    v5, cmean, cmin = score_of(Z['v5'], ce)
    R = {'rule_t6': rule6, 'floor': FLOOR}
    base, cand = {}, {}
    for p in ('dev', 'holdout'):
        prefs = sorted(s for s in gold if cp[s] == p)
        pos = {s: i for i, s in enumerate(prefs)}
        idx = np.flatnonzero(part10 == p)
        rix = np.array([pos[r] for r in r10[idx]])
        ch = decide(rix, prob[idx], rule6[0], rule6[1], len(prefs))
        pred = defaultdict(set)
        for i, t in zip(rix[ch], t10[idx][ch]):
            pred[prefs[i]].add(t)
        base[p] = (prefs, pred)
        R[f'{p}_t6'] = round(float(np.mean([f05(gold[s], pred.get(s, set())) for s in prefs])), 6)
        cand[p] = np.flatnonzero(Z['part'].astype(str) == p)
    log('T6 reproduced', {k: v for k, v in R.items() if k.endswith('_t6')})
    held = {p: set().union(*base[p][1].values()) for p in base}

    def run(p, rule):
        prefs, pred = base[p]
        j = cand[p]
        acc = j[accept(v5[j], cmean[j], cmin[j], rule)]
        best = {}
        for i in acc:  # one reference per record, highest v5 first; never a record another reference holds
            t = Z['target'][i]
            if t in held[p]:
                continue
            if t not in best or v5[i] > v5[best[t]]:
                best[t] = i
        add = defaultdict(set)
        for t, i in best.items():
            add[Z['ref'][i]].add(t)
        tp = sum(int(Z['y'][i]) for i in best.values())
        f = np.mean([f05(gold[s], pred.get(s, set()) | add.get(s, set())) for s in prefs])
        return float(f), len(best), tp
    res = {p: {r: run(p, r) for r in GRID} for p in ('dev', 'holdout')}
    for a, b in (('dev', 'holdout'), ('holdout', 'dev')):
        g = max(GRID, key=lambda r: (res[a][r][0], -res[a][r][1]))
        R[f'chosen_on_{a}'] = {'rule': g, a: round(res[a][g][0], 6), b: round(res[b][g][0], 6),
                               f'{b}_gain': round(res[b][g][0] - R[f'{b}_t6'], 6),
                               f'{b}_added': res[b][g][1], f'{b}_true': res[b][g][2]}
    R['top_by_dev'] = [(r, round(res['dev'][r][0] - R['dev_t6'], 6), res['dev'][r][1:], round(res['holdout'][r][0] - R['holdout_t6'], 6),
                        res['holdout'][r][1:]) for r in sorted(GRID, key=lambda r: -res['dev'][r][0])[:12]]
    R['gold_rank11_20'] = {p: int(Z['y'][cand[p]].sum()) for p in cand}
    (OUT / 'eval.json').write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps(R, indent=1, default=str), flush=True)


def apply(base_tsv, out_dir):
    import csv
    E = json.load(open(OUT / 'eval.json'))
    rule = tuple(E['chosen_on_dev']['rule'])
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    Z = np.load(OUT / 'test_rows.npz', allow_pickle=True)
    ref, tgt = Z['ref'].astype(str), Z['target'].astype(str)
    ce = load_ce('test', ref, tgt)
    v5, cmean, cmin = score_of(Z['v5'], ce)
    base = load_lists(base_tsv)
    held = set().union(*map(set, base.values()))
    acc = np.flatnonzero(accept(v5, cmean, cmin, rule))
    best = {}
    for i in acc:
        if tgt[i] in held:
            continue
        if tgt[i] not in best or v5[i] > v5[best[tgt[i]]]:
            best[tgt[i]] = i
    add = defaultdict(list)
    for t, i in best.items():
        add[ref[i]].append(t)
    cty = {}
    with open(ME / 'data/dataset/test/test_source1.tsv', encoding='utf-8', newline='') as fh:
        for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
            cty[row['entity_id']] = row['country']
    top10 = load_lists(ART / 'submission_v5_top10/candidate_pairs.tsv')
    extra = defaultdict(list)
    for r, t in zip(ref, tgt):
        extra[r].append(t)
    with open(out / 'matching_results.tsv', 'w', encoding='utf-8') as f, \
            open(out / 'candidate_pairs.tsv', 'w', encoding='utf-8') as g:
        f.write('source1_entity_id\tmatched_entity_ids\n'); g.write('source1_entity_id\tcandidate_entity_ids\n')
        for r in base:
            f.write(r + '\t' + ','.join(base[r] + add.get(r, [])) + '\n')
            g.write(r + '\t' + ','.join(top10[r] + extra.get(r, [])) + '\n')
    st = Counter()
    for r, v in add.items():
        st[cty[r]] += len(v); st[cty[r] + '_refs'] += 1; st[cty[r] + '_was_empty'] += not base[r]
    n = Counter(cty.values())
    R = {'rule': rule, 'scored_pairs': len(ref), 'accepted_before_exclusivity': int(len(acc)), 'added': len(best),
         'by_country': {c: {'added': st[c], 'per_ref': round(st[c] / n[c], 5), 'refs': st[c + '_refs'],
                            'refs_that_were_empty': st[c + '_was_empty']} for c in n},
         'candidates_per_ref': round((sum(map(len, top10.values())) + len(ref)) / len(top10), 4)}
    (out / 'rescue_report.json').write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1), flush=True)


if __name__ == '__main__':
    {'prep': prep, 'eval': evaluate}.get(sys.argv[1], lambda: apply(sys.argv[2], sys.argv[3]))()
