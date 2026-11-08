"""Why does T1 leave 16.9k references empty that S5 matched with near-certainty?

For each such reference: S5's match, and T1's view of every candidate (stage-3b p3, L2, v5, chunk2, listwise), with
text. Summary: listwise logit distribution on these pairs vs pairs every decider agrees on, and by candidate position
in the list (a position or length effect would show here). Writes fusion/t1_empty_check.txt.
"""
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HOME = Path.home()
A = HOME / 'scratch/amazon_ml_2026/artifacts'
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/test'
VCAND = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/submission_v5_top10/candidate_pairs.tsv')
out = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); out.append(s)


def load(f):
    d = {}
    with open(f, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            s, m = (line.rstrip('\n').split('\t') + [''])[:2]
            d[s] = [x for x in m.split(',') if x]
    return d


def main():
    t1 = load(A / 'submission_T1_s3b_c2lw/expected_0.6/matching_results.tsv')
    s5 = load(A / 'submission_S5/threshold_0.68/matching_results.tsv')
    Z = np.load(A / 'submission_T1_s3b_c2lw/late_scores.npz', allow_pickle=True)
    ri, tg = Z['ref_idx'], Z['target'].astype(str)
    p3, l2, c2, v5 = Z['prob'], Z['l2'], Z['ce'], Z['stage1']
    refs, cty = [], {}
    with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, ic = head.index('entity_id'), head.index('country')
        for line in fh:
            q = line.rstrip('\n').split('\t')
            refs.append(q[ie]); cty[q[ie]] = q[ic]
    pos = {r: i for i, r in enumerate(refs)}
    lw = {}
    for k in (0, 1):
        z = np.load(A / f'listwise/ce_listwise_shard{k}of2.npz', allow_pickle=True)
        for r, t, v in zip(z['ref'], z['target'], z['logit']):
            lw[(str(r), str(t))] = float(v)
    say('listwise test pairs', len(lw))
    cand = {}
    with open(VCAND, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            s, c = line.rstrip('\n').split('\t')
            cand[s] = c.split(',') if c else []
    newly = [r for r in refs if not t1.get(r) and s5.get(r)]
    say('newly empty', len(newly))
    idx = defaultdict(dict)
    nset = {pos[r] for r in newly}
    for j in np.flatnonzero(np.isin(ri, list(nset))):
        idx[ri[j]][tg[j]] = j
    L_new, P_new = [], Counter()
    for r in newly:
        for t in s5[r]:
            L_new.append(lw.get((r, t), np.nan))
            P_new[cand[r].index(t) + 1 if t in cand[r] else -1] += 1
    L_new = np.array(L_new)
    ok = (v5 >= 0.99) & (c2 >= 9)
    samp = np.random.default_rng(0).choice(np.flatnonzero(ok), 200000, replace=False)
    L_ok = np.array([lw.get((refs[ri[j]], tg[j]), np.nan) for j in samp])
    q = lambda a: {k: round(float(np.nanquantile(a, k)), 2) for k in (0.1, 0.25, 0.5, 0.75, 0.9)}
    say('listwise logit on the dropped S5 matches:', q(L_new), 'nan', int(np.isnan(L_new).sum()))
    say('listwise logit on pairs with v5>=0.99 & chunk2>=9:', q(L_ok))
    say('candidate-list position of the dropped S5 match:', dict(sorted(P_new.items())))
    pos_ok = Counter()
    for j in samp[:50000]:
        r = refs[ri[j]]
        pos_ok[cand[r].index(tg[j]) + 1 if tg[j] in cand[r] else -1] += 1
    say('candidate-list position of agreed pairs:', dict(sorted(pos_ok.items())))
    say('by country:', Counter(cty[r] for r in newly))
    need = set(newly[:4000]) | {t for r in newly[:4000] for t in cand[r]}
    text = {}
    for k in (1, 2, 3):
        with open(RAW / f'test_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
            for line in fh:
                q_ = line.rstrip('\n').split('\t')
                if len(q_) > iaddr and q_[ie] in need:
                    text[q_[ie]] = (q_[iname], q_[iaddr])
    rng = np.random.default_rng(1)
    for r in rng.choice(newly[:4000], 30, replace=False):
        rn, ra = text.get(r, ('', ''))
        say(f'\n**{r}** [{cty[r]}] {rn} | {ra}')
        for n, t in enumerate(cand[r], 1):
            j = idx[pos[r]].get(t)
            tn, ta = text.get(t, ('', ''))
            flag = 'S5' if t in s5[r] else '  '
            vals = f'p3={p3[j]:.2f} l2={l2[j]:.2f} v5={v5[j]:.2f} c2={c2[j]:5.1f} lw={lw.get((r, t), float("nan")):5.1f}' if j is not None else ''
            say(f'  {n:2d} {flag} {vals} | {tn} | {ta}')
    (A / 'fusion/t1_empty_check.txt').write_text('\n'.join(out))


if __name__ == '__main__':
    main()
