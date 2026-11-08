"""Address-evidence fix for countries without training labels (every country except US and India; on test: France).

France loss analysis (france_loss.py): France is expected to miss 0.161 links per reference against 0.063 actually missed
in the US/India holdout. Two of the excess types are decided by the address, which the France-robust / France-trained
stage-3b does not see (address features removed as France-shifted):
  same_alias_invented   invented name (words never used in any Source-1 name of the country) at the same house number
                        and near-identical address
  addr_other            same house number, address written differently (region / department names, abbreviations)
The full-feature stage-3b T6 keeps the address features and is 99.5-99.9% precise on both types in the US/India
holdout. A rejected candidate of these types is added when T6 >= TAU, no reference holds the record, and (among new
links) the reference has the highest T6 probability for it. Remove-only steps (census, exclusivity prune) are untouched.
Usage: nonus_fix.py <base matching_results.tsv> <out dir>   (env TAU, default 0.9)
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
ART = ME / 'artifacts'
TAU = float(os.environ.get('TAU', 0.9))
TYPES = {'same_alias_invented', 'addr_other'}
LABELLED = {'US', 'India'}
sys.path.insert(0, str(ME / 'business_entity_resolution/src'))
from france_loss import read, rtype, words  # noqa: E402


def f05(tp, pred, gold):
    if not gold and not pred:
        return 1.0
    if tp == 0:
        return 0.0
    p, r = tp / pred, tp / gold
    return 1.25 * p * r / (0.25 * p + r)


def main():
    base_path, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    text, vocab = read('test')
    refs, tgts = [], []
    with open(ART / 'submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
        next(f)
        for line in f:
            s, c = line.rstrip('\n').split('\t')
            refs.append(s); tgts.extend(c.split(','))
    n = len(refs)
    T = np.array(tgts, dtype=object).reshape(n, 10)
    p6 = np.load(ART / 'submission_T6_own/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)
    pF = np.load(ART / 'submission_FRtrained_w4/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)
    rows, base = [], {}
    with open(base_path, encoding='utf-8') as f:
        next(f)
        for l in f:
            r, m = l.rstrip('\n').split('\t'); rows.append(r); base[r] = [x for x in m.split(',') if x]
    held = {t for v in base.values() for t in v}
    cand = {}
    st = Counter()
    for i, r in enumerate(refs):
        rn, ra, c = text[r]
        if c in LABELLED:
            continue
        acc = set(base[r])
        for k in range(10):
            t = T[i, k]
            if t in acc or t in held:
                continue
            ty = rtype(rn, ra, text[t][0], text[t][1], vocab[c])
            if ty not in TYPES:
                continue
            if ty == 'addr_other':
                # hand-check: decoys swap in a real business-type word (Club -> Comite); copies only add garbled or
                # invented words. Keep only records whose extra words never occur in a Source-1 name of the country.
                extra = set(words(text[t][0])) - set(words(rn))
                if any(w in vocab[c] for w in extra):
                    st['addr_other dropped: real extra word'] += 1
                    continue
            st[f'{ty} rejected & free'] += 1
            if p6[i, k] >= TAU:
                st[f'{ty} T6>={TAU}'] += 1
                if t not in cand or p6[i, k] > cand[t][1]:
                    cand[t] = (r, float(p6[i, k]), float(pF[i, k]), ty, i, k)
    add = defaultdict(list)
    for t, (r, q, pf, ty, i, k) in cand.items():
        add[r].append(t)
    with open(out / 'matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for r in rows:
            f.write(r + '\t' + ','.join(base[r] + add.get(r, [])) + '\n')
    added = [(r, t, q, pf, ty) for t, (r, q, pf, ty, i, k) in cand.items()]
    by = Counter(ty for *_, ty in added)
    # expected effect per added link, if the rest of the reference is right: gain when the link is a true copy the
    # reference was missing (k accepted of k+1 true), loss when it is false (k of k true)
    gain, loss = Counter(), Counter()
    for r, t, q, pf, ty in added:
        k = len(base[r]) + len(add[r]) - 1
        gain[ty] += 1 - f05(k, k, k + 1)
        loss[ty] += 1 - f05(k, k + 1, k)
    nfr = sum(1 for r in rows if text[r][2] not in LABELLED)
    R = {'tau': TAU, 'counts': dict(st), 'added': len(added), 'added_by_type': dict(by),
         'added_per_ref': round(len(added) / nfr, 5), 'refs_changed': len(add),
         'refs_that_were_empty': sum(1 for r in add if not base[r]),
         'sum_gain_if_true_by_type': {k: round(v, 1) for k, v in gain.items()},
         'sum_loss_if_false_by_type': {k: round(v, 1) for k, v in loss.items()},
         'n_country_refs': nfr}
    (out / 'fix_report.json').write_text(json.dumps(R, indent=1))
    rng = np.random.default_rng(11)
    ia = [j for j, a in enumerate(added) if a[4] == 'addr_other']
    ib = [j for j, a in enumerate(added) if a[4] != 'addr_other']
    pick = list(rng.choice(ia, size=min(45, len(ia)), replace=False)) + list(rng.choice(ib, size=min(15, len(ib)), replace=False))
    with open(out / 'handcheck_sample.txt', 'w', encoding='utf-8') as f:
        for j in pick:
            r, t, q, pf, ty = added[j]
            f.write(f'--- {ty} p6={q:.3f} pFR={pf:.3f} | we already accept {len(base[r])}\n')
            f.write(f'  REF: {text[r][0]} | {text[r][1]}\n  ADD: {text[t][0]} | {text[t][1]}\n')
            for x in base[r][:6]:
                f.write(f'    has: {text[x][0]} | {text[x][1]}\n')
    print(json.dumps(R, indent=1), flush=True)


if __name__ == '__main__':
    main()
