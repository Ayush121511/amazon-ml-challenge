"""France pseudo-labels for training a France-aware stage-3b: one label per test candidate pair, only where sure.

Two label sources, census first (it encodes the French generator patterns the US/India-trained stage-3b cannot see):
  census   pairs at the S1's own house number (france_census.py tests, applied whether or not the pair is chosen)
             positive  strict same-address copy with only filler words added, or the identical name
             negative  a business-type word added and no filler word (different business in the same building)
           strict-test pairs that census itself would skip (record owned by another S1, another France S1 with the same
           core name, no decoys elsewhere for an edited name) stay unlabeled.
  extreme  every decider at an extreme value (france_pseudo.py): positive p3 >= 0.995, v5 >= 0.9, chunk2 >= 4 and in the
           final matches; negative p3 <= 0.001, v5 <= 0.02, chunk2 <= -4.
With CENSUS_V2=1 the census positives also take the v2 paths (street typo, type-word drop, initials). Round 2 of
self-training: point SUB_DIR at the France-trained run so the extreme labels come from its probabilities.
Census overrides extreme. A record labeled positive for two S1 entities is unlabeled for both. Uncertain pairs are
never labeled. Env SUB_DIR (late_scores.npz of the base run, same pair order as every stage-3b test run), FINAL_DIR
(folder whose rule subfolder holds the final matching_results.tsv), OUT. Writes OUT/fr_labels.npz (idx, target,
label, src) + fr_labels_stats.json.
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault('OUT_DIR', os.environ['OUT'])
from france_census import FILLER, RAW, TYPES, V2, core, forms, house, load, near, street_words, tokens_addr

SRC = {'census_pos': 1, 'census_neg': 2, 'extreme_pos': 3, 'extreme_neg': 4}


def read_france(ri, tg):
    """Test S1 ids, (name, address) of France S1s and their candidate records, France-pair mask, S1 core names."""
    refs, cty, text = [], {}, {}
    with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, iname, iaddr, ic = (head.index(c) for c in ('entity_id', 'business_name', 'business_address', 'country'))
        for line in fh:
            q = line.rstrip('\n').split('\t')
            refs.append(q[ie]); cty[q[ie]] = q[ic]
            if q[ic] == 'France':
                text[q[ie]] = (q[iname], q[iaddr])
    fr = np.array([cty[r] == 'France' for r in refs])
    m = fr[ri]
    need = set(tg[m])
    for k in (2, 3):
        with open(RAW / f'test_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if q[ie] in need:
                    text[q[ie]] = (q[iname], q[iaddr] if len(q) > iaddr else '')
    s1_core = {r: core(text[r][0]) for r, f in zip(refs, fr) if f}
    return refs, text, m, s1_core


def elsewhere_map(refs, ri, tg, m, text):
    """S1 has candidate records at another house number (its shifted decoy cluster), as in france_census.py."""
    e = defaultdict(bool)
    for j in np.flatnonzero(m):
        r = refs[ri[j]]
        h1, h2 = house(text[r][1])[0], house(text.get(tg[j], ('', ''))[1])[0]
        if h1 and h2 and h1 != h2:
            e[r] = True
    return e


def main():
    OUT = Path(os.environ['OUT']); OUT.mkdir(parents=True, exist_ok=True)
    Z = np.load(Path(os.environ['SUB_DIR']) / 'late_scores.npz', allow_pickle=True)
    ri, tg = Z['ref_idx'], Z['target'].astype(str)
    p3, v5, c2 = Z['prob'].astype(np.float64), Z['stage1'].astype(np.float64), np.nan_to_num(Z['ce'].astype(np.float64))
    fd = Path(os.environ['FINAL_DIR'])
    final = load(next(d for d in fd.iterdir() if (d / 'matching_results.tsv').exists()) / 'matching_results.tsv')
    refs, text, m, s1_core = read_france(ri, tg)
    freq = Counter(t for c in s1_core.values() for t in set(c))
    key_owner = Counter(' '.join(sorted(set(c))) for c in s1_core.values())
    owner = defaultdict(set)
    for r, ts in final.items():
        for t in ts:
            owner[t].add(r)
    elsewhere = elsewhere_map(refs, ri, tg, m, text)
    lab = np.full(len(tg), -1, np.int8)
    src = np.zeros(len(tg), np.int8)
    R = Counter()
    for j in np.flatnonzero(m):
        r, t = refs[ri[j]], tg[j]
        rn, ra = text[r]; tn, ta = text.get(t, ('', ''))
        if not ta.strip():
            continue
        (h1, s1n), (h2, s2n) = house(ra), house(ta)
        if not h1 or h1 != h2:
            continue
        rc, tc = s1_core[r], core(tn)
        added, dropped = set(tc) - set(rc), set(rc) - set(tc)
        if added & TYPES and not added & FILLER:
            lab[j], src[j] = 0, SRC['census_neg']
            continue
        if s1n and s2n and s1n != s2n:
            continue
        sw, ta_t = street_words(ra), tokens_addr(ta)
        if not sw or not rc or not tc:
            continue
        exact_st = sw <= ta_t
        if not exact_st and not (V2 and all(w in ta_t or near(w, ta_t) for w in sw)):
            continue
        initials = V2 and exact_st and len(tc) == 1 and len(rc) >= 2 and tc[0] == ''.join(w[0] for w in rc)
        if not initials:
            if rc[0] != tc[0] or not added <= FILLER:
                continue
            if min(rc, key=lambda x: (freq[x], x)) not in tc and not (V2 and exact_st and dropped and dropped <= TYPES):
                continue
        rf, tf = forms(rn), forms(tn)
        identical = not added and not dropped and not (rf and tf and rf != tf)
        if not identical and not elsewhere[r]:
            R['skip_no_decoys_elsewhere'] += 1
            continue
        if owner.get(t) and r not in owner[t]:
            R['skip_owned_by_other_s1'] += 1
            continue
        if added and key_owner[' '.join(sorted(set(tc)))] > 0:
            R['skip_other_s1_same_name'] += 1
            continue
        lab[j], src[j] = 1, SRC['census_pos']
    pos_refs = defaultdict(list)
    for j in np.flatnonzero(lab == 1):
        pos_refs[tg[j]].append(j)
    for t, js in pos_refs.items():
        if len(js) > 1:
            lab[js], src[js] = -1, 0
            R['unlabeled_census_pos_two_owners'] += len(js)
    in_final = np.zeros(len(tg), bool)
    for j in np.flatnonzero(m):
        in_final[j] = tg[j] in set(final.get(refs[ri[j]], ()))
    free = m & (lab == -1)
    ep = free & in_final & (p3 >= 0.995) & (v5 >= 0.9) & (c2 >= 4)
    en = free & (p3 <= 0.001) & (v5 <= 0.02) & (c2 <= -4)
    lab[ep], src[ep] = 1, SRC['extreme_pos']
    lab[en], src[en] = 0, SRC['extreme_neg']
    idx = np.flatnonzero(lab >= 0)
    np.savez(OUT / 'fr_labels.npz', idx=idx, target=tg[idx], label=lab[idx], src=src[idx])
    for k, v in SRC.items():
        s = src == v
        R[k] = int(s.sum())
        if s.any():
            R[k + '_p3_q10_50_90'] = [round(float(x), 3) for x in np.quantile(p3[s], [0.1, 0.5, 0.9])]
    R['census_pos_model_rejects_p3_lt_0.5'] = int(((src == 1) & (p3 < 0.5)).sum())
    R['census_neg_model_accepts_p3_ge_0.5'] = int(((src == 2) & (p3 >= 0.5)).sum())
    R.update({'france_pairs': int(m.sum()), 'labeled': int(len(idx)), 'labeled_share': round(len(idx) / int(m.sum()), 4),
              'france_refs_with_label': int(len(set(ri[idx])))})
    (OUT / 'fr_labels_stats.json').write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1), flush=True)


if __name__ == '__main__':
    main()
