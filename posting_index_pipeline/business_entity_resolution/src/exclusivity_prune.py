"""Drop France links whose record is claimed by several S1 entities at once, when nothing breaks the tie.

A record belongs to at most one S1, so its stage-3b probabilities over S1 lists should sum to at most 1. France is
over-committed 12x more often than US/India (overcommit_check.py), mostly templated names ('Tourcoing Club SAS',
'Nantes Theatre SARL') held by several S1 entities, where a blank-address copy is a coin flip. An accepted France link
is removed when: sum S of p over S1s for its record > 1.2, the record's address is blank or not on the S1's street at the
S1's number (a missing number on the right street still breaks the tie), p / S < 0.77 (the F0.5 break-even for an extra link), and the S1 keeps at
least one other link. Remove-only (no re-decision), so no new links appear. Env LATE, SUB_DIR (rule folder parent of
the pre-census submission), OUT_DIR, COUNTRIES (default France).
"""
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault('OUT_DIR', os.environ.get('OUT_DIR', '.'))
from fr_labels import read_france
from france_census import house, load, street_words, tokens_addr


def main():
    Z = np.load(os.environ['LATE'], allow_pickle=True)
    ri, tg, p = Z['ref_idx'], Z['target'].astype(str), Z['prob'].astype(np.float64)
    _, inv = np.unique(tg, return_inverse=True)
    S = np.bincount(inv, weights=p)[inv]
    sub = Path(os.environ['SUB_DIR'])
    rd = next(d for d in sub.iterdir() if (d / 'matching_results.tsv').exists())
    match = load(rd / 'matching_results.tsv')
    refs, text, m, _ = read_france(ri, tg)
    R = Counter(); drop = set()
    for j in np.flatnonzero(m & (S > 1.2)):
        r, t = refs[ri[j]], tg[j]
        acc = match.get(r, [])
        if t not in acc:
            continue
        R['accepted_overcommitted'] += 1
        ra, ta = text[r][1], text.get(t, ('', ''))[1]
        if ta.strip():
            h1, h2 = house(ra)[0], house(ta)[0]
            sw = street_words(ra)
            if sw and sw <= tokens_addr(ta) and (not h2 or h1 == h2):
                R['kept_same_street_number_equal_or_missing'] += 1
                continue
        if p[j] / S[j] >= 0.77:
            R['kept_above_breakeven'] += 1
            continue
        if len(acc) < 2:
            R['kept_only_link'] += 1
            continue
        drop.add((r, t)); R['dropped_blank' if not ta.strip() else 'dropped_other_address'] += 1
    dest = Path(os.environ['OUT_DIR']) / rd.name
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / 'matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for r in refs:
            f.write(r + '\t' + ','.join(t for t in match.get(r, []) if (r, t) not in drop) + '\n')
    shutil.copy(rd / 'candidate_pairs.tsv', dest / 'candidate_pairs.tsv')
    link = Path(os.environ['OUT_DIR']) / 'late_scores.npz'
    if not link.exists():
        os.symlink(Path(os.environ['LATE']).resolve(), link)
    (Path(os.environ['OUT_DIR']) / 'prune_report.json').write_text(json.dumps(dict(R), indent=1))
    print(json.dumps(dict(R), indent=1), flush=True)


if __name__ == '__main__':
    main()
