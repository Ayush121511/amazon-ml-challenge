"""Gen-4 EDA, test side (label-free): where does T2 (public 0.988085) differ from S5 (0.983415), and what is still
uncertain in France after the census fix?

  - T2 vs S5: references whose match set differs, counts by country and change type, 30 sampled per country with names
    and addresses (removed / added links, T1b stage-3b probability of each).
  - France after census: France pairs with T1b probability 0.2-0.8, the final T2 decision, and the kind of edit between
    reference and candidate (legal form only / filler word / type word / affiliate word / house number differs / other).
Writes OUT/test_side.json, OUT/cases_test_T2_vs_S5_<country>.tsv, OUT/cases_test_france_uncertain.tsv.
"""
import csv
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

A = Path('/scratch/scai/mtech/aib262015/amazon_ml_2026')
ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
OUT = Path(os.environ.get('OUT', str(ME / 'artifacts/gen4_eda')))
RAW = ME / 'data/dataset/test'
S5 = A / 'artifacts/submission_S5/threshold_0.68/matching_results.tsv'
T2 = ME / 'artifacts/submission_T2_census_ayush/expected_0.6/matching_results.tsv'
T1B = A / 'artifacts/submission_T1b_s3b_c2w1/late_scores.npz'
LEGAL = set('sarl sas sasu eurl sci snc sa ei scop selarl inc llc ltd pvt private limited co corp'.split())
FILLER = set('fils associes cie services groupe developpement france dba aka formerly com compagnie'.split())
TYPES = set('club ecole amicale comite maison centre union sportive college amis primaire federation pharmacie sante '
            'parents lycee maternelle association fetes institut anciens ehpad loisirs culturelle clinique jeunes sport '
            'groupement medical foyer culture section collectif team danse musique theatre cercle atelier cabinet espace'.split())
AFFIL = set('group holdings holding international internationale development sons enterprises entreprises industries '
            'ventures global associates solutions technologies management consulting distribution participations'.split())


def fold(s):
    s = unicodedata.normalize('NFKD', s or '')
    return re.sub(r'[^\w]+', ' ', ''.join(c for c in s if not unicodedata.combining(c)).lower()).split()


def first_num(a):
    m = re.search(r'\d+', a or '')
    return str(int(m.group(0))) if m else ''


def load(p):
    with open(p, encoding='utf-8') as f:
        next(f)
        return {l.split('\t')[0]: [x for x in l.rstrip('\n').split('\t')[1].split(',') if x] for l in f}


def kind(rn, ra, tn, ta):
    a, b = set(fold(rn)), set(fold(tn))
    added, dropped = b - a, a - b
    diff = added | dropped
    n1, n2 = first_num(ra), first_num(ta)
    tag = []
    if n1 and n2 and n1 != n2:
        tag.append('house_number_differs')
    if not diff:
        tag.append('same_words')
    elif diff <= LEGAL:
        tag.append('legal_form_only')
    elif added & TYPES and not added & FILLER:
        tag.append('type_word_added')
    elif added & FILLER:
        tag.append('filler_word_added')
    elif diff & AFFIL:
        tag.append('affiliate_word')
    else:
        tag.append('other_name_edit')
    if not (ta or '').strip():
        tag.append('blank_address')
    return '+'.join(tag)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    refs, cty, text = [], {}, {}
    for src in ('test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv'):
        with open(RAW / src, encoding='utf-8', newline='') as fh:
            for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                text[row['entity_id']] = (row['business_name'] or '', row['business_address'] or '')
                if src == 'test_source1.tsv':
                    refs.append(row['entity_id']); cty[row['entity_id']] = row['country']
    s5, t2 = load(S5), load(T2)
    Z = np.load(T1B, allow_pickle=True)
    ri, tg, pr = Z['ref_idx'], Z['target'].astype(str), Z['prob'].astype(np.float64)
    prob = {(refs[i], t): float(p) for i, t, p in zip(ri, tg, pr)}
    R = {'T2_vs_S5': {}}
    rng = np.random.default_rng(0)
    for c in ('France', 'India', 'US'):
        rs = [r for r in refs if cty[r] == c]
        diff = [r for r in rs if set(s5[r]) != set(t2[r])]
        ch = Counter()
        rem_k, add_k = Counter(), Counter()
        for r in diff:
            a, b = set(s5[r]), set(t2[r])
            ch['emptied' if a and not b else 'filled' if b and not a else 'removed_only' if b < a else 'added_only' if a < b else 'swapped'] += 1
            for t in a - b:
                rem_k[kind(*text[r], *text.get(t, ('', '')))] += 1
            for t in b - a:
                add_k[kind(*text[r], *text.get(t, ('', '')))] += 1
        R['T2_vs_S5'][c] = {'refs': len(rs), 'refs_changed': len(diff), 'change_type': dict(ch),
                            'removed_link_kinds': dict(rem_k.most_common(10)), 'added_link_kinds': dict(add_k.most_common(10))}
        with open(OUT / f'cases_test_T2_vs_S5_{c}.tsv', 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(['ref', 'ref_name', 'ref_addr', 'change', 'target', 'tgt_name', 'tgt_addr', 'p_T1b', 'kind'])
            for r in [diff[i] for i in rng.permutation(len(diff))[:30]]:
                a, b = set(s5[r]), set(t2[r])
                for t, lab in [(t, 'removed_by_T2') for t in sorted(a - b)] + [(t, 'added_by_T2') for t in sorted(b - a)] + \
                              [(t, 'kept') for t in sorted(a & b)]:
                    w.writerow([r, *text[r], lab, t, *text.get(t, ('', '')), round(prob.get((r, t), float('nan')), 3),
                                kind(*text[r], *text.get(t, ('', '')))])
    # France uncertainty after census
    fr = np.array([cty[refs[i]] == 'France' for i in ri])
    unc = np.flatnonzero(fr & (pr >= 0.2) & (pr <= 0.8))
    kk, dec = Counter(), Counter()
    rows = []
    for j in unc:
        r, t = refs[ri[j]], tg[j]
        k = kind(*text[r], *text.get(t, ('', '')))
        chosen = t in set(t2[r])
        kk[k] += 1; dec[(k, chosen)] += 1
        rows.append((r, t, k, chosen, pr[j]))
    R['france_uncertain_0.2_0.8'] = {'pairs': int(len(unc)), 'of_france_pairs': int(fr.sum()),
                                     'by_kind': {k: [v, dec[(k, True)]] for k, v in kk.most_common(15)},
                                     'note': 'by_kind = [pairs, of which chosen in T2]'}
    with open(OUT / 'cases_test_france_uncertain.tsv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['ref', 'ref_name', 'ref_addr', 'target', 'tgt_name', 'tgt_addr', 'p_T1b', 'chosen_T2', 'kind'])
        by = defaultdict(list)
        for x in rows:
            by[x[2]].append(x)
        for k, xs in sorted(by.items(), key=lambda kv: -len(kv[1]))[:8]:
            for r, t, k_, c_, p in [xs[i] for i in rng.permutation(len(xs))[:6]]:
                w.writerow([r, *text[r], t, *text.get(t, ('', '')), round(p, 3), int(c_), k_])
    (OUT / 'test_side.json').write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
