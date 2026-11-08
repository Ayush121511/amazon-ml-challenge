"""Where France loses: record types from the manual audit, France vs US/India, test vs labelled holdout.

Each candidate (reference, record) gets one type:
  held_other      record accepted for another reference
  initials_web    record name = initials / web form / compact form of the reference name
  blank_same | blank_other          record address blank; same name words or not
  same_*  (same first house number and near-identical address):
     same_name | rec_adds (legal/filler added) | rec_drops | overlap (type-word swap) |
     alias_invented (no shared word; record words never used in any Source-1 name of the country) | alias_real
  diff_same_shift (same name, both numbers, |diff| <= 13) | diff_same_far (|diff| > 13) | diff_other
  nonum_same | nonum_other           a house number missing on one side
  addr_other      same number but a different address
The generator's copy recipes are country-independent (train US and India: identical link counts and histograms), so
true links per reference of each type should be about the same in France as in the US/India holdout.
Output per type: holdout true/ref, T6 accepted/ref and precision; test accepted/ref per country (submission),
France expected misses/ref (sum of p over rejected) and expected false links/ref (sum of 1-p over accepted).
Usage: france_loss.py <matching_results.tsv>
"""
import csv
import gzip
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
AY = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
ART = ME / 'artifacts'
OUT = Path('/scratch/scai/mtech/aib262467/france_loss')
sys.path[:0] = [str(AY / 'business_entity_resolution/src')]
csv.field_size_limit(10 ** 9)
STOP = set('sarl sas sasu eurl sa sci snc selarl scop inc llc ltd limited corp corporation co company pvt private llp '
           'plc lp pllc pc de du des la le les l d et and the of en au aux'.split())
NUM = re.compile(r'\d+')


def fold(s):
    s = unicodedata.normalize('NFKD', (s or '').lower())
    return ''.join(c for c in s if not unicodedata.combining(c))


def words(n):
    return [w for w in re.findall(r'[a-z0-9]+', fold(n)) if w not in STOP and not w.isdigit()]


def house(a):
    m = NUM.search(a or '')
    return int(m.group(0)) if m else None


def rtype(rn, ra, tn, ta, vocab):
    A, B = words(rn), words(tn)
    sa, sb = set(A), set(B)
    ca, cb = ''.join(A), ''.join(B)
    ini = ''.join(w[0] for w in A)
    if cb and (cb == ini or (len(cb) >= 4 and (ca in cb or cb in ca))) and sa != sb:
        return 'initials_web'
    same = bool(sa) and sa == sb
    if not (ta or '').strip():
        return 'blank_same' if same else 'blank_other'
    h1, h2 = house(ra), house(ta)
    if h1 is None or h2 is None:
        return 'nonum_same' if same else 'nonum_other'
    if h1 != h2:
        if same:
            return 'diff_same_shift' if abs(h1 - h2) <= 13 else 'diff_same_far'
        return 'diff_other'
    if fuzz.token_set_ratio(fold(ra), fold(ta)) < 85:
        return 'addr_other'
    if same:
        return 'same_name'
    if sa and sa < sb:
        return 'same_rec_adds'
    if sb and sb < sa:
        return 'same_rec_drops'
    if sa & sb:
        return 'same_overlap'
    if sb and all(len(w) >= 4 and w not in vocab for w in sb):
        return 'same_alias_invented'
    return 'same_alias_real'


def read(split, keep=None):
    text, vocab = {}, defaultdict(set)
    for k in (1, 2, 3):
        with open(ME / f'data/dataset/{split}/{split}_source{k}.tsv', encoding='utf-8', newline='') as fh:
            for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                e = row['entity_id']
                if k == 1:
                    vocab[row['country']].update(words(row['business_name']))
                if keep is None or e in keep:
                    text[e] = (row['business_name'], row['business_address'], row['country'])
    return text, vocab


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    text, vocab = read('test')
    refs, tgts = [], []
    with open(ART / 'submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
        next(f)
        for line in f:
            s, c = line.rstrip('\n').split('\t')
            refs.append(s); tgts.extend(c.split(','))
    n = len(refs)
    T = np.array(tgts, dtype=object).reshape(n, 10)
    pF = np.load(ART / 'submission_FRtrained_w4/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)
    pB = np.load('/scratch/scai/mtech/aib262467/t6bag/test/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)
    p6 = np.load(ART / 'submission_T6_own/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)
    sub = {}
    with open(sys.argv[1], encoding='utf-8') as f:
        next(f)
        for l in f:
            r, m = l.rstrip('\n').split('\t'); sub[r] = set(x for x in m.split(',') if x)
    held = defaultdict(set)
    for r, v in sub.items():
        for t in v:
            held[t].add(r)
    rng = random.Random(0)
    st = defaultdict(Counter)
    nref = Counter()
    fr_rows = []
    for i, r in enumerate(refs):
        rn, ra, c = text[r]
        if c != 'France' and rng.random() > 0.25:
            continue
        nref[c] += 1
        for k in range(10):
            t = T[i, k]
            tn, ta, _ = text[t]
            a = t in sub[r]
            ty = 'held_other' if (not a and held[t] - {r}) else rtype(rn, ra, tn, ta, vocab[c])
            p = pF[i, k] if c == 'France' else pB[i, k]
            s = st[(c, ty)]
            s['n'] += 1; s['acc'] += a; s['exp_fn'] += float((not a) * p); s['exp_fp'] += float(a * (1 - p))
            s['p6_acc_if_free'] += (not a) and ty != 'held_other' and p6[i, k] >= 0.74
            if c == 'France':
                fr_rows.append((i, k, ty, int(a), float(pF[i, k]), float(p6[i, k])))
    test = {f'{c}|{ty}': {kk: v / nref[c] for kk, v in s.items()} for (c, ty), s in st.items()}
    json.dump([(i, k, ty, a, pf, q) for i, k, ty, a, pf, q in fr_rows], open(OUT / 'france_rows.json', 'w'))
    del text
    # ---- holdout
    from distractor_check import decide, partition_of
    gold = {k: set(v) for k, v in json.loads((ART / 'matcher_v4/pairs/gold.json').read_text()).items()}
    cp = partition_of(gold)
    m10 = np.load(ART / 'prune_v1/mask_top10.npy').astype(bool)
    pr, pt = [], []
    with gzip.open(ART / 'matcher_v4/pairs/pairs.tsv.gz', 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if m10[i]:
                a, b = line.rstrip('\n').split('\t'); pr.append(a); pt.append(b)
    pr, pt = np.array(pr, dtype=object), np.array(pt, dtype=object)
    S1 = np.load(ART / 'fusion/stage1.npz')
    y = np.load(ART / 'matcher_v4/pairs/y.npy')[S1['rows']] > 0
    ph = np.load(ART / 'fusion/boundary_models_t6/stage3b_prob.npy')
    rule = json.load(open(ART / 'fusion/boundary_models_t6/stage3b_meta.json'))['rule']
    prefs = sorted(s for s in gold if cp[s] == 'holdout')
    pos = {s: i for i, s in enumerate(prefs)}
    idx = np.flatnonzero(np.array([cp[r] == 'holdout' for r in pr]))
    ch = np.zeros(len(pr), bool)
    ch[idx] = decide(np.array([pos[r] for r in pr[idx]]), ph[idx], rule[0], rule[1], len(prefs))
    heldh = defaultdict(set)
    for j in idx[ch[idx]]:
        heldh[pt[j]].add(pr[j])
    stext, svocab = read('train', set(pr[idx]) | set(pt[idx]))
    hs = defaultdict(Counter)
    nh = Counter(stext[s][2] for s in prefs)
    for j in idx:
        rn, ra, c = stext[pr[j]]
        tn, ta, _ = stext[pt[j]]
        ty = 'held_other' if (not ch[j] and heldh[pt[j]] - {pr[j]}) else rtype(rn, ra, tn, ta, svocab[c])
        h = hs[(c, ty)]
        h['n'] += 1; h['true'] += int(y[j]); h['acc'] += int(ch[j]); h['tp'] += int(ch[j] and y[j])
        h['fn'] += int(y[j] and not ch[j])
    hold = {f'{c}|{ty}': {kk: v / nh[c] for kk, v in h.items()} for (c, ty), h in hs.items()}
    types = sorted({k.split('|')[1] for k in list(test) + list(hold)})
    rows = []
    for ty in types:
        H = {c: hold.get(f'{c}|{ty}', {}) for c in ('US', 'India')}
        Tt = {c: test.get(f'{c}|{ty}', {}) for c in ('US', 'India', 'France')}
        true_ui = (H['US'].get('true', 0) + H['India'].get('true', 0)) / 2
        rows.append({'type': ty,
                     'hold_true_per_ref_US/IN': [round(H['US'].get('true', 0), 4), round(H['India'].get('true', 0), 4)],
                     'hold_T6_acc_per_ref_US/IN': [round(H['US'].get('acc', 0), 4), round(H['India'].get('acc', 0), 4)],
                     'hold_T6_prec': round((H['US'].get('tp', 0) + H['India'].get('tp', 0)) /
                                           max(1e-9, H['US'].get('acc', 0) + H['India'].get('acc', 0)), 3),
                     'hold_missed_per_ref': round((H['US'].get('fn', 0) + H['India'].get('fn', 0)) / 2, 4),
                     'test_acc_per_ref_US/IN/FR': [round(Tt[c].get('acc', 0), 4) for c in ('US', 'India', 'France')],
                     'test_cand_per_ref_US/IN/FR': [round(Tt[c].get('n', 0), 4) for c in ('US', 'India', 'France')],
                     'FR_acc_minus_USIN_true': round(Tt['France'].get('acc', 0) - true_ui, 4),
                     'FR_exp_miss_per_ref': round(Tt['France'].get('exp_fn', 0), 4),
                     'FR_exp_false_per_ref': round(Tt['France'].get('exp_fp', 0), 4),
                     'FR_T6_would_add_per_ref': round(Tt['France'].get('p6_acc_if_free', 0), 4)})
    rows.sort(key=lambda d: d['FR_acc_minus_USIN_true'])
    (OUT / 'report.json').write_text(json.dumps(rows, indent=1, default=float))
    hdr = f"{'type':22s} {'holdTrue US/IN':>15s} {'holdAcc US/IN':>15s} {'prec':>5s} {'testAcc US/IN/FR':>24s} {'FR-USINtrue':>11s} {'FRexpMiss':>9s} {'FRexpFalse':>10s} {'T6add':>6s}"
    print(hdr)
    for d in rows:
        print(f"{d['type']:22s} {d['hold_true_per_ref_US/IN'][0]:.4f}/{d['hold_true_per_ref_US/IN'][1]:.4f} "
              f"{d['hold_T6_acc_per_ref_US/IN'][0]:.4f}/{d['hold_T6_acc_per_ref_US/IN'][1]:.4f} {d['hold_T6_prec']:5.3f} "
              f"{d['test_acc_per_ref_US/IN/FR'][0]:.4f}/{d['test_acc_per_ref_US/IN/FR'][1]:.4f}/{d['test_acc_per_ref_US/IN/FR'][2]:.4f} "
              f"{d['FR_acc_minus_USIN_true']:+11.4f} {d['FR_exp_miss_per_ref']:9.4f} {d['FR_exp_false_per_ref']:10.4f} {d['FR_T6_would_add_per_ref']:6.4f}",
              flush=True)


if __name__ == '__main__':
    main()
