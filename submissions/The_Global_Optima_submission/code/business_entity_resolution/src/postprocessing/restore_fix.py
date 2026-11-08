"""U7 restore rule: re-add same-address copies that the old leaderboard file accepted and the France model removed.

Hand-check of 70 French references changed between the 0.989635 upload (H2v2 = T5 + census v2) and U6: 18 of the 23
wrong changes were removals of copies at the business's own address (aliases, typo'd names, filler swaps) that T5 had
accepted with p >= 0.85; every correct removal was a shifted-number decoy, a type-word swap or an exclusivity coin flip.
Restore (countries other than US/India) a link in OLD but not in NEW when:
  same first house number as the reference (both present, record address not blank), T5 p >= 0.85,
  no other reference holds the record in NEW, and every word the record adds to the name is either never used in a
  Source-1 name of the country (garbled / invented) or a filler word (FILLER); one reference per record (highest T5).
Usage: restore_fix.py <old tsv> <new tsv> <out dir>
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
ART = ME / 'artifacts'
LABELLED = {'US', 'India'}
P_MIN = 0.85
FILLER = set('services service cie compagnie france groupe fils associes developpement trust foundation enterprises'.split())
sys.path.insert(0, str(ME / 'business_entity_resolution/src'))
from france_loss import house, read, words  # noqa: E402


def f05(tp, pred, gold):
    if not gold and not pred:
        return 1.0
    if tp == 0:
        return 0.0
    p, r = tp / pred, tp / gold
    return 1.25 * p * r / (0.25 * p + r)


def load(p):
    rows, d = [], {}
    with open(p, encoding='utf-8') as f:
        next(f)
        for l in f:
            r, m = l.rstrip('\n').split('\t'); rows.append(r); d[r] = [x for x in m.split(',') if x]
    return rows, d


def main():
    _, old = load(sys.argv[1])
    rows, new = load(sys.argv[2])
    out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    text, vocab = read('test')
    refs, tgts = [], []
    with open(ART / 'submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
        next(f)
        for line in f:
            s, c = line.rstrip('\n').split('\t')
            refs.append(s); tgts.extend(c.split(','))
    n = len(refs)
    T = np.array(tgts, dtype=object).reshape(n, 10)
    p5 = np.load(ART / 'submission_T5_own/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)
    pF = np.load(ART / 'submission_FRtrained_w4/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)
    pos = {r: i for i, r in enumerate(refs)}
    held = defaultdict(set)
    for r, v in new.items():
        for t in v:
            held[t].add(r)
    st, cand = Counter(), {}
    for r in rows:
        rn, ra, c = text[r]
        if c in LABELLED:
            continue
        dropped = set(old[r]) - set(new[r])
        if not dropped:
            continue
        i = pos[r]
        for k in range(10):
            t = T[i, k]
            if t not in dropped:
                continue
            st['dropped (old had, new lacks)'] += 1
            tn, ta, _ = text[t]
            h1, h2 = house(ra), house(ta)
            if not (ta or '').strip() or h1 is None or h1 != h2:
                st['skip: blank / no number / different number'] += 1; continue
            if p5[i, k] < P_MIN:
                st['skip: T5 < %.2f' % P_MIN] += 1; continue
            if held[t] - {r}:
                st['skip: held by another reference'] += 1; continue
            extra = set(words(tn)) - set(words(rn))
            if any(w in vocab[c] and w not in FILLER for w in extra):
                st['skip: adds a real business word'] += 1; continue
            st['restore candidate'] += 1
            if t not in cand or p5[i, k] > cand[t][1]:
                cand[t] = (r, float(p5[i, k]), float(pF[i, k]))
    add = defaultdict(list)
    for t, (r, q, pf) in cand.items():
        add[r].append(t)
    with open(out / 'matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for r in rows:
            f.write(r + '\t' + ','.join(new[r] + add.get(r, [])) + '\n')
    gain = loss = 0.0
    for r, ts in add.items():
        for _ in ts:
            k = len(new[r]) + len(ts) - 1
            gain += 1 - f05(k, k, k + 1); loss += 1 - f05(k, k + 1, k)
    nfr = sum(1 for r in rows if text[r][2] not in LABELLED)
    R = {'counts': dict(st), 'restored': len(cand), 'refs_changed': len(add),
         'sum_gain_if_true': round(gain, 1), 'sum_loss_if_false': round(loss, 1),
         'breakeven_precision': round(loss / (gain + loss), 3) if gain + loss else None, 'n_country_refs': nfr}
    (out / 'restore_report.json').write_text(json.dumps(R, indent=1))
    items = list(cand.items())
    rng = np.random.default_rng(5)
    with open(out / 'handcheck_sample.txt', 'w', encoding='utf-8') as f:
        for j in rng.choice(len(items), size=min(60, len(items)), replace=False):
            t, (r, q, pf) = items[j]
            f.write(f'--- pT5={q:.2f} pFR={pf:.2f} | U6 accepts {len(new[r])}\n  REF: {text[r][0]} | {text[r][1]}\n'
                    f'  ADD: {text[t][0]} | {text[t][1]}\n')
            for x in new[r][:6]:
                f.write(f'    has: {text[x][0]} | {text[x][1]}\n')
    print(json.dumps(R, indent=1), flush=True)


if __name__ == '__main__':
    main()
