"""Blank-address misses under stage-3b: do they sit where the deciders disagree?

Deciders per pair: v5 probability, cross-encoder logit, S5 L2 probability, stage-3b probability (all out-of-sample on
holdout). On holdout blank-address rows (non-blank for contrast): v5 x cross-encoder agreement grid with positive rate,
stage-3b mean probability, misses (FN) and false matches (FP); FN/TN/TP profiles on the context features (rival rank
and gap, name multiplicity, exact name equality with the reference or an accepted sibling, decoy words); and the
label-ceiling gain per disagreement cell (decide that cell by label, everything else as stage-3b).
Needs boundary_exp2.py outputs (exp2_cache.npz, stage3b_prob.npy, stage3b_meta.json). Writes fusion/blank_disagree.json.
"""
import gzip
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from boundary_exp import CE, FUS, NAMES, OUT, PRUNE, RAW, V4
from boundary_exp2 import NUM, REV
from distractor_check import decide, f05, partition_of

DECOY = {'group', 'groupe', 'holdings', 'holding', 'enterprises', 'entreprises', 'industries', 'ventures', 'global',
         'associates', 'trading', 'solutions', 'international', 'technologies', 'management', 'gestion', 'systems',
         'systemes', 'consulting', 'conseil', 'sons', 'fils', 'development', 'developpement', 'distribution',
         'participations', 'centre', 'usa', 'america', 'of'}


def toks(s):
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r'[^\w]+', ' ', s).split()


def main():
    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    cp = partition_of(gold)
    mask = np.load(PRUNE / 'mask_top10.npy')
    refs, tgts = [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t')
                refs.append(r); tgts.append(t)
    refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
    part = np.array([cp[r] for r in refs], dtype=object)
    S = np.load(FUS / 'stage1.npz')
    y = np.load(V4 / 'pairs/y.npy')[S['rows']] > 0
    v5 = np.where(S['stage1'] < 0.02, 0.0, S['stage1']).astype(np.float64)
    ce = np.nan_to_num(np.load(CE).astype(np.float64), nan=0.0)
    C = np.load(OUT / 'exp2_cache.npz')
    F, Frev, P, blank = C['F'], C['Frev'], C['P'], C['blank']
    P3 = np.load(OUT / 'stage3b_prob.npy')
    rule = json.load(open(OUT / 'stage3b_meta.json'))['rule']
    fi = {n: i for i, n in enumerate(NAMES)}
    ri = {n: i for i, n in enumerate(REV)}
    ho = np.flatnonzero(part == 'holdout')
    H = sorted(s for s in gold if cp[s] == 'holdout')
    pos = {s: i for i, s in enumerate(H)}
    rix = np.array([pos[r] for r in refs[ho]])
    ch = np.zeros(len(y), dtype=bool)
    ch[ho] = decide(rix, P3[ho], rule[0], rule[1], len(H))
    s5 = np.zeros(len(y), dtype=bool)
    s5[ho] = P[ho] >= 0.68

    need = set(refs[ho]) | set(tgts[ho])
    name = {}
    for k in (1, 2, 3):
        with open(RAW / f'train_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname = head.index('entity_id'), head.index('business_name')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if q[ie] in need:
                    name[q[ie]] = q[iname]
    by_ref = defaultdict(list)
    for j in ho:
        by_ref[refs[j]].append(j)
    ex_ref = np.zeros(len(y), dtype=bool); ex_sib = np.zeros(len(y), dtype=bool); decoy = np.zeros(len(y), dtype=bool)
    for r, js in by_ref.items():
        rt = set(toks(name.get(r, '')))
        acc = [j for j in js if ch[j] and not blank[j]]
        for j in js:
            tt = set(toks(name.get(tgts[j], '')))
            ex_ref[j] = bool(tt) and tt == rt
            ex_sib[j] = any(set(toks(name.get(tgts[q], ''))) == tt for q in acc if q != j) and bool(tt)
            decoy[j] = bool((tt - rt) & DECOY)

    def vbin(v):
        return np.select([v < 0.1, v < 0.5, v < 0.9], ['v5<0.1', 'v5 0.1-0.5', 'v5 0.5-0.9'], 'v5>=0.9')

    def cbin(c):
        return np.select([c < -2, c < 0, c < 2], ['ce<-2', 'ce -2..0', 'ce 0..2'], 'ce>=2')

    def macro(sel):
        return float(np.mean([f05(gold[s], {tgts[j] for j in by_ref.get(s, []) if sel[j]}) for s in H]))

    base = macro(ch)
    R = {'rule': rule, 'stage3b_holdout': round(base, 6)}
    vb, cb = vbin(v5), cbin(ce)
    for seg, m in (('blank', blank), ('nonblank', ~blank)):
        grid = {}
        for a in ('v5<0.1', 'v5 0.1-0.5', 'v5 0.5-0.9', 'v5>=0.9'):
            for b in ('ce<-2', 'ce -2..0', 'ce 0..2', 'ce>=2'):
                js = ho[m[ho] & (vb[ho] == a) & (cb[ho] == b)]
                if not len(js):
                    continue
                fix = ch.copy(); fix[js] = y[js]
                grid[f'{a} | {b}'] = {'n': int(len(js)), 'pos_rate': round(float(y[js].mean()), 4),
                                      'mean_p3': round(float(P3[js].mean()), 4), 'fn': int((y[js] & ~ch[js]).sum()),
                                      'fp': int((~y[js] & ch[js]).sum()), 's5_fn': int((y[js] & ~s5[js]).sum()),
                                      'label_ceiling_gain': round(macro(fix) - base, 6)}
        R[f'{seg}_grid'] = grid
    hb = ho[blank[ho]]
    groups = {'FN': hb[y[hb] & ~ch[hb]], 'TP': hb[y[hb] & ch[hb]], 'TN_band': hb[~y[hb] & ~ch[hb] & (P3[hb] >= 0.1)],
              'FP': hb[~y[hb] & ch[hb]]}
    prof = {}
    for g, js in groups.items():
        if not len(js):
            continue
        prof[g] = {'n': int(len(js)), 'p3_median': round(float(np.median(P3[js])), 3),
                   'v5_median': round(float(np.median(v5[js])), 3), 'ce_median': round(float(np.median(ce[js])), 2),
                   'rev_name_rank1': round(float((Frev[js, ri['rn_rank']] == 1).mean()), 3),
                   'rev_name_gap_median': round(float(np.median(Frev[js, ri['rn_gap']])), 4),
                   'rev_name_close_median': float(np.median(Frev[js, ri['rn_close']])),
                   'name_unique_s1': round(float((F[js, fi['mult_s1_tgt']] <= np.log1p(1)).mean()), 3),
                   'exact_name_ref': round(float(ex_ref[js].mean()), 3), 'exact_name_sibling': round(float(ex_sib[js].mean()), 3),
                   'decoy_word': round(float(decoy[js].mean()), 3), 'rank_median': float(np.median(F[js, fi['rank']])),
                   'n_acc_same_src0': round(float((F[js, fi['n_acc_same_src']] == 0).mean()), 3)}
    R['blank_profiles'] = prof
    (FUS / 'blank_disagree.json').write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
