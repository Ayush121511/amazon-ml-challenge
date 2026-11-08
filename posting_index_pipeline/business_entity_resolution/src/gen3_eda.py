"""Gen-3 matcher EDA: where does stage-3b (v5 + chunk2 + listwise, list/global-name/reverse/house-number features,
expected@0.60) still lose on holdout, compared with S5 (L2 v5 + mdeberta-full, threshold@0.68)?

Same loss split as gen2_eda.py (blocking / ranking / boundary: over, under, missed_all, singleton_fp), ceilings,
calibration, strata, FP typing, rank positions; plus for gen-3 errors: three-decider agreement (v5, chunk2, listwise),
label ceilings per blank x agreement cell, feature profiles of FN / FP against correct rows, and case files of every
error reference for manual review. Inputs are the saved out-of-sample probabilities, no training.
Env DATA (folder with the downloaded artifacts), TRAIN_RAW. Writes DATA/gen3_eda.json, DATA/gen3_cases.md.
"""
import gzip
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from distractor_check import decide, f05, partition_of

D = Path(os.environ['DATA'])
RAW = Path(os.environ['TRAIN_RAW'])
NAMES27 = ['p', 'logit_p', 's1f', 'ce', 'rank', 'p_prev', 'p_next', 'sum_p', 'n_acc', 'n_acc_same_src',
           'n_acc_other_src', 'is_s3', 'tgt_addr_empty', 'ref_addr_empty', 'name_sim', 'addr_sim', 'key_equal',
           'max_name_sim_acc', 'max_addr_sim_acc', 'name_sim_top1', 'n_chain_list', 'is_chain', 'mult_s1_ref',
           'mult_s1_tgt', 'mult_s23_tgt', 'n_same_key_list', 'decoy_word']
REV = ['rn_rank', 'rn_gap', 'rn_close', 'rn_cos', 'rf_rank', 'rf_gap', 'rf_close', 'rf_cos']
NUM = ['num_eq_ref', 'num_group', 'num_eq_acc', 'n_num_acc', 'tgt_has_num']


def q(a):
    a = np.asarray(a, dtype=np.float64)
    return {k: round(float(np.quantile(a, v)), 4) for k, v in (('p10', .1), ('p50', .5), ('p90', .9))} | {'n': int(len(a))} if len(a) else {'n': 0}


def best_k(order_t, g):
    if not g:
        return 1.0, 0
    best, bk, tp = 0.0, 0, 0
    for k, t in enumerate(order_t, 1):
        tp += t in g
        v = 1.25 * tp / (0.25 * len(g) + k)
        if v > best + 1e-12:
            best, bk = v, k
    return best, bk


def toks(s):
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    return frozenset(re.sub(r'[^\w]+', ' ', s).split())


def main():
    gold = {k: set(v) for k, v in json.loads((D / 'gold.json').read_text()).items()}
    countries = json.loads((D / 'countries.json').read_text())
    cp = partition_of(gold)
    mask = np.load(D / 'mask_top10.npy')
    refs, tgts = [], []
    with gzip.open(D / 'pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t')
                refs.append(r); tgts.append(t)
    refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
    part = np.array([cp[r] for r in refs], dtype=object)
    S = np.load(D / 'stage1.npz')
    y = np.load(D / 'y.npy')[S['rows']] > 0
    v5 = np.where(S['stage1'] < 0.02, 0.0, S['stage1']).astype(np.float64)
    c2, lw = np.nan_to_num(np.load(D / 'c2.npy')), np.nan_to_num(np.load(D / 'lw.npy'))
    C = np.load(D / 'cache_c2lw.npz')
    F = np.column_stack([C['F'], C['Frev'], C['Fnum']])
    names = NAMES27 + REV + NUM
    fi = {n: i for i, n in enumerate(names)}
    blank = C['blank']
    P3 = np.load(D / 'p3_c2lw.npy')
    rule3 = json.load(open(D / 'meta_c2lw.json'))['rule']
    P5 = np.load(D / 'cache_s3b.npz')['P']
    owner = {}
    with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            qq = line.rstrip('\n').split('\t')
            if len(qq) > 1 and qq[1]:
                for t in qq[1].split(','):
                    owner[t.strip()] = qq[0]
    ho = np.flatnonzero(part == 'holdout')
    H = sorted(s for s in gold if cp[s] == 'holdout')
    pos = {s: i for i, s in enumerate(H)}
    rix = np.array([pos[r] for r in refs[ho]])
    need = set(refs[ho]) | set(tgts[ho]) | {owner[t] for t in tgts[ho] if t in owner}
    text = {}
    for k in (1, 2, 3):
        with open(RAW / f'train_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
            for line in fh:
                qq = line.rstrip('\n').split('\t')
                if len(qq) > iaddr and qq[ie] in need:
                    text[qq[ie]] = (qq[iname], qq[iaddr])
    print('loaded', len(ho), 'holdout rows', flush=True)
    by_ref = defaultdict(list)
    for j in ho:
        by_ref[refs[j]].append(j)
    ch = {'S5': np.zeros(len(y), bool), 'gen3': np.zeros(len(y), bool)}
    ch['S5'][ho] = P5[ho] >= 0.68
    ch['gen3'][ho] = decide(rix, P3[ho], rule3[0], rule3[1], len(H))
    PR = {'S5': P5, 'gen3': P3}
    E = {'rule_gen3': rule3, 'n_refs': len(H), 'n_rows': int(len(ho))}
    kinds_by_ref = {}

    for gname in ('S5', 'gen3'):
        P, sel = PR[gname], ch[gname]
        loss, cnt, ceil = Counter(), Counter(), defaultdict(float)
        margins, strata = defaultdict(list), defaultdict(lambda: [0, 0.0])
        rank_c = defaultdict(Counter)
        fpt = Counter()
        sep = 0
        for s in H:
            g = gold[s]
            js = by_ref.get(s, [])
            order = sorted(js, key=lambda j: -P[j])
            ot = [tgts[j] for j in order]
            cand = set(ot)
            pred = {tgts[j] for j in js if sel[j]}
            kp, sc = len(pred), f05(g, pred)
            perfect = f05(g, g & cand)
            ok, ks = best_k(ot, g)
            ceil['achieved'] += sc; ceil['oracle_topk'] += ok; ceil['blocking'] += perfect
            gi = [r for r, t in enumerate(ot) if t in g]; ng = [r for r, t in enumerate(ot) if t not in g]
            sep += (not gi or not ng or max(gi) < min(ng))
            loss['blocking'] += 1 - perfect; loss['ranking'] += perfect - ok
            kind = ('singleton_fp' if not g and kp else 'missed_all' if g and not kp else 'over' if kp > ks
                    else 'under' if kp < ks else 'none')
            b = ok - sc
            if b > 1e-12:
                loss['boundary_' + kind] += b; cnt[kind] += 1
                if gname == 'gen3':
                    kinds_by_ref[s] = kind
                if kind in ('over', 'singleton_fp'):
                    margins[kind] += [P[j] for j in order[ks:kp]]
                else:
                    margins[kind] += [P[j] for j in order[kp:ks] if tgts[j] in g]
            elif ok - sc <= 1e-12 and perfect - ok > 1e-12 and gname == 'gen3':
                kinds_by_ref[s] = 'ranking'
            for r, j in enumerate(order):
                t = tgts[j]
                lab = 'TP' if t in g and sel[j] else 'FN' if t in g else 'FP' if sel[j] else 'TN'
                rank_c[lab][r + 1] += 1
                if lab == 'FP':
                    o = owner.get(t)
                    fpt['decoy' if o is None else 'owned_same_name' if toks(text.get(o, ('', ''))[0]) == toks(text.get(t, ('', ''))[0])
                        else 'owned_other'] += 1
            rn = toks(text.get(s, ('', ''))[0])
            exact = any(rn and rn == toks(text.get(t, ('', ''))[0]) for t in ot)
            for key in (f'country={countries.get(s, "UNK")}', f'gold={min(len(g), 6)}',
                        f'exact_name_in_cands={exact}', f'has_blank_cand={any(blank[j] for j in js)}'):
                strata[key][0] += 1; strata[key][1] += 1 - sc
        n = len(H)
        G = {'holdout': round(ceil['achieved'] / n, 6), 'ceilings': {k: round(v / n, 6) for k, v in ceil.items()},
             'loss_points': {k: round(v / n * 100, 4) for k, v in loss.items()},
             'boundary_error_refs': dict(cnt), 'boundary_margins': {k: q(v) for k, v in margins.items()},
             'separable_refs': round(sep / n, 5), 'fp_type': dict(fpt),
             'rank_TP_FN_FP': {r: [rank_c['TP'][r], rank_c['FN'][r], rank_c['FP'][r]] for r in range(1, 11)},
             'strata': {k: {'refs': v[0], 'f05': round(1 - v[1] / v[0], 5), 'loss_points': round(v[1] / n * 100, 4)}
                        for k, v in sorted(strata.items())}}
        G['loss_points']['total'] = round(100 - ceil['achieved'] / n * 100, 4)
        yh, ph = y[ho], P[ho]
        edges = [0, .02, .1, .3, .5, .7, .9, .98, 1.0001]
        bb = np.digitize(ph, edges) - 1
        G['calibration'] = [{'bin': f'{edges[i]:.2f}-{min(edges[i + 1], 1):.2f}', 'n': int((bb == i).sum()),
                             'mean_p': round(float(ph[bb == i].mean()), 4), 'pos_rate': round(float(yh[bb == i].mean()), 4)}
                            for i in range(len(edges) - 1) if (bb == i).any()]
        m = sel[ho]
        G['row_errors'] = {seg: {'fn': int((sg & yh & ~m).sum()), 'fp': int((sg & ~yh & m).sum())}
                           for seg, sg in (('blank', blank[ho]), ('nonblank', ~blank[ho]))}
        E[gname] = G
        print(gname, json.dumps({k: G[k] for k in ('holdout', 'ceilings', 'loss_points', 'row_errors')}), flush=True)

    # ---------- gen-3 error anatomy ----------
    sel = ch['gen3']
    vote = (v5 >= 0.5).astype(int) + (c2 >= 0).astype(int) + (lw >= 0).astype(int)
    base = E['gen3']['holdout']

    def macro(s_):
        return float(np.mean([f05(gold[s], {tgts[j] for j in by_ref.get(s, []) if s_[j]}) for s in H]))

    grid = {}
    for seg, sg in (('blank', blank), ('nonblank', ~blank)):
        for v in range(4):
            js = ho[sg[ho] & (vote[ho] == v)]
            if not len(js):
                continue
            fix = sel.copy(); fix[js] = y[js]
            grid[f'{seg} votes_yes={v}'] = {'n': int(len(js)), 'pos_rate': round(float(y[js].mean()), 4),
                                            'mean_p3': round(float(P3[js].mean()), 4), 'fn': int((y[js] & ~sel[js]).sum()),
                                            'fp': int((~y[js] & sel[js]).sum()), 'label_ceiling_gain': round(macro(fix) - base, 6)}
    E['gen3_vote_grid'] = grid
    groups = {}
    for seg, sg in (('blank', blank), ('nonblank', ~blank)):
        h = ho[sg[ho]]
        groups[f'{seg}_FN'] = h[y[h] & ~sel[h]]
        groups[f'{seg}_FP'] = h[~y[h] & sel[h]]
        groups[f'{seg}_TP'] = h[y[h] & sel[h]]
        groups[f'{seg}_TN_band'] = h[~y[h] & ~sel[h] & (P3[h] >= 0.1)]
    cols = ['rn_rank', 'rn_gap', 'rn_close', 'rf_rank', 'rf_gap', 'rf_close', 'mult_s1_tgt', 'n_acc_same_src', 'rank',
            'key_equal', 'name_sim', 'addr_sim', 'num_eq_ref', 'num_group', 'decoy_word', 'is_s3']
    E['gen3_profiles'] = {g: {'n': int(len(js)), 'p3': round(float(np.median(P3[js])), 3), 'v5': round(float(np.median(v5[js])), 3),
                              'c2': round(float(np.median(c2[js])), 2), 'lw': round(float(np.median(lw[js])), 2),
                              **{c: round(float(np.median(F[js, fi[c]])), 3) for c in cols},
                              'rn_rank1': round(float((F[js, fi['rn_rank']] == 1).mean()), 3)}
                          for g, js in groups.items() if len(js)}
    E['gen3_error_ref_kinds'] = dict(Counter(kinds_by_ref.values()))
    (D / 'gen3_eda.json').write_text(json.dumps(E, indent=1, default=str))

    # ---------- case files ----------
    out = ['# Gen-3 error references (holdout, stage-3b chunk2+listwise, expected@0.60)\n',
           'Per reference: candidates sorted by p3. * = chosen, G = gold. v5 / chunk2 / listwise shown. rival = rn_rank,rn_gap,rn_close\n']
    rng = np.random.default_rng(3)
    for kind in ('under', 'over', 'missed_all', 'singleton_fp', 'ranking'):
        rs = [s for s, k in kinds_by_ref.items() if k == kind]
        rs = list(rng.choice(rs, min(40, len(rs)), replace=False)) if rs else []
        out.append(f'\n## {kind} ({sum(1 for k in kinds_by_ref.values() if k == kind)} refs, showing {len(rs)})\n')
        for s in rs:
            rn, ra = text.get(s, ('', ''))
            out.append(f'\n**{s}** {rn} | {ra}  (gold {len(gold[s])}, {countries.get(s, "")})')
            for j in sorted(by_ref.get(s, []), key=lambda j: -P3[j]):
                tn, ta = text.get(tgts[j], ('', ''))
                out.append(f'- {"*" if sel[j] else " "}{"G" if tgts[j] in gold[s] else " "} p3={P3[j]:.2f} v5={v5[j]:.2f} '
                           f'c2={c2[j]:5.1f} lw={lw[j]:5.1f} rival={int(F[j, fi["rn_rank"]])},{F[j, fi["rn_gap"]]:+.3f},'
                           f'{int(F[j, fi["rn_close"]])} {tgts[j][:2]} | {tn} | {ta}')
    (D / 'gen3_cases.md').write_text('\n'.join(out))
    print('done', flush=True)


if __name__ == '__main__':
    main()
