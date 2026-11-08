"""Error sweep over every stage of the final pipeline, on the labelled sample (actual loss) and on test (expected loss).

Stages: retrieval (true record not in the ~290 pool), top-10 prune (in the pool, cut), stage-3b decision (FN / FP inside
the list), and for test the post-processing steps (exclusivity finalize, France prune, France census).
Train (MODE=train): T6 expected@0.74 on dev + holdout. Each S1's loss 1 - F0.5 is shared equally over its error items;
every item gets pattern attributes (target blank, name relation, address relation, house-number shift class, S1 name
multiplicity in the full S1 table, p band, S1 case). Also the label-free expected loss of the same decisions (exact
E[1 - F0.5] under independent calibrated p, shared over pairs by error probability), so the estimator can be checked
against the actual loss per cell before it is applied to test. Per-cell logit recalibration: shift fitted on dev,
holdout re-decided once.
Test (MODE=test): U2's final decisions with T6 p (US/India) and FRtrained_w4 p (France): expected loss per cell.
Writes OUT (json) and OUT.txt (examples).
"""
import gzip
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
sys.path.insert(0, str(Path(__file__).parent))
from distractor_check import decide, f05, partition_of
from france_census import core, house, load
from usin_errors import EN_FILLER, rel

H = Path('/home/scai/mtech/aib262015/scratch')
V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
A = H / 'amazon_ml_2026/artifacts'
SHIFTS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
PB = [0, 0.05, 0.2, 0.4, 0.6, 0.74, 0.9, 0.98, 1.01]
G = {}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def mkey(name):
    return ' '.join(sorted(set(core(name)) - EN_FILLER))


def num(a):
    h = house(a)[0]
    m = re.match(r'\d+', h)
    return m.group(0) if m else ''


def shift_class(ra, ta):
    a, b = num(ra), num(ta)
    if not a or not b:
        return 'no_num'
    if a == b:
        return 'same'
    if len(a) != len(b) and (a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a)):
        return 'digit_drop'
    d = abs(int(a) - int(b))
    return 'decoy_shift' if d in SHIFTS else ('small_other' if d <= 30 else 'far')


def mult_bucket(m):
    return '1' if m <= 1 else ('2' if m == 2 else ('3-5' if m <= 5 else '6+'))


def pband(p):
    for lo, hi in zip(PB, PB[1:]):
        if p < hi:
            return f'{lo:.2f}-{min(hi, 1):.2f}'
    return '0.98-1.00'


def attrs(rn, ra, cty, tn, ta, tid, mult):
    nm, ad = rel(rn, ra, tn, ta)
    return dict(country=cty, src=tid[:2], blank='blank' if not ta.strip() else 'nonblank', name=nm, addr=ad,
                shift=shift_class(ra, ta) if ta.strip() else 'blank', mult=mult_bucket(mult))


def attr_chunk(items):
    T, M = G['text'], G['mult']
    out = []
    for k, r, t in items:
        rn, ra, cty = T.get(r, ('', '', ''))
        tn, ta, _ = T.get(t, ('', '', ''))
        out.append((k, attrs(rn, ra, cty, tn, ta, t, M.get((cty, mkey(rn)), 1))))
    return out


def attr_parallel(items, nproc):
    n = max(1, nproc)
    chunks = [items[i::n * 4] for i in range(n * 4)]
    res = {}
    with mp.get_context('fork').Pool(n) as pool:
        for part in pool.imap_unordered(attr_chunk, chunks):
            res.update(part)
    return res


def expected_f(Pm, Dm):
    """E[F0.5] per row, independent Bernoulli p, decision mask D; padded slots have p = 0 and D = False."""
    m, n = Pm.shape
    pin = np.zeros((m, n + 1)); pin[:, 0] = 1
    pout = np.zeros((m, n + 1)); pout[:, 0] = 1
    for c in range(n):
        for pmf, q in ((pin, Pm[:, c] * Dm[:, c]), (pout, Pm[:, c] * ~Dm[:, c])):
            pmf[:, 1:] = pmf[:, 1:] * (1 - q)[:, None] + pmf[:, :-1] * q[:, None]
            pmf[:, 0] *= 1 - q
    d = Dm.sum(1).astype(np.float64)
    E = np.zeros(m)
    for a in range(n + 1):
        for b in range(n + 1):
            g = a + b
            if a == 0:
                f = np.where((d == 0) & (g == 0), 1.0, 0.0)
            else:
                pr = a / np.maximum(d, 1); rc = a / g
                f = np.where(d >= a, 1.25 * pr * rc / (0.25 * pr + rc), 0.0)
            E += pin[:, a] * pout[:, b] * f
    return E


def pad(lists, P, D, n=10):
    Pm = np.zeros((len(lists), n)); Dm = np.zeros((len(lists), n), dtype=bool)
    for i, js in enumerate(lists):
        js = js[:n]
        Pm[i, :len(js)] = P[js]; Dm[i, :len(js)] = D[js]
    return Pm, Dm


def share_expected(lists, P, D, EL):
    """Pair share of the S1's expected loss, proportional to the pair's error probability."""
    sh = np.zeros(len(P))
    for i, js in enumerate(lists):
        js = np.asarray(js)
        e = np.where(D[js], 1 - P[js], P[js])
        if e.sum() > 0:
            sh[js] = EL[i] * e / e.sum()
    return sh


def read_text(split, need, all_s1=True):
    raw = H / f'amazon_ml_2026/data/dataset/{split}'
    text, s1 = {}, []
    for k in (1, 2, 3):
        with open(raw / f'{split}_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if k == 1:
                    s1.append(q[ie])
                if (k == 1 and all_s1) or q[ie] in need:
                    text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
    return text, s1


def multiplicity(text, s1):
    c = Counter()
    for r in s1:
        n, _, cty = text[r]
        c[(cty, mkey(n))] += 1
    return c


def agg(df, keys, cols):
    g = df.groupby(keys, dropna=False)[cols].sum().reset_index()
    return g


def train_part(out):
    nproc = int(os.environ.get('NPROC', 8))
    gold = {k: set(v) for k, v in json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text()).items()}
    cp = partition_of(gold)
    evalrefs = {s for s in gold if cp[s] in ('dev', 'holdout')}
    gp = {(s, t) for s in evalrefs for t in gold[s]}
    mask = np.load(A / 'prune_v1_copy/mask_top10.npy')
    refs, tgts, inpool = [], [], set()
    with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            r, t = line.rstrip('\n').split('\t')
            if (r, t) in gp:
                inpool.add((r, t))
            if mask[i]:
                refs.append(r); tgts.append(t)
    refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
    log('pairs read', len(refs), 'gold eval pairs', len(gp), 'in pool', len(inpool))
    P = np.load(V / 'fusion/boundary_models_t6/stage3b_prob.npy').astype(np.float64)
    assert len(P) == len(refs), (len(P), len(refs))
    need = set(tgts) | {t for s in evalrefs for t in gold[s]}
    text, s1 = read_text('train', need)
    M = multiplicity(text, s1)
    G.update(text=text, mult=M)
    log('text + multiplicity', len(text))
    thr = float(os.environ.get('RULE_T', '0.74'))
    part = np.array([cp[r] for r in refs], dtype=object)
    rows, blk, s1rows, R = [], [], [], {}
    rng = random.Random(1)
    ex = defaultdict(list)
    decided = {}
    for p in ('dev', 'holdout'):
        ix = np.flatnonzero(part == p)
        prefs = sorted(s for s in gold if cp[s] == p)
        pos = {s: i for i, s in enumerate(prefs)}
        ch = decide(np.array([pos[r] for r in refs[ix]]), P[ix], 'expected', thr, len(prefs))
        D = np.zeros(len(P), dtype=bool); D[ix[ch]] = True
        decided[p] = (ix, prefs, pos, D)
        lists = defaultdict(list)
        for j in ix:
            lists[refs[j]].append(j)
        L = [sorted(lists.get(s, []), key=lambda j: -P[j]) for s in prefs]
        Pm, Dm = pad(L, P, D)
        EL = 1 - expected_f(Pm, Dm)
        esh = share_expected(L, P, D, EL)
        items = []
        for i, s in enumerate(prefs):
            g = gold[s]; js = L[i]
            pred = {tgts[j] for j in js if D[j]}
            f = f05(g, pred)
            cs = {tgts[j] for j in js}
            errs = [j for j in js if (tgts[j] in g) != D[j]]
            miss = [t for t in g if t not in cs]
            n_err = len(errs) + len(miss)
            case = 'singleton' if not g else ('pred_empty' if not pred else 'partial')
            cty = text.get(s, ('', '', ''))[2]
            s1rows.append(dict(part=p, country=cty, loss=1 - f, exp_loss=EL[i], ngold=len(g), npred=len(pred), ncand=len(js),
                               nmiss=len(miss), case=case))
            for j in js:
                yv = tgts[j] in g
                rows.append(dict(part=p, j=int(j), s1=s, t=tgts[j], y=int(yv), p=P[j], D=int(D[j]), case=case,
                                 kind=('FP' if D[j] else 'FN') if yv != D[j] else ('TP' if yv else 'TN'),
                                 loss=(1 - f) / n_err if yv != D[j] else 0.0, exp_loss=esh[j]))
                items.append((len(rows) - 1, s, tgts[j]))
            for t in miss:
                blk.append(dict(part=p, s1=s, t=t, stage='prune' if (s, t) in inpool else 'retrieval', loss=(1 - f) / n_err,
                                case=case, ngold=len(g)))
                items.append((('b', len(blk) - 1), s, t))
        log(p, 'refs', len(prefs), 'actual F', round(1 - np.mean([1 - f05(gold[s], {tgts[j] for j in L[i] if D[j]}) for i, s in enumerate(prefs)]), 5),
            'expected F (in-list)', round(float(1 - EL.mean()), 5))
        A_ = attr_parallel(items, nproc)
        for k, a in A_.items():
            if isinstance(k, tuple):
                blk[k[1]].update(a)
            else:
                rows[k].update(a)
    df = pd.DataFrame(rows); bf = pd.DataFrame(blk); sf = pd.DataFrame(s1rows)
    df['pband'] = df.p.map(pband)
    df['err'] = df.kind.isin(['FP', 'FN']).astype(int)
    df['ekind'] = np.where(df.D == 1, 'FP', 'FN')
    n_s1 = sf.groupby(['part', 'country']).size().to_dict()
    R['n_s1'] = {f'{a}|{b}': int(v) for (a, b), v in n_s1.items()}
    R['totals'] = {p: dict(loss_points=round(100 * sf[sf.part == p].loss.mean(), 4),
                           exp_points_inlist=round(100 * sf[sf.part == p].exp_loss.mean(), 4),
                           blocking_points=round(100 * bf[bf.part == p].loss.sum() / (sf.part == p).sum(), 4),
                           by_stage={st: round(100 * bf[(bf.part == p) & (bf.stage == st)].loss.sum() / (sf.part == p).sum(), 4) for st in ('retrieval', 'prune')},
                           decision_FN=round(100 * df[(df.part == p) & (df.kind == 'FN')].loss.sum() / (sf.part == p).sum(), 4),
                           decision_FP=round(100 * df[(df.part == p) & (df.kind == 'FP')].loss.sum() / (sf.part == p).sum(), 4),
                           case_points={c: round(100 * sf[(sf.part == p) & (sf.case == c)].loss.sum() / (sf.part == p).sum(), 4) for c in ('singleton', 'pred_empty', 'partial')},
                           gold_over_10=int(((sf.part == p) & (sf.ngold > 10)).sum()))
                   for p in ('dev', 'holdout')}
    df.drop(columns=['j']).to_pickle(str(out) + '.train_pairs.pkl')
    bf.to_pickle(str(out) + '.train_blocking.pkl'); sf.to_pickle(str(out) + '.train_s1.pkl')
    log('frames written')

    # per-cell logit recalibration: shift fitted on dev, holdout re-decided once
    cellk = ['country', 'blank', 'name', 'addr']
    unc = df[(df.p > 0.02) & (df.p < 0.98)]
    shifts = {}
    for key, g in unc[unc.part == 'dev'].groupby(cellk):
        if len(g) < 150:
            continue
        z = np.log(np.clip(g.p, 1e-6, 1 - 1e-6) / (1 - np.clip(g.p, 1e-6, 1 - 1e-6))).values; yv = g.y.values
        b = 0.0
        for _ in range(30):
            q = 1 / (1 + np.exp(-(z + b)))
            b -= (q - yv).sum() / max((q * (1 - q)).sum(), 1e-9)
            b = float(np.clip(b, -3, 3))
        shifts[key] = b
    ix, prefs, pos, _ = decided['holdout']
    base_F = 1 - sf[sf.part == 'holdout'].loss.mean()

    def holdout_F(Pnew):
        ch = decide(np.array([pos[r] for r in refs[ix]]), Pnew[ix], 'expected', thr, len(prefs))
        pred = defaultdict(set)
        for j, c in zip(ix, ch):
            if c:
                pred[refs[j]].add(tgts[j])
        return float(np.mean([f05(gold[s], pred.get(s, set())) for s in prefs]))

    keyarr = list(zip(*(df[k] for k in cellk)))
    jarr = df.j.values
    recal = []
    Pall = P.copy()
    for key, b in sorted(shifts.items(), key=lambda kv: -abs(kv[1])):
        Pn = P.copy()
        sel = np.array([k == key for k in keyarr]) & (df.p.values > 0.02) & (df.p.values < 0.98)
        js = jarr[sel]
        z = np.log(P[js] / (1 - P[js]))
        Pn[js] = 1 / (1 + np.exp(-(z + b)))
        Pall[js] = Pn[js]
        recal.append([str(key), round(b, 3), int(sel.sum()), round(holdout_F(Pn) - base_F, 6)])
    R['recalibration'] = dict(holdout_base=round(base_F, 6), all_cells_delta=round(holdout_F(Pall) - base_F, 6), cells=recal)
    log('recalibration', R['recalibration']['all_cells_delta'])

    for kind_, frame in (('decision', df[df.err == 1]), ('blocking', bf)):
        for key, g in frame.groupby(['blank', 'name', 'addr'] if kind_ == 'blocking' else ['kind', 'blank', 'name', 'addr']):
            g = g[g.part == 'holdout']
            if len(g):
                smp = g.sample(min(12, len(g)), random_state=1)
                for _, r in smp.iterrows():
                    s = r.s1; accepted = [x for x in df[(df.s1 == s) & (df.D == 1)].t if x != r.t][:4]
                    ex[(kind_,) + tuple(key)].append(
                        (f"p={r.p:.2f} " if 'p' in r else f"[{r.stage}] ") + f"{text.get(s, ('', '', ''))[0]} | {text.get(s, ('', '', ''))[1]}  =>  "
                        f"{text.get(r.t, ('', '', ''))[0]} | {text.get(r.t, ('', '', ''))[1]}   [acc: "
                        + '; '.join(f"{text.get(x, ('', ''))[0]} | {text.get(x, ('', ''))[1]}" for x in accepted) + ']')
    json.dump(R, open(out, 'w'), indent=1, ensure_ascii=False, default=str)
    big = sorted(ex, key=lambda k: -len(ex[k]))
    with open(str(out) + '.txt', 'w', encoding='utf-8') as f:
        for k in big:
            f.write(f'\n## {k}\n' + '\n'.join(ex[k]) + '\n')
    log('train done')


def test_part(out):
    nproc = int(os.environ.get('NPROC', 8))
    T6 = np.load(V / 'submission_T6_own/late_scores.npz', allow_pickle=True)
    ri, tg, p6 = T6['ref_idx'], T6['target'].astype(str).astype(object), T6['prob'].astype(np.float64)
    FR = np.load(V / 'submission_FRtrained_w4/late_scores.npz', allow_pickle=True)
    assert np.array_equal(FR['ref_idx'], ri)
    pfr = FR['prob'].astype(np.float64)
    log('test pairs', len(ri))
    text, s1 = read_text('test', set(tg))
    M = multiplicity(text, s1)
    G.update(text=text, mult=M)
    cty = np.array([text[r][2] for r in s1], dtype=object)
    fr = cty[ri] == 'France'
    P = np.where(fr, pfr, p6)
    final = load(A / 'submission_U2_final/matching_results.tsv')
    t6_dec = load(V / 'submission_T6_own/expected_0.74/matching_results.tsv')
    fr_model = load(V / 'submission_FRtrained_w4/expected_0.64/matching_results.tsv')
    fr_prune = load(A / 'submission_FRw4_xprune/expected_0.64/matching_results.tsv')
    refs_arr = np.array(s1, dtype=object)[ri]

    def member(d):
        ps = {(r, t) for r, ts in d.items() for t in ts}
        return np.fromiter(((r, t) in ps for r, t in zip(refs_arr, tg)), dtype=bool, count=len(tg))
    D = member(final)
    raw_ch = np.zeros(len(P), dtype=bool)
    for m_, pp, th in ((~fr, p6, 0.74), (fr, pfr, 0.64)):
        ix = np.flatnonzero(m_)
        raw_ch[ix] = decide(ri[ix], pp[ix], 'expected', th, len(s1))
    log('decisions loaded; final links', int(D.sum()), 'raw decide links', int(raw_ch.sum()))
    step = np.full(len(P), 'model', dtype=object)
    in_t6 = member(t6_dec)
    log('US/India final vs T6 decision mismatches', int(((D != in_t6) & ~fr).sum()))
    in_frm = fr & member(fr_model)
    in_frp = fr & member(fr_prune)
    fin_model = np.where(fr, in_frm, in_t6)
    step[raw_ch & ~fin_model] = 'exclusivity_drop'
    step[fr & in_frm & ~in_frp] = 'xprune_drop'
    step[fr & in_frp & ~D] = 'census_reject'
    step[fr & ~in_frp & D] = 'census_restore'
    log('steps', Counter(step))
    by = defaultdict(list)
    for j, r in enumerate(ri):
        by[r].append(j)
    order = sorted(by)
    L = [sorted(by[r], key=lambda j: -P[j]) for r in order]
    Pm, Dm = pad(L, P, D)
    EL = 1 - expected_f(Pm, Dm)
    esh = share_expected(L, P, D, EL)
    s1c = cty[np.array(order)]
    npred = Dm.sum(1)
    sfr = pd.DataFrame(dict(country=s1c, exp_loss=EL, npred=npred, top_p=Pm[:, 0], n10=(Pm > 0).sum(1)))
    R = {'n_s1': {k: int(v) for k, v in Counter(cty).items()},
         'expected_points_inlist': {c: round(100 * float(sfr[sfr.country == c].exp_loss.mean()), 4) for c in sorted(set(cty))},
         'expected_points_public_inlist': round(100 * float(EL.sum() / len(s1)), 4),
         'pred_empty_rate': {c: round(float((sfr[sfr.country == c].npred == 0).mean()), 5) for c in sorted(set(cty))},
         'all10_accepted': {c: int(((sfr.country == c) & (sfr.npred >= 10)).sum()) for c in sorted(set(cty))},
         'steps': {k: int(v) for k, v in Counter(step).items()}}
    e = np.where(D, 1 - P, P)
    sel = np.flatnonzero((e > 0.003) | (step != 'model'))
    log('uncertain / changed pairs for attributes', len(sel), 'of', len(P))
    items = [(int(j), s1[ri[j]], tg[j]) for j in sel]
    A_ = attr_parallel(items, nproc)
    rows = []
    for j in sel:
        a = A_[int(j)]
        rows.append(dict(a, j=int(j), s1=s1[ri[j]], t=tg[j], p=P[j], D=int(D[j]), step=step[j], exp_loss=esh[j],
                         ekind='FP' if D[j] else 'FN'))
    df = pd.DataFrame(rows)
    df['pband'] = df.p.map(pband)
    rest = np.ones(len(P), dtype=bool); rest[sel] = False
    R['confident_rest_points'] = {c: round(100 * float(esh[rest & (cty[ri] == c)].sum()) / len(s1), 5) for c in sorted(set(cty))}
    df.to_pickle(str(out) + '.test_pairs.pkl'); sfr.to_pickle(str(out) + '.test_s1.pkl')
    rng = random.Random(2)
    ex = defaultdict(list); W = {}
    for key, g in df.groupby(['country', 'ekind', 'blank', 'name', 'addr', 'step']):
        w = W[key] = g.exp_loss.sum()
        if w * 100 / len(s1) < 0.002:
            continue
        smp = g.sample(min(10, len(g)), weights=g.exp_loss + 1e-9, random_state=1)
        for _, r in smp.iterrows():
            acc = [x for x in final.get(r.s1, []) if x != r.t][:4]
            ex[key].append(f"p={r.p:.2f} D={r.D} {text[r.s1][0]} | {text[r.s1][1]}  =>  {text.get(r.t, ('', ''))[0]} | "
                           f"{text.get(r.t, ('', ''))[1]}   [acc: " + '; '.join(f"{text.get(x, ('', ''))[0]} | {text.get(x, ('', ''))[1]}" for x in acc) + ']')
    json.dump(R, open(out, 'w'), indent=1, ensure_ascii=False, default=str)
    with open(str(out) + '.txt', 'w', encoding='utf-8') as f:
        for k in sorted(ex, key=lambda k: -W[k]):
            f.write(f'\n## {k}\n' + '\n'.join(ex[k]) + '\n')
    log('test done', json.dumps(R))


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    (train_part if os.environ.get('MODE', 'train') == 'train' else test_part)(Path(os.environ['OUT']))
