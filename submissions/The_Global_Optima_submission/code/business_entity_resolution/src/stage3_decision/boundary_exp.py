"""Learned stop decision on top of the S5 ranking (gen2_eda: ~76% of the S5 holdout loss is the stop decision).

The per-pair probability is already calibrated, so a smarter threshold on it alone cannot help (exact expected-F0.5
DP gave nothing). The stop decision needs information the pairwise score lacks:
  list context   - prefix mass, accepted count per source, candidate-to-accepted-candidate similarity,
                   same-name-other-address 'chain' candidates in the list
  global context - how many S1 entities / S2+S3 records share this (order-free, suffix-free) name key
                   (a blank-address record whose name belongs to one S1 entity is that entity's)
Two learners on the same features (S5 L2 probability enters out-of-fold on train):
  stage3  - XGBoost pair model -> dev-chosen threshold/expected rule
  cut     - Choppy-style cut transformer (Bahri et al. 2020): list of <=10 candidates in S5 order -> softmax over
            cut positions 0..10, loss = -expected F0.5 of the cut
Rule/epoch chosen on dev, holdout scored once. Writes fusion/boundary_exp.json and fusion/boundary_models/.
"""
import gzip
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).parent))
from distractor_check import decide, f05, partition_of

HOME = Path.home()
V4 = HOME / 'scratch/vishal_v4_artifacts'
PRUNE = HOME / 'scratch/amazon_ml_2026/artifacts/prune_v1_copy'
FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
CE = os.environ.get('CE_FILE', '/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/mdeberta_full/ce_logit_sample.npy')
W = float(os.environ.get('DECOY_W', 3))
OUT = FUS / os.environ.get('BOUNDARY_OUT', 'boundary_models')
STOP = {'inc', 'llc', 'ltd', 'pvt', 'private', 'limited', 'co', 'corp', 'corporation', 'company', 'lp', 'llp',
        'pllc', 'plc', 'sa', 'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'ei', 'scop', 'selarl', 'cie', 'the', 'and',
        'of'}
# 'S1 name + one of these' is 0% positive in train even at the same house number (affix_check.py); French
# equivalents added for the test-only France records. Kept out of STOP so such decoys never share a name key.
DECOY_WORDS = {'group', 'groupe', 'holdings', 'holding', 'enterprises', 'entreprises', 'industries', 'ventures',
               'global', 'trading', 'solutions', 'international', 'technologies', 'management', 'systems',
               'consulting', 'sons', 'fils', 'development', 'developpement', 'distribution', 'participations',
               'associates', 'usa', 'america'}
NAMES = ['p', 'logit_p', 's1f', 'ce', 'rank', 'p_prev', 'p_next', 'sum_p', 'n_acc', 'n_acc_same_src',
         'n_acc_other_src', 'is_s3', 'tgt_addr_empty', 'ref_addr_empty', 'name_sim', 'addr_sim', 'key_equal',
         'max_name_sim_acc', 'max_addr_sim_acc', 'name_sim_top1', 'n_chain_list', 'is_chain', 'mult_s1_ref',
         'mult_s1_tgt', 'mult_s23_tgt', 'n_same_key_list', 'decoy_word',
         'a_s1_ref', 'a_s1_tgt', 'a_s23_names', 'skey_eq', 'unit_eq', 'no_name_overlap',
         'squash_match', 'core_swap', 'rare_key_jacc', 'freq_tok_jacc',
         'num_shift', 'shift_up_1_13', 'num2_eq']
# Address-key statistics over the whole split (filled by read_sources): S1 entities per street key, distinct name keys
# of S2/S3 records per address key (see addr_keys).
ADDR = {'s1': Counter(), 's23': Counter()}
GENERIC = {'street', 'st', 'road', 'rd', 'avenue', 'ave', 'av', 'drive', 'dr', 'lane', 'ln', 'boulevard', 'blvd', 'bd',
           'rue', 'route', 'rte', 'chemin', 'impasse', 'imp', 'allee', 'place', 'pl', 'court', 'ct', 'way', 'unit',
           'suite', 'ste', 'apt', 'floor', 'fl', 'flat', 'bldg', 'building', 'office', 'no', 'north', 'south', 'east',
           'west', 'the', 'of', 'and', 'de', 'la', 'le', 'des', 'du', 'bis', 'ter', 'near', 'opp', 'behind'}
RARE = 3000
UNIT = re.compile(r'\b(?:unit|suite|ste|apt|fl|floor|flat|bldg|office)\s+(\w+)')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def norm(s):
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r'[^\w]+', ' ', s).strip()


def key(name):
    return ' '.join(sorted({t for t in norm(name).split() if t not in STOP}))


def addr_keys(na, f1):
    """Order-free address keys: house number + each rare street word (seen in 1..RARE-1 S1 addresses), so city and
    state words drop out and field order / abbreviation style of the source does not matter."""
    t = na.split()
    num = next((str(int(x)) for x in t if x.isdigit()), '')  # '0014' (French S2 style) == '14'
    if not num or num == '0':
        return ()
    return tuple(sorted({f'{num}|{x}' for x in t if len(x) >= 3 and not x.isdigit() and x not in GENERIC
                         and 0 < f1.get(x, 0) < RARE}))


def unit_of(na):
    m = UNIT.search(na)
    return m.group(1) if m else ''


def read_sources(raw, split, need):
    """text[id] = (norm name, norm addr, name key, address keys, unit); name-key counts over S1 and over S2+S3;
    address-key statistics into ADDR (S1 entities per key, distinct S2/S3 name keys per key)."""
    f1 = Counter()
    with open(raw / f'{split}_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        iaddr = head.index('business_address')
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if len(q) > iaddr:
                f1.update({x for x in norm(q[iaddr]).split() if len(x) >= 3 and not x.isdigit()})
    text, c1, c23 = {}, Counter(), Counter()
    ADDR['s1'], s23 = Counter(), set()
    for k in (1, 2, 3):
        with open(raw / f'{split}_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) <= iaddr:
                    continue
                kk = key(q[iname])
                (c1 if k == 1 else c23)[kk] += 1
                na = norm(q[iaddr])
                ak = addr_keys(na, f1)
                if k == 1:
                    ADDR['s1'].update(ak)
                else:
                    s23.update((x, kk) for x in ak)
                if q[ie] in need:
                    text[q[ie]] = (norm(q[iname]), na, kk, ak, unit_of(na))
    ADDR['s23'] = Counter(x for x, _ in s23)
    ADDR['f1'] = f1
    return text, c1, c23


def nums(na):
    """First two house-number tokens, zero padding removed. Decoys sit at the S1's number shifted up by 1-13 and
    India decoys keep the first number but change a sub-number."""
    t = [str(int(x)) for x in na.split() if x.isdigit()]
    return (t + ['', ''])[:2]


def squash(name_norm):
    return ''.join(t for t in name_norm.split() if t not in STOP and t not in ('com', 'www'))


def common_toks(na):
    f1 = ADDR.get('f1', {})
    return {x for x in na.split() if len(x) >= 3 and not x.isdigit() and x not in GENERIC and f1.get(x, 0) >= RARE}


def list_features(p, s1f, ce, tgt_ids, ref_id, text, c1, c23):
    """One reference, candidates already sorted by p descending -> (n, len(NAMES)) float32."""
    n = len(p)
    rn, ra, rk, rsk, ru = text.get(ref_id, ('', '', '', (), ''))
    tn = [text.get(t, ('', '', '', '', ''))[0] for t in tgt_ids]
    rset = set(rn.split())
    rcore = rset - STOP
    tsk = [text.get(t, ('', '', '', '', ''))[3] or () for t in tgt_ids]
    tu = [text.get(t, ('', '', '', '', ''))[4] for t in tgt_ids]
    a_ref = np.log1p(min(ADDR['s1'].get(x, 0) for x in rsk)) if rsk else -1.0
    rks = set(rsk)
    rsq = squash(rn)
    rct = common_toks(ra)
    rn1, rn2 = nums(ra)
    ta = [text.get(t, ('', '', '', '', ''))[1] for t in tgt_ids]
    tk = [text.get(t, ('', '', '', '', ''))[2] for t in tgt_ids]
    tsq = [squash(x) for x in tn]
    tcore = [set(x.split()) - STOP for x in tn]
    tct = [common_toks(x) for x in ta]
    tnn = [nums(x) for x in ta]
    s3 = np.array([t.startswith('S3') for t in tgt_ids], dtype=np.float32)
    ns = process.cdist([rn], tn, scorer=fuzz.token_sort_ratio)[0] if rn else np.zeros(n)
    asim = process.cdist([ra], ta, scorer=fuzz.token_set_ratio)[0] if ra else np.zeros(n)
    asim = np.where([bool(a) and bool(ra) for a in ta], asim, -1.0)
    nn_ = process.cdist(tn, tn, scorer=fuzz.token_sort_ratio)
    aa = process.cdist(ta, ta, scorer=fuzz.token_set_ratio)
    empty_a = np.array([not a for a in ta])
    acc = p >= 0.5
    chain = (ns >= 90) & ~empty_a & (asim >= 0) & (asim < 60)
    kc = Counter(tk)
    F = np.zeros((n, len(NAMES)), dtype=np.float32)
    for i in range(n):
        others = acc.copy(); others[i] = False
        oa = others & ~empty_a
        F[i] = [p[i], np.log(max(p[i], 1e-6) / max(1 - p[i], 1e-6)), s1f[i], ce[i], i + 1,
                p[i - 1] if i else 1.0, p[i + 1] if i + 1 < n else 0.0, p.sum(), acc.sum(),
                (others & (s3 == s3[i])).sum(), (others & (s3 != s3[i])).sum(), s3[i], float(empty_a[i]),
                float(not ra), ns[i], asim[i], float(bool(rk) and rk == tk[i]),
                nn_[i][others].max() if others.any() else -1.0,
                aa[i][oa].max() if oa.any() and not empty_a[i] else -1.0,
                nn_[i][0] if i else -1.0, chain.sum(), float(chain[i]), np.log1p(c1.get(rk, 0)),
                np.log1p(c1.get(tk[i], 0)), np.log1p(c23.get(tk[i], 0)), kc[tk[i]],
                float(bool((set(tn[i].split()) - rset) & DECOY_WORDS)),
                a_ref, np.log1p(min(ADDR['s1'].get(x, 0) for x in tsk[i])) if tsk[i] else -1.0,
                np.log1p(min(ADDR['s23'].get(x, 0) for x in tsk[i])) if tsk[i] else -1.0,
                float(bool(rks & set(tsk[i]))) if tsk[i] else -1.0,
                (float(tu[i] == ru) if ru and tu[i] else -1.0),
                float(not (rcore & (set(tn[i].split()) - STOP))),
                float(len(rsq) >= 4 and len(tsq[i]) >= 4 and (tsq[i] in rsq or rsq in tsq[i])),
                float(len(rcore - tcore[i]) == 1 and len(tcore[i] - rcore) == 1),
                len(rks & set(tsk[i])) / len(rks | set(tsk[i])) if rks and tsk[i] else -1.0,
                len(rct & tct[i]) / len(rct | tct[i]) if rct and tct[i] else -1.0,
                float(np.clip(int(tnn[i][0]) - int(rn1), -50, 50)) if rn1 and tnn[i][0] else -99.0,
                float(1 <= int(tnn[i][0]) - int(rn1) <= 13) if rn1 and tnn[i][0] else -1.0,
                float(tnn[i][1] == rn2) if rn2 and tnn[i][1] else -1.0]
    return F


class Cut(torch.nn.Module):
    """Position 0 = CLS = cut before the first candidate; position k = cut after candidate k."""
    def __init__(self, d_in, d=64):
        super().__init__()
        self.inp = torch.nn.Linear(d_in, d)
        self.cls = torch.nn.Parameter(torch.zeros(1, 1, d))
        self.pos = torch.nn.Embedding(11, d)
        layer = torch.nn.TransformerEncoderLayer(d, 4, 128, 0.1, batch_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, 2)
        self.head = torch.nn.Linear(d, 1)

    def forward(self, x, m):
        h = torch.cat([self.cls.expand(len(x), -1, -1), self.inp(x)], 1) + self.pos.weight[None, :x.shape[1] + 1]
        pad = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device), ~m], 1)
        return self.head(self.enc(h, src_key_padding_mask=pad)).squeeze(-1).masked_fill(pad, -1e4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
    y = np.load(V4 / 'pairs/y.npy')[S['rows']].astype(np.float32)
    s1f = np.where(S['stage1'].astype(np.float64) < 0.02, 0.0, S['stage1'].astype(np.float64))
    ce = np.nan_to_num(np.load(CE).astype(np.float64), nan=0.0)
    owned = set()
    with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if len(q) > 1 and q[1]:
                owned.update(t.strip() for t in q[1].split(','))
    wt = np.where((y == 0) & np.array([t not in owned for t in tgts]), W, 1.0).astype(np.float32)
    trA, trB, dv = part == 'trainA', part == 'trainB', part == 'dev'
    tr = trA | trB

    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                  max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)
    X2 = np.column_stack([s1f, ce]).astype(np.float32)

    def fit(X, fit_rows, **kw):
        d = xgb.QuantileDMatrix(X[fit_rows], y[fit_rows], weight=wt[fit_rows])
        b = xgb.train(params | kw, d, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=d), 'dev')],
                      early_stopping_rounds=100, verbose_eval=False)
        return b, (0, b.best_iteration + 1)

    P = np.zeros(len(y))
    for a, b in ((trA, trB), (trB, trA)):
        m, it = fit(X2, a)
        P[b] = m.inplace_predict(X2[b], iteration_range=it)
    m, it = fit(X2, tr)
    P[~tr] = m.inplace_predict(X2[~tr], iteration_range=it)
    log('L2 out-of-fold probabilities done')

    text, c1, c23 = read_sources(RAW, 'train', set(refs) | set(tgts))
    log('text + name-key counts', len(text), len(c1), len(c23))
    by_ref = defaultdict(list)
    for j, r in enumerate(refs):
        by_ref[r].append(j)
    F = np.zeros((len(y), len(NAMES)), dtype=np.float32)
    ords = {}
    for r, js in by_ref.items():
        js = np.array(js)
        o = js[np.argsort(-P[js], kind='stable')]
        ords[r] = o
        F[o] = list_features(P[o], s1f[o], ce[o], list(tgts[o]), r, text, c1, c23)
    log('features built')
    np.save(OUT / 'feature_names.npy', np.array(NAMES))

    idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
    prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
    rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}

    def macro_sel(p, chosen):
        pred = defaultdict(set)
        for i, tt in zip(rix[p][chosen], tgts[idx[p]][chosen]):
            pred[i].add(tt)
        return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

    grid = [(ru, round(float(t), 2)) for ru in ('expected', 'threshold') for t in np.arange(0.3, 0.981, 0.02)]

    def by_rule(name, prob):
        dev = {g: macro_sel('dev', decide(rix['dev'], prob[idx['dev']], *g, len(prefs['dev']))) for g in grid}
        g = max(dev, key=lambda k: (dev[k], k[1]))
        R[name] = {'rule': g, 'dev': round(dev[g], 6),
                   'holdout': round(macro_sel('holdout', decide(rix['holdout'], prob[idx['holdout']], *g,
                                                              len(prefs['holdout']))), 6)}
        log(name, R[name])
        (FUS / 'boundary_exp.json').write_text(json.dumps(R, indent=1))

    R = {'features': NAMES, 'baseline_S5': 'threshold@0.68 dev 0.98772 / holdout 0.98622'}
    by_rule('L2_oof_recheck', P)

    Xs = F.astype(np.float32)
    m3, it3 = fit(Xs, tr)
    P3 = np.asarray(m3.inplace_predict(Xs, iteration_range=it3), dtype=np.float64)
    m3.save_model(str(OUT / 'stage3.json'))
    json.dump({'best_iteration': int(m3.best_iteration)}, open(OUT / 'stage3_meta.json', 'w'))
    by_rule('stage3', P3)
    g = m3.get_score(importance_type='total_gain')
    R['stage3_gain'] = {NAMES[int(k[1:])]: round(v) for k, v in sorted(g.items(), key=lambda kv: -kv[1])[:15]}

    # ---- cut transformer ----
    dev_t = torch.device('cuda')
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    np.savez(OUT / 'cut_norm.npz', mu=mu, sd=sd)
    keys = {'train': sorted(set(refs[tr])), 'dev': sorted(set(refs[part == 'dev'])),
            'holdout': sorted(set(refs[part == 'holdout']))}
    # refs with no candidate rows still count in the macro (1 if singleton else 0)
    extra = {p: sum(f05(gold[s], set()) for s in prefs[p] if s not in by_ref) for p in ('dev', 'holdout')}

    def cut_macro(p, logits):
        return (float(D[p][2].gather(1, logits.argmax(1, keepdim=True)).sum()) + extra[p]) / len(prefs[p])

    def pack(rs):
        X = np.zeros((len(rs), 10, len(NAMES)), dtype=np.float32)
        M = np.zeros((len(rs), 10), dtype=bool)
        T = np.zeros((len(rs), 11), dtype=np.float32)
        for i, r in enumerate(rs):
            o = ords[r][:10]
            X[i, :len(o)] = (F[o] - mu) / sd
            M[i, :len(o)] = True
            g, tp = gold[r], 0
            T[i, 0] = 1.0 if not g else 0.0
            for k, j in enumerate(o, 1):
                tp += tgts[j] in g
                T[i, k] = 1.25 * tp / (0.25 * len(g) + k) if g else 0.0
        return [torch.tensor(a, device=dev_t) for a in (X, M, T)]

    D = {p: pack(keys[p]) for p in keys}
    for p in ('dev', 'holdout'):
        R[f'cut_oracle_{p}'] = round((float(D[p][2].max(1).values.sum()) + extra[p]) / len(prefs[p]), 6)

    def run_seed(seed):
        torch.manual_seed(seed)
        net = Cut(len(NAMES)).to(dev_t)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.01)
        X, M, T = D['train']
        best, state = -1, None
        for ep in range(40):
            net.train()
            perm = torch.randperm(len(X), device=dev_t)
            for b in range(0, len(X), 256):
                ib = perm[b:b + 256]
                loss = -(torch.softmax(net(X[ib], M[ib]), 1) * T[ib]).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                dv_ = cut_macro('dev', net(*D['dev'][:2]))
            if dv_ > best:
                best, state = dv_, {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(state)
        torch.save(state, OUT / f'cut_seed{seed}.pt')
        return net, best

    nets = [run_seed(s) for s in range(3)]
    R['cut_single_dev'] = [round(b, 6) for _, b in nets]
    with torch.no_grad():
        for p in ('dev', 'holdout'):
            pr = sum(torch.softmax(n_(*D[p][:2]), 1) for n_, _ in nets)
            R[f'cut_ens_{p}'] = round(cut_macro(p, pr), 6)
    log('cut', {k: v for k, v in R.items() if k.startswith('cut')})
    (FUS / 'boundary_exp.json').write_text(json.dumps(R, indent=1))
    log('done')


if __name__ == '__main__':
    main()
