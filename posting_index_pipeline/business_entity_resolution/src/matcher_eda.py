import csv
import gzip
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from submission_pipeline import decide, f05

HOME = Path.home()
V4 = HOME / 'scratch/vishal_v4_artifacts'
VISHAL = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
OUT = HOME / 'scratch/amazon_ml_2026/artifacts/matcher_eda'
SET = sys.argv[1] if __name__ == '__main__' and len(sys.argv) > 1 else 'top10'
NTHREAD = int(sys.argv[2]) if __name__ == '__main__' and len(sys.argv) > 2 else 8

KEY_FEATURES = ['name_folded_token_sort', 'name_folded_jaro_winkler', 'name_exact', 'name_core_token_set',
                'name_translit_ratio', 'address_folded_token_set', 'address_exact', 'address_numbers_token_set',
                'address_numbers_ref_empty', 'address_numbers_target_empty', 'address_folded_target_empty',
                'postal_candidates_token_set', 'emb_cos', 'emb_rank', 'dense_score', 'fused_rank', 'rev_rank',
                'route_hits', 'strong_candidates', 'proxy', 'target_source3', 'ref_name_non_ascii',
                'target_name_non_ascii']


def log(*a):
    print(*a, flush=True)


def partition_of(gold):
    import hashlib
    parent = {s: s for s in gold}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    owner = {}
    for s, targets in gold.items():
        for t in targets:
            if t in owner:
                parent[find(s)] = find(owner[t])
            owner[t] = s

    def part(s):
        h = int(hashlib.sha256(find(s).encode()).hexdigest()[:8], 16) % 100
        return 'train' if h < 70 else 'dev' if h < 85 else 'holdout'
    return {s: part(s) for s in gold}


def norm(s):
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r'[^\w]+', ' ', s).strip()


def non_ascii_frac(s):
    return sum(ord(c) > 127 for c in s) / max(len(s), 1)


def f05_parts(gold_set, pred_set):
    tp = len(gold_set & pred_set)
    return tp, len(pred_set) - tp, len(gold_set) - tp


def quantiles(a):
    a = np.asarray(a, dtype=np.float64)
    if not len(a):
        return None
    q = np.percentile(a, [10, 25, 50, 75, 90])
    return {'n': int(len(a)), 'mean': round(float(a.mean()), 4), 'p10': round(float(q[0]), 4),
            'p25': round(float(q[1]), 4), 'p50': round(float(q[2]), 4), 'p75': round(float(q[3]), 4),
            'p90': round(float(q[4]), 4)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    masks = {'top10': 'mask_top10.npy', 'top20': 'mask_top20.npy',
             'adaptive': 'mask_adaptive_t0.005_min3_max15.npy'}
    models = {'top10': 'model_top10', 'top20': 'model_top20', 'adaptive': 'model_adaptive_t0.005_min3_max15'}
    model_dir = VISHAL / 'prune_v1' / models[SET]
    report = json.loads((model_dir / 'report.json').read_text())
    names = report['features']
    rule, thr = report['decision']['rule'], report['decision']['threshold']
    log('set', SET, 'rule', rule, thr)

    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    countries = json.loads((V4 / 'pairs/countries.json').read_text())
    partition = partition_of(gold)
    mask = np.load(VISHAL / 'prune_v1' / masks[SET])
    stage1 = np.load(VISHAL / 'prune_v1/stage1_scores.npy', mmap_mode='r')

    rows, refs, tgts = [], [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if not mask[i]:
                continue
            r, t = line.rstrip('\n').split('\t')
            if partition[r] != 'holdout':
                continue
            rows.append(i)
            refs.append(r)
            tgts.append(t)
    rows = np.array(rows)
    refs = np.array(refs, dtype=object)
    tgts = np.array(tgts, dtype=object)
    log('holdout rows', len(rows))

    x_all = np.load(V4 / 'pairs/x.npy', mmap_mode='r')
    x = np.asarray(x_all[rows])
    y = np.load(V4 / 'pairs/y.npy')[rows]
    s1 = np.asarray(stage1[rows], dtype=np.float64)
    booster = xgb.Booster()
    booster.load_model(str(model_dir / 'model.json'))
    booster.set_param({'device': 'cpu', 'nthread': NTHREAD})
    prob = np.asarray(booster.inplace_predict(x, iteration_range=(0, report['best_iteration'] + 1)),
                      dtype=np.float64)
    log('predicted')

    hold_refs = sorted(s for s in gold if partition[s] == 'holdout')
    pos = {s: i for i, s in enumerate(hold_refs)}
    ref_idx = np.array([pos[s] for s in refs])
    chosen = decide(ref_idx, prob, rule, thr, len(hold_refs))
    fi = {n: i for i, n in enumerate(names)}

    by_ref = defaultdict(list)
    for j in range(len(rows)):
        by_ref[refs[j]].append(j)

    E = {'set': SET, 'decision': {'rule': rule, 'threshold': thr}, 'n_refs': len(hold_refs),
         'n_pairs': int(len(rows)), 'reported_holdout_f05': report.get('macro_f05_holdout')}

    # ---------- per-reference outcome ----------
    ref_rows = []
    oracle_k, oracle_card, perfect = [], [], []
    for s in hold_refs:
        g = gold[s]
        js = by_ref.get(s, [])
        cand = {tgts[j] for j in js}
        pred = {tgts[j] for j in js if chosen[j]}
        tp, fp, fn = f05_parts(g, pred)
        blk_miss = len(g - cand)
        score = f05(g, pred)
        order = sorted(js, key=lambda j: -prob[j])
        best = 1.0 if not g else 0.0
        for k in range(1, len(order) + 1):
            best = max(best, f05(g, {tgts[j] for j in order[:k]}))
        oracle_k.append(best)
        oracle_card.append(f05(g, {tgts[j] for j in order[:len(g)]}) if g else 1.0)
        perfect.append(f05(g, g & cand))
        if score == 1.0:
            kind = 'perfect'
        elif not g:
            kind = 'singleton_false_match'
        elif fp and (fn - blk_miss) > 0:
            kind = 'mixed'
        elif fp:
            kind = 'false_positive_only'
        elif fn - blk_miss > 0:
            kind = 'missed_candidate_only'
        else:
            kind = 'blocking_only'
        ref_rows.append({'ref': s, 'country': countries.get(s, 'UNK'), 'gold': len(g), 'pred': len(pred),
                         'tp': tp, 'fp': fp, 'fn': fn, 'blk_miss': blk_miss, 'score': score, 'kind': kind,
                         'top_prob': float(prob[order[0]]) if order else 0.0})
    R = pd.DataFrame(ref_rows)
    n = len(R)
    E['macro_f05'] = float(R.score.mean())
    E['ceilings'] = {'current': float(R.score.mean()), 'oracle_topk_given_ranking': float(np.mean(oracle_k)),
                     'oracle_cardinality_known': float(np.mean(oracle_card)),
                     'perfect_matcher_on_candidates': float(np.mean(perfect))}
    R['loss'] = 1 - R.score
    E['loss_by_kind'] = {k: {'refs': int((R.kind == k).sum()), 'loss_points': round(float(v.sum() / n * 100), 4)}
                         for k, v in R.groupby('kind').loss}
    E['loss_by_country'] = {c: {'refs': int(len(d)), 'f05': round(float(d.score.mean()), 5),
                                'loss_points_share': round(float(d.loss.sum() / n * 100), 4)}
                            for c, d in R.groupby('country')}
    E['loss_by_gold_count'] = {int(k): {'refs': int(len(d)), 'f05': round(float(d.score.mean()), 5),
                                        'loss_points': round(float(d.loss.sum() / n * 100), 4)}
                               for k, d in R.groupby('gold')}
    E['pred_minus_gold'] = {int(k): int(v) for k, v in sorted(Counter((R.pred - R.gold).clip(-5, 5)).items())}
    E['avg_pred_per_ref'] = float(R.pred.mean())
    E['avg_gold_per_ref'] = float(R.gold.mean())
    E['singleton_rate'] = float((R.gold == 0).mean())
    E['singleton_top_prob'] = quantiles(R.top_prob[R.gold == 0])
    E['singleton_predicted_nonempty'] = int(((R.gold == 0) & (R.pred > 0)).sum())
    E['totals'] = {'tp': int(R.tp.sum()), 'fp': int(R.fp.sum()), 'fn': int(R.fn.sum()),
                   'blocking_missed': int(R.blk_miss.sum()), 'gold_links': int(R.gold.sum())}

    # ---------- pair buckets ----------
    ref_has_tp = {r.ref: r.tp > 0 for r in R.itertuples()}
    bucket = np.full(len(rows), 'TN', dtype=object)
    bucket[chosen & (y == 1)] = 'TP'
    bucket[chosen & (y == 0)] = 'FP'
    bucket[~chosen & (y == 1)] = 'FN'
    hard = (~chosen) & (y == 0) & (prob >= 0.1)
    bucket[hard] = 'HARD_NEG'
    E['bucket_counts'] = {k: int(v) for k, v in Counter(bucket).items()}

    bins = np.linspace(0, 1, 11)
    E['prob_hist'] = {lab: np.histogram(prob[y == v], bins=bins)[0].tolist() for lab, v in (('pos', 1), ('neg', 0))}
    rel = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (prob >= a) & (prob < b if b < 1 else prob <= b)
        if m.sum():
            rel.append({'bin': f'{a:.1f}-{b:.1f}', 'n': int(m.sum()), 'mean_pred': round(float(prob[m].mean()), 4),
                        'frac_pos': round(float(y[m].mean()), 4)})
    E['reliability'] = rel
    E['ece'] = round(float(sum(r['n'] * abs(r['mean_pred'] - r['frac_pos']) for r in rel) / len(prob)), 5)
    E['fn_prob'] = quantiles(prob[bucket == 'FN'])
    E['fp_prob'] = quantiles(prob[bucket == 'FP'])
    fn_idx = np.flatnonzero(bucket == 'FN')
    E['fn_in_ref_with_accepted_tp'] = int(sum(ref_has_tp[refs[j]] for j in fn_idx))
    rank_in_ref = np.zeros(len(rows), dtype=int)
    for s, js in by_ref.items():
        for rk, j in enumerate(sorted(js, key=lambda j: -prob[j]), 1):
            rank_in_ref[j] = rk
    E['fn_rank_in_ref'] = {int(k): int(v) for k, v in sorted(Counter(rank_in_ref[fn_idx]).items())}
    E['fp_rank_in_ref'] = {int(k): int(v) for k, v in sorted(Counter(rank_in_ref[bucket == 'FP']).items())}
    E['error_by_source'] = {b: {'S2': int(((bucket == b) & (x[:, fi['target_source3']] == 0)).sum()),
                                'S3': int(((bucket == b) & (x[:, fi['target_source3']] == 1)).sum())}
                            for b in ('TP', 'FP', 'FN', 'HARD_NEG')}
    E['feature_by_bucket'] = {f: {b: quantiles(x[bucket == b, fi[f]]) for b in ('TP', 'FN', 'FP', 'HARD_NEG')}
                              for f in KEY_FEATURES if f in fi}
    E['feature_by_bucket']['stage1_score'] = {b: quantiles(s1[bucket == b]) for b in ('TP', 'FN', 'FP', 'HARD_NEG')}
    E['feature_by_bucket']['matcher_prob'] = {b: quantiles(prob[bucket == b]) for b in ('TP', 'FN', 'FP', 'HARD_NEG')}
    gains = report.get('feature_gain', {})
    E['top_gain'] = sorted(gains.items(), key=lambda kv: -kv[1])[:25]
    E['zero_gain_features'] = [k for k, v in gains.items() if v == 0]

    # ---------- raw text join ----------
    need_t = set(tgts)
    need_r = set(hold_refs)
    owner = {}
    with open(RAW / 'train_ground_truth.tsv', newline='') as f:
        rd = csv.reader(f, delimiter='\t')
        next(rd)
        for row in rd:
            if len(row) < 2 or not row[1]:
                continue
            for t in row[1].split(','):
                t = t.strip()
                if t in need_t:
                    owner[t] = row[0]
    need_r |= set(owner.values())
    text = {}
    s1_name_counts = Counter()
    for src in ('train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv'):
        df = pd.read_csv(RAW / src, sep='\t', dtype=str, keep_default_na=False, quoting=3)
        df.columns = [c.strip() for c in df.columns]
        if src == 'train_source1.tsv':
            s1_name_counts = Counter(norm(v) for v in df['business_name'])
        keep = df['entity_id'].isin(need_r if src == 'train_source1.tsv' else need_t)
        for e, nm, ad in zip(df.entity_id[keep], df.business_name[keep], df.business_address[keep]):
            text[e] = (nm, ad)
        log('loaded', src, int(keep.sum()))
        del df

    def pair_patterns(r, t):
        rn, ra = text.get(r, ('', ''))
        tn, ta = text.get(t, ('', ''))
        nrn, ntn, nra, nta = norm(rn), norm(tn), norm(ra), norm(ta)
        rnum, tnum = set(re.findall(r'\d+', ra)), set(re.findall(r'\d+', ta))
        rtok, ttok = set(nrn.split()), set(ntn.split())
        return {
            'name_equal': nrn == ntn and bool(nrn),
            'name_subset': bool(rtok) and bool(ttok) and (rtok <= ttok or ttok <= rtok) and nrn != ntn,
            'name_sim': fuzz.token_sort_ratio(nrn, ntn) / 100,
            'addr_sim': fuzz.token_set_ratio(nra, nta) / 100,
            'addr_empty_either': not nra or not nta,
            'numbers_disjoint': bool(rnum) and bool(tnum) and not (rnum & tnum),
            'numbers_equal': bool(rnum) and rnum == tnum,
            'cross_script': abs(non_ascii_frac(rn) - non_ascii_frac(tn)) > 0.3,
            'same_name_diff_number': nrn == ntn and bool(rnum) and bool(tnum) and not (rnum & tnum),
            'chain_size': s1_name_counts.get(nrn, 0),
        }

    rng = np.random.default_rng(7)
    pat = {}
    samples = {}
    for b in ('TP', 'FN', 'FP', 'HARD_NEG'):
        idx = np.flatnonzero(bucket == b)
        if len(idx) > 6000:
            idx = rng.choice(idx, 6000, replace=False)
        P = pd.DataFrame([pair_patterns(refs[j], tgts[j]) for j in idx])
        pat[b] = {c: round(float(P[c].mean()), 4) for c in P.columns if c != 'chain_size'}
        pat[b]['chain_size_median'] = float(P.chain_size.median())
        pat[b]['chain_size_gt1'] = round(float((P.chain_size > 1).mean()), 4)
        pat[b]['chain_size_gt10'] = round(float((P.chain_size > 10).mean()), 4)
        pat[b]['n'] = int(len(P))
        pick = rng.choice(idx, min(30, len(idx)), replace=False)
        samples[b] = [{'ref_name': text.get(refs[j], ('', ''))[0], 'ref_addr': text.get(refs[j], ('', ''))[1],
                       'tgt_name': text.get(tgts[j], ('', ''))[0], 'tgt_addr': text.get(tgts[j], ('', ''))[1],
                       'country': countries.get(refs[j], ''), 'prob': round(float(prob[j]), 3),
                       'rank': int(rank_in_ref[j]), 'src': 'S3' if x[j, fi['target_source3']] else 'S2',
                       'owner_is_other_ref': b in ('FP', 'HARD_NEG') and tgts[j] in owner,
                       'owner_name': text.get(owner.get(tgts[j], ''), ('', ''))[0] if b in ('FP', 'HARD_NEG') else ''}
                      for j in pick]
    E['text_patterns'] = pat
    E['samples'] = samples

    fp_idx = np.flatnonzero(bucket == 'FP')
    owned = [tgts[j] in owner for j in fp_idx]
    E['fp_target_owned_by_other_ref'] = round(float(np.mean(owned)), 4) if len(fp_idx) else None
    osim = [fuzz.token_sort_ratio(norm(text.get(refs[j], ('', ''))[0]), norm(text.get(owner[tgts[j]], ('', ''))[0])) / 100
            for j in fp_idx if tgts[j] in owner]
    E['fp_ref_vs_true_owner_name_sim'] = quantiles(osim)

    # chain-size loss analysis
    R['chain'] = [s1_name_counts.get(norm(text.get(s, ('', ''))[0]), 0) for s in R.ref]
    cb = pd.cut(R.chain, [-1, 1, 5, 50, 10**9], labels=['1', '2-5', '6-50', '50+'])
    E['loss_by_chain_size'] = {str(k): {'refs': int(len(d)), 'f05': round(float(d.score.mean()), 5),
                                        'loss_points': round(float(d.loss.sum() / n * 100), 4),
                                        'fp_per_ref': round(float(d.fp.mean()), 4)}
                               for k, d in R.groupby(cb, observed=True)}

    # cluster consistency: missed true vs accepted true of same ref
    sims = []
    for j in fn_idx:
        acc = [k for k in by_ref[refs[j]] if bucket[k] == 'TP']
        if not acc:
            continue
        tn = norm(text.get(tgts[j], ('', ''))[0])
        sims.append(max(fuzz.token_sort_ratio(tn, norm(text.get(tgts[k], ('', ''))[0])) / 100 for k in acc))
    E['fn_vs_accepted_sibling_name_sim'] = quantiles(sims)
    fps = []
    for j in fp_idx:
        acc = [k for k in by_ref[refs[j]] if bucket[k] == 'TP']
        if not acc:
            continue
        tn = norm(text.get(tgts[j], ('', ''))[0])
        fps.append(max(fuzz.token_sort_ratio(tn, norm(text.get(tgts[k], ('', ''))[0])) / 100 for k in acc))
    E['fp_vs_accepted_sibling_name_sim'] = quantiles(fps)

    (OUT / f'eda_{SET}.json').write_text(json.dumps(E, indent=1, default=str, ensure_ascii=False))
    log('wrote', OUT / f'eda_{SET}.json')


if __name__ == '__main__':
    main()
