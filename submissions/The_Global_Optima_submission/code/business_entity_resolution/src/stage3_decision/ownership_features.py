"""Ownership features: who else could own this candidate record?

S5's remaining loss is mostly the stop decision on blank-address candidates: a record with only a name may be this
reference's copy, the copy of another business with the same or a similar name, or a decoy. Exact name-key counts miss
the dataset's name noise (typos, case, punctuation, dropped/added words), so this compares the candidate's folded name
with every S1 reference that could own it: its reverse top-10 (reverse_index.py, searched over the split's full Source 1)
plus the reference of the pair. Features per (reference, candidate) pair, label-free and computed the same way on train
and test:

  own_in_rev          reference is in the candidate's reverse top-10
  own_rev_n           size of that reverse list
  own_sim_sort/ratio/set   name similarity reference vs candidate (token_sort, char ratio, token_set; 0-100)
  own_best_other_sort/ratio   best similarity of any other reference in the reverse list
  own_margin_sort/ratio       own minus best other (> 0: this reference is the closest name)
  own_n95_sort, own_n90_sort, own_n90_set   references (reverse list + this one) with a near-identical name
  own_is_best         this reference has the highest char ratio (ties count as best)
  own_n_tied_best     references within 1 point of the best char ratio (ambiguity)
  own_t_addr_blank    candidate has no address

Address block (own_addr_{stage}.npy; for alias names at a shared address, e.g. "Jaxkorpyrax" at the reference's
address): the same owner set, compared by folded address. NaN when the candidate or reference address is blank.
  own_addr_set/ratio        address similarity reference vs candidate (token_set, char ratio)
  own_best_other_addr       best token_set address similarity of any other possible owner
  own_addr_margin           own minus best other (> 0: this reference is the closest address)
  own_n_addr85              possible owners at (almost) the same address (token_set >= 85; st vs street scores ~90)
  own_addr_is_best          this reference has the best address similarity (ties count as best)
  own_name_tok_overlap      share of the candidate's name words found in the reference name (0 = pure alias)

Raw-name block (own_raw_{stage}.npy): folded names drop case, punctuation and spacing, but a noisy copy tends to keep
its own parent's formatting ("Leandra J. Seng, D.D.S. LLC" vs "Leandra J Seng DDS LLC"), which can break same-name ties.
  own_raw_ratio             char ratio of the raw names (case, punctuation, spacing kept)
  own_raw_margin            own minus best other possible owner
  own_raw_is_best           this reference has the best raw ratio (ties count as best)
  own_raw_n_tied            possible owners within 1 point of the best raw ratio

  sample: ownership_features.py sample OUT      (1M top-10 rows of the 100k train sample, mask_top10 order)
  test:   ownership_features.py test OUT        (17.3M rows of submission_v5_top10/candidate_pairs.tsv, file order)
  eval:   ownership_features.py eval OUT        (S5 late fusion with / without these features, S5 protocol)
"""
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
AY = Path('/scratch/scai/mtech/aib262015/amazon_ml_2026')
PIDX = ME / 'artifacts/posting_index_v1'
V4 = Path('/scratch/scai/mtech/aib262015/vishal_v4_artifacts')
PRUNE = AY / 'artifacts/prune_v1_copy'
CANDS = ME / 'artifacts/submission_v5_top10/candidate_pairs.tsv'
NAMES = ['own_in_rev', 'own_rev_n', 'own_sim_sort', 'own_sim_ratio', 'own_sim_set', 'own_best_other_sort',
         'own_best_other_ratio', 'own_margin_sort', 'own_margin_ratio', 'own_n95_sort', 'own_n90_sort',
         'own_n90_set', 'own_is_best', 'own_n_tied_best', 'own_t_addr_blank']
ADDR_NAMES = ['own_addr_set', 'own_addr_ratio', 'own_best_other_addr', 'own_addr_margin', 'own_n_addr85',
              'own_addr_is_best', 'own_name_tok_overlap']
RAW_NAMES = ['own_raw_ratio', 'own_raw_margin', 'own_raw_is_best', 'own_raw_n_tied']
G = {}  # shared with forked workers


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def index_side(directory):
    """Entity ids, folded names, folded addresses and blank-address flags of one country of a posting index."""
    ids = [i.decode('utf-8') for i in np.load(directory / 'ids.npy')]
    names, raws, addrs, blank = [], [], [], []
    with gzip.open(directory / 'targets.tsv.gz', 'rt', encoding='utf-8', newline='') as f:
        for k, row in enumerate(csv.DictReader(f, delimiter='\t')):
            if k < 5:
                assert row['entity_id'] == ids[k], (directory, k)
            names.append(row['name_folded'] or '')
            raws.append((row['business_name'] or '').strip())
            addrs.append((row['address_folded'] or '').strip())
            blank.append(not addrs[-1])
    assert len(names) == len(ids), directory
    return ids, names, raws, addrs, np.array(blank)


def work(chunk):
    tn, rn, t_blank, ta, ra = G['t_names'], G['r_names'], G['t_blank'], G['t_addrs'], G['r_addrs']
    tr_, rr_ = G['t_raws'], G['r_raws']
    rev_start, rev_r = G['rev_start'], G['rev_r']
    ut, row_start, row_ref, row_idx = G['ut'], G['row_start'], G['row_ref'], G['row_idx']
    out_idx, out, out_a, out_r = [], [], [], []
    nan = float('nan')
    for u in chunk:
        t = ut[u]
        name = tn[t]
        rev = rev_r[rev_start[t]:rev_start[t + 1]]
        refs = np.unique(np.concatenate([rev, row_ref[row_start[u]:row_start[u + 1]]]))
        ratio = np.array([fuzz.ratio(name, rn[r]) for r in refs])
        srt = np.array([fuzz.token_sort_ratio(name, rn[r]) for r in refs])
        tset = np.array([fuzz.token_set_ratio(name, rn[r]) for r in refs])
        in_rev = set(rev.tolist())
        top = ratio.max()
        n95, n90, n90s, tied = (srt >= 95).sum(), (srt >= 90).sum(), (tset >= 90).sum(), (ratio >= top - 1).sum()
        addr = ta[t]
        if addr:
            aset = np.array([fuzz.token_set_ratio(addr, ra[r]) if ra[r] else nan for r in refs])
            arat = np.array([fuzz.ratio(addr, ra[r]) if ra[r] else nan for r in refs])
            n_a90 = float(np.nansum(aset >= 85))
        t_tok = set(name.split())
        raw = np.array([fuzz.ratio(tr_[t], rr_[r]) for r in refs])
        raw_top = raw.max()
        raw_tied = float((raw >= raw_top - 1).sum())
        for k in range(row_start[u], row_start[u + 1]):
            j = int(np.searchsorted(refs, row_ref[k]))
            if len(refs) > 1:
                other = np.ones(len(refs), dtype=bool)
                other[j] = False
                bs, br = srt[other].max(), ratio[other].max()
            else:
                bs = br = 0.0
            out.append([float(row_ref[k] in in_rev), len(rev), srt[j], ratio[j], tset[j], bs, br, srt[j] - bs,
                        ratio[j] - br, n95, n90, n90s, float(ratio[j] >= br), tied, float(t_blank[t])])
            overlap = len(t_tok & set(rn[row_ref[k]].split())) / max(1, len(t_tok))
            if addr:
                oth = np.delete(aset, j)
                oth = oth[~np.isnan(oth)]
                bo = float(oth.max()) if len(oth) else nan
                mine = aset[j]
                is_best = nan if np.isnan(mine) else float(np.isnan(bo) or mine >= bo)
                out_a.append([mine, arat[j], bo, mine - bo, n_a90, is_best, overlap])
            else:
                out_a.append([nan, nan, nan, nan, nan, nan, overlap])
            ob = float(np.delete(raw, j).max()) if len(refs) > 1 else 0.0
            out_r.append([raw[j], raw[j] - ob, float(raw[j] >= ob), raw_tied])
            out_idx.append(row_idx[k])
    return (np.array(out_idx, dtype=np.int64), np.array(out, dtype=np.float32).reshape(-1, len(NAMES)),
            np.array(out_a, dtype=np.float32).reshape(-1, len(ADDR_NAMES)),
            np.array(out_r, dtype=np.float32).reshape(-1, len(RAW_NAMES)))


def features(split, rows_ref, rows_tgt, workers):
    feats = np.full((len(rows_ref), len(NAMES)), np.nan, dtype=np.float32)
    afeats = np.full((len(rows_ref), len(ADDR_NAMES)), np.nan, dtype=np.float32)
    rfeats = np.full((len(rows_ref), len(RAW_NAMES)), np.nan, dtype=np.float32)
    ref_root, tgt_root, rev_root = PIDX / f'{split}_ref_index', PIDX / f'{split}_index_c', PIDX / f'{split}_reverse_t'
    manifest = json.loads((ref_root / 'manifest.json').read_text())
    report = {}
    for country, entry in sorted(manifest['countries'].items()):
        started = time.monotonic()
        d = entry['dir']
        r_ids, r_names, r_raws, r_addrs, _ = index_side(ref_root / d)
        t_ids, t_names, t_raws, t_addrs, t_blank = index_side(tgt_root / d)
        rpos = {x: i for i, x in enumerate(r_ids)}
        tpos = {x: i for i, x in enumerate(t_ids)}
        ri = np.fromiter((rpos.get(r, -1) for r in rows_ref), dtype=np.int64, count=len(rows_ref))
        sel = np.flatnonzero(ri >= 0)
        ti = np.fromiter((tpos.get(rows_tgt[i], -1) for i in sel), dtype=np.int64, count=len(sel))
        keep = ti >= 0
        sel, ti, ri = sel[keep], ti[keep], ri[sel][keep]
        z = np.load(rev_root / d / 'reverse.npz')
        # The reverse search ran against an earlier build of the target index; its ordinals must match this one.
        rev_report = json.loads((rev_root / 'report.json').read_text())['countries'][country]
        assert rev_report['targets'] == len(t_ids) and int(z['t_ord'].max()) < len(t_ids), (country, rev_report, len(t_ids))
        assert int(z['r_ord'].max()) < len(r_ids), (country, len(r_ids))
        order = np.argsort(z['t_ord'], kind='stable')
        rev_t = z['t_ord'][order].astype(np.int64)
        G.update(t_names=t_names, r_names=r_names, t_blank=t_blank, t_addrs=t_addrs, r_addrs=r_addrs, t_raws=t_raws, r_raws=r_raws, rev_r=z['r_ord'][order].astype(np.int64),
                 rev_start=np.searchsorted(rev_t, np.arange(len(t_ids) + 1)))
        ut, inv = np.unique(ti, return_inverse=True)
        o = np.argsort(inv, kind='stable')
        G.update(ut=ut, row_start=np.searchsorted(inv[o], np.arange(len(ut) + 1)), row_ref=ri[o], row_idx=sel[o])
        log(country, 'rows', len(sel), 'targets', len(ut), 'reverse pairs', len(rev_t), f'{time.monotonic() - started:.0f}s')
        chunks = [np.arange(s, min(s + 20000, len(ut))) for s in range(0, len(ut), 20000)]
        if workers <= 1:  # also the path for local tests (Windows spawns workers without G)
            results = map(work, chunks)
        else:
            pool = Pool(workers)
            results = pool.imap_unordered(work, chunks)
        for idx, f, fa, fr in results:
            feats[idx] = f
            afeats[idx] = fa
            rfeats[idx] = fr
        if workers > 1:
            pool.close()
            pool.join()
        report[country] = {'rows': int(len(sel)), 'targets': int(len(ut)), 'seconds': round(time.monotonic() - started)}
        log(country, 'done', report[country])
        G.clear()
    return feats, afeats, rfeats, report


def sample_rows():
    mask = np.load(PRUNE / 'mask_top10.npy')
    refs, tgts = [], []
    with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t')
                refs.append(r); tgts.append(t)
    return refs, tgts


def test_rows():
    refs, tgts = [], []
    with open(CANDS, encoding='utf-8') as f:
        next(f)
        for line in f:
            s, c = line.rstrip('\n').split('\t')
            for t in (c.split(',') if c else []):
                refs.append(s); tgts.append(t)
    return refs, tgts


def evaluate(out):
    """S5 late fusion (v5 prob + mdeberta_full logit, decoy w3) with and without the ownership features."""
    import xgboost as xgb
    from collections import defaultdict
    sys.path[:0] = [str(ME / 'business_entity_resolution/src'), str(AY / 'business_entity_resolution/src')]
    from distractor_check import decide, f05, partition_of
    gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
    cp = partition_of(gold)
    refs, tgts = sample_rows()
    refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
    part = np.array([cp[r] for r in refs], dtype=object)
    S = np.load(AY / 'artifacts/fusion/stage1.npz')
    y = np.load(V4 / 'pairs/y.npy')[S['rows']].astype(np.float32)
    s1 = S['stage1'].astype(np.float64)
    s1f = np.where(s1 < 0.02, 0.0, s1)
    ce = np.load(ME / 'artifacts/mdeberta_full/ce_logit_sample.npy').astype(np.float64)
    own = np.load(out / 'own_sample.npy')
    own_a = np.load(out / 'own_addr_sample.npy')
    own_r = np.load(out / 'own_raw_sample.npy')
    owned = set()
    with open(ME / 'data/dataset/train/train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if len(q) > 1 and q[1]:
                owned.update(t.strip() for t in q[1].split(','))
    wt = np.where((y == 0) & np.array([t not in owned for t in tgts]), 3.0, 1.0).astype(np.float32)
    tr, dv = np.isin(part, ['trainA', 'trainB']), part == 'dev'
    idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
    prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
    rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}
    grid = [(ru, round(float(t), 3)) for ru in ('expected', 'threshold') for t in np.arange(0.5, 0.981, 0.02)]
    params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                  max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)

    def macro(p, prob, ru, t):
        c = decide(rix[p], prob, ru, t, len(prefs[p]))
        pred = defaultdict(set)
        for i, tt in zip(rix[p][c], tgts[idx[p]][c]):
            pred[i].add(tt)
        return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

    R = {'coverage_rows_with_features': float((~np.isnan(own[:, 0])).mean())}
    blank = own[:, -1] == 1
    for name, X in (('S5_L2', np.column_stack([s1f, ce])), ('S5_L2+ownership', np.column_stack([s1f, ce, own])),
                    ('S5_L2+ownership+addr', np.column_stack([s1f, ce, own, own_a])),
                    ('S5_L2+ownership+addr+raw', np.column_stack([s1f, ce, own, own_a, own_r]))):
        X = X.astype(np.float32)
        dtr = xgb.QuantileDMatrix(X[tr], y[tr], weight=wt[tr])
        bst = xgb.train(params, dtr, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=dtr), 'dev')],
                        early_stopping_rounds=100, verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        pdv = np.asarray(bst.inplace_predict(X[idx['dev']], iteration_range=it), dtype=np.float64)
        pho = np.asarray(bst.inplace_predict(X[idx['holdout']], iteration_range=it), dtype=np.float64)
        dev = {g: macro('dev', pdv, *g) for g in grid}
        best = max(dev, key=lambda k: (dev[k], k[1]))
        R[name] = {'rule': list(best), 'dev': round(dev[best], 6), 'holdout': round(macro('holdout', pho, *best), 6),
                   'best_iteration': int(bst.best_iteration)}
        if name.endswith('raw'):
            g = bst.get_score(importance_type='total_gain')
            names = ['s1f', 'ce'] + NAMES + ADDR_NAMES + RAW_NAMES
            R['gain'] = {names[int(k[1:])]: round(v) for k, v in sorted(g.items(), key=lambda kv: -kv[1])}
        log(name, R[name])
    # How the uncertain blank-address rows behave (holdout, S5 L2 band 0.1-0.9 is not recomputed here: raw rates).
    hb = np.flatnonzero((part == 'holdout') & blank)
    for label, m in (('unique_best_name', (own[hb, 12] == 1) & (own[hb, 13] == 1)),
                     ('tied_best_name', own[hb, 13] >= 2), ('not_in_reverse', own[hb, 0] == 0)):
        R[f'holdout_blank_{label}'] = {'rows': int(m.sum()), 'match_rate': round(float(y[hb][m].mean()), 4) if m.any() else None}
    (out / 'own_eval.json').write_text(json.dumps(R, indent=1))
    log(json.dumps(R))


def main():
    stage, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    workers = int(os.environ.get('NCPUS', 4))
    if stage == 'eval':
        return evaluate(out)
    refs, tgts = sample_rows() if stage == 'sample' else test_rows()
    # Row-order fingerprint so consumers can verify alignment before joining by row:
    # sha256 over "ref<TAB>target<LF>" for every row in order.
    digest = hashlib.sha256()
    for r, t in zip(refs, tgts):
        digest.update(f'{r}\t{t}\n'.encode('utf-8'))
    log(stage, 'rows', len(refs), 'row sha256', digest.hexdigest())
    feats, afeats, rfeats, report = features('train' if stage == 'sample' else 'test', refs, tgts, workers)
    np.save(out / f'own_{stage}.npy', feats)
    np.save(out / f'own_addr_{stage}.npy', afeats)
    np.save(out / f'own_raw_{stage}.npy', rfeats)
    report.update(rows=len(refs), row_sha256=digest.hexdigest(), missing=int(np.isnan(feats[:, 0]).sum()), features=NAMES, addr_features=ADDR_NAMES, raw_features=RAW_NAMES,
                  row_order='mask_top10 rows of pairs.tsv.gz' if stage == 'sample' else 'candidate_pairs.tsv expanded')
    (out / f'own_{stage}.json').write_text(json.dumps(report, indent=1))
    log('saved', out / f'own_{stage}.npy', 'missing', report['missing'])


if __name__ == '__main__':
    main()
