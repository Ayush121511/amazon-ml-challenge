"""Multi-member late fusion, same protocol as late_full_train.py (train) + late_predict.py (test).

L2 = XGBoost on [v5 stage-1 probability floored at 0.02, one cross-encoder logit column per member] (a config
`mean:a+b` uses the members' mean logit as one column instead). Members must be out-of-sample on every 100k-sample row,
i.e. full-train models (train_ddp.py / train_chunk.py). Each config is trained on the trainA+trainB rows with decoy
negatives (targets owned by no train reference) weighted by w, early-stopped on dev; the rule is chosen on dev and
holdout is scored once.

  train:   MEMBERS='name=sample.npy;...' CONFIGS='full21;full21+decoyw2;mean:full21+decoyw2' WEIGHTS=2,3 late_multi.py train OUT
  predict: late_multi.py predict OUT CONFIG_DIR 'name=glob|glob;...'  -> OUT/<config>/{matching_results,candidate_pairs}.tsv

Paths are the same files the fusion scripts read (the v4 pairs, the prune_v1 mask and the stage-1 OOF file).
"""
import csv
import glob
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
AY = Path('/scratch/scai/mtech/aib262015/amazon_ml_2026')
sys.path[:0] = [str(ME / 'business_entity_resolution/src'), str(AY / 'business_entity_resolution/src')]
from distractor_check import decide as decide_eval, f05, partition_of  # noqa: E402  (evaluation helpers)
from submission_pipeline import finalize  # noqa: E402

V4 = Path('/scratch/scai/mtech/aib262015/vishal_v4_artifacts')
PRUNE = AY / 'artifacts/prune_v1_copy'
STAGE1 = AY / 'artifacts/fusion/stage1.npz'
RAW = ME / 'data/dataset'
CANDS = ME / 'artifacts/submission_v5_top10/candidate_pairs.tsv'
S1_TEST = [ME / 'artifacts/full_v5_s0/top10/scores.npz', ME / 'artifacts/full_v5_s1/top10/scores.npz']
PARAMS = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
              max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def parse_members(text):
    return dict(item.split('=', 1) for item in text.split(';') if item)


def columns(cfg, get):
    """cfg 'a+b' -> one column per member (a 2-D member such as the ownership features adds all its columns);
    'mean:a+b' -> one column with the mean logit; 'mean:a+b|own' -> that mean plus the members after '|'."""
    head, _, extra = cfg.partition('|')
    if head.startswith('mean:'):
        names = head[5:].split('+')
        cols = [np.mean([get(n) for n in names], axis=0)]
    else:
        names = head.split('+')
        cols = []
        for n in names:
            v = get(n)
            cols.extend(v.T if v.ndim == 2 else [v])
    for n in (extra.split('+') if extra else []):
        v = get(n)
        cols.extend(v.T if v.ndim == 2 else [v])
        names.append(n)
    return cols, names


def train(out):
    members = parse_members(os.environ['MEMBERS'])
    configs = [c for c in os.environ['CONFIGS'].split(';') if c]
    weights = [float(w) for w in os.environ.get('WEIGHTS', '2,3').split(',')]
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
    S = np.load(STAGE1)
    y = np.load(V4 / 'pairs/y.npy')[S['rows']].astype(np.float32)
    s1 = S['stage1'].astype(np.float64)
    s1f = np.where(s1 < 0.02, 0.0, s1)
    cache = {}

    def get(n):
        if n not in cache:
            v = np.load(members[n]).astype(np.float64)
            if v.ndim == 2:  # extra feature block (e.g. ownership features); NaN = missing, handled by XGBoost
                assert v.shape[0] == len(s1), (n, v.shape, s1.shape)
            else:
                assert v.shape == s1.shape and not np.isnan(v).any(), (n, v.shape, s1.shape)
            cache[n] = v
        return cache[n]

    owned = set()
    with open(RAW / 'train/train_ground_truth.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            q = line.rstrip('\n').split('\t')
            if len(q) > 1 and q[1]:
                owned.update(t.strip() for t in q[1].split(','))
    decoy = (y == 0) & np.array([t not in owned for t in tgts])
    tr = np.isin(part, ['trainA', 'trainB'])
    dv = part == 'dev'
    idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
    prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
    rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}
    grid = [(ru, round(float(t), 3)) for ru in ('expected', 'threshold') for t in np.arange(0.5, 0.981, 0.02)]

    def macro(p, prob, ru, t):
        c = decide_eval(rix[p], prob, ru, t, len(prefs[p]))
        pred = defaultdict(set)
        for i, tt in zip(rix[p][c], tgts[idx[p]][c]):
            pred[i].add(tt)
        return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

    report_path = out / 'late_multi.json'
    R = json.loads(report_path.read_text()) if report_path.exists() else {}
    R['protocol'] = 'L2 on trainA+trainB (all members full-train, out-of-sample), rule on dev, holdout once'
    R['members'] = members
    for cfg in configs:
        cols, names = columns(cfg, get)
        X = np.column_stack([s1f] + cols).astype(np.float32)
        for w in weights:
            wt = np.where(decoy, w, 1.0).astype(np.float32)
            dtr = xgb.QuantileDMatrix(X[tr], y[tr], weight=wt[tr])
            bst = xgb.train(PARAMS, dtr, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=dtr), 'dev')],
                            early_stopping_rounds=100, verbose_eval=False)
            it = (0, bst.best_iteration + 1)
            pdv = np.asarray(bst.inplace_predict(X[idx['dev']], iteration_range=it), dtype=np.float64)
            pho = np.asarray(bst.inplace_predict(X[idx['holdout']], iteration_range=it), dtype=np.float64)
            dev = {g: macro('dev', pdv, *g) for g in grid}
            best = max(dev, key=lambda k: (dev[k], k[1]))
            name = f'{cfg.replace(":", "_").replace("|", "__")}_w{w:g}'
            res = {'rule': list(best), 'dev': round(dev[best], 6), 'holdout': round(macro('holdout', pho, *best), 6),
                   'best_iteration': int(bst.best_iteration)}
            md = out / 'models' / name
            md.mkdir(parents=True, exist_ok=True)
            bst.save_model(str(md / 'model.json'))
            (md / 'config.json').write_text(json.dumps({'name': name, 'config': cfg, 'members': names,
                                                        'decoy_weight': w, **res}, indent=1))
            R[name] = res
            log(name, res)
            report_path.write_text(json.dumps(R, indent=1))
    ranked = sorted(((k, v) for k, v in R.items() if isinstance(v, dict) and 'dev' in v), key=lambda kv: -kv[1]['dev'])
    log('by dev:', [(k, v['dev'], v['holdout']) for k, v in ranked[:8]])


def lookup(files, pairs, field):
    """Value per (ref, target) of `pairs` (DataFrame r, t) from npz shards with ref/target/field; NaN if absent."""
    parts = []
    for fn in files:
        z = np.load(fn, allow_pickle=True)
        parts.append(pd.DataFrame({'r': z['ref'].astype(str), 't': z['target'].astype(str),
                                   'v': z[field].astype(np.float64)}))
    table = pd.concat(parts, ignore_index=True).drop_duplicates(['r', 't'])
    return pairs.merge(table, on=['r', 't'], how='left')['v'].to_numpy()


def predict(out, model_dir, member_globs):
    cfg = json.loads((model_dir / 'config.json').read_text())
    globs = parse_members(member_globs)
    cands_out, refs_arr, tgts = {}, [], []
    with open(CANDS, encoding='utf-8') as f:
        next(f)
        for line in f:
            s, c = line.rstrip('\n').split('\t')
            cands_out[s] = c
            for t in (c.split(',') if c else []):
                refs_arr.append(s); tgts.append(t)
    pairs = pd.DataFrame({'r': refs_arr, 't': tgts})
    log('pairs', len(pairs))
    with open(RAW / 'test/test_source1.tsv', encoding='utf-8', newline='') as fh:
        refs = [{'entity_id': r['entity_id'], 'country': r['country']}
                for r in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE)]
    position = {r['entity_id']: i for i, r in enumerate(refs)}
    assert set(position) == set(cands_out), 'candidate file and test_source1 disagree'
    ref_idx = np.array([position[r] for r in refs_arr], dtype=np.int64)
    s1 = lookup(S1_TEST, pairs, 'prob')
    log('stage-1 probabilities found for', int((~np.isnan(s1)).sum()), 'of', len(s1))
    s1f = np.nan_to_num(s1, nan=0.0)
    s1f = np.where(s1f < 0.02, 0.0, s1f)
    cache = {}

    def get(n):
        if n not in cache and globs[n].endswith('.npy'):  # feature block already in candidate_pairs row order
            v = np.load(globs[n]).astype(np.float64)
            assert v.shape[0] == len(pairs), (n, v.shape, len(pairs))
            log('member', n, globs[n], 'shape', v.shape, 'missing rows', int(np.isnan(v.reshape(len(v), -1)[:, 0]).sum()))
            cache[n] = v
        if n not in cache:
            files = sorted({f for pat in globs[n].split('|') for f in glob.glob(pat)})
            v = lookup(files, pairs, 'logit')
            miss = int(np.isnan(v).sum())
            log('member', n, 'files', files, 'missing', miss)
            if not files or miss > 0.001 * len(v):
                raise SystemExit(f'ABORT: member {n}: {miss}/{len(v)} pairs without a logit')
            cache[n] = np.nan_to_num(v, nan=-20.0)
        return cache[n]

    cols, names = columns(cfg['config'], get)
    X = np.column_stack([s1f] + cols).astype(np.float32)
    bst = xgb.Booster()
    bst.load_model(str(model_dir / 'model.json'))
    bst.set_param({'device': 'cuda'})
    prob = np.asarray(bst.inplace_predict(X, iteration_range=(0, cfg['best_iteration'] + 1)), dtype=np.float32)
    dest = out / cfg['name']
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / 'late_scores.npz', ref_idx=ref_idx, target=np.array(tgts, dtype=str), prob=prob,
             stage1=s1f.astype(np.float32), **{f'ce_{n}': get(n).astype(np.float32) for n in names})
    rule, thr = cfg['rule']
    stats = Counter()
    finalize(refs, cands_out, ref_idx, np.array(tgts, dtype=object), prob, {'decision': {'rule': rule, 'threshold': thr}},
             True, dest, stats, {'late_model': str(model_dir), 'rule': [rule, thr], 'started': time.monotonic()}, {})
    log('wrote', dest, dict((k, v) for k, v in stats.items() if not isinstance(v, dict)))


def main():
    stage, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if stage == 'train':
        train(out)
    else:
        predict(out, Path(sys.argv[3]), sys.argv[4])


if __name__ == '__main__':
    main()
