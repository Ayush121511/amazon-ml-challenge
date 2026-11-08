"""LLM judge for France only (no French labels; the public score is the check). US/India are left exactly as U5.

MODE=pairs_fr: France pairs of the FRtrained_w4 lists worth a second opinion: U5-accepted with p < 0.95 (veto
candidates, the S1 keeps another link) and U5-rejected with p > 0.10 whose record no S1 holds in U5 (rescue candidates).
Blank-address records of S1 names shared by another French S1 are skipped (no reader can pick the owner).
Scoring: llm_probe.py MODE=score SPLIT=test.
MODE=thresholds: LLM yes-no logit thresholds from the labelled US/India band (llm_probe.py train scores): the lowest
t_hi with P(true | llm >= t_hi) >= PREC_HI and the highest t_lo with P(true | llm <= t_lo) <= PREC_LO.
MODE=apply_fr: rescue if llm >= t_hi, veto if llm <= t_lo; writes OUT_DIR (U5 US/India rows, new France rows, U5
candidates), a flip sample for manual reading, and counts.
"""
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
sys.path.insert(0, str(Path(__file__).parent))
from france_census import fold, load

H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
FRW4 = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/submission_FRtrained_w4/late_scores.npz')
U5 = Path(os.environ.get('BASE_DIR', '/scratch/scai/mtech/aib262467/final_u5/out'))
nrm = lambda s: re.sub(r'[^a-z0-9]+', ' ', fold(s or '')).strip()


def test_text():
    s1, rec = [], {}
    for k in (1, 2, 3):
        with open(H / f'data/dataset/test/test_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                rec[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
                if k == 1:
                    s1.append(q[ie])
    return s1, rec


def pairs_fr():
    Z = np.load(FRW4, allow_pickle=True)
    ri, tg, p = Z['ref_idx'], Z['target'].astype(str), Z['prob'].astype(np.float64)
    s1, rec = test_text()
    u5 = load(U5 / 'matching_results.tsv')
    owner = {t: s for s, ts in u5.items() for t in ts}
    fr_names = Counter(nrm(rec[s][0]) for s in s1 if rec[s][2] == 'France')
    keep, kind = [], []
    R = Counter()
    for j in np.flatnonzero(np.array([rec[s1[i]][2] == 'France' for i in ri])):
        s, t = s1[ri[j]], tg[j]
        acc = t in set(u5.get(s, []))
        if not rec.get(t, ('', ''))[1].strip() and fr_names[nrm(rec[s][0])] > 1:
            R['skip_blank_twin'] += 1
            continue
        if acc and p[j] < 0.95 and len(u5[s]) >= 2:
            keep.append(j); kind.append('veto')
        elif not acc and p[j] > 0.10 and t not in owner:
            keep.append(j); kind.append('rescue')
    keep = np.array(keep)
    np.savez(os.environ['PAIRS'], s1=np.array([s1[ri[j]] for j in keep]), t=tg[keep], row=keep, kind=np.array(kind), p=p[keep])
    R.update(Counter(kind))
    print('France pairs to judge', len(keep), dict(R), flush=True)


def shards(prefix, n):
    out = np.full(n, np.nan)
    for f in sorted(Path(prefix).parent.glob(Path(prefix).name + '_*.npy')):
        a = np.load(f); out[a[:, 0].astype(int)] = a[:, 1]
    return out


def thresholds():
    import pandas as pd
    df = pd.read_pickle(f"{os.environ['A']}/train.json.train_pairs.pkl")
    band = df[(df.p > 0.02) & (df.p < 0.98)]
    llm = shards(os.environ['TRAIN_OUT'], len(band))
    ok = ~np.isnan(llm); y = band.y.values[ok]; v = llm[ok]
    ph, pl = float(os.environ.get('PREC_HI', 0.92)), float(os.environ.get('PREC_LO', 0.05))
    o = np.argsort(-v); prec_top = np.cumsum(y[o]) / np.arange(1, len(o) + 1)
    hi = [v[o][i] for i in range(len(o)) if prec_top[i] >= ph and i >= 30]
    t_hi = float(min(hi)) if hi else float('inf')
    o2 = np.argsort(v); rate_bot = np.cumsum(y[o2]) / np.arange(1, len(o2) + 1)
    lo = [v[o2][i] for i in range(len(o2)) if rate_bot[i] <= pl and i >= 30]
    t_lo = float(max(lo)) if lo else float('-inf')
    R = {'n': int(ok.sum()), 't_hi': t_hi, 't_lo': t_lo, 'share_above_hi': float((v >= t_hi).mean()), 'share_below_lo': float((v <= t_lo).mean()),
         'true_rate_above_hi': float(y[v >= t_hi].mean()) if (v >= t_hi).any() else None,
         'true_rate_below_lo': float(y[v <= t_lo].mean()) if (v <= t_lo).any() else None,
         'auc_llm_in_band': float(pd.Series(v).rank().values[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / max(1, y.sum() * (len(y) - y.sum()))}
    print(json.dumps(R, indent=1), flush=True)
    json.dump(R, open(os.environ['THR'], 'w'), indent=1)


def apply_fr():
    B = np.load(os.environ['PAIRS'], allow_pickle=True)
    s1a, ta, kind, p = B['s1'].astype(str), B['t'].astype(str), B['kind'].astype(str), B['p']
    llm = shards(os.environ['OUT'], len(ta))
    T = json.load(open(os.environ['THR']))
    miss = int(np.isnan(llm).sum())
    assert miss == 0, f'missing {miss}'
    u5 = load(U5 / 'matching_results.tsv')
    new = {s: list(ts) for s, ts in u5.items()}
    owner = {t: s for s, ts in u5.items() for t in ts}
    s1, rec = test_text()
    R = Counter(); flips = []
    for i in np.argsort(-np.abs(llm)):
        s, t = s1a[i], ta[i]
        if kind[i] == 'rescue' and llm[i] >= T['t_hi'] and t not in owner:
            new[s].append(t); owner[t] = s; R['rescued'] += 1; flips.append(('ADD', i))
        elif kind[i] == 'veto' and llm[i] <= T['t_lo'] and len(new[s]) >= 2 and t in new[s]:
            new[s].remove(t); owner.pop(t, None); R['vetoed'] += 1; flips.append(('DROP', i))
    out = Path(os.environ['OUT_DIR']); out.mkdir(parents=True, exist_ok=True)
    order = [line.split('\t', 1)[0] for line in open(U5 / 'matching_results.tsv', encoding='utf-8')][1:]
    with open(out / 'matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s in order:
            f.write(s + '\t' + ','.join(new[s]) + '\n')
    shutil.copy(U5 / 'candidate_pairs.tsv', out / 'candidate_pairs.tsv')
    rng = np.random.default_rng(0)
    with open(out / 'flip_sample.txt', 'w', encoding='utf-8') as f:
        for tag in ('ADD', 'DROP'):
            idx = [i for k, i in flips if k == tag]
            for i in rng.choice(idx, min(40, len(idx)), replace=False) if idx else []:
                s, t = s1a[i], ta[i]
                others = [x for x in u5[s] if x != t][:4]
                f.write(f'{tag} p={p[i]:.2f} llm={llm[i]:+.1f} | {rec[s][0]} | {rec[s][1]}  =>  {rec[t][0]} | {rec[t][1] or "(blank)"}'
                        f'   [U5 links: ' + '; '.join(f'{rec[x][0]} | {rec[x][1]}' for x in others) + ']\n')
    R['records_in_2_rows'] = sum(v > 1 for v in Counter(t for ts in new.values() for t in ts).values())
    R['llm_mean_rescue_pairs'] = float(llm[kind == 'rescue'].mean()); R['llm_mean_veto_pairs'] = float(llm[kind == 'veto'].mean())
    print(json.dumps(dict(R), indent=1), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    {'pairs_fr': pairs_fr, 'thresholds': thresholds, 'apply_fr': apply_fr}[os.environ['MODE']]()
