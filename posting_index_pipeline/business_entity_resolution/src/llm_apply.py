"""Test path of the LLM judge (only run if llm_probe.py MODE=eval gains on dev and holdout).

MODE=pairs_test: US/India pairs of the T6-bag test lists with 0.02 < p < 0.98 -> PAIRS npz (s1, t, row).
MODE=apply: replaces those pairs' p with the dev-fitted combination sigma(a*logit(p) + b*llm + c*blank + d*blank*llm + e),
re-decides with the bag's rule + global exclusivity (Vishal's finalize; first checked to reproduce the bag's own
US/India decisions from the unchanged p), re-adds U5's US/India rescue links (ranks 11-20) whose record is still free,
takes France rows from FR_TSV (U5 or U6) and writes OUT_DIR/matching_results.tsv + U5's candidate_pairs.tsv.
"""
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault('SUB_DIR', '.')
sys.path.insert(0, str(Path(__file__).parent))
H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
BAG = Path('/scratch/scai/mtech/aib262467/t6bag/test')
U5 = Path('/scratch/scai/mtech/aib262467/final_u5')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def s1_table():
    refs = []
    with open(H / 'data/dataset/test/test_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, ic = head.index('entity_id'), head.index('country')
        for line in fh:
            q = line.rstrip('\n').split('\t'); refs.append({'entity_id': q[ie], 'country': q[ic]})
    return refs


def pairs_test():
    Z = np.load(BAG / 'late_scores.npz', allow_pickle=True)
    ri, tg, p = Z['ref_idx'], Z['target'].astype(str), Z['prob'].astype(np.float64)
    refs = s1_table()
    cty = np.array([r['country'] for r in refs], dtype=object)[ri]
    m = np.flatnonzero((cty != 'France') & (p > 0.02) & (p < 0.98))
    np.savez(os.environ['PAIRS'], s1=np.array([refs[i]['entity_id'] for i in ri[m]]), t=tg[m], row=m)
    log('test band pairs', len(m), Counter(cty[m]))


def apply():
    os.environ.setdefault('OUT_DIR', '.')
    from france_census import load
    from submission_pipeline import finalize
    Z = np.load(BAG / 'late_scores.npz', allow_pickle=True)
    ri, tg, p = Z['ref_idx'], Z['target'].astype(str).astype(object), Z['prob'].astype(np.float64)
    B = np.load(os.environ['PAIRS'], allow_pickle=True)
    row = B['row']
    llm = np.full(len(row), np.nan)
    L = Path(os.environ['OUT'])
    for f in sorted(L.parent.glob(L.name + '_*.npy')):
        a = np.load(f); llm[a[:, 0].astype(int)] = a[:, 1]
    assert not np.isnan(llm).any(), f'missing LLM scores: {int(np.isnan(llm).sum())}'
    c = json.load(open(os.environ['EVAL']))['coef']
    tx = {}
    need = set(tg[row])
    for k in (2, 3):
        with open(H / f'data/dataset/test/test_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, ia = head.index('entity_id'), head.index('business_address')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if q[ie] in need:
                    tx[q[ie]] = q[ia] if len(q) > ia else ''
    blank = np.array([0.0 if tx.get(t, '').strip() else 1.0 for t in tg[row]])
    lp = np.log(p[row] / (1 - p[row]))
    z = c['logit_p'] * lp + c['llm'] * llm + c['blank'] * blank + c['blank_x_llm'] * blank * llm + c['const']
    p2 = p.copy(); p2[row] = 1 / (1 + np.exp(-z))
    log('band pairs', len(row), 'mean p', round(float(p[row].mean()), 4), '->', round(float(p2[row].mean()), 4))
    refs = s1_table()
    cands = {}
    with open(BAG / 'candidates.tsv', encoding='utf-8') as f:
        for line in f:
            s, cc = line.rstrip('\n').split('\t'); cands[s] = cc
    out = Path(os.environ['OUT_DIR']); out.mkdir(parents=True, exist_ok=True)
    rep = {'decision': {'rule': 'expected', 'threshold': 0.68}}
    chk = out / 'repro_check'; chk.mkdir(exist_ok=True)
    finalize(refs, dict(cands), ri, tg, p.astype(np.float32), rep, True, chk, Counter(), {'started': time.monotonic()})
    a, b = load(chk / 'matching_results.tsv'), load(BAG / 'expected_0.68/matching_results.tsv')
    cty = {r['entity_id']: r['country'] for r in refs}
    mism = sum(set(a[s]) != set(b[s]) for s in a if cty[s] != 'France')
    log('reproduction of bag US/India decisions: mismatching S1', mism)
    if mism > 100:
        raise SystemExit('ABORT: finalize does not reproduce the bag decisions')
    new = out / 'decided'; new.mkdir(exist_ok=True)
    finalize(refs, dict(cands), ri, tg, p2.astype(np.float32), rep, True, new, Counter(), {'started': time.monotonic()})
    dec = load(new / 'matching_results.tsv')
    u5 = load(U5 / 'out/matching_results.tsv'); base = load(U5 / 'base.tsv')
    fr = load(os.environ.get('FR_TSV', str(U5 / 'out/matching_results.tsv')))
    order = [r['entity_id'] for r in refs]
    res = {s: (list(fr[s]) if cty[s] == 'France' else list(dec[s])) for s in order}
    owner = Counter(t for ts in res.values() for t in ts)
    R = Counter()
    for s in order:
        if cty[s] == 'France':
            continue
        for t in set(u5[s]) - set(base[s]):
            if owner[t]:
                R['rescue_dropped'] += 1
            else:
                res[s].append(t); owner[t] += 1; R['rescue_kept'] += 1
    R['records_in_2_rows'] = sum(v > 1 for v in owner.values())
    for k in ('US', 'India', 'France'):
        R[f'{k}_added_vs_U5'] = sum(len(set(res[s]) - set(u5[s])) for s in order if cty[s] == k)
        R[f'{k}_removed_vs_U5'] = sum(len(set(u5[s]) - set(res[s])) for s in order if cty[s] == k)
        R[f'{k}_emptied_vs_U5'] = sum(bool(u5[s]) and not res[s] for s in order if cty[s] == k)
    with open(out / 'matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s in order:
            f.write(s + '\t' + ','.join(res[s]) + '\n')
    shutil.copy(os.environ.get('CAND_TSV', str(U5 / 'out/candidate_pairs.tsv')), out / 'candidate_pairs.tsv')
    print(json.dumps(dict(R), indent=1), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    {'pairs_test': pairs_test, 'apply': apply}[os.environ['MODE']]()
