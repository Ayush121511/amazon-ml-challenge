"""Native-script candidate channel for India: records whose business name is written in an Indic script.

Character TF-IDF shares no n-gram between 'అర్బన్ ప్రాపర్టీస్ ప్రైవేట్ లిమిటెడ్' and 'Urban Properties Private Limited',
so retrieval misses these copies. Here every India S2/S3 record with an Indic-script name is transliterated
(indic_translit.py), reduced to a consonant skeleton without legal words, and matched to India S1 names by skeleton
similarity (rapidfuzz ratio >= SIM) inside the same first-skeleton-word bucket, with address agreement (a shared
number token and a shared place word). A record is linked only to its single best S1 (ties are skipped).
MODE=train: holdout S1s of the labelled sample; precision of proposals, how many are outside the top-10 candidates,
and holdout F0.5 of T6's decisions with / without the extra links. MODE=test: writes extra candidate pairs and the
accepted extra links for a submission (records owned by another S1 are skipped). Env MODE, SIM, OUT, SUB (test).
"""
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz.fuzz import ratio

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
sys.path.insert(0, str(Path(__file__).parent))
from indic_translit import INDIC, skeleton

H = Path('/home/scai/mtech/aib262015/scratch')
V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
LEGAL_SK = {'frvt', 'fvt', 'fr', 'lmtd', 'ltd', 'lmt', 'l', 'lf', 'llf', 'kmfn', 'kmfny', 'krf', 'krfrsn', 'nk', 'inc',
            'nkrfrtd', 'k', 'ms', 'm', 's', 'smt', 'sr', 'srmt', 'sri', 'mr', 'lmtt', 'frvtt', 'frt', 'fvtltd'}
PLACE_STOP = {'road', 'rd', 'no', 'flat', 'floor', 'plot', 'near', 'house', 'h', 'building', 'nagar', 'colony', 'street',
              'india', 'null', 'na', 'st', 'the', 'and', 'of', 'main', 'opp', 'door'}


def core_sk(name):
    return ' '.join(w for w in skeleton(name).split() if w not in LEGAL_SK and len(w) > 1)


def addr_keys(a):
    a = re.sub(r'[^0-9a-z]+', ' ', a.lower())
    nums = {t for t in re.findall(r'\d+', a) if t.lstrip('0')}
    nums = {t.lstrip('0') for t in nums}
    words = {t for t in a.split() if t.isalpha() and len(t) > 3 and t not in PLACE_STOP}
    return nums, words


def read(split):
    D = H / f'amazon_ml_2026/data/dataset/{split}'
    s1, rec = {}, {}
    for k in (1, 2, 3):
        with open(D / f'{split}_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) <= c or q[c] != 'India':
                    continue
                if k == 1:
                    s1[q[ie]] = (q[n], q[a])
                elif INDIC.search(q[n]):
                    rec[q[ie]] = (q[n], q[a] if len(q) > a else '')
    return s1, rec


def propose(s1, rec, sim):
    bucket = defaultdict(list)
    info = {}
    for s, (n, a) in s1.items():
        sk = core_sk(n)
        if sk:
            info[s] = (sk, addr_keys(a))
            for num in info[s][1][0]:
                bucket[(sk.split()[0], num)].append(s)
    out = {}
    for t, (n, a) in rec.items():
        sk = core_sk(n)
        if not sk or len(sk) < 4:
            continue
        nums, words = addr_keys(a)
        best, second = (0, None), 0
        seen = set()
        for s in (x for num in nums for x in bucket.get((sk.split()[0], num), ())):
            if s in seen:
                continue
            seen.add(s)
            ssk, (snums, swords) = info[s]
            if not (words & swords):
                continue
            r = ratio(sk, ssk)
            if r > best[0]:
                second, best = best[0], (r, s)
            elif r > second:
                second = r
        if best[1] and best[0] >= sim and best[0] - second >= 5:
            out[t] = (best[1], best[0])
    return out


def main():
    mode, sim = os.environ.get('MODE', 'train'), float(os.environ.get('SIM', '85'))
    R = {}
    if mode == 'train':
        from distractor_check import decide, f05, partition_of
        gold = {k: set(v) for k, v in json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text()).items()}
        cp = partition_of(gold)
        s1, rec = read('train')
        prop = propose(s1, rec, sim)
        owner = {t: s for s, ts in gold.items() for t in ts}
        S = np.load(H / 'amazon_ml_2026/artifacts/fusion/stage1.npz')
        mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
        refs, tgts = [], []
        with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t'); refs.append(r); tgts.append(t)
        refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
        P = np.load(V / 'fusion/boundary_models_t6/stage3b_prob.npy')
        for part in ('dev', 'holdout'):
            prefs = sorted(s for s in gold if cp[s] == part)
            pset = set(prefs)
            ix = np.flatnonzero(np.isin(refs, prefs))
            pos = {s: i for i, s in enumerate(prefs)}
            ch = decide(np.array([pos[r] for r in refs[ix]]), P[ix], 'expected', 0.74, len(prefs))
            pred, cand = defaultdict(set), defaultdict(set)
            for j, c in zip(ix, ch):
                cand[refs[j]].add(tgts[j])
                if c:
                    pred[refs[j]].add(tgts[j])
            predowner = {t: s for s, ts in pred.items() for t in ts}
            C = Counter(); new = defaultdict(set)
            for t, (s, r) in prop.items():
                if s not in pset:
                    continue
                ok = owner.get(t) == s
                C['proposed'] += 1; C['proposed_true'] += ok
                if t in cand[s]:
                    C['already_candidate'] += 1
                    continue
                C['new_candidate'] += 1; C['new_true'] += ok
                if t in predowner:
                    C['new_owned_elsewhere'] += 1
                    continue
                new[s].add(t)
            base = np.mean([f05(gold[s], pred.get(s, set())) for s in prefs])
            plus = np.mean([f05(gold[s], pred.get(s, set()) | new.get(s, set())) for s in prefs])
            C['F_base'] = round(float(base), 6); C['F_with_channel'] = round(float(plus), 6)
            C['gain'] = round(float(plus - base), 6)
            R[part] = dict(C)
            print(part, dict(C), flush=True)
        R['total_native_records'] = len(rec); R['proposed_all'] = len(prop)
    else:
        s1, rec = read('test')
        prop = propose(s1, rec, sim)
        from france_census import load
        match = load(os.environ['SUB'])
        owner = {t: r for r, ts in match.items() for t in ts}
        extra_cand = defaultdict(list); extra_acc = defaultdict(list)
        for t, (s, r) in prop.items():
            extra_cand[s].append(t)
            if t not in owner and t not in match.get(s, []):
                extra_acc[s].append(t)
        json.dump({'candidates': extra_cand, 'accepted': extra_acc}, open(os.environ['OUT'] + '.pairs.json', 'w'))
        R = {'native_records': len(rec), 'proposed': len(prop), 's1_with_extra_candidates': len(extra_cand),
             'accepted_links': sum(len(v) for v in extra_acc.values())}
    open(os.environ['OUT'], 'w').write(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1), flush=True)


if __name__ == '__main__':
    main()
