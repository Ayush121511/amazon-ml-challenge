"""Two deterministic candidate channels for copies that character retrieval misses (US + India), found by reading the
holdout blocking misses:
  web   web / handle names built from the S1 name: 'aesociety.com' = A(chintya) E(ducational) + Society,
        'btprivate.com' = B(ig) T(echnology) + Private, '#LITTLEAUTO', 'energyprivate.c0m'. S1 keys: concatenations of
        contiguous word runs, initials of the other words + one word, and the anagram of all words; the record key is
        its name without www/TLD/punctuation, leet digits mapped back (0->o, 1->l, 5->s ...).
  leet  names with leet digits inside words ('b1ack constructions', '6URU BENEFIT', 'Ltd 5ai It'): both names are
        leet-normalised and matched on their core word set.
A proposal needs address agreement (a shared number or a shared place word) and a single best S1. MODE=train reports
precision, new candidates outside the top 10 and holdout / dev F0.5 of T6 with the extra links; MODE=test writes the
extra pairs for a submission (records owned by another S1 skipped). Env CHANNEL, MODE, OUT, SUB (test).
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

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
sys.path.insert(0, str(Path(__file__).parent))
from native_channel import addr_keys

H = Path('/home/scai/mtech/aib262015/scratch')
V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '6': 'g', '8': 'b', '@': 'a', '$': 's'})
LEGAL = {'llc', 'inc', 'ltd', 'limited', 'pvt', 'private', 'co', 'company', 'corp', 'corporation', 'llp', 'lp', 'pllc',
         'pc', 'the', 'of', 'and', 'm', 's', 'ms', 'smt', 'sri', 'shri', 'mr', 'dr', 'plc', 'incorporated'}
SUFFIX = {'limited', 'ltd', 'llc', 'inc', 'co', 'corp', 'company', 'llp', 'lp', 'pllc', 'plc', 'corporation'}
WEBLIKE = re.compile(r'(\.(com|c0m|co|in|net|org|biz|io)\b|www|^[@#]|\s[@#])', re.I)


def fold(s):
    s = unicodedata.normalize('NFKD', s or '')
    return ''.join(c for c in s if not unicodedata.combining(c)).lower()


def words(name, drop_legal=True):
    w = re.sub(r'[^a-z0-9]+', ' ', fold(name)).split()
    return [x for x in w if not (drop_legal and x in LEGAL)]


def web_keys(name):
    ks = set()
    full = words(name, False)
    tail = list(full)
    while tail and tail[-1] in SUFFIX:
        tail.pop()
    for w in (words(name), full, tail):
        w = [x.translate(LEET) for x in w]
        n = len(w)
        if n == 0:
            continue
        for i in range(n):
            for j in range(i + 1, n + 1):
                if j - i >= 1:
                    ks.add(''.join(w[i:j]))
            ks.add(''.join(x[0] for k, x in enumerate(w) if k != i) + w[i])
        ks.add(''.join(x[0] for x in w))
        ks.add('A:' + ''.join(sorted(''.join(w))))
    return {k for k in ks if len(k.lstrip('A:')) >= 6}


def web_record_key(name):
    s = fold(name)
    s = re.sub(r'\.(com|c0m|co|in|net|org|biz|io)\b', ' ', s)
    s = re.sub(r'\bwww\b', ' ', s)
    s = re.sub(r'[^a-z0-9]+', '', s).translate(LEET)
    return {s, 'A:' + ''.join(sorted(s))} if len(s) >= 6 else set()


def leet_key(name):
    w = [x.translate(LEET) if re.search(r'[a-z]', x) else x for x in words(name)]
    return ' '.join(sorted(set(w))) if w else ''


def read(split):
    D = H / f'amazon_ml_2026/data/dataset/{split}'
    s1, rec = {}, {}
    for k in (1, 2, 3):
        with open(D / f'{split}_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) <= c or q[c] not in ('US', 'India'):
                    continue
                if k == 1:
                    s1[q[ie]] = (q[n], q[a], q[c])
                else:
                    rec[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c])
    return s1, rec


def propose(s1, rec, channel):
    index = defaultdict(set)
    ak = {}
    if channel == 'web':
        want = {t: web_record_key(v[0]) for t, v in rec.items() if WEBLIKE.search(v[0]) or (' ' not in v[0].strip() and len(v[0]) >= 8)}
        need = set().union(*want.values()) if want else set()
        for s, (n, a, c) in s1.items():
            for k in web_keys(n) & need:
                index[(c, k)].add(s)
    else:
        want = {t: {leet_key(v[0])} for t, v in rec.items() if re.search(r'[a-z][0-9]|[0-9][a-z]', fold(v[0])) and leet_key(v[0])}
        need = set().union(*want.values()) if want else set()
        for s, (n, a, c) in s1.items():
            k = leet_key(n)
            if k in need and k != ' '.join(sorted(set(words(n)))) or k in need:
                index[(c, k)].add(s)
    out = {}
    for t, keys in want.items():
        n, a, c = rec[t]
        cands = set().union(*[index.get((c, k), set()) for k in keys]) if keys else set()
        if not cands:
            continue
        nums, ws = addr_keys(a)
        ok = []
        for s in cands:
            if s not in ak:
                ak[s] = addr_keys(s1[s][1])
            snums, sws = ak[s]
            if (nums & snums) or (ws & sws):
                ok.append(s)
        if len(ok) == 1:
            out[t] = (ok[0], 100)
    return out


def main():
    ch, mode = os.environ['CHANNEL'], os.environ.get('MODE', 'train')
    R = {'channel': ch}
    if mode == 'train':
        from distractor_check import decide, f05, partition_of
        gold = {k: set(v) for k, v in json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text()).items()}
        cp = partition_of(gold)
        s1, rec = read('train')
        prop = propose(s1, rec, ch)
        owner = {t: s for s, ts in gold.items() for t in ts}
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
            chs = decide(np.array([pos[r] for r in refs[ix]]), P[ix], 'expected', 0.74, len(prefs))
            pred, cand = defaultdict(set), defaultdict(set)
            for j, c in zip(ix, chs):
                cand[refs[j]].add(tgts[j])
                if c:
                    pred[refs[j]].add(tgts[j])
            po = {t: s for s, ts in pred.items() for t in ts}
            C = Counter(); new = defaultdict(set); ex = []
            for t, (s, _) in prop.items():
                if s not in pset:
                    continue
                okk = owner.get(t) == s
                C['proposed'] += 1; C['proposed_true'] += okk
                if t in cand[s]:
                    C['already_candidate'] += 1
                    continue
                C['new_candidate'] += 1; C['new_true'] += okk
                if t in po:
                    C['new_owned_elsewhere'] += 1
                    continue
                new[s].add(t)
                if len(ex) < 15:
                    ex.append(f"{int(okk)} {s1[s][0]} | {s1[s][1][:50]} => {rec[t][0]} | {rec[t][1][:50]}")
            base = np.mean([f05(gold[s], pred.get(s, set())) for s in prefs])
            plus = np.mean([f05(gold[s], pred.get(s, set()) | new.get(s, set())) for s in prefs])
            C['gain'] = round(float(plus - base), 6)
            R[part] = dict(C); R[part + '_examples'] = ex
            print(ch, part, dict(C), flush=True)
    else:
        s1, rec = read('test')
        prop = propose(s1, rec, ch)
        from france_census import load
        match = load(os.environ['SUB'])
        owner = {t: r for r, ts in match.items() for t in ts}
        ec, ea = defaultdict(list), defaultdict(list)
        for t, (s, _) in prop.items():
            ec[s].append(t)
            if t not in owner:
                ea[s].append(t)
        json.dump({'candidates': ec, 'accepted': ea}, open(os.environ['OUT'] + '.pairs.json', 'w'))
        R.update({'proposed': len(prop), 'accepted_links': sum(len(v) for v in ea.values())})
    open(os.environ['OUT'], 'w').write(json.dumps(R, indent=1, ensure_ascii=False))
    print(json.dumps(R, indent=1, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
