"""France blocking check: exact-key copies (same core name, same house number, same street words) missing from the
S1's candidate list. Train says such pairs are 99.99% true, so a miss is a blocking loss no matcher can recover.
Every France S2/S3 record is keyed (not only candidates); a key shared by exactly one France S1 is matched to it.
Reports, per country: key matches, how many are in the candidates, in the submission, owned by another S1.
Env LATE (late_scores.npz, candidate order), SUB (matching_results.tsv), OUT, COUNTRIES (default France).
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np

os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
from france_census import RAW, core, house, load, street_words


def key(name, addr):
    c = core(name)
    h = house(addr)[0] if addr.strip() else ''
    sw = street_words(addr)
    return (' '.join(sorted(set(c))), h, ' '.join(sorted(sw))) if c and h and sw else None


def main():
    C = set(os.environ.get('COUNTRIES', 'France').split(','))
    Z = np.load(os.environ['LATE'], allow_pickle=True)
    ri, tg = Z['ref_idx'], Z['target'].astype(str)
    refs, s1key, cty = [], {}, {}
    with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
        for line in fh:
            q = line.rstrip('\n').split('\t')
            refs.append(q[ie]); cty[q[ie]] = q[c]
            if q[c] in C:
                k = key(q[n], q[a])
                if k:
                    s1key[q[ie]] = k
    owners = defaultdict(list)
    for r, k in s1key.items():
        owners[k].append(r)
    cand = defaultdict(set)
    for i, t in zip(ri, tg):
        if cty[refs[i]] in C:
            cand[refs[i]].add(t)
    match = load(os.environ['SUB'])
    owner = {t: r for r, ts in match.items() for t in ts}
    R = Counter(); ex = []
    for k in (2, 3):
        with open(RAW / f'test_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) <= c or q[c] not in C:
                    continue
                kk = key(q[n], q[a] if len(q) > a else '')
                if not kk or kk not in owners:
                    continue
                o = owners[kk]
                if len(o) > 1:
                    R[q[c] + ' key_shared_by_several_s1'] += 1
                    continue
                r, t = o[0], q[ie]
                R[q[c] + ' key_match'] += 1
                inc = t in cand[r]
                R[q[c] + ' in_candidates'] += inc
                R[q[c] + ' in_submission'] += t in match.get(r, ())
                if not inc:
                    R[q[c] + ' missing_owned_elsewhere'] += t in owner
                    if len(ex) < 25:
                        ex.append(f'{r} {t} | {q[n]} | {q[a] if len(q) > a else ""}')
    R = dict(R); R['examples_missing'] = ex
    out = json.dumps(R, indent=1, ensure_ascii=False)
    open(os.environ['OUT'], 'w', encoding='utf-8').write(out)
    print(out, flush=True)


if __name__ == '__main__':
    main()
