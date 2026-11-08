"""France audits and France-model reports.

Each command is one analysis script. Run `python france_audits.py <command> [args]`; `python france_audits.py` lists the commands.
"""
import sys


def cmd_affix_check():
    """Is 'S1 name + one extra word' a true match or a generated decoy? Measured with train labels, by added word.

France test lists are full of 'X Holding / International / Groupe / Développement / Et Fils' records near the S1
address that the cross-encoder rejects and v5 accepts. If the generator builds affiliate decoys the same way in
train (English: Group, International, Development, Holding, & Sons), their train positive rate is ~0 and the
cross-encoder is right on France. For every train S2/S3 record whose name minus one word equals an S1 name key
(keys shared by <= 20 S1), label = the record is that S1's gold match; split by same / different house number.
Legal forms (removed from keys) are measured on exact-key pairs: extra legal token on the record vs none.
Writes fusion/affix_check.json.
"""
    import json
    import re
    import unicodedata
    from collections import Counter, defaultdict
    from pathlib import Path

    HOME = Path.home()
    RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
    OUT = HOME / 'scratch/amazon_ml_2026/artifacts/fusion/affix_check.json'
    LEG = {'llc', 'inc', 'corp', 'co', 'ltd', 'lp', 'llp', 'pllc', 'pc', 'pa', 'pvt', 'private', 'limited', 'corporation',
           'incorporated', 'company', 'l', 'c', 'p'}
    AFFIX = {'group', 'international', 'intl', 'development', 'developments', 'holding', 'holdings', 'sons', 'distribution',
             'participations', 'services', 'service', 'partners', 'associates', 'enterprises', 'ventures', 'global',
             'solutions', 'industries', 'trading', 'center', 'centre', 'the', 'and', 'of', 'india', 'usa', 'america',
             'com', 'dba', 'aka', 'systems', 'technologies', 'consulting', 'management', 'foundation', 'trust'}


    def toks(s):
        s = unicodedata.normalize('NFKD', s or '')
        s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
        return re.sub(r'[^\w]+', ' ', s).split()


    def num(a):
        m = re.search(r'\d+', a or '')
        return m.group(0) if m else ''


    def main():
        owner = {}
        with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) > 1 and q[1]:
                    for t in q[1].split(','):
                        owner[t.strip()] = q[0]
        idx = defaultdict(list)
        with open(RAW / 'train_source1.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) > iaddr:
                    t = toks(q[iname])
                    idx[' '.join(sorted(set(t) - LEG))].append((q[ie], num(q[iaddr]), set(t)))
        idx = {k: v for k, v in idx.items() if k and len(v) <= 20}
        print('S1 keys', len(idx), flush=True)
        stat = defaultdict(lambda: [0, 0])
        for k in (2, 3):
            with open(RAW / f'train_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if len(q) <= iaddr:
                        continue
                    e, n = q[ie], num(q[iaddr])
                    blank = not q[iaddr].strip()
                    T = set(toks(q[iname]))
                    core = T - LEG
                    own = owner.get(e)
                    for s, sn, st in idx.get(' '.join(sorted(core)), []):
                        where = 'blank' if blank else 'same_num' if n and n == sn else 'diff_num'
                        extra = (T & LEG) - st
                        tag = 'exact_key+legal_added' if extra else 'exact_key'
                        stat[(tag, where)][0] += 1; stat[(tag, where)][1] += own == s
                    for w in core & AFFIX:
                        for s, sn, st in idx.get(' '.join(sorted(core - {w})), []):
                            if w in st:
                                continue
                            where = 'blank' if blank else 'same_num' if n and n == sn else 'diff_num'
                            stat[(w, where)][0] += 1; stat[(w, where)][1] += own == s
            print('source', k, 'done', flush=True)
        R = {}
        for (w, where), (n, pos) in sorted(stat.items()):
            R.setdefault(w, {})[where] = {'n': n, 'pos_rate': round(pos / n, 4)}
        OUT.write_text(json.dumps(R, indent=1))
        for w in sorted(R, key=lambda w: -sum(v['n'] for v in R[w].values())):
            print(w, R[w], flush=True)


    if __name__ == '__main__':
        main()



def cmd_census_review():
    """Manual-review sheet for the France census fixes: sampled restored / rejected pairs with each S1's full candidate list.

Strata (the census logic says where the risk is):
  restore, S1 has records at other house numbers (its decoy cluster is accounted for)      -> expected safe
  restore, S1 has none elsewhere (a same-address record may be the decoy)                  -> risky
  reject, S1 keeps other matches                                                           -> expected safe
  reject, rejection leaves the S1 with no match                                            -> risky
Marks: * accepted by the base, + restored, - rejected; house number, p, name, address.
Env BASE_DIR (base submission folder), FIX_DIR (census output folder). Writes FIX_DIR/census_review.md.
"""
    import os
    import re
    import unicodedata
    from collections import defaultdict
    from pathlib import Path

    import numpy as np

    HOME = Path.home()
    RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/test'
    BASE, FIX = Path(os.environ['BASE_DIR']), Path(os.environ['FIX_DIR'])


    def load(f):
        d = {}
        with open(f, encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                s, m = (line.rstrip('\n').split('\t') + [''])[:2]
                d[s] = set(x for x in m.split(',') if x)
        return d


    def num(a):
        m = re.search(r'(?<![\d/])0*(\d+)', unicodedata.normalize('NFKD', a or '').lower())
        return m.group(1) if m else ''


    def main():
        rd = next(d for d in BASE.iterdir() if (d / 'matching_results.tsv').exists())
        base = load(rd / 'matching_results.tsv')
        fix = load(FIX / rd.name / 'matching_results.tsv')
        Z = np.load(BASE / 'late_scores.npz', allow_pickle=True)
        ri, tg, p = Z['ref_idx'], Z['target'].astype(str), Z['prob']
        refs, cty, text = [], {}, {}
        with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr, ic = (head.index(c) for c in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                refs.append(q[ie]); cty[q[ie]] = q[ic]
                if q[ic] == 'France':
                    text[q[ie]] = (q[iname], q[iaddr])
        fr = [i for i, r in enumerate(refs) if cty[r] == 'France']
        frs = set(fr)
        cand = defaultdict(list)
        for j in np.flatnonzero(np.isin(ri, fr)):
            cand[refs[ri[j]]].append((tg[j], float(p[j])))
        need = {t for r in cand for t, _ in cand[r]}
        for k in (2, 3):
            with open(RAW / f'test_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if q[ie] in need:
                        text[q[ie]] = (q[iname], q[iaddr] if len(q) > iaddr else '')
        strata = defaultdict(list)
        for r in (refs[i] for i in fr):
            added, removed = fix[r] - base[r], base[r] - fix[r]
            n1 = num(text[r][1])
            elsewhere = sum(1 for t, _ in cand[r] if num(text.get(t, ('', ''))[1]) not in ('', n1))
            for t in added:
                strata['restore | S1 has records at other numbers' if elsewhere else 'restore | S1 has NO records elsewhere'].append((r, t))
            for t in removed:
                strata['reject | S1 keeps other matches' if fix[r] else 'reject | leaves S1 EMPTY'].append((r, t))
        rng = np.random.default_rng(4)
        out = ['# France census fixes: manual review sheet\n',
               'Marks: `*` accepted by base, `+` restored by census, `-` rejected by census. num = first house number.\n']
        for k in sorted(strata):
            v = strata[k]
            out.append(f'\n## {k}: {len(v)} pairs (showing 12)\n')
            for i in rng.choice(len(v), min(12, len(v)), replace=False):
                r, t = v[i]
                rn, ra = text[r]
                out.append(f'\n**{r}** {rn} | {ra}  (num {num(ra)})  -> pair {t}')
                for c, pv in sorted(cand[r], key=lambda x: -x[1]):
                    mk = '+' if c in fix[r] - base[r] else '-' if c in base[r] - fix[r] else '*' if c in base[r] else ' '
                    cn, ca = text.get(c, ('', ''))
                    out.append(f'- `{mk}` p={pv:.2f} num={num(ca) or "-":>5s} {c[:2]} | {cn} | {ca}')
        (FIX / 'census_review.md').write_text('\n'.join(out))
        print('\n'.join(f'{k}: {len(v)}' for k, v in sorted(strata.items())))


    if __name__ == '__main__':
        main()



def cmd_colocated_check():
    """Co-located unrelated names: alias or same-building distractor? True rate by how many such records share the S1's address.

A record whose core name shares no word with the S1's, at the S1's house number and street, is an alias in some cases
and a different business in the same building in others. Train (labelled sample): true rate of these pairs by k, the
number of such records in the S1's candidate list, and by country. France (a submission): pairs and accepted links by
the same k, with p quantiles, so the France decision can be set against the train rate. Env LATE, SUB, OUT.
"""
    import gzip
    import json
    import os
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    from fr_labels import read_france
    from france_census import core, house, load, street_words, tokens_addr

    H = Path('/home/scai/mtech/aib262015/scratch')


    def colocated(ra, tn, ta, rc):
        if not ta.strip():
            return False
        h1, h2 = house(ra)[0], house(ta)[0]
        sw = street_words(ra)
        return bool(h1 and h1 == h2 and sw and sw <= tokens_addr(ta) and rc and not set(rc) & set(core(tn)))


    def main():
        R = {}
        S = np.load(H / 'amazon_ml_2026/artifacts/fusion/stage1.npz')
        y = np.load(H / 'vishal_v4_artifacts/pairs/y.npy')[S['rows']]
        mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
        pr = []
        with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    pr.append(line.rstrip('\n').split('\t'))
        need = {x for p in pr for x in p}
        text = {}
        for k in (1, 2, 3):
            with open(H / f'amazon_ml_2026/data/dataset/train/train_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if q[ie] in need:
                        text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
        by = defaultdict(list)
        V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
        PM = {'t5': np.load(V / 'fusion/boundary_models_t5/stage3b_prob.npy'), 'frrobust': np.load(V / 'submission_FRrobust/robust_sample_prob.npy')}
        from distractor_check import partition_of
        gold = json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text())
        cp = partition_of({k: set(v) for k, v in gold.items()})
        hi = Counter(); hip = Counter()
        rows = defaultdict(list)
        for i, ((r, t), lab) in enumerate(zip(pr, y)):
            rn, ra, cty = text[r]; tn, ta, _ = text.get(t, ('', '', ''))
            if colocated(ra, tn, ta, core(rn)):
                by[r].append(int(lab)); rows[r].append((i, int(lab)))
        for r, lst in rows.items():
            if cp[r] == 'train':
                continue
            k = min(len(lst), 4)
            for mdl, P in PM.items():
                for i, lab in lst:
                    if P[i] >= 0.9:
                        hi[(text[r][2], mdl, k)] += 1; hip[(text[r][2], mdl, k)] += lab
        R['train_p_ge_0.9_devholdout'] = {str(k): [hi[k], round(hip[k] / hi[k], 3)] for k in sorted(hi, key=str)}
        cell, pos = Counter(), Counter()
        for r, labs in by.items():
            k = min(len(labs), 6); cty = text[r][2]
            cell[(cty, k)] += len(labs); pos[(cty, k)] += sum(labs)
            cell[(cty, 'refs', k)] += 1; pos[(cty, 'refs', k)] += sum(labs)
        R['train'] = {str(k): [cell[k], round(pos[k] / cell[k], 4)] for k in sorted(cell, key=str) if 'refs' not in k}
        R['train_true_per_ref'] = {str(k): round(pos[k] / cell[k], 3) for k in sorted(cell, key=str) if 'refs' in k}
        Z = np.load(os.environ['LATE'], allow_pickle=True)
        ri, tg, p3 = Z['ref_idx'], Z['target'].astype(str), Z['prob']
        match = load(os.environ['SUB'])
        refs, ftext, m, s1_core = read_france(ri, tg)
        fb = defaultdict(list)
        for j in np.flatnonzero(m):
            r, t = refs[ri[j]], tg[j]
            tn, ta = ftext.get(t, ('', ''))
            if colocated(ftext[r][1], tn, ta, s1_core[r]):
                fb[r].append((j, t in match.get(r, ())))
        fc, fa, fp = Counter(), Counter(), defaultdict(list)
        for r, js in fb.items():
            k = min(len(js), 6)
            fc[k] += len(js); fa[k] += sum(a for _, a in js); fp[k] += [p3[j] for j, _ in js]
        R['france'] = {str(k): {'pairs': fc[k], 'accepted': fa[k], 'acc_rate': round(fa[k] / fc[k], 3),
                                'p_q10_50_90': np.quantile(fp[k], [0.1, 0.5, 0.9]).round(3).tolist()} for k in sorted(fc)}
        out = json.dumps(R, indent=1)
        open(os.environ['OUT'], 'w').write(out)
        print(out, flush=True)


    if __name__ == '__main__':
        main()



def cmd_cross_source_check():
    """Cross-source link completion, sized before building: rejected candidates that duplicate an accepted sibling.

For an S1, a rejected candidate whose (core name, house number, street words) key equals the key of a record already
accepted for that S1 from the other source (S2 vs S3), or from the same source, may be a missed copy. Train (T5's own
dev/holdout decisions, labelled): pool size and true rate of such rejected candidates. France (a submission): pool size,
owned-by-another-S1 share, p quantiles, examples. Env LATE, SUB, OUT.
"""
    import gzip
    import json
    import os
    import random
    import sys
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    sys.path.insert(0, str(Path(__file__).parent))
    from distractor_check import decide, partition_of
    from fr_labels import read_france
    from france_census import core, house, load, street_words

    H = Path('/home/scai/mtech/aib262015/scratch')
    V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')


    def key(name, addr):
        c = core(name)
        if not c or not addr.strip():
            return None
        return (' '.join(sorted(set(c))), house(addr)[0], ' '.join(sorted(street_words(addr))))


    def pool(refs_of, cands, accepted, text):
        """(ref, target, same_source) for rejected candidates whose key equals an accepted sibling's key."""
        out = []
        for r in refs_of:
            acc = accepted.get(r, set())
            if not acc:
                continue
            keys = defaultdict(set)
            for a in acc:
                k = key(*text.get(a, ('', ''))[:2])
                if k:
                    keys[k].add(a[:2])
            for t in cands[r]:
                if t in acc:
                    continue
                k = key(*text.get(t, ('', ''))[:2])
                if k and k in keys:
                    out.append((r, t, t[:2] in keys[k]))
        return out


    def main():
        R = {}
        S = np.load(H / 'amazon_ml_2026/artifacts/fusion/stage1.npz')
        y = np.load(H / 'vishal_v4_artifacts/pairs/y.npy')[S['rows']]
        gold = {k: set(v) for k, v in json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text()).items()}
        cp = partition_of(gold)
        mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
        refs, tgts = [], []
        with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t'); refs.append(r); tgts.append(t)
        refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
        need = set(refs) | set(tgts)
        text = {}
        for k in (1, 2, 3):
            with open(H / f'amazon_ml_2026/data/dataset/train/train_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if q[ie] in need:
                        text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
        P = np.load(V / 'fusion/boundary_models_t5/stage3b_prob.npy')
        lab = {(r, t): int(v) for r, t, v in zip(refs, tgts, y)}
        part = np.array([cp[r] for r in refs], dtype=object)
        for p in ('dev', 'holdout'):
            ix = np.flatnonzero(part == p)
            prefs = sorted(s for s in gold if cp[s] == p)
            pos = {s: i for i, s in enumerate(prefs)}
            ch = decide(np.array([pos[r] for r in refs[ix]]), P[ix], 'expected', 0.66, len(prefs))
            cands, accepted = defaultdict(list), defaultdict(set)
            for j, c in zip(ix, ch):
                cands[refs[j]].append(tgts[j])
                if c:
                    accepted[refs[j]].add(tgts[j])
            pl = pool(cands.keys(), cands, accepted, text)
            cnt = Counter(); tru = Counter()
            for r, t, same in pl:
                k = (text[r][2], 'same_src' if same else 'other_src')
                cnt[k] += 1; tru[k] += lab[(r, t)]
            R['train_' + p] = {str(k): [cnt[k], round(tru[k] / cnt[k], 3)] for k in sorted(cnt)}
            print('train', p, R['train_' + p], flush=True)
        Z = np.load(os.environ['LATE'], allow_pickle=True)
        ri, tg, p3 = Z['ref_idx'], Z['target'].astype(str), Z['prob']
        match = load(os.environ['SUB'])
        owner = {t: r for r, ts in match.items() for t in ts}
        frefs, ftext, m, _ = read_france(ri, tg)
        cands = defaultdict(list); pj = {}
        for j in np.flatnonzero(m):
            cands[frefs[ri[j]]].append(tg[j]); pj[(frefs[ri[j]], tg[j])] = j
        accepted = {r: set(match.get(r, [])) for r in cands}
        pl = pool(cands.keys(), cands, accepted, ftext)
        c = Counter((('same_src' if s else 'other_src'), 'owned_elsewhere' if t in owner else 'free') for _, t, s in pl)
        R['france'] = {str(k): v for k, v in c.items()}
        free = [x for x in pl if x[1] not in owner]
        R['france_free_p_q10_50_90'] = np.quantile([p3[pj[(r, t)]] for r, t, _ in free], [0.1, 0.5, 0.9]).round(3).tolist() if free else []
        random.seed(4)
        R['france_examples'] = [f"{p3[pj[(r, t)]]:.2f} {'S' if s else 'O'} {ftext[r][0]} | {ftext[r][1]}  =>  {ftext[t][0]} | {ftext[t][1]}"
                                for r, t, s in random.sample(free, min(20, len(free)))]
        out = json.dumps(R, indent=1, ensure_ascii=False)
        open(os.environ['OUT'], 'w', encoding='utf-8').write(out)
        print(out, flush=True)


    if __name__ == '__main__':
        main()



def cmd_delta_audit():
    """Hand-check sheet for the France score change between the uploaded leaderboard file and a new submission.

U_new score = LB_old + delta(US/India) + delta(France). delta(France) only involves French references whose matches
differ between the two files. This lists them (counts by kind of change) and prints a random sample with every candidate,
which file accepts it ([O]ld / [N]ew), and the scores, for judging each candidate by hand.
Usage: delta_audit.py <old matching_results.tsv> <new matching_results.tsv> <out.txt> [N]
"""
    import csv
    import random
    import sys
    from collections import Counter
    from pathlib import Path

    import numpy as np

    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    ART = ME / 'artifacts'
    csv.field_size_limit(10 ** 9)


    def load(p):
        with open(p, encoding='utf-8') as f:
            next(f)
            return {l.split('\t')[0]: [x for x in l.rstrip('\n').split('\t')[1].split(',') if x] for l in f}


    def main():
        old, new, out = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
        N = int(sys.argv[4]) if len(sys.argv) > 4 else 70
        text, cty = {}, {}
        for k in (1, 2, 3):
            with open(ME / f'data/dataset/test/test_source{k}.tsv', encoding='utf-8', newline='') as fh:
                for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                    text[row['entity_id']] = (row['business_name'], row['business_address'])
                    if k == 1:
                        cty[row['entity_id']] = row['country']
        refs, tgts = [], []
        with open(ART / 'submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
            next(f)
            for line in f:
                s, c = line.rstrip('\n').split('\t')
                refs.append(s); tgts.extend(c.split(','))
        n = len(refs)
        T = np.array(tgts, dtype=object).reshape(n, 10)
        pos = {r: i for i, r in enumerate(refs)}
        pF = np.load(ART / 'submission_FRtrained_w4/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)
        p5 = np.load(ART / 'submission_T5_own/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)
        st = Counter()
        changed = {c: [] for c in ('US', 'India', 'France')}
        for r in old:
            a, b = set(old[r]), set(new[r])
            if a != b:
                changed[cty[r]].append(r)
                c = cty[r]
                st[f'{c} refs changed'] += 1; st[f'{c} links added'] += len(b - a); st[f'{c} links removed'] += len(a - b)
                st[f'{c} ' + ('add only' if a < b else 'remove only' if b < a else 'both')] += 1
        fr = sorted(changed['France'])
        rng = random.Random(4242)
        pick = rng.sample(fr, min(N, len(fr)))
        with open(out, 'w', encoding='utf-8') as f:
            f.write(' | '.join(f'{k}: {v}' for k, v in sorted(st.items())) + '\n')
            f.write(f'France references {sum(1 for r in cty.values() if r == "France")}\n')
            f.write('line: [O]=old file accepts [N]=new file accepts  pFR=France model  pT5=T5 | name | address\n\n')
            for q, r in enumerate(pick, 1):
                i = pos[r]
                a, b = set(old[r]), set(new[r])
                f.write(f'=== #{q} {r}: {text[r][0]} | {text[r][1]}   old {len(a)} -> new {len(b)}\n')
                seen = set()
                for k in np.argsort(-pF[i]):
                    t = T[i, k]; seen.add(t)
                    if pF[i, k] < 0.01 and p5[i, k] < 0.01 and t not in a | b:
                        continue
                    f.write(f"  {'O' if t in a else '.'}{'N' if t in b else '.'} pFR={pF[i, k]:.2f} pT5={p5[i, k]:.2f} | "
                            f"{text[t][0]} | {text[t][1]}\n")
                for t in sorted((a | b) - seen):
                    f.write(f"  {'O' if t in a else '.'}{'N' if t in b else '.'} rank>10 | {text[t][0]} | {text[t][1]}\n")
                f.write('\n')
        print(dict(st), flush=True)


    if __name__ == '__main__':
        main()



def cmd_filler_test():
    """Label test for the France-trained pattern 'first word kept, type word dropped, filler added, same address'.

French example: 'Gaule SARL Et Fils' for S1 'Gaule Ecole SARL' at the S1's own address. The France model accepts these;
the US/India model gives p=0. The generator is shared across countries, so the US/India training labels decide:
among sample candidate pairs at the same first house number with a near-identical address, where the record keeps some
of the S1's name words, drops at least one, and every added word is a filler/affiliate word, how many are true links?
Also the same-address pattern with nothing dropped (filler only added) and the shifted-number version, for contrast.
"""
    import csv
    import gzip
    import json
    import re
    import unicodedata
    from collections import Counter
    from pathlib import Path

    import numpy as np
    from rapidfuzz import fuzz

    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    ART = ME / 'artifacts'
    csv.field_size_limit(10 ** 9)
    LEGAL = set('inc llc ltd limited corp corporation co company pvt private llp plc lp pllc pc'.split())
    STOP = set('the of and de la'.split())
    FILLER = set('sons son group groupe holdings holding associates associes partners brothers bros enterprises enterprise '
                 'services service trust foundation solutions international global india usa america fils cie compagnie '
                 'developpement development france'.split())
    NUM = re.compile(r'\d+')


    def fold(s):
        s = unicodedata.normalize('NFKD', (s or '').lower())
        return ''.join(c for c in s if not unicodedata.combining(c))


    def words(n):
        return [w for w in re.findall(r'[a-z0-9]+', fold(n)) if w not in LEGAL and w not in STOP]


    def house(a):
        m = NUM.search(a or '')
        return int(m.group(0)) if m else None


    def main():
        y = np.load(ART / 'matcher_v4/pairs/y.npy')
        need, rows = set(), []
        with gzip.open(ART / 'matcher_v4/pairs/pairs.tsv.gz', 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                r, t = line.rstrip('\n').split('\t')
                rows.append((r, t))
        text = {}
        for k in (1, 2, 3):
            with open(ME / f'data/dataset/train/train_source{k}.tsv', encoding='utf-8', newline='') as fh:
                for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                    text[row['entity_id']] = (row['business_name'], row['business_address'], row['country'])
        st = Counter()
        ex = {'drop+filler_same': [], 'drop+filler_shift': []}
        for i, (r, t) in enumerate(rows):
            rn, ra, c = text[r]
            tn, ta, _ = text[t]
            A, B = set(words(rn)), set(words(tn))
            if not A or not B:
                continue
            added, dropped, kept = B - A, A - B, A & B
            if not added or not kept or not added <= FILLER:
                continue
            h1, h2 = house(ra), house(ta)
            if h1 is None or h2 is None or not (ta or '').strip():
                continue
            if h1 == h2 and fuzz.token_set_ratio(fold(ra), fold(ta)) >= 80:
                loc = 'same'
            elif h1 != h2 and abs(h1 - h2) <= 13:
                loc = 'shift'
            else:
                continue
            kind = ('drop+filler' if dropped else 'filler_only') + '_' + loc
            st[(c, kind, 'n')] += 1; st[(c, kind, 'true')] += int(y[i])
            if kind in ex and len(ex[kind]) < 12 and i % 3 == 0:
                ex[kind].append(f"{'TRUE ' if y[i] else 'false'} {rn} | {ra}  <-  {tn} | {ta}")
        R = {}
        for c in ('US', 'India'):
            for kind in ('drop+filler_same', 'filler_only_same', 'drop+filler_shift', 'filler_only_shift'):
                n = st[(c, kind, 'n')]
                R[f'{c} {kind}'] = {'n': n, 'true': st[(c, kind, 'true')], 'true_share': round(st[(c, kind, 'true')] / n, 3) if n else None}
        print(json.dumps(R, indent=1), flush=True)
        for k, v in ex.items():
            print('==', k); [print('  ', e) for e in v]


    if __name__ == '__main__':
        main()



def cmd_france_anatomy():
    """France rejection anatomy: where does France reject pairs the v5 matcher believes, and are such pairs usually true?

Pairs in the top 10 with v5 p >= 0.5 that the final submission rejects are put in buckets:
  claimed    the record is accepted for another reference (exclusivity)       -> own bucket
  address    record address blank / same first house number / different number / no number to compare
  name       after folding and dropping legal forms: same words / record adds words / record drops words /
             some overlap / none
  ce         cross-encoder (chunk2) logit >= 0 or < 0
Per bucket: rejected pairs per reference on test for US, India, France (submission 4), and on the US/India holdout the
same rate for T6 plus the share of those rejected pairs that are true links (labels). A bucket where France rejects far
more per reference than US/India and whose holdout true share is high is where France loses links.
Writes artifacts/gap_diag/france_anatomy.json and examples for the top excess buckets.
"""
    import csv
    import gzip
    import json
    import re
    import sys
    import unicodedata
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np

    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    AY = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
    ART = ME / 'artifacts'
    sys.path[:0] = [str(AY / 'business_entity_resolution/src')]
    csv.field_size_limit(10 ** 9)
    LEGAL = set('sarl sas sasu eurl sa sci snc selarl scop inc llc ltd limited corp corporation co company pvt private llp '
                'plc lp pllc pc'.split())
    NUM = re.compile(r'\d+')


    def fold(s):
        s = unicodedata.normalize('NFKD', s.lower())
        s = ''.join(c for c in s if not unicodedata.combining(c))
        return [w for w in re.findall(r'[a-z0-9]+', s) if w not in LEGAL]


    def house(a):
        m = NUM.search(a or '')
        return m.group(0) if m else ''


    def bucket(rn, ra, tn, ta, ce):
        A, B = set(fold(rn)), set(fold(tn))
        name = ('same' if A == B else 'rec_adds' if A < B else 'rec_drops' if B < A else 'overlap' if A & B else 'none')
        if not (ta or '').strip():
            addr = 'blank'
        else:
            h1, h2 = house(ra), house(ta)
            addr = 'same_no' if h1 and h1 == h2 else 'diff_no' if h1 and h2 else 'no_no'
        return f'{addr}|{name}|{"ce+" if ce >= 0 else "ce-"}'


    def read_text(split, keep=None):
        out = {}
        for k in (1, 2, 3):
            with open(ME / f'data/dataset/{split}/{split}_source{k}.tsv', encoding='utf-8', newline='') as fh:
                for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                    if keep is None or row['entity_id'] in keep:
                        out[row['entity_id']] = (row['business_name'], row['business_address'], row['country'])
        return out


    def load_lists(path):
        with open(path, encoding='utf-8') as f:
            next(f)
            return {l.split('\t')[0]: [x for x in l.rstrip('\n').split('\t')[1].split(',') if x] for l in f}


    def main():
        R = {}
        # ---- test side
        refs, tgts = [], []
        with open(ART / 'submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
            next(f)
            for line in f:
                s, c = line.rstrip('\n').split('\t')
                for t in c.split(','):
                    refs.append(s); tgts.append(t)
        v5 = np.load(ART / 'v5raw/v5_test_raw.npz')['v5_prob']
        Z6 = np.load(ART / 'submission_T6_own/late_scores.npz', allow_pickle=True)
        ZF = np.load(ART / 'submission_FRtrained_w4/late_scores.npz', allow_pickle=True)
        assert len(v5) == len(refs) == len(Z6['prob'])
        ce_t = Z6['ce'].astype(np.float64)
        sub = load_lists(sys.argv[1])
        acc = {(r, t) for r, v in sub.items() for t in v}
        held = {t for v in sub.values() for t in v}
        text = read_text('test')
        n_ref = Counter(v[2] for k, v in text.items() if k.startswith('S1-'))
        test_b = defaultdict(Counter)
        ex = defaultdict(list)
        for j in np.flatnonzero(v5 >= 0.5):
            r, t = refs[j], tgts[j]
            if (r, t) in acc:
                continue
            rn, ra, c = text[r]
            tn, ta, _ = text[t]
            b = 'claimed' if t in held else bucket(rn, ra, tn, ta, ce_t[j])
            test_b[c][b] += 1
            if c == 'France' and len(ex[b]) < 25 and j % 7 == 0:
                ex[b].append({'ref': f'{rn} | {ra}', 'rec': f'{tn} | {ta}', 'v5': round(float(v5[j]), 3),
                              'p_final': round(float(ZF['prob'][j]), 3), 'ce': round(float(ce_t[j]), 2)})
        del text
        # ---- holdout side (US/India, labels)
        from distractor_check import decide, partition_of
        gold = {k: set(v) for k, v in json.loads((ART / 'matcher_v4/pairs/gold.json').read_text()).items()}
        cp = partition_of(gold)
        m10 = np.load(ART / 'prune_v1/mask_top10.npy').astype(bool)
        pr, pt = [], []
        with gzip.open(ART / 'matcher_v4/pairs/pairs.tsv.gz', 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if m10[i]:
                    a, b = line.rstrip('\n').split('\t'); pr.append(a); pt.append(b)
        pr, pt = np.array(pr, dtype=object), np.array(pt, dtype=object)
        S = np.load(ART / 'fusion/stage1.npz')
        y = np.load(ART / 'matcher_v4/pairs/y.npy')[S['rows']] > 0
        s1 = S['stage1'].astype(np.float64)
        ce_s = np.load('/home/scai/mtech/aib262140/scratch/amazon_ml_2026/ftchunk2/ce_logit_sample.npy').astype(np.float64)
        p6 = np.load(ART / 'fusion/boundary_models_t6/stage3b_prob.npy')
        rule = json.load(open(ART / 'fusion/boundary_models_t6/stage3b_meta.json'))['rule']
        assert len(pr) == len(s1) == len(ce_s) == len(p6)
        prefs = sorted(s for s in gold if cp[s] == 'holdout')
        pos = {s: i for i, s in enumerate(prefs)}
        idx = np.flatnonzero(np.array([cp[r] == 'holdout' for r in pr]))
        ch = np.zeros(len(pr), bool)
        ch[idx] = decide(np.array([pos[r] for r in pr[idx]]), p6[idx], rule[0], rule[1], len(prefs))
        held_s = set(pt[ch])
        stext = read_text('train', set(pr[idx]) | set(pt[idx]))
        hb, ht = defaultdict(Counter), defaultdict(Counter)
        for j in idx[(s1[idx] >= 0.5) & ~ch[idx]]:
            r, t = pr[j], pt[j]
            rn, ra, c = stext[r]
            tn, ta, _ = stext[t]
            b = 'claimed' if t in held_s else bucket(rn, ra, tn, ta, np.nan_to_num(ce_s[j]))
            hb[c][b] += 1; ht[c][b] += int(y[j])
        n_hold = Counter(stext[s][2] for s in prefs)
        rows = []
        for b in set(test_b['France']) | set(test_b['US']) | set(hb['US']):
            per = {c: test_b[c][b] / n_ref[c] for c in ('US', 'India', 'France')}
            hr = {c: hb[c][b] / max(1, n_hold[c]) for c in ('US', 'India')}
            tr = (ht['US'][b] + ht['India'][b]) / max(1, hb['US'][b] + hb['India'][b])
            usin = (per['US'] + per['India']) / 2
            rows.append({'bucket': b, 'test_per_ref': {k: round(v, 5) for k, v in per.items()},
                         'holdout_per_ref': {k: round(v, 5) for k, v in hr.items()},
                         'holdout_true_share': round(tr, 3), 'holdout_n': hb['US'][b] + hb['India'][b],
                         'france_excess_per_ref': round(per['France'] - usin, 5),
                         'est_missed_true_per_ref_excess': round((per['France'] - usin) * tr, 5)})
        rows.sort(key=lambda d: -d['est_missed_true_per_ref_excess'])
        R['buckets'] = rows
        R['totals_rejected_v5believed_per_ref'] = {c: round(sum(test_b[c].values()) / n_ref[c], 4) for c in n_ref}
        R['est_france_excess_missed_true_per_ref'] = round(sum(max(0, d['est_missed_true_per_ref_excess']) for d in rows), 4)
        R['examples'] = {d['bucket']: ex[d['bucket']] for d in rows[:6]}
        (ART / 'gap_diag').mkdir(exist_ok=True)
        (ART / 'gap_diag/france_anatomy.json').write_text(json.dumps(R, indent=1, ensure_ascii=False))
        print('bucket'.ljust(28), 'US/ref  IN/ref  FR/ref | hold US/ref true% n | FR excess -> est missed true/ref')
        for d in rows[:30]:
            t, h = d['test_per_ref'], d['holdout_per_ref']
            print(d['bucket'].ljust(28), f"{t['US']:.4f} {t['India']:.4f} {t['France']:.4f} | {h['US']:.4f} "
                  f"{100 * d['holdout_true_share']:5.1f}% {d['holdout_n']:5d} | {d['france_excess_per_ref']:+.4f} -> "
                  f"{d['est_missed_true_per_ref_excess']:+.4f}", flush=True)
        print('totals', R['totals_rejected_v5believed_per_ref'], 'est France excess missed true/ref',
              R['est_france_excess_missed_true_per_ref'], flush=True)


    if __name__ == '__main__':
        main()



def cmd_france_audit():
    """Manual France audit sheet for a submission: French references with every top-10 candidate (+ accepted rescue links),
our decision and the scores, for reading by eye (no labels exist for France).

Sample: N_RAND random French references + N_CONT 'contested' ones (a candidate with 0.05 < p < 0.95, a v5-believed
candidate rejected, or an accepted candidate the cross-encoder rejects). Also prints per-reference flags: how many French
references hold each record in their top 10, and whether another reference was given the record.
Usage: france_audit.py <matching_results.tsv> <out.txt>
"""
    import csv
    import random
    import sys
    from collections import defaultdict
    from pathlib import Path

    import numpy as np

    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    ART = ME / 'artifacts'
    N_RAND, N_CONT = 30, 40
    csv.field_size_limit(10 ** 9)


    def main():
        sub_path, out_path = sys.argv[1], sys.argv[2]
        text, cty = {}, {}
        for k in (1, 2, 3):
            with open(ME / f'data/dataset/test/test_source{k}.tsv', encoding='utf-8', newline='') as fh:
                for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                    if row['country'] == 'France':
                        text[row['entity_id']] = (row['business_name'], row['business_address'])
                        if k == 1:
                            cty[row['entity_id']] = 1
        refs, tgts = [], []
        with open(ART / 'submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
            next(f)
            for line in f:
                s, c = line.rstrip('\n').split('\t')
                for t in c.split(','):
                    refs.append(s); tgts.append(t)
        v5 = np.load(ART / 'v5raw/v5_test_raw.npz')['v5_prob']
        ZF = np.load(ART / 'submission_FRtrained_w4/late_scores.npz', allow_pickle=True)
        Z6 = np.load(ART / 'submission_T6_own/late_scores.npz', allow_pickle=True)
        pF, ce = ZF['prob'], Z6['ce']
        sub = {}
        with open(sub_path, encoding='utf-8') as f:
            next(f)
            for l in f:
                r, m = l.rstrip('\n').split('\t')
                sub[r] = [x for x in m.split(',') if x]
        holder = defaultdict(list)
        for r, v in sub.items():
            for t in v:
                holder[t].append(r)
        by = defaultdict(list)
        nref = defaultdict(int)
        for j, r in enumerate(refs):
            if r in cty:
                by[r].append(j); nref[tgts[j]] += 1
        contested = []
        for r, js in by.items():
            acc = set(sub[r])
            if any((0.05 < pF[j] < 0.95) or (v5[j] >= 0.5 and tgts[j] not in acc) or (tgts[j] in acc and ce[j] < 0)
                   for j in js):
                contested.append(r)
        rng = random.Random(2027)
        fr = sorted(by)
        pick = [('RANDOM', r) for r in rng.sample(fr, N_RAND)] + [('CONTESTED', r) for r in rng.sample(sorted(contested), N_CONT)]
        with open(out_path, 'w', encoding='utf-8') as out:
            out.write(f'France refs {len(fr)}, contested {len(contested)} ({100 * len(contested) / len(fr):.1f}%)\n'
                      'cand line: [A]=accepted by us  src  p_final v5 ce(chunk2 logit)  nFR(refs with it in top10)  '
                      'held_by_other | name | address\n\n')
            for kind, r in pick:
                n, a = text[r]
                acc = sub[r]
                out.write(f'=== {kind} {r}: {n} | {a}   (we accept {len(acc)})\n')
                seen = set()
                for j in sorted(by[r], key=lambda j: -pF[j]):
                    t = tgts[j]; seen.add(t)
                    tn, ta = text.get(t, ('?', '?'))
                    other = [x for x in holder.get(t, []) if x != r]
                    out.write(f"  {'[A]' if t in acc else '[ ]'} {t[:2]} p={pF[j]:.3f} v5={v5[j]:.3f} ce={ce[j]:+.1f} "
                              f"n={nref[t]} {'HELD:' + text.get(other[0], ('?', ''))[0][:30] if other else ''} | {tn} | {ta}\n")
                for t in acc:
                    if t not in seen:
                        tn, ta = text.get(t, ('?', '?'))
                        out.write(f'  [A] {t[:2]} RESCUE(rank 11-20) | {tn} | {ta}\n')
                out.write('\n')
        print('wrote', out_path, 'contested', len(contested), flush=True)


    if __name__ == '__main__':
        main()



def cmd_france_block_check():
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



def cmd_france_cases():
    """France case file for manual reading: businesses where the decision is shakiest, one block per S1.

Groups (30 S1 each, fixed seed): U uncertain (a candidate with 0.15 < p < 0.85), L low-link (<= 1 accepted but >= 3
candidates at the S1's house number), B blank copy with 0.3 < p < 0.85, E emptied (no match), R random.
Per candidate: accepted mark, stage-3b p, v5, chunk2 logit, house-number relation (= same, ~ other, 0 blank/none),
census/extreme label source, name | address. Candidates shown: accepted, p >= 0.01, or at the S1's number.
Env LATE (late_scores.npz), SUB (matching_results.tsv), LABELS (fr_labels.npz), OUT (text file).
"""
    import os
    from collections import defaultdict

    import numpy as np

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    from fr_labels import read_france
    from france_census import house, load

    TAG = {0: '', 1: 'cP', 2: 'cN', 3: 'xP', 4: 'xN'}


    def main():
        Z = np.load(os.environ['LATE'], allow_pickle=True)
        ri, tg = Z['ref_idx'], Z['target'].astype(str)
        p3, v5, c2 = Z['prob'], Z['stage1'], np.nan_to_num(Z['ce'])
        match = load(os.environ['SUB'])
        L = np.load(os.environ['LABELS'], allow_pickle=True)
        src = np.zeros(len(tg), np.int8); src[L['idx']] = L['src']
        refs, text, m, _ = read_france(ri, tg)
        by = defaultdict(list)
        for j in np.flatnonzero(m):
            by[ri[j]].append(j)
        rel = {}
        for i, js in by.items():
            h1 = house(text[refs[i]][1])[0]
            for j in js:
                ta = text.get(tg[j], ('', ''))[1]
                h2 = house(ta)[0] if ta.strip() else ''
                rel[j] = '=' if h1 and h2 == h1 else ('~' if h1 and h2 else '0')
        groups = {'U': [], 'L': [], 'B': [], 'E': [], 'R': list(by)}
        for i, js in by.items():
            acc = set(match.get(refs[i], []))
            if any(0.15 < p3[j] < 0.85 for j in js):
                groups['U'].append(i)
            if len(acc) <= 1 and sum(rel[j] == '=' for j in js) >= 3:
                groups['L'].append(i)
            if any(0.3 < p3[j] < 0.85 and not text.get(tg[j], ('', ''))[1].strip() for j in js):
                groups['B'].append(i)
            if not acc:
                groups['E'].append(i)
        rng = np.random.default_rng(7)
        with open(os.environ['OUT'], 'w', encoding='utf-8') as f:
            for g, ids in groups.items():
                f.write(f'#### group {g}: {len(ids)} S1 in France\n')
                for i in rng.choice(ids, min(30, len(ids)), replace=False):
                    r = refs[i]; acc = set(match.get(r, []))
                    f.write(f'\n{r} | {text[r][0]} | {text[r][1]}\n')
                    for j in sorted(by[i], key=lambda j: -p3[j]):
                        t = tg[j]
                        if not (t in acc or p3[j] >= 0.01 or rel[j] == '='):
                            continue
                        n, a = text.get(t, ('', ''))
                        f.write(f"  {'A' if t in acc else '.'} {p3[j]:.2f} v{v5[j]:.2f} c{c2[j]:+.0f} {rel[j]} {TAG[src[j]]:2s} {n} | {a}\n")
                f.write('\n')


    if __name__ == '__main__':
        main()



def cmd_france_check():
    """Label-free per-country look at the S5 test scores: is France (no training data) less certain than US/India?

On late_scores.npz (pre-exclusivity L2 probability, v5 probability, cross-encoder logit per test pair) and the
threshold@0.68 decision, per country: pair-probability bands, uncertain candidates per reference, best-probability
quantiles, expected matches (sum of p) vs chosen, v5 / cross-encoder disagreement, and the expected F0.5 of the
decision if p were calibrated (ratio-of-expectations approximation). Writes fusion/france_check.json.
"""
    import json
    import sys
    import time
    from pathlib import Path

    import numpy as np

    HOME = Path.home()
    S5 = HOME / 'scratch/amazon_ml_2026/artifacts/submission_S5/late_scores.npz'
    RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/test/test_source1.tsv'
    OUT = HOME / 'scratch/amazon_ml_2026/artifacts/fusion/france_check.json'
    THR = 0.68


    def log(*a):
        print(time.strftime('%H:%M:%S'), *a, flush=True)


    def main():
        Z = np.load(S5, allow_pickle=True)
        ri, p, s1, ce = Z['ref_idx'], Z['prob'].astype(np.float64), Z['stage1'].astype(np.float64), Z['ce'].astype(np.float64)
        countries = []
        with open(RAW, encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ic = head.index('country')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                countries.append(q[ic] if len(q) > ic else '')
        countries = np.array(countries, dtype=object)
        n = len(countries)
        log('pairs', len(p), 'refs', n)
        ch = p >= THR
        sp = np.bincount(ri, p, n)
        k = np.bincount(ri, ch.astype(float), n)
        tp = np.bincount(ri, p * ch, n)
        none = np.exp(np.bincount(ri, np.log(np.clip(1 - p, 1e-12, 1)), n))
        ef = np.where(k > 0, 1.25 * tp / np.maximum(0.25 * sp + k, 1e-12), none)
        best = np.zeros(n); np.maximum.at(best, ri, p)
        unc = np.bincount(ri, ((p >= 0.1) & (p < 0.9)).astype(float), n)
        pc = countries[ri]
        bands = [0, .02, .1, .3, .5, .68, .9, .98, 1.0001]
        R = {'threshold': THR}
        for c in sorted(set(countries)):
            m, mp = countries == c, pc == c
            top = np.full(n, -np.inf); np.maximum.at(top, ri[mp], ce[mp])
            top1 = np.zeros(n); np.maximum.at(top1, ri[mp], s1[mp])
            R[c] = {
                'refs': int(m.sum()),
                'expected_f05_if_calibrated': round(float(ef[m].mean()), 5),
                'mean_sum_p': round(float(sp[m].mean()), 4), 'mean_chosen': round(float(k[m].mean()), 4),
                'refs_no_match': round(float((k[m] == 0).mean()), 4),
                'refs_with_uncertain_cand': round(float((unc[m] > 0).mean()), 4),
                'mean_uncertain_cands': round(float(unc[m].mean()), 4),
                'pair_band_share': {f'{bands[i]}-{min(bands[i + 1], 1)}': round(float(((p[mp] >= bands[i]) & (p[mp] < bands[i + 1])).mean()), 5)
                                    for i in range(len(bands) - 1)},
                'best_p_quantiles': {q: round(float(np.quantile(best[m], q)), 4) for q in (0.01, 0.03, 0.05, 0.07, 0.1, 0.15)},
                'top_ce_logit_quantiles': {q: round(float(np.quantile(top[m], q)), 3) for q in (0.05, 0.1, 0.25, 0.5)},
                'top_v5_quantiles': {q: round(float(np.quantile(top1[m], q)), 4) for q in (0.05, 0.1, 0.25, 0.5)},
                'v5_hi_ce_lo': round(float(((s1[mp] >= 0.9) & (ce[mp] < 0)).mean()), 5),
                'v5_lo_ce_hi': round(float(((s1[mp] < 0.1) & (ce[mp] > 2)).mean()), 5),
                'ef_loss_share_uncertain_refs': round(float((1 - ef[m & (unc > 0)]).sum() / max((1 - ef[m]).sum(), 1e-9)), 4),
            }
            log(c, R[c])
        OUT.write_text(json.dumps(R, indent=1))
        log('done')


    if __name__ == '__main__':
        main()



def cmd_france_diff_cases():
    """Read what a new France submission changed: random France links added / removed vs a base, with the S1's accepted
siblings for context, plus accepted links whose house number differs from the S1's (the shifted-decoy family).
Env LATE (new run's late_scores.npz), NEW, BASE (matching_results.tsv), OUT, N (per group, default 40).
"""
    import os
    import random
    from collections import defaultdict

    import numpy as np

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    from fr_labels import read_france
    from france_census import house, load


    def main():
        Z = np.load(os.environ['LATE'], allow_pickle=True)
        ri, tg, p3 = Z['ref_idx'], Z['target'].astype(str), Z['prob']
        new, base = load(os.environ['NEW']), load(os.environ['BASE'])
        refs, text, m, _ = read_france(ri, tg)
        pj = {(refs[ri[j]], tg[j]): j for j in np.flatnonzero(m)}
        fr = {refs[ri[j]] for j in np.flatnonzero(m)}
        add = [(r, t) for r in fr for t in set(new.get(r, [])) - set(base.get(r, []))]
        rem = [(r, t) for r in fr for t in set(base.get(r, [])) - set(new.get(r, []))]
        shifted = []
        for r in fr:
            h1 = house(text[r][1])[0]
            for t in new.get(r, []):
                a = text.get(t, ('', ''))[1]
                h2 = house(a)[0] if a.strip() else ''
                if h1 and h2 and h1 != h2:
                    shifted.append((r, t))
        n = int(os.environ.get('N', 40))
        random.seed(11)
        with open(os.environ['OUT'], 'w', encoding='utf-8') as f:
            for g, pairs in (('ADDED', add), ('REMOVED', rem), ('ACCEPTED_NUMBER_DIFFERS', shifted)):
                f.write(f'#### {g}: {len(pairs)}\n')
                for r, t in random.sample(pairs, min(n, len(pairs))):
                    j = pj.get((r, t))
                    f.write(f"\n{text[r][0]} | {text[r][1]}\n  {'+' if g != 'REMOVED' else '-'} {p3[j] if j is not None else -1:.2f} "
                            f"{text.get(t, ('?', ''))[0]} | {text.get(t, ('', ''))[1]}\n")
                    for s in new.get(r, []):
                        if s != t:
                            f.write(f"    = {text.get(s, ('?', ''))[0]} | {text.get(s, ('', ''))[1]}\n")
                f.write('\n')


    if __name__ == '__main__':
        main()



def cmd_france_eda():
    """France deep dive on the S5 test scores (France has no training data, so everything here is label-free).

A. data profile per country and source: blank rates, legal forms, name templates (city in name), name-key and
   address-key multiplicity in S1, address field layout, casing
B. positive noise model: token edits between an S1 name/address and its confident matches (p>=0.98, v5>=0.95,
   ce>=5), per country - which words get added/dropped, legal-form swaps, abbreviations, number changes
C. France failure buckets vs US/India: CE veto (v5>=0.9, ce<0), CE override (v5<0.1, ce>2), uncertain not chosen,
   marginal chosen, empty references, contested targets (>=2 references at p>=0.68); per bucket the edit patterns,
   blank rate, S1 name/address uniqueness, source mix, top added/dropped tokens
D. case files: full candidate lists (p, v5, ce, chosen) for sampled France references per bucket
Writes fusion/france_eda.md and fusion/france_cases.md.
"""
    import os
    import re
    import unicodedata
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np

    HOME = Path.home()
    RAW = Path(os.environ.get('TEST_RAW', str(HOME / 'scratch/amazon_ml_2026/data/dataset/test')))
    S5 = Path(os.environ.get('S5_SCORES', str(HOME / 'scratch/amazon_ml_2026/artifacts/submission_S5/late_scores.npz')))
    OUT = Path(os.environ.get('OUT_DIR', str(HOME / 'scratch/amazon_ml_2026/artifacts/fusion')))
    THR = 0.68
    LEG = {'sarl', 'sas', 'sasu', 'sci', 'eurl', 'ei', 'sa', 'snc', 'scop', 'scm', 'selarl', 'llc', 'inc', 'corp', 'co',
           'ltd', 'lp', 'llp', 'pllc', 'pc', 'pa', 'pvt', 'private', 'limited', 'l', 'c', 'p'}
    ABBR = re.compile(r'(^|[\s,])(r|r\.|av|av\.|bd|bd\.|n°|no|no\.|st|st\.|ave|rd|dr|ln|blvd|#)(?=\s)', re.I)
    rng = np.random.default_rng(7)
    md = []


    def say(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        md.append(s)


    def fold(s):
        s = unicodedata.normalize('NFKD', s or '')
        return ''.join(c for c in s if not unicodedata.combining(c)).lower()


    def toks(s):
        return re.sub(r'[^\w]+', ' ', fold(s)).split()


    def nkey(n):
        return ' '.join(sorted(set(toks(n)) - LEG))


    def num(a):
        m = re.search(r'\d+', a or '')
        return m.group(0) if m else ''


    def akey(a):
        t = [x for x in toks(a) if not x.isdigit()]
        return num(a) + '|' + ' '.join(sorted(set(t)))


    def pct(x):
        return f'{100 * x:.1f}%'


    def table(head, rows):
        say('| ' + ' | '.join(head) + ' |')
        say('|' + '---|' * len(head))
        for r in rows:
            say('| ' + ' | '.join(str(v) for v in r) + ' |')
        say('')


    def main():
        # ---------- A. profile (streaming) ----------
        say('# France deep dive (S5 test scores, label-free)\n')
        prof = defaultdict(Counter)
        legal = defaultdict(Counter)
        fields = defaultdict(Counter)
        s1 = {}
        s1_country = {}
        nk_count, ak_count = defaultdict(Counter), defaultdict(Counter)
        text = {}
        for k in (1, 2, 3):
            with open(RAW / f'test_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, iname, iaddr, ic = (head.index(c) for c in ('entity_id', 'business_name', 'business_address', 'country'))
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if len(q) <= ic:
                        continue
                    e, n, a, c = q[ie], q[iname], q[iaddr], q[ic]
                    g = (c, f'S{k}')
                    prof[g]['records'] += 1
                    prof[g]['blank_addr'] += not a.strip()
                    prof[g]['upper_name'] += n.isupper()
                    prof[g]['upper_addr'] += a.isupper()
                    prof[g]['has_number'] += bool(num(a))
                    prof[g]['abbrev'] += bool(ABBR.search(' ' + a + ' '))
                    prof[g]['fields'] += a.count(',') + 1 if a.strip() else 0
                    prof[g]['non_ascii_name'] += any(ord(ch) > 127 for ch in n)
                    for t in set(toks(n)) & LEG:
                        legal[g][t] += 1
                    if a.strip():
                        for fpart in a.split(','):
                            fpart = fpart.strip()
                            if fpart and not re.search(r'\d', fpart) and len(fpart.split()) <= 3:
                                fields[g][fpart] += 1
                    if k == 1:
                        s1_country[e] = c
                        nk_count[c][nkey(n)] += 1
                        ak_count[c][akey(a)] += 1
                        at = set(toks(a))
                        prof[g]['city_in_name'] += bool(set(toks(n)) & {x for x in at if len(x) > 3 and not x.isdigit()})
                    text[e] = (n, a)
        say('## A. Data profile\n')
        rows = []
        for g in sorted(prof):
            P = prof[g]; n = P['records']
            rows.append([g[0], g[1], f'{n:,}', pct(P['blank_addr'] / n), pct(P['has_number'] / n), pct(P['abbrev'] / n),
                         f"{P['fields'] / max(n - P['blank_addr'], 1):.2f}", pct(P['upper_name'] / n), pct(P['upper_addr'] / n),
                         pct(P['non_ascii_name'] / n), pct(P['city_in_name'] / n) if g[1] == 'S1' else '-'])
        table(['country', 'src', 'records', 'blank addr', 'has number', 'abbrev', 'addr fields', 'UPPER name',
               'UPPER addr', 'non-ascii name', 'city word in name'], rows)
        say('Top legal forms per country/source (share of records):\n')
        table(['country', 'src', 'top legal-form tokens'],
              [[g[0], g[1], ', '.join(f'{t} {pct(v / prof[g]["records"])}' for t, v in legal[g].most_common(6))] for g in sorted(legal)])
        say('Most frequent text-only address fields (admin areas):\n')
        table(['country', 'src', 'top fields'],
              [[g[0], g[1], ', '.join(f'{t} ({v:,})' for t, v in fields[g].most_common(8))] for g in sorted(fields)])
        say('S1 name-key and address-key multiplicity (share of S1 entities whose key is shared by m entities):\n')
        rows = []
        for c in sorted(nk_count):
            for nm, cnt in (('name key', nk_count[c]), ('address key', ak_count[c])):
                m = Counter()
                for v in cnt.values():
                    m['1' if v == 1 else '2-5' if v <= 5 else '6-20' if v <= 20 else '21+'] += v
                tot = sum(m.values())
                rows.append([c, nm, pct(m['1'] / tot), pct(m['2-5'] / tot), pct(m['6-20'] / tot), pct(m['21+'] / tot),
                             ', '.join(f'"{k}" x{v}' for k, v in cnt.most_common(4))])
        table(['country', 'key', 'm=1', '2-5', '6-20', '21+', 'most shared'], rows)

        # ---------- scores ----------
        Z = np.load(S5, allow_pickle=True)
        ri, p, v5, ce = Z['ref_idx'], Z['prob'].astype(np.float64), Z['stage1'].astype(np.float64), Z['ce'].astype(np.float64)
        tg = Z['target'].astype(str)
        refs = []
        with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie = head.index('entity_id')
            for line in fh:
                refs.append(line.rstrip('\n').split('\t')[ie])
        refs = np.array(refs, dtype=object)
        rc = np.array([s1_country.get(r, '') for r in refs], dtype=object)
        pc = rc[ri]
        ch = p >= THR
        n_ref = len(refs)
        kref = np.bincount(ri, ch.astype(float), n_ref)
        claim = Counter(tg[ch])
        contested = np.array([claim[t] >= 2 for t in tg]) & ch

        def edits(j):
            r, t = refs[ri[j]], tg[j]
            rn, ra = text.get(r, ('', '')); tn, ta = text.get(t, ('', ''))
            a, b = set(toks(rn)), set(toks(tn))
            return {'added': b - a, 'dropped': a - b, 'legal_change': (a & LEG) != (b & LEG),
                    'core_added': bool((b - a) - LEG), 'core_dropped': bool((a - b) - LEG),
                    'blank': not ta.strip(), 'num_eq': bool(num(ta)) and num(ta) == num(ra),
                    'num_diff': bool(num(ta)) and bool(num(ra)) and num(ta) != num(ra),
                    'abbrev': bool(ABBR.search(' ' + ta + ' ')), 'upper': tn.isupper(),
                    'name_uniq': nk_count[rc[ri[j]]][nkey(rn)] == 1, 'addr_uniq': ak_count[rc[ri[j]]][akey(ra)] == 1,
                    'src3': t.startswith('S3')}

        def profile(js, label):
            E = [edits(j) for j in js]
            n = max(len(E), 1)
            add, drop = Counter(), Counter()
            for e in E:
                add.update(e['added']); drop.update(e['dropped'])
            keys = ['core_added', 'core_dropped', 'legal_change', 'blank', 'num_eq', 'num_diff', 'abbrev', 'upper',
                    'name_uniq', 'addr_uniq', 'src3']
            return [label, f'{len(E):,}'] + [pct(sum(e[k] for e in E) / n) for k in keys], add, drop, keys

        # ---------- B. positive noise model ----------
        say('## B. Positive noise model (confident pairs: p>=0.98, v5>=0.95, ce>=5)\n')
        rows, adds = [], {}
        for c in ('France', 'US', 'India'):
            js = np.flatnonzero((pc == c) & (p >= 0.98) & (v5 >= 0.95) & (ce >= 5))
            js = rng.choice(js, min(40000, len(js)), replace=False)
            row, add, drop, keys = profile(js, c)
            rows.append(row); adds[c] = (add, drop, len(js))
        table(['country', 'pairs'] + keys, rows)
        for c, (add, drop, n) in adds.items():
            say(f'- **{c}** added tokens: ' + ', '.join(f'{t} {pct(v / n)}' for t, v in add.most_common(15)))
            say(f'- **{c}** dropped tokens: ' + ', '.join(f'{t} {pct(v / n)}' for t, v in drop.most_common(15)))
        say('')

        # ---------- C. failure buckets ----------
        say('## C. Failure buckets\n')
        buckets = {
            'CE veto (v5>=0.9, ce<0)': (v5 >= 0.9) & (ce < 0),
            'CE override (v5<0.1, ce>2)': (v5 < 0.1) & (ce > 2),
            'uncertain not chosen (0.2<=p<0.68)': (p >= 0.2) & (p < THR),
            'marginal chosen (0.68<=p<0.9)': (p >= THR) & (p < 0.9),
            'contested chosen (>=2 refs claim target)': contested,
        }
        say('Rate per 100 references:\n')
        table(['bucket', 'France', 'US', 'India'],
              [[b] + [f'{100 * (m & (pc == c)).sum() / max((rc == c).sum(), 1):.2f}' for c in ('France', 'US', 'India')]
               for b, m in buckets.items()] +
              [['empty references'] + [pct(((kref == 0) & (rc == c)).sum() / max((rc == c).sum(), 1)) for c in ('France', 'US', 'India')]])
        tops = {}
        for b, m in buckets.items():
            rows = []
            for c in ('France', 'US', 'India'):
                js = np.flatnonzero(m & (pc == c))
                js = rng.choice(js, min(20000, len(js)), replace=False) if len(js) else js
                row, add, drop, keys = profile(js, c)
                rows.append(row)
                if c == 'France':
                    tops[b] = (add, drop, len(js))
            say(f'### {b}\n')
            table(['country', 'pairs'] + keys, rows)
            add, drop, n = tops[b]
            say('- France added: ' + ', '.join(f'{t} {pct(v / max(n, 1))}' for t, v in add.most_common(12)))
            say('- France dropped: ' + ', '.join(f'{t} {pct(v / max(n, 1))}' for t, v in drop.most_common(12)))
            say('')
        say('### Empty references: best candidate\n')
        best = np.full(n_ref, -1); bp = np.zeros(n_ref)
        order = np.lexsort((-p, ri))
        first = np.r_[True, ri[order][1:] != ri[order][:-1]]
        best[ri[order][first]] = order[first]; bp[ri[order][first]] = p[order][first]
        rows = []
        for c in ('France', 'US', 'India'):
            js = best[(kref == 0) & (rc == c) & (best >= 0)]
            js = rng.choice(js, min(20000, len(js)), replace=False)
            row, _, _, keys = profile(js, c)
            rows.append(row + [f'{np.median(p[js]):.3f}'])
        table(['country', 'pairs'] + keys + ['median best p'], rows)

        # ---------- D. case files ----------
        by_ref = defaultdict(list)
        fr = np.flatnonzero(pc == 'France')
        for j in fr:
            by_ref[ri[j]].append(j)
        cases = ['# France case files (S5 test)\n', 'Per reference: all candidates sorted by p; * = chosen at 0.68.\n']
        pick = {b: rng.choice(np.unique(ri[m & (pc == 'France')]), 12, replace=False) for b, m in buckets.items() if (m & (pc == 'France')).any()}
        pick['empty references'] = rng.choice(np.flatnonzero((kref == 0) & (rc == 'France')), 12, replace=False)
        for b, rs in pick.items():
            cases.append(f'\n## {b}\n')
            for r in rs:
                rn, ra = text.get(refs[r], ('', ''))
                cases.append(f'\n**{refs[r]}** {rn} | {ra}  (name shared by {nk_count["France"][nkey(rn)]} S1, address by {ak_count["France"][akey(ra)]} S1)\n')
                for j in sorted(by_ref[r], key=lambda j: -p[j]):
                    tn, ta = text.get(tg[j], ('', ''))
                    cases.append(f'- {"*" if ch[j] else " "} p={p[j]:.2f} v5={v5[j]:.2f} ce={ce[j]:5.1f} claim={claim[tg[j]]} {tg[j][:2]} | {tn} | {ta}')
        (OUT / 'france_eda.md').write_text('\n'.join(md))
        (OUT / 'france_cases.md').write_text('\n'.join(cases))
        print('done')


    if __name__ == '__main__':
        main()



def cmd_france_noise():
    """France diagnosis (label-free): which name words make the cross-encoder reject pairs the v5 model is sure about?

Reads S5's test late_scores.npz (v5 stage-1 probability and mdeberta logit per top-10 test pair). Among pairs with
v5 probability >= 0.9, 'rejected' = cross-encoder logit < 0. For each country: rejection rate, and the words present
on one side only (after lowercasing/accent folding) whose share among rejected pairs most exceeds their share among
accepted pairs (lift). Writes france_noise.json and france_noise_examples.tsv.

  france_noise.py S5_DIR OUT
"""
    import csv
    import json
    import re
    import sys
    import unicodedata
    from collections import Counter
    from pathlib import Path

    import numpy as np

    RAW = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/data/dataset/test')


    def fold(s):
        s = unicodedata.normalize('NFKD', s or '')
        return re.sub(r'[^\w]+', ' ', ''.join(c for c in s if not unicodedata.combining(c)).lower()).split()


    def main():
        s5, out = Path(sys.argv[1]), Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        refs, country, name = [], [], {}
        for src in ('test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv'):
            with open(RAW / src, encoding='utf-8', newline='') as fh:
                for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                    name[row['entity_id']] = row['business_name']
                    if src == 'test_source1.tsv':
                        refs.append(row['entity_id']); country.append(row['country'])
        country = np.array(country, dtype=object)
        z = np.load(s5 / 'late_scores.npz', allow_pickle=True)
        ref_idx, tgt, s1, ce = z['ref_idx'], z['target'], z['stage1'], z['ce']
        sure = np.flatnonzero(s1 >= 0.9)
        R, rows = {}, []
        for c in sorted(set(country)):
            m = sure[country[ref_idx[sure]] == c]
            rej = m[ce[m] < 0]
            acc = m[ce[m] >= 0]
            only_rej, only_acc = Counter(), Counter()
            for pool, counter in ((rej, only_rej), (acc[np.random.default_rng(0).permutation(len(acc))[:200000]], only_acc)):
                for j in pool:
                    a, b = set(fold(name.get(refs[ref_idx[j]], ''))), set(fold(name.get(str(tgt[j]), '')))
                    counter.update(f'+{w}' for w in b - a)   # word added in the candidate
                    counter.update(f'-{w}' for w in a - b)   # word missing from the candidate
            n_rej, n_acc = max(1, len(rej)), max(1, min(len(acc), 200000))
            lift = {w: (only_rej[w] / n_rej) / max(only_acc[w] / n_acc, 1e-4) for w in only_rej if only_rej[w] >= 20}
            top = sorted(lift, key=lambda w: -only_rej[w] * np.log(max(lift[w], 1.0)))[:40]
            R[c] = {'confident_v5_pairs': int(len(m)), 'ce_rejects': int(len(rej)),
                    'reject_rate': round(len(rej) / max(1, len(m)), 4),
                    'words': [(w, only_rej[w], round(only_rej[w] / n_rej, 4), round(lift[w], 1)) for w in top]}
            for j in rej[:: max(1, len(rej) // 40)][:40]:
                rows.append([c, refs[ref_idx[j]], name.get(refs[ref_idx[j]], ''), str(tgt[j]), name.get(str(tgt[j]), ''),
                             f'{s1[j]:.3f}', f'{ce[j]:.2f}'])
            print(c, {k: v for k, v in R[c].items() if k != 'words'}, flush=True)
            print('  top words (word, rejected pairs, share, lift):', R[c]['words'][:20], flush=True)
        (out / 'france_noise.json').write_text(json.dumps(R, indent=1, ensure_ascii=False))
        with open(out / 'france_noise_examples.tsv', 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(['country', 'ref', 'ref_name', 'target', 'target_name', 'v5_prob', 'ce_logit'])
            w.writerows(rows)


    if __name__ == '__main__':
        main()



def cmd_france_numdiff():
    """France pairs by (name relation, house-number/street relation), accepted vs rejected in a submission.

Train says a record with the S1's core name (legal forms ignored) on the S1's street at a different house number is
95-96% a true match (alias_check.py), so rejected France pairs of that kind are restore candidates. Prints counts,
accepted share and p quantiles per cell, plus 30 rejected same-name / other-number / same-street examples with the
number pair. Env LATE, SUB, OUT.
"""
    import os
    import random
    from collections import Counter, defaultdict

    import numpy as np

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    from fr_labels import read_france
    from france_census import core, house, load, street_words, tokens_addr

    SHIFT = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


    def main():
        Z = np.load(os.environ['LATE'], allow_pickle=True)
        ri, tg, p3 = Z['ref_idx'], Z['target'].astype(str), Z['prob']
        match = load(os.environ['SUB'])
        owner = {t: r for r, ts in match.items() for t in ts}
        refs, text, m, s1_core = read_france(ri, tg)
        cell, acc, ps = Counter(), Counter(), defaultdict(list)
        cand = []
        acc_diff = defaultdict(Counter)
        shifted = []
        for j in np.flatnonzero(m):
            r, t = refs[ri[j]], tg[j]
            ra = text[r][1]; tn, ta = text.get(t, ('', ''))
            if not ta.strip():
                continue
            h1, h2 = house(ra)[0], house(ta)[0]
            sw = street_words(ra)
            if not h1 or not h2 or not sw:
                continue
            st = ('same_no' if h1 == h2 else 'diff_no') + ('_street' if sw <= tokens_addr(ta) else '_other')
            rc, tc = set(s1_core[r]), set(core(tn))
            nm = 'same' if rc == tc else ('none' if not rc & tc else 'edit')
            k = (nm, st); a = t in match.get(r, ())
            cell[k] += 1; acc[k] += a; ps[k].append(p3[j])
            if k == ('same', 'diff_no_street') and not a:
                cand.append((j, r, t, h1, h2))
            if nm in ('same', 'edit', 'none') and st == 'diff_no_street' and a:
                try:
                    d = int(''.join(ch for ch in h2 if ch.isdigit())) - int(''.join(ch for ch in h1 if ch.isdigit()))
                except ValueError:
                    d = None
                acc_diff[nm][d if d is None or abs(d) < 30 else ('big+' if d > 0 else 'big-')] += 1
                if nm == 'same' and d in SHIFT:
                    own_same = sum(1 for x in match.get(r, ()) if house(text.get(x, ('', ''))[1])[0] == h1)
                    shifted.append((j, r, t, h1, h2, own_same))
        lines = []
        for k in sorted(cell):
            q = np.quantile(ps[k], [0.1, 0.5, 0.9]).round(3).tolist()
            lines.append(f'{k} pairs {cell[k]} accepted {acc[k] / cell[k]:.3f} p_q10/50/90 {q}')
        free = [c for c in cand if c[2] not in owner]
        lines.append(f'rejected same-name other-number same-street: {len(cand)}, of which record unowned: {len(free)}')
        diffs = Counter()
        for _, _, _, h1, h2 in free:
            try:
                diffs[min(abs(int(''.join(ch for ch in h2 if ch.isdigit())) - int(''.join(ch for ch in h1 if ch.isdigit()))), 99)] += 1
            except ValueError:
                diffs['suffix'] += 1
        lines.append('abs number difference (unowned): ' + str(sorted(diffs.items(), key=lambda kv: -kv[1])[:20]))
        random.seed(2)
        for j, r, t, h1, h2 in random.sample(free, min(30, len(free))):
            lines.append(f'  {p3[j]:.2f} [{h1} -> {h2}] {text[r][0]} | {text[r][1]}  =>  {text[t][0]} | {text[t][1]}')
        for nm, c in acc_diff.items():
            lines.append(f'ACCEPTED {nm} diff_no_street signed diff: ' + str(sorted(c.items(), key=lambda kv: -kv[1])[:25]))
        lines.append(f'ACCEPTED same-name same-street shift-up in SHIFT: {len(shifted)}; S1 also has a link at its own number: '
                     f'{sum(1 for x in shifted if x[5] > 0)}; p q10/50/90 '
                     f'{np.quantile([p3[x[0]] for x in shifted], [0.1, 0.5, 0.9]).round(3).tolist() if shifted else []}')
        for j, r, t, h1, h2, o in random.sample(shifted, min(25, len(shifted))):
            lines.append(f'  S {p3[j]:.2f} own={o} [{h1} -> {h2}] {text[r][0]} | {text[r][1]}  =>  {text[t][0]} | {text[t][1]}')
        out = '\n'.join(lines)
        open(os.environ['OUT'], 'w', encoding='utf-8').write(out)
        print(out, flush=True)


    if __name__ == '__main__':
        main()



def cmd_france_patterns():
    """Size four France patterns found by reading case files, on every France pair (label-free counts + p quantiles).

  typo    census restore test passes except a street word, which has a close typo in the record (similarity >= 0.8)
  drop    only a business-type word dropped and only filler words added, everything else as the strict restore test
          (town + type names, where the rarest-word guard rejects the copy)
  init    record name = initials of the S1's core name, same house number and street words
  blank   blank-address record, p >= 0.8, not accepted: how many are owned by another S1 in the submission
Per pattern: pairs, not accepted now, p quantiles, 12 examples. Env LATE, SUB, OUT.
"""
    import json
    import os
    from collections import Counter, defaultdict

    import numpy as np
    from rapidfuzz.fuzz import ratio

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    from fr_labels import read_france
    from france_census import FILLER, TYPES, core, house, load, street_words, tokens_addr


    def near(w, toks):
        return any(ratio(w, t) >= 80 for t in toks if abs(len(t) - len(w)) <= 2)


    def main():
        Z = np.load(os.environ['LATE'], allow_pickle=True)
        ri, tg, p3 = Z['ref_idx'], Z['target'].astype(str), Z['prob'].astype(np.float64)
        match = load(os.environ['SUB'])
        owner = {t: r for r, ts in match.items() for t in ts}
        refs, text, m, s1_core = read_france(ri, tg)
        freq = Counter(t for c in s1_core.values() for t in set(c))
        hits = defaultdict(list)
        for j in np.flatnonzero(m):
            r, t = refs[ri[j]], tg[j]
            rn, ra = text[r]; tn, ta = text.get(t, ('', ''))
            acc = t in match.get(r, ())
            if not ta.strip():
                if p3[j] >= 0.8 and not acc:
                    hits['blank_owned_elsewhere' if t in owner else 'blank_unowned'].append((j, r, t))
                continue
            (h1, s1n), (h2, s2n) = house(ra), house(ta)
            if not h1 or h1 != h2 or (s1n and s2n and s1n != s2n):
                continue
            rc, tc = s1_core[r], core(tn)
            sw, ta_t = street_words(ra), tokens_addr(ta)
            if not sw:
                continue
            exact_st = sw <= ta_t
            fuzzy_st = all(w in ta_t or near(w, ta_t) for w in sw)
            added, dropped = set(tc) - set(rc), set(rc) - set(tc)
            if rc and tc and len(tc) == 1 and len(rc) >= 2 and tc[0] == ''.join(w[0] for w in rc) and exact_st:
                hits['init'].append((j, r, t))
                continue
            if not rc or not tc or not added <= FILLER:
                continue
            rare_ok = min(rc, key=lambda x: (freq[x], x)) in tc
            first_ok = rc[0] == tc[0]
            if first_ok and rare_ok and fuzzy_st and not exact_st:
                hits['typo'].append((j, r, t))
            elif exact_st and dropped and dropped <= TYPES and first_ok and not rare_ok:
                hits['drop'].append((j, r, t))
        R, ex = {}, {}
        rng = np.random.default_rng(3)
        for k, v in hits.items():
            js = np.array([x[0] for x in v])
            na = [x for x in v if x[2] not in match.get(x[1], ())]
            R[k] = {'pairs': len(v), 'not_accepted': len(na),
                    'p_q10_50_90': [round(float(x), 3) for x in np.quantile(p3[js], [0.1, 0.5, 0.9])],
                    'owned_by_other_s1': sum(x[2] in owner and owner[x[2]] != x[1] for x in na)}
            pick = [na[i] for i in rng.choice(len(na), min(12, len(na)), replace=False)] if na else []
            ex[k] = [f"{p3[j]:.2f} {text[r][0]} | {text[r][1]}  =>  {text[t][0]} | {text[t][1]}"
                     + (f"  [owner {text.get(owner[t], ('?',))[0]} | {text.get(owner[t], ('', '?'))[1]}]" if t in owner else '')
                     for j, r, t in pick]
        out = json.dumps({'stats': R, 'examples': ex}, indent=1, ensure_ascii=False)
        open(os.environ['OUT'], 'w', encoding='utf-8').write(out)
        print(out, flush=True)


    if __name__ == '__main__':
        main()



def cmd_france_patterns2():
    """Size four France patterns found by reading case files, on every France pair (label-free counts + p quantiles).

  typo    census restore test passes except a street word, which has a close typo in the record (similarity >= 0.8)
  drop    only a business-type word dropped and only filler words added, everything else as the strict restore test
          (town + type names, where the rarest-word guard rejects the copy)
  init    record name = initials of the S1's core name, same house number and street words
  blank   blank-address record, p >= 0.8, not accepted: how many are owned by another S1 in the submission
Per pattern: pairs, not accepted now, p quantiles, 12 examples. Env LATE, SUB, OUT.
"""
    import json
    import os
    from collections import Counter, defaultdict

    import numpy as np
    from rapidfuzz.fuzz import ratio

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    from fr_labels import read_france
    from france_census import FILLER, TYPES, core, house, load, street_words, tokens_addr


    def near(w, toks):
        return any(ratio(w, t) >= 80 for t in toks if abs(len(t) - len(w)) <= 2)


    def main():
        Z = np.load(os.environ['LATE'], allow_pickle=True)
        ri, tg, p3 = Z['ref_idx'], Z['target'].astype(str), Z['prob'].astype(np.float64)
        match = load(os.environ['SUB'])
        owner = {t: r for r, ts in match.items() for t in ts}
        refs, text, m, s1_core = read_france(ri, tg)
        freq = Counter(t for c in s1_core.values() for t in set(c))
        hits = defaultdict(list)
        for j in np.flatnonzero(m):
            r, t = refs[ri[j]], tg[j]
            rn, ra = text[r]; tn, ta = text.get(t, ('', ''))
            acc = t in match.get(r, ())
            if not ta.strip():
                if p3[j] >= 0.8 and not acc:
                    hits['blank_owned_elsewhere' if t in owner else 'blank_unowned'].append((j, r, t))
                continue
            (h1, s1n), (h2, s2n) = house(ra), house(ta)
            if not h1 or h1 != h2 or (s1n and s2n and s1n != s2n):
                continue
            rc, tc = s1_core[r], core(tn)
            sw, ta_t = street_words(ra), tokens_addr(ta)
            if not sw:
                continue
            exact_st = sw <= ta_t
            fuzzy_st = all(w in ta_t or near(w, ta_t) for w in sw)
            added, dropped = set(tc) - set(rc), set(rc) - set(tc)
            if exact_st and rc and tc and dropped and not added & TYPES:
                unmatched = {w for w in added if not any(ratio(w, d) >= 75 for d in dropped)}
                if all(any(ratio(d, w) >= 75 for w in added) for d in dropped) and unmatched <= FILLER:
                    hits['nametypo'].append((j, r, t))
                    continue
            if rc and tc and len(tc) == 1 and len(rc) >= 2 and tc[0] == ''.join(w[0] for w in rc) and exact_st:
                hits['init'].append((j, r, t))
                continue
            if not rc or not tc or not added <= FILLER:
                continue
            rare_ok = min(rc, key=lambda x: (freq[x], x)) in tc
            first_ok = rc[0] == tc[0]
            if first_ok and rare_ok and fuzzy_st and not exact_st:
                hits['typo'].append((j, r, t))
            elif exact_st and dropped and dropped <= TYPES and first_ok and not rare_ok:
                hits['drop'].append((j, r, t))
        R, ex = {}, {}
        rng = np.random.default_rng(3)
        for k, v in hits.items():
            js = np.array([x[0] for x in v])
            na = [x for x in v if x[2] not in match.get(x[1], ())]
            R[k] = {'pairs': len(v), 'not_accepted': len(na),
                    'p_q10_50_90': [round(float(x), 3) for x in np.quantile(p3[js], [0.1, 0.5, 0.9])],
                    'owned_by_other_s1': sum(x[2] in owner and owner[x[2]] != x[1] for x in na)}
            pick = [na[i] for i in rng.choice(len(na), min(12, len(na)), replace=False)] if na else []
            ex[k] = [f"{p3[j]:.2f} {text[r][0]} | {text[r][1]}  =>  {text[t][0]} | {text[t][1]}"
                     + (f"  [owner {text.get(owner[t], ('?',))[0]} | {text.get(owner[t], ('', '?'))[1]}]" if t in owner else '')
                     for j, r, t in pick]
        out = json.dumps({'stats': R, 'examples': ex}, indent=1, ensure_ascii=False)
        open(os.environ['OUT'], 'w', encoding='utf-8').write(out)
        print(out, flush=True)


    if __name__ == '__main__':
        main()



def cmd_france_pseudo():
    """France self-training set from T1's test predictions: only pairs that every decider is very sure about.

France has no labels; the rules allow self-training on the provided test records. To limit confirmation bias, a pair
enters only when four independent signals agree at extreme values, and uncertain pairs (the ones we want to fix)
are never used:
  positive  stage-3b p >= 0.995, v5 >= 0.9, chunk2 logit >= 4, listwise logit >= 4, and the pair is in T1's final
            matches (it won global exclusivity)
  negative  stage-3b p <= 0.001, v5 <= 0.02, chunk2 logit <= -4, listwise logit <= -4
  lists     France references whose every candidate is a very-confident positive or negative (for the listwise model)
Only the cross-encoders are fine-tuned on this; fusion and stage-3b stay trained on train labels only.
Env T1_DIR. Writes artifacts/fr_pseudo/{pairs.npz, lists.npz, stats.json}.
"""
    import json
    import os
    from collections import defaultdict
    from pathlib import Path

    import numpy as np

    HOME = Path.home()
    T1 = Path(os.environ['T1_DIR'])
    RAW = Path(os.environ.get('TEST_RAW', str(HOME / 'scratch/amazon_ml_2026/data/dataset/test')))
    OUT = Path(os.environ.get('OUT_DIR', str(HOME / 'scratch/amazon_ml_2026/artifacts/fr_pseudo')))
    NO_LW = os.environ.get('NO_LW', '') == '1'  # base without a listwise column (e.g. T1b): use the three other deciders


    def main():
        OUT.mkdir(parents=True, exist_ok=True)
        Z = np.load(T1 / 'late_scores.npz', allow_pickle=True)
        ri, tg = Z['ref_idx'], Z['target'].astype(str)
        p3, v5, c2 = Z['prob'].astype(np.float64), Z['stage1'].astype(np.float64), Z['ce'].astype(np.float64)
        lw = None if NO_LW else np.load(T1 / 'features.npy', mmap_mode='r')[:, -1].astype(np.float64)
        refs, cty = [], []
        with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, ic = head.index('entity_id'), head.index('country')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                refs.append(q[ie]); cty.append(q[ic] if len(q) > ic else '')
        refs, cty = np.array(refs, dtype=object), np.array(cty, dtype=object)
        sub = next(d for d in T1.iterdir() if (d / 'matching_results.tsv').exists())
        final = set()
        with open(sub / 'matching_results.tsv', encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                s, m = line.rstrip('\n').split('\t') if '\t' in line else (line.strip(), '')
                final.update((s, t) for t in m.split(',') if t)
        fr = cty[ri] == 'France'
        rid = refs[ri]
        in_final = np.array([(r, t) in final for r, t in zip(rid, tg)]) & fr
        pos = fr & in_final & (p3 >= 0.995) & (v5 >= 0.9) & (c2 >= 4)
        neg = fr & (p3 <= 0.001) & (v5 <= 0.02) & (c2 <= -4)
        if lw is not None:
            pos &= lw >= 4
            neg &= lw <= -4
        sure = pos | neg
        by_ref = defaultdict(list)
        for j in np.flatnonzero(fr):
            by_ref[ri[j]].append(j)
        full = [r for r, js in by_ref.items() if all(sure[j] for j in js) and any(pos[j] for j in js)]
        lj = np.array([j for r in full for j in by_ref[r]], dtype=np.int64)
        pj = np.flatnonzero(sure)
        np.savez(OUT / 'pairs.npz', ref=rid[pj].astype(str), target=tg[pj], label=pos[pj].astype(np.int8))
        np.savez(OUT / 'lists.npz', ref=rid[lj].astype(str), target=tg[lj], label=pos[lj].astype(np.int8))
        R = {'france_refs': int((cty == 'France').sum()), 'france_pairs': int(fr.sum()),
             'france_pairs_in_T1_matches': int(in_final.sum()), 'pseudo_pos': int(pos.sum()), 'pseudo_neg': int(neg.sum()),
             'pseudo_share_of_france_pairs': round(float(sure.sum() / fr.sum()), 4),
             'pos_share_of_T1_france_matches': round(float(pos.sum() / max(in_final.sum(), 1)), 4),
             'fully_confident_lists': len(full), 'list_rows': int(len(lj)),
             'excluded_uncertain_pairs': int((fr & ~sure).sum())}
        (OUT / 'stats.json').write_text(json.dumps(R, indent=1))
        print(json.dumps(R, indent=1), flush=True)


    if __name__ == '__main__':
        main()



def cmd_france_rates():
    """Per-country, label-free: links per reference, and how often each kind of candidate is accepted (T5 + census):
same-words blank-address copies, legal-form-only, filler words at the same address, same house number on another street,
type-word swaps. If France accepts a copy kind far less often than US/India, that kind is where France under-links."""
    import csv, json, re, sys
    from collections import Counter, defaultdict
    from pathlib import Path
    import numpy as np
    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    sys.path.insert(0, str(ME / 'business_entity_resolution/src/ayush'))
    from gen4_test_side import kind, fold  # noqa: E402
    refs, cty, text = [], {}, {}
    for src in ('test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv'):
        with open(ME / 'data/dataset/test' / src, encoding='utf-8', newline='') as fh:
            for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                text[row['entity_id']] = (row['business_name'] or '', row['business_address'] or '')
                if src == 'test_source1.tsv':
                    refs.append(row['entity_id']); cty[row['entity_id']] = row['country']
    def load(p):
        with open(p, encoding='utf-8') as f:
            next(f); return {l.split('\t')[0]: set(x for x in l.rstrip('\n').split('\t')[1].split(',') if x) for l in f}
    m = load(ME / 'artifacts/submission_T5_census/expected_0.66/matching_results.tsv')
    Z = np.load(ME / 'artifacts/submission_T5_own/late_scores.npz', allow_pickle=True)
    ri, tg, pr = Z['ref_idx'], Z['target'].astype(str), Z['prob']
    def street(a):
        f = [x for x in fold(a) if not x.isdigit()]
        return ' '.join(f[:3])
    st = defaultdict(lambda: defaultdict(Counter))
    for i, t, p in zip(ri, tg, pr):
        r = refs[i]; c = cty[r]
        rn, ra = text[r]; tn, ta = text.get(t, ('', ''))
        k = kind(rn, ra, tn, ta)
        rnum = re.search(r'\d+', ra or ''); tnum = re.search(r'\d+', ta or '')
        if 'blank_address' not in k and rnum and tnum and int(rnum.group()) == int(tnum.group()) and street(ra) and street(ta) and \
                len(set(fold(ra)) & set(fold(ta)) - {rnum.group(), tnum.group()}) <= 1:
            k = 'same_number_other_street:' + k
        acc = t in m[r]
        b = 'hi' if p >= 0.5 else 'mid' if p >= 0.1 else 'lo'
        st[c][k]['n'] += 1; st[c][k]['acc'] += acc; st[c][k]['p_' + b] += 1; st[c][k]['acc_' + b] += acc
    R = {}
    for c in ('France', 'India', 'US'):
        rs = [r for r in refs if cty[r] == c]
        dist = Counter(min(len(m[r]), 8) for r in rs)
        R[c] = {'refs': len(rs), 'links_per_ref': round(sum(len(m[r]) for r in rs) / len(rs), 3),
                'links_dist': {k: round(v / len(rs), 4) for k, v in sorted(dist.items())},
                'kinds': {k: {'n': v['n'], 'acc_rate': round(v['acc'] / v['n'], 4), 'per_ref': round(v['n'] / len(rs), 3),
                              'accepted_per_ref': round(v['acc'] / len(rs), 3), 'mid_p_n': v['p_mid'], 'mid_p_acc': v['acc_mid'],
                              'hi_p_n': v['p_hi'], 'hi_p_acc': v['acc_hi']}
                          for k, v in sorted(st[c].items(), key=lambda kv: -kv[1]['acc']) if v['n'] >= 200}}
    (ME / 'artifacts/gen4_eda/france_rates.json').write_text(json.dumps(R, indent=1))
    for c, v in R.items():
        print('==', c, 'links/ref', v['links_per_ref'], 'dist', v['links_dist'])
        for k, s in list(v['kinds'].items())[:18]:
            print('   %-55s n/ref %.3f acc %.3f acc/ref %.3f | mid %d->%d hi %d->%d' % (k, s['per_ref'], s['acc_rate'], s['accepted_per_ref'], s['mid_p_n'], s['mid_p_acc'], s['hi_p_n'], s['hi_p_acc']))



def cmd_france_review():
    """Readable French case dump for manual review (label-free): T5 + census decisions vs S5, with T5 probabilities.

Groups: A random refs, B uncertain same-words blank pairs, C uncertain other-name edits, D refs emptied vs S5,
E uncertain legal-form pairs, F census restores (low p, now accepted), G census rejections (high p, now rejected),
H refs with the most accepted links. Writes OUT/france_review.txt and france_review_stats.json.
"""
    import csv, json, re, sys, unicodedata
    from collections import Counter, defaultdict
    from pathlib import Path
    import numpy as np
    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    A = Path('/scratch/scai/mtech/aib262015/amazon_ml_2026')
    OUT = ME / 'artifacts/gen4_eda'
    sys.path.insert(0, str(ME / 'business_entity_resolution/src/ayush'))
    from gen4_test_side import kind  # noqa: E402
    RAW = ME / 'data/dataset/test'
    def load(p):
        with open(p, encoding='utf-8') as f:
            next(f); return {l.split('\t')[0]: set(x for x in l.rstrip('\n').split('\t')[1].split(',') if x) for l in f}
    refs, cty, text = [], {}, {}
    for src in ('test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv'):
        with open(RAW / src, encoding='utf-8', newline='') as fh:
            for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                text[row['entity_id']] = (row['business_name'] or '', row['business_address'] or '')
                if src == 'test_source1.tsv':
                    refs.append(row['entity_id']); cty[row['entity_id']] = row['country']
    t5c = load(ME / 'artifacts/submission_T5_census/expected_0.66/matching_results.tsv')
    t5r = load(ME / 'artifacts/submission_T5_own/expected_0.66/matching_results.tsv')
    s5 = load(A / 'artifacts/submission_S5/threshold_0.68/matching_results.tsv')
    Z = np.load(ME / 'artifacts/submission_T5_own/late_scores.npz', allow_pickle=True)
    ri, tg, pr = Z['ref_idx'], Z['target'].astype(str), Z['prob'].astype(np.float64)
    fr_idx = np.flatnonzero(np.array([cty[refs[i]] == 'France' for i in ri]))
    cands = defaultdict(list)
    for j in fr_idx:
        cands[refs[ri[j]]].append((tg[j], pr[j]))
    rng = np.random.default_rng(7)
    frefs = sorted(cands)
    groups = defaultdict(list)
    for r in rng.permutation(frefs)[:25]:
        groups['A random French references'].append(r)
    unc = defaultdict(list)
    for r in frefs:
        for t, p in cands[r]:
            k = kind(*text[r], *text.get(t, ('', '')))
            if 0.2 <= p <= 0.8:
                unc[k].append(r)
            if t in t5c[r] and t not in t5r[r]:
                unc['__restored'].append(r)
            if t in t5r[r] and t not in t5c[r] and p > 0.9:
                unc['__rejected'].append(r)
    def pick(lst, n):
        lst = sorted(set(lst)); return [lst[i] for i in rng.permutation(len(lst))[:n]]
    groups['B uncertain: same words + blank address'] = pick(unc['same_words+blank_address'], 20)
    groups['C uncertain: other name edit'] = pick(unc['other_name_edit'], 15)
    groups['D emptied vs S5'] = pick([r for r in frefs if s5[r] and not t5c[r]], 15)
    groups['E uncertain: legal form (with or without address)'] = pick(unc['legal_form_only'] + unc['legal_form_only+blank_address'], 15)
    groups['F census restores (low p, now accepted)'] = pick(unc['__restored'], 12)
    groups['G census rejections (p > 0.9, now rejected)'] = pick(unc['__rejected'], 12)
    groups['H most accepted links'] = sorted(frefs, key=lambda r: -len(t5c[r]))[:6]
    lines = []
    for g, rs in groups.items():
        lines.append('#' * 110); lines.append(f'# {g} ({len(rs)} references)'); lines.append('#' * 110)
        for r in rs:
            rn, ra = text[r]
            lines.append(f'\nREF {r} | {rn} | {ra}   [T5c {len(t5c[r])} links, S5 {len(s5[r])}]')
            for t, p in sorted(cands[r], key=lambda x: -x[1]):
                tn, ta = text.get(t, ('', ''))
                mark = ('T5c' if t in t5c[r] else '   ') + ('/S5' if t in s5[r] else '   ')
                lines.append(f'  {mark} p={p:.3f} {t[:2]} | {tn} | {ta or "<blank>"}   ({kind(rn, ra, tn, ta)})')
    (OUT / 'france_review.txt').write_text('\n'.join(lines), encoding='utf-8')
    nlinks = Counter(min(len(t5c[r]), 8) for r in frefs)
    stats = {'france_refs': len(frefs), 'links_per_ref_dist_T5c': dict(sorted(nlinks.items())),
             'uncertain_pairs_by_kind': {k: len(v) for k, v in unc.items()}}
    (OUT / 'france_review_stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))



def cmd_france_robust_report():
    """France-robust vs T5 on France: links per reference, empty share, added/removed links (pre-census vs T5 raw, and
with census vs T5 + census), and the France-only transfer audit of the probability (robust vs T5, KS on pairs v5 and
chunk2 agree on, sample vs test France). Writes artifacts/submission_FRrobust/france_report.json."""
    import csv
    import json
    from pathlib import Path

    import numpy as np

    A = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    OUT = ME / 'artifacts/submission_FRrobust'


    def ks(a, b):
        a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
        g = np.unique(np.concatenate([a, b]))
        if len(g) > 4000:
            g = np.quantile(g, np.linspace(0, 1, 4000))
        return float(np.max(np.abs(np.searchsorted(a, g, side='right') / len(a) - np.searchsorted(b, g, side='right') / len(b))))


    def load(p):
        with open(p, encoding='utf-8') as f:
            next(f)
            return {l.split('\t')[0]: set(x for x in l.rstrip('\n').split('\t')[1].split(',') if x) for l in f}


    def main():
        cty, refs = {}, []
        with open(ME / 'data/dataset/test/test_source1.tsv', encoding='utf-8', newline='') as fh:
            for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                refs.append(row['entity_id']); cty[row['entity_id']] = row['country']
        fr = [r for r in refs if cty[r] == 'France']
        subs = {'T5 raw': next((ME / 'artifacts/submission_T5_own').glob('expected_*')) / 'matching_results.tsv',
                'T5 + census': next((ME / 'artifacts/submission_T5_census').glob('expected_*')) / 'matching_results.tsv',
                'robust raw': next(d for d in OUT.iterdir() if d.is_dir() and (d / 'matching_results.tsv').exists()) / 'matching_results.tsv',
                'robust + census': next((ME / 'artifacts/submission_FRrobust_census').glob('*/matching_results.tsv'))}
        M = {k: load(v) for k, v in subs.items()}
        R = {'france_refs': len(fr)}
        for k, m in M.items():
            R[k] = {'links_per_ref': round(sum(len(m[r]) for r in fr) / len(fr), 3),
                    'empty_share': round(sum(not m[r] for r in fr) / len(fr), 4)}
        for a, b in (('robust raw', 'T5 raw'), ('robust + census', 'T5 + census')):
            add = sum(len(M[a][r] - M[b][r]) for r in fr); rem = sum(len(M[b][r] - M[a][r]) for r in fr)
            R[f'{a} vs {b}'] = {'links_added': add, 'links_removed': rem,
                                'refs_changed': sum(M[a][r] != M[b][r] for r in fr),
                                'refs_emptied': sum(bool(M[b][r]) and not M[a][r] for r in fr),
                                'refs_filled': sum(bool(M[a][r]) and not M[b][r] for r in fr)}
        us_in_same = all(M['robust + census'][r] == M['T5 + census'][r] for r in refs if cty[r] != 'France')
        R['US_India_identical_to_T5_census'] = us_in_same
        # France-only transfer audit of the probability
        S = np.load(A / 'artifacts/fusion/stage1.npz')
        v5s = np.where(S['stage1'] < 0.02, 0.0, S['stage1'])
        c2s = np.nan_to_num(np.load('/scratch/scai/mtech/aib262140/amazon_ml_2026/ftchunk2/ce_logit_sample.npy'))
        ps = {'robust': np.load(OUT / 'robust_sample_prob.npy'), 'T5': np.load(ME / 'artifacts/fusion/boundary_models_t5/stage3b_prob.npy')}
        Z = np.load(OUT / 'late_scores.npz', allow_pickle=True)
        pt = {'robust': Z['prob'], 'T5': Z['prob_t5']}
        frr = np.array([cty[refs[i]] == 'France' for i in Z['ref_idx']])
        v5t, c2t = Z['stage1'], Z['ce']
        groups = {'sure_match': ((v5s >= 0.99) & (c2s >= 9), frr & (v5t >= 0.99) & (c2t >= 9)),
                  'sure_nonmatch': ((v5s <= 0.02) & (c2s <= -5), frr & (v5t <= 0.02) & (c2t <= -5))}
        rng = np.random.default_rng(0)
        R['france_prob_audit_KS'] = {}
        for g, (ms, mt) in groups.items():
            js = rng.choice(np.flatnonzero(ms), min(200000, int(ms.sum())), replace=False)
            jt = rng.choice(np.flatnonzero(mt), min(400000, int(mt.sum())), replace=False)
            R['france_prob_audit_KS'][g] = {m: round(ks(ps[m][js], pt[m][jt]), 4) for m in ('T5', 'robust')}
        uj = frr & (Z['prob_t5'] >= 0.2) & (Z['prob_t5'] <= 0.8)
        R['france_uncertain_band_rows'] = {'T5': int(uj.sum()), 'robust': int((frr & (Z['prob'] >= 0.2) & (Z['prob'] <= 0.8)).sum())}
        (OUT / 'france_report.json').write_text(json.dumps(R, indent=1))
        print(json.dumps(R, indent=1))


    if __name__ == '__main__':
        main()



def cmd_france_trained_report():
    """France-trained vs France-robust (both with census) on France: links per reference, empty share, added/removed links,
and how the unlabelled pairs (not in fr_labels) moved: probability shift and acceptance by kind (blank copies especially).
Writes artifacts/submission_FRtrained/france_report.json."""
    import csv
    import json
    import sys
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np

    A = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    sys.path.insert(0, str(ME / 'business_entity_resolution/src/ayush'))
    from gen4_test_side import kind  # noqa: E402

    import os
    OUT = ME / 'artifacts' / os.environ.get('FRT_OUT', 'submission_FRtrained')


    def load(p):
        with open(p, encoding='utf-8') as f:
            next(f)
            return {l.split('\t')[0]: set(x for x in l.rstrip('\n').split('\t')[1].split(',') if x) for l in f}


    def first(d):
        return next(x for x in d.iterdir() if x.is_dir() and (x / 'matching_results.tsv').exists()) / 'matching_results.tsv'


    def main():
        refs, cty, text = [], {}, {}
        for src in ('test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv'):
            with open(ME / 'data/dataset/test' / src, encoding='utf-8', newline='') as fh:
                for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                    if src == 'test_source1.tsv':
                        refs.append(row['entity_id']); cty[row['entity_id']] = row['country']
                    text[row['entity_id']] = (row['business_name'] or '', row['business_address'] or '')
        fr = [r for r in refs if cty[r] == 'France']
        M = {'FRtrained raw': load(first(OUT)), 'FRtrained + census': load(first(ME / 'artifacts' / (OUT.name + '_census'))),
             'FRrobust + census': load(first(ME / 'artifacts/submission_FRrobust_census')),
             'FRrobust + census v2': load(first(ME / 'artifacts/submission_FRrobust_census_v2')),
             'T5 + census': load(first(ME / 'artifacts/submission_T5_census'))}
        R = {'france_refs': len(fr)}
        for k, m in M.items():
            R[k] = {'links_per_ref': round(sum(len(m[r]) for r in fr) / len(fr), 3), 'empty_share': round(sum(not m[r] for r in fr) / len(fr), 4)}
        a, b = M['FRtrained + census'], M['FRrobust + census v2']
        R['FRtrained+census v2 vs FRrobust+census v2'] = {
            'links_added': sum(len(a[r] - b[r]) for r in fr), 'links_removed': sum(len(b[r] - a[r]) for r in fr),
            'refs_changed': sum(a[r] != b[r] for r in fr), 'refs_emptied': sum(bool(b[r]) and not a[r] for r in fr),
            'refs_filled': sum(bool(a[r]) and not b[r] for r in fr)}
        b1 = M['FRrobust + census']
        R['FRtrained+census v2 vs FRrobust+census v1'] = {'links_added': sum(len(a[r] - b1[r]) for r in fr),
                                                          'links_removed': sum(len(b1[r] - a[r]) for r in fr)}
        R['US_India_identical_to_T5_census'] = all(M['FRtrained + census'][r] == M['T5 + census'][r] for r in refs if cty[r] != 'France')
        # unlabelled France pairs
        Z = np.load(OUT / 'late_scores.npz', allow_pickle=True)
        ri, tg, pn, prb = Z['ref_idx'], Z['target'].astype(str), Z['prob'], Z['prob_robust']
        L = np.load(A / 'artifacts/fr_labels/fr_labels.npz', allow_pickle=True)
        labelled = np.zeros(len(tg), dtype=bool); labelled[L['idx']] = True
        frr = np.array([cty[refs[i]] == 'France' for i in ri])
        un = np.flatnonzero(frr & ~labelled)
        edges = [0, 0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98, 1.0001]
        R['unlabelled_p_histogram'] = {'bins': [f'{edges[i]}-{min(edges[i + 1], 1.0)}' for i in range(len(edges) - 1)],
                                       'FRrobust': np.histogram(prb[un], edges)[0].tolist(),
                                       'FRtrained': np.histogram(pn[un], edges)[0].tolist()}
        st = defaultdict(Counter)
        for j in un:
            r, t = refs[ri[j]], tg[j]
            k = kind(*text[r], *text.get(t, ('', '')))
            c = st['ALL unlabelled']; d = st[k]
            for s in (c, d):
                s['n'] += 1; s['acc_before'] += t in b[r]; s['acc_after'] += t in a[r]
                s['p_up_0.1'] += pn[j] - prb[j] >= 0.1; s['p_down_0.1'] += prb[j] - pn[j] >= 0.1
                s['sum_p_before'] += float(prb[j]); s['sum_p_after'] += float(pn[j])
        R['unlabelled_france_pairs'] = {k: {'n': v['n'], 'accepted_before': v['acc_before'], 'accepted_after': v['acc_after'],
                                            'mean_p_before': round(v['sum_p_before'] / v['n'], 4), 'mean_p_after': round(v['sum_p_after'] / v['n'], 4),
                                            'p_up_ge_0.1': v['p_up_0.1'], 'p_down_ge_0.1': v['p_down_0.1']}
                                        for k, v in sorted(st.items(), key=lambda kv: -kv[1]['n']) if v['n'] >= 300}
        (OUT / 'france_report.json').write_text(json.dumps(R, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
        print(json.dumps(R, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


    if __name__ == '__main__':
        main()



def cmd_france_veto():
    """France AND-veto: a France match survives only if both the base submission and the self-trained view accept it.

Self-training on a country without labels is used only to veto. US/India matches are the base's, unchanged. The output folder
mirrors a submission folder (rule subfolder + late_scores.npz link) so france_census.py can run on it next.
Env BASE_DIR, SELF_DIR, OUT_DIR. Writes OUT_DIR/<rule>/matching_results.tsv, candidate_pairs.tsv, veto_report.json.
"""
    import json
    import os
    import shutil
    from pathlib import Path

    HOME = Path.home()
    RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/test'
    BASE, SELF, OUT = Path(os.environ['BASE_DIR']), Path(os.environ['SELF_DIR']), Path(os.environ['OUT_DIR'])


    def load(f):
        d = {}
        with open(f, encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                s, m = (line.rstrip('\n').split('\t') + [''])[:2]
                d[s] = [x for x in m.split(',') if x]
        return d


    def main():
        rd = next(d for d in BASE.iterdir() if (d / 'matching_results.tsv').exists())
        sd = next(d for d in SELF.iterdir() if (d / 'matching_results.tsv').exists())
        base, selfv = load(rd / 'matching_results.tsv'), load(sd / 'matching_results.tsv')
        refs, cty = [], {}
        with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, ic = head.index('entity_id'), head.index('country')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                refs.append(q[ie]); cty[q[ie]] = q[ic]
        out, R = {}, {'france_links_base': 0, 'france_links_vetoed': 0, 'france_refs_emptied_by_veto': 0,
                      'france_links_self_only_ignored': 0}
        for r in refs:
            b = base.get(r, [])
            if cty[r] != 'France':
                out[r] = b
                continue
            s = set(selfv.get(r, []))
            keep = [t for t in b if t in s]
            R['france_links_base'] += len(b)
            R['france_links_vetoed'] += len(b) - len(keep)
            R['france_refs_emptied_by_veto'] += bool(b) and not keep
            R['france_links_self_only_ignored'] += len(s - set(b))
            out[r] = keep
        dest = OUT / rd.name
        dest.mkdir(parents=True, exist_ok=True)
        with open(dest / 'matching_results.tsv', 'w', encoding='utf-8') as f:
            f.write('source1_entity_id\tmatched_entity_ids\n')
            for r in refs:
                f.write(r + '\t' + ','.join(out[r]) + '\n')
        shutil.copy(rd / 'candidate_pairs.tsv', dest / 'candidate_pairs.tsv')
        link = OUT / 'late_scores.npz'
        if not link.exists():
            os.symlink((BASE / 'late_scores.npz').resolve(), link)
        (OUT / 'veto_report.json').write_text(json.dumps(R, indent=1))
        print(json.dumps(R, indent=1), flush=True)


    if __name__ == '__main__':
        main()



def cmd_gen2_eda():
    """Matcher EDA for the S5 generation: is the remaining loss in the ranking or in the stop decision?

Same questions as matcher_eda.py (v5 alone), asked of v5 alone ('gen1') and of the S5 late fusion ('gen2': v5 prob +
full-train mdeberta logit, L2 XGBoost, decoy w=3, trained on trainA+trainB like late_full_train.py). Rule chosen on
dev, everything reported on holdout. Loss is split per reference into
  blocking  = 1 - F05(gold, gold & candidates)
  ranking   = blocking ceiling - best top-k F05 under the model's own ordering
  boundary  = best top-k - achieved   (singleton_fp / missed_all / over / under)
Plus: calibration, margins of boundary errors, singleton detection, strata, FP typing, error examples.
Env MEMBERS 'name=path;...' adds cross-encoder logit columns to gen2. Writes fusion/gen2_eda.json + gen2_examples.tsv.
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
    import xgboost as xgb

    sys.path.insert(0, str(Path(__file__).parent))
    from distractor_check import decide, f05, partition_of

    HOME = Path.home()
    V4 = HOME / 'scratch/vishal_v4_artifacts'
    PRUNE = HOME / 'scratch/amazon_ml_2026/artifacts/prune_v1_copy'
    FUS = HOME / 'scratch/amazon_ml_2026/artifacts/fusion'
    RAW = HOME / 'scratch/amazon_ml_2026/data/dataset/train'
    MEMBERS = os.environ.get('MEMBERS', 'mdebfull=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/mdeberta_full/ce_logit_sample.npy')
    W = float(os.environ.get('DECOY_W', 3))


    def log(*a):
        print(time.strftime('%H:%M:%S'), *a, flush=True)


    def norm(s):
        s = unicodedata.normalize('NFKD', s or '')
        s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
        return re.sub(r'[^\w]+', ' ', s).strip()


    def q(a):
        a = np.asarray(a, dtype=np.float64)
        return {k: round(float(np.quantile(a, v)), 4) for k, v in (('p10', .1), ('p25', .25), ('p50', .5), ('p75', .75), ('p90', .9))} | {'n': int(len(a))} if len(a) else {'n': 0}


    def best_k(order_t, g):
        """Best top-k F05 and smallest k achieving it; F05 = 1.25 tp / (0.25 |g| + k)."""
        if not g:
            return 1.0, 0
        best, bk, tp = 0.0, 0, 0
        for k, t in enumerate(order_t, 1):
            tp += t in g
            v = 1.25 * tp / (0.25 * len(g) + k)
            if v > best + 1e-12:
                best, bk = v, k
        return best, bk


    def main():
        gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
        countries = json.loads((V4 / 'pairs/countries.json').read_text())
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
        cols, names = [s1f], []
        for item in MEMBERS.split(';'):
            n, p = item.split('=', 1)
            if Path(p).exists():
                cols.append(np.nan_to_num(np.load(p).astype(np.float64), nan=0.0)); names.append(n)
            else:
                log('skip missing', n)
        owned = set()
        with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                r = line.rstrip('\n').split('\t')
                if len(r) > 1 and r[1]:
                    owned.update(t.strip() for t in r[1].split(','))
        decoy = np.array([t not in owned for t in tgts])
        wt = np.where((y == 0) & decoy, W, 1.0).astype(np.float32)
        tr, dv = np.isin(part, ['trainA', 'trainB']), part == 'dev'
        idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
        prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
        rix = {p: np.array([{s: i for i, s in enumerate(prefs[p])}[r] for r in refs[idx[p]]]) for p in idx}

        X = np.column_stack(cols).astype(np.float32)
        params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                      max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)
        dtr = xgb.QuantileDMatrix(X[tr], y[tr], weight=wt[tr])
        bst = xgb.train(params, dtr, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=dtr), 'dev')],
                        early_stopping_rounds=100, verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        P2 = {p: np.asarray(bst.inplace_predict(X[idx[p]], iteration_range=it), dtype=np.float64) for p in idx}
        P1 = {p: s1f[idx[p]] for p in idx}
        CE = {p: cols[1][idx[p]] for p in idx} if names else None
        log('L2 trained', names, bst.best_iteration)

        ho = idx['holdout']
        need = set(refs[ho]) | set(tgts[ho])
        text = {}
        for k in (1, 2, 3):
            with open(RAW / f'train_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
                for line in fh:
                    r = line.rstrip('\n').split('\t')
                    if len(r) > iaddr and r[ie] in need:
                        text[r[ie]] = (r[iname], r[iaddr])
        log('text loaded', len(text))

        def macro(p, prob, ru, t):
            c = decide(rix[p], prob, ru, t, len(prefs[p]))
            pred = defaultdict(set)
            for i, tt in zip(rix[p][c], tgts[idx[p]][c]):
                pred[i].add(tt)
            return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])])), c

        grid = [(ru, round(float(t), 2)) for ru in ('expected', 'threshold') for t in np.arange(0.3, 0.981, 0.02)]
        by_ref = defaultdict(list)
        for j, r in enumerate(refs[ho]):
            by_ref[r].append(j)
        H = prefs['holdout']
        n = len(H)
        yh, dh, th = y[ho], decoy[ho], tgts[ho]
        E = {'members': names, 'best_iteration': int(bst.best_iteration), 'n_holdout_refs': n, 'n_holdout_pairs': int(len(ho))}
        examples, nex = [], Counter()

        for gname, P in (('gen1_v5', P1), ('gen2_s5', P2)):
            dev = {g: macro('dev', P['dev'], *g)[0] for g in grid}
            rule = max(dev, key=lambda k: (dev[k], k[1]))
            score, chosen = macro('holdout', P['holdout'], *rule)
            ph = P['holdout']
            G = {'rule': rule, 'dev': round(dev[rule], 6), 'holdout': round(score, 6)}
            loss = Counter()
            cnt = Counter()
            ceil = defaultdict(float)
            margins = defaultdict(list)
            strata = defaultdict(lambda: [0, 0.0])
            top_s, top_ns = [], []
            rank_tp, rank_fn, rank_fp = Counter(), Counter(), Counter()
            fp_type = Counter()
            kdiff = Counter()
            sep = 0
            for s in H:
                g = gold[s]
                js = by_ref.get(s, [])
                order = sorted(js, key=lambda j: -ph[j])
                ot = [th[j] for j in order]
                cand = set(ot)
                pred = {th[j] for j in js if chosen[j]}
                kp = len(pred)
                sc = f05(g, pred)
                perfect = f05(g, g & cand)
                ok, ks = best_k(ot, g)
                ceil['achieved'] += sc; ceil['oracle_topk'] += ok; ceil['blocking'] += perfect
                ceil['oracle_card'] += f05(g, set(ot[:len(g)])) if g else 1.0
                for nm, alt in (('oracle_topk_rank_v5', P1['holdout']), ('oracle_topk_rank_ce', CE['holdout'] if CE else None)):
                    if alt is not None:
                        ceil[nm] += best_k([th[j] for j in sorted(js, key=lambda j: -alt[j])], g)[0]
                ceil['singletons_perfect'] += (1.0 if not g else sc)
                ceil['never_empty'] += (f05(g, {ot[0]}) if g and not pred and ot else sc)
                gi = [r for r, t in enumerate(ot) if t in g]
                ngi = [r for r, t in enumerate(ot) if t not in g]
                sep += (not gi or not ngi or max(gi) < min(ngi))
                loss['blocking'] += 1 - perfect
                loss['ranking'] += perfect - ok
                b = ok - sc
                if not g and kp:
                    kind = 'singleton_fp'
                elif g and not kp:
                    kind = 'missed_all'
                elif kp > ks:
                    kind = 'over'
                elif kp < ks:
                    kind = 'under'
                else:
                    kind = 'none'
                if b > 1e-12:
                    loss['boundary_' + kind] += b
                    cnt[kind] += 1
                    if kind in ('over', 'singleton_fp'):
                        margins[kind].extend(ph[j] for j in order[ks:kp])
                    else:
                        margins[kind].extend(ph[j] for j in order[kp:ks] if th[j] in g)
                    if nex[gname] < 3000:
                        nex[gname] += len(order)
                        rn, ra = text.get(s, ('', ''))
                        for r, j in enumerate(order):
                            tn, ta = text.get(th[j], ('', ''))
                            examples.append((gname, kind, s, rn, ra, r + 1, th[j], tn, ta, round(ph[j], 3),
                                             int(th[j] in g), int(bool(chosen[j]))))
                (top_ns if g else top_s).append(ph[order[0]] if order else 0.0)
                if g:
                    kdiff[max(-3, min(3, kp - len(g & cand)))] += 1
                for r, j in enumerate(order):
                    t = th[j]
                    if t in g and chosen[j]:
                        rank_tp[r + 1] += 1
                    elif t in g:
                        rank_fn[r + 1] += 1
                    elif chosen[j]:
                        rank_fp[r + 1] += 1
                        fp_type['decoy' if dh[j] else 'owned_by_other'] += 1
                rn, ra = text.get(s, ('', ''))
                exact = any(norm(rn) and norm(rn) == norm(text.get(t, ('', ''))[0]) for t in ot)
                for key in (f'country={countries.get(s, "UNK")}',
                            f'gold={min(len(g), 6)}' + ('+' if len(g) >= 6 else ''),
                            f'ref_addr_empty={not ra.strip()}', f'exact_name_in_cands={exact}',
                            f'ncand={"<5" if len(js) < 5 else "5-9" if len(js) < 10 else "10"}'):
                    strata[key][0] += 1; strata[key][1] += 1 - sc
            G['ceilings'] = {k: round(v / n, 6) for k, v in ceil.items()}
            G['loss_points'] = {k: round(v / n * 100, 4) for k, v in loss.items()}
            G['loss_points']['total'] = round((1 - score) * 100, 4)
            G['boundary_error_refs'] = dict(cnt)
            G['boundary_margins'] = {k: q(v) for k, v in margins.items()}
            G['perfectly_separable_refs'] = round(sep / n, 5)
            G['k_pred_minus_k_reachable'] = {int(k): v for k, v in sorted(kdiff.items())}
            G['singleton_top_prob'] = q(top_s)
            G['nonsingleton_top_prob'] = q(top_ns)
            ts, tn = np.array(top_s), np.array(top_ns)
            G['singleton_auc_top_prob'] = round(float((tn[:, None] > ts[None, :]).mean() + 0.5 * (tn[:, None] == ts[None, :]).mean()), 5) if len(ts) and len(tn) else None
            G['rank_tp_fn_fp'] = {r: [rank_tp[r], rank_fn[r], rank_fp[r]] for r in range(1, 11)}
            G['fp_type'] = dict(fp_type)
            G['strata_loss_points'] = {k: {'refs': v[0], 'f05': round(1 - v[1] / v[0], 5), 'loss_points': round(v[1] / n * 100, 4)}
                                       for k, v in sorted(strata.items())}
            edges = [0, .02, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, 1.0001]
            b = np.digitize(ph, edges) - 1
            G['calibration'] = [{'bin': f'{edges[i]:.2f}-{min(edges[i + 1], 1):.2f}', 'n': int((b == i).sum()),
                                 'mean_p': round(float(ph[b == i].mean()), 4), 'pos_rate': round(float(yh[b == i].mean()), 4),
                                 'pos_rate_decoy': round(float(yh[(b == i) & dh].mean()), 4) if ((b == i) & dh).any() else None}
                                for i in range(len(edges) - 1) if (b == i).any()]
            E[gname] = G
            log(gname, json.dumps({k: G[k] for k in ('rule', 'dev', 'holdout', 'ceilings', 'loss_points')}))
            (FUS / 'gen2_eda.json').write_text(json.dumps(E, indent=1, default=str))
        with open(FUS / 'gen2_examples.tsv', 'w', encoding='utf-8') as f:
            f.write('gen\tkind\tref\tref_name\tref_addr\trank\ttarget\ttgt_name\ttgt_addr\tp\tgold\tchosen\n')
            for e in examples:
                f.write('\t'.join(str(x).replace('\t', ' ') for x in e) + '\n')
        log('done')


    if __name__ == '__main__':
        main()



def cmd_gen4_eda():
    """Gen-4 error EDA of T4's stage-3b (holdout 0.99037), holdout side (US/India, labels).

Pools of remaining loss, each with a label-fix ceiling (decide that pool by label, everything else as T4) and
label-free diagnostics: loss split (S5 / T4 / T5 = T4 + ownership if trained), row errors by segment, blank-address
ties and tie-breakers, same-address aliases, decoy anatomy, feature separation + SHAP on error rows, calibration by
segment with a dev-chosen per-segment recalibration. Writes OUT/gen4_eda.json, OUT/cases_<pool>.tsv.
Env OUT (default artifacts/gen4_eda), T5_DIR (optional boundary_models_t5 with stage3b_prob.npy).
"""
    import csv
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
    import xgboost as xgb

    A = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    sys.path.insert(0, str(A / 'business_entity_resolution/src'))
    from distractor_check import decide, f05, partition_of  # noqa: E402

    V4 = Path('/scratch/scai/mtech/aib262015/vishal_v4_artifacts')
    B4 = A / 'artifacts/fusion/boundary_models_t4'
    RAW = ME / 'data/dataset/train'
    OUT = Path(os.environ.get('OUT', str(ME / 'artifacts/gen4_eda')))
    OWN = ME / 'artifacts/ownership_v1'
    LEGAL = set('inc llc ltd pvt private limited co corp corporation company lp llp pllc plc sa sarl sas eurl sasu sci '
                'the and of pc pa'.split())


    def log(*a):
        print(time.strftime('%H:%M:%S'), *a, flush=True)


    def fold(s):
        s = unicodedata.normalize('NFKD', s or '')
        return re.sub(r'[^\w]+', ' ', ''.join(c for c in s if not unicodedata.combining(c)).lower()).split()


    def best_k(order_t, g):
        if not g:
            return 1.0, 0
        best, bk, tp = 0.0, 0, 0
        for k, t in enumerate(order_t, 1):
            tp += t in g
            v = 1.25 * tp / (0.25 * len(g) + k)
            if v > best + 1e-12:
                best, bk = v, k
        return best, bk


    def main():
        OUT.mkdir(parents=True, exist_ok=True)
        gold = {k: set(v) for k, v in json.loads((V4 / 'pairs/gold.json').read_text()).items()}
        countries = json.loads((V4 / 'pairs/countries.json').read_text())
        cp = partition_of(gold)
        mask = np.load(A / 'artifacts/prune_v1_copy/mask_top10.npy')
        refs, tgts = [], []
        with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t')
                    refs.append(r); tgts.append(t)
        refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
        part = np.array([cp[r] for r in refs], dtype=object)
        S = np.load(A / 'artifacts/fusion/stage1.npz')
        y = np.load(V4 / 'pairs/y.npy')[S['rows']].astype(bool)
        s1f = np.where(S['stage1'] < 0.02, 0.0, S['stage1']).astype(np.float64)
        C = np.load(B4 / 'exp2_cache.npz')
        meta = json.load(open(B4 / 'stage3b_meta.json'))
        names55 = meta['features']
        ce2 = [np.load(ME / f'artifacts/{m}/ce_logit_sample.npy').astype(np.float32) for m in ('mdeberta_full21', 'mdeberta_decoyw2')]
        X55 = np.column_stack([C['F'], C['Frev'], C['Fnum']] + ce2).astype(np.float32)
        assert X55.shape[1] == len(names55)
        col = {n: X55[:, i] for i, n in enumerate(names55)}
        blank = C['blank']
        p4 = np.load(B4 / 'stage3b_prob.npy')
        own = np.column_stack([np.load(OWN / f) for f in ('own_sample.npy', 'own_addr_sample.npy', 'own_raw_sample.npy')])
        own_names = json.load(open(OWN / 'own_sample.json'))
        names26 = own_names['features'] + own_names['addr_features'] + own_names['raw_features']
        for i, n in enumerate(names26):
            col[n] = own[:, i]
        owned = set()
        with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) > 1 and q[1]:
                    owned.update(t.strip() for t in q[1].split(','))
        decoy = ~y & np.array([t not in owned for t in tgts])
        is_s3 = np.array([t.startswith('S3') for t in tgts])
        # S5 L2 on the sample (my late_multi reproduction of S5: v5 prob + mdeberta_full logit, threshold 0.68)
        m5 = ME / 'artifacts/final_night/models/full_w3'
        b5 = xgb.Booster(); b5.load_model(str(m5 / 'model.json'))
        cfg5 = json.load(open(m5 / 'config.json'))
        ce_full = np.load(ME / 'artifacts/mdeberta_full/ce_logit_sample.npy').astype(np.float32)
        p5 = np.asarray(b5.inplace_predict(np.column_stack([s1f, ce_full]).astype(np.float32),
                                           iteration_range=(0, cfg5['best_iteration'] + 1)), dtype=np.float64)
        models = {'S5': (p5, tuple(cfg5['rule'])), 'T4': (p4, tuple(meta['rule']))}
        t5 = Path(os.environ.get('T5_DIR', str(ME / 'artifacts/fusion/boundary_models_t5')))
        if (t5 / 'stage3b_prob.npy').exists():
            m5meta = json.load(open(t5 / 'stage3b_meta.json'))
            models['T5 (T4+ownership)'] = (np.load(t5 / 'stage3b_prob.npy'), tuple(m5meta['rule']))
        log('loaded', {k: v[1] for k, v in models.items()})

        H = np.flatnonzero(part == 'holdout')
        D = np.flatnonzero(part == 'dev')
        prefs = {p: sorted(s for s in gold if cp[s] == p) for p in ('dev', 'holdout')}
        rix = {}
        for p, idx in (('dev', D), ('holdout', H)):
            pos = {s: i for i, s in enumerate(prefs[p])}
            rix[p] = np.array([pos[r] for r in refs[idx]])

        def macro(p, idx, chosen):
            pred = defaultdict(set)
            for i, tt in zip(rix[p][chosen], tgts[idx][chosen]):
                pred[i].add(tt)
            return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

        need = set(refs[H]) | set(tgts[H])
        text = {}
        for k in (1, 2, 3):
            with open(RAW / f'train_source{k}.tsv', encoding='utf-8', newline='') as fh:
                for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                    if row['entity_id'] in need:
                        text[row['entity_id']] = (row['business_name'] or '', row['business_address'] or '')
        log('text', len(text))
        R = {'model_rules': {k: list(v[1]) for k, v in models.items()}}

        # ---------- 1. loss split per model ----------
        by_ref = defaultdict(list)
        for j in H:
            by_ref[refs[j]].append(j)
        chosen_h = {}
        for name, (prob, rule) in models.items():
            ch = decide(rix['holdout'], prob[H], rule[0], rule[1], len(prefs['holdout']))
            cfull = np.zeros(len(y), dtype=bool); cfull[H] = ch
            chosen_h[name] = cfull
            loss, cnt = Counter(), Counter()
            for s in prefs['holdout']:
                g = gold[s]; js = by_ref.get(s, [])
                order = sorted(js, key=lambda j: -prob[j]); ot = [tgts[j] for j in order]
                pred = {tgts[j] for j in js if cfull[j]}
                sc, perfect = f05(g, pred), f05(g, g & set(ot))
                ok, ks = best_k(ot, g)
                loss['blocking'] += 1 - perfect; loss['ranking'] += perfect - ok
                b, kp = ok - sc, len(pred)
                kind = ('singleton_fp' if not g and kp else 'missed_all' if g and not kp else
                        'stopped_late' if kp > ks else 'stopped_early' if kp < ks else 'none')
                if b > 1e-12:
                    loss[kind] += b; cnt[kind] += 1
            n = len(prefs['holdout'])
            R[f'loss_split_{name}'] = {'holdout': round(macro('holdout', H, cfull[H]), 6),
                                       'points': {k: round(v / n * 100, 4) for k, v in loss.items()},
                                       'refs': dict(cnt)}
            log(name, R[f'loss_split_{name}'])

        ch4 = chosen_h['T4']
        base = macro('holdout', H, ch4[H])
        fn, fp = y & ~ch4, ~y & ch4
        rank = np.zeros(len(y), dtype=np.int16)
        for s, js in by_ref.items():
            for r, j in enumerate(sorted(js, key=lambda j: -p4[j]), 1):
                rank[j] = r

        # ---------- 2. pools with label-fix ceilings ----------
        alias = (col['no_name_overlap'] == 1) & ~blank & (np.nan_to_num(col['own_addr_set'], nan=0) >= 85)
        pools = {
            'blank_missed': blank & fn, 'blank_false': blank & fp,
            'nonblank_missed': ~blank & fn, 'nonblank_false': ~blank & fp,
            'alias_missed': alias & fn, 'alias_false': alias & fp,
            'decoy_false': decoy & fp, 'owned_by_other_false': ~y & ~decoy & fp,
            'blank_missed_tied': blank & fn & (col['rn_close'] >= 1),
            'blank_missed_untied': blank & fn & (col['rn_close'] < 1),
        }
        R['pools'] = {}
        for nm, m in pools.items():
            mh = m[H]
            fixed = ch4[H].copy(); fixed[mh] = y[H][mh]
            R['pools'][nm] = {'rows': int(mh.sum()), 'label_fix_gain_points': round((macro('holdout', H, fixed) - base) * 100, 4),
                              'mean_p': round(float(p4[H][mh].mean()), 3) if mh.any() else None}
        log('pools', R['pools'])

        # ---------- 3. row errors by segment ----------
        seg = defaultdict(Counter)
        gcount = np.array([len(gold[r]) for r in refs])
        for j in H:
            key = (('blank' if blank[j] else 'addr'), ('S3' if is_s3[j] else 'S2'), countries.get(refs[j], '?'),
                   'g0' if gcount[j] == 0 else 'g1' if gcount[j] == 1 else 'g2-3' if gcount[j] <= 3 else 'g4+',
                   f'rank{min(int(rank[j]), 6)}')
            c = seg['|'.join(key)]
            c['rows'] += 1; c['pos'] += int(y[j]); c['fn'] += int(fn[j]); c['fp'] += int(fp[j])
        for dim, pos_ in (('blank', 0), ('source', 1), ('country', 2), ('gold_count', 3), ('rank', 4)):
            agg = defaultdict(Counter)
            for k, c in seg.items():
                agg[k.split('|')[pos_]].update(c)
            R[f'errors_by_{dim}'] = {k: dict(v) for k, v in sorted(agg.items())}
        top = sorted(seg.items(), key=lambda kv: -(kv[1]['fn'] + kv[1]['fp']))[:15]
        R['errors_top_cells'] = {k: dict(v) for k, v in top}

        # ---------- 4. blank ties and tie-breakers (uncertain band) ----------
        band = (p4 > 0.1) & (p4 < 0.9)
        Hm = np.zeros(len(y), dtype=bool); Hm[H] = True
        bb = Hm & blank & band

        def group_rates(mask_, groups):
            out = {}
            for gname, gm in groups.items():
                m = mask_ & gm
                if m.sum() >= 20:
                    rate, pr = float(y[m].mean()), float(p4[m].mean())
                    out[gname] = {'rows': int(m.sum()), 'match_rate': round(rate, 3), 'mean_p': round(pr, 3),
                                  'gap': round(rate - pr, 3), 'FIX?': abs(rate - pr) >= 0.1}
            return out
        raw_eq_ref = np.nan_to_num(col['own_raw_ratio'], nan=0) >= 99.5
        fold_eq_ref = np.nan_to_num(col['own_sim_ratio'], nan=0) >= 99.5
        fold_eq_rival = np.nan_to_num(col['own_best_other_ratio'], nan=0) >= 99.5
        nacc = col['n_acc']
        R['blank_ties'] = {
            'missed_blank_rows': int((Hm & blank & fn).sum()),
            'missed_blank_rival_within_0.02': int((Hm & blank & fn & (col['rn_close'] >= 1)).sum()),
            'rn_close_dist_missed': {str(k): int(v) for k, v in sorted(Counter(np.minimum(col['rn_close'][Hm & blank & fn], 5).astype(int)).items())},
            'tie_breakers_band_0.1_0.9': group_rates(bb, {
                'exact_raw_name_eq_ref': raw_eq_ref, 'not_exact_raw_eq_ref': ~raw_eq_ref,
                'folded_eq_ref_and_rival': fold_eq_ref & fold_eq_rival, 'folded_eq_ref_not_rival': fold_eq_ref & ~fold_eq_rival,
                'folded_eq_rival_not_ref': ~fold_eq_ref & fold_eq_rival,
                'ref_is_closest_raw_unique': (col['own_raw_is_best'] == 1) & (col['own_raw_n_tied'] <= 1),
                'raw_tied_2plus': col['own_raw_n_tied'] >= 2,
                'ref_is_closest_folded_unique': (col['own_is_best'] == 1) & (col['own_n_tied_best'] <= 1),
                'n_acc_0': nacc == 0, 'n_acc_1': nacc == 1, 'n_acc_2': nacc == 2, 'n_acc_3plus': nacc >= 3,
                'src_S2': ~is_s3, 'src_S3': is_s3,
                'same_src_accepted_0': col['n_acc_same_src'] == 0, 'same_src_accepted_1plus': col['n_acc_same_src'] >= 1,
                'rival_close_0': col['rn_close'] == 0, 'rival_close_1': col['rn_close'] == 1, 'rival_close_2plus': col['rn_close'] >= 2,
                'mult_s1_tgt_1': col['mult_s1_tgt'] <= np.log1p(1) + 1e-6, 'mult_s1_tgt_2plus': col['mult_s1_tgt'] > np.log1p(1) + 1e-6,
            })}

        # ---------- 5. aliases at the same address ----------
        Ha = Hm & alias
        R['aliases'] = {'rows': int(Ha.sum()), 'pos': int(y[Ha].sum()), 'missed': int((Ha & fn).sum()), 'false': int((Ha & fp).sum()),
                        'by_signal_band_0.05_0.95': group_rates(Ha & (p4 > 0.05) & (p4 < 0.95), {
                            'unit_eq': col['unit_eq'] == 1, 'unit_not_eq': col['unit_eq'] != 1,
                            'addr_owners_1': col['own_n_addr85'] <= 1, 'addr_owners_2plus': col['own_n_addr85'] >= 2,
                            'addr_is_best_unique': (col['own_addr_is_best'] == 1) & (col['own_n_addr85'] <= 1),
                            'addr_margin_pos': np.nan_to_num(col['own_addr_margin'], nan=-1) > 0,
                            'addr_margin_le0': np.nan_to_num(col['own_addr_margin'], nan=-1) <= 0,
                            'a_s1_tgt_1': col['a_s1_tgt'] <= np.log1p(1) + 1e-6, 'a_s1_tgt_2plus': col['a_s1_tgt'] > np.log1p(1) + 1e-6})}

        # ---------- 6. decoy anatomy (false matches on unowned records) ----------
        kinds = Counter(); kinds_all_decoy = Counter()

        def edit_kind(j):
            rn, tn = set(fold(text.get(refs[j], ('', ''))[0])), set(fold(text.get(tgts[j], ('', ''))[0]))
            diff = rn ^ tn
            if col['shift_up_1_13'][j] == 1:
                return 'number_shift_up_1_13'
            if col['num_shift'][j] not in (0,) and not np.isnan(col['num_shift'][j]) and col['num_eq_ref'][j] == 0 and col['tgt_has_num'][j] == 1:
                return 'number_shift_other'
            if col['decoy_word'][j] == 1:
                return 'affiliate_word'
            if diff and diff <= LEGAL:
                return 'legal_form_swap'
            if col['core_swap'][j] == 1 or (col['num_eq_ref'][j] == 1 and diff):
                return 'same_number_word_swap'
            return 'other'
        for j in np.flatnonzero(Hm & decoy & fp):
            kinds[edit_kind(j)] += 1
        for j in np.flatnonzero(Hm & decoy & (p4 > 0.05)):
            kinds_all_decoy[edit_kind(j)] += 1
        s13 = Hm & (col['shift_up_1_13'] == 1)
        R['decoy_anatomy'] = {'false_decoys': int((Hm & decoy & fp).sum()), 'false_owned_by_other': int((Hm & ~y & ~decoy & fp).sum()),
                              'false_decoy_edit': dict(kinds.most_common()), 'decoys_p>0.05_edit': dict(kinds_all_decoy.most_common()),
                              'shift_up_1_13': {'rows': int(s13.sum()), 'pos_rate': round(float(y[s13].mean()), 4) if s13.any() else None,
                                                'false_matches': int((s13 & fp).sum()), 'missed': int((s13 & fn).sum()),
                                                'mean_p': round(float(p4[s13].mean()), 4) if s13.any() else None}}

        # ---------- 7. feature separation + SHAP ----------
        allnames = names55 + names26

        def sep(a, b):
            out = {}
            for n in allnames:
                va, vb = col[n][a], col[n][b]
                va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
                if len(va) < 20 or len(vb) < 20:
                    continue
                sd = np.sqrt((va.var() + vb.var()) / 2) + 1e-9
                out[n] = round(float((va.mean() - vb.mean()) / sd), 3)
            return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))
        missed_band = Hm & band & fn
        rejected_band = Hm & band & ~y & ~ch4
        false_m = Hm & fp
        true_m = Hm & y & ch4 & band
        R['separation_missed_vs_correctly_rejected_band'] = {'n': [int(missed_band.sum()), int(rejected_band.sum())],
                                                             'effect_size': sep(missed_band, rejected_band)}
        R['separation_false_vs_true_match_band'] = {'n': [int(false_m.sum()), int(true_m.sum())], 'effect_size': sep(false_m, true_m)}
        b4 = xgb.Booster(); b4.load_model(str(B4 / 'stage3b.json'))
        err = np.flatnonzero(Hm & (fn | fp))
        allb = np.flatnonzero(Hm & band)
        for nm, rows in (('errors', err), ('band', allb)):
            contrib = b4.predict(xgb.DMatrix(X55[rows]), pred_contribs=True, iteration_range=(0, meta['best_iteration'] + 1))
            R[f'shap_mean_abs_{nm}'] = dict(sorted(((n, round(float(np.abs(contrib[:, i]).mean()), 4)) for i, n in enumerate(names55)),
                                                  key=lambda kv: -kv[1]))
        R['features_no_signal_on_errors'] = [n for n in names55 if R['shap_mean_abs_errors'].get(n, 0) < 0.01
                                              and abs(R['separation_missed_vs_correctly_rejected_band']['effect_size'].get(n, 0)) < 0.05
                                              and abs(R['separation_false_vs_true_match_band']['effect_size'].get(n, 0)) < 0.05]

        # ---------- 8. calibration by segment + per-segment recalibration (dev-chosen) ----------
        def calib(m):
            out = {}
            for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
                mm = m & (p4 >= lo) & (p4 < hi)
                if mm.sum() >= 30:
                    out[f'{lo}-{hi}'] = [int(mm.sum()), round(float(p4[mm].mean()), 3), round(float(y[mm].mean()), 3)]
            return out
        segs = {'blank': blank, 'nonblank': ~blank, 'alias': alias, 'US': np.array([countries.get(r) == 'US' for r in refs]),
                'India': np.array([countries.get(r) == 'India' for r in refs]), 'S2': ~is_s3, 'S3': is_s3}
        R['calibration_holdout [n, mean_p, rate]'] = {k: calib(Hm & v) for k, v in segs.items()}
        lg = np.log(np.clip(p4, 1e-6, 1 - 1e-6) / np.clip(1 - p4, 1e-6, 1))
        grid = [(ru, round(float(t), 2)) for ru in ('expected', 'threshold') for t in np.arange(0.4, 0.961, 0.02)]
        Dm = np.zeros(len(y), dtype=bool); Dm[D] = True
        for sname in ('blank', 'S3', 'US'):
            s = segs[sname]
            pr = p4.copy()
            for mseg in (s, ~s):
                m = Dm & mseg & band
                if m.sum() < 100:
                    continue
                # 1-D Platt on the logit, fitted on dev rows of the segment
                a, b = 1.0, 0.0
                xv, yv = lg[m], y[m].astype(np.float64)
                for _ in range(200):
                    z = 1 / (1 + np.exp(-(a * xv + b)))
                    ga, gb = ((z - yv) * xv).mean(), (z - yv).mean()
                    a -= 0.5 * ga; b -= 0.5 * gb
                sel = mseg & band
                pr[sel] = 1 / (1 + np.exp(-(a * lg[sel] + b)))
            dev = {g: macro('dev', D, decide(rix['dev'], pr[D], *g, len(prefs['dev']))) for g in grid}
            g = max(dev, key=lambda k: (dev[k], k[1]))
            R[f'recalibrate_by_{sname}'] = {'rule': list(g), 'dev': round(dev[g], 6),
                                             'holdout': round(macro('holdout', H, decide(rix['holdout'], pr[H], *g, len(prefs['holdout']))), 6),
                                             'T4_dev_holdout': [None, round(base, 6)]}
            log('recalibrate', sname, R[f'recalibrate_by_{sname}'])

        # ---------- 9. case files (~40 per pool) ----------
        rng = np.random.default_rng(0)
        keep = ['rn_gap', 'rn_close', 'n_acc', 'mult_s1_tgt', 'no_name_overlap', 'num_eq_ref', 'shift_up_1_13', 'decoy_word',
                'own_margin_ratio', 'own_n_tied_best', 'own_raw_margin', 'own_addr_margin', 'own_n_addr85']
        for nm, m in pools.items():
            rows = np.flatnonzero(m & Hm)
            rows = rows[rng.permutation(len(rows))[:40]]
            with open(OUT / f'cases_{nm}.tsv', 'w', encoding='utf-8', newline='') as f:
                w = csv.writer(f, delimiter='\t')
                w.writerow(['ref', 'ref_name', 'ref_addr', 'target', 'tgt_name', 'tgt_addr', 'rank', 'p_T4', 'p_S5', 'gold',
                            'decoy', 'gold_count'] + keep)
                for j in rows:
                    rn, ra = text.get(refs[j], ('', '')); tn, ta = text.get(tgts[j], ('', ''))
                    w.writerow([refs[j], rn, ra, tgts[j], tn, ta, int(rank[j]), round(float(p4[j]), 3), round(float(p5[j]), 3),
                                int(y[j]), int(decoy[j]), int(gcount[j])] + [round(float(col[k][j]), 3) for k in keep])
        (OUT / 'gen4_eda.json').write_text(json.dumps(R, indent=1, default=str))
        log('done')


    if __name__ == '__main__':
        main()



def cmd_swap_france():
    """Ablation build: U5 with its France rows replaced by another France version, keeping U5's France rescue links.

U5 France = U2 France (FRtrained_w4 -> exclusivity prune -> census v2) + rescue links (ranks 11-20). U6 = U5 US/India
rows + FR_TSV's France rows (default: U1's France, census v2 without the prune) + the same rescue links, each dropped if
its record is already held by another S1 in the new rows (a record belongs to one S1). Candidates = U5's.
Env U5_DIR, U2_TSV, FR_TSV, OUT_DIR.
"""
    import os
    import shutil
    from collections import Counter
    from pathlib import Path

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    from france_census import load

    H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
    u5d = Path(os.environ['U5_DIR'])
    u5, u2, fr = load(u5d / 'matching_results.tsv'), load(os.environ['U2_TSV']), load(os.environ['FR_TSV'])
    cand = load(u5d / 'candidate_pairs.tsv')
    rows, cty = [], {}
    with open(H / 'data/dataset/test/test_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, ic = head.index('entity_id'), head.index('country')
        for line in fh:
            q = line.rstrip('\n').split('\t'); cty[q[ie]] = q[ic]
    order = []
    with open(u5d / 'matching_results.tsv', encoding='utf-8') as fh:
        next(fh)
        order = [line.split('\t', 1)[0].rstrip('\n') for line in fh]
    new = {r: (list(fr.get(r, [])) if cty[r] == 'France' else list(u5[r])) for r in order}
    owner = Counter(t for ts in new.values() for t in ts)
    R = Counter()
    for r in order:
        if cty[r] != 'France':
            continue
        for t in set(u5[r]) - set(u2[r]):
            if owner[t]:
                R['rescue_dropped_record_held'] += 1
                continue
            new[r].append(t); owner[t] += 1; R['rescue_kept'] += 1
    R['records_in_2_rows'] = sum(v > 1 for v in owner.values())
    R['matches_outside_candidates'] = sum(len(set(new[r]) - set(cand.get(r, []))) for r in order)
    for c in ('France', 'India', 'US'):
        a = sum(len(set(new[r]) - set(u5[r])) for r in order if cty[r] == c)
        d = sum(len(set(u5[r]) - set(new[r])) for r in order if cty[r] == c)
        R[f'{c}_vs_U5_added'] = a; R[f'{c}_vs_U5_removed'] = d
        R[f'{c}_empty'] = sum(not new[r] for r in order if cty[r] == c)
    out = Path(os.environ['OUT_DIR']); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for r in order:
            f.write(r + '\t' + ','.join(new[r]) + '\n')
    shutil.copy(u5d / 'candidate_pairs.tsv', out / 'candidate_pairs.tsv')
    print(dict(R), flush=True)



def cmd_train_fr_aug():
    """French-noise augmentation fine-tune of a pair cross-encoder (France has no training data).

The France deep dive (france_eda.py, affix_check.py) showed the cross-encoder handles French abbreviations,
départements and affiliate decoys, but hesitates on French legal forms added/dropped at the same address (logits -1..-4)
and is lenient on different house numbers. Train pairs are rewritten French-style on the fly, keeping their label:
  names    - English legal forms removed, a French form (SARL/SAS/EURL/SASU/SA/SCI/SNC/EI) placed as prefix/suffix,
             independently on each side (dropped, swapped, dotted, bracketed) so the form carries no label signal
  address  - '<num> <name> Street' -> '<num> Rue <name>' with French type words and abbreviations (R., Av., Bd, Imp),
             'N°'/'No'/'#' number prefixes, casing kept
  decoys   - some augmented positives become negatives: target name + Groupe/Développement/International/Holding/
             Distribution/Participations/Et Fils (0% positive in train as English affixes), house number shifted
Mixed with untouched pairs (AUG_P). Starts from BASE (chunk2 model), low LR, time-budgeted DDP like train_ddp.py;
then scores the 100k-sample rows unaugmented -> ce_logit_sample.npy (no US/India regression check).
Launch: python -m torch.distributed.run --nproc_per_node=2 train_fr_aug.py
"""
    import gzip
    import os
    import random
    import re
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    HOME = Path.home()
    SCR = Path(os.environ.get('ER_SCRATCH', str(HOME / 'scratch')))  # data root, read-only is enough
    PAIRS = SCR / 'amazon_ml_2026/artifacts/fulltrain_pairs'
    RAW = SCR / 'amazon_ml_2026/data/dataset/train'
    V4 = SCR / 'vishal_v4_artifacts'
    MASK = SCR / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy'
    BASE = Path(os.environ.get('BASE', '/home/scai/mtech/aib262140/scratch/amazon_ml_2026/ftchunk2/model'))
    OUT = Path(os.environ.get('OUT_DIR', str(SCR / 'amazon_ml_2026/artifacts/cross_encoder_v1/mdeb_fr_aug')))
    MAX_MIN = float(os.environ.get('MAX_MIN', 60))
    TRAIN_PAIRS = int(os.environ.get('TRAIN_PAIRS', 4_000_000))
    LR = float(os.environ.get('LR', 1e-5))
    BATCH = int(os.environ.get('BATCH', 128))
    AUG_P = float(os.environ.get('AUG_P', 0.5))
    DECOY_P = float(os.environ.get('DECOY_P', 0.2))
    PSEUDO = os.environ.get('PSEUDO', '')  # France self-training pairs (france_pseudo.py), test split, real French text
    PSEUDO_FRAC = float(os.environ.get('PSEUDO_FRAC', 0.4))
    MAX_LEN = 128

    FR_FORMS = ['SARL', 'SAS', 'EURL', 'SASU', 'SA', 'SCI', 'SNC', 'EI']
    FORM_VARIANTS = {'SAS': ['SAS', 'S.A.S.', 'S.A.S', 'Sas'], 'SARL': ['SARL', 'S.A.R.L.', 'Sarl', 'sarl'],
                     'EURL': ['EURL', 'E.U.R.L.', 'Eurl'], 'SASU': ['SASU', 'S.A.S.U.', 'Sasu'], 'SA': ['SA', 'S.A.'],
                     'SCI': ['SCI', 'S.C.I.'], 'SNC': ['SNC', 'S.N.C.'], 'EI': ['EI', 'E.I.']}
    EN_LEGAL = re.compile(r'\b(l\.?l\.?c\.?|inc\.?|incorporated|corp\.?|corporation|ltd\.?|limited|pvt\.?|private|'
                          r'llp|pllc|p\.?c\.?|lp)(?=\W|$)', re.I)
    EN_TYPES = {'street': 'Rue', 'st': 'Rue', 'avenue': 'Avenue', 'ave': 'Avenue', 'road': 'Route', 'rd': 'Route',
                'drive': 'Allée', 'dr': 'Allée', 'lane': 'Impasse', 'ln': 'Impasse', 'boulevard': 'Boulevard',
                'blvd': 'Boulevard', 'court': 'Cour', 'ct': 'Cour', 'place': 'Place', 'pl': 'Place', 'way': 'Chemin'}
    FR_ABBR = {'Rue': ['Rue', 'R', 'R.', 'Rue'], 'Avenue': ['Avenue', 'Av.', 'Av', 'Avenue'], 'Boulevard': ['Boulevard', 'Bd'],
               'Impasse': ['Impasse', 'Imp', 'Imp.'], 'Allée': ['Allée', 'Allee', 'All.'], 'Route': ['Route', 'Rte'],
               'Place': ['Place', 'Pl', 'Pl.'], 'Cour': ['Cour'], 'Chemin': ['Chemin', 'Ch.']}
    STREET = re.compile(r'^\s*(\d+\w?)\s+(.+?)\s+(' + '|'.join(EN_TYPES) + r')\.?\s*$', re.I)
    AFFIL = ['Groupe', 'Développement', 'International', 'Holding', 'Distribution', 'Participations', 'Et Fils', '& Fils']
    # French category words, from the France test S1 names (unsupervised test statistics; decoy words removed). In train, a
    # one-word swap at the same house number is a true match 80% of the time (swap_check), so swaps are applied to matches
    # and non-matches alike: the category word carries no label signal and the address decides.
    FR_CATS = ['Club', 'École', 'Amicale', 'Comité', 'Maison', 'Centre', 'Union', 'Sportive', 'Collège', 'Amis', 'Primaire',
               'Fédération', 'Pharmacie', 'Santé', 'Parents', 'Lycée', 'Maternelle', 'Association', 'Fêtes', 'Institut',
               'Anciens', 'Compagnie', 'EHPAD', 'Loisirs', 'Culturelle', 'Clinique', 'Jeunes', 'Sport', 'Groupement',
               'Médical', 'Élémentaire', 'Hospitalier', 'Foyer', 'Culture', 'Section', 'Collectif', 'Service', 'Team',
               'Chasse', 'Danse', 'Patrimoine', 'Musique', 'Communale', 'Services', 'Théâtre', 'Cercle', 'Atelier',
               'Publique', 'Soins', 'Défense', 'Tennis', 'Culturel', 'Sportif', 'Ateliers', 'Agricole', 'Solidarité']
    SWAP_P = float(os.environ.get('SWAP_P', 0.2))


    def swap_word(name, rng):
        t = name.split()
        idx = [i for i, w in enumerate(t) if len(w) >= 3 and w.upper().strip('.()[]') not in FORM_VARIANTS]
        if len(idx) < 2:
            return name
        t[rng.choice(idx)] = rng.choice(FR_CATS)
        return ' '.join(t)


    def log(*a):
        if int(os.environ.get('RANK', 0)) == 0:
            print(time.strftime('%H:%M:%S'), *a, flush=True)


    def fr_name(name, form, rng):
        core = re.sub(r'\s+', ' ', EN_LEGAL.sub(' ', name)).strip(' ,.-')
        if not form:
            return core
        f = rng.choice(FORM_VARIANTS[form])
        x = rng.random()
        if x < 0.15:
            return f'{f} {core}'
        if x < 0.25:
            return f'{core} ({f})' if rng.random() < 0.5 else f'{core} [{f}]'
        return f'{core} {f}'


    def fr_addr(addr, rng, shift=0):
        parts = [p.strip() for p in addr.split(',')]
        out = []
        for p in parts:
            m = STREET.match(p)
            if m:
                num, street, typ = m.group(1), m.group(2), EN_TYPES[m.group(3).lower()]
                if shift:
                    d = re.match(r'\d+', num)
                    num = str(int(d.group(0)) + shift) + num[d.end():]
                pre = rng.choice(['', '', '', 'N° ', 'No ', '# '])
                p = f'{pre}{num} {rng.choice(FR_ABBR[typ])} {street}'
                if parts and addr.isupper():
                    p = p.upper()
            out.append(p)
        return ', '.join(out)


    def frenchify(ref, tgt, label, rng):
        (rn, ra), (tn, ta) = ref, tgt
        rf = rng.choice(FR_FORMS) if rng.random() < 0.75 else ''
        x = rng.random()
        tf = rf if x < 0.55 else '' if x < 0.75 else rng.choice(FR_FORMS)
        new_r = (fr_name(rn, rf, rng), fr_addr(ra, rng))
        shift = 0
        if label > 0 and rng.random() < DECOY_P:
            label = 0.0
            shift = rng.randint(1, 12) if rng.random() < 0.6 else 0
            core = fr_name(tn, '', rng)
            tn2 = f'{core} {rng.choice(AFFIL)}' + (f' {rng.choice(FORM_VARIANTS[tf])}' if tf else '')
        else:
            tn2 = fr_name(tn, tf, rng)
            if rng.random() < SWAP_P:
                tn2 = swap_word(tn2, rng)
        return new_r, (tn2, fr_addr(ta, rng, shift)), label


    def load_text(need, split='train'):
        text = {}
        for k in (1, 2, 3):
            with open(RAW.parent / split / f'{split}_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if len(q) > iaddr and q[ie] in need:
                        text[q[ie]] = (q[iname], q[iaddr])
        return text


    def enc(tok, L, R):
        return tok([f'{a} | {b}' for a, b in L], [f'{a} | {b}' for a, b in R], truncation=True, max_length=MAX_LEN,
                   padding=True, return_tensors='pt')


    def main():
        dist.init_process_group('nccl')
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local)
        dev = torch.device('cuda', local)
        OUT.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        refs, tgts, labels = [], [], []
        for f in sorted(PAIRS.glob('chunk*.npz')):
            z = np.load(f)
            refs.append(z['ref']); tgts.append(z['target']); labels.append(z['label'])
        refs, tgts = np.concatenate(refs), np.concatenate(tgts)
        y = np.concatenate(labels).astype(np.float32)
        perm = np.random.default_rng(77).permutation(len(y))[:min(TRAIN_PAIRS, len(y))]
        mine = perm[rank::world]
        mask = np.load(MASK)
        sr, st = [], []
        with gzip.open(V4 / 'pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t')
                    sr.append(r); st.append(t)
        text = load_text(set(refs[mine]) | set(tgts[mine]) | set(sr) | set(st))
        n_ps = 0
        if PSEUDO:
            zp = np.load(PSEUDO)
            pr_, pt_, pl_ = zp['ref'], zp['target'], zp['label'].astype(np.float32)
            ttext = load_text(set(pr_) | set(pt_), 'test')
            prng = np.random.default_rng(500 + rank)
            n_ps = int(BATCH * PSEUDO_FRAC)
            log('pseudo pairs', len(pl_), 'positives', int(pl_.sum()), 'per batch', n_ps)
        log('world', world, 'base', BASE, 'pairs', len(mine) * world, f'{time.time() - t0:.0f}s')

        tok = AutoTokenizer.from_pretrained(BASE)
        net = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1)
        model = torch.nn.parallel.DistributedDataParallel(net.to(dev), device_ids=[local])
        planned = len(mine) // (BATCH - n_ps)
        warm = max(1, int(0.03 * min(planned, 20000)))
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        lossf = torch.nn.BCEWithLogitsLoss()
        rng = random.Random(1000 + rank)
        model.train()
        t1 = time.time()
        step, n_aug, n_decoy = 0, 0, 0
        while step < planned:
            frac = min(1.0, (step + 1) / warm) * max(0.0, (planned - step) / planned)
            for g in opt.param_groups:
                g['lr'] = LR * frac
            L, R, Y = [], [], []
            nb = BATCH - n_ps
            for k in (prng.integers(0, len(pl_), n_ps) if n_ps else []):
                L.append(ttext.get(pr_[k], ('', ''))); R.append(ttext.get(pt_[k], ('', ''))); Y.append(float(pl_[k]))
            for i in mine[step * nb:(step + 1) * nb]:
                a, b, lab = text.get(refs[i], ('', '')), text.get(tgts[i], ('', '')), float(y[i])
                if rng.random() < AUG_P:
                    n_aug += 1
                    pos = lab > 0
                    a, b, lab = frenchify(a, b, lab, rng)
                    n_decoy += pos and lab == 0
                L.append(a); R.append(b); Y.append(lab)
            e = enc(tok, L, R).to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logit = model(**e).logits.squeeze(-1)
            loss = lossf(logit.float(), torch.tensor(Y, device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step == 200:
                sps = torch.tensor([(time.time() - t1) / 200], device=dev)
                dist.all_reduce(sps, op=dist.ReduceOp.MAX)
                left = MAX_MIN * 60 - (time.time() - t1)
                plan_t = torch.tensor([int(min(planned, 200 + left / (float(sps) * 1.03)))], device=dev)
                dist.broadcast(plan_t, src=0)
                planned = int(plan_t.item())
                log(f'{float(sps):.3f} s/step -> planned {planned} steps')
            if step % 2000 == 0:
                log(f'step {step}/{planned} loss {loss.item():.4f} aug {n_aug} decoys {n_decoy}')
        net.eval()
        dist.barrier()
        if rank == 0:
            net.save_pretrained(OUT / 'model'); tok.save_pretrained(OUT / 'model')
        log('trained', step, f'{time.time() - t1:.0f}s')
        idx = np.arange(rank, len(sr), world)
        SL = [text.get(sr[i], ('', '')) for i in idx]
        SR = [text.get(st[i], ('', '')) for i in idx]
        part = np.zeros(len(idx), dtype=np.float32)
        order = np.argsort([len(a[0]) + len(a[1]) + len(b[0]) + len(b[1]) for a, b in zip(SL, SR)], kind='stable')
        with torch.no_grad():
            for b in range(0, len(order), 512):
                o = order[b:b + 512]
                e = enc(tok, [SL[i] for i in o], [SR[i] for i in o]).to(dev)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    part[o] = net(**e).logits.squeeze(-1).float().cpu().numpy()
        np.save(OUT / f'sample_part{rank}.npy', part)
        dist.barrier()
        if rank == 0:
            ce = np.zeros(len(sr), dtype=np.float32)
            for r in range(world):
                ce[np.arange(r, len(sr), world)] = np.load(OUT / f'sample_part{r}.npy')
            np.save(OUT / 'ce_logit_sample.npy', ce)
            log('sample rows scored', len(ce))
        dist.barrier()
        dist.destroy_process_group()


    if __name__ == '__main__':
        main()



def cmd_typo_street_check():
    """Train check for census v2's typo path: is a street-word typo copy noise or a decoy signal in US/India labels?

On the labelled train sample (v5 top-10 pairs), pairs at the S1's first house number with a non-blank address, split by
name relation (identical core name / only filler or legal words added / other edit) and street relation (exact street
words / every street word matched up to a typo / no match). Prints the true-match rate and count per cell.
"""
    import gzip
    import sys
    from collections import Counter
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).parent))
    import os
    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    from france_census import FILLER, core, house, near, street_words, tokens_addr

    H = Path.home() / 'scratch'
    EN_FILLER = FILLER | {'group', 'services', 'and', 'sons', 'co', 'company', 'associates', 'enterprises', 'llc', 'inc',
                          'corp', 'ltd', 'pvt', 'private', 'limited', 'the'}


    def main():
        S = np.load(H / 'amazon_ml_2026/artifacts/fusion/stage1.npz')
        y = np.load(H / 'vishal_v4_artifacts/pairs/y.npy')[S['rows']]
        mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
        pr = []
        with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    pr.append(line.rstrip('\n').split('\t'))
        need = {x for p in pr for x in p}
        text = {}
        for k in (1, 2, 3):
            with open(H / f'amazon_ml_2026/data/dataset/train/train_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if q[ie] in need:
                        text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
        cell = Counter(); pos = Counter()
        for (r, t), lab in zip(pr, y):
            rn, ra, cty = text[r]; tn, ta, _ = text.get(t, ('', '', ''))
            if not ta.strip():
                continue
            h1, h2 = house(ra)[0], house(ta)[0]
            if not h1 or h1 != h2:
                continue
            sw, tt = street_words(ra), tokens_addr(ta)
            if not sw:
                continue
            st = 'exact' if sw <= tt else ('typo' if all(w in tt or near(w, tt) for w in sw) else 'other')
            rc, tc = set(core(rn)), set(core(tn))
            nm = 'same' if rc == tc else ('filler' if rc <= tc and (tc - rc) <= EN_FILLER else 'edit')
            k = (cty, nm, st)
            cell[k] += 1; pos[k] += int(lab)
        for k in sorted(cell):
            print(k, cell[k], 'true', round(pos[k] / cell[k], 4), flush=True)


    if __name__ == '__main__':
        main()



def cmd_u8_check():
    """Independent check of U8 against U7: per-country diff, structural checks, expected gain, hand-check sample.

Structural: every S1 row once, no record matched to two S1 rows, every match inside candidate_pairs.tsv.
Expected gain per added link as in nonus_fix.py (the reference's other links assumed right).
Usage: u8_check.py <U7 matching> <U8 matching> <U8 candidates> <out sample txt>
"""
    import csv
    import json
    import sys
    from collections import Counter
    from pathlib import Path

    import numpy as np

    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    ART = ME / 'artifacts'
    csv.field_size_limit(10 ** 9)


    def f05(tp, pred, gold):
        if not gold and not pred:
            return 1.0
        if tp == 0:
            return 0.0
        p, r = tp / pred, tp / gold
        return 1.25 * p * r / (0.25 * p + r)


    def load(p):
        rows, d = [], {}
        with open(p, encoding='utf-8') as f:
            next(f)
            for l in f:
                r, m = l.rstrip('\n').split('\t'); rows.append(r); d[r] = [x for x in m.split(',') if x]
        return rows, d


    def main():
        _, u7 = load(sys.argv[1])
        rows, u8 = load(sys.argv[2])
        _, cand = load(sys.argv[3])
        text, cty = {}, {}
        for k in (1, 2, 3):
            with open(ME / f'data/dataset/test/test_source{k}.tsv', encoding='utf-8', newline='') as fh:
                for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                    text[row['entity_id']] = (row['business_name'], row['business_address'])
                    if k == 1:
                        cty[row['entity_id']] = row['country']
        R = {'rows': len(rows), 'unique_rows': len(set(rows)), 'rows_equal_test_S1': len(set(rows)) == len(cty)}
        owner = Counter(t for v in u8.values() for t in v)
        R['records_in_two_rows'] = sum(1 for c in owner.values() if c > 1)
        R['matches_outside_candidates'] = sum(len(set(u8[r]) - set(cand[r])) for r in rows)
        st = Counter()
        added = []
        for r in rows:
            a, b = set(u7[r]), set(u8[r])
            if a != b:
                st[f'{cty[r]} refs changed'] += 1
                st[f'{cty[r]} links added'] += len(b - a); st[f'{cty[r]} links removed'] += len(a - b)
                for t in b - a:
                    added.append((r, t))
        R['diff_vs_U7'] = dict(st)
        gain = loss = 0.0
        for r, t in added:
            k = len(u8[r]) - 1
            gain += 1 - f05(k, k, k + 1); loss += 1 - f05(k, k + 1, k)
        R['sum_gain_if_true'] = round(gain, 1); R['sum_loss_if_false'] = round(loss, 1)
        R['breakeven_precision'] = round(loss / (gain + loss), 3) if added else None
        # T6 bag probabilities of the added links, for reference
        refs, tgts = [], []
        with open(ART / 'submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
            next(f)
            for line in f:
                s, c = line.rstrip('\n').split('\t')
                refs.append(s); tgts.extend(c.split(','))
        n = len(refs)
        T = np.array(tgts, dtype=object).reshape(n, 10)
        pos = {r: i for i, r in enumerate(refs)}
        pb = np.load('/scratch/scai/mtech/aib262467/t6bag/test/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)
        pF = np.load(ART / 'submission_FRtrained_w4/late_scores.npz', allow_pickle=True)['prob'].reshape(n, 10)

        def probs(r, t):
            i = pos[r]
            k = np.flatnonzero(T[i] == t)
            return (float(pb[i, k[0]]), float(pF[i, k[0]])) if len(k) else (None, None)
        bag = [probs(r, t)[0] for r, t in added]
        bag = [x for x in bag if x is not None]
        R['added_bag_p_quantiles'] = [round(float(np.quantile(bag, q)), 3) for q in (0, 0.1, 0.5, 0.9)] if bag else None
        rng = np.random.default_rng(8)
        with open(sys.argv[4], 'w', encoding='utf-8') as f:
            for j in rng.choice(len(added), size=min(60, len(added)), replace=False):
                r, t = added[j]
                pbv, pfv = probs(r, t)
                f.write(f"--- bag={pbv if pbv is None else round(pbv, 2)} pFR={pfv if pfv is None else round(pfv, 2)} "
                        f"| U7 accepts {len(u7[r])}\n  REF: {text[r][0]} | {text[r][1]}\n  ADD: {text[t][0]} | {text[t][1]}\n")
                for x in u7[r][:6]:
                    f.write(f'    has: {text[x][0]} | {text[x][1]}\n')
        print(json.dumps(R, indent=1), flush=True)


    if __name__ == '__main__':
        main()



COMMANDS = {'affix_check': cmd_affix_check, 'census_review': cmd_census_review, 'colocated_check': cmd_colocated_check, 'cross_source_check': cmd_cross_source_check, 'delta_audit': cmd_delta_audit, 'filler_test': cmd_filler_test, 'france_anatomy': cmd_france_anatomy, 'france_audit': cmd_france_audit, 'france_block_check': cmd_france_block_check, 'france_cases': cmd_france_cases, 'france_check': cmd_france_check, 'france_diff_cases': cmd_france_diff_cases, 'france_eda': cmd_france_eda, 'france_noise': cmd_france_noise, 'france_numdiff': cmd_france_numdiff, 'france_patterns': cmd_france_patterns, 'france_patterns2': cmd_france_patterns2, 'france_pseudo': cmd_france_pseudo, 'france_rates': cmd_france_rates, 'france_review': cmd_france_review, 'france_robust_report': cmd_france_robust_report, 'france_trained_report': cmd_france_trained_report, 'france_veto': cmd_france_veto, 'gen2_eda': cmd_gen2_eda, 'gen4_eda': cmd_gen4_eda, 'swap_france': cmd_swap_france, 'train_fr_aug': cmd_train_fr_aug, 'typo_street_check': cmd_typo_street_check, 'u8_check': cmd_u8_check}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('commands:', ', '.join(COMMANDS))
        sys.exit(0 if len(sys.argv) < 2 else 1)
    name = sys.argv.pop(1)
    sys.argv[0] = f'{sys.argv[0]} {name}'
    COMMANDS[name]()
