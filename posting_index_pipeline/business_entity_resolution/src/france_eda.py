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
