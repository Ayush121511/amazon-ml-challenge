"""France same-address census fixes: label-free post-processing of a submission (France references only).

France has no labels. A same-address rule tuned on US/India fails there because France has its own generator
patterns: at the S1's own address,
  - filler words are noise:   & Fils, & Associés, & Cie, Services, Groupe, Développement, France, dba/aka, .com
    -> such records are true matches the models tend to reject              => RESTORE when currently rejected
  - a type word added at the S1's house number marks a decoy ('different business in the same building': Club ->
    École, Amicale -> Comité) -> models accept these arbitrarily              => REJECT when currently accepted
REJECT needs the S1's first house number (incl. bis/ter/letter). RESTORE needs a strict same-address test: same first
house number, same sub-number when both have one, every distinctive street word of the S1 present, same first and
rarest core-name words (other words may be dropped), and only filler words added; for edited names the S1 must have
records at another house number (otherwise its decoy may be the same-address record itself). A restore is skipped when the record is already assigned elsewhere, or when another France S1 carries the
record's exact core name (it may be that entity's). Env SUB_DIR (folder with late_scores.npz and the rule subfolder
holding matching_results.tsv + candidate_pairs.tsv), OUT_DIR. Writes a new submission folder + census_report.json.
"""
import json
import os
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# The earlier 'distinctive rarest word' check (B) was dropped after a manual review of 28 skipped restores: every one
# was a same-address type-word drop + filler add with the S1's decoy cluster at a shifted number, i.e. a true copy.
# CENSUS_V2=1 adds three restore paths found by reading France case files (france_patterns.py sizes them), each still
# behind every guard below: a street word may carry a typo (similarity >= 80), a dropped business-type word may be the
# rarest core word (town + type names such as 'Bordeaux Anciens SAS' -> 'Bordeaux SAS & Fils'), and a record named by
# the initials of the S1's core name at the exact street address.
V2 = os.environ.get('CENSUS_V2', '') == '1'
HOME = Path.home()
RAW = Path(os.environ.get('TEST_RAW', str(HOME / 'scratch/amazon_ml_2026/data/dataset/test')))
SUB = Path(os.environ['SUB_DIR'])
OUT = Path(os.environ['OUT_DIR'])
LEGAL = {'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'sa', 'ei', 'scop', 'selarl', 's', 'a', 'r', 'l', 'u', 'e', 'i'}
STOPW = {'de', 'la', 'le', 'les', 'des', 'du', 'd', 'l', 'et', 'en', 'au', 'aux', 'a'}
FILLER = {'fils', 'associes', 'cie', 'services', 'groupe', 'developpement', 'france', 'dba', 'aka', 'formerly', 'com',
          'co', 'compagnie'}
TYPES = {'club', 'ecole', 'amicale', 'comite', 'maison', 'centre', 'union', 'sportive', 'college', 'amis', 'primaire',
         'federation', 'pharmacie', 'sante', 'parents', 'lycee', 'maternelle', 'association', 'fetes', 'institut',
         'anciens', 'ehpad', 'loisirs', 'culturelle', 'clinique', 'jeunes', 'sport', 'groupement', 'medical',
         'elementaire', 'hospitalier', 'foyer', 'culture', 'section', 'collectif', 'team', 'chasse', 'danse',
         'patrimoine', 'musique', 'communale', 'theatre', 'cercle', 'atelier', 'publique', 'soins', 'defense', 'tennis',
         'culturel', 'sportif', 'ateliers', 'agricole', 'solidarite', 'societe', 'service', 'gestion',
         'conseil', 'enseignement', 'commerce', 'sportifs', 'secours', 'cabinet', 'espace', 'fondation', 'syndicat'}
GENERIC_ST = {'rue', 'r', 'avenue', 'av', 'ave', 'boulevard', 'bd', 'bld', 'place', 'pl', 'impasse', 'imp', 'allee',
              'all', 'chemin', 'ch', 'route', 'rte', 'quai', 'q', 'square', 'sq', 'cours', 'crs', 'passage', 'pass',
              'residence', 'res', 'bis', 'ter', 'no', 'n', 'ndeg', 'b', 'c'} | STOPW


def fold(s):
    s = unicodedata.normalize('NFKD', s or '')
    return ''.join(c for c in s if not unicodedata.combining(c)).lower()


FORMS = {'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'sa', 'ei', 'scop', 'selarl'}


def forms(name):
    """Legal forms written in a name, dotted spellings joined (S.A.R.L. -> sarl)."""
    return set(re.sub(r'[^a-z0-9]+', ' ', fold(name).replace('.', '')).split()) & FORMS


def core(name):
    return [t for t in re.sub(r'[^a-z0-9]+', ' ', fold(name)).split() if t not in LEGAL and t not in STOPW]


def house(addr):
    """(first number with bis/ter/letter suffix, sub-number or '') from a French address; zero padding removed."""
    a = fold(addr)
    m = re.search(r'(?<![\d/])0*(\d+)(?:\s*(bis|ter|[a-h](?![a-z])))?(?:\s*[-/]\s*0*(\d+)|\s+0*(\d+)(?=\s+[a-z]))?', a)
    if not m:
        return '', ''
    suf = {'b': 'bis', 't': 'ter'}.get(m.group(2) or '', m.group(2) or '')
    return m.group(1) + suf, m.group(3) or m.group(4) or ''


def street_words(addr):
    """Distinctive words of the field holding the house number."""
    for f in fold(addr).split(','):
        if re.search(r'\d', f):
            return {t for t in re.sub(r'[^a-z0-9]+', ' ', f).split() if not any(ch.isdigit() for ch in t) and t not in GENERIC_ST and len(t) > 1}
    return set()


def tokens_addr(addr):
    return set(re.sub(r'[^a-z0-9]+', ' ', fold(addr)).split())


def near(w, toks):
    """A street word matched by a record token up to a typo (rapidfuzz ratio >= 80, length within 2)."""
    from rapidfuzz.fuzz import ratio
    return any(ratio(w, t) >= 80 for t in toks if abs(len(t) - len(w)) <= 2)


def load(f):
    d = {}
    with open(f, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            s, m = (line.rstrip('\n').split('\t') + [''])[:2]
            d[s] = [x for x in m.split(',') if x]
    return d


def main():
    rule_dir = next(d for d in SUB.iterdir() if (d / 'matching_results.tsv').exists())
    match = load(rule_dir / 'matching_results.tsv')
    Z = np.load(SUB / 'late_scores.npz', allow_pickle=True)
    ri, tg, p = Z['ref_idx'], Z['target'].astype(str), Z['prob'].astype(np.float64)
    refs, cty = [], {}
    text = {}
    with open(RAW / 'test_source1.tsv', encoding='utf-8') as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ie, iname, iaddr, ic = (head.index(c) for c in ('entity_id', 'business_name', 'business_address', 'country'))
        for line in fh:
            q = line.rstrip('\n').split('\t')
            refs.append(q[ie]); cty[q[ie]] = q[ic]
            if q[ic] == 'France':
                text[q[ie]] = (q[iname], q[iaddr])
    fr_idx = {i for i, r in enumerate(refs) if cty[r] == 'France'}
    m = np.array([a in fr_idx for a in ri])
    need = set(tg[m])
    for k in (2, 3):
        with open(RAW / f'test_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if q[ie] in need:
                    text[q[ie]] = (q[iname], q[iaddr] if len(q) > iaddr else '')
    s1_core = {r: core(text[r][0]) for r in fr_idx_ids(refs, fr_idx)}
    freq = Counter(t for c in s1_core.values() for t in set(c))
    key_owner = Counter(' '.join(sorted(set(c))) for c in s1_core.values())
    owner = {t: r for r, ts in match.items() for t in ts}
    elsewhere = defaultdict(bool)  # A: S1 has candidate records at another house number (its shifted decoy cluster)
    for j in np.flatnonzero(m):
        r = refs[ri[j]]
        h1, h2 = house(text[r][1])[0], house(text.get(tg[j], ('', ''))[1])[0]
        if h1 and h2 and h1 != h2:
            elsewhere[r] = True
    R = Counter()
    restore, reject, weak = [], [], []
    for j in np.flatnonzero(m):
        r, t = refs[ri[j]], tg[j]
        rn, ra = text[r]; tn, ta = text.get(t, ('', ''))
        if not ta.strip():
            continue
        (h1, s1n), (h2, s2n) = house(ra), house(ta)
        if not h1 or h1 != h2:
            continue
        rc, tc = s1_core[r], core(tn)
        added, dropped = set(tc) - set(rc), set(rc) - set(tc)
        chosen = t in match.get(r, [])
        if chosen and added & TYPES and not added & FILLER:
            reject.append((r, t, p[j], tuple(sorted(added & TYPES))))
            continue
        if chosen or (s1n and s2n and s1n != s2n):
            continue
        sw, ta_t = street_words(ra), tokens_addr(ta)
        if not sw:
            continue
        exact_st = sw <= ta_t
        if not exact_st and not (V2 and all(w in ta_t or near(w, ta_t) for w in sw)):
            continue
        initials = (V2 and exact_st and len(tc) == 1 and len(rc) >= 2 and tc[0] == ''.join(w[0] for w in rc))
        if not initials:
            if not rc or not tc or rc[0] != tc[0]:
                continue
            rare = min(rc, key=lambda x: (freq[x], x))
            if rare not in tc and not (V2 and exact_st and dropped and dropped <= TYPES):
                continue
        path = 'initials' if initials else ('typo' if not exact_st else ('type_drop' if min(rc, key=lambda x: (freq[x], x)) not in tc else ''))
        rf, tf = forms(rn), forms(tn)
        identical = not added and not dropped and not (rf and tf and rf != tf)
        if identical:
            R['restore_identical_name'] += 1  # only a legal form added/dropped: exact name + exact address is identity
        elif not elsewhere[r]:
            R['restore_skipped_no_decoys_elsewhere'] += 1
            continue
        R['strict_same_address_rejected_pairs'] += 1
        if added <= FILLER or initials:
            if owner.get(t) is not None:
                R['restore_skipped_assigned_elsewhere'] += 1
                continue
            if added and key_owner[' '.join(sorted(set(tc)))] > 0:
                R['restore_skipped_other_s1_same_name'] += 1
                continue
            if path:
                R['v2_restore_' + path] += 1
            restore.append((r, t, p[j], tuple(sorted(added))))
    best = {}
    for r, t, pv, a in restore:
        if t not in best or pv > best[t][2]:
            best[t] = (r, t, pv, a)
    restore = list(best.values())
    for r, t, _, _ in restore:
        match.setdefault(r, []).append(t)
    for r, t, _, _ in reject:
        match[r] = [x for x in match[r] if x != t]
    R = dict(R)
    R.update({'restored': len(restore), 'rejected': len(reject),
              'restored_p_quantiles': {q: round(float(np.quantile([x[2] for x in restore], q)), 3) for q in (0.1, 0.5, 0.9)} if restore else {},
              'restore_added_words': dict(Counter(w for x in restore for w in x[3]).most_common(15)),
              'reject_word_pairs': {'>'.join(k): v for k, v in Counter(x[3] for x in reject).most_common(15)},
              'reject_p_quantiles': {q: round(float(np.quantile([x[2] for x in reject], q)), 3) for q in (0.1, 0.5, 0.9)} if reject else {}})
    dest = OUT / rule_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / 'matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for r in refs:
            f.write(r + '\t' + ','.join(match.get(r, [])) + '\n')
    shutil.copy(rule_dir / 'candidate_pairs.tsv', dest / 'candidate_pairs.tsv')
    rng = np.random.default_rng(0)
    ex = {'restore': [(r, text[r], t, text.get(t)) for r, t, _, _ in (restore[i] for i in rng.choice(len(restore), min(15, len(restore)), replace=False))] if restore else [],
          'reject': [(r, text[r], t, text.get(t)) for r, t, _, _ in (reject[i] for i in rng.choice(len(reject), min(15, len(reject)), replace=False))] if reject else []}
    with open(OUT / 'skipped_weak_name.tsv', 'w', encoding='utf-8') as f:
        f.write('ref\ttarget\tp\tdropped\tadded\ts1_has_records_elsewhere\n')
        for x in weak:
            f.write('\t'.join(str(v) for v in x) + '\n')
    (OUT / 'census_report.json').write_text(json.dumps({'stats': R, 'examples': ex}, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(R, indent=1, default=str), flush=True)


def fr_idx_ids(refs, fr_idx):
    return [refs[i] for i in sorted(fr_idx)]


if __name__ == '__main__':
    main()
