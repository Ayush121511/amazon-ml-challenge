"""Error sweeps and training-label evidence for the rules.

Each command is one analysis script. Run `python rule_evidence.py <command> [args]`; `python rule_evidence.py` lists the commands.
"""
import sys


def cmd_alias_base():
    """Training-label rates for alias records at a business's address (true / other business / no business)."""
    import re, unicodedata, sys
    from collections import Counter, defaultdict
    D='/Users/ayushsrivastava/Documents/AmazonMLChallenge/data/student_resource/dataset/train/'
    fold=lambda s:''.join(c for c in unicodedata.normalize('NFKD',s or '') if not unicodedata.combining(c)).lower()
    toks=lambda s:re.sub(r'[^a-z0-9]+',' ',fold(s)).split()
    STOP={'inc','llc','ltd','corp','co','pvt','private','limited','the','and','of','company','corporation','lp','llp','pc','group','services','service','center','partners','holdings','enterprises','india'}
    def akey(a):
        t=toks(a); nums=[x for x in t if x.isdigit()]
        if not nums: return None
        words=sorted(set(x for x in t if not any(ch.isdigit() for ch in x) and len(x)>2))
        return (str(int(nums[0])), ' '.join(words[:6]))
    owner={}
    f=open(D+'train_ground_truth.tsv'); next(f)
    for l in f:
        q=l.rstrip('\n').split('\t')
        for t in (q[1].split(',') if len(q)>1 else []):
            if t.strip(): owner[t.strip()]=q[0]
    s1={}; idx=defaultdict(list)
    f=open(D+'train_source1.tsv',encoding='utf-8'); h=f.readline().rstrip('\n').split('\t'); ie,n,a,c=(h.index(x) for x in ('entity_id','business_name','business_address','country'))
    for l in f:
        q=l.rstrip('\n').split('\t'); k=akey(q[a]) if len(q)>a else None
        s1[q[ie]]=q[n]
        if k: idx[(q[c],)+k].append(q[ie])
    R=Counter()
    for src in (2,3):
        f=open(D+f'train_source{src}.tsv',encoding='utf-8'); h=f.readline().rstrip('\n').split('\t'); ie,n,a,c=(h.index(x) for x in ('entity_id','business_name','business_address','country'))
        for l in f:
            q=l.rstrip('\n').split('\t')
            if len(q)<=a or not q[a].strip(): continue
            k=akey(q[a]); hit=idx.get((q[c],)+k) if k else None
            if not hit or len(hit)!=1: continue
            s=hit[0]; rt=set(toks(q[n]))-STOP; st=set(toks(s1[s]))-STOP
            if not rt or rt & st: continue
            single = len(toks(q[n]))==1 and toks(q[n])[0].isalpha()
            kind='invented_single_word' if single else 'no_shared_word_multi'
            R[(kind, 'true' if owner.get(q[ie])==s else ('other_S1' if q[ie] in owner else 'nobody'))]+=1
    for kind in ('invented_single_word','no_shared_word_multi'):
        tot=sum(v for (k,_),v in R.items() if k==kind)
        print(kind, 'n', tot, {o: f'{R[(kind,o)]/max(tot,1):.3f}' for o in ('true','other_S1','nobody')})



def cmd_alias_check_france_audit():
    """Patterns found by the manual France audit, measured on test (per country) and on the labelled US/India holdout.

  alias       record name shares no word with the reference name (not initials, not a web form), same first house
              number and near-identical address  -> invented-name copy at the business's own address?
  exact_diff  same name words, both house numbers present and different                  -> shifted-number decoy group?
  blank_solo  same name words, record address blank, record in only one reference's top 10 (test) -> unclaimed copy?
Test: candidates per reference, share accepted by the submission, mean France-model (FRw4) and T6 probabilities.
Holdout: true share, T6 acceptance and precision, true share by T6 probability bin.
Also writes France alias pairs with both probabilities for the fix: /scratch/.../alias/france_alias.npz
Usage: alias_check.py <matching_results.tsv>
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
    from rapidfuzz import fuzz

    ME = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026')
    AY = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
    ART = ME / 'artifacts'
    OUT = Path('/scratch/scai/mtech/aib262467/alias')
    sys.path[:0] = [str(AY / 'business_entity_resolution/src')]
    csv.field_size_limit(10 ** 9)
    STOP = set('sarl sas sasu eurl sa sci snc selarl scop inc llc ltd limited corp corporation co company pvt private llp '
               'plc lp pllc pc de du des la le les l d et and the of en au aux'.split())
    NUM = re.compile(r'\d+')


    def fold(s):
        s = unicodedata.normalize('NFKD', (s or '').lower())
        return ''.join(c for c in s if not unicodedata.combining(c))


    def words(n):
        return [w for w in re.findall(r'[a-z0-9]+', fold(n)) if w not in STOP and not w.isdigit()]


    def house(a):
        m = NUM.search(a or '')
        return str(int(m.group(0))) if m else ''


    def kind(rn, ra, tn, ta):
        A, B = words(rn), words(tn)
        sa, sb = set(A), set(B)
        if not sa or not sb:
            return None
        blank = not (ta or '').strip()
        if sa == sb:
            if blank:
                return 'blank_same'
            h1, h2 = house(ra), house(ta)
            return 'exact_diff' if h1 and h2 and h1 != h2 else None
        if sa & sb or blank:
            return None
        comp_a, comp_b = ''.join(A), ''.join(B)
        initials = ''.join(w[0] for w in A)
        if comp_b == initials or comp_b.startswith(initials) or comp_a in comp_b or comp_b in comp_a:
            return None
        h1, h2 = house(ra), house(ta)
        if h1 and h1 == h2 and fuzz.token_set_ratio(fold(ra), fold(ta)) >= 85:
            return 'alias'
        return None


    def load_sub(p):
        with open(p, encoding='utf-8') as f:
            next(f)
            return {l.split('\t')[0]: set(x for x in l.rstrip('\n').split('\t')[1].split(',') if x) for l in f}


    def main():
        OUT.mkdir(parents=True, exist_ok=True)
        R = {}
        text = {}
        for k in (1, 2, 3):
            with open(ME / f'data/dataset/test/test_source{k}.tsv', encoding='utf-8', newline='') as fh:
                for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                    text[row['entity_id']] = (row['business_name'], row['business_address'], row['country'])
        refs, tgts = [], []
        with open(ART / 'submission_v5_top10/candidate_pairs.tsv', encoding='utf-8') as f:
            next(f)
            for line in f:
                s, c = line.rstrip('\n').split('\t')
                for t in c.split(','):
                    refs.append(s); tgts.append(t)
        ntop = Counter(tgts)
        pF = np.load(ART / 'submission_FRtrained_w4/late_scores.npz', allow_pickle=True)['prob']
        p6 = np.load(ART / 'submission_T6_own/late_scores.npz', allow_pickle=True)['prob']
        sub = load_sub(sys.argv[1])
        held = defaultdict(set)
        for r, v in sub.items():
            for t in v:
                held[t].add(r)
        nref = Counter(v[2] for k, v in text.items() if k.startswith('S1-'))
        st = defaultdict(Counter)
        fr_rows = []
        ex = []
        for j in range(len(refs)):
            r, t = refs[j], tgts[j]
            rn, ra, c = text[r]
            tn, ta, _ = text[t]
            k = kind(rn, ra, tn, ta)
            if k == 'blank_same':
                k = 'blank_solo' if ntop[t] == 1 else None
            if not k:
                continue
            a = t in sub[r]
            other = bool(held[t] - {r})
            key = (c, k)
            st[key]['n'] += 1; st[key]['acc'] += a; st[key]['other'] += other
            st[key]['pF'] += float(pF[j]); st[key]['p6'] += float(p6[j])
            st[key]['p6_ge_074'] += p6[j] >= 0.74; st[key]['pF_ge_064'] += pF[j] >= 0.64
            if c == 'France' and k == 'alias':
                fr_rows.append((j, int(a), int(other), float(pF[j]), float(p6[j]), ntop[t]))
                if not a and not other and len(ex) < 25 and j % 5 == 0:
                    ex.append(f'{rn} | {ra}  <-  {tn} | {ta}   pF={pF[j]:.3f} p6={p6[j]:.3f} n={ntop[t]}')
        R['test'] = {f'{c} {k}': {'per_ref': round(v['n'] / nref[c], 5), 'accepted_share': round(v['acc'] / v['n'], 3),
                                  'held_by_other_share': round(v['other'] / v['n'], 3), 'mean_pF': round(v['pF'] / v['n'], 3),
                                  'mean_p6': round(v['p6'] / v['n'], 3), 'p6>=0.74': round(v['p6_ge_074'] / v['n'], 3),
                                  'pF>=0.64': round(v['pF_ge_064'] / v['n'], 3), 'n': v['n']} for (c, k), v in sorted(st.items())}
        a = np.array(fr_rows, dtype=np.float64) if fr_rows else np.zeros((0, 6))
        np.save(OUT / 'france_alias.npy', a)
        R['france_alias_examples_rejected'] = ex
        del text
        # ---- holdout (labels)
        from distractor_check import decide, partition_of
        gold = {k: set(v) for k, v in json.loads((ART / 'matcher_v4/pairs/gold.json').read_text()).items()}
        cp = partition_of(gold)
        m10 = np.load(ART / 'prune_v1/mask_top10.npy').astype(bool)
        pr, pt = [], []
        with gzip.open(ART / 'matcher_v4/pairs/pairs.tsv.gz', 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if m10[i]:
                    x, y_ = line.rstrip('\n').split('\t'); pr.append(x); pt.append(y_)
        pr, pt = np.array(pr, dtype=object), np.array(pt, dtype=object)
        S = np.load(ART / 'fusion/stage1.npz')
        y = np.load(ART / 'matcher_v4/pairs/y.npy')[S['rows']] > 0
        ph = np.load(ART / 'fusion/boundary_models_t6/stage3b_prob.npy')
        rule = json.load(open(ART / 'fusion/boundary_models_t6/stage3b_meta.json'))['rule']
        prefs = sorted(s for s in gold if cp[s] == 'holdout')
        pos = {s: i for i, s in enumerate(prefs)}
        idx = np.flatnonzero(np.array([cp[r] == 'holdout' for r in pr]))
        ch = np.zeros(len(pr), bool)
        ch[idx] = decide(np.array([pos[r] for r in pr[idx]]), ph[idx], rule[0], rule[1], len(prefs))
        keep = set(pr[idx]) | set(pt[idx])
        stext = {}
        for k in (1, 2, 3):
            with open(ME / f'data/dataset/train/train_source{k}.tsv', encoding='utf-8', newline='') as fh:
                for row in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                    if row['entity_id'] in keep:
                        stext[row['entity_id']] = (row['business_name'], row['business_address'], row['country'])
        ntop_s = Counter(pt[idx])
        hs = defaultdict(Counter)
        bins = defaultdict(lambda: np.zeros((5, 2)))
        edges = [0, 0.3, 0.5, 0.64, 0.8, 1.01]
        for j in idx:
            rn, ra, c = stext[pr[j]]
            tn, ta, _ = stext[pt[j]]
            k = kind(rn, ra, tn, ta)
            if k == 'blank_same':
                k = 'blank_solo' if ntop_s[pt[j]] == 1 else None
            if not k:
                continue
            key = f'{c} {k}'
            hs[key]['n'] += 1; hs[key]['true'] += int(y[j]); hs[key]['acc'] += int(ch[j]); hs[key]['acc_true'] += int(ch[j] and y[j])
            b = np.searchsorted(edges, ph[j], side='right') - 1
            bins[key][b] += (1, int(y[j]))
        n_hold = Counter(stext[s][2] for s in prefs)
        R['holdout'] = {k: {'per_ref': round(v['n'] / n_hold[k.split()[0]], 5), 'true_share': round(v['true'] / v['n'], 3),
                            'T6_accepted_share': round(v['acc'] / v['n'], 3),
                            'T6_precision': round(v['acc_true'] / max(1, v['acc']), 3),
                            'true_share_by_p6_bin[0-.3,.3-.5,.5-.64,.64-.8,.8-1]':
                                [f'{int(t)}/{int(n)}' for n, t in bins[k]], 'n': v['n']} for k, v in sorted(hs.items())}
        (OUT / 'report.json').write_text(json.dumps(R, indent=1, ensure_ascii=False))
        print(json.dumps(R, indent=1, ensure_ascii=False), flush=True)


    if __name__ == '__main__':
        main()



def cmd_alias_check_typo_street():
    """Train check (alias variant) for census v2's typo path: is a street-word typo copy noise or a decoy signal in US/India labels?

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
            if not h1 or not h2:
                continue
            sw, tt = street_words(ra), tokens_addr(ta)
            if not sw:
                continue
            st = ('same_no' if h1 == h2 else 'diff_no') + ('_street' if sw <= tt else '_other')
            rc, tc = set(core(rn)), set(core(tn))
            nm = 'same' if rc == tc else ('filler' if rc <= tc and (tc - rc) <= EN_FILLER else ('none' if not rc & tc else 'edit'))
            k = (cty, nm, st)
            cell[k] += 1; pos[k] += int(lab)
        for k in sorted(cell):
            print(k, cell[k], 'true', round(pos[k] / cell[k], 4), flush=True)


    if __name__ == '__main__':
        main()



def cmd_alias_cond():
    """Alias rule, conditioned on the model: among US/India dev + holdout top-10 pairs (error_sweep frames, T6 decisions),
records whose name is one Latin word (after dropping non-Latin script) sharing no word with the S1 name, at the S1's
exact address key (first house number + street words), with that S1 the only S1 at the key. True rate split by T6's
decision D and by country, and the F0.5 change on dev / holdout if every rejected flagged pair is added."""
    import os
    import re
    import unicodedata
    from collections import defaultdict
    import numpy as np
    import pandas as pd

    H = '/home/scai/mtech/aib262015/scratch/amazon_ml_2026/data/dataset/train/'
    fold = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s or '') if not unicodedata.combining(c)).lower()
    toks = lambda s: re.sub(r'[^a-z0-9]+', ' ', fold(s)).split()
    STOP = {'inc', 'llc', 'ltd', 'corp', 'co', 'pvt', 'private', 'limited', 'the', 'and', 'of', 'company', 'corporation', 'lp', 'llp',
            'pc', 'group', 'services', 'service', 'center', 'partners', 'holdings', 'enterprises', 'india'}


    def akey(a):
        t = toks(a); nums = [x for x in t if x.isdigit()]
        if not nums:
            return None
        return (str(int(nums[0])), ' '.join(sorted(set(x for x in t if not any(ch.isdigit() for ch in x) and len(x) > 2))[:6]))


    df = pd.read_pickle(f"{os.environ['A']}/train.json.train_pairs.pkl")
    bf = pd.read_pickle(f"{os.environ['A']}/train.json.train_blocking.pkl")
    need = set(df.t)
    s1, rec, idx = {}, {}, defaultdict(int)
    for k in (1, 2, 3):
        f = open(H + f'train_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
        ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
        for l in f:
            q = l.rstrip('\n').split('\t'); ad = q[a] if len(q) > a else ''
            if k == 1:
                s1[q[ie]] = (q[n], ad, q[c]); kk = akey(ad)
                if kk:
                    idx[(q[c],) + kk] += 1
            elif q[ie] in need:
                rec[q[ie]] = (q[n], ad)


    def flag(s, t):
        n, a, c = s1[s]; tn, ta = rec.get(t, ('', ''))
        if not ta.strip():
            return False
        k1, k2 = akey(a), akey(ta)
        if not k1 or k1 != k2 or idx[(c,) + k1] != 1:
            return False
        tk = toks(tn)
        return len(tk) == 1 and tk[0].isalpha() and not (set(tk) & (set(toks(n)) - STOP))


    df['flag'] = [flag(s, t) for s, t in zip(df.s1, df.t)]
    F = df[df.flag]
    print('flagged', len(F))
    print(F.groupby(['country', 'D']).agg(n=('y', 'size'), true_rate=('y', 'mean'), mean_p=('p', 'mean')).round(3).to_string())
    miss = bf.groupby('s1').size(); g = df.groupby('s1')
    S = pd.DataFrame({'part': g.part.first(), 'ytrue': g.y.sum(), 'npred': g.D.sum(), 'tp': (df.y * df.D).groupby(df.s1).sum()})
    S['ngold'] = S.ytrue + miss.reindex(S.index).fillna(0).astype(int)
    add = F[F.D == 0].groupby('s1').agg(a=('y', 'size'), b=('y', 'sum')); S = S.join(add).fillna({'a': 0, 'b': 0})


    def Fs(tp, npred, ngold):
        tp, npred, ngold = map(np.asarray, (tp, npred, ngold))
        pr = np.where(npred > 0, tp / np.maximum(npred, 1), 0); rc = np.where(ngold > 0, tp / np.maximum(ngold, 1), 0)
        f = np.where(tp > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
        return np.where((npred == 0) & (ngold == 0), 1.0, f)


    for p in ('dev', 'holdout'):
        s = S[S.part == p]
        print(p, 'add', int(s.a.sum()), 'true', int(s.b.sum()), 'F0.5 delta', round(float(Fs(s.tp + s.b, s.npred + s.a, s.ngold).mean() - Fs(s.tp, s.npred, s.ngold).mean()), 6))



def cmd_alias_link():
    """Alias link-up: a sibling record "X dba/fka/aka <S1 name>" names X as an alias of that S1, so another candidate record
of the same S1 whose name is X belongs to it (blank address or not). The pair models never see sibling alias text.

Train (US/India dev + holdout, T6 decisions from error_sweep.py frames): flagged pairs (candidate name == an alias
announced by an ACCEPTED sibling of the same S1), their true rate, how many T6 rejects, F0.5 change when the rejected
flagged pairs are added. Test: the same flags on U5 (candidates + accepted links), count of links it would add per
country, and records already held by another S1. Env A (sweep dir), U5_DIR.
"""
    import os
    import re
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np
    import pandas as pd

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    from france_census import core, load

    H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
    A = os.environ['A']
    SEP = re.compile(r'\s+(?:d\s*/\s*b\s*/\s*a|dba|f\s*/\s*k\s*/\s*a|fka|a\s*/\s*k\s*/\s*a|aka|formerly(?:\s+known\s+as)?|'
                     r'doing\s+business\s+as|trading\s+as|t\s*/\s*a)\.?\s+', re.I)


    def ckey(name):
        return ' '.join(sorted(set(core(name))))


    def compact(name):
        return re.sub(r'[^a-z0-9]', '', ' '.join(core(name)))


    def alias_of(rec_name, s1_name):
        """Alias side of 'X dba Y' when the other side is the S1's name, else None."""
        parts = SEP.split(rec_name, maxsplit=1)
        if len(parts) != 2:
            return None
        s = set(core(s1_name))
        if not s:
            return None
        j = [len(set(core(p)) & s) / max(1, len(set(core(p)) | s)) for p in parts]
        k = int(np.argmax(j))
        if j[k] < 0.5:
            return None
        al = parts[1 - k]
        return compact(al) or None


    def flags(s1_name, cands, accepted, text):
        """cands: list of record ids; accepted: set. Returns set of candidate ids named like an alias that an accepted
    sibling announces."""
        als = {}
        for t in cands:
            if t in accepted:
                a = alias_of(text.get(t, ('',))[0], s1_name)
                if a and len(a) >= 3:
                    als[a] = t
        if not als:
            return set()
        out = set()
        for t in cands:
            c = compact(text.get(t, ('',))[0])
            if c and c in als and als[c] != t:
                out.add(t)
        return out


    def read_text(split, need):
        text, cty = {}, {}
        for k in (1, 2, 3):
            with open(H / f'data/dataset/{split}/{split}_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if k == 1:
                        cty[q[ie]] = q[c]
                    if k == 1 or q[ie] in need:
                        text[q[ie]] = (q[n], q[a] if len(q) > a else '')
        return text, cty


    def F(tp, npred, ngold):
        tp, npred, ngold = map(np.asarray, (tp, npred, ngold))
        pr = np.where(npred > 0, tp / np.maximum(npred, 1), 0); rc = np.where(ngold > 0, tp / np.maximum(ngold, 1), 0)
        f = np.where(tp > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
        return np.where((npred == 0) & (ngold == 0), 1.0, f)


    def train():
        df = pd.read_pickle(f'{A}/train.json.train_pairs.pkl')
        bf = pd.read_pickle(f'{A}/train.json.train_blocking.pkl')
        text, _ = read_text('train', set(df.t))
        miss = bf.groupby('s1').size()
        rows = []
        for s, g in df.groupby('s1'):
            acc = set(g.t[g.D == 1])
            fl = flags(text[s][0], list(g.t), acc, text)
            for t, y, d, p, blank in zip(g.t, g.y, g.D, g.p, g.blank):
                if t in fl:
                    rows.append(dict(s1=s, t=t, y=y, D=d, p=p, blank=blank, part=g.part.iloc[0], country=g.country.iloc[0]))
        fl = pd.DataFrame(rows)
        print('TRAIN flagged pairs', len(fl))
        if not len(fl):
            return
        print(fl.groupby(['part', 'D']).agg(n=('y', 'size'), true_rate=('y', 'mean'), mean_p=('p', 'mean')).to_string())
        print('flagged & rejected by blank:', fl[fl.D == 0].groupby('blank').agg(n=('y', 'size'), true_rate=('y', 'mean')).to_string())
        g = df.groupby('s1')
        S = pd.DataFrame({'part': g.part.first(), 'ytrue': g.y.sum(), 'npred': g.D.sum(), 'tp': (df.y * df.D).groupby(df.s1).sum()})
        S['ngold'] = S.ytrue + miss.reindex(S.index).fillna(0).astype(int)
        add = fl[fl.D == 0].groupby('s1').agg(a=('y', 'size'), b=('y', 'sum'))
        S = S.join(add).fillna({'a': 0, 'b': 0})
        for p in ('dev', 'holdout'):
            s = S[S.part == p]
            d = F(s.tp + s.b, s.npred + s.a, s.ngold).mean() - F(s.tp, s.npred, s.ngold).mean()
            print(f'  {p}: add {int(s.a.sum())} links ({int(s.b.sum())} true) -> F0.5 {d:+.6f}  (public-scale x0.85 {0.85 * d:+.6f})')
        print('examples:')
        for _, r in fl[fl.D == 0].head(15).iterrows():
            print('   y', r.y, f'p={r.p:.2f}', text[r.s1][0], '|', text[r.s1][1][:40], ' => ', text.get(r.t, ('', ''))[0], '|', text.get(r.t, ('', ''))[1][:40])


    def test():
        d = Path(os.environ['U5_DIR'])
        u5, cand = load(d / 'matching_results.tsv'), load(d / 'candidate_pairs.tsv')
        need = {t for ts in cand.values() for t in ts}
        text, cty = read_text('test', need)
        owner = {t: s for s, ts in u5.items() for t in ts}
        R = Counter(); ex = defaultdict(list)
        for s, cs in cand.items():
            acc = set(u5.get(s, []))
            for t in flags(text[s][0], cs, acc, text):
                c = cty[s]
                if t in acc:
                    R[(c, 'already_accepted')] += 1
                elif t in owner:
                    R[(c, 'held_by_other_S1')] += 1
                else:
                    R[(c, 'would_add')] += 1
                    if len(ex[c]) < 12:
                        ex[c].append(f"{text[s][0]} | {text[s][1][:40]}  =>  {text.get(t, ('', ''))[0]} | {text.get(t, ('', ''))[1][:40]}")
        print('TEST', dict(R))
        for c, e in ex.items():
            print(' ', c); [print('   ', x) for x in e]


    if __name__ == '__main__':
        import sys
        sys.stdout.reconfigure(line_buffering=True)
        (train if os.environ.get('MODE') == 'train' else test)()



def cmd_alias_rate():
    """Train-label check for the U6/U7 France restores: true rate of invented-name aliases (no shared name word) and of
filler/edit copies at the S1's own house number, non-blank, by T6 p band (US/India dev + holdout, error_sweep frames)."""
    import os
    import pandas as pd
    df = pd.read_pickle(f"{os.environ['A']}/train.json.train_pairs.pkl")
    m = (df.blank == 'nonblank') & df.addr.isin(['same_no_street', 'same_no_other'])
    df['pb'] = pd.cut(df.p, [0, 0.3, 0.64, 0.85, 0.95, 1.0001])
    print(df[m].groupby(['name', 'pb']).agg(n=('y', 'size'), true_rate=('y', 'mean'), mean_p=('p', 'mean')).round(3).to_string())



def cmd_alias_test():
    """Counts and samples of alias records at a business's address in test."""
    import re, unicodedata
    from collections import Counter, defaultdict
    D='/Users/ayushsrivastava/Documents/AmazonMLChallenge/'
    T=D+'data/student_resource/dataset/test/'
    fold=lambda s:''.join(c for c in unicodedata.normalize('NFKD',s or '') if not unicodedata.combining(c)).lower()
    toks=lambda s:re.sub(r'[^a-z0-9]+',' ',fold(s)).split()
    STOP={'inc','llc','ltd','corp','co','pvt','private','limited','the','and','of','company','corporation','lp','llp','pc','group','services','service','center','partners','holdings','enterprises','india','sarl','sas','sasu','eurl','sci','snc','sa','ei','de','la','le','les','des','du','et'}
    def akey(a):
        t=toks(a); nums=[x for x in t if x.isdigit()]
        if not nums: return None
        words=sorted(set(x for x in t if not any(ch.isdigit() for ch in x) and len(x)>2))
        return (str(int(nums[0])), ' '.join(words[:6]))
    def load(f):
        d={}
        for i,l in enumerate(open(f,encoding='utf-8')):
            if i: s,m=(l.rstrip('\n').split('\t')+[''])[:2]; d[s]=[x for x in m.split(',') if x]
        return d
    u7=load('u7_matching.tsv'); cand=load(D+'submissions/final_u5/candidate_pairs.tsv')
    link={t:s for s,ts in u7.items() for t in ts}; cs={(s,t) for s,ts in cand.items() for t in ts}
    s1={}; cty={}; idx=defaultdict(list)
    f=open(T+'test_source1.tsv',encoding='utf-8'); h=f.readline().rstrip('\n').split('\t'); ie,n,a,c=(h.index(x) for x in ('entity_id','business_name','business_address','country'))
    for l in f:
        q=l.rstrip('\n').split('\t'); k=akey(q[a]) if len(q)>a else None
        s1[q[ie]]=(q[n],q[a]); cty[q[ie]]=q[c]
        if k: idx[(q[c],)+k].append(q[ie])
    R=Counter(); ex=defaultdict(list); add=[]
    for src in (2,3):
        f=open(T+f'test_source{src}.tsv',encoding='utf-8'); h=f.readline().rstrip('\n').split('\t'); ie,n,a,c=(h.index(x) for x in ('entity_id','business_name','business_address','country'))
        for l in f:
            q=l.rstrip('\n').split('\t')
            if len(q)<=a or not q[a].strip(): continue
            k=akey(q[a]); hit=idx.get((q[c],)+k) if k else None
            if not hit or len(hit)!=1: continue
            s=hit[0]; tk=toks(q[n])
            if not (len(tk)==1 and tk[0].isalpha()): continue
            if set(tk)&(set(toks(s1[s][0]))-STOP): continue
            r=q[ie]
            st='agree' if link.get(r)==s else ('other_S1' if r in link else ('unlinked_in_cand' if (s,r) in cs else 'unlinked_not_cand'))
            R[(q[c],st)]+=1
            if st!='agree' and len(ex[(q[c],st)])<5: ex[(q[c],st)].append(f'{s1[s][0]} | {s1[s][1][:40]}  <=  {q[n]} | {q[a][:40]}' + (f'   [U7 gave to {s1[link[r]][0]} | {s1[link[r]][1][:30]}]' if r in link else ''))
            if st=='unlinked_in_cand': add.append((s,r))
    for k in sorted(R): print(k, R[k])
    for k,v in ex.items():
        print(k); [print('   ',x) for x in v]
    open('alias_adds.tsv','w').write(''.join(f'{s}\t{r}\n' for s,r in add))



def cmd_alias_union_test():
    """New links from the accepted-copy-address alias rule beyond an earlier file, with samples."""
    import sys, random
    from collections import Counter
    sys.path.insert(0, '.')
    from alias_union import run
    def load(f):
        d = {}
        for i, l in enumerate(open(f, encoding='utf-8')):
            if i: s, m = (l.rstrip('\n').split('\t') + [''])[:2]; d[s] = [x for x in m.split(',') if x]
        return d
    u12 = load('U12/matching_results.tsv'); cand = load('U12/candidate_pairs.tsv')
    R, new, s1, rec = run('test', u12)
    print('TEST new alias links beyond U12:', dict(R))
    cs = Counter(); 
    for s, t in new: cs[(s1[s][2], t in set(cand[s]))] += 1
    print('  by (country, already in candidates):', dict(cs))
    random.seed(3)
    fr = [x for x in new if s1[x[0]][2] == 'France']
    for s, t in random.sample(fr, min(12, len(fr))):
        print(f'   {s1[s][0]} | {s1[s][1][:34]}  =>  {rec[t][0]} | {rec[t][1][:34]}')
    import pickle; pickle.dump(new, open('alias_union_new.pkl', 'wb'))



def cmd_blank_eda():
    """Blank-address candidates under the S5 decision: how much loss, and is any of it recoverable?

A blank-address S2/S3 record can only be tied to its S1 entity by name. It is recoverable when the name points to one
S1 entity (name ambiguity 1), or when it agrees with records already accepted for the reference (relational context,
collective ER); it is a coin flip when several S1 entities share the name. Measured on holdout with the S5 L2
probability (trained on trainA+trainB, threshold@0.68):
  ceilings    - S5 decision everywhere except blank-address rows decided by the label (and the same for non-blank)
  by bucket   - positive rate vs mean probability for blank rows, split by name ambiguity over the FULL train S1
                (exact order-free suffix-free key; a looser key), key equality with the reference, sibling agreement
                with accepted candidates, and whether the reference already has an accepted record from that source
  negatives   - blank false candidates in the ambiguous band: owned by another S1 entity vs decoy
Writes fusion/blank_eda.json.
"""
    import gzip
    import json
    import re
    import sys
    import time
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np
    import xgboost as xgb
    from rapidfuzz import fuzz

    sys.path.insert(0, str(Path(__file__).parent))
    from boundary_exp import CE, FUS, PRUNE, RAW, V4, W, key, norm
    from distractor_check import f05, partition_of


    def log(*a):
        print(time.strftime('%H:%M:%S'), *a, flush=True)


    def loose(name):
        return ' '.join(sorted({t for t in re.sub(r'\d+', ' ', key(name)).split() if len(t) >= 3}))


    def main():
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
        S = np.load(FUS / 'stage1.npz')
        y = np.load(V4 / 'pairs/y.npy')[S['rows']].astype(np.float32)
        s1f = np.where(S['stage1'].astype(np.float64) < 0.02, 0.0, S['stage1'].astype(np.float64))
        ce = np.nan_to_num(np.load(CE).astype(np.float64), nan=0.0)
        owner = {}
        with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) > 1 and q[1]:
                    for t in q[1].split(','):
                        owner[t.strip()] = q[0]
        wt = np.where((y == 0) & np.array([t not in owner for t in tgts]), W, 1.0).astype(np.float32)
        tr, dv, ho = np.isin(part, ['trainA', 'trainB']), part == 'dev', np.flatnonzero(part == 'holdout')
        X = np.column_stack([s1f, ce]).astype(np.float32)
        params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                      max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)
        d = xgb.QuantileDMatrix(X[tr], y[tr], weight=wt[tr])
        b = xgb.train(params, d, 3000, evals=[(xgb.QuantileDMatrix(X[dv], y[dv], ref=d), 'dev')],
                      early_stopping_rounds=100, verbose_eval=False)
        P = np.asarray(b.inplace_predict(X[ho], iteration_range=(0, b.best_iteration + 1)), dtype=np.float64)
        log('L2 done')

        need = set(refs[ho]) | set(tgts[ho]) | {owner[t] for t in tgts[ho] if t in owner}
        text, kc, lc = {}, Counter(), Counter()
        for k in (1, 2, 3):
            with open(RAW / f'train_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, iname, iaddr = head.index('entity_id'), head.index('business_name'), head.index('business_address')
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if len(q) <= iaddr:
                        continue
                    if k == 1:
                        kc[key(q[iname])] += 1; lc[loose(q[iname])] += 1
                    if q[ie] in need:
                        text[q[ie]] = (q[iname], q[iaddr])
        log('text + S1 name counts', len(text), len(kc))

        th, yh, rh = tgts[ho], y[ho], refs[ho]
        chosen = P >= 0.68
        blank = np.array([not text.get(t, ('', ''))[1].strip() for t in th])
        by_ref = defaultdict(list)
        for j, r in enumerate(rh):
            by_ref[r].append(j)
        H = sorted(s for s in gold if cp[s] == 'holdout')

        def macro(sel):
            return float(np.mean([f05(gold[s], {th[j] for j in by_ref.get(s, []) if sel[j]}) for s in H]))

        E = {'n_holdout_rows': int(len(ho)), 'blank_rows': int(blank.sum()), 'blank_pos': int(yh[blank].sum()),
             'blank_share_rows': round(float(blank.mean()), 4)}
        base = macro(chosen)
        E['ceilings'] = {'s5': round(base, 6),
                         'blank_rows_by_label': round(macro(np.where(blank, yh > 0, chosen)), 6),
                         'nonblank_rows_by_label': round(macro(np.where(blank, chosen, yh > 0)), 6),
                         'all_by_label': round(macro(yh > 0), 6)}
        E['errors'] = {'blank_fn': int((blank & (yh > 0) & ~chosen).sum()), 'blank_fp': int((blank & (yh == 0) & chosen).sum()),
                       'nonblank_fn': int((~blank & (yh > 0) & ~chosen).sum()), 'nonblank_fp': int((~blank & (yh == 0) & chosen).sum())}
        log(E)

        feats = defaultdict(dict)
        for r, js in by_ref.items():
            rn = text.get(r, ('', ''))[0]
            rk = key(rn)
            acc = [j for j in js if chosen[j] and not blank[j]]
            for j in js:
                if not blank[j]:
                    continue
                tn = text.get(th[j], ('', ''))[0]
                tk, tl = key(tn), loose(tn)
                sib = max((fuzz.token_sort_ratio(norm(tn), norm(text.get(th[q], ('', ''))[0])) for q in acc if q != j), default=-1)
                same_src = any(th[q][:2] == th[j][:2] for q in acc)
                m = kc.get(tk, 0)
                ml = lc.get(tl, 0)
                feats[j] = {'amb_key': '0' if m == 0 else '1' if m == 1 else '2' if m == 2 else '3-5' if m <= 5 else '6+',
                            'amb_loose': '0' if ml == 0 else '1' if ml == 1 else '2' if ml == 2 else '3-5' if ml <= 5 else '6+',
                            'key_eq_ref': str(bool(rk) and tk == rk),
                            'sibling_sim': 'none' if sib < 0 else '<70' if sib < 70 else '70-89' if sib < 90 else '90+',
                            'ref_has_same_src': str(same_src)}
        band = [j for j in feats if 0.1 <= P[j] < 0.9]
        E['blank_band_rows'] = len(band)
        for f in ('amb_key', 'amb_loose', 'key_eq_ref', 'sibling_sim', 'ref_has_same_src'):
            for nm, rows in (('all', list(feats)), ('band', band)):
                g = defaultdict(list)
                for j in rows:
                    g[feats[j][f]].append(j)
                E[f'{f}/{nm}'] = {v: {'n': len(js), 'pos_rate': round(float(yh[js].mean()), 4),
                                      'mean_p': round(float(P[js].mean()), 4), 'fn': int(((yh[js] > 0) & ~chosen[js]).sum()),
                                      'fp': int(((yh[js] == 0) & chosen[js]).sum())} for v, js in sorted(g.items())}
        neg = Counter()
        for j in band:
            if yh[j] == 0:
                o = owner.get(th[j])
                if o is None:
                    neg['decoy'] += 1
                else:
                    on = key(text.get(o, ('', ''))[0]) if o in text else None
                    neg['owned_other_same_key' if on is not None and on == key(text.get(th[j], ('', ''))[0])
                        else 'owned_other'] += 1
        E['band_negatives'] = dict(neg)
        (FUS / 'blank_eda.json').write_text(json.dumps(E, indent=1))
        log('done')


    if __name__ == '__main__':
        main()



def cmd_blank_expert():
    """Fixes for blank-address misses on top of stage-3b (cached features from boundary_exp2.py).

  A  per-segment cutoffs: separate dev-chosen thresholds for blank / non-blank rows on the stage-3b probability
  B  stage-3b + disagreement/identity features on all rows: decoy word added (train: 0% positives), benign noise
     word added, exact name = reference, exact name = an accepted non-blank sibling, same-name siblings in the list,
     v5 - sigmoid(ce), |.|, cross-encoder rank and v5 rank within the list
  C  blank expert: model B's features, trained on blank rows only (non-blank rows keep model B)
Rule chosen on dev, holdout once. Writes fusion/blank_expert.json.
"""
    import gzip
    import json
    import sys
    import time
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import xgboost as xgb

    sys.path.insert(0, str(Path(__file__).parent))
    from blank_disagree import DECOY, toks
    from boundary_exp import CE, FUS, OUT, PRUNE, RAW, V4, W
    from distractor_check import decide, f05, partition_of

    BENIGN = {'center', 'services', 'service', 'partners', 'trust', 'the', 'and', 'com'}


    def log(*a):
        print(time.strftime('%H:%M:%S'), *a, flush=True)


    def main():
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
        S = np.load(FUS / 'stage1.npz')
        y = np.load(V4 / 'pairs/y.npy')[S['rows']].astype(np.float32)
        v5 = np.where(S['stage1'] < 0.02, 0.0, S['stage1']).astype(np.float64)
        ce = np.nan_to_num(np.load(CE).astype(np.float64), nan=0.0)
        C = np.load(OUT / 'exp2_cache.npz')
        base = np.column_stack([C['F'], C['Frev'], C['Fnum']]).astype(np.float32)
        P, blank = C['P'], C['blank']
        owned = set()
        with open(RAW / 'train_ground_truth.tsv', encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if len(q) > 1 and q[1]:
                    owned.update(t.strip() for t in q[1].split(','))
        wt = np.where((y == 0) & np.array([t not in owned for t in tgts]), W, 1.0).astype(np.float32)
        need = set(refs) | set(tgts)
        name = {}
        for k in (1, 2, 3):
            with open(RAW / f'train_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, iname = head.index('entity_id'), head.index('business_name')
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if q[ie] in need:
                        name[q[ie]] = frozenset(toks(q[iname]))
        log('names')
        by_ref = defaultdict(list)
        for j, r in enumerate(refs):
            by_ref[r].append(j)
        X = np.zeros((len(y), 9), dtype=np.float32)
        sig = 1 / (1 + np.exp(-ce))
        for r, js in by_ref.items():
            rt = name.get(r, frozenset())
            tn = [name.get(tgts[j], frozenset()) for j in js]
            acc = [P[j] >= 0.5 and not blank[j] for j in js]
            rc = np.empty(len(js)); rc[np.argsort(-ce[js], kind='stable')] = np.arange(1, len(js) + 1)
            rv = np.empty(len(js)); rv[np.argsort(-v5[js], kind='stable')] = np.arange(1, len(js) + 1)
            for q, j in enumerate(js):
                t = tn[q]
                X[j] = [bool((t - rt) & DECOY), bool((t - rt) & BENIGN), bool(t) and t == rt,
                        any(acc[z] and z != q and tn[z] == t for z in range(len(js))) and bool(t),
                        sum(tn[z] == t and not blank[js[z]] for z in range(len(js)) if z != q),
                        v5[j] - sig[j], abs(v5[j] - sig[j]), rc[q], rv[q]]
        log('extra features')
        XB = np.column_stack([base, X])
        tr, dv = np.isin(part, ['trainA', 'trainB']), part == 'dev'
        params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
                      max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, seed=3)

        def fit(Xm, rows, dev_rows):
            d = xgb.QuantileDMatrix(Xm[rows], y[rows], weight=wt[rows])
            b = xgb.train(params, d, 3000, evals=[(xgb.QuantileDMatrix(Xm[dev_rows], y[dev_rows], ref=d), 'dev')],
                          early_stopping_rounds=100, verbose_eval=False)
            return np.asarray(b.inplace_predict(Xm, iteration_range=(0, b.best_iteration + 1)), dtype=np.float64)

        idx = {p: np.flatnonzero(part == p) for p in ('dev', 'holdout')}
        prefs = {p: sorted(s for s in gold if cp[s] == p) for p in idx}
        rix = {}
        for p in idx:
            pos = {s: i for i, s in enumerate(prefs[p])}
            rix[p] = np.array([pos[r] for r in refs[idx[p]]])

        def macro(p, chosen):
            pred = defaultdict(set)
            for i, tt in zip(rix[p][chosen], tgts[idx[p]][chosen]):
                pred[i].add(tt)
            return float(np.mean([f05(gold[s], pred.get(i, set())) for i, s in enumerate(prefs[p])]))

        grid = [(ru, round(float(t), 2)) for ru in ('expected', 'threshold') for t in np.arange(0.4, 0.961, 0.02)]
        R = {'ref': 'stage3b expected@0.60 dev 0.99124 / holdout 0.98985'}

        def by_rule(nm, prob):
            dev = {g: macro('dev', decide(rix['dev'], prob[idx['dev']], *g, len(prefs['dev']))) for g in grid}
            g = max(dev, key=lambda k: (dev[k], k[1]))
            R[nm] = {'rule': g, 'dev': round(dev[g], 6),
                     'holdout': round(macro('holdout', decide(rix['holdout'], prob[idx['holdout']], *g, len(prefs['holdout']))), 6)}
            log(nm, R[nm])
            (FUS / 'blank_expert.json').write_text(json.dumps(R, indent=1))

        P3 = np.load(OUT / 'stage3b_prob.npy')
        by_rule('stage3b_recheck', P3)
        # A: per-segment thresholds on stage-3b
        ts = np.arange(0.4, 0.961, 0.04)
        bl = {p: blank[idx[p]] for p in idx}
        dev = {(a, b): macro('dev', np.where(bl['dev'], P3[idx['dev']] >= a, P3[idx['dev']] >= b)) for a in ts for b in ts}
        a, b = max(dev, key=dev.get)
        R['A_segment_thresholds'] = {'blank_t': round(float(a), 2), 'nonblank_t': round(float(b), 2), 'dev': round(dev[(a, b)], 6),
                                     'holdout': round(macro('holdout', np.where(bl['holdout'], P3[idx['holdout']] >= a, P3[idx['holdout']] >= b)), 6)}
        log('A', R['A_segment_thresholds'])
        PB = fit(XB, tr, dv)
        by_rule('B_stage3b+identity', PB)
        PC = PB.copy()
        PC[blank] = fit(XB, tr & blank, dv & blank)[blank]
        by_rule('C_blank_expert', PC)
        np.save(OUT / 'blank_expert_prob.npy', PC)
        log('done')


    if __name__ == '__main__':
        main()



def cmd_blind_sample():
    """Blind judge set: 80 holdout pairs from T6's decision band (0.15 < p < 0.85), with the context an LLM prompt would
get (S1 record, candidate record, the S1's other candidates, how many other S1s share the S1's exact normalized name).
Questions file has no label and no p; the key file (id, y, p, blank) is written separately. Env A (sweep dir), OUT_Q,
OUT_K.
"""
    import os
    import re
    import sys
    from collections import Counter
    from pathlib import Path

    import pandas as pd

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    sys.path.insert(0, str(Path(__file__).parent))
    from france_census import fold

    H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
    nrm = lambda s: re.sub(r'[^a-z0-9]+', ' ', fold(s or '')).strip()
    df = pd.read_pickle(f"{os.environ['A']}/train.json.train_pairs.pkl")
    band = df[(df.part == 'holdout') & (df.p > 0.15) & (df.p < 0.85)]
    smp = pd.concat([band[band.blank == 'blank'].sample(45, random_state=7), band[band.blank != 'blank'].sample(35, random_state=7)])
    smp = smp.sample(frac=1, random_state=3).reset_index(drop=True)
    need = set(df[df.s1.isin(smp.s1)].t) | set(smp.s1)
    text, s1names = {}, Counter()
    for k in (1, 2, 3):
        with open(H / f'data/dataset/train/train_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if k == 1:
                    s1names[(q[c], nrm(q[n]))] += 1
                if q[ie] in need:
                    text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
    with open(os.environ['OUT_Q'], 'w', encoding='utf-8') as fq, open(os.environ['OUT_K'], 'w') as fk:
        fk.write('id\ty\tp\tblank\n')
        for i, r in smp.iterrows():
            n, a, c = text[r.s1]
            tw = s1names[(c, nrm(n))] - 1
            others = [t for t in df[df.s1 == r.s1].t if t != r.t]
            fq.write(f'Q{i:02d} | S1: {n} | {a} | {c} | other S1s with this exact name: {tw}\n')
            fq.write(f'     CANDIDATE [{r.t[:2]}]: {text[r.t][0]} | {text[r.t][1] or "(blank)"}\n')
            for t in others:
                fq.write(f'       other cand [{t[:2]}]: {text[t][0]} | {text[t][1] or "(blank)"}\n')
            fk.write(f'Q{i:02d}\t{r.y}\t{r.p:.4f}\t{r.blank}\n')
    print('written', len(smp))



def cmd_empty_guard():
    """Empty-guard check on T6 dev/holdout: when the expected-F0.5 rule leaves an S1 empty, add its top candidate if p >= x.

Holdout loss has 0.225 points on S1s predicted empty that have matches vs 0.007 on true singletons given a link, so
the empty choice may be too eager (prod(1 - p) treats candidates as independent, ignoring that 94% of S1s have a match).
Uses error_sweep.py frames. Also profiles the wrongly-empty S1s. Env A (sweep dir).
"""
    import os
    from collections import Counter

    import numpy as np
    import pandas as pd

    A = os.environ['A']
    df = pd.read_pickle(f'{A}/train.json.train_pairs.pkl')
    bf = pd.read_pickle(f'{A}/train.json.train_blocking.pkl')
    miss = bf.groupby('s1').size()
    g = df.groupby('s1')
    S = pd.DataFrame({'part': g.part.first(), 'country': g.country.first(), 'ytrue': g.y.sum(), 'npred': g.D.sum(),
                      'tp': (df.y * df.D).groupby(df.s1).sum(), 'top_p': g.p.max(), 'n': g.size()})
    S['ngold'] = S.ytrue + miss.reindex(S.index).fillna(0).astype(int)
    top = df.sort_values('p', ascending=False).drop_duplicates('s1').set_index('s1')
    S['top_y'] = top.y.reindex(S.index); S['top_blank'] = top.blank.reindex(S.index); S['top_name'] = top.name.reindex(S.index)
    S['sum_p'] = g.p.sum()


    def F(tp, npred, ngold):
        tp, npred, ngold = map(np.asarray, (tp, npred, ngold))
        pr = np.where(npred > 0, tp / np.maximum(npred, 1), 0); rc = np.where(ngold > 0, tp / np.maximum(ngold, 1), 0)
        f = np.where(tp > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
        return np.where((npred == 0) & (ngold == 0), 1.0, f)


    S['F'] = F(S.tp, S.npred, S.ngold)
    for p in ('dev', 'holdout'):
        s = S[S.part == p]
        e = s[s.npred == 0]
        print(p, 'F', round(s.F.mean(), 6), 'pred-empty', len(e), 'of', len(s), 'wrong-empty', int((e.ngold > 0).sum()),
              'true singletons', int((s.ngold == 0).sum()), 'singleton FP', int(((s.ngold == 0) & (s.npred > 0)).sum()))
        print('  wrong-empty: top_y', Counter(e[e.ngold > 0].top_y), 'top blank', Counter(e[e.ngold > 0].top_blank),
              'ngold', Counter(e[e.ngold > 0].ngold.clip(upper=6)), 'ytrue in list', Counter(e[e.ngold > 0].ytrue.clip(upper=4)))
        for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .74), (.74, 1.01)):
            b = e[(e.top_p >= lo) & (e.top_p < hi)]
            print(f'  empty, top_p {lo:.2f}-{hi:.2f}: n {len(b)} has match {int((b.ngold > 0).sum())} top true {int((b.top_y == 1).sum())}'
                  f' sum_p mean {b.sum_p.mean():.2f}')
    for x in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        out = []
        for p in ('dev', 'holdout'):
            s = S[S.part == p]
            add = (s.npred == 0) & (s.top_p >= x)
            f2 = np.where(add, F(s.tp + s.top_y, s.npred + 1, s.ngold), s.F)
            out.append(f'{p} {f2.mean() - s.F.mean():+.6f} (n {int(add.sum())})')
        print('guard top_p >=', x, ' | '.join(out))
    for x in (0.6, 0.8, 1.0, 1.2, 1.5):
        out = []
        for p in ('dev', 'holdout'):
            s = S[S.part == p]
            add = (s.npred == 0) & (s.sum_p >= x)
            f2 = np.where(add, F(s.tp + s.top_y, s.npred + 1, s.ngold), s.F)
            out.append(f'{p} {f2.mean() - s.F.mean():+.6f} (n {int(add.sum())})')
        print('guard sum_p >=', x, ' | '.join(out))



def cmd_error_sweep_report():
    """Tables from error_sweep.py outputs: loss per stage and pattern cell, train (actual, dev + holdout) vs test (expected),
and the public-score ceiling of fixing each cell.

Public impact, US/India: holdout loss rate per country x test S1 share of that country (score units, 1.0 = perfect).
Cross-check: test expected loss in the cell x the cell's holdout actual/expected ratio. France: test expected only
(no labels; census-changed pairs are counted at the model's own p, i.e. as the model would score them).
Env TRAIN (json of MODE=train), TEST (json of MODE=test), OUT (markdown).
"""
    import json
    import os

    import pandas as pd

    TR, TE = os.environ['TRAIN'], os.environ['TEST']
    R = json.load(open(TR)); RT = json.load(open(TE))
    df = pd.read_pickle(TR + '.train_pairs.pkl'); bf = pd.read_pickle(TR + '.train_blocking.pkl')
    sf = pd.read_pickle(TR + '.train_s1.pkl')
    tf = pd.read_pickle(TE + '.test_pairs.pkl')
    NT = RT['n_s1']; NTOT = sum(NT.values())
    NH = {c: int((sf.part.eq('holdout') & sf.country.eq(c)).sum()) for c in ('US', 'India')}
    ND = {c: int((sf.part.eq('dev') & sf.country.eq(c)).sum()) for c in ('US', 'India')}
    USIN = ('US', 'India')
    out = []


    def w(s=''):
        out.append(s)


    def public_from_holdout(g, part='holdout', col='loss'):
        N = NH if part == 'holdout' else ND
        return sum(g[(g.part == part) & (g.country == c)][col].sum() / N[c] * NT.get(c, 0) / NTOT for c in USIN)


    # ---------------- stage summary ----------------
    w('# Error sweep: T6 (US/India) + U2 France, train actual vs test expected')
    w()
    w(f"Test S1: {NT} (total {NTOT}). Holdout S1 US {NH['US']} India {NH['India']}; dev US {ND['US']} India {ND['India']}.")
    w()
    w('## 1. Stage totals (loss points = 100 x mean(1 - F0.5) over S1)')
    w()
    for p in ('dev', 'holdout'):
        t = R['totals'][p]
        w(f"- **{p}**: total {t['loss_points']} (in-list expected {t['exp_points_inlist']}); blocking {t['blocking_points']} "
          f"{t['by_stage']}; decision FN {t['decision_FN']}, FP {t['decision_FP']}; by S1 case {t['case_points']}; S1 with >10 gold: {t['gold_over_10']}")
    w(f"- **test expected in-list points** by country: {RT['expected_points_inlist']}; public-weighted {RT['expected_points_public_inlist']}")
    w(f"- test pred-empty rate {RT['pred_empty_rate']}; S1 with all 10 accepted {RT['all10_accepted']}; post-processing steps {RT['steps']}")
    hold_exp = sf[sf.part == 'holdout'].groupby('country').exp_loss.mean() * 100
    hold_act = sf[sf.part == 'holdout'].groupby('country').loss.mean() * 100
    w(f"- holdout actual vs in-list expected by country: " + ', '.join(f"{c}: {hold_act[c]:.3f} vs {hold_exp[c]:.3f}" for c in USIN))
    w()

    # ---------------- decision cells ----------------
    E = df[df.err == 1]
    keys = ['ekind', 'blank', 'name', 'addr']
    TU = tf[tf.country.isin(USIN)]
    te = TU.groupby(keys).agg(test_n=('p', 'size'), test_exp=('exp_loss', 'sum'))
    tr = df.groupby(keys).agg(train_n=('p', 'size'), exp=('exp_loss', 'sum'))
    rows = []
    for key, ge in E.groupby(keys):
        r = dict(zip(keys, key))
        r['err_hold'] = int((ge.part == 'holdout').sum()); r['err_dev'] = int((ge.part == 'dev').sum())
        r['pub_hold'] = public_from_holdout(ge); r['pub_dev'] = public_from_holdout(ge, 'dev')
        ex = tr.exp.get(key, 0.0)
        r['act/exp'] = ge.loss.sum() / ex if ex > 0 else float('nan')
        r['test_exp_pub'] = te.test_exp.get(key, 0.0) / NTOT
        r['test_cal_pub'] = r['test_exp_pub'] * (r['act/exp'] if ex > 0 else 1)
        r['share_tr'] = tr.train_n.get(key, 0) / len(df); r['share_te'] = te.test_n.get(key, 0) / max(1, len(TU))
        rows.append(r)
    C = pd.DataFrame(rows).sort_values('pub_hold', ascending=False)
    w('## 2. US/India decision errors by cell (public-score ceiling if the cell were fixed)')
    w()
    w('pub_* = score units on the public LB (0.001 = +0.1 points). pub_hold/pub_dev from actual labelled loss reweighted to '
      'the test country mix; test_exp_pub = label-free expected loss on test; test_cal_pub = test_exp x holdout act/exp.')
    w()
    w(C.head(30).to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    w()
    w(f"Sum pub_hold (decision, US/India): {C.pub_hold.sum():.5f}; sum test_cal_pub: {C.test_cal_pub.sum():.5f}")
    w()

    # ---------------- one-way breakdowns ----------------
    w('## 3. One-way breakdowns of US/India decision loss (pub_hold | pub_dev | test_exp_pub)')
    for k in ('mult', 'shift', 'pband', 'case', 'src', 'country'):
        w()
        w(f'### by {k}')
        rr = []
        for key, g in E.groupby([k, 'ekind']):
            tu = tf[(tf.country.isin(USIN)) & (tf[k] == key[0]) & (tf.ekind == key[1])] if k in tf else None
            rr.append({k: key[0], 'ekind': key[1], 'err_hold': int((g.part == 'holdout').sum()), 'pub_hold': public_from_holdout(g),
                       'pub_dev': public_from_holdout(g, 'dev'),
                       'test_exp_pub': tu.exp_loss.sum() / NTOT if tu is not None else float('nan')})
        w(pd.DataFrame(rr).sort_values('pub_hold', ascending=False).to_string(index=False, float_format=lambda x: f'{x:.5f}'))

    # ---------------- blank-address ties: multiplicity x name relation ----------------
    w()
    w('## 4. Blank-address errors by S1 name multiplicity x name relation (pub_hold)')
    B = E[E.blank == 'blank']
    w(B.groupby(['ekind', 'mult', 'name']).apply(lambda g: pd.Series({'err_hold': int((g.part == 'holdout').sum()), 'pub_hold': public_from_holdout(g), 'pub_dev': public_from_holdout(g, 'dev')}))
      .reset_index().sort_values('pub_hold', ascending=False).head(20).to_string(index=False, float_format=lambda x: f'{x:.5f}'))

    # ---------------- blocking ----------------
    w()
    w('## 5. Blocking misses (true record not in the top-10 list)')
    rr = []
    for key, g in bf.groupby(['stage', 'blank', 'name', 'addr']):
        rr.append(dict(zip(['stage', 'blank', 'name', 'addr'], key), n_hold=int((g.part == 'holdout').sum()),
                       pub_hold=public_from_holdout(g), pub_dev=public_from_holdout(g, 'dev')))
    w(pd.DataFrame(rr).sort_values('pub_hold', ascending=False).head(20).to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    w()
    w('by case / gold size: ' + str(bf[bf.part == 'holdout'].groupby('case').loss.sum().round(2).to_dict()) +
      ' | gold size of missed: ' + str(bf[bf.part == 'holdout'].ngold.clip(upper=12).value_counts().sort_index().to_dict()))

    # ---------------- calibration ----------------
    w()
    w('## 6. Calibration of uncertain pairs (0.05 < p < 0.95), dev + holdout, US/India')
    U = df[(df.p > 0.05) & (df.p < 0.95)]
    cal = U.groupby(['blank', 'name', 'addr']).agg(n=('y', 'size'), mean_p=('p', 'mean'), true_rate=('y', 'mean')).reset_index()
    cal = cal[cal.n >= 100].sort_values('n', ascending=False)
    w(cal.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    w()
    rc = R['recalibration']
    w(f"Per-cell logit recalibration (dev-fitted, holdout once): base {rc['holdout_base']}, all cells {rc['all_cells_delta']:+.6f}")
    w(pd.DataFrame(rc['cells'], columns=['cell', 'shift', 'n', 'holdout_delta']).head(15).to_string(index=False))

    # ---------------- France ----------------
    w()
    w('## 7. France (test, expected loss at FRtrained_w4 p, U2 decisions)')
    F = tf[tf.country == 'France']
    fr = F.groupby(['ekind', 'blank', 'name', 'addr', 'step']).agg(n=('p', 'size'), mean_p=('p', 'mean'), exp_pub=('exp_loss', 'sum')).reset_index()
    fr['exp_pub'] /= NTOT
    w(fr.sort_values('exp_pub', ascending=False).head(30).to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    w()
    w('France by step: ' + str((F.groupby('step').exp_loss.sum() / NTOT).round(5).to_dict()))
    w('France by shift: ' + str((F.groupby(['ekind', 'shift']).exp_loss.sum() / NTOT).round(5).sort_values(ascending=False).head(12).to_dict()))
    w('France by mult: ' + str((F.groupby(['ekind', 'mult']).exp_loss.sum() / NTOT).round(5).sort_values(ascending=False).to_dict()))
    w()
    w('US/India test by step: ' + str((tf[tf.country.isin(USIN)].groupby('step').exp_loss.sum() / NTOT).round(5).to_dict()))
    open(os.environ['OUT'], 'w', encoding='utf-8').write('\n'.join(out))
    print('\n'.join(out))



def cmd_excl_empty():
    """Exclusivity finalize on test: when a record is claimed by several S1s, the highest p keeps it and the rest re-decide.
How often does the loser S1 end with NO link (full F0.5 = 0 if it has matches), who won the record, with what p, and
how many links the winner keeps. Uses error_sweep.py test frames + U2 final + the T6 / FRtrained_w4 probabilities.
Env A (sweep dir).
"""
    import os
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import pandas as pd

    A = os.environ['A']
    V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
    H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
    tf = pd.read_pickle(f'{A}/test.json.test_pairs.pkl')
    final = {}
    with open(H / 'artifacts/submission_U2_final/matching_results.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            s, m = (line.rstrip('\n').split('\t') + [''])[:2]
            final[s] = [x for x in m.split(',') if x]
    owner = {t: s for s, ts in final.items() for t in ts}
    X = tf[tf.step == 'exclusivity_drop'].copy()
    need = set(X.t)
    s1 = []
    with open(H / 'data/dataset/test/test_source1.tsv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            s1.append(line.split('\t', 1)[0])
    P = {}
    for f, fr in ((V / 'submission_T6_own/late_scores.npz', False), (V / 'submission_FRtrained_w4/late_scores.npz', True)):
        Z = np.load(f, allow_pickle=True)
        tg = Z['target'].astype(str); ri = Z['ref_idx']; p = Z['prob']
        for j in np.flatnonzero(np.isin(tg, list(need))):
            P[(s1[ri[j]], tg[j], fr)] = float(p[j])
    X['winner'] = X.t.map(owner)
    X['L_final'] = X.s1.map(lambda s: len(final.get(s, [])))
    X['W_final'] = X.winner.map(lambda s: len(final.get(s, [])) if isinstance(s, str) else -1)
    X['pW'] = [P.get((w, t, c == 'France'), np.nan) if isinstance(w, str) else np.nan for w, t, c in zip(X.winner, X.t, X.country)]
    X['nclaim'] = X.t.map(tf[tf.p > 0.5].groupby('t').size()).fillna(0)
    for c, g in X.groupby('country'):
        e = g[g.L_final == 0]
        print(f'{c}: dropped {len(g)} | winner none {int(g.winner.isna().sum())} | loser emptied {len(e)} ({e.s1.nunique()} S1) '
              f'| emptied & winner keeps >=2 links {int((e.W_final >= 2).sum())} | pL mean {e.p.mean():.3f} pW mean {e.pW.mean():.3f}')
        print('   emptied by blank/name:', dict(Counter(zip(e.blank, e.name)).most_common(6)))
        print('   emptied pL bands:', dict(Counter(pd.cut(e.p, [0, .5, .74, .9, .98, 1.01]).astype(str))))
        print('   emptied winner link count:', dict(Counter(e.W_final.clip(upper=6))))
        print('   emptied mult:', dict(Counter(e.mult)), ' claimants(p>0.5):', dict(Counter(e.nclaim.clip(upper=5))))
        print('   public points if every emptied S1 were a true non-singleton owning t (ceiling):',
              round(100 * e.s1.nunique() / 1732544, 4))



def cmd_nocand_usin():
    """US/India copy-type records that retrieval never returned: counts and a trial file."""
    import sys, os
    from collections import Counter
    sys.path.insert(0, '.')
    import pattern_audit as pa
    def load_rows(f):
        rows, d = [], {}
        for i, l in enumerate(open(f, encoding='utf-8')):
            if i:
                s, m = (l.rstrip('\n').split('\t') + [''])[:2]; rows.append(s); d[s] = [x for x in m.split(',') if x]
        return rows, d
    order, u = load_rows('U15b/matching_results.tsv'); _, cand = load_rows('U15b/candidate_pairs.tsv')
    owner = {t: s for s, ts in u.items() for t in ts}; cs = {s: set(v) for s, v in cand.items()}
    OWN_OK = {'exact', 'same_core', 'dropped_word', 'initials', 'handle'}
    NONUM_OK = {'exact', 'same_core', 'dropped_word', 'initials'}
    s1, own, street, names = pa.build_index('test')
    R = Counter(); adds = []
    for k in (2, 3):
        f = open(pa.D + f'test/test_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
        ie, n, a, c = (h.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
        for l in f:
            q = l.rstrip('\n').split('\t'); rid, rn, ra, cc = q[ie], q[n], (q[a] if len(q) > a else ''), q[c]
            if cc == 'France' or rid in owner: continue
            tag = None
            if ra.strip():
                num, words = pa.akey(ra)
                hit = own.get((cc, num, words)) if num and words else None
                if hit and len(hit) == 1:
                    r = pa.rel(s1[hit[0]][0], rn)[0]
                    if r in OWN_OK and rid not in cs[hit[0]]: tag = ('own_' + r, hit[0])
                elif not num and words:
                    h2 = street.get((cc, words))
                    if h2 and len(h2) == 1:
                        r = pa.rel(s1[h2[0]][0], rn)[0]
                        if r in NONUM_OK and rid not in cs[h2[0]]: tag = ('nonum_' + r, h2[0])
            else:
                hit = names.get((cc, pa.nrm(rn)))
                if hit and len(hit) == 1 and rid not in cs[hit[0]]: tag = ('blank_unique_exact', hit[0])
            if tag:
                R[(cc, tag[0])] += 1; adds.append((tag[1], rid, cc + ':' + tag[0])); owner[rid] = tag[1]
    for k_ in sorted(R): print(k_, R[k_])
    for s, rid, tg in adds:
        u[s].append(rid); cand[s].append(rid)
    os.makedirs('U15c', exist_ok=True)
    for fn, d, col in (('matching_results.tsv', u, 'matched_entity_ids'), ('candidate_pairs.tsv', cand, 'candidate_entity_ids')):
        with open('U15c/' + fn, 'w', encoding='utf-8') as fo:
            fo.write(f'source1_entity_id\t{col}\n')
            for s in order: fo.write(s + '\t' + ','.join(d[s]) + '\n')
    open('U15c/added_vs_U15b.tsv', 'w').write('source1_entity_id\tentity_id\trule\n' + ''.join(f'{a}\t{b}\t{c}\n' for a, b, c in adds))
    print('added', len(adds), 'dups', sum(v > 1 for v in Counter(t for ts in u.values() for t in ts).values()))



def cmd_pattern_cond():
    """Pattern cells (pattern_audit.py definitions) on the US/India dev + holdout top-10 pairs, split by T6's decision D:
true rate of the pairs T6 rejects / accepts per cell, and the F0.5 change on dev and holdout if a cell's rejected pairs
were all added (or accepted pairs all dropped)."""
    import os, sys
    from collections import defaultdict
    import numpy as np, pandas as pd
    sys.path.insert(0, os.path.dirname(__file__))
    import pattern_audit as pa
    pa.D = '/home/scai/mtech/aib262015/scratch/amazon_ml_2026/data/dataset/'
    s1, own, street, names = pa.build_index('train')
    df = pd.read_pickle(f"{os.environ['A']}/train.json.train_pairs.pkl"); bf = pd.read_pickle(f"{os.environ['A']}/train.json.train_blocking.pkl")
    need = set(df.t); rec = {}
    for k in (2, 3):
        f = open(pa.D + f'train/train_source{k}.tsv', encoding='utf-8'); h = f.readline().rstrip('\n').split('\t')
        ie, n, a = (h.index(x) for x in ('entity_id', 'business_name', 'business_address'))
        for l in f:
            q = l.rstrip('\n').split('\t')
            if q[ie] in need: rec[q[ie]] = (q[n], q[a] if len(q) > a else '')
    def cell(s, t):
        rn, ra = rec.get(t, ('', '')); sn, sa, cc = s1[s]
        if ra.strip():
            num, words = pa.akey(ra); hit = own.get((cc, num, words)) if num and words else None
            if hit and len(hit) == 1:
                return ('A_own_address', pa.rel(sn, rn)[0]) if hit[0] == s else None
            hit = street.get((cc, words)) if words else None
            if hit and len(hit) == 1 and hit[0] == s and num:
                snum = pa.akey(sa)[0]
                if snum:
                    return ('B_street_' + pa.shift(snum, num), pa.rel(sn, rn)[0])
            return None
        hit = names.get((cc, pa.nrm(rn)))
        return ('C_blank_unique_name', 'exact') if hit and len(hit) == 1 and hit[0] == s else None
    df['cell'] = [cell(s, t) for s, t in zip(df.s1, df.t)]
    d = df[df.cell.notna()].copy(); d['cell'] = d.cell.map(lambda c: c[0] + ':' + c[1])
    T = d.groupby(['cell', 'D']).agg(n=('y', 'size'), true_rate=('y', 'mean')).unstack('D')
    T = T[(T[('n', 0)].fillna(0) + T[('n', 1)].fillna(0)) >= 50]
    pd.set_option('display.width', 200); print(T.round(3).to_string())
    miss = bf.groupby('s1').size(); g = df.groupby('s1')
    S = pd.DataFrame({'part': g.part.first(), 'ytrue': g.y.sum(), 'npred': g.D.sum(), 'tp': (df.y * df.D).groupby(df.s1).sum()})
    S['ngold'] = S.ytrue + miss.reindex(S.index).fillna(0).astype(int)
    def Fs(tp, npred, ngold):
        tp, npred, ngold = map(np.asarray, (tp, npred, ngold))
        pr = np.where(npred > 0, tp / np.maximum(npred, 1), 0); rc = np.where(ngold > 0, tp / np.maximum(ngold, 1), 0)
        f = np.where(tp > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
        return np.where((npred == 0) & (ngold == 0), 1.0, f)
    print('\nper-cell action on dev / holdout (ADD = add all T6-rejected pairs of the cell; DROP = drop all accepted):')
    for c in sorted(d.cell.unique()):
        for act, dv in (('ADD', 0), ('DROP', 1)):
            sel = d[(d.cell == c) & (d.D == dv)]
            if len(sel) < 15: continue
            a = sel.groupby('s1').agg(k=('y', 'size'), b=('y', 'sum'))
            out = []
            for p in ('dev', 'holdout'):
                s = S[S.part == p].join(a).fillna({'k': 0, 'b': 0})
                if act == 'ADD': new = Fs(s.tp + s.b, s.npred + s.k, s.ngold)
                else: new = Fs(s.tp - s.b, s.npred - s.k, s.ngold)
                out.append(f'{p} {new.mean() - Fs(s.tp, s.npred, s.ngold).mean():+.6f}')
            print(f'  {act:4s} {c:40s} n={len(sel):5d} true={sel.y.mean():.3f}  ' + '  '.join(out))



def cmd_rowsweep():
    """Every-row check of U5 + the textual-decidability ceiling for any reader (LLM or human).

1. Exact-copy rule. Key = (country, normalized full name, normalized address); records whose key matches exactly one
   S1 (non-blank address). Train (FULL ground truth, 2.2M S1): how often is the record that S1's? Also a looser key
   (core-name token set + address token set). Test: U5 status of every such record (linked to that S1 / to another S1 /
   unlinked but in candidates / unlinked and not in candidates).
2. Twins. An S1 whose normalized full name is shared by another S1 of the same country cannot be told apart from it for
   a blank-address record by any reader. Share of the uncertain band and of holdout error loss (error_sweep.py frames)
   on such pairs = the part no LLM can fix.
Env A (sweep dir), U5_DIR.
"""
    import os
    import re
    import sys
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np
    import pandas as pd

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    sys.path.insert(0, str(Path(__file__).parent))
    from france_census import core, fold, load

    H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')


    def nrm(s):
        return re.sub(r'[^a-z0-9]+', ' ', fold(s or '')).strip()


    def keys(name, addr, cty):
        return ((cty, nrm(name), nrm(addr)),
                (cty, ' '.join(sorted(set(core(name)))), ' '.join(sorted(set(nrm(addr).split())))))


    def read(split):
        s1, rec = {}, {}
        for k in (1, 2, 3):
            with open(H / f'data/dataset/{split}/{split}_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    v = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
                    (s1 if k == 1 else rec)[q[ie]] = v
        return s1, rec


    def exact_matches(s1, rec):
        idx = [defaultdict(list), defaultdict(list)]
        for e, (n, a, c) in s1.items():
            if a.strip():
                for i, k in enumerate(keys(n, a, c)):
                    idx[i][k].append(e)
        out = [[], []]
        for r, (n, a, c) in rec.items():
            if not a.strip():
                continue
            for i, k in enumerate(keys(n, a, c)):
                hit = idx[i].get(k)
                if hit and len(hit) == 1:
                    out[i].append((r, hit[0]))
        return out


    def twins(s1):
        c = Counter((v[2], nrm(v[0])) for v in s1.values())
        return {e for e, v in s1.items() if c[(v[2], nrm(v[0]))] > 1}


    def main():
        A = os.environ['A']
        s1, rec = read('train')
        gold = {}
        with open(H / 'data/dataset/train/train_ground_truth.tsv', encoding='utf-8') as fh:
            next(fh)
            for line in fh:
                q = line.rstrip('\n').split('\t')
                gold[q[0]] = set(x.strip() for x in (q[1] if len(q) > 1 else '').split(',') if x.strip())
        owner = {t: s for s, ts in gold.items() for t in ts}
        print('TRAIN records', len(rec), 'S1', len(s1))
        for i, pairs in enumerate(exact_matches(s1, rec)):
            y = np.array([owner.get(r) == s for r, s in pairs])
            other = np.array([owner.get(r) not in (None, s) for r, s in pairs])
            print(f'  key{i} ({"strict" if i == 0 else "loose"}): matches {len(pairs)}, true {y.mean():.5f}, '
                  f'owned by another S1 {other.mean():.5f}, owned by nobody {1 - y.mean() - other.mean():.5f}')
        tw = twins(s1)
        print('  train S1 with an exact-name twin:', len(tw), f'({len(tw) / len(s1):.3%})')
        df = pd.read_pickle(f'{A}/train.json.train_pairs.pkl')
        df['twin'] = df.s1.isin(tw)
        band = df[(df.p > 0.02) & (df.p < 0.98)]
        err = df[df.kind.isin(['FN', 'FP']) & (df.part == 'holdout')]
        for nm, d in (('uncertain band pairs', band), ('holdout error loss', err)):
            w = d.loss if nm.startswith('holdout') else pd.Series(1.0, index=d.index)
            cell = d.assign(w=w).groupby(['blank', 'twin']).w.sum()
            print(f'  {nm}: share by (blank, S1 has exact-name twin):', (cell / cell.sum()).round(4).to_dict())
        del s1, rec, gold, owner
        s1, rec = read('test')
        u5, cand = load(Path(os.environ['U5_DIR']) / 'matching_results.tsv'), load(Path(os.environ['U5_DIR']) / 'candidate_pairs.tsv')
        link = {t: s for s, ts in u5.items() for t in ts}
        candset = {(s, t) for s, ts in cand.items() for t in ts}
        print('TEST records', len(rec), 'S1', len(s1), 'linked records', len(link))
        for i, pairs in enumerate(exact_matches(s1, rec)):
            R = Counter(); ex = defaultdict(list)
            for r, s in pairs:
                c = s1[s][2]
                if link.get(r) == s:
                    k = 'agree'
                elif r in link:
                    k = 'linked_to_other_S1'
                elif (s, r) in candset:
                    k = 'unlinked_in_candidates'
                else:
                    k = 'unlinked_not_candidate'
                R[(c, k)] += 1
                if k != 'agree' and len(ex[(c, k)]) < 6:
                    o = link.get(r)
                    ex[(c, k)].append(f'{s1[s][0]} | {s1[s][1][:45]}  <=  {rec[r][0]} | {rec[r][1][:45]}' +
                                      (f'   [U5 gave it to: {s1[o][0]} | {s1[o][1][:40]}]' if o else ''))
            print(f'  key{i}:', dict(sorted(R.items())))
            for k, e in sorted(ex.items()):
                print('   ', k); [print('      ', x) for x in e]


    if __name__ == '__main__':
        sys.stdout.reconfigure(line_buffering=True)
        main()



def cmd_u5_check():
    """Pre-upload audit of U5 against U2: per-country link diff, exclusivity (a record in at most one S1 row),
candidate superset, France rows = U2 France + additions only, and what the rescue links look like (name / address
relation to the S1, house-number shift class, S1 name multiplicity), with examples. Env U5 (dir with the two TSVs).
"""
    import os
    import random
    from collections import Counter

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    from error_sweep import mkey, shift_class
    from france_census import load
    from usin_errors import rel

    H = '/home/scai/mtech/aib262015/scratch/amazon_ml_2026'
    U5 = os.environ['U5']
    BASE = os.environ.get('BASE', f'{H}/artifacts/submission_U2_final')
    u5, u2 = load(f'{U5}/matching_results.tsv'), load(f'{BASE}/matching_results.tsv')
    c5, c2 = load(f'{U5}/candidate_pairs.tsv'), load(f'{BASE}/candidate_pairs.tsv')
    text, cty, s1 = {}, {}, []
    for k in (1, 2, 3):
        with open(f'{H}/data/dataset/test/test_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                text[q[ie]] = (q[n], q[a] if len(q) > a else '')
                if k == 1:
                    cty[q[ie]] = q[c]; s1.append(q[ie])
    mult = Counter((cty[r], mkey(text[r][0])) for r in s1)
    print('rows', len(u5), len(u2), 'same S1 set', set(u5) == set(u2))
    own = Counter(t for ts in u5.values() for t in ts)
    print('records in >1 S1 row (U5):', sum(v > 1 for v in own.values()), '| U2:', sum(v > 1 for v in Counter(t for ts in u2.values() for t in ts).values()))
    print('matches outside own candidates:', sum(len(set(u5[r]) - set(c5.get(r, []))) for r in u5))
    print('U2 candidates not in U5 candidates:', sum(len(set(c2[r]) - set(c5.get(r, []))) for r in c2),
          '| extra U5 candidates:', sum(len(set(c5[r]) - set(c2.get(r, []))) for r in c5))
    diff = Counter(); rescue = []; emptied = Counter()
    for r in u5:
        a, b = set(u5[r]), set(u2[r])
        diff[(cty[r], 'added')] += len(a - b); diff[(cty[r], 'removed')] += len(b - a)
        if b and not a:
            emptied[cty[r]] += 1
        for t in a - b:
            if t not in set(c2.get(r, [])):
                rescue.append((r, t))
    print('diff', dict(diff), 'emptied', dict(emptied))
    print('France: U2 France rows subset of U5:', all(set(u2[r]) <= set(u5[r]) for r in u5 if cty[r] == 'France'))
    rng = random.Random(0)
    for C in ('France', 'India', 'US'):
        R = [(r, t) for r, t in rescue if cty[r] == C]
        A = Counter(); S = Counter(); M = Counter(); acc_same = Counter()
        ex = []
        for r, t in R:
            rn, ra = text[r]; tn, ta = text.get(t, ('', ''))
            nm, ad = rel(rn, ra, tn, ta)
            A[(nm, ad)] += 1; S[shift_class(ra, ta) if ta.strip() else 'blank'] += 1
            m = mult[(cty[r], mkey(rn))]; M['1' if m <= 1 else ('2-5' if m <= 5 else '6+')] += 1
            others = [x for x in u5[r] if x != t]
            acc_same[len(others)] += 1
            ex.append(f"{rn} | {ra}  =>  {tn} | {ta}   [other links {len(others)}: " + '; '.join(f'{text.get(x, ("", ""))[0]} | {text.get(x, ("", ""))[1]}' for x in others[:3]) + ']')
        print(f'\n== {C} rescue links {len(R)}')
        print('  name x addr:', dict(A.most_common(12)))
        print('  number shift:', dict(S)); print('  S1 name multiplicity:', dict(M)); print('  other links of the S1:', dict(sorted(Counter(min(k, 6) for k in acc_same.elements()).items())))
        for e in rng.sample(ex, min(25 if C == 'France' else 10, len(ex))):
            print('   ', e[:300])



def cmd_usin_patterns():
    """Two generator patterns behind T6's non-blank US/India errors, measured on labelled train pairs (dev + holdout).

alias   the S1 has a candidate 'ALIAS aka/a.k.a./formerly/fka/dba <S1 name>'; another candidate whose core name is the
        alias part (the bare alias) is a sibling of the same alias. True rate, and how many T6 rejects.
digit   house number differs from the S1's: one-digit deletion/insertion (2118 -> 118, 27 -> 278), one-digit
        substitution, small upward shift (+1..+21, the decoy family), other. True rate per type and name relation,
        and T6 decisions per type.
Env OUT, PROB (T6 sample probabilities), RULE_T.
"""
    import gzip
    import json
    import os
    import re
    import sys
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    sys.path.insert(0, str(Path(__file__).parent))
    from distractor_check import decide, partition_of
    from france_census import core, house, street_words, tokens_addr
    from usin_errors import EN_FILLER

    H = Path('/home/scai/mtech/aib262015/scratch')
    V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')
    AKA = re.compile(r'\b(?:a\s*/\s*k\s*/\s*a|a\.?k\.?a\.?|aka|f\s*/\s*k\s*/\s*a|f\.?k\.?a\.?|fka|formerly(?:\s+known\s+as)?|'
                     r'd\s*/\s*b\s*/\s*a|d\.?b\.?a\.?|dba|doing business as|née|nee)\b[:\s]*', re.I)


    def num_rel(a, b):
        a, b = re.sub(r'\D', '', a), re.sub(r'\D', '', b)
        if not a or not b or a == b:
            return 'same_or_none'
        if len(a) == len(b) + 1 and any(a[:i] + a[i + 1:] == b for i in range(len(a))):
            return 'digit_drop'
        if len(b) == len(a) + 1 and any(b[:i] + b[i + 1:] == a for i in range(len(b))):
            return 'digit_insert'
        if 1 <= int(b) - int(a) <= 21:
            return 'shift_up_1_21'
        if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1:
            return 'digit_sub'
        return 'other'


    def main():
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
        P = np.load(os.environ.get('PROB', str(V / 'fusion/boundary_models_t6/stage3b_prob.npy')))
        thr = float(os.environ.get('RULE_T', '0.74'))
        part = np.array([cp[r] for r in refs], dtype=object)
        chosen = np.zeros(len(refs), bool)
        for p in ('dev', 'holdout'):
            ix = np.flatnonzero(part == p)
            prefs = sorted(s for s in gold if cp[s] == p)
            pos = {s: i for i, s in enumerate(prefs)}
            chosen[ix] = decide(np.array([pos[r] for r in refs[ix]]), P[ix], 'expected', thr, len(prefs))
        by = defaultdict(list)
        for j in np.flatnonzero(part != 'train'):
            by[refs[j]].append(j)
        A = Counter(); D = Counter(); ex = defaultdict(list)
        for s, js in by.items():
            rn, ra, cty = text[s]
            rc = set(core(rn))
            aliases = set()
            for j in js:
                tn = text.get(tgts[j], ('', ''))[0]
                parts = AKA.split(tn)
                if len(parts) >= 2:
                    left, right = set(core(parts[0])), set(core(' '.join(parts[1:])))
                    if left and right and len(right & rc) >= max(1, len(rc) // 2) and not left & rc:
                        aliases.add(frozenset(left))
            for j in js:
                tn, ta, _ = text.get(tgts[j], ('', '', ''))
                tc = set(core(tn))
                if aliases and tc and not tc & rc and len(AKA.split(tn)) == 1 and any(tc <= al or al <= tc for al in aliases):
                    k = (cty, 'bare_alias_sibling')
                    A[k + ('n',)] += 1; A[k + ('true',)] += int(y[j]); A[k + ('t6_acc',)] += int(chosen[j])
                    A[k + ('t6_missed',)] += int(y[j] and not chosen[j]); A[k + ('t6_false',)] += int(chosen[j] and not y[j])
                    if len(ex['alias']) < 12:
                        ex['alias'].append(f'{int(y[j])} {P[j]:.2f} {rn} => {tn} | {ta}')
                if not ta.strip():
                    continue
                h1, h2 = house(ra)[0], house(ta)[0]
                if not h1 or not h2 or h1 == h2:
                    continue
                nr = num_rel(h1, h2)
                nm = 'same' if rc == tc else ('filler' if (tc - rc) <= EN_FILLER and (rc - tc) <= EN_FILLER else ('none' if not rc & tc else 'edit'))
                sw = street_words(ra)
                st = 'street' if sw and sw <= tokens_addr(ta) else 'other'
                k = (cty, nm, st, nr)
                D[k + ('n',)] += 1; D[k + ('true',)] += int(y[j]); D[k + ('missed',)] += int(y[j] and not chosen[j])
                D[k + ('false_acc',)] += int(chosen[j] and not y[j])
        def table(C):
            keys = sorted({k[:-1] for k in C})
            return {str(k): {m: C[k + (m,)] for m in ('n', 'true', 't6_acc', 't6_missed', 't6_false', 'missed', 'false_acc') if k + (m,) in C}
                    for k in keys}
        R = {'alias': table(A), 'digit': {k: v for k, v in table(D).items() if v['n'] >= 30}, 'examples': ex}
        out = json.dumps(R, indent=1, ensure_ascii=False)
        open(os.environ['OUT'], 'w', encoding='utf-8').write(out)
        print(out, flush=True)


    if __name__ == '__main__':
        main()



def cmd_usin_read():
    """Every non-blank US/India holdout error of T6, for reading one by one, plus the blocking misses.

For each holdout S1 with F0.5 < 1: its false links (with the S1 that truly owns the record, if any), its missed links
inside the candidates, and its true records outside the candidate list (blocking), each with p, name/address relation
and the S1's accepted records. Blank-address records are skipped. Writes a text file sorted by the S1's F0.5 loss,
and counts of blocking misses by relation. Env OUT, PROB, RULE_T.
"""
    import gzip
    import json
    import os
    import sys
    from collections import Counter, defaultdict
    from pathlib import Path

    import numpy as np

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    sys.path.insert(0, str(Path(__file__).parent))
    from distractor_check import decide, f05, partition_of
    from usin_errors import rel

    H = Path('/home/scai/mtech/aib262015/scratch')
    V = Path('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts')


    def main():
        S = np.load(H / 'amazon_ml_2026/artifacts/fusion/stage1.npz')
        y = np.load(H / 'vishal_v4_artifacts/pairs/y.npy')[S['rows']]
        gold = {k: set(v) for k, v in json.loads((H / 'vishal_v4_artifacts/pairs/gold.json').read_text()).items()}
        owner = {t: s for s, ts in gold.items() for t in ts}
        cp = partition_of(gold)
        mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
        refs, tgts = [], []
        with gzip.open(H / 'vishal_v4_artifacts/pairs/pairs.tsv.gz', 'rt') as f:
            for i, line in enumerate(f):
                if mask[i]:
                    r, t = line.rstrip('\n').split('\t'); refs.append(r); tgts.append(t)
        refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
        need = set(refs) | set(tgts) | {t for s in gold if cp[s] == 'holdout' for t in gold[s]}
        text = {}
        for k in (1, 2, 3):
            with open(H / f'amazon_ml_2026/data/dataset/train/train_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if q[ie] in need:
                        text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
        P = np.load(os.environ.get('PROB', str(V / 'fusion/boundary_models_t6/stage3b_prob.npy')))
        thr = float(os.environ.get('RULE_T', '0.74'))
        ix = np.flatnonzero(np.array([cp[r] for r in refs], dtype=object) == 'holdout')
        prefs = sorted(s for s in gold if cp[s] == 'holdout')
        pos = {s: i for i, s in enumerate(prefs)}
        ch = decide(np.array([pos[r] for r in refs[ix]]), P[ix], 'expected', thr, len(prefs))
        pred, cand, pj = defaultdict(set), defaultdict(set), {}
        for j, c in zip(ix, ch):
            cand[refs[j]].add(tgts[j]); pj[(refs[j], tgts[j])] = j
            if c:
                pred[refs[j]].add(tgts[j])
        blocks, lines = Counter(), []
        fmt = lambda t: f"{text.get(t, ('?', ''))[0]} | {text.get(t, ('', ''))[1]}"
        for s in sorted(prefs, key=lambda s: f05(gold[s], pred.get(s, set()))):
            f = f05(gold[s], pred.get(s, set()))
            if f == 1:
                break
            rn, ra, cty = text.get(s, ('', '', ''))
            items = []
            for t in sorted(pred.get(s, set()) - gold[s]):
                if text.get(t, ('', ''))[1].strip():
                    o = owner.get(t)
                    items.append(f"  FP {P[pj[(s, t)]]:.2f} {rel(rn, ra, *text.get(t, ('', ''))[:2])} {fmt(t)}   <true owner: {fmt(o) if o else 'none (distractor)'}>")
            for t in sorted(gold[s] - pred.get(s, set())):
                if not text.get(t, ('', ''))[1].strip():
                    continue
                r2 = rel(rn, ra, *text.get(t, ('', ''))[:2])
                if t in cand[s]:
                    items.append(f"  FN {P[pj[(s, t)]]:.2f} {r2} {fmt(t)}")
                else:
                    items.append(f"  BLOCK {r2} {fmt(t)}"); blocks[(cty,) + r2] += 1
            if items:
                acc = '; '.join(fmt(t) for t in sorted(pred.get(s, set()) & gold[s])[:4])
                lines.append(f"\n[{cty} F={f:.2f}] {rn} | {ra}\n    accepted ok: {acc}\n" + '\n'.join(items))
        open(os.environ['OUT'], 'w', encoding='utf-8').write(''.join(lines))
        print('S1 blocks written', len(lines), flush=True)
        print(json.dumps(sorted(((str(k), v) for k, v in blocks.items()), key=lambda x: -x[1])[:25], indent=0), flush=True)


    if __name__ == '__main__':
        main()



COMMANDS = {'alias_base': cmd_alias_base, 'alias_check_france_audit': cmd_alias_check_france_audit, 'alias_check_typo_street': cmd_alias_check_typo_street, 'alias_cond': cmd_alias_cond, 'alias_link': cmd_alias_link, 'alias_rate': cmd_alias_rate, 'alias_test': cmd_alias_test, 'alias_union_test': cmd_alias_union_test, 'blank_eda': cmd_blank_eda, 'blank_expert': cmd_blank_expert, 'blind_sample': cmd_blind_sample, 'empty_guard': cmd_empty_guard, 'error_sweep_report': cmd_error_sweep_report, 'excl_empty': cmd_excl_empty, 'nocand_usin': cmd_nocand_usin, 'pattern_cond': cmd_pattern_cond, 'rowsweep': cmd_rowsweep, 'u5_check': cmd_u5_check, 'usin_patterns': cmd_usin_patterns, 'usin_read': cmd_usin_read}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('commands:', ', '.join(COMMANDS))
        sys.exit(0 if len(sys.argv) < 2 else 1)
    name = sys.argv.pop(1)
    sys.argv[0] = f'{sys.argv[0]} {name}'
    COMMANDS[name]()
