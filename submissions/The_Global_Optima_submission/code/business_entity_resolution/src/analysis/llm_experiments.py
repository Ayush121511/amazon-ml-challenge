"""LLM judge experiments.

Each command is one analysis script. Run `python llm_experiments.py <command> [args]`; `python llm_experiments.py` lists the commands.
"""
import sys


def cmd_delta_merge():
    """Apply the LLM-judge changes to the best file as a delta, keeping every other rule's links.

US/India delta = (LLM-blended decisions) - (T6 bag decisions reproduced by the same finalize), from llm_apply.py's
OUT_DIR/decided and OUT_DIR/repro_check. France delta = the ADD/DROP list written by llm_france.py (optional).
Adds only when the record is unowned; removes only when the S1 keeps another link. Env BASE_DIR, USIN_DIR, FR_DIR,
OUT_DIR.
"""
    import os, shutil
    from collections import Counter
    from pathlib import Path


    def load_rows(f):
        rows, d = [], {}
        for i, l in enumerate(open(f, encoding='utf-8')):
            if i:
                s, m = (l.rstrip('\n').split('\t') + [''])[:2]; rows.append(s); d[s] = [x for x in m.split(',') if x]
        return rows, d


    base = Path(os.environ['BASE_DIR'])
    order, u = load_rows(base / 'matching_results.tsv')
    owner = {t: s for s, ts in u.items() for t in ts}
    R = Counter(); adds, drops = [], []
    if os.environ.get('USIN_DIR'):
        U = Path(os.environ['USIN_DIR'])
        _, rep = load_rows(U / 'repro_check/matching_results.tsv'); _, dec = load_rows(U / 'decided/matching_results.tsv')
        cty = {}
        for i, l in enumerate(open('/home/scai/mtech/aib262015/scratch/amazon_ml_2026/data/dataset/test/test_source1.tsv', encoding='utf-8')):
            if i: q = l.rstrip('\n').split('\t'); cty[q[0]] = q[3]
        for s in rep:
            if cty.get(s) == 'France': continue
            a, b = set(dec[s]), set(rep[s])
            adds += [(s, t, 'usin') for t in a - b]; drops += [(s, t, 'usin') for t in b - a]
    if os.environ.get('FR_DIR'):
        F = Path(os.environ['FR_DIR'])
        _, fr_new = load_rows(F / 'matching_results.tsv'); _, fr_old = load_rows(Path(os.environ['FR_BASE']) / 'matching_results.tsv')
        for s in fr_new:
            a, b = set(fr_new[s]), set(fr_old[s])
            adds += [(s, t, 'fr') for t in a - b]; drops += [(s, t, 'fr') for t in b - a]
    for s, t, tag in drops:
        if t in u[s] and len(u[s]) >= 2:
            u[s].remove(t); owner.pop(t, None); R[tag + '_drop'] += 1
        else:
            R[tag + '_drop_skipped'] += 1
    for s, t, tag in adds:
        if t not in owner:
            u[s].append(t); owner[t] = s; R[tag + '_add'] += 1
        else:
            R[tag + '_add_skipped'] += 1
    out = Path(os.environ['OUT_DIR']); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'matching_results.tsv', 'w', encoding='utf-8') as fo:
        fo.write('source1_entity_id\tmatched_entity_ids\n')
        for s in order:
            fo.write(s + '\t' + ','.join(u[s]) + '\n')
    shutil.copy(base / 'candidate_pairs.tsv', out / 'candidate_pairs.tsv')
    print(dict(R), 'records in >1 row', sum(v > 1 for v in Counter(t for ts in u.values() for t in ts).values()), flush=True)



def cmd_llm_apply():
    """Test path of the LLM judge (only run if llm_probe.py MODE=eval gains on dev and holdout).

MODE=pairs_test: US/India pairs of the T6-bag test lists with 0.02 < p < 0.98 -> PAIRS npz (s1, t, row).
MODE=apply: replaces those pairs' p with the dev-fitted combination sigma(a*logit(p) + b*llm + c*blank + d*blank*llm + e),
re-decides with the bag's rule + global exclusivity (the pipeline's finalize; first checked to reproduce the bag's own
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



def cmd_llm_ctx():
    """LLM judge v2: the same context stage-3b has + the data's rules + worked examples (US/India band, labelled).

v1 (llm_probe.py) saw one pair alone and scored AUC 0.54 in the band. v2 prompt: data facts (how copies and decoys are
made, that several reference businesses share names), 6 worked examples from dev (one per pattern: digit-drop copy,
shifted-number decoy, invented alias true / false, blank copy of a unique name, blank record of a shared name), then
the question with the reference's other candidates and how many other references share its exact name.
Writes OUT_<shard>.npy in the band order of llm_probe.py, so MODE=eval there scores it. Env A, OUT, SHARD, NSHARD.
"""
    import os
    import re
    import sys
    import time
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import pandas as pd

    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    sys.path.insert(0, str(Path(__file__).parent))
    from france_census import fold

    H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
    MODEL = os.environ.get('MODEL', '/home/scai/mtech/aib262015/scratch/models/qwen2.5-7b-instruct')
    nrm = lambda s: re.sub(r'[^a-z0-9]+', ' ', fold(s or '')).strip()
    SYS = ('You judge whether a candidate record is a copy of a reference business in a synthetic entity-resolution '
           'dataset. Facts about this data:\n'
           '- Each reference business usually has 2-6 copies spread over two sources. Copies carry noise: typos, reordered '
           'words, accents, legal-form changes (Inc/LLC/Corp/Pvt Ltd), filler words (Services, Center, Partners, The), '
           'native-script or transliterated names, domain or handle names (name.com, #name), initials, and occasionally an '
           'invented alias name (e.g. "Zephcira") at the reference address. Addresses may be reformatted, abbreviated, '
           'missing parts, blank, or have a dropped digit in the house number (2118 -> 118).\n'
           '- Decoys are different businesses built to look similar: the same or similar name at a house number shifted by '
           'a small amount on the same street, often with a legal-form change or an added word (Holdings, Group, '
           'International, Development, Uptown, East); a swapped business-type word; or another business sharing the '
           'address.\n'
           '- Several different reference businesses can have exactly the same name; a record with a blank address then '
           'belongs to at most one of them.\n'
           'Answer Yes if the candidate is a copy of the reference, otherwise No.')


    def rec_line(t, text):
        n, a, _ = text.get(t, ('', '', ''))
        return f'[{t[:2]}] {n} | {a or "(blank)"}'


    def question(s, t, others, twins, text):
        n, a, c = text[s]
        ctx = '\n'.join('  - ' + rec_line(o, text) for o in others[:9]) or '  (none)'
        return (f'Reference: {n} | {a} | {c}\nOther references with exactly this name: {twins}\n'
                f'Other candidate records of this reference:\n{ctx}\nCandidate: {rec_line(t, text)}\n'
                'Is the candidate a copy of the reference?')


    def main():
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        df = pd.read_pickle(f"{os.environ['A']}/train.json.train_pairs.pkl")
        band = df[(df.p > 0.02) & (df.p < 0.98)]
        dev = df[df.part == 'dev']
        shots_spec = [((dev['shift'] == 'digit_drop') & (dev.y == 1)), ((dev['shift'] == 'decoy_shift') & (dev.y == 0)),
                      ((dev['name'] == 'none') & (dev.blank == 'nonblank') & (dev.y == 1)),
                      ((dev['name'] == 'none') & (dev.blank == 'nonblank') & (dev.y == 0)),
                      ((dev.blank == 'blank') & (dev.mult == '1') & (dev.y == 1)), ((dev.blank == 'blank') & (dev.mult == '6+') & (dev.y == 0))]
        shots = [dev[m].sample(1, random_state=11).iloc[0] for m in shots_spec]
        sh, ns = int(os.environ.get('SHARD', 0)), int(os.environ.get('NSHARD', 1))
        ix = np.arange(sh, len(band), ns)
        need_s1 = set(band.s1.values[ix]) | {r.s1 for r in shots}
        sib = df[df.s1.isin(need_s1)].groupby('s1').t.apply(list).to_dict()
        need = need_s1 | {t for s in need_s1 for t in sib.get(s, [])}
        text, names = {}, Counter()
        for k in (1, 2, 3):
            with open(H / f'data/dataset/train/train_source{k}.tsv', encoding='utf-8') as fh:
                head = fh.readline().rstrip('\n').split('\t')
                ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
                for line in fh:
                    q = line.rstrip('\n').split('\t')
                    if k == 1:
                        names[(q[c], nrm(q[n]))] += 1
                    if q[ie] in need:
                        text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
        tw = lambda s: names[(text[s][2], nrm(text[s][0]))] - 1
        tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
        msgs = [{'role': 'system', 'content': SYS}]
        for r in shots:
            msgs += [{'role': 'user', 'content': question(r.s1, r.t, [o for o in sib[r.s1] if o != r.t], tw(r.s1), text)},
                     {'role': 'assistant', 'content': 'Yes' if r.y else 'No'}]
        s1v, tv = band.s1.values, band.t.values
        prompts = [tok.apply_chat_template(msgs + [{'role': 'user', 'content': question(s1v[j], tv[j], [o for o in sib[s1v[j]] if o != tv[j]], tw(s1v[j]), text)}],
                                           tokenize=False, add_generation_prompt=True) for j in ix]
        yes = [tok.encode(w, add_special_tokens=False)[0] for w in ('Yes', ' Yes')]
        no = [tok.encode(w, add_special_tokens=False)[0] for w in ('No', ' No')]
        order = np.argsort([len(p) for p in prompts]); out = np.zeros(len(ix), dtype=np.float32)
        bs = int(os.environ.get('BS', 24)); t0 = time.time()
        with torch.inference_mode():
            for b in range(0, len(order), bs):
                o = order[b:b + bs]
                enc = tok([prompts[i] for i in o], return_tensors='pt', padding=True).to('cuda')
                lg = model(**enc, num_logits_to_keep=1).logits[:, -1, :].float()
                out[o] = (torch.logsumexp(lg[:, yes], 1) - torch.logsumexp(lg[:, no], 1)).cpu().numpy()
                if b // bs % 100 == 0:
                    print(time.strftime('%H:%M:%S'), f'shard {sh}: {b + len(o)}/{len(order)}, {(b + len(o)) / (time.time() - t0):.1f}/s', flush=True)
        np.save(f"{os.environ['OUT']}_{sh}.npy", np.column_stack([ix, out]))
        print('done', sh, len(ix), 'prompt tokens ~', len(tok(prompts[order[-1]]).input_ids), flush=True)


    if __name__ == '__main__':
        main()



def cmd_llm_lexicon():
    """Qwen2.5-7B-Instruct labels single French words from business names: A business type/activity, B generic filler,
C proper name / other, D unsure. Next-token probabilities over A-D; a label is used only if its probability >= 0.7.
Validated on the census lists (TYPES -> A, FILLER -> B) before use."""
    import json, os, sys, time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    M = '/home/scai/mtech/aib262015/scratch/models/qwen2.5-7b-instruct'
    W = json.load(open(sys.argv[1]))
    SYS = 'You classify single French words (possibly misspelled, abbreviated or without accents) that appear in names of French businesses and associations.'
    Q = ("Word: \"{w}\"\nA = a kind of business, organisation or activity (e.g. club, ecole, pharmacie, restaurant, garage, residence, "
         "association, hopital, boulangerie, sportive, amicale, comite)\nB = a generic filler that does not say what the business is "
         "(e.g. groupe, fils, freres, services, compagnie, cie, developpement, etablissements, associes, france, holding)\n"
         "C = a proper name (place, person, saint, town) or anything else\nD = unsure\nAnswer with one letter.")
    tok = AutoTokenizer.from_pretrained(M); tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16).cuda().eval()
    ids = [tok.encode(x, add_special_tokens=False)[0] for x in 'ABCD']
    allw = W['unknown'] + W['types'] + W['filler']
    out = {}
    with torch.inference_mode():
        for b in range(0, len(allw), 64):
            ws = allw[b:b + 64]
            pr = [tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': Q.format(w=w)}], tokenize=False, add_generation_prompt=True) for w in ws]
            enc = tok(pr, return_tensors='pt', padding=True).to('cuda')
            p = torch.softmax(model(**enc, num_logits_to_keep=1).logits[:, -1, ids].float(), -1).cpu().numpy()
            for w, row in zip(ws, p):
                out[w] = ['ABCD'[int(row.argmax())], float(row.max())]
    acc_t = sum(out[w][0] == 'A' for w in W['types']) / len(W['types']); acc_f = sum(out[w][0] == 'B' for w in W['filler']) / len(W['filler'])
    print('validation: TYPES labelled A', round(acc_t, 3), '| FILLER labelled B', round(acc_f, 3), flush=True)
    print('misses:', [(w, out[w]) for w in W['types'] + W['filler'] if out[w][0] != ('A' if w in W['types'] else 'B')], flush=True)
    json.dump(out, open(sys.argv[2], 'w'), indent=0)
    from collections import Counter
    print('unknown words:', Counter(out[w][0] + ('' if out[w][1] >= 0.7 else '?') for w in W['unknown']), flush=True)



COMMANDS = {'delta_merge': cmd_delta_merge, 'llm_apply': cmd_llm_apply, 'llm_ctx': cmd_llm_ctx, 'llm_lexicon': cmd_llm_lexicon}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('commands:', ', '.join(COMMANDS))
        sys.exit(0 if len(sys.argv) < 2 else 1)
    name = sys.argv.pop(1)
    sys.argv[0] = f'{sys.argv[0]} {name}'
    COMMANDS[name]()
