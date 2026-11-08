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
