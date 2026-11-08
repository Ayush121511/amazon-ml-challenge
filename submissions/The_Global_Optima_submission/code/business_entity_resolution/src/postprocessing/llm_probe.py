"""LLM judge probe: does a zero-shot instruct LLM (Qwen2.5-7B-Instruct, Apache-2.0, 7.6B) separate true from false
links among the pairs stage-3b is unsure about, beyond stage-3b's own probability?

MODE=score: yes/no next-token logit (logit(Yes) - logit(No)) for the pairs in PAIRS (npz with s1, t arrays) using the
source texts of SPLIT; shard SHARD of NSHARD (one process per GPU); writes OUT_<shard>.npy.
MODE=pairs_train: writes the uncertain band (0.02 < p < 0.98) of the T6 dev + holdout lists (error_sweep.py frames).
MODE=eval: fits z = a*logit(p) + b*llm + c (+ blank interaction) on dev band pairs, applies it to holdout band pairs,
re-decides every S1 with expected@RULE_T and reports F0.5 change on dev and holdout plus in-band AUCs.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

H = Path('/home/scai/mtech/aib262015/scratch/amazon_ml_2026')
MODEL = os.environ.get('MODEL', '/home/scai/mtech/aib262015/scratch/models/qwen2.5-7b-instruct')
SYS = 'You are an expert in business entity resolution. Records come from noisy business registries.'
Q = ('Do these two records refer to the same business entity (the same business at the same location)? '
     'Names may contain typos, abbreviations, reordered words, legal-form changes or transliteration; '
     'addresses may be abbreviated, reordered, partially missing or blank.\n'
     'Record A: {n1} | {a1} | {c1}\nRecord B: {n2} | {a2}\nAnswer with Yes or No.')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def read_text(split, need):
    text = {}
    for k in (1, 2, 3):
        with open(H / f'data/dataset/{split}/{split}_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if q[ie] in need:
                    text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
    return text


def score():
    """TASKS='pairs|split|out;...' (defaults to PAIRS|SPLIT|OUT): one model load per process, several pair sets."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tasks = [t.split('|') for t in os.environ.get('TASKS', f"{os.environ.get('PAIRS')}|{os.environ.get('SPLIT')}|{os.environ.get('OUT')}").split(';') if t]
    sh, ns = int(os.environ.get('SHARD', 0)), int(os.environ.get('NSHARD', 1))
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ('Yes', ' Yes')]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ('No', ' No')]
    bs = int(os.environ.get('BS', 96))
    for pairs, split, outp in tasks:
        Z = np.load(pairs, allow_pickle=True)
        s1, tg = Z['s1'].astype(str), Z['t'].astype(str)
        ix = np.arange(sh, len(s1), ns)
        text = read_text(split, set(s1[ix]) | set(tg[ix]))
        prompts = []
        for j in ix:
            n1, a1, c1 = text.get(s1[j], ('', '', '')); n2, a2, _ = text.get(tg[j], ('', '', ''))
            msg = [{'role': 'system', 'content': SYS},
                   {'role': 'user', 'content': Q.format(n1=n1, a1=a1 or '(blank)', c1=c1, n2=n2, a2=a2 or '(blank)')}]
            prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        order = np.argsort([len(q) for q in prompts])
        out = np.zeros(len(ix), dtype=np.float32); t0 = time.time()
        with torch.inference_mode():
            for b in range(0, len(order), bs):
                o = order[b:b + bs]
                enc = tok([prompts[i] for i in o], return_tensors='pt', padding=True).to('cuda')
                lg = model(**enc, num_logits_to_keep=1).logits[:, -1, :].float()
                out[o] = (torch.logsumexp(lg[:, yes], 1) - torch.logsumexp(lg[:, no], 1)).cpu().numpy()
                if b // bs % 50 == 0:
                    log(f'shard {sh} {Path(pairs).name}: {b + len(o)}/{len(order)} pairs, {(b + len(o)) / (time.time() - t0):.1f}/s')
        np.save(f'{outp}_{sh}.npy', np.column_stack([ix, out]))
        log('shard', sh, Path(pairs).name, 'done', len(ix))


def pairs_train():
    import pandas as pd
    A = os.environ['A']
    df = pd.read_pickle(f'{A}/train.json.train_pairs.pkl')
    band = df[(df.p > 0.02) & (df.p < 0.98)]
    np.savez(os.environ['PAIRS'], s1=band.s1.values.astype(str), t=band.t.values.astype(str))
    log('band pairs', len(band), band.groupby('part').size().to_dict())


def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def fit_logit(X, y, w=None, it=50):
    w = np.ones(len(y)) if w is None else w
    b = np.zeros(X.shape[1])
    for _ in range(it):
        q = 1 / (1 + np.exp(-(X @ b)))
        g = X.T @ (w * (q - y)) + 1e-3 * b
        Hs = (X * (w * q * (1 - q))[:, None]).T @ X + 1e-3 * np.eye(X.shape[1])
        b -= np.linalg.solve(Hs, g)
    return b


def ev():
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent))
    os.environ.setdefault('SUB_DIR', '.'); os.environ.setdefault('OUT_DIR', '.')
    from distractor_check import decide
    A = os.environ['A']
    df = pd.read_pickle(f'{A}/train.json.train_pairs.pkl').reset_index(drop=True)
    bf = pd.read_pickle(f'{A}/train.json.train_blocking.pkl')
    band = df[(df.p > 0.02) & (df.p < 0.98)]
    llm = np.full(len(band), np.nan)
    for f in sorted(Path(os.environ['OUT']).parent.glob(Path(os.environ['OUT']).name + '_*.npy')):
        a = np.load(f); llm[a[:, 0].astype(int)] = a[:, 1]
    assert not np.isnan(llm).any(), 'missing LLM scores'
    band = band.assign(llm=llm)
    lp = np.log(band.p / (1 - band.p)).values
    bl = (band.blank == 'blank').values.astype(float)
    X = np.column_stack([lp, band.llm.values, bl, bl * band.llm.values, np.ones(len(band))])
    y = band.y.values.astype(float)
    dv = (band.part == 'dev').values
    R = {'n_band': {p: int((band.part == p).sum()) for p in ('dev', 'holdout')}}
    for p in ('dev', 'holdout'):
        m = (band.part == p).values
        R[f'auc_{p}'] = {'t6': auc(y[m], lp[m]), 'llm': auc(y[m], band.llm.values[m])}
        for k in ('blank', 'nonblank'):
            mk = m & ((band.blank == k).values)
            R[f'auc_{p}_{k}'] = {'t6': auc(y[mk], lp[mk]), 'llm': auc(y[mk], band.llm.values[mk])}
    b = fit_logit(X[dv], y[dv])
    R['coef'] = dict(zip(['logit_p', 'llm', 'blank', 'blank_x_llm', 'const'], b.round(4).tolist()))
    pn = 1 / (1 + np.exp(-(X @ b)))
    ll = lambda q, yy: float(-np.mean(yy * np.log(np.clip(q, 1e-9, 1)) + (1 - yy) * np.log(np.clip(1 - q, 1e-9, 1))))
    ho = ~dv
    R['holdout_logloss'] = {'t6': ll(band.p.values[ho], y[ho]), 'combined': ll(pn[ho], y[ho])}
    R['auc_holdout_combined'] = auc(y[ho], pn[ho])
    miss = bf.groupby('s1').size()
    P2 = df.p.values.copy(); P2[band.index.values] = pn
    thr = float(os.environ.get('RULE_T', '0.74'))
    for p in ('dev', 'holdout'):
        d = df[df.part == p]
        refs = sorted(d.s1.unique()); pos = {s: i for i, s in enumerate(refs)}
        ri = np.array([pos[s] for s in d.s1])
        gold_n = d.groupby('s1').y.sum().reindex(refs).values + miss.reindex(refs).fillna(0).values
        res = {}
        for name, pp in (('t6', d.p.values), ('combined', P2[d.index.values])):
            ch = decide(ri, pp, 'expected', thr, len(refs))
            tp = np.bincount(ri, weights=ch * d.y.values, minlength=len(refs))
            npred = np.bincount(ri, weights=ch.astype(float), minlength=len(refs))
            pr = np.where(npred > 0, tp / np.maximum(npred, 1), 0); rc = np.where(gold_n > 0, tp / np.maximum(gold_n, 1), 0)
            f = np.where(tp > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
            f = np.where((npred == 0) & (gold_n == 0), 1.0, f)
            res[name] = float(f.mean())
        R[f'F_{p}'] = {**{k: round(v, 6) for k, v in res.items()}, 'delta': round(res['combined'] - res['t6'], 6)}
    print(json.dumps(R, indent=1), flush=True)
    json.dump(R, open(os.environ['OUT'] + '_eval.json', 'w'), indent=1)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    {'score': score, 'pairs_train': pairs_train, 'eval': ev}[os.environ['MODE']]()
