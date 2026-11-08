"""LoRA-tuned LLM judge (Qwen2.5-7B-Instruct, Apache-2.0) for the pairs stage-3b is unsure about.

MODE=train: training set = US/India trainA/trainB top-10 pairs of the labelled sample, 80% from T6's uncertain band
(0.02 < p < 0.98) and 20% sure pairs; LoRA r16/alpha32 on attention projections; loss = BCE on
logit(Yes) - logit(No) at the answer position. Saves the adapter to ADAPTER, then scores the dev + holdout band of
llm_probe.py (same row order) into OUT_0.npy so `llm_probe.py MODE=eval` reports AUC and the F0.5 change.
MODE=score: loads base + adapter and scores shard SHARD of NSHARD of PAIRS (npz with s1, t) from SPLIT's sources.
"""
import gzip, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
H = Path('/home/scai/mtech/aib262015/scratch')
MODEL = str(H / 'models/qwen2.5-7b-instruct')
SYS = 'You are an expert in business entity resolution for noisy synthetic business registries.'
Q = ('Is record B a copy of business A (same business, possibly with typos, abbreviations, aliases, legal-form or filler '
     'changes, or a partial or blank address), rather than a different business?\n'
     'A: {n1} | {a1} | {c1}\nB: {n2} | {a2}\nAnswer Yes or No.')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def read_text(split, need):
    text = {}
    for k in (1, 2, 3):
        with open(H / f'amazon_ml_2026/data/dataset/{split}/{split}_source{k}.tsv', encoding='utf-8') as fh:
            head = fh.readline().rstrip('\n').split('\t')
            ie, n, a, c = (head.index(x) for x in ('entity_id', 'business_name', 'business_address', 'country'))
            for line in fh:
                q = line.rstrip('\n').split('\t')
                if q[ie] in need:
                    text[q[ie]] = (q[n], q[a] if len(q) > a else '', q[c] if len(q) > c else '')
    return text


def prompts(tok, s1, tg, text):
    out = []
    for s, t in zip(s1, tg):
        n1, a1, c1 = text.get(s, ('', '', '')); n2, a2, _ = text.get(t, ('', '', ''))
        msg = [{'role': 'system', 'content': SYS},
               {'role': 'user', 'content': Q.format(n1=n1, a1=a1 or '(blank)', c1=c1, n2=n2, a2=a2 or '(blank)')}]
        out.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
    return out


def load_model(adapter=None, train=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    from peft import LoraConfig, PeftModel, get_peft_model
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    elif train:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type='CAUSAL_LM',
                                                 target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ('Yes', ' Yes')]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ('No', ' No')]
    return tok, model, yes, no


def margin(model, enc, yes, no):
    try:
        lg = model(**enc, num_logits_to_keep=1).logits[:, -1, :].float()
    except TypeError:  # transformers < 4.45 has no num_logits_to_keep
        lg = model(**enc).logits[:, -1, :].float()
    return torch.logsumexp(lg[:, yes], 1) - torch.logsumexp(lg[:, no], 1)


def score_prompts(tok, model, yes, no, pr, bs=64, tag=''):
    model.eval(); order = np.argsort([len(p) for p in pr]); out = np.zeros(len(pr), dtype=np.float32); t0 = time.time()
    with torch.inference_mode():
        for b in range(0, len(order), bs):
            o = order[b:b + bs]
            enc = tok([pr[i] for i in o], return_tensors='pt', padding=True).to('cuda')
            out[o] = margin(model, enc, yes, no).cpu().numpy()
            if b // bs % 100 == 0:
                log(f'{tag} {b + len(o)}/{len(order)} {(b + len(o)) / (time.time() - t0):.1f}/s')
    return out


def train():
    from distractor_check import partition_of
    V4 = H / 'vishal_v4_artifacts/pairs'
    gold = {k: set(v) for k, v in json.loads((V4 / 'gold.json').read_text()).items()}
    cp = partition_of(gold)
    mask = np.load(H / 'amazon_ml_2026/artifacts/prune_v1_copy/mask_top10.npy')
    refs, tgts = [], []
    with gzip.open(V4 / 'pairs.tsv.gz', 'rt') as f:
        for i, line in enumerate(f):
            if mask[i]:
                r, t = line.rstrip('\n').split('\t'); refs.append(r); tgts.append(t)
    refs, tgts = np.array(refs, dtype=object), np.array(tgts, dtype=object)
    P = np.load('/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/fusion/boundary_models_t6/stage3b_prob.npy')
    y = np.array([t in gold[r] for r, t in zip(refs, tgts)], dtype=np.float32)
    part = np.array([cp[r] for r in refs], dtype=object)
    tr = np.isin(part, ['trainA', 'trainB'])
    band = np.flatnonzero(tr & (P > 0.02) & (P < 0.98)); sure = np.flatnonzero(tr & ~((P > 0.02) & (P < 0.98)))
    rng = np.random.default_rng(0)
    NB = int(os.environ.get('N_BAND', 24000))
    idx = np.concatenate([rng.choice(band, min(NB, len(band)), replace=False), rng.choice(sure, NB // 4, replace=False)])
    rng.shuffle(idx)
    log('train rows', len(idx), 'band available', len(band), 'pos rate', float(y[idx].mean()))
    text = read_text('train', set(refs[idx]) | set(tgts[idx]))
    tok, model, yes, no = load_model(train=True)
    pr = prompts(tok, refs[idx], tgts[idx], text); yy = torch.tensor(y[idx])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(os.environ.get('LR', 1e-4)), weight_decay=0.0)
    bs = int(os.environ.get('BS', 16)); steps = (len(pr) + bs - 1) // bs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / steps))
    model.train(); t0 = time.time(); lossf = torch.nn.BCEWithLogitsLoss()
    for s in range(steps):
        b = slice(s * bs, (s + 1) * bs)
        enc = tok(pr[b], return_tensors='pt', padding=True).to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            m = margin(model, enc, yes, no)
        loss = lossf(m, yy[b].cuda())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
        if s % 100 == 0:
            log(f'step {s}/{steps} loss {loss.item():.4f} {(s + 1) * bs / (time.time() - t0):.1f} ex/s')
        if time.time() - t0 > float(os.environ.get('MAX_SEC', 1800)):
            log('time budget reached at step', s); break
    A = os.environ['ADAPTER']; model.save_pretrained(A); Path(A, 'DONE').write_text('ok')
    log('adapter saved', A)
    Z = np.load(os.environ['DEV_PAIRS'], allow_pickle=True)
    s1, tg = Z['s1'].astype(str), Z['t'].astype(str)
    dtext = read_text('train', set(s1) | set(tg))
    sc = score_prompts(tok, model, yes, no, prompts(tok, s1, tg, dtext), tag='devband')
    np.save(os.environ['OUT'] + '_0.npy', np.column_stack([np.arange(len(s1)), sc]))
    log('dev/holdout band scored', len(s1))


def score():
    Z = np.load(os.environ['PAIRS'], allow_pickle=True)
    s1, tg = Z['s1'].astype(str), Z['t'].astype(str)
    sh, ns = int(os.environ['SHARD']), int(os.environ['NSHARD'])
    ix = np.arange(sh, len(s1), ns)
    text = read_text(os.environ.get('SPLIT', 'test'), set(s1[ix]) | set(tg[ix]))
    tok, model, yes, no = load_model(adapter=os.environ['ADAPTER'])
    sc = score_prompts(tok, model, yes, no, prompts(tok, s1[ix], tg[ix], text), tag=f'shard{sh}')
    np.save(f"{os.environ['OUT']}_{sh}.npy", np.column_stack([ix, sc]))
    log('shard', sh, 'done', len(ix))


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    {'train': train, 'score': score}[os.environ['MODE']]()
