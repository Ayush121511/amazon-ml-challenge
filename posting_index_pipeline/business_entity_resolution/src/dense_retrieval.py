"""Dense (embedding) retrieval on the GPU with a fine-tuned multilingual-e5-small.

Stages (run in order by padum/dense_chain.pbs):
  finetune  Contrastive fine-tuning (in-batch negatives, batches within one country) on train
            (reference, matched target) pairs. References used by the matcher's train/dev/holdout
            sample and earlier evaluation samples are EXCLUDED, so no evaluation leakage.
  embed     fp16 L2-normalised vectors for Source 1 and for each country's targets. Target
            vectors follow the posting index's ordinal order (Source 2 then Source 3 rows of
            that country), so they line up with TargetTable / index ordinals.
  search    Exact matrix-multiply top-k per country on the GPU, both directions:
            forward (reference -> top targets) and reverse (target -> top references).

Text is the original name and address, so any script (Latin, Devanagari, Tamil, French
accents) goes to the multilingual model as-is. Label use: only `finetune` reads labels.
"""
import argparse
import csv
import gzip
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from posting_index import country_dir, records

MAX_TOKENS = 64


def text(row):
    return f"query: {row['business_name']} | {row['business_address']}"


def load_model(path, device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).to(device)
    return tokenizer, model


def encode_batch(model, tokenizer, texts, device):
    import torch
    batch = tokenizer(texts, max_length=MAX_TOKENS, truncation=True, padding=True, return_tensors='pt').to(device)
    out = model(**batch).last_hidden_state
    mask = batch['attention_mask'].unsqueeze(-1).to(out.dtype)
    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
    return torch.nn.functional.normalize(pooled, dim=-1)


# ------------------------------------------------------------------ finetune ----

def stage_finetune(a):
    import torch
    started = time.monotonic()
    excluded = set()
    for path in a.exclude_ids:
        excluded.update(Path(path).read_text().split())
    gold = {}
    with open(a.gold, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            ids = [i for i in row['matched_entity_ids'].split(',') if i]
            if ids and row['source1_entity_id'] not in excluded:
                gold[row['source1_entity_id']] = ids
    rng = random.Random('dense-finetune')
    chosen = rng.sample(sorted(gold), min(a.refs, len(gold)))
    wanted_targets = {t for s in chosen for t in gold[s]}
    ref_text, ref_country, target_text = {}, {}, {}
    chosen_set = set(chosen)
    for row in records(a.data / 'train_source1.tsv.gz'):
        if row['entity_id'] in chosen_set:
            ref_text[row['entity_id']] = text(row)
            ref_country[row['entity_id']] = row['country']
    for source in (2, 3):
        for row in records(a.data / f'train_source{source}.tsv.gz'):
            if row['entity_id'] in wanted_targets:
                target_text[row['entity_id']] = text(row)
    pairs_by_country = defaultdict(list)
    for s in chosen:
        for t in gold[s]:
            pairs_by_country[ref_country[s]].append((ref_text[s], target_text[t]))
    batches = []
    for country, pairs in pairs_by_country.items():
        rng.shuffle(pairs)
        batches += [pairs[i:i + a.batch] for i in range(0, len(pairs) - a.batch + 1, a.batch)]
    rng.shuffle(batches)
    batches = batches[:a.max_steps] if a.max_steps else batches
    print(f'Fine-tune: {len(chosen):,} references, {sum(map(len, pairs_by_country.values())):,} pairs, '
          f'{len(batches):,} steps; prepared in {time.monotonic() - started:.0f}s', flush=True)
    device = 'cuda'
    tokenizer, model = load_model(a.model, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warmup = max(1, len(batches) // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) * max(0.0, 1 - step / len(batches)))
    scaler = torch.cuda.amp.GradScaler()
    labels = torch.arange(a.batch, device=device)
    for step, batch in enumerate(batches):
        left, right = zip(*batch)
        with torch.autocast('cuda', dtype=torch.float16):
            q = encode_batch(model, tokenizer, list(left), device)
            d = encode_batch(model, tokenizer, list(right), device)
            logits = (q @ d.T).float() / a.temperature
            # Symmetric InfoNCE: every other target in the batch is a negative (same country).
            loss = (torch.nn.functional.cross_entropy(logits, labels)
                    + torch.nn.functional.cross_entropy(logits.T, labels)) / 2
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        schedule.step()
        if step % 200 == 0:
            print(f'step {step}/{len(batches)} loss {loss.item():.4f} {time.monotonic() - started:.0f}s', flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.output / 'model')
    tokenizer.save_pretrained(a.output / 'model')
    (a.output / 'finetune_report.json').write_text(json.dumps(
        {'references': len(chosen), 'steps': len(batches), 'batch': a.batch, 'lr': a.lr,
         'temperature': a.temperature, 'excluded_references': len(excluded),
         'seconds': time.monotonic() - started}, indent=2))


# --------------------------------------------------------------------- embed ----

def embed_texts(model, tokenizer, texts, batch, device):
    import torch
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    order = np.argsort([len(t) for t in texts], kind='stable')  # similar lengths -> less padding
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            vecs = encode_batch(model, tokenizer, [texts[i] for i in idx], device)
            out[idx] = vecs.to(torch.float16).cpu().numpy()
    return out


def stage_embed(a):
    started = time.monotonic()
    device = 'cuda'
    tokenizer, model = load_model(a.model, device)
    model.eval().half()
    out = a.output / a.split
    out.mkdir(parents=True, exist_ok=True)
    rows = list(records(a.data / f'{a.split}_source1.tsv.gz'))
    np.save(out / 's1_vectors.npy', embed_texts(model, tokenizer, [text(r) for r in rows], a.batch, device))
    np.save(out / 's1_ids.npy', np.array([r['entity_id'] for r in rows], dtype='S'))
    np.save(out / 's1_country.npy', np.array([r['country'] for r in rows], dtype='S'))
    print(f'{a.split} Source 1: {len(rows):,} in {time.monotonic() - started:.0f}s', flush=True)
    del rows
    by_country = defaultdict(list)
    for source in (2, 3):  # Same order as posting_index ordinals: Source 2 then Source 3, per country.
        for row in records(a.data / f'{a.split}_source{source}.tsv.gz'):
            by_country[row['country']].append((row['entity_id'], text(row)))
    for country, items in sorted(by_country.items()):
        vecs = embed_texts(model, tokenizer, [t for _, t in items], a.batch, device)
        np.save(out / f'targets_{country_dir(country)}.npy', vecs)
        np.save(out / f'targets_{country_dir(country)}_ids.npy', np.array([i for i, _ in items], dtype='S'))
        print(f'{a.split} {country}: {len(items):,} targets, {time.monotonic() - started:.0f}s', flush=True)


# -------------------------------------------------------------------- search ----

def topk_chunks(queries, keys, k, budget_bytes=3 << 30):
    """Exact cosine top-k of every query row against all key rows, on the GPU."""
    import torch
    keys_gpu = torch.from_numpy(keys).cuda()
    k = min(k, keys.shape[0])
    chunk = max(64, int(budget_bytes // (2 * max(1, keys.shape[0]))))
    scores = np.zeros((len(queries), k), dtype=np.float16)
    index = np.zeros((len(queries), k), dtype=np.int32)
    for start in range(0, len(queries), chunk):
        q = torch.from_numpy(queries[start:start + chunk]).cuda()
        s, i = (q @ keys_gpu.T).topk(k, dim=1)
        scores[start:start + len(q)] = s.cpu().numpy()
        index[start:start + len(q)] = i.cpu().numpy()
    del keys_gpu
    torch.cuda.empty_cache()
    return scores, index


def stage_search(a):
    started = time.monotonic()
    base = a.output / a.split
    s1 = np.load(base / 's1_vectors.npy')
    s1_ids = np.load(base / 's1_ids.npy')
    s1_country = np.load(base / 's1_country.npy')
    s1_dir = np.array([country_dir(c.decode()) for c in s1_country], dtype=object)
    report = {}
    for path in sorted(base.glob('targets_*_ids.npy')):
        name = path.name[len('targets_'):-len('_ids.npy')]
        targets = np.load(base / f'targets_{name}.npy')
        rows = np.flatnonzero(s1_dir == name)  # Same-country references only.
        if not len(rows):
            report[name] = {'references': 0}
            continue
        fs, fi = topk_chunks(s1[rows], targets, a.forward_k)
        rs, ri = topk_chunks(targets, s1[rows], a.reverse_k)
        np.savez(base / f'dense_{name}.npz', ref_ids=s1_ids[rows], fwd_score=fs, fwd_ord=fi,
                 rev_score=rs, rev_ref=ri)
        report[name] = {'references': int(len(rows)), 'targets': int(len(targets)),
                        'seconds': time.monotonic() - started}
        print(f'{a.split} {name}: {report[name]}', flush=True)
    (base / 'search_report.json').write_text(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['finetune', 'embed', 'search'])
    p.add_argument('--data', type=Path, required=True, help='Preprocessed v2 split directory')
    p.add_argument('--model', type=Path, help='Base (finetune) or fine-tuned (embed) model directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'test'], default='train')
    p.add_argument('--gold', type=Path)
    p.add_argument('--exclude-ids', type=Path, action='append', default=[])
    p.add_argument('--refs', type=int, default=400000, help='Fine-tuning references')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--temperature', type=float, default=0.05)
    p.add_argument('--forward-k', type=int, default=50)
    p.add_argument('--reverse-k', type=int, default=10)
    a = p.parse_args()
    {'finetune': stage_finetune, 'embed': stage_embed, 'search': stage_search}[a.stage](a)


if __name__ == '__main__':
    main()
