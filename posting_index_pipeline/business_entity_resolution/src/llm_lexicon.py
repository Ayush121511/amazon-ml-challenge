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
