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
