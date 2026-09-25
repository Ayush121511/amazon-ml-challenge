"""End-to-end entity resolution: posting-index blocking -> pair features -> LightGBM -> TSVs.

Stages (all label-free except `train`, which reads labels only after candidates exist):
  pairs    Sample train references, generate candidates + features, then attach labels.
  train    Entity-grouped train/dev/holdout split, LightGBM, decision rule chosen on dev only.
  predict  Every reference of a split -> matching_results.tsv and candidate_pairs.tsv.

Blocking is posting_index.py (name / address / anchor routes, per country, hashed terms so
unseen countries such as France work). Each route keeps its top_k targets; the routes are
fused by reciprocal rank and the fused top `keep` targets are the candidates fed to the model.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from posting_index import CountryIndex, country_dir, load_manifest, raw_counts, route_text, weigh, ROUTES, records
from transliterate import phonetic, transliterate

TARGET_FIELDS = ('entity_id', 'name_norm', 'name_folded', 'address_folded',
                 'address_numbers', 'postal_candidates', 'address_missing')
RRF_K = 60
LEGAL = frozenset('private pvt limited ltd llc inc incorporated corp corporation co company the and '
                  'llp plc sarl sas sa eurl enterprises enterprise services service'.split())
REVERSE_FIELDS = ('rev_rank', 'rev_rrf', 'rev_score_name', 'rev_score_address', 'rev_score_anchor')


def core_name(name):
    """Name without legal-form and generic business words."""
    return ' '.join(w for w in name.split() if w not in LEGAL)


# ------------------------------------------------------------ blocking ----

class TargetTable:
    """Preprocessed target fields for one country, in the index's posting-ordinal order."""

    def __init__(self, data_dir, split, country, index_ids):
        columns = {f: [] for f in TARGET_FIELDS}
        for source in (2, 3):
            for row in records(Path(data_dir) / f'{split}_source{source}.tsv.gz'):
                if row['country'] == country:
                    for f in TARGET_FIELDS:
                        columns[f].append(row[f])
        if len(columns['entity_id']) != len(index_ids) or any(
                a != b.decode('utf-8') for a, b in zip(columns['entity_id'], index_ids)):
            raise ValueError(f'{country}: preprocessed targets do not match index ordinals')
        columns['name_core'] = [core_name(x) for x in columns['name_folded']]
        columns['name_translit'] = [transliterate(x) for x in columns['name_folded']]
        columns['address_translit'] = [transliterate(x) for x in columns['address_folded']]
        columns['name_phonetic'] = [phonetic(x) for x in columns['name_folded']]
        self.fields = {f: np.array(v, dtype=object) for f, v in columns.items()}
        self.source3 = np.array([i.startswith('S3-') for i in columns['entity_id']])


def candidates(index, rows, top_k, top_terms, keep, threads, routes=None):
    """Fused top-`keep` candidates for a batch of reference rows.

    Returns dict of equal-length arrays: ref (row position), ord (target ordinal),
    per-route score and rank (0 = route did not return the target), rrf, fused rank.
    """
    from sparse_dot_topn import sp_matmul_topn
    parts = []
    tops = {route: np.zeros(len(rows), dtype=np.float32) for route in ROUTES}
    for r, route in enumerate(ROUTES):
        if route not in index.routes or (routes and route not in routes):  # Absent or not requested.
            continue
        postings, idf, keep_terms = index.routes[route]
        queries = weigh(raw_counts(route, [route_text(route, x) for x in rows]), idf, keep_terms, top_terms)
        if queries.nnz == 0 or postings.nnz == 0:
            continue
        scores = sp_matmul_topn(queries, postings, top_n=top_k, threshold=0.0, sort=True,
                                n_threads=threads)
        counts = np.diff(scores.indptr)
        ref = np.repeat(np.arange(len(rows)), counts)
        rank = np.arange(scores.nnz) - np.repeat(scores.indptr[:-1], counts) + 1
        tops[route][ref[rank == 1]] = scores.data[rank == 1]
        parts.append((ref, scores.indices.astype(np.int64), np.full(scores.nnz, r),
                      rank, scores.data.astype(np.float32)))
    out = {k: np.zeros(0, dtype=t) for k, t in (('ref', np.int64), ('ord', np.int64), ('rrf', np.float32),
                                                  ('fused_rank', np.int32), ('route_hits', np.int8))}
    for route in ROUTES:
        out[f'score_{route}'] = np.zeros(0, dtype=np.float32)
        out[f'rank_{route}'] = np.zeros(0, dtype=np.int32)
    out['top'] = tops
    if not parts:
        return out
    ref, ord_, route, rank, score = (np.concatenate(x) for x in zip(*parts))
    key = (ref << 32) | ord_
    pairs, inverse = np.unique(key, return_inverse=True)
    n = len(pairs)
    rrf = np.zeros(n, dtype=np.float64)
    np.add.at(rrf, inverse, 1.0 / (RRF_K + rank))
    per_route = {}
    for r, name in enumerate(ROUTES):
        s, k = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.int32)
        m = route == r
        s[inverse[m]] = score[m]
        k[inverse[m]] = rank[m]
        per_route[name] = (s, k)
    pair_ref, pair_ord = pairs >> 32, pairs & 0xFFFFFFFF
    order = np.lexsort((pair_ord, -rrf, pair_ref))
    sorted_ref = pair_ref[order]
    starts = np.searchsorted(sorted_ref, sorted_ref, side='left')
    position = np.arange(n) - starts + 1
    chosen = order[position <= keep]
    out = {'ref': pair_ref[chosen], 'ord': pair_ord[chosen], 'rrf': rrf[chosen].astype(np.float32),
           'fused_rank': position[position <= keep].astype(np.int32)}
    hits = np.zeros(len(chosen), dtype=np.int8)
    for name, (s, k) in per_route.items():
        out[f'score_{name}'], out[f'rank_{name}'] = s[chosen], k[chosen]
        hits += (k[chosen] > 0)
    out['route_hits'] = hits
    out['top'] = tops
    return out


def merge_reverse(c, rev):
    """Union of forward candidates and reverse candidates (targets whose top-M references
    include this reference). Output is sorted by (ref, ord). Absent values are 0; a
    reverse-only pair has fused_rank 0, a forward-only pair has rev_rank 0."""
    key_f = (c['ref'].astype(np.int64) << 32) | c['ord'].astype(np.int64)
    key_r = (rev['ref'].astype(np.int64) << 32) | rev['t_ord'] if len(rev['ref']) else np.zeros(0, np.int64)
    keys = np.union1d(key_f, key_r)
    out = {'ref': keys >> 32, 'ord': keys & 0xFFFFFFFF, 'top': c['top']}
    fi = np.searchsorted(keys, key_f)
    for k, v in c.items():
        if k in ('ref', 'ord', 'top'):
            continue
        out[k] = np.zeros(len(keys), dtype=v.dtype)
        out[k][fi] = v
    ri = np.searchsorted(keys, key_r)
    sources = {'rev_rank': 'rank', 'rev_rrf': 'rrf', 'rev_score_name': 'score_name',
               'rev_score_address': 'score_address', 'rev_score_anchor': 'score_anchor'}
    for k, src in sources.items():
        out[k] = np.zeros(len(keys), dtype=np.float32)
        if len(key_r):
            out[k][ri] = rev[src]
    return out


class Dense:
    """Embedding retrieval results + vectors for one split and country (dense_retrieval.py)."""

    def __init__(self, root, split, country, table_ids):
        base = Path(root) / split
        name = country_dir(country)
        self.ok = (base / f'dense_{name}.npz').exists()
        if not self.ok:
            return
        ids = np.load(base / f'targets_{name}_ids.npy')
        if len(ids) != len(table_ids) or not np.array_equal(ids, np.asarray(table_ids, dtype=ids.dtype)):
            raise ValueError(f'{country}: dense target vectors do not match index ordinals')
        self.targets = np.load(base / f'targets_{name}.npy')
        d = np.load(base / f'dense_{name}.npz')
        self.ref_ids = d['ref_ids']
        self.position = {r.decode(): i for i, r in enumerate(self.ref_ids)}
        s1_ids = np.load(base / 's1_ids.npy')
        s1_rows = {r: i for i, r in enumerate(s1_ids)}
        s1 = np.load(base / 's1_vectors.npy', mmap_mode='r')
        self.refs = np.asarray(s1[[s1_rows[r] for r in self.ref_ids]])
        self.fwd_score, self.fwd_ord = d['fwd_score'], d['fwd_ord']
        rev_ref, rev_score = d['rev_ref'], d['rev_score']
        n_t, k = rev_ref.shape
        flat_ref = rev_ref.ravel()
        order = np.argsort(flat_ref, kind='stable')
        self.rev_t = np.repeat(np.arange(n_t), k)[order]
        self.rev_rank = np.tile(np.arange(1, k + 1), n_t)[order]
        self.rev_score = rev_score.ravel()[order]
        self.rev_bounds = np.searchsorted(flat_ref[order], np.arange(len(self.ref_ids) + 1))

    def lookup(self, rows):
        """Forward and reverse dense candidates, plus each row's reference vector."""
        fwd = {'ref': [], 't_ord': [], 'rank': [], 'score': []}
        rev = {'ref': [], 't_ord': [], 'rank': [], 'score': []}
        vecs = np.zeros((len(rows), self.targets.shape[1]), dtype=np.float32)
        for i, row in enumerate(rows):
            n = self.position.get(row['entity_id'])
            if n is None:
                continue
            vecs[i] = self.refs[n]
            k = self.fwd_ord.shape[1]
            fwd['ref'].append(np.full(k, i)); fwd['t_ord'].append(self.fwd_ord[n])
            fwd['rank'].append(np.arange(1, k + 1)); fwd['score'].append(self.fwd_score[n])
            lo, hi = self.rev_bounds[n], self.rev_bounds[n + 1]
            rev['ref'].append(np.full(hi - lo, i)); rev['t_ord'].append(self.rev_t[lo:hi])
            rev['rank'].append(self.rev_rank[lo:hi]); rev['score'].append(self.rev_score[lo:hi])
        cat = lambda d: {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in d.items()}
        return cat(fwd), cat(rev), vecs


def merge_source(c, src, fields):
    """Union candidates `c` with another source (ref, t_ord, ...); fields maps new name -> src key."""
    key_c = (c['ref'].astype(np.int64) << 32) | c['ord'].astype(np.int64)
    key_s = ((src['ref'].astype(np.int64) << 32) | src['t_ord'].astype(np.int64)) if len(src['ref']) \
        else np.zeros(0, np.int64)
    keys = np.union1d(key_c, key_s)
    out = {'ref': keys >> 32, 'ord': keys & 0xFFFFFFFF, 'top': c['top']}
    ci = np.searchsorted(keys, key_c)
    for k, v in c.items():
        if k in ('ref', 'ord', 'top'):
            continue
        out[k] = np.zeros(len(keys), dtype=v.dtype)
        out[k][ci] = v
    si = np.searchsorted(keys, key_s)
    for name, key in fields.items():
        out[name] = np.zeros(len(keys), dtype=np.float32)
        if len(key_s):
            out[name][si] = src[key]
    return out


DENSE_FIELDS = ('dense_rank', 'dense_score', 'dense_rev_rank', 'dense_rev_score')


def merge_dense(c, fwd, rev):
    c = merge_source(c, fwd, {'dense_rank': 'rank', 'dense_score': 'score'})
    return merge_source(c, rev, {'dense_rev_rank': 'rank', 'dense_rev_score': 'score'})


# ------------------------------------------------------------ features ----

FEATURE_NAMES = []


def _group_max(values, ref, n_refs):
    result = np.full(n_refs, -np.inf, dtype=np.float32)
    np.maximum.at(result, ref, values)
    return result[ref]


ANCHOR_FIELDS = ('name_translit', 'address_folded', 'address_numbers')


def group_features(cols, ref, n_refs, target_field, threads):
    """Candidate-vs-anchor similarity. Anchors are each reference's two most convincing
    candidates by a label-free text score. A reference's true targets are usually
    near-duplicates of each other (Source 2 and Source 3 copies of one business), so a
    candidate resembling a strong anchor is likely a match even if it resembles the
    reference itself less."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    proxy = (np.clip(cols['name_translit_token_set'], 0, None) + np.clip(cols['address_folded_token_set'], 0, None)
             + 0.5 * np.clip(cols['address_numbers_token_set'], 0, None)).astype(np.float32)
    n = len(ref)
    order = np.lexsort((np.arange(n), -proxy, ref))
    sorted_ref = ref[order]
    position = np.arange(n) - np.searchsorted(sorted_ref, sorted_ref, side='left')
    rank = np.empty(n, dtype=np.int32)
    rank[order] = position + 1
    cols['proxy'] = proxy
    cols['proxy_rank'] = rank
    best = np.full(n_refs, -np.inf, dtype=np.float32)
    np.maximum.at(best, ref, proxy)
    cols['proxy_gap'] = proxy - best[ref]
    cols['strong_candidates'] = np.bincount(ref, weights=proxy >= 1.6, minlength=n_refs)[ref]
    texts = {f: target_field(f) for f in ANCHOR_FIELDS}
    empty = {f: np.fromiter((len(x) == 0 for x in texts[f]), dtype=bool, count=n) for f in ANCHOR_FIELDS}
    for j in (1, 2):
        anchor = np.full(n_refs, -1, dtype=np.int64)
        pick = position == j - 1
        anchor[sorted_ref[pick]] = order[pick]
        idx = anchor[ref]
        valid = idx >= 0
        safe = np.where(valid, idx, 0)
        for f in ANCHOR_FIELDS:
            v = cpdist(texts[f], texts[f][safe], scorer=fuzz.token_set_ratio, workers=threads, dtype=np.float32) / 100
            cols[f'anchor{j}_{f}'] = np.where(valid & ~empty[f] & ~empty[f][safe], v, -1.0)
        cols[f'is_anchor{j}'] = idx == np.arange(n)
        cols[f'anchor{j}_proxy'] = np.where(valid, proxy[safe], -1.0)


def embedding_features(cols, cands, ref, ord_, n_refs, ref_vecs, target_vecs):
    """Cosine of the fine-tuned multilingual embeddings for every pair (any script), plus
    its gap/rank within the reference and the dense-retrieval ranks."""
    n = len(ref)
    emb = np.zeros(n, dtype=np.float32)
    if ref_vecs is not None:
        for lo in range(0, n, 200000):
            hi = min(n, lo + 200000)
            emb[lo:hi] = np.einsum('ij,ij->i', ref_vecs[ref[lo:hi]],
                                   target_vecs[ord_[lo:hi]].astype(np.float32))
    cols['emb_cos'] = emb
    cols['emb_gap'] = emb - _group_max(emb, ref, n_refs)
    order = np.lexsort((-emb, ref))
    sorted_ref = ref[order]
    rank = np.empty(n, dtype=np.int32)
    rank[order] = np.arange(n) - np.searchsorted(sorted_ref, sorted_ref, side='left') + 1
    cols['emb_rank'] = rank
    for k in DENSE_FIELDS:
        cols[k] = cands[k] if k in cands else np.zeros(n, np.float32)
    cols['dense_inv_rank'] = np.where(cols['dense_rank'] > 0, 1.0 / np.maximum(cols['dense_rank'], 1), 0.0)
    cols['dense_rev_inv_rank'] = np.where(cols['dense_rev_rank'] > 0,
                                          1.0 / np.maximum(cols['dense_rev_rank'], 1), 0.0)
    cols['dense_rev_top1'] = cols['dense_rev_rank'] == 1


def features(cands, rows, table, threads, ref_vecs=None, target_vecs=None):
    """Label-free pair features; column order is FEATURE_NAMES."""
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz.process import cpdist
    ref, ord_ = cands['ref'], cands['ord']
    n_refs = len(rows)
    cols = {}
    for route in ROUTES:
        s, k = cands[f'score_{route}'], cands[f'rank_{route}']
        cols[f'score_{route}'] = s
        cols[f'inv_rank_{route}'] = np.where(k > 0, 1.0 / np.maximum(k, 1), 0.0)
        cols[f'gap_{route}'] = s - cands['top'][route][ref]  # vs the route's top-1; independent of keep
    cols['rrf'] = cands['rrf']
    cols['rrf_ratio'] = cands['rrf'] / np.maximum(_group_max(cands['rrf'], ref, n_refs), 1e-9)
    cols['fused_rank'] = cands['fused_rank']
    cols['route_hits'] = cands['route_hits']
    cols['in_forward'] = cands['fused_rank'] > 0
    for k in REVERSE_FIELDS:
        cols[k] = cands[k] if k in cands else np.zeros(len(ref), np.float32)
    cols['rev_inv_rank'] = np.where(cols['rev_rank'] > 0, 1.0 / np.maximum(cols['rev_rank'], 1), 0.0)
    cols['rev_top1'] = cols['rev_rank'] == 1

    def side(field):
        if field == 'name_core':
            a = np.array([core_name(r['name_folded']) for r in rows], dtype=object)[ref]
        elif field == 'name_phonetic':
            a = np.array([phonetic(r['name_folded']) for r in rows], dtype=object)[ref]
        elif field in ('name_translit', 'address_translit'):
            source = field.replace('translit', 'folded')
            a = np.array([transliterate(r[source]) for r in rows], dtype=object)[ref]
        else:
            a = np.array([r[field] for r in rows], dtype=object)[ref]
        return a, table.fields[field][ord_]

    scorers = {'ratio': fuzz.ratio, 'token_sort': fuzz.token_sort_ratio,
               'token_set': fuzz.token_set_ratio, 'partial': fuzz.partial_ratio}
    lengths = {}
    for field, names in (('name_folded', ('ratio', 'token_sort', 'token_set', 'partial')),
                         ('name_norm', ('token_set',)),
                         ('name_core', ('ratio', 'token_set')),
                         ('name_translit', ('ratio', 'token_set', 'partial')),
                         ('name_phonetic', ('ratio', 'token_set')),
                         ('address_translit', ('token_set',)),
                         ('address_folded', ('ratio', 'token_set', 'partial')),
                         ('address_numbers', ('token_set',)),
                         ('postal_candidates', ('token_set',))):
        a, b = side(field)
        la = np.fromiter((len(x) for x in a), dtype=np.int32, count=len(a))
        lb = np.fromiter((len(x) for x in b), dtype=np.int32, count=len(b))
        lengths[field] = (la, lb)
        both = (la > 0) & (lb > 0)
        for name in names:
            values = cpdist(a, b, scorer=scorers[name], workers=threads, dtype=np.float32) / 100
            cols[f'{field}_{name}'] = np.where(both, values, -1.0)
        if field == 'name_folded':
            jw = cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=threads, dtype=np.float32)
            cols['name_folded_jaro_winkler'] = np.where(both, jw, -1.0)
            cols['name_exact'] = (a == b) & both
            cols['name_first_token_equal'] = np.fromiter(
                (x.split(' ', 1)[0] == y.split(' ', 1)[0] for x, y in zip(a, b)), dtype=bool, count=len(a)) & both
        if field == 'address_folded':
            cols['address_exact'] = (a == b) & both
        cols[f'{field}_len_ratio'] = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)
        cols[f'{field}_ref_empty'] = la == 0
        cols[f'{field}_target_empty'] = lb == 0
    a, b = side('name_folded')
    cols['target_name_non_ascii'] = np.fromiter((not x.isascii() for x in b), dtype=bool, count=len(b))
    cols['ref_name_non_ascii'] = np.fromiter((not x.isascii() for x in a), dtype=bool, count=len(a))
    cols['target_source3'] = table.source3[ord_]
    group_features(cols, ref, n_refs, lambda f: table.fields[f][ord_], threads)
    embedding_features(cols, cands, ref, ord_, n_refs, ref_vecs, target_vecs)
    if not FEATURE_NAMES:  # pairs stage: every feature; predict: the model's own list
        FEATURE_NAMES[:] = list(cols)
    missing = [n for n in FEATURE_NAMES if n not in cols]
    if missing:
        raise ValueError(f'Model expects features this code does not compute: {missing}')
    return np.column_stack([np.asarray(cols[n], dtype=np.float32) for n in FEATURE_NAMES])


def generate(index_root, data_dir, split, refs, args, on_chunk):
    """Groups references by country, blocks and featurises them, calls on_chunk per batch."""
    manifest = load_manifest(index_root)
    by_country = defaultdict(list)
    for r in refs:
        by_country[r['country']].append(r)
    timing = Counter()
    for country in sorted(by_country):
        rows = by_country[country]
        if country not in manifest['countries']:
            for start in range(0, len(rows), args.batch):
                on_chunk(rows[start:start + args.batch], None, None, None)
            continue
        t = time.monotonic()
        index = CountryIndex(index_root, country, manifest)
        table = TargetTable(data_dir, split, country, index.ids)
        reverse = None
        if getattr(args, 'reverse', None):
            from reverse_index import Reverse
            reverse = Reverse(args.reverse, args.ref_index, manifest['countries'][country]['dir'], country)
        dense = None
        if getattr(args, 'dense', None):
            dense = Dense(args.dense, split, country, index.ids)
            dense = dense if dense.ok else None
        timing[f'load_{country}'] += time.monotonic() - t
        for start in range(0, len(rows), args.batch):
            batch = rows[start:start + args.batch]
            t = time.monotonic()
            cands = candidates(index, batch, args.top_k, args.top_terms, args.keep, args.threads,
                               routes=route_list(getattr(args, 'routes', '')))
            if reverse is not None:
                cands = merge_reverse(cands, reverse.lookup(batch))
            ref_vecs = None
            if dense is not None:
                fwd, rev, ref_vecs = dense.lookup(batch)
                cands = merge_dense(cands, fwd, rev)
            timing['block'] += time.monotonic() - t
            t = time.monotonic()
            x = features(cands, batch, table, args.threads, ref_vecs,
                         dense.targets if dense is not None else None) if len(cands['ref']) else None
            timing['features'] += time.monotonic() - t
            t = time.monotonic()
            on_chunk(batch, cands, x, table)
            timing['consume'] += time.monotonic() - t
            print(f'{country}: {start + len(batch):,}/{len(rows):,} references; '
                  f'{dict((k, round(v)) for k, v in timing.items())}', flush=True)
        del index, table, dense
    return dict(timing)


# ------------------------------------------------------------- stage: pairs ----

def sample(source1, per_country, seed, excluded):
    pools, counts, rngs = {}, Counter(), {}
    for row in records(source1):
        if row['entity_id'] in excluded:
            continue
        c = row['country']
        if c not in pools:
            pools[c], rngs[c] = [], random.Random(f'{seed}:{c}')
        counts[c] += 1
        if len(pools[c]) < per_country:
            pools[c].append(row)
        else:
            j = rngs[c].randrange(counts[c])
            if j < per_country:
                pools[c][j] = row
    return sorted((r for p in pools.values() for r in p), key=lambda r: r['entity_id'])


def read_gold(path, wanted):
    gold = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['source1_entity_id'] in wanted:
                gold[row['source1_entity_id']] = set(filter(None, row['matched_entity_ids'].split(',')))
    if set(gold) != wanted:
        raise ValueError('Missing ground truth for sampled references')
    return gold


def stage_pairs(a):
    started = time.monotonic()
    excluded = set()
    for path in a.exclude_ids:
        excluded.update(path.read_text(encoding='utf-8').split())
    refs = sample(a.data / 'train_source1.tsv.gz', a.per_country, a.seed, excluded)
    a.output.mkdir(parents=True)
    (a.output / 'reference_ids.txt').write_text('\n'.join(r['entity_id'] for r in refs) + '\n')
    xs, ref_ids, target_ids = [], [], []

    def consume(batch, cands, x, table):
        if cands is None or x is None:
            return
        xs.append(x)
        ref_ids.append(np.array([batch[i]['entity_id'] for i in cands['ref']], dtype=object))
        target_ids.append(table.fields['entity_id'][cands['ord']])

    timing = generate(a.index, a.data, 'train', refs, a, consume)
    x = np.vstack(xs)
    ref_ids, target_ids = np.concatenate(ref_ids), np.concatenate(target_ids)
    # Labels are read only now, after every candidate list is fixed.
    gold = read_gold(a.gold, {r['entity_id'] for r in refs})
    y = np.fromiter((t in gold[r] for r, t in zip(ref_ids, target_ids)), dtype=np.int8, count=len(ref_ids))
    np.save(a.output / 'x.npy', x)
    np.save(a.output / 'y.npy', y)
    with gzip.open(a.output / 'pairs.tsv.gz', 'wt', encoding='utf-8') as f:
        for r, t in zip(ref_ids, target_ids):
            f.write(f'{r}\t{t}\n')
    (a.output / 'gold.json').write_text(json.dumps({k: sorted(v) for k, v in gold.items()}))
    (a.output / 'countries.json').write_text(json.dumps({r['entity_id']: r['country'] for r in refs}))
    total = sum(map(len, gold.values()))
    rank = x[:, FEATURE_NAMES.index('fused_rank')]
    forward = rank > 0
    recall = {f'recall_at_{k}': float(y[forward & (rank <= k)].sum() / total)
              for k in (10, 20, 50, 100, 150, 200, 300) if k <= a.keep}
    recall['forward_plus_reverse'] = float(y.sum() / total)
    recall['reverse_only_pairs'] = int((~forward).sum())
    recall['reverse_only_positives'] = int(y[~forward].sum())
    rev_rank = x[:, FEATURE_NAMES.index('rev_rank')]
    for k in (100, 200):
        if k <= a.keep:
            recall[f'forward_at_{k}_plus_reverse'] = float(y[(forward & (rank <= k)) | (rev_rank > 0)].sum() / total)
    by_country = {}
    country_of = {r['entity_id']: r['country'] for r in refs}
    pair_country = np.array([country_of[r] for r in ref_ids], dtype=object)
    for c in sorted(set(country_of.values())):
        m = pair_country == c
        tot = sum(len(gold[r['entity_id']]) for r in refs if r['country'] == c)
        by_country[c] = {f'recall_at_{k}': float(y[m & forward & (rank <= k)].sum() / tot) if tot else None
                         for k in (50, 100, 200) if k <= a.keep}
        by_country[c]['forward_plus_reverse'] = float(y[m].sum() / tot) if tot else None
    report = {'references': len(refs), 'pairs': int(len(y)), 'positives': int(y.sum()),
              'gold_links': total, 'singletons': sum(not v for v in gold.values()),
              'candidate_recall': recall, 'candidate_recall_by_country': by_country,
              'features': FEATURE_NAMES, 'timing': timing, 'seconds': time.monotonic() - started,
              'parameters': {k: str(v) for k, v in vars(a).items()}}
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


# ------------------------------------------------------------- stage: train ----

def f05(truth, pred):
    if not truth and not pred:
        return 1.0
    tp = len(truth & pred)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def decide(ref_idx, prob, rule, threshold, n_refs):
    """Selected mask. 'threshold': p >= t. 'expected': per-reference top-k maximising
    approximate expected F0.5, and only if it beats the no-match expectation; t is a floor."""
    selected = prob >= threshold
    if rule == 'threshold':
        return selected
    order = np.lexsort((-prob, ref_idx))
    result = np.zeros(len(prob), dtype=bool)
    bounds = np.searchsorted(ref_idx[order], np.arange(n_refs + 1))
    for i in range(n_refs):
        idx = order[bounds[i]:bounds[i + 1]]
        if not len(idx):
            continue
        p = prob[idx].astype(np.float64)
        expected_size = p.sum()
        k = np.arange(1, len(p) + 1)
        value = 1.25 * np.cumsum(p) / (0.25 * expected_size + k)
        best = int(np.argmax(value))
        empty = float(np.prod(1 - np.clip(p, 0, 1 - 1e-9)))
        if value[best] > empty and p[0] >= threshold:
            result[idx[:best + 1]] = True
    return result


def macro_score(ref_names, ref_idx, targets, mask, gold):
    pred = defaultdict(set)
    for i, t in zip(ref_idx[mask], targets[mask]):
        pred[i].add(t)
    return float(np.mean([f05(gold[name], pred.get(i, set())) for i, name in enumerate(ref_names)]))


class Scorer:
    """One interface for the two matcher backends: LightGBM (CPU) and XGBoost (GPU if present)."""

    def __init__(self, backend, booster, threads=8, iteration_range=None):
        self.backend, self.booster, self.threads = backend, booster, threads
        self.iteration_range = iteration_range

    def predict(self, x):
        if self.backend == 'xgboost':
            kwargs = {'iteration_range': self.iteration_range} if self.iteration_range else {}
            return np.asarray(self.booster.inplace_predict(x, **kwargs), dtype=np.float64)
        return self.booster.predict(x, num_threads=self.threads)

    def save(self, directory):
        if self.backend == 'xgboost':
            self.booster.save_model(str(directory / 'model.json'))
        else:
            self.booster.save_model(str(directory / 'model.txt'))

    def gain(self, names):
        if self.backend == 'xgboost':
            score = self.booster.get_score(importance_type='total_gain')
            return {n: round(float(score.get(n, 0.0)), 1) for n in names}
        return dict(zip(names, self.booster.feature_importance('gain').round(1).tolist()))


def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def load_scorer(model_dir, threads):
    report = json.loads((model_dir / 'report.json').read_text())
    if report.get('backend') == 'xgboost':
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(model_dir / 'model.json'))
        booster.set_param({'device': 'cuda' if gpu_available() else 'cpu', 'nthread': threads})
        return Scorer('xgboost', booster, threads, (0, report['best_iteration'] + 1))
    import lightgbm as lgb
    return Scorer('lightgbm', lgb.Booster(model_file=str(model_dir / 'model.txt')), threads)


def fit_matcher(a, x_train, y_train, x_dev, y_dev, names):
    """Returns (scorer, params, best_iteration)."""
    if getattr(a, 'backend', 'lightgbm') == 'xgboost':
        import xgboost as xgb
        device = 'cuda' if gpu_available() else 'cpu'
        params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                      device=device, grow_policy='lossguide', max_depth=0, max_leaves=a.num_leaves,
                      learning_rate=a.learning_rate, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=5.0, min_child_weight=1.0, max_bin=256, seed=2026, nthread=a.threads)
        train_set = xgb.QuantileDMatrix(x_train, y_train, feature_names=names)
        dev_set = xgb.QuantileDMatrix(x_dev, y_dev, ref=train_set, feature_names=names)
        booster = xgb.train(params, train_set, num_boost_round=a.rounds, evals=[(dev_set, 'dev')],
                            early_stopping_rounds=a.early_stopping, verbose_eval=100)
        best = booster.best_iteration
        return Scorer('xgboost', booster, a.threads, (0, best + 1)), params, best
    import lightgbm as lgb
    params = dict(objective='binary', learning_rate=a.learning_rate, num_leaves=a.num_leaves,
                  min_data_in_leaf=a.min_data_in_leaf,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  num_threads=a.threads, verbose=-1, seed=2026)
    train_set = lgb.Dataset(x_train, y_train, feature_name=names, free_raw_data=False)
    dev_set = lgb.Dataset(x_dev, y_dev, reference=train_set)
    booster = lgb.train(params, train_set, num_boost_round=a.rounds, valid_sets=[dev_set],
                        callbacks=[lgb.early_stopping(a.early_stopping), lgb.log_evaluation(100)])
    return Scorer('lightgbm', booster, a.threads), params, booster.best_iteration


def stage_train(a):
    started = time.monotonic()
    report_in = json.loads((a.pairs / 'report.json').read_text())
    names = report_in['features']
    x, y = np.load(a.pairs / 'x.npy'), np.load(a.pairs / 'y.npy')
    pairs = [line.rstrip('\n').split('\t') for line in gzip.open(a.pairs / 'pairs.tsv.gz', 'rt')]
    pair_ref = np.array([p[0] for p in pairs], dtype=object)
    pair_target = np.array([p[1] for p in pairs], dtype=object)
    gold = {k: set(v) for k, v in json.loads((a.pairs / 'gold.json').read_text()).items()}
    # Components: references sharing a gold target stay in one partition.
    parent = {s: s for s in gold}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    owner = {}
    for s, targets in gold.items():
        for t in targets:
            if t in owner:
                parent[find(s)] = find(owner[t])
            owner[t] = s

    def part(s):
        h = int(hashlib.sha256(find(s).encode()).hexdigest()[:8], 16) % 100
        return 'train' if h < 70 else 'dev' if h < 85 else 'holdout'
    partition = {s: part(s) for s in gold}
    if a.train_refs:
        # Learning-curve runs: keep a fixed random subset of training references; dev and
        # holdout are untouched, so scores stay comparable across training-set sizes.
        pool = sorted(s for s, q in partition.items() if q == 'train')
        keep_ids = set(random.Random('train-refs').sample(pool, min(a.train_refs, len(pool))))
        partition = {s: ('unused' if q == 'train' and s not in keep_ids else q) for s, q in partition.items()}
    pair_part = np.array([partition[s] for s in pair_ref], dtype=object)
    in_range = (x[:, names.index('fused_rank')] <= a.max_rank) if a.max_rank else np.ones(len(y), bool)
    m = {p: (pair_part == p) & in_range for p in ('train', 'dev', 'holdout')}
    scorer, params, best_iteration = fit_matcher(a, x[m['train']], y[m['train']], x[m['dev']], y[m['dev']], names)

    def evaluate(p, rules):
        refs = sorted(s for s in gold if partition[s] == p)
        position = {s: i for i, s in enumerate(refs)}
        idx = np.flatnonzero(m[p])
        prob = scorer.predict(x[idx])
        ref_idx = np.array([position[s] for s in pair_ref[idx]])
        return {(rule, t): macro_score(refs, ref_idx, pair_target[idx], decide(ref_idx, prob, rule, t, len(refs)),
                                       gold)
                for rule, t in rules}
    grid = [(rule, round(float(t), 3)) for rule in ('threshold', 'expected') for t in np.arange(0.05, 0.96, 0.05)]
    dev = evaluate('dev', grid)
    best = max(dev, key=lambda k: (dev[k], k[1]))
    holdout = evaluate('holdout', [best])[best]
    by_country = {}
    countries_path = a.pairs / 'countries.json'
    if countries_path.exists():
        countries = json.loads(countries_path.read_text())
        refs = sorted(s for s in gold if partition[s] == 'holdout')
        position = {s: i for i, s in enumerate(refs)}
        idx = np.flatnonzero(m['holdout'])
        ref_idx = np.array([position[s] for s in pair_ref[idx]])
        chosen = decide(ref_idx, scorer.predict(x[idx]), best[0], best[1], len(refs))
        pred = defaultdict(set)
        for i, t in zip(ref_idx[chosen], pair_target[idx][chosen]):
            pred[i].add(t)
        scores = defaultdict(list)
        for i, s in enumerate(refs):
            scores[countries[s]].append(f05(gold[s], pred.get(i, set())))
        by_country = {c: {'references': len(v), 'macro_f05': float(np.mean(v))} for c, v in sorted(scores.items())}
    a.output.mkdir(parents=True)
    scorer.save(a.output)
    importance = scorer.gain(names)
    keep = a.max_rank or int(report_in['parameters']['keep'])
    report = {'decision': {'rule': best[0], 'threshold': best[1]}, 'keep': keep,
              'routes': report_in['parameters'].get('routes', ''),
              'macro_f05_dev': dev[best], 'macro_f05_holdout': holdout,
              'holdout_by_country': by_country,
              'dev_grid': {f'{r}@{t}': v for (r, t), v in sorted(dev.items())},
              'references': dict(Counter(partition.values())),
              'pairs': {p: int(v.sum()) for p, v in m.items()},
              'best_iteration': best_iteration, 'features': names,
              'backend': getattr(a, 'backend', 'lightgbm'),
              'lightgbm_params': {k: v for k, v in params.items() if k not in ('num_threads', 'nthread')},
              'train_refs_used': int(sum(q == 'train' for q in partition.values())),
              'feature_gain': dict(sorted(importance.items(), key=lambda kv: -kv[1])),
              'candidate_recall': report_in['candidate_recall'],
              'note': 'Holdout scored once with the dev-selected rule. Macro F0.5 counts gold links that '
                      'blocking missed and singleton references, so it is an end-to-end estimate.',
              'seconds': time.monotonic() - started}
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


# ----------------------------------------------------------- stage: predict ----

def stage_predict(a):
    started = time.monotonic()
    model_report = json.loads((a.model / 'report.json').read_text())
    booster = load_scorer(a.model, a.threads)
    rule, threshold = model_report['decision']['rule'], model_report['decision']['threshold']
    FEATURE_NAMES[:] = model_report['features']
    a.keep = a.keep or model_report['keep']
    a.routes = getattr(a, 'routes', '') or model_report.get('routes', '')
    refs = list(records(a.data / f'{a.split}_source1.tsv.gz'))
    if a.limit:
        refs = refs[:a.limit]
    if a.sample:
        refs = sorted(random.Random(a.sample_seed).sample(refs, min(a.sample, len(refs))),
                      key=lambda r: r['entity_id'])
    if a.shard:
        i, n = map(int, a.shard.split('/'))
        refs = refs[i::n]
    position = {r['entity_id']: i for i, r in enumerate(refs)}
    a.output.mkdir(parents=True)
    cands_out = {r['entity_id']: '' for r in refs}
    kept = {'ref': [], 'target': [], 'prob': []}
    stats = Counter()

    def consume(batch, cands, x, table):
        if cands is None or x is None:
            stats['refs_without_candidates'] += len(batch)
            return
        prob = booster.predict(x).astype(np.float32)
        ids = table.fields['entity_id'][cands['ord']]
        bounds = np.searchsorted(cands['ref'], np.arange(len(batch) + 1))  # grouped by ref
        for i, r in enumerate(batch):
            cands_out[r['entity_id']] = ','.join(ids[bounds[i]:bounds[i + 1]])
        # Pairs this unlikely can never be selected; drop them to bound memory.
        m = prob >= 0.02
        kept['ref'].append(np.array([position[batch[i]['entity_id']] for i in cands['ref'][m]], dtype=np.int64))
        kept['target'].append(ids[m])
        kept['prob'].append(prob[m])
        stats['pairs'] += len(prob)

    timing = generate(a.index, a.data, a.split, refs, a, consume)
    ref_idx = np.concatenate(kept['ref']) if kept['ref'] else np.zeros(0, np.int64)
    target = np.concatenate(kept['target']) if kept['target'] else np.zeros(0, object)
    prob = np.concatenate(kept['prob']) if kept['prob'] else np.zeros(0, np.float32)
    params = {k: str(v) for k, v in vars(a).items()}
    if a.shard:
        # Exclusivity and the decision rule need every reference: `merge` does them.
        ids = np.array([r['entity_id'] for r in refs])
        np.savez(a.output / 'scores.npz', ref=ids[ref_idx], target=target.astype(str), prob=prob)
        with open(a.output / 'candidates.tsv', 'w', encoding='utf-8', newline='') as f:
            for r in refs:
                f.write(r['entity_id'] + '\t' + cands_out[r['entity_id']] + '\n')
        (a.output / 'report.json').write_text(json.dumps(
            {'shard': a.shard, 'references': len(refs), 'stats': dict(stats), 'timing': timing,
             'seconds': time.monotonic() - started, 'parameters': params}, indent=2))
        return
    finalize(refs, cands_out, ref_idx, target, prob, model_report, a.exclusive, a.output, stats,
             {'split': a.split, 'timing': timing, 'started': started, 'parameters': params},
             parse_country_thresholds(a.country_threshold))


def stage_merge(a):
    """Combine `predict --shard` outputs: global exclusivity, decision rule, final TSVs."""
    started = time.monotonic()
    model_report = json.loads((a.model / 'report.json').read_text())
    refs = list(records(a.data / f'{a.split}_source1.tsv.gz'))
    position = {r['entity_id']: i for i, r in enumerate(refs)}
    cands_out, parts, stats = {}, [], Counter()
    for shard in a.shards:
        with open(shard / 'candidates.tsv', encoding='utf-8') as f:
            for line in f:
                sid, cands = line.rstrip('\n').split('\t')
                if sid in cands_out:
                    raise ValueError(f'{sid} appears in more than one shard')
                cands_out[sid] = cands
        parts.append(np.load(shard / 'scores.npz'))
        stats.update(json.loads((shard / 'report.json').read_text())['stats'])
    missing = [r['entity_id'] for r in refs if r['entity_id'] not in cands_out]
    if missing:
        raise ValueError(f'{len(missing)} references missing from shards, e.g. {missing[:3]}')
    ref_idx = np.array([position[s] for p in parts for s in p['ref']], dtype=np.int64)
    target = np.concatenate([p['target'] for p in parts]).astype(object)
    prob = np.concatenate([p['prob'] for p in parts])
    a.output.mkdir(parents=True)
    finalize(refs, cands_out, ref_idx, target, prob, model_report, a.exclusive, a.output, stats,
             {'split': a.split, 'shards': [str(s) for s in a.shards], 'started': started,
              'parameters': {k: str(v) for k, v in vars(a).items()}},
             parse_country_thresholds(a.country_threshold))


def route_list(text):
    """'name,address,combined' -> tuple; empty means every route the index has."""
    return tuple(r for r in text.split(',') if r) if text else None


def parse_country_thresholds(text):
    """'France=0.5,India=0.6' -> {'France': 0.5, 'India': 0.6}."""
    return {k: float(v) for k, v in (item.split('=') for item in text.split(',') if item)} if text else {}


def finalize(refs, cands_out, ref_idx, target, prob, model_report, exclusive, output, stats, extra,
             country_thresholds=None):
    rule, threshold = model_report['decision']['rule'], model_report['decision']['threshold']
    country_thresholds = country_thresholds or {}
    if exclusive and len(prob):
        # Ground truth links each target to at most one reference: the most probable one keeps it.
        # Exactly one winner per target; ties (e.g. duplicate references) go to the lowest ref index.
        _, t_idx = np.unique(target.astype(str), return_inverse=True)
        order = np.lexsort((ref_idx, -prob, t_idx))
        first = np.ones(len(order), dtype=bool)
        first[1:] = t_idx[order][1:] != t_idx[order][:-1]
        losers = np.ones(len(prob), dtype=bool)
        losers[order[first]] = False
        stats['exclusivity_zeroed_pairs'] = int(losers.sum())
        stats['exclusivity_zeroed_pairs_above_threshold'] = int((losers & (prob >= threshold)).sum())
        prob = np.where(losers, 0.0, prob).astype(np.float32)
    ref_country = np.array([r['country'] for r in refs], dtype=object)
    pair_country = ref_country[ref_idx] if len(ref_idx) else np.zeros(0, object)
    chosen = np.zeros(len(prob), dtype=bool)
    for country in sorted(set(ref_country)):
        m = pair_country == country
        if m.any():  # A per-country cutoff overrides the dev-chosen one (label-free calibration).
            chosen[m] = decide(ref_idx[m], prob[m], rule, country_thresholds.get(country, threshold), len(refs))
    # Per-country spread of each reference's best probability, to guide such cutoffs.
    best = np.zeros(len(refs), dtype=np.float32)
    if len(prob):
        np.maximum.at(best, ref_idx, prob)
    stats['best_prob_quantiles'] = {
        country: {q: float(np.quantile(best[ref_country == country], float(q)))
                  for q in ('0.05', '0.1', '0.25', '0.5')}
        for country in sorted(set(ref_country))}
    stats['country_thresholds'] = {c: country_thresholds.get(c, threshold) for c in sorted(set(ref_country))}
    selected = defaultdict(list)
    for i, t in zip(ref_idx[chosen], target[chosen]):
        selected[i].append(t)
    matches = {r['entity_id']: ','.join(sorted(selected.get(i, []))) for i, r in enumerate(refs)}
    stats['matches'] = int(chosen.sum())
    for name, column, table in (('matching_results.tsv', 'matched_entity_ids', matches),
                                ('candidate_pairs.tsv', 'candidate_entity_ids', cands_out)):
        partial = output / (name + '.partial')
        with open(partial, 'w', encoding='utf-8', newline='') as f:
            f.write(f'source1_entity_id\t{column}\n')
            for r in refs:
                f.write(r['entity_id'] + '\t' + table[r['entity_id']] + '\n')
        os.replace(partial, output / name)
    counts = Counter(v.count(',') + 1 if v else 0 for v in matches.values())
    per_country = defaultdict(Counter)
    for r in refs:
        n = matches[r['entity_id']].count(',') + 1 if matches[r['entity_id']] else 0
        c = cands_out[r['entity_id']].count(',') + 1 if cands_out[r['entity_id']] else 0
        pc = per_country[r['country']]
        pc['references'] += 1
        pc['no_match'] += n == 0
        pc['matches'] += n
        pc['candidates'] += c
    diagnostics = {country: {'references': pc['references'],
                             'no_match_rate': pc['no_match'] / pc['references'],
                             'mean_matches': pc['matches'] / pc['references'],
                             'mean_candidates': pc['candidates'] / pc['references']}
                   for country, pc in sorted(per_country.items())}
    started = extra.pop('started')
    report = {**extra, 'references': len(refs), 'decision': model_report['decision'],
              'stats': dict(stats), 'references_with_no_match': counts[0],
              'match_count_distribution': {str(k): v for k, v in sorted(counts.items())[:20]},
              'per_country': diagnostics,
              'train_reference_rates': 'no-match 5.6%, mean 3.46 matches per reference',
              'seconds': time.monotonic() - started}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    for name in ('pairs', 'predict'):
        s = sub.add_parser(name)
        s.add_argument('--data', type=Path, required=True, help='Preprocessed v2 split directory')
        s.add_argument('--index', type=Path, required=True)
        s.add_argument('--output', type=Path, required=True)
        s.add_argument('--top-k', type=int, default=200, help='Targets kept per route')
        s.add_argument('--top-terms', type=int, default=64)
        s.add_argument('--keep', type=int, default=0,
                       help='Fused candidates per reference (pairs: required; predict: default = model keep)')
        s.add_argument('--batch', type=int, default=4096)
        s.add_argument('--threads', type=int, default=8)
        s.add_argument('--routes', default='', help='Comma list of index routes to use (default: all; '
                       'predict: the routes the model was trained with)')
    s = sub.choices['pairs']
    s.add_argument('--gold', type=Path, required=True)
    s.add_argument('--per-country', type=int, default=30000)
    s.add_argument('--seed', default='matcher-v1')
    s.add_argument('--exclude-ids', type=Path, action='append', default=[])
    s = sub.choices['predict']
    s.add_argument('--split', choices=['train', 'test'], default='test')
    s.add_argument('--model', type=Path, required=True)
    s.add_argument('--limit', type=int, default=0, help='Only the first N references (smoke runs)')
    s.add_argument('--sample', type=int, default=0, help='Random subset of N references (trial runs)')
    s.add_argument('--sample-seed', default='predict-sample-v1')
    s.add_argument('--no-exclusive', dest='exclusive', action='store_false',
                   help='Do not restrict each target to its most probable reference')
    s.add_argument('--shard', default='', help='i/n: score every n-th reference from i; finish with `merge`')
    s.add_argument('--country-threshold', default='', help='Per-country decision cutoffs, e.g. France=0.5')
    s = sub.add_parser('merge')
    s.add_argument('--data', type=Path, required=True)
    s.add_argument('--split', choices=['train', 'test'], default='test')
    s.add_argument('--model', type=Path, required=True)
    s.add_argument('--shards', type=Path, nargs='+', required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--no-exclusive', dest='exclusive', action='store_false')
    s.add_argument('--country-threshold', default='', help='Per-country decision cutoffs, e.g. France=0.5')
    for name in ('pairs', 'predict'):
        s = sub.choices[name]
        s.add_argument('--ref-index', type=Path, help='Source 1 posting index (reverse blocking)')
        s.add_argument('--reverse', type=Path, help='reverse_index.py output for this split')
        s.add_argument('--dense', type=Path, help='dense_retrieval.py output root (embedding candidates + vectors)')
    s = sub.add_parser('train')
    s.add_argument('--pairs', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--rounds', type=int, default=1000)
    s.add_argument('--max-rank', type=int, default=0, help='Train/evaluate on fused rank <= N only')
    s.add_argument('--train-refs', type=int, default=0, help='Use only N training references (0 = all)')
    s.add_argument('--num-leaves', type=int, default=63)
    s.add_argument('--learning-rate', type=float, default=0.05)
    s.add_argument('--min-data-in-leaf', type=int, default=100)
    s.add_argument('--early-stopping', type=int, default=50)
    s.add_argument('--backend', choices=['lightgbm', 'xgboost'], default='lightgbm',
                   help='xgboost trains/scores on the GPU when one is available')
    s.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    if a.stage == 'pairs' and a.keep < 1:
        p.error('pairs needs --keep')
    if a.stage in ('pairs', 'predict') and bool(a.reverse) != bool(a.ref_index):
        p.error('--reverse and --ref-index go together')
    if a.output.exists():
        raise FileExistsError(f'{a.output} exists; choose a new output directory')
    {'pairs': stage_pairs, 'train': stage_train, 'predict': stage_predict, 'merge': stage_merge}[a.stage](a)


if __name__ == '__main__':
    main()
