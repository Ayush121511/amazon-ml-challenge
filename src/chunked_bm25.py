"""BM25 retrieval with name+address reciprocal-rank fusion, built the way
production BM25 systems actually scale: index the target pool ONCE per
country, then batch only the query side.

Earlier version of this module chunked the *target* side and rebuilt a full
BM25 index per chunk -- backwards from how BM25/Lucene/Elasticsearch scale in
practice ("index once, query millions of times"; rebuilding the index per
batch pays its full construction cost N times over for no reason). Confirmed
via research: bm25s itself has been benchmarked directly against 2M+-document
corpora, comfortably inside our per-country target pool sizes (~2-3M), so the
target side doesn't need chunking at our scale -- only the query side does,
since S1 has 2.2M entities and we don't want one giant retrieve() call.
"""
from collections import defaultdict

import bm25s
import pandas as pd

from bm25s_blocking import char_ngrams

RRF_K = 60  # reciprocal rank fusion constant, matches the teammate's own choice


def bm25_topk_batched_queries(query_tokens, query_ids, other_df, text_field,
                               top_k, query_batch_size=20000, label=""):
    """Index `other_df` ONCE, then serve queries in batches. Returns
    {query_id: [(score, other_id), ...]} sorted best-first."""
    corpus_tokens = [char_ngrams(t) for t in other_df[text_field]]
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    other_ids = other_df["entity_id"].to_numpy()
    k = min(top_k, len(other_df))

    out = {}
    n = len(query_tokens)
    for lo in range(0, n, query_batch_size):
        batch_tokens = query_tokens[lo:lo + query_batch_size]
        batch_ids = query_ids[lo:lo + query_batch_size]
        results, scores = retriever.retrieve(
            batch_tokens, corpus=other_ids, k=k, n_threads=-1, show_progress=False,
        )
        for i, qid in enumerate(batch_ids):
            out[qid] = list(zip(scores[i].tolist(), results[i].tolist()))
    return out


def rrf_fuse(name_ranked, addr_ranked, top_k, rrf_k=RRF_K):
    """Reciprocal rank fusion of two per-query ranked lists into one, same
    formula the teammate used: score += 1 / (rrf_k + rank)."""
    fused = {}
    for qid in set(name_ranked) | set(addr_ranked):
        scores = defaultdict(float)
        for rank, (_, other_id) in enumerate(name_ranked.get(qid, []), 1):
            scores[other_id] += 1.0 / (rrf_k + rank)
        for rank, (_, other_id) in enumerate(addr_ranked.get(qid, []), 1):
            scores[other_id] += 1.0 / (rrf_k + rank)
        fused[qid] = sorted(scores, key=lambda oid: -scores[oid])[:top_k]
    return fused


def build_candidates_fused(s1, other, top_k=75, query_batch_size=20000, label="fused"):
    """Country-partitioned: name-channel + address-channel BM25 (each
    indexed once), RRF-fused, query side batched."""
    parts = []
    countries = set(s1["country"]) & set(other["country"])
    for country in countries:
        s1_sub = s1[s1["country"] == country]
        other_sub = other[other["country"] == country]
        if len(s1_sub) == 0 or len(other_sub) == 0:
            continue

        query_ids = s1_sub["entity_id"].to_numpy()
        name_query_tokens = [char_ngrams(t) for t in s1_sub["name_normalized"]]
        addr_query_tokens = [char_ngrams(t) for t in s1_sub["address_normalized"]]

        name_topk = bm25_topk_batched_queries(name_query_tokens, query_ids, other_sub, "name_normalized",
                                               top_k, query_batch_size, label=f"{label}-name-{country}")
        addr_topk = bm25_topk_batched_queries(addr_query_tokens, query_ids, other_sub, "address_normalized",
                                               top_k, query_batch_size, label=f"{label}-addr-{country}")
        fused = rrf_fuse(name_topk, addr_topk, top_k)

        print(f"  [{label}] country={country}: {len(s1_sub):,} queries x {len(other_sub):,} targets "
              f"(index built once, {(len(query_ids)+query_batch_size-1)//query_batch_size} query batches)",
              flush=True)

        for qid, ranked_ids in fused.items():
            for oid in ranked_ids:
                parts.append((qid, oid))

    if not parts:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    df = pd.DataFrame(parts, columns=["source1_entity_id", "candidate_entity_id"])
    return df.drop_duplicates()
