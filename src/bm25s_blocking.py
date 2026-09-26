"""BM25 (via bm25s) blocking on character n-grams, country-partitioned.

Same n-gram tokenization as tfidf_topn_blocking.py for a fair comparison --
the only variable that changes is the scoring function (BM25's term-frequency
saturation + length normalization vs plain TF-IDF cosine similarity).
"""
import bm25s
import pandas as pd

NGRAM_RANGE = (2, 4)  # reverted from (3,4): GPU matmul already solves the CPU speed problem (8x, validated),
# so we don't need to trade recall for it -- (2,4) keeps the +0.04pt recall (2,4) had over (3,4) on dev sample.
# Cached indexes (data/processed/bm25_indexes/) were built with (2,4); this must match at query time.


def char_ngrams(text, ngram_range=NGRAM_RANGE):
    tokens = []
    for word in text.split():
        padded = f" {word} "
        for n in range(ngram_range[0], ngram_range[1] + 1):
            tokens.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
    return tokens if tokens else [""]


def build_candidates(s1, other, top_n=75, label="bm25s"):
    parts = []
    countries = set(s1["country"]) & set(other["country"])
    for country in countries:
        s1_sub = s1[s1["country"] == country]
        other_sub = other[other["country"] == country]
        if len(s1_sub) == 0 or len(other_sub) == 0:
            continue

        corpus_tokens = [char_ngrams(t) for t in other_sub["name_normalized"]]
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)

        query_tokens = [char_ngrams(t) for t in s1_sub["name_normalized"]]
        k = min(top_n, len(other_sub))
        results, scores = retriever.retrieve(
            query_tokens, corpus=other_sub["entity_id"].to_numpy(),
            k=k, n_threads=-1, show_progress=False,
        )
        print(f"  [{label}] country={country}: {len(s1_sub):,} x {len(other_sub):,} -> "
              f"k={k} per query", flush=True)

        s1_ids = s1_sub["entity_id"].to_numpy()
        for i in range(len(s1_sub)):
            for j in range(results.shape[1]):
                parts.append((s1_ids[i], results[i, j]))

    if not parts:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    df = pd.DataFrame(parts, columns=["source1_entity_id", "candidate_entity_id"])
    return df.drop_duplicates()
