"""MinHash + LSH (Jaccard similarity) blocking — the set-overlap counterpart
to simhash_blocking.py's cosine-similarity approach.

Each business name becomes a SET of character 3-grams (shingles), so this is
typo-robust the same way n-gram TF-IDF is. MinHash compresses each set into a
short signature such that the probability two signatures agree on a given
hash function equals the true Jaccard similarity of the underlying sets —
then LSH bands the signature into buckets so only records with a high
probability of being similar ever get compared, without ever computing a
full pairwise Jaccard matrix.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import blocking_precheck as precheck_mod

NGRAM_RANGE = (3, 3)
NUM_HASHES = 24
NUM_BANDS = 8
ROWS_PER_BAND = NUM_HASHES // NUM_BANDS
SEED = 42
BAND_PRIME = 2_147_483_647  # large prime, keeps composite band ids well-spread


def fit_shingle_vectorizer(*text_series):
    corpus = pd.concat(list(text_series), ignore_index=True)
    vec = CountVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE, binary=True, min_df=2)
    vec.fit(corpus)
    return vec


def compute_minhash_signatures(shingle_matrix, num_hashes=NUM_HASHES, seed=SEED):
    """Vectorized MinHash: one groupby-min pass computes all `num_hashes` signatures at once."""
    X = shingle_matrix.tocsr()
    n_rows, vocab_size = X.shape
    rng = np.random.default_rng(seed)

    # One random hash value per (shingle, hash function) — independent permutations.
    H = rng.integers(0, np.iinfo(np.int64).max, size=(vocab_size, num_hashes), dtype=np.int64)

    row_idx = np.repeat(np.arange(n_rows), np.diff(X.indptr))
    col_idx = X.indices
    values = H[col_idx]  # (nnz, num_hashes)

    df = pd.DataFrame(values)
    df["row"] = row_idx
    sigs = df.groupby("row").min()
    # rows with zero shingles (empty name) get no group — reindex and fill with a
    # sentinel so they simply never collide with anything.
    sigs = sigs.reindex(range(n_rows), fill_value=np.iinfo(np.int64).max)
    return sigs.to_numpy()


def lsh_bucket_ids(signatures, num_bands=NUM_BANDS, rows_per_band=ROWS_PER_BAND):
    """Combine each band's rows_per_band signature values into one composite bucket id."""
    n_rows, num_hashes = signatures.shape
    assert num_bands * rows_per_band == num_hashes
    out = np.empty((n_rows, num_bands), dtype=np.int64)
    for b in range(num_bands):
        band = signatures[:, b * rows_per_band:(b + 1) * rows_per_band]
        bucket = np.zeros(n_rows, dtype=np.int64)
        for r in range(rows_per_band):
            bucket = (bucket * BAND_PRIME + band[:, r]) % np.iinfo(np.int64).max
        out[:, b] = bucket
    return out


def build_candidates(s1, other, s1_buckets, other_buckets, label="minhash"):
    num_bands = s1_buckets.shape[1]
    parts = []
    for b in range(num_bands):
        s1_b = pd.DataFrame({
            "entity_id": s1["entity_id"].values,
            "country": s1["country"].values,
            "bucket": s1_buckets[:, b],
        })
        other_b = pd.DataFrame({
            "entity_id": other["entity_id"].values,
            "country": other["country"].values,
            "bucket": other_buckets[:, b],
        })

        report = precheck_mod.precheck(s1_b, other_b, ["bucket", "country"], label=f"{label} band {b}")
        precheck_mod.print_report(report)
        if not report["safe"]:
            print(f"  [{label} band {b}] SKIPPED — would exceed safety cap", flush=True)
            continue

        m = s1_b.merge(other_b, on=["bucket", "country"], suffixes=("_s1", ""))
        m = m[["entity_id_s1", "entity_id"]].rename(
            columns={"entity_id_s1": "source1_entity_id", "entity_id": "candidate_entity_id"}
        )
        parts.append(m)

    if not parts:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    return pd.concat(parts, ignore_index=True).drop_duplicates()
