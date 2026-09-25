"""TF-IDF (character n-gram) + Random Projection LSH (SimHash) blocking.

Why this over the exact-match blockers in blocking.py: those need an
explicit, hand-tuned frequency cap per key because their cost is driven by
however skewed the data's key values happen to be (this is what blew up
twice on HPC). SimHash's cost is instead governed by parameters we choose
(bits per table, number of tables) — a common word no longer dominates a
bucket, because the hash depends on the whole weighted TF-IDF vector across
many random directions at once, not any single token.

Character n-grams (not whole words) also make this typo-robust in a way the
exact-match/whole-token blockers in blocking.py are not: "Technolgies" still
shares most of its 3-4 char chunks with "Technologies".
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

import blocking_precheck as precheck_mod

NGRAM_RANGE = (2, 4)
NUM_TABLES = 6
BITS_PER_TABLE = 16
SEED = 42


def fit_vectorizer(*text_series):
    """Fit one TfidfVectorizer across all sources so column indices align."""
    corpus = pd.concat(list(text_series), ignore_index=True)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE, min_df=2)
    vec.fit(corpus)
    return vec


def _random_projection_matrices(vocab_size, num_tables, bits_per_table, seed):
    rng = np.random.default_rng(seed)
    # One (vocab_size x bits_per_table) random hyperplane matrix per table.
    return [rng.standard_normal((vocab_size, bits_per_table)).astype(np.float32) for _ in range(num_tables)]


def _bucket_ids(tfidf_matrix, projections):
    """Return an (n_records, num_tables) int array of bucket ids, one per table."""
    n = tfidf_matrix.shape[0]
    out = np.empty((n, len(projections)), dtype=np.int64)
    for t, proj in enumerate(projections):
        signs = (tfidf_matrix @ proj) > 0  # sparse @ dense -> dense (n, bits)
        bits = np.asarray(signs, dtype=np.int64)
        # pack bits into a single integer bucket id
        weights = (1 << np.arange(bits.shape[1]))
        out[:, t] = bits @ weights
    return out


def build_candidates(s1, other, s1_vecs, other_vecs, num_tables=NUM_TABLES,
                      bits_per_table=BITS_PER_TABLE, seed=SEED, label="simhash"):
    """s1_vecs/other_vecs: TF-IDF sparse matrices already aligned to a shared vocab."""
    projections = _random_projection_matrices(s1_vecs.shape[1], num_tables, bits_per_table, seed)
    s1_buckets = _bucket_ids(s1_vecs, projections)
    other_buckets = _bucket_ids(other_vecs, projections)

    parts = []
    for t in range(num_tables):
        s1_t = pd.DataFrame({
            "entity_id": s1["entity_id"].values,
            "country": s1["country"].values,
            "bucket": s1_buckets[:, t],
        })
        other_t = pd.DataFrame({
            "entity_id": other["entity_id"].values,
            "country": other["country"].values,
            "bucket": other_buckets[:, t],
        })

        report = precheck_mod.precheck(s1_t, other_t, ["bucket", "country"], label=f"{label} table {t}")
        precheck_mod.print_report(report)

        # Prune only the offending (bucket, country) values, not the whole
        # table — a handful of oversized buckets shouldn't cost us every
        # other bucket's safe candidates.
        safe = precheck_mod.safe_keys(s1_t, other_t, ["bucket", "country"])
        n_dropped = report["n_shared_keys"] - len(safe)
        if n_dropped:
            print(f"  [{label} table {t}] pruned {n_dropped}/{report['n_shared_keys']} oversized buckets, "
                  f"kept {len(safe)}", flush=True)
        if len(safe) == 0:
            continue
        keep_df = safe.to_frame(index=False)

        s1_t = s1_t.merge(keep_df, on=["bucket", "country"])
        other_t = other_t.merge(keep_df, on=["bucket", "country"])
        m = s1_t.merge(other_t, on=["bucket", "country"], suffixes=("_s1", ""))
        m = m[["entity_id_s1", "entity_id"]].rename(
            columns={"entity_id_s1": "source1_entity_id", "entity_id": "candidate_entity_id"}
        )
        parts.append(m)

    if not parts:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    return pd.concat(parts, ignore_index=True).drop_duplicates()
