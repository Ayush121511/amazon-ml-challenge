"""TF-IDF (character n-gram) cosine similarity blocking via sparse_dot_topn.

Unlike the exact-match blockers (blocking.py) and the hash-bucket approaches
(simhash_blocking.py, minhash_lsh_blocking.py), this one is safe by
construction rather than by a post-hoc frequency cap: sp_matmul_topn never
computes more than `top_n` results per Source-1 row, so the output size is
hard-bounded at (n_s1_rows * top_n) regardless of how skewed the data is.
That's the tradeoff sparse_dot_topn makes for being exact (real cosine scores)
rather than probabilistic (LSH) — see the earlier research/discussion.
"""
import pandas as pd
import sparse_dot_topn as sdt


def build_candidates(s1, other, s1_vecs, other_vecs, top_n=50, threshold=0.0, label="tfidf-topn"):
    """Partition by country (proven safe/recall-neutral) and take each S1
    row's top_n cosine-similarity matches within its own country subset."""
    parts = []
    countries = set(s1["country"]) & set(other["country"])
    for country in countries:
        s1_mask = (s1["country"] == country).to_numpy()
        other_mask = (other["country"] == country).to_numpy()
        if not s1_mask.any() or not other_mask.any():
            continue

        A = s1_vecs[s1_mask]
        B = other_vecs[other_mask].T.tocsr()
        C = sdt.sp_matmul_topn(A, B, top_n=top_n, threshold=threshold, sort=False, n_threads=-1)

        coo = C.tocoo()
        s1_ids = s1["entity_id"].to_numpy()[s1_mask]
        other_ids = other["entity_id"].to_numpy()[other_mask]
        print(f"  [{label}] country={country}: {A.shape[0]:,} x {B.shape[1]:,} -> {C.nnz:,} pairs "
              f"(<= {A.shape[0]*top_n:,} hard cap)", flush=True)

        parts.append(pd.DataFrame({
            "source1_entity_id": s1_ids[coo.row],
            "candidate_entity_id": other_ids[coo.col],
        }))

    if not parts:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    return pd.concat(parts, ignore_index=True).drop_duplicates()
