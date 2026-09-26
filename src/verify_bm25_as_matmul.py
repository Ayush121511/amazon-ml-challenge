"""Prove BM25 scoring is exactly a sparse matrix multiply, not an approximation.

bm25s.BM25._compute_relevance_from_scores (confirmed via its own source) does:
    scores = zeros(num_docs)
    for each query token id: scores[doc_ids_for_that_term] += weight_for_that_term
This is precisely scores = M.T @ q where M is the (num_terms x num_docs) CSC
matrix bm25s already builds and saves at index time, and q is a sparse vector
counting each query token id's occurrences. Batched across many queries it's
Scores = M.T @ Q -- ordinary sparse matmul, which is why this is GPU-portable
with zero recall change: same numbers, different processor.

This script builds a small BM25 index locally, reconstructs M and Q by hand
from bm25s's own internal arrays, computes scores via scipy sparse matmul,
and checks the result is numerically identical to bm25s.retrieve()'s own
output. Also tries the same matmul via PyTorch on this Mac's GPU (MPS
backend) as a smoke test that sparse matmul executes correctly on *some*
GPU -- MPS is a different architecture than the A100 we'd actually deploy
on (CUDA/cuSPARSE), so this proves the reformulation works end-to-end on
GPU hardware in general, not A100 throughput specifically.
"""
import sys
import time

import bm25s
import numpy as np
import pandas as pd
from scipy import sparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")
from bm25s_blocking import char_ngrams

DEV = "data/processed/dev_sample"


def main():
    s2 = pd.read_csv(f"{DEV}/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = s2[s2["country"] == "India"].head(20000).reset_index(drop=True)
    s1 = pd.read_csv(f"{DEV}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1 = s1[s1["country"] == "India"].head(500).reset_index(drop=True)

    sys.path.insert(0, "src")
    from normalize import normalize_name
    corpus_text = [normalize_name(t)["normalized"] for t in s2["business_name"]]
    query_text = [normalize_name(t)["normalized"] for t in s1["business_name"]]

    corpus_tokens = [char_ngrams(t) for t in corpus_text]
    query_tokens = [char_ngrams(t) for t in query_text]

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)

    k = 20
    ref_results, ref_scores = retriever.retrieve(query_tokens, k=k, n_threads=1, show_progress=False)

    data = retriever.scores["data"]
    indices = retriever.scores["indices"]
    indptr = retriever.scores["indptr"]
    num_docs = retriever.scores["num_docs"]
    num_terms = len(indptr) - 1

    # M: (num_docs x num_terms) CSC matrix -- indptr is compressed by TERM
    # (length num_terms+1), confirmed empirically, so terms are the columns.
    M = sparse.csc_matrix((data, indices, indptr), shape=(num_docs, num_terms))
    print(f"M shape (docs x terms): {M.shape}, nnz={M.nnz:,}")

    vocab = retriever.vocab_dict
    unk_id = -1
    query_token_ids = [
        [vocab[t] for t in toks if t in vocab] for toks in query_tokens
    ]

    n_queries = len(query_token_ids)
    rows, cols, vals = [], [], []
    for qi, ids in enumerate(query_token_ids):
        for tid in ids:
            rows.append(tid)
            cols.append(qi)
            vals.append(1.0)
    Q = sparse.csc_matrix((vals, (rows, cols)), shape=(num_terms, n_queries))
    print(f"Q shape (terms x queries): {Q.shape}, nnz={Q.nnz:,}")

    t0 = time.time()
    Scores = (M @ Q).toarray()  # (num_docs x num_queries) -- dense only because this test corpus is small
    print(f"scipy sparse matmul done in {time.time()-t0:.3f}s")

    mismatches = 0
    for qi in range(n_queries):
        matmul_top = np.argsort(-Scores[:, qi])[:k]
        matmul_top_scores = Scores[matmul_top, qi]
        ref_top_scores = ref_scores[qi][:len(matmul_top)]
        # compare score sets (order can tie-break differently), not doc identity,
        # since np.add.at accumulation order vs scipy matmul order can differ in
        # float rounding at the ~1e-6 level on tied/near-tied entries
        if not np.allclose(sorted(matmul_top_scores, reverse=True),
                            sorted(ref_top_scores, reverse=True), rtol=1e-4, atol=1e-5):
            mismatches += 1

    print(f"\nqueries checked: {n_queries}, mismatches (top-{k} score sets differ beyond float tolerance): {mismatches}")
    print("PASS: sparse matmul reproduces bm25s's own CPU scores" if mismatches == 0 else "FAIL: investigate")

    # bonus: same matmul on this Mac's GPU (Metal/MPS) -- different hardware
    # than the A100 target, just proves sparse matmul executes correctly on GPU
    try:
        import torch
        if torch.backends.mps.is_available():
            M_coo = M.tocoo()
            Q_coo = Q.tocoo()
            M_t = torch.sparse_coo_tensor(
                np.vstack([M_coo.row, M_coo.col]), M_coo.data.astype(np.float32), M_coo.shape
            ).coalesce().to("mps")
            Q_t = torch.sparse_coo_tensor(
                np.vstack([Q_coo.row, Q_coo.col]), Q_coo.data.astype(np.float32), Q_coo.shape
            ).coalesce().to("mps")
            t0 = time.time()
            Scores_gpu = torch.sparse.mm(M_t, Q_t).to_dense().cpu().numpy()
            torch.mps.synchronize()
            print(f"\nPyTorch MPS (Apple GPU) sparse matmul done in {time.time()-t0:.3f}s")
            gpu_mismatches = int(np.sum(~np.isclose(Scores_gpu, Scores, rtol=1e-3, atol=1e-4)))
            total = Scores.size
            print(f"GPU vs scipy-CPU numeric agreement: {total - gpu_mismatches}/{total} entries within tolerance")
        else:
            print("\nMPS not available, skipping GPU smoke test")
    except Exception as e:
        print(f"\nGPU smoke test skipped/failed (non-fatal, correctness proof above stands): {e}")


if __name__ == "__main__":
    main()
