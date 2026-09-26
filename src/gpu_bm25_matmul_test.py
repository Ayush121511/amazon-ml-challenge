"""HPC-side test: load a cached BM25 index (built by hpc_build_indexes.py),
reconstruct its scoring as a sparse matmul on the A100 via CuPy/cuSPARSE, and
verify it reproduces bm25s's own CPU retrieve() output exactly. Also times
CPU-vs-GPU for the same query batch, at real full-scale corpus size (not a
toy sample), since that's the actual question: does GPU help at the size
that matters.
"""
import os
import sys
import time

import bm25s
import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy import sparse as csp
from scipy import sparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")
from bm25s_blocking import char_ngrams

INDEX_ROOT = "data/processed/bm25_indexes"


def tiny_sanity_check():
    """Hand-built 3x3 example with a known answer. If this fails, the bug is
    fundamental to this CuPy/CUDA environment, not our specific data."""
    A_cpu = sparse.csc_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32))
    B_cpu = sparse.csc_matrix(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    expected = (A_cpu @ B_cpu).toarray()

    A_gpu = csp.csr_matrix(A_cpu.tocsr())
    B_gpu = csp.csr_matrix(B_cpu.tocsr())
    got = cp.asnumpy((A_gpu @ B_gpu).toarray())
    ok = np.allclose(expected, got)
    print(f"TINY SANITY CHECK: expected=\n{expected}\ngot=\n{got}\nPASS={ok}", flush=True)
    return ok


def main():
    country = sys.argv[1] if len(sys.argv) > 1 else "India"
    n_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    print(f"CuPy version: {cp.__version__}")
    print(cp.cuda.runtime.getDeviceProperties(0)["name"])

    if not tiny_sanity_check():
        print("ABORTING: even the tiny hand-built sparse matmul failed on this GPU/CuPy setup.")
        return

    d = f"{INDEX_ROOT}/s2/{country}/name"
    t0 = time.time()
    retriever = bm25s.BM25.load(d, mmap=False, load_corpus=False, show_progress=False)
    entity_ids = np.load(f"{d}/entity_ids.npy", allow_pickle=True)
    print(f"loaded cached index ({country}/name, {len(entity_ids):,} docs) in {time.time()-t0:.1f}s")

    s1 = pd.read_parquet("data/processed/train_source1.parquet")
    s1 = s1[s1["country"] == country].head(n_queries).reset_index(drop=True)
    query_tokens = [char_ngrams(t) for t in s1["name_normalized"]]
    print(f"queries: {len(query_tokens):,}")

    k = 75
    n_threads = min(4, os.cpu_count() or 4)
    print(f"using n_threads={n_threads} for CPU baseline (matches this job's cpu allocation)")
    t0 = time.time()
    ref_results, ref_scores = retriever.retrieve(query_tokens, k=k, n_threads=n_threads, show_progress=False)
    cpu_time = time.time() - t0
    print(f"\nCPU (bm25s native) retrieve: {cpu_time:.2f}s for {len(query_tokens):,} queries "
          f"({cpu_time/len(query_tokens)*1000:.2f}ms/query)")

    data = retriever.scores["data"]
    indices = retriever.scores["indices"]
    indptr = retriever.scores["indptr"]
    num_docs = retriever.scores["num_docs"]
    num_terms = len(indptr) - 1
    vocab = retriever.vocab_dict

    t0 = time.time()
    # Build the CPU (SciPy) matrix first and let SciPy's own .tocsr() put it in
    # canonical form (sorted indices, deduplicated) before handing to CuPy --
    # confirmed via a tiny hand-built sanity check that wrapping an
    # already-canonical SciPy CSR object is the reliable path; constructing
    # raw CuPy arrays directly produced a silently empty (nnz=0) result.
    M_cpu = sparse.csc_matrix((data, indices, indptr), shape=(num_docs, num_terms)).astype(np.float32).tocsr()
    M_gpu = csp.csr_matrix(M_cpu)
    print(f"M uploaded to GPU: shape={M_gpu.shape}, nnz={M_gpu.nnz:,} in {time.time()-t0:.2f}s (one-time cost)")

    query_token_ids = [[vocab[t] for t in toks if t in vocab] for toks in query_tokens]
    rows, cols, vals = [], [], []
    for qi, ids in enumerate(query_token_ids):
        for tid in ids:
            rows.append(tid)
            cols.append(qi)
            vals.append(1.0)
    Q_cpu = sparse.csc_matrix((vals, (rows, cols)), shape=(num_terms, len(query_token_ids))).astype(np.float32)

    # Q is small (num_terms x n_queries, only a handful of nonzeros per query)
    # -- densify it and do sparse x dense (SpMM), not sparse x sparse (SpGEMM).
    # SpGEMM at this scale (2M x 225K, 122M nnz) silently returned an empty
    # result (nnz=0) despite a passing tiny hand-built sanity check and a
    # passing pure-CPU/scipy version at smaller scale -- looks like a
    # scale-specific CuPy/cuSPARSE limitation, not a construction bug. SpMM is
    # a simpler, more mature GPU operation and sidesteps it entirely.
    # Process queries in column-batches for both the matmul and top-k steps --
    # doing all 2,000 queries' dense (2M x 2000, ~16GB) scores plus argpartition
    # scratch in one shot pushed past the A100's 80GB (OOM). Batching keeps
    # peak memory bounded regardless of total query count, which is also what
    # a real production run (2.2M queries) will need.
    QUERY_BATCH = 200
    n_queries_total = len(query_token_ids)
    mempool = cp.get_default_memory_pool()

    t0 = time.time()
    mismatches = 0
    check_n = min(200, n_queries_total)
    debug_top5 = None
    for lo in range(0, n_queries_total, QUERY_BATCH):
        hi = min(lo + QUERY_BATCH, n_queries_total)
        Q_batch_dense = cp.asarray(Q_cpu[:, lo:hi].toarray())
        Scores_batch = M_gpu.dot(Q_batch_dense)  # (num_docs x batch_size)
        del Q_batch_dense

        if lo == 0:
            debug_top5 = sorted(cp.asnumpy(Scores_batch[:, 0]), reverse=True)[:5]

        top_idx_batch = cp.argpartition(-Scores_batch, min(k, Scores_batch.shape[0] - 1), axis=0)[:k, :]
        for j in range(hi - lo):
            qi = lo + j
            if qi >= check_n:
                break
            idx = cp.asnumpy(top_idx_batch[:, j])
            gpu_top_scores = sorted(cp.asnumpy(Scores_batch[:, j])[idx], reverse=True)
            cpu_top_scores = sorted(ref_scores[qi][:len(idx)], reverse=True)
            if not np.allclose(gpu_top_scores[:min(len(gpu_top_scores), len(cpu_top_scores))],
                                cpu_top_scores[:min(len(gpu_top_scores), len(cpu_top_scores))],
                                rtol=1e-3, atol=1e-4):
                mismatches += 1
        del Scores_batch, top_idx_batch
        mempool.free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    total_gpu_time = time.time() - t0

    print(f"debug: query 0 top-5 GPU scores: {debug_top5}")
    print(f"debug: query 0 top-5 CPU  scores: {sorted(ref_scores[0][:5].tolist(), reverse=True)}")
    print(f"\nTOTAL GPU (matmul + topk, batched, index already loaded): {total_gpu_time:.2f}s for {n_queries_total:,} queries "
          f"({total_gpu_time/n_queries_total*1000:.2f}ms/query)")
    print(f"speedup vs CPU bm25s native: {cpu_time/total_gpu_time:.2f}x")
    print(f"\ncorrectness check ({check_n} queries): {mismatches} mismatches")
    print("PASS: GPU matmul reproduces CPU bm25s scores" if mismatches == 0 else "FAIL: investigate")


if __name__ == "__main__":
    main()
