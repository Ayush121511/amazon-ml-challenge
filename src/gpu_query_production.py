"""Production GPU-accelerated BM25 name+address RRF fusion query pipeline.

Validated approach (src/gpu_bm25_matmul_test.py, job 1066121, 25 Sep 2026):
BM25 scoring is exactly a sparse matmul (bm25s's own cached index IS the
precomputed term-weight matrix); running it as sparse x dense (SpMM) on an
A100 reproduces bm25s's CPU scores exactly (0/200 mismatches) at 8.18x
speedup over CPU bm25s.retrieve(). Sparse x sparse (SpGEMM) silently returned
an empty result at this scale -- SpMM is the correct/safe approach here.

Checkpointed per query-batch: each batch's result is written to its own
parquet file immediately (atomic write-then-rename), and already-complete
batches are skipped on a rerun -- a walltime kill or crash only loses the
in-flight batch, not the whole run.

Usage: python src/gpu_query_production.py <source e.g. s2> <n_queries or 'all'> <top_k> <query_batch_size>
"""
import os
import sys
import time

import bm25s
import cupy as cp
import numpy as np
import pandas as pd
from scipy import sparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")
from bm25s_blocking import char_ngrams

INDEX_ROOT = "data/processed/bm25_indexes"
RRF_K = 60
SEED = 42

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "s2"
N_QUERIES = sys.argv[2] if len(sys.argv) > 2 else "all"
TOP_K = int(sys.argv[3]) if len(sys.argv) > 3 else 75
QUERY_BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 200


def load_gpu_index(country, channel):
    d = f"{INDEX_ROOT}/{SOURCE}/{country}/{channel}"
    retriever = bm25s.BM25.load(d, mmap=False, load_corpus=False, show_progress=False)
    entity_ids = np.load(f"{d}/entity_ids.npy", allow_pickle=True)
    data = retriever.scores["data"]
    indices = retriever.scores["indices"]
    indptr = retriever.scores["indptr"]
    num_docs = retriever.scores["num_docs"]
    num_terms = len(indptr) - 1
    vocab = retriever.vocab_dict
    # canonical SciPy CSR first, then wrap for CuPy -- validated as the
    # reliable construction path (raw-array construction silently produced
    # nnz=0 results at this scale).
    M_cpu = sparse.csc_matrix((data, indices, indptr), shape=(num_docs, num_terms)).astype(np.float32).tocsr()
    import cupyx.scipy.sparse as csp
    M_gpu = csp.csr_matrix(M_cpu)
    return M_gpu, vocab, entity_ids, num_terms


def gpu_topk_scores(M_gpu, vocab, num_terms, query_tokens, top_k, query_batch=QUERY_BATCH):
    """Returns list of (scores_array, doc_indices_array) per query, top_k each,
    sorted best-first. doc_indices index into the index's own entity_ids array."""
    mempool = cp.get_default_memory_pool()
    n = len(query_tokens)
    query_token_ids = [[vocab[t] for t in toks if t in vocab] for toks in query_tokens]
    out = [None] * n

    for lo in range(0, n, query_batch):
        hi = min(lo + query_batch, n)
        rows, cols, vals = [], [], []
        for j, ids in enumerate(query_token_ids[lo:hi]):
            for tid in ids:
                rows.append(tid)
                cols.append(j)
                vals.append(1.0)
        Q_cpu = sparse.csc_matrix((vals, (rows, cols)), shape=(num_terms, hi - lo)).astype(np.float32)
        Q_dense_gpu = cp.asarray(Q_cpu.toarray())
        Scores_batch = M_gpu.dot(Q_dense_gpu)  # (num_docs x batch_size)
        del Q_dense_gpu

        k = min(top_k, Scores_batch.shape[0])
        top_idx_batch = cp.argpartition(-Scores_batch, k - 1, axis=0)[:k, :]
        for j in range(hi - lo):
            idx = top_idx_batch[:, j]
            scores = Scores_batch[idx, j]
            order = cp.argsort(-scores)
            out[lo + j] = (cp.asnumpy(scores[order]), cp.asnumpy(idx[order]))
        del Scores_batch, top_idx_batch
        mempool.free_all_blocks()
    return out


def rrf_fuse_one(name_result, addr_result, name_ids, addr_ids, top_k):
    scores = {}
    if name_result is not None:
        s, idx = name_result
        for rank, i in enumerate(idx, 1):
            oid = name_ids[i]
            scores[oid] = scores.get(oid, {})
            scores[oid]["name_score"] = float(s[rank - 1])
            scores[oid]["name_rank"] = rank
    if addr_result is not None:
        s, idx = addr_result
        for rank, i in enumerate(idx, 1):
            oid = addr_ids[i]
            scores[oid] = scores.get(oid, {})
            scores[oid]["addr_score"] = float(s[rank - 1])
            scores[oid]["addr_rank"] = rank
    for d in scores.values():
        rrf = 0.0
        if "name_rank" in d:
            rrf += 1.0 / (RRF_K + d["name_rank"])
        if "addr_rank" in d:
            rrf += 1.0 / (RRF_K + d["addr_rank"])
        d["rrf_score"] = rrf
    top_oids = sorted(scores, key=lambda o: -scores[o]["rrf_score"])[:top_k]
    return [(oid, scores[oid]) for oid in top_oids]


def process_country(country, s1_sub, top_k, batch_dir):
    out_path = f"{batch_dir}/{country}.parquet"
    if os.path.exists(out_path):
        print(f"[{country}] already complete, skipping -> {out_path}", flush=True)
        return out_path

    t0 = time.time()
    name_M, name_vocab, name_ids, name_terms = load_gpu_index(country, "name")
    addr_M, addr_vocab, addr_ids, addr_terms = load_gpu_index(country, "address")
    print(f"[{country}] indexes loaded to GPU in {time.time()-t0:.1f}s "
          f"(name: {name_M.shape}, addr: {addr_M.shape})", flush=True)

    name_tokens = [char_ngrams(t) for t in s1_sub["name_normalized"]]
    addr_tokens = [char_ngrams(t) for t in s1_sub["address_normalized"]]
    qids = s1_sub["entity_id"].to_numpy()

    t0 = time.time()
    name_results = gpu_topk_scores(name_M, name_vocab, name_terms, name_tokens, top_k)
    del name_M
    cp.get_default_memory_pool().free_all_blocks()
    addr_results = gpu_topk_scores(addr_M, addr_vocab, addr_terms, addr_tokens, top_k)
    del addr_M
    cp.get_default_memory_pool().free_all_blocks()
    elapsed = time.time() - t0
    print(f"[{country}] {len(s1_sub):,} queries scored (both channels) in {elapsed:.1f}s "
          f"({elapsed/len(s1_sub)*1000:.2f}ms/query)", flush=True)

    parts = []
    for i, qid in enumerate(qids):
        fused = rrf_fuse_one(name_results[i], addr_results[i], name_ids, addr_ids, top_k)
        for oid, d in fused:
            parts.append((qid, oid, d.get("name_score"), d.get("name_rank"),
                          d.get("addr_score"), d.get("addr_rank"), d["rrf_score"]))

    df = pd.DataFrame(parts, columns=["source1_entity_id", "candidate_entity_id",
                                       "name_score", "name_rank", "addr_score", "addr_rank", "rrf_score"])
    tmp_path = out_path + ".tmp"
    df.to_parquet(tmp_path)
    os.replace(tmp_path, out_path)
    print(f"[{country}] saved {len(df):,} candidate rows -> {out_path}", flush=True)
    return out_path


def recall_of(cand_df, true_pairs, label):
    merged = true_pairs.merge(cand_df, on=["source1_entity_id", "candidate_entity_id"], how="left", indicator=True)
    found = (merged["_merge"] == "both").sum()
    total = len(true_pairs)
    per_entity = merged.groupby("source1_entity_id")["_merge"].apply(lambda s: (s == "both").all())
    print(f"[{label}] pair recall: {found}/{total} = {found/max(total,1):.4f} | "
          f"entities with ALL true matches recovered: {per_entity.mean():.4f} | candidate pairs: {len(cand_df):,}")


def main():
    print(f"CuPy version: {cp.__version__}, GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}", flush=True)
    t0 = time.time()
    s1_full = pd.read_parquet("data/processed/train_source1.parquet")
    print(f"loaded S1={len(s1_full):,} in {time.time()-t0:.1f}s, top_k={TOP_K}, query_batch={QUERY_BATCH}", flush=True)

    if N_QUERIES == "all":
        s1_sample = s1_full
    else:
        n = int(N_QUERIES)
        parts = []
        for country, g in s1_full.groupby("country"):
            k = min(len(g), max(1, round(n * len(g) / len(s1_full))))
            parts.append(g.sample(n=k, random_state=SEED))
        s1_sample = pd.concat(parts, ignore_index=True)
    print(f"S1 query sample: {len(s1_sample):,}  {s1_sample['country'].value_counts().to_dict()}", flush=True)

    batch_dir = f"{INDEX_ROOT}/{SOURCE}_gpu_production_n{len(s1_sample)}"
    os.makedirs(batch_dir, exist_ok=True)
    print(f"checkpoint dir: {batch_dir} (already-complete countries are skipped on rerun)", flush=True)

    t0 = time.time()
    country_paths = []
    for country in sorted(s1_sample["country"].unique()):
        s1_sub = s1_sample[s1_sample["country"] == country]
        country_paths.append(process_country(country, s1_sub, TOP_K, batch_dir))
    elapsed = time.time() - t0
    print(f"\nall countries done in {elapsed:.1f}s wall time", flush=True)

    cand = pd.concat([pd.read_parquet(p) for p in country_paths], ignore_index=True)
    print(f"total candidate pairs: {len(cand):,}, avg/entity: {len(cand)/len(s1_sample):.1f}", flush=True)

    gt = pd.read_csv("data/student_resource/dataset/train/train_ground_truth.tsv", sep="\t",
                      dtype=str, keep_default_na=False)
    gt_sample = gt[gt["source1_entity_id"].isin(set(s1_sample["entity_id"]))]
    prefix = SOURCE.upper() + "-"
    pairs = []
    for row in gt_sample.itertuples(index=False):
        if row.matched_entity_ids:
            for mid in row.matched_entity_ids.split(","):
                if mid.startswith(prefix):
                    pairs.append((row.source1_entity_id, mid))
    true_pairs = pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])
    print(f"true pairs for sampled queries: {len(true_pairs)}", flush=True)
    recall_of(cand, true_pairs, f"GPU fused vs FULL {SOURCE.upper()}")

    per_query_s = elapsed / len(s1_sample)
    print(f"\nper-query wall time: {per_query_s*1000:.2f}ms")
    full_scale_estimate_hr = per_query_s * 2_206_821 / 3600
    print(f"naive full-S1-scale extrapolation at this rate: {full_scale_estimate_hr:.2f}h")

    final_path = f"{INDEX_ROOT}/{SOURCE}_gpu_production_result_n{len(s1_sample)}.parquet"
    cand.to_parquet(final_path)
    print(f"saved merged candidates (with scores) to {final_path}", flush=True)


if __name__ == "__main__":
    main()
