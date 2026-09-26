"""Query the cached BM25 indexes (built by hpc_build_indexes.py) with a
sharded, aggressively parallel S1 query pool. Each shard is its own OS
process (fork) that loads the cached index read-only (mmap where possible),
runs its slice of queries for both channels, and RRF-fuses -- this is the
"index once, query many, in parallel" architecture applied at HPC-node scale
(dozens of cores, hundreds of GB RAM), unlike the local 16GB machine where
process-per-shard blew memory.

Usage: python src/hpc_query_shards.py <source e.g. s2> <n_queries or 'all'> <top_k> <n_workers>
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import bm25s
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")
from bm25s_blocking import char_ngrams

INDEX_ROOT = "data/processed/bm25_indexes"
RRF_K = 60
SEED = 42

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "s2"
N_QUERIES = sys.argv[2] if len(sys.argv) > 2 else "20000"
TOP_K = int(sys.argv[3]) if len(sys.argv) > 3 else 75
N_WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else (os.cpu_count() or 8)


def _load_index(country, channel):
    d = f"{INDEX_ROOT}/{SOURCE}/{country}/{channel}"
    retriever = bm25s.BM25.load(d, mmap=True, load_corpus=False, show_progress=False)
    ids = np.load(f"{d}/entity_ids.npy", allow_pickle=True)
    return retriever, ids


def _shard_path(shard_dir, shard_id):
    return f"{shard_dir}/shard_{shard_id:03d}.parquet"


def _query_shard(args):
    shard_id, s1_shard, shard_dir = args
    out_path = _shard_path(shard_dir, shard_id)
    if os.path.exists(out_path):
        print(f"[shard {shard_id}] already complete on disk, skipping -> {out_path}", flush=True)
        return out_path
    parts = []
    for country in sorted(s1_shard["country"].unique()):
        sub = s1_shard[s1_shard["country"] == country]
        if len(sub) == 0:
            continue
        name_retriever, name_ids = _load_index(country, "name")
        addr_retriever, addr_ids = _load_index(country, "address")

        name_tokens = [char_ngrams(t) for t in sub["name_normalized"]]
        addr_tokens = [char_ngrams(t) for t in sub["address_normalized"]]
        qids = sub["entity_id"].to_numpy()

        k_name = min(TOP_K, len(name_ids))
        k_addr = min(TOP_K, len(addr_ids))
        name_res, name_scores = name_retriever.retrieve(name_tokens, corpus=name_ids, k=k_name,
                                                          n_threads=1, show_progress=False)
        addr_res, addr_scores = addr_retriever.retrieve(addr_tokens, corpus=addr_ids, k=k_addr,
                                                          n_threads=1, show_progress=False)

        for i, qid in enumerate(qids):
            cand = {}  # oid -> {name_score, name_rank, addr_score, addr_rank}
            for rank, (oid, sc) in enumerate(zip(name_res[i].tolist(), name_scores[i].tolist()), 1):
                cand.setdefault(oid, {})["name_score"] = sc
                cand[oid]["name_rank"] = rank
            for rank, (oid, sc) in enumerate(zip(addr_res[i].tolist(), addr_scores[i].tolist()), 1):
                cand.setdefault(oid, {})["addr_score"] = sc
                cand[oid]["addr_rank"] = rank
            for d in cand.values():
                rrf = 0.0
                if "name_rank" in d:
                    rrf += 1.0 / (RRF_K + d["name_rank"])
                if "addr_rank" in d:
                    rrf += 1.0 / (RRF_K + d["addr_rank"])
                d["rrf_score"] = rrf
            top_ids = sorted(cand, key=lambda o: -cand[o]["rrf_score"])[:TOP_K]
            for oid in top_ids:
                d = cand[oid]
                parts.append((qid, oid, d.get("name_score"), d.get("name_rank"),
                              d.get("addr_score"), d.get("addr_rank"), d["rrf_score"]))
    df = pd.DataFrame(parts, columns=["source1_entity_id", "candidate_entity_id",
                                       "name_score", "name_rank", "addr_score", "addr_rank", "rrf_score"])
    # write-to-temp-then-rename: a walltime kill mid-write leaves only the .tmp
    # file behind, never a corrupt/partial file at the real path, so a resumed
    # run's os.path.exists() check above is always trustworthy.
    tmp_path = out_path + ".tmp"
    df.to_parquet(tmp_path)
    os.replace(tmp_path, out_path)
    print(f"[shard {shard_id}] {len(s1_shard):,} queries done -> {len(parts):,} candidate rows, saved to {out_path}",
          flush=True)
    return out_path


def recall_of(cand_df, true_pairs, label):
    merged = true_pairs.merge(cand_df, on=["source1_entity_id", "candidate_entity_id"], how="left", indicator=True)
    found = (merged["_merge"] == "both").sum()
    total = len(true_pairs)
    per_entity = merged.groupby("source1_entity_id")["_merge"].apply(lambda s: (s == "both").all())
    print(f"[{label}] pair recall: {found}/{total} = {found/max(total,1):.4f} | "
          f"entities with ALL true matches recovered: {per_entity.mean():.4f} | candidate pairs: {len(cand_df):,}")


def main():
    t0 = time.time()
    s1_full = pd.read_parquet("data/processed/train_source1.parquet")
    print(f"loaded S1={len(s1_full):,} in {time.time()-t0:.1f}s, n_workers={N_WORKERS}, top_k={TOP_K}", flush=True)

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

    shard_dir = f"{INDEX_ROOT}/{SOURCE}_shards_n{len(s1_sample)}_w{N_WORKERS}"
    os.makedirs(shard_dir, exist_ok=True)
    print(f"shard checkpoint dir: {shard_dir} (already-complete shards are skipped on rerun)", flush=True)

    shard_ids = np.array_split(s1_sample.index.to_numpy(), N_WORKERS)
    shards = [(i, s1_sample.loc[idx], shard_dir) for i, idx in enumerate(shard_ids) if len(idx) > 0]

    ctx = get_context("fork")
    t0 = time.time()
    shard_paths = []
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        for path in ex.map(_query_shard, shards):
            shard_paths.append(path)
    elapsed = time.time() - t0
    print(f"\nall {len(shards)} shards done in {elapsed:.1f}s wall time", flush=True)

    cand = pd.concat([pd.read_parquet(p) for p in shard_paths], ignore_index=True)
    print(f"total candidate pairs: {len(cand):,}, avg/entity: {len(cand)/len(s1_sample):.1f}", flush=True)

    gt = pd.read_csv("data/student_resource/dataset/train/train_ground_truth.tsv", sep="\t",
                      dtype=str, keep_default_na=False)
    gt_sample = gt[gt["source1_entity_id"].isin(set(s1_sample["entity_id"]))]
    pairs = []
    prefix = SOURCE.upper().replace("S", "S") + "-"
    for row in gt_sample.itertuples(index=False):
        if row.matched_entity_ids:
            for mid in row.matched_entity_ids.split(","):
                if mid.startswith(prefix):
                    pairs.append((row.source1_entity_id, mid))
    true_pairs = pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])
    print(f"true pairs for sampled queries: {len(true_pairs)}", flush=True)
    recall_of(cand, true_pairs, f"Fused vs FULL {SOURCE.upper()}")

    per_query_s = elapsed / len(s1_sample)
    print(f"\nper-query wall time: {per_query_s*1000:.2f}ms")
    full_scale_estimate_hr = per_query_s * 2_206_821 / 3600
    print(f"naive full-S1-scale extrapolation at this rate: {full_scale_estimate_hr:.2f}h")

    out_path = f"data/processed/bm25_indexes/{SOURCE}_query_shard_result_n{len(s1_sample)}.parquet"
    cand.to_parquet(out_path)
    print(f"saved candidates to {out_path}", flush=True)


if __name__ == "__main__":
    main()
