"""Intermediate-scale validation: fixed S1 sample vs FULL target pool, same
methodology the teammate used for their own benchmark (10k refs vs full 10.3M
target pool). This tells us real per-query cost at production corpus size,
instead of guessing from dev-sample-scale numbers.

Local parallelism: name-channel and address-channel BM25 (index build +
query) are independent per country, so run them as separate OS processes via
`fork` (copy-on-write, avoids re-pickling the multi-GB target DataFrame that
macOS's default `spawn` start method would force).
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")
from chunked_bm25 import bm25_topk_batched_queries, rrf_fuse

N_QUERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 75
QUERY_BATCH_SIZE = 20000
SEED = 42


def _channel_task(args):
    query_tokens, query_ids, other_df, text_field, top_k, label = args
    t0 = time.time()
    result = bm25_topk_batched_queries(query_tokens, query_ids, other_df, text_field,
                                        top_k, QUERY_BATCH_SIZE, label=label)
    return label, result, time.time() - t0


def build_candidates_fused_parallel(s1, other, top_k, label="fused"):
    ctx = get_context("fork")
    parts = []
    timings = {}
    countries = sorted(set(s1["country"]) & set(other["country"]))
    for country in countries:
        s1_sub = s1[s1["country"] == country]
        other_sub = other[other["country"] == country]
        if len(s1_sub) == 0 or len(other_sub) == 0:
            continue

        from bm25s_blocking import char_ngrams
        query_ids = s1_sub["entity_id"].to_numpy()
        name_tokens = [char_ngrams(t) for t in s1_sub["name_normalized"]]
        addr_tokens = [char_ngrams(t) for t in s1_sub["address_normalized"]]

        tasks = [
            (name_tokens, query_ids, other_sub, "name_normalized", top_k, f"{label}-name-{country}"),
            (addr_tokens, query_ids, other_sub, "address_normalized", top_k, f"{label}-addr-{country}"),
        ]
        print(f"[{country}] launching name+address channels in parallel: "
              f"{len(s1_sub):,} queries x {len(other_sub):,} targets", flush=True)

        with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
            results = list(ex.map(_channel_task, tasks))

        by_label = {lbl: (res, dt) for lbl, res, dt in results}
        name_topk, name_dt = by_label[f"{label}-name-{country}"]
        addr_topk, addr_dt = by_label[f"{label}-addr-{country}"]
        timings[country] = {"name_s": name_dt, "addr_s": addr_dt}
        print(f"[{country}] name channel: {name_dt:.1f}s | address channel: {addr_dt:.1f}s "
              f"(ran concurrently, wall time ~= max of the two)", flush=True)

        fused = rrf_fuse(name_topk, addr_topk, top_k)
        for qid, ranked_ids in fused.items():
            for oid in ranked_ids:
                parts.append((qid, oid))

    df = pd.DataFrame(parts, columns=["source1_entity_id", "candidate_entity_id"]).drop_duplicates()
    return df, timings


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
    s2_full = pd.read_parquet("data/processed/train_source2.parquet")
    print(f"loaded full S1={len(s1_full):,} S2={len(s2_full):,} in {time.time()-t0:.1f}s", flush=True)

    s1_sample = s1_full.groupby("country", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), max(1, round(N_QUERIES * len(g) / len(s1_full)))), random_state=SEED)
    ).reset_index(drop=True)
    print(f"S1 sample: {len(s1_sample):,} queries (stratified by country) vs FULL S2 pool {len(s2_full):,}", flush=True)
    print(s1_sample["country"].value_counts().to_dict(), flush=True)

    gt = pd.read_csv("data/student_resource/dataset/train/train_ground_truth.tsv", sep="\t",
                      dtype=str, keep_default_na=False)
    gt_sample = gt[gt["source1_entity_id"].isin(set(s1_sample["entity_id"]))]
    pairs = []
    for row in gt_sample.itertuples(index=False):
        if row.matched_entity_ids:
            for mid in row.matched_entity_ids.split(","):
                if mid.startswith("S2-"):
                    pairs.append((row.source1_entity_id, mid))
    true_s2 = pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])
    print(f"true S2 pairs for sampled queries: {len(true_s2)}", flush=True)

    t0 = time.time()
    cand2, timings = build_candidates_fused_parallel(s1_sample, s2_full, top_k=TOP_K, label="interm")
    elapsed = time.time() - t0
    print(f"\nbuilt in {elapsed:.1f}s total wall time", flush=True)
    recall_of(cand2, true_s2, "Fused vs FULL S2")
    print(f"avg candidates per S1: {len(cand2)/len(s1_sample):.1f}")

    per_query_s = elapsed / len(s1_sample)
    print(f"\nper-query wall time: {per_query_s*1000:.2f}ms")
    full_scale_estimate_hr = per_query_s * 2_206_821 / 3600
    print(f"naive full-S1-scale extrapolation (this per-query rate x 2,206,821): {full_scale_estimate_hr:.1f}h")


if __name__ == "__main__":
    main()
