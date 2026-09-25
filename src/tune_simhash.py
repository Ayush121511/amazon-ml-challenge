"""Sweep SimHash's two real hyperparameters (bits per hash table, number of
tables) on the dev sample. Vectorizes once, reuses the same TF-IDF vectors
for every combination — only the hashing/bucketing step changes.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")
from normalize import normalize_name
import simhash_blocking as sh
import blocking_precheck as precheck_mod

DEV = "data/processed/dev_sample"


def build(df):
    names = df["business_name"].apply(normalize_name)
    return pd.DataFrame({
        "entity_id": df["entity_id"], "country": df["country"],
        "name_normalized": [d["normalized"] for d in names],
    })


def recall_of(cand_df, true_pairs):
    merged = true_pairs.merge(cand_df, on=["source1_entity_id", "candidate_entity_id"], how="left", indicator=True)
    found = (merged["_merge"] == "both").sum()
    total = len(true_pairs)
    per_entity = merged.groupby("source1_entity_id")["_merge"].apply(lambda s: (s == "both").all())
    return found / max(total, 1), per_entity.mean()


def build_candidates_silent(s1, other, s1_vecs, other_vecs, num_tables, bits_per_table, seed=42):
    """Same logic as simhash_blocking.build_candidates but no per-table printing,
    and returns whether every table was safe."""
    projections = sh._random_projection_matrices(s1_vecs.shape[1], num_tables, bits_per_table, seed)
    s1_buckets = sh._bucket_ids(s1_vecs, projections)
    other_buckets = sh._bucket_ids(other_vecs, projections)

    parts = []
    n_unsafe = 0
    for t in range(num_tables):
        s1_t = pd.DataFrame({"entity_id": s1["entity_id"].values, "country": s1["country"].values, "bucket": s1_buckets[:, t]})
        other_t = pd.DataFrame({"entity_id": other["entity_id"].values, "country": other["country"].values, "bucket": other_buckets[:, t]})
        report = precheck_mod.precheck(s1_t, other_t, ["bucket", "country"])
        if not report["safe"]:
            n_unsafe += 1
            continue
        m = s1_t.merge(other_t, on=["bucket", "country"], suffixes=("_s1", ""))
        m = m[["entity_id_s1", "entity_id"]].rename(columns={"entity_id_s1": "source1_entity_id", "entity_id": "candidate_entity_id"})
        parts.append(m)

    if not parts:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"]), n_unsafe
    return pd.concat(parts, ignore_index=True).drop_duplicates(), n_unsafe


def main():
    s1_raw = pd.read_csv(f"{DEV}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2_raw = pd.read_csv(f"{DEV}/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3_raw = pd.read_csv(f"{DEV}/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(f"{DEV}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1, s2, s3 = build(s1_raw), build(s2_raw), build(s3_raw)

    vec = sh.fit_vectorizer(s1["name_normalized"], s2["name_normalized"], s3["name_normalized"])
    s1_vecs, s2_vecs, s3_vecs = vec.transform(s1["name_normalized"]), vec.transform(s2["name_normalized"]), vec.transform(s3["name_normalized"])
    print(f"vectorized once, vocab={len(vec.vocabulary_):,}\n")

    pairs = []
    for row in gt.itertuples(index=False):
        if row.matched_entity_ids:
            for mid in row.matched_entity_ids.split(","):
                pairs.append((row.source1_entity_id, mid))
    true_pairs = pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])
    true_s2 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S2-")]
    true_s3 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S3-")]

    bits_options = [10, 12, 14, 16, 20, 24]
    table_options = [4, 6, 8, 12]

    print(f"{'bits':>5} {'tables':>7} {'recall_S2':>10} {'entfull_S2':>11} {'cand/ent_S2':>12} {'recall_S3':>10} {'entfull_S3':>11} {'cand/ent_S3':>12} {'unsafe_tbl':>11} {'time_s':>8}")
    results = []
    for bits in bits_options:
        for tables in table_options:
            t0 = time.time()
            c2, unsafe2 = build_candidates_silent(s1, s2, s1_vecs, s2_vecs, tables, bits)
            c3, unsafe3 = build_candidates_silent(s1, s3, s1_vecs, s3_vecs, tables, bits)
            dt = time.time() - t0
            r2, e2 = recall_of(c2, true_s2)
            r3, e3 = recall_of(c3, true_s3)
            cand_per_ent2 = len(c2) / len(s1)
            cand_per_ent3 = len(c3) / len(s1)
            unsafe_total = unsafe2 + unsafe3
            print(f"{bits:>5} {tables:>7} {r2:>10.4f} {e2:>11.4f} {cand_per_ent2:>12.1f} {r3:>10.4f} {e3:>11.4f} {cand_per_ent3:>12.1f} {unsafe_total:>11} {dt:>8.2f}")
            results.append(dict(bits=bits, tables=tables, recall_s2=r2, entfull_s2=e2, cand_s2=cand_per_ent2,
                                 recall_s3=r3, entfull_s3=e3, cand_s3=cand_per_ent3, unsafe=unsafe_total, time=dt))

    df = pd.DataFrame(results)
    df["avg_recall"] = (df.recall_s2 + df.recall_s3) / 2
    print("\nTop 5 by avg recall (with 0 unsafe tables):")
    safe = df[df.unsafe == 0].sort_values("avg_recall", ascending=False)
    print(safe.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
