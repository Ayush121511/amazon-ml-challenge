"""Test MinHash + LSH blocking against the dev sample, same measurement
pattern as dev_pipeline_check.py / dev_simhash_check.py for direct comparison.
"""
import sys
import time

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")
from normalize import normalize_name
import minhash_lsh_blocking as mh

DEV = "data/processed/dev_sample"


def build(df):
    names = df["business_name"].apply(normalize_name)
    return pd.DataFrame({
        "entity_id": df["entity_id"], "country": df["country"],
        "name_normalized": [d["normalized"] for d in names],
    })


def recall_of(cand_df, true_pairs, label):
    merged = true_pairs.merge(cand_df, on=["source1_entity_id", "candidate_entity_id"], how="left", indicator=True)
    found = (merged["_merge"] == "both").sum()
    total = len(true_pairs)
    per_entity = merged.groupby("source1_entity_id")["_merge"].apply(lambda s: (s == "both").all())
    print(f"[{label}] pair recall: {found}/{total} = {found/max(total,1):.4f} | "
          f"entities with ALL true matches recovered: {per_entity.mean():.4f} | candidate pairs: {len(cand_df):,}")


def main():
    t0 = time.time()
    s1_raw = pd.read_csv(f"{DEV}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2_raw = pd.read_csv(f"{DEV}/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3_raw = pd.read_csv(f"{DEV}/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(f"{DEV}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    print(f"loaded dev sample S1={len(s1_raw)} S2={len(s2_raw)} S3={len(s3_raw)} in {time.time()-t0:.1f}s")

    t0 = time.time()
    s1, s2, s3 = build(s1_raw), build(s2_raw), build(s3_raw)
    print(f"normalized in {time.time()-t0:.1f}s")

    t0 = time.time()
    vec = mh.fit_shingle_vectorizer(s1["name_normalized"], s2["name_normalized"], s3["name_normalized"])
    print(f"fit shingle vectorizer: vocab size = {len(vec.vocabulary_):,} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    s1_shingles = vec.transform(s1["name_normalized"])
    s2_shingles = vec.transform(s2["name_normalized"])
    s3_shingles = vec.transform(s3["name_normalized"])
    print(f"transformed all sources in {time.time()-t0:.1f}s")

    t0 = time.time()
    s1_sig = mh.compute_minhash_signatures(s1_shingles)
    s2_sig = mh.compute_minhash_signatures(s2_shingles)
    s3_sig = mh.compute_minhash_signatures(s3_shingles)
    print(f"computed MinHash signatures ({mh.NUM_HASHES} hashes) in {time.time()-t0:.1f}s")

    s1_buckets = mh.lsh_bucket_ids(s1_sig)
    s2_buckets = mh.lsh_bucket_ids(s2_sig)
    s3_buckets = mh.lsh_bucket_ids(s3_sig)
    print(f"LSH bands: {mh.NUM_BANDS} bands x {mh.ROWS_PER_BAND} rows/band")

    pairs = []
    for row in gt.itertuples(index=False):
        if row.matched_entity_ids:
            for mid in row.matched_entity_ids.split(","):
                pairs.append((row.source1_entity_id, mid))
    true_pairs = pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])
    true_s2 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S2-")]
    true_s3 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S3-")]
    print(f"true pairs: {len(true_pairs)}")

    print("\n--- MinHash+LSH vs S2 ---")
    t0 = time.time()
    cand2 = mh.build_candidates(s1, s2, s1_buckets, s2_buckets, label="minhash-S2")
    print(f"built in {time.time()-t0:.1f}s")
    recall_of(cand2, true_s2, "MinHash+LSH vs S2")

    print("\n--- MinHash+LSH vs S3 ---")
    t0 = time.time()
    cand3 = mh.build_candidates(s1, s3, s1_buckets, s3_buckets, label="minhash-S3")
    print(f"built in {time.time()-t0:.1f}s")
    recall_of(cand3, true_s3, "MinHash+LSH vs S3")

    print(f"\navg candidates per S1 (S2): {len(cand2)/len(s1):.1f}")
    print(f"avg candidates per S1 (S3): {len(cand3)/len(s1):.1f}")


if __name__ == "__main__":
    main()
