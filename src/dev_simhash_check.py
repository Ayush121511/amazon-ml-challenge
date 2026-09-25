"""Test TF-IDF + SimHash blocking against the dev sample, measuring recall
the same way dev_pipeline_check.py does for the exact-match blockers, so the
two approaches are directly comparable.
"""
import sys
import time

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")
from normalize import normalize_address, normalize_name
import simhash_blocking

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
    vec = simhash_blocking.fit_vectorizer(s1["name_normalized"], s2["name_normalized"], s3["name_normalized"])
    print(f"fit TF-IDF vectorizer: vocab size = {len(vec.vocabulary_):,} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    s1_vecs = vec.transform(s1["name_normalized"])
    s2_vecs = vec.transform(s2["name_normalized"])
    s3_vecs = vec.transform(s3["name_normalized"])
    print(f"transformed all sources in {time.time()-t0:.1f}s")

    pairs = []
    for row in gt.itertuples(index=False):
        if row.matched_entity_ids:
            for mid in row.matched_entity_ids.split(","):
                pairs.append((row.source1_entity_id, mid))
    true_pairs = pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])
    true_s2 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S2-")]
    true_s3 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S3-")]
    print(f"true pairs: {len(true_pairs)}")

    print("\n--- SimHash vs S2 ---")
    t0 = time.time()
    cand2 = simhash_blocking.build_candidates(s1, s2, s1_vecs, s2_vecs, label="simhash-S2")
    print(f"built in {time.time()-t0:.1f}s")
    recall_of(cand2, true_s2, "SimHash vs S2")

    print("\n--- SimHash vs S3 ---")
    t0 = time.time()
    cand3 = simhash_blocking.build_candidates(s1, s3, s1_vecs, s3_vecs, label="simhash-S3")
    print(f"built in {time.time()-t0:.1f}s")
    recall_of(cand3, true_s3, "SimHash vs S3")

    print(f"\navg candidates per S1 (S2): {len(cand2)/len(s1):.1f}")
    print(f"avg candidates per S1 (S3): {len(cand3)/len(s1):.1f}")


if __name__ == "__main__":
    main()
