"""Full-scale TF-IDF + SimHash (Random Projection LSH) blocking, with the
pre-check gate on every hash table before it merges. Measures recall against
train_ground_truth.tsv the same way eval_blocking.py does for the exact-match
blockers, so the two are directly comparable.

Reuses the already-normalized parquet files from stage 1 (name_normalized,
country, entity_id) — no re-normalization needed.
"""
import sys
import time

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")
import simhash_blocking

PROC = "data/processed"


def recall_of(cand_df, true_pairs, label):
    merged = true_pairs.merge(cand_df, on=["source1_entity_id", "candidate_entity_id"], how="left", indicator=True)
    found = (merged["_merge"] == "both").sum()
    total = len(true_pairs)
    per_entity = merged.groupby("source1_entity_id")["_merge"].apply(lambda s: (s == "both").all())
    print(f"[{label}] pair recall: {found}/{total} = {found/max(total,1):.4f} | "
          f"entities with ALL true matches recovered: {per_entity.mean():.4f} | candidate pairs: {len(cand_df):,}")


def main():
    t0 = time.time()
    s1 = pd.read_parquet(f"{PROC}/train_source1.parquet")[["entity_id", "country", "name_normalized"]]
    s2 = pd.read_parquet(f"{PROC}/train_source2.parquet")[["entity_id", "country", "name_normalized"]]
    s3 = pd.read_parquet(f"{PROC}/train_source3.parquet")[["entity_id", "country", "name_normalized"]]
    print(f"loaded S1={len(s1):,} S2={len(s2):,} S3={len(s3):,} in {time.time()-t0:.1f}s")

    t0 = time.time()
    vec = simhash_blocking.fit_vectorizer(s1["name_normalized"], s2["name_normalized"], s3["name_normalized"])
    print(f"fit TF-IDF vectorizer: vocab size = {len(vec.vocabulary_):,} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    s1_vecs = vec.transform(s1["name_normalized"])
    s2_vecs = vec.transform(s2["name_normalized"])
    s3_vecs = vec.transform(s3["name_normalized"])
    print(f"transformed all sources in {time.time()-t0:.1f}s")

    t0 = time.time()
    gt = pd.read_csv("data/student_resource/dataset/train/train_ground_truth.tsv",
                      sep="\t", dtype=str, keep_default_na=False)
    pairs = []
    for row in gt.itertuples(index=False):
        if row.matched_entity_ids:
            for mid in row.matched_entity_ids.split(","):
                pairs.append((row.source1_entity_id, mid))
    true_pairs = pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])
    true_s2 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S2-")]
    true_s3 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S3-")]
    print(f"loaded ground truth: {len(true_pairs):,} true pairs ({time.time()-t0:.1f}s)")

    print("\n--- SimHash vs S2 (full scale) ---")
    t0 = time.time()
    cand2 = simhash_blocking.build_candidates(s1, s2, s1_vecs, s2_vecs, label="simhash-S2")
    print(f"built in {time.time()-t0:.1f}s")
    recall_of(cand2, true_s2, "SimHash vs S2")

    print("\n--- SimHash vs S3 (full scale) ---")
    t0 = time.time()
    cand3 = simhash_blocking.build_candidates(s1, s3, s1_vecs, s3_vecs, label="simhash-S3")
    print(f"built in {time.time()-t0:.1f}s")
    recall_of(cand3, true_s3, "SimHash vs S3")

    print(f"\navg candidates per S1 (S2): {len(cand2)/len(s1):.1f}")
    print(f"avg candidates per S1 (S3): {len(cand3)/len(s1):.1f}")


if __name__ == "__main__":
    main()
