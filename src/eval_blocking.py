"""Measure blocking recall on the training set: for each blocking strategy
(and their union), what fraction of true S2/S3 matches survive into the
candidate set? This determines the recall ceiling for everything downstream.
"""
import sys
import time

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)  # visible progress even when redirected to a file (PBS job log)

sys.path.insert(0, "src")
import blocking

PROC = "data/processed"


def load_ground_truth():
    gt = pd.read_csv(
        f"data/student_resource/dataset/train/train_ground_truth.tsv",
        sep="\t", dtype=str, keep_default_na=False,
    )
    pairs = []
    for row in gt.itertuples(index=False):
        s1 = row.source1_entity_id
        if row.matched_entity_ids:
            for mid in row.matched_entity_ids.split(","):
                pairs.append((s1, mid))
    true_pairs = pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])
    n_s1_with_matches = gt[gt["matched_entity_ids"] != ""]["source1_entity_id"].nunique()
    n_singletons = gt[gt["matched_entity_ids"] == ""]["source1_entity_id"].nunique()
    return true_pairs, n_s1_with_matches, n_singletons


def recall_of(cand_df, true_pairs, label):
    merged = true_pairs.merge(cand_df, on=["source1_entity_id", "candidate_entity_id"], how="left", indicator=True)
    found = (merged["_merge"] == "both").sum()
    total = len(true_pairs)
    per_entity = merged.groupby("source1_entity_id")["_merge"].apply(lambda s: (s == "both").all())
    full_recall_entities = per_entity.mean()
    print(f"[{label}] pair recall: {found}/{total} = {found/total:.4f} | "
          f"entities with ALL true matches recovered: {full_recall_entities:.4f} | "
          f"candidate pairs: {len(cand_df)}")
    return found / total


def main():
    s1 = pd.read_parquet(f"{PROC}/train_source1.parquet")
    s2 = pd.read_parquet(f"{PROC}/train_source2.parquet")
    s3 = pd.read_parquet(f"{PROC}/train_source3.parquet")
    print(f"S1={len(s1)} S2={len(s2)} S3={len(s3)}")

    true_pairs, n_matched, n_singleton = load_ground_truth()
    print(f"true pairs: {len(true_pairs)} | S1 with matches: {n_matched} | singletons: {n_singleton}")
    true_s2 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S2-")]
    true_s3 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S3-")]

    for name, fn in [
        ("compact_name", blocking.compact_name_candidates),
        ("postal_country", blocking.postal_country_candidates),
        ("first_token_country", blocking.first_token_country_candidates),
    ]:
        t0 = time.time()
        c2 = fn(s1, s2)
        c3 = fn(s1, s3)
        recall_of(c2, true_s2, f"{name} vs S2")
        recall_of(c3, true_s3, f"{name} vs S3")
        print(f"  ({time.time()-t0:.1f}s)")

    print("\n--- token blocking ---")
    t0 = time.time()
    tok2 = blocking.token_candidates(s1, s2)
    print(f"token vs S2 built in {time.time()-t0:.1f}s ({len(tok2):,} pairs)")
    recall_of(tok2, true_s2, "token vs S2")
    t0 = time.time()
    tok3 = blocking.token_candidates(s1, s3)
    print(f"token vs S3 built in {time.time()-t0:.1f}s ({len(tok3):,} pairs)")
    recall_of(tok3, true_s3, "token vs S3")

    print("\n--- union of all strategies (reusing token candidates above) ---")
    t0 = time.time()
    union2 = pd.concat([
        blocking.compact_name_candidates(s1, s2),
        blocking.postal_country_candidates(s1, s2),
        blocking.first_token_country_candidates(s1, s2),
        tok2,
    ], ignore_index=True).drop_duplicates()
    union3 = pd.concat([
        blocking.compact_name_candidates(s1, s3),
        blocking.postal_country_candidates(s1, s3),
        blocking.first_token_country_candidates(s1, s3),
        tok3,
    ], ignore_index=True).drop_duplicates()
    print(f"union built in {time.time()-t0:.1f}s")
    recall_of(union2, true_s2, "UNION vs S2")
    recall_of(union3, true_s3, "UNION vs S3")
    print(f"avg candidates per S1 (S2): {len(union2)/len(s1):.1f}")
    print(f"avg candidates per S1 (S3): {len(union3)/len(s1):.1f}")


if __name__ == "__main__":
    main()
