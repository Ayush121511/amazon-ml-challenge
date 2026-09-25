"""Fast local smoke test: run normalization + all blocking strategies + recall
measurement against the small dev sample (data/processed/dev_sample/),
instead of paying the full-scale HPC round-trip for every code change.
"""
import sys
import time

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")
from normalize import normalize_address, normalize_name
import blocking

DEV = "data/processed/dev_sample"


def build(df):
    names = df["business_name"].apply(normalize_name)
    addrs = df["business_address"].apply(normalize_address)
    return pd.DataFrame({
        "entity_id": df["entity_id"], "country": df["country"],
        "name_normalized": [d["normalized"] for d in names],
        "name_core": [d["core"] for d in names],
        "name_compact": [d["compact"] for d in names],
        "name_first_token": [d["core_tokens"][0] if d["core_tokens"] else "" for d in names],
        "address_normalized": [d["normalized"] for d in addrs],
        "address_postal": [d["postal"] for d in addrs],
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

    pairs = []
    for row in gt.itertuples(index=False):
        if row.matched_entity_ids:
            for mid in row.matched_entity_ids.split(","):
                pairs.append((row.source1_entity_id, mid))
    true_pairs = pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])
    true_s2 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S2-")]
    true_s3 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S3-")]
    print(f"true pairs: {len(true_pairs)}")

    for name, fn in [
        ("compact_name", blocking.compact_name_candidates),
        ("postal_country", blocking.postal_country_candidates),
        ("first_token_country", blocking.first_token_country_candidates),
        ("token", blocking.token_candidates),
    ]:
        t0 = time.time()
        c2 = fn(s1, s2)
        c3 = fn(s1, s3)
        recall_of(c2, true_s2, f"{name} vs S2")
        recall_of(c3, true_s3, f"{name} vs S3")
        print(f"  ({time.time()-t0:.1f}s)")

    print("\n--- union ---")
    t0 = time.time()
    union2 = blocking.generate_candidates(s1, s2)
    union3 = blocking.generate_candidates(s1, s3)
    print(f"union built in {time.time()-t0:.1f}s")
    recall_of(union2, true_s2, "UNION vs S2")
    recall_of(union3, true_s3, "UNION vs S3")
    print(f"avg candidates per S1 (S2): {len(union2)/len(s1):.1f}")
    print(f"avg candidates per S1 (S3): {len(union3)/len(s1):.1f}")


if __name__ == "__main__":
    main()
