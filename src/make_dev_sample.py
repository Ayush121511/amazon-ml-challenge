"""Build a small, representative dev sample of the training data for fast
local iteration on blocking/features/model, instead of paying the full-scale
HPC round-trip for every experiment.

Two things a generic random subsample would get wrong for entity resolution
specifically:
  1. Match integrity: if we include a Source-1 entity, we must include ALL
     of its true Source-2/3 matches too, or we invent false negatives that
     don't exist in the real data — corrupting any recall number we compute.
  2. Representativeness: EDA showed train/test have different country splits
     and a fixed ~5.6% singleton rate — the sample should preserve both via
     stratified sampling, not pure uniform random.

We also add a random "background" slice of S2/S3 (independent of who's
matched) so token/name collision *rates* stay structurally similar to full
scale — useful for sanity-checking blocking behavior, though absolute
collision counts still shrink with the sample and can't fully replace a
true full-scale check for the memory-safety caps in blocking.py.
"""
import os
import sys

import numpy as np
import pandas as pd

RAW = "data/student_resource/dataset"
OUT = "data/processed/dev_sample"
SEED = 42
S1_FRACTION = 0.03          # ~66k of 2.2M train S1 entities
BACKGROUND_FRACTION = 0.03  # independent random background from S2/S3


def stratified_sample_ids(s1_df, gt_df, fraction, seed):
    s1 = s1_df[["entity_id", "country"]].merge(
        gt_df[["source1_entity_id", "matched_entity_ids"]],
        left_on="entity_id", right_on="source1_entity_id", how="left",
    )
    s1["is_singleton"] = s1["matched_entity_ids"].fillna("") == ""
    sampled = s1.groupby(["country", "is_singleton"], group_keys=False).apply(
        lambda g: g.sample(frac=fraction, random_state=seed), include_groups=False
    )
    return set(sampled["entity_id"])


def make_test_sample(fraction, seed):
    print("\nloading raw test files...", flush=True)
    s1 = pd.read_csv(f"{RAW}/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{RAW}/test/test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{RAW}/test/test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    print(f"loaded S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}", flush=True)

    # No ground truth on test, so no match-integrity constraint here — just
    # stratify S1 by country (France included) and independently subsample
    # S2/S3 at the same rate so blocking has a realistic pool to search.
    s1_sample = s1.groupby("country", group_keys=False).apply(
        lambda g: g.sample(frac=fraction, random_state=seed), include_groups=False
    )
    s1_out = s1.loc[s1_sample.index]
    s2_out = s2.sample(frac=fraction, random_state=seed)
    s3_out = s3.sample(frac=fraction, random_state=seed)

    s1_out.to_csv(f"{OUT}/test_source1.tsv", sep="\t", index=False)
    s2_out.to_csv(f"{OUT}/test_source2.tsv", sep="\t", index=False)
    s3_out.to_csv(f"{OUT}/test_source3.tsv", sep="\t", index=False)
    print(f"wrote test dev sample: S1={len(s1_out):,} S2={len(s2_out):,} S3={len(s3_out):,}", flush=True)
    print(f"  country dist (S1): {s1_out['country'].value_counts().to_dict()}", flush=True)


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)

    print("loading raw train files...", flush=True)
    s1 = pd.read_csv(f"{RAW}/train/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{RAW}/train/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{RAW}/train/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(f"{RAW}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    print(f"loaded S1={len(s1):,} S2={len(s2):,} S3={len(s3):,} GT={len(gt):,}", flush=True)

    sampled_s1_ids = stratified_sample_ids(s1, gt, S1_FRACTION, SEED)
    print(f"stratified-sampled {len(sampled_s1_ids):,} S1 entities "
          f"({100*len(sampled_s1_ids)/len(s1):.2f}% of train)", flush=True)

    gt_sample = gt[gt["source1_entity_id"].isin(sampled_s1_ids)].copy()
    positive_ids = set()
    for ids in gt_sample["matched_entity_ids"]:
        if ids:
            positive_ids.update(ids.split(","))
    positive_s2 = {i for i in positive_ids if i.startswith("S2-")}
    positive_s3 = {i for i in positive_ids if i.startswith("S3-")}
    print(f"true matches to preserve: {len(positive_s2):,} S2 + {len(positive_s3):,} S3", flush=True)

    background_s2 = set(s2["entity_id"].sample(frac=BACKGROUND_FRACTION, random_state=SEED))
    background_s3 = set(s3["entity_id"].sample(frac=BACKGROUND_FRACTION, random_state=SEED))
    keep_s2 = positive_s2 | background_s2
    keep_s3 = positive_s3 | background_s3
    print(f"final S2 sample: {len(keep_s2):,} ({100*len(keep_s2)/len(s2):.2f}% of full) | "
          f"final S3 sample: {len(keep_s3):,} ({100*len(keep_s3)/len(s3):.2f}% of full)", flush=True)

    s1_out = s1[s1["entity_id"].isin(sampled_s1_ids)]
    s2_out = s2[s2["entity_id"].isin(keep_s2)]
    s3_out = s3[s3["entity_id"].isin(keep_s3)]

    s1_out.to_csv(f"{OUT}/train_source1.tsv", sep="\t", index=False)
    s2_out.to_csv(f"{OUT}/train_source2.tsv", sep="\t", index=False)
    s3_out.to_csv(f"{OUT}/train_source3.tsv", sep="\t", index=False)
    gt_sample.to_csv(f"{OUT}/train_ground_truth.tsv", sep="\t", index=False)

    print(f"\nwrote dev sample to {OUT}/:", flush=True)
    print(f"  S1={len(s1_out):,} S2={len(s2_out):,} S3={len(s3_out):,} GT rows={len(gt_sample):,}", flush=True)
    print(f"  country dist (S1): {s1_out['country'].value_counts().to_dict()}", flush=True)
    print(f"  singleton rate: {(gt_sample['matched_entity_ids'] == '').mean():.4f} "
          f"(full train: 0.0559)", flush=True)

    make_test_sample(S1_FRACTION, SEED)


if __name__ == "__main__":
    main()
