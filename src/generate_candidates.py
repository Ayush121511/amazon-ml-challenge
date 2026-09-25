"""Production candidate-generation pipeline: the union of every blocking
channel we've validated, turned into the actual candidate_pairs.tsv the
submission format requires.

Channels combined:
- Exact-match union (blocking.py): compact_name, postal_country,
  first_token_country, token — all frequency-capped, catches near-identical
  names cheaply.
- TF-IDF top-N cosine similarity (tfidf_topn_blocking.py): safe by
  construction (hard-capped at top_n candidates per S1 row regardless of data
  skew, unlike the exact-match keys which need MAX_KEY_FREQ/MAX_PAIR_PRODUCT
  caps), and catches typo/reordering matches the exact-match keys miss
  entirely (e.g. "Technolgies" vs "Technologies").

SimHash (simhash_blocking.py) is deliberately left out of this production
union for now: its full-scale recall was still "in progress" as of the last
progress log and it is a probabilistic approximation of the same cosine
signal TF-IDF top-N computes exactly. Re-add it here (as a third `parts`
entry) once full-scale numbers confirm it's worth the extra candidates.

Usage:
    python src/generate_candidates.py --split train --eval
    python src/generate_candidates.py --split test
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
import blocking
import simhash_blocking as sh  # TF-IDF vectorizer fitting, shared with tfidf_topn_blocking
import tfidf_topn_blocking as topn

PROC = "data/processed"
TOP_N = 75


def load_split(split):
    s1 = pd.read_parquet(f"{PROC}/{split}_source1.parquet")
    s2 = pd.read_parquet(f"{PROC}/{split}_source2.parquet")
    s3 = pd.read_parquet(f"{PROC}/{split}_source3.parquet")
    return s1, s2, s3


def build_final_candidates(s1, s2, s3, top_n=TOP_N, use_tokens=True):
    """Union of the exact-match blockers and TF-IDF top-N, per other-source."""
    t0 = time.time()
    exact2 = blocking.generate_candidates(s1, s2, use_tokens=use_tokens)
    exact3 = blocking.generate_candidates(s1, s3, use_tokens=use_tokens)
    print(f"exact-match union built in {time.time()-t0:.1f}s "
          f"(S2: {len(exact2):,} pairs, S3: {len(exact3):,} pairs)")

    t0 = time.time()
    vec = sh.fit_vectorizer(s1["name_normalized"], s2["name_normalized"], s3["name_normalized"])
    s1_vecs = vec.transform(s1["name_normalized"])
    s2_vecs = vec.transform(s2["name_normalized"])
    s3_vecs = vec.transform(s3["name_normalized"])
    print(f"TF-IDF vectorized (vocab={len(vec.vocabulary_):,}) in {time.time()-t0:.1f}s")

    t0 = time.time()
    topn2 = topn.build_candidates(s1, s2, s1_vecs, s2_vecs, top_n=top_n, label="tfidf-topn-S2")
    topn3 = topn.build_candidates(s1, s3, s1_vecs, s3_vecs, top_n=top_n, label="tfidf-topn-S3")
    print(f"TF-IDF top-{top_n} built in {time.time()-t0:.1f}s "
          f"(S2: {len(topn2):,} pairs, S3: {len(topn3):,} pairs)")

    cand2 = pd.concat([exact2, topn2], ignore_index=True).drop_duplicates()
    cand3 = pd.concat([exact3, topn3], ignore_index=True).drop_duplicates()
    return cand2, cand3, {"exact2": exact2, "exact3": exact3, "topn2": topn2, "topn3": topn3}


def write_candidate_pairs(s1, cand2, cand3, out_path):
    """One row per S1 entity (including zero-candidate rows), comma-joined
    S2+S3 candidate ids, matching the candidate_pairs.tsv spec exactly."""
    all_cands = pd.concat([cand2, cand3], ignore_index=True)
    grouped = all_cands.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda s: ",".join(sorted(set(s)))
    )
    out = pd.DataFrame({"source1_entity_id": s1["entity_id"]})
    out["candidate_entity_ids"] = out["source1_entity_id"].map(grouped).fillna("")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    n_with_cands = (out["candidate_entity_ids"] != "").sum()
    print(f"wrote {out_path}: {len(out):,} S1 entities, "
          f"{n_with_cands:,} with >=1 candidate ({n_with_cands/len(out):.4f})")


def load_train_ground_truth():
    gt = pd.read_csv(
        "data/student_resource/dataset/train/train_ground_truth.tsv",
        sep="\t", dtype=str, keep_default_na=False,
    )
    pairs = []
    for row in gt.itertuples(index=False):
        if row.matched_entity_ids:
            for mid in row.matched_entity_ids.split(","):
                pairs.append((row.source1_entity_id, mid))
    return pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])


def recall_of(cand_df, true_pairs, label):
    merged = true_pairs.merge(cand_df, on=["source1_entity_id", "candidate_entity_id"], how="left", indicator=True)
    found = (merged["_merge"] == "both").sum()
    total = len(true_pairs)
    per_entity = merged.groupby("source1_entity_id")["_merge"].apply(lambda s: (s == "both").all())
    print(f"[{label}] pair recall: {found}/{total} = {found/max(total,1):.4f} | "
          f"entities with ALL true matches recovered: {per_entity.mean():.4f} | "
          f"candidate pairs: {len(cand_df):,}")


def evaluate(cand2, cand3, channels):
    true_pairs = load_train_ground_truth()
    true_s2 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S2-")]
    true_s3 = true_pairs[true_pairs["candidate_entity_id"].str.startswith("S3-")]

    print("\n--- per-channel recall (for comparison) ---")
    recall_of(channels["exact2"], true_s2, "exact-match only vs S2")
    recall_of(channels["exact3"], true_s3, "exact-match only vs S3")
    recall_of(channels["topn2"], true_s2, "tfidf-topn only vs S2")
    recall_of(channels["topn3"], true_s3, "tfidf-topn only vs S3")

    print("\n--- combined (production) recall ---")
    recall_of(cand2, true_s2, "COMBINED vs S2")
    recall_of(cand3, true_s3, "COMBINED vs S3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval", action="store_true", help="print recall diagnostics (train only)")
    args = ap.parse_args()

    if args.eval and args.split != "train":
        ap.error("--eval requires --split train (no ground truth for test)")

    t0 = time.time()
    s1, s2, s3 = load_split(args.split)
    print(f"loaded {args.split}: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,} in {time.time()-t0:.1f}s")

    cand2, cand3, channels = build_final_candidates(s1, s2, s3, top_n=args.top_n)

    out_path = args.out or f"{PROC}/candidate_pairs_{args.split}.tsv"
    write_candidate_pairs(s1, cand2, cand3, out_path)

    if args.eval:
        evaluate(cand2, cand3, channels)


if __name__ == "__main__":
    main()
