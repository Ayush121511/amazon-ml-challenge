"""A/B test: does narrowing char n-gram range from (2,4) to (3,4) speed up
BM25 fusion meaningfully, and what does it cost in recall? Fewer, longer
n-grams per document/query directly cuts token volume at both index-build
and query time -- this is a compute-cost lever, not a recall lever, since
recall is already strong (96-98% at full scale).
"""
import sys
import time

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")
from normalize import normalize_address, normalize_name
import bm25s_blocking as bm
import chunked_bm25 as cb

DEV = "data/processed/dev_sample"
TOP_K = 75


def build(df):
    names = df["business_name"].apply(normalize_name)
    addrs = df["business_address"].apply(normalize_address)
    return pd.DataFrame({
        "entity_id": df["entity_id"], "country": df["country"],
        "name_normalized": [d["normalized"] for d in names],
        "address_normalized": [d["normalized"] for d in addrs],
    })


def recall_of(cand_df, true_pairs, label):
    merged = true_pairs.merge(cand_df, on=["source1_entity_id", "candidate_entity_id"], how="left", indicator=True)
    found = (merged["_merge"] == "both").sum()
    total = len(true_pairs)
    per_entity = merged.groupby("source1_entity_id")["_merge"].apply(lambda s: (s == "both").all())
    print(f"[{label}] pair recall: {found}/{total} = {found/max(total,1):.4f} | "
          f"entities with ALL true matches recovered: {per_entity.mean():.4f}")


def main():
    s1_raw = pd.read_csv(f"{DEV}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2_raw = pd.read_csv(f"{DEV}/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(f"{DEV}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)

    pairs = []
    for row in gt.itertuples(index=False):
        if row.matched_entity_ids:
            for mid in row.matched_entity_ids.split(","):
                if mid.startswith("S2-"):
                    pairs.append((row.source1_entity_id, mid))
    true_s2 = pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])

    s1, s2 = build(s1_raw), build(s2_raw)

    for ngram_range in ((2, 4), (3, 4)):
        bm.NGRAM_RANGE = ngram_range
        import bm25s_blocking
        bm25s_blocking.NGRAM_RANGE = ngram_range

        def patched_char_ngrams(text, _r=ngram_range):
            tokens = []
            for word in text.split():
                padded = f" {word} "
                for n in range(_r[0], _r[1] + 1):
                    tokens.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
            return tokens if tokens else [""]

        cb.char_ngrams = patched_char_ngrams
        label = f"ngram{ngram_range}"
        t0 = time.time()
        cand2 = cb.build_candidates_fused(s1, s2, top_k=TOP_K, query_batch_size=20000, label=label)
        elapsed = time.time() - t0
        print(f"\n--- {label} --- built in {elapsed:.1f}s")
        recall_of(cand2, true_s2, label)


if __name__ == "__main__":
    main()
