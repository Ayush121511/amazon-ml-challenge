"""A/B test: does applying Vishal's offline transliteration (Devanagari/Indic
-> Latin, via Unicode character names, no external lookup) to the name field
before BM25 tokenizing improve recall, especially for India (where our EDA
found the Devanagari 'Private Limited' tokenization artifact)? transliterate()
is a no-op on pure-ASCII text, so this should only ever help, never hurt US.
"""
import sys
import time

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")
from normalize import normalize_address, normalize_name
from transliterate import transliterate
import chunked_bm25 as cb

DEV = "data/processed/dev_sample"
TOP_K = 75


def build(df, use_translit):
    names = df["business_name"].apply(normalize_name)
    addrs = df["business_address"].apply(normalize_address)
    name_field = [d["normalized"] for d in names]
    if use_translit:
        name_field = [transliterate(t) for t in name_field]
    return pd.DataFrame({
        "entity_id": df["entity_id"], "country": df["country"],
        "name_normalized": name_field,
        "address_normalized": [d["normalized"] for d in addrs],
    })


def recall_of(cand_df, true_pairs, label, s1_countries):
    merged = true_pairs.merge(cand_df, on=["source1_entity_id", "candidate_entity_id"], how="left", indicator=True)
    merged = merged.merge(s1_countries, left_on="source1_entity_id", right_on="entity_id", how="left")
    found = (merged["_merge"] == "both").sum()
    total = len(true_pairs)
    print(f"[{label}] overall pair recall: {found}/{total} = {found/max(total,1):.4f}")
    for country, g in merged.groupby("country"):
        f = (g["_merge"] == "both").sum()
        t = len(g)
        print(f"  [{label}] {country}: {f}/{t} = {f/max(t,1):.4f}")


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
    s1_countries = s1_raw[["entity_id", "country"]]
    print(f"true S2 pairs: {len(true_s2)}")

    for use_translit in (False, True):
        label = "with-translit" if use_translit else "baseline"
        s1 = build(s1_raw, use_translit)
        s2 = build(s2_raw, use_translit)
        t0 = time.time()
        cand2 = cb.build_candidates_fused(s1, s2, top_k=TOP_K, query_batch_size=20000, label=label)
        print(f"\n--- {label} --- built in {time.time()-t0:.1f}s")
        recall_of(cand2, true_s2, label, s1_countries)


if __name__ == "__main__":
    main()
