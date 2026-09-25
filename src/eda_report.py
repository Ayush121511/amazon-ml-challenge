"""Basic EDA across all 6 source files + ground truth, written to a markdown
report. Reads the normalized parquet files produced by build_normalized.py
(stage 1) plus the raw ground truth TSV.
"""
import random
import sys
import time

import pandas as pd

PROC = "data/processed"
RAW = "data/student_resource/dataset"
OUT_PATH = "notebooks/eda_basic.md"

SOURCES = ["train_source1", "train_source2", "train_source3",
           "test_source1", "test_source2", "test_source3"]


def load_all():
    return {name: pd.read_parquet(f"{PROC}/{name}.parquet") for name in SOURCES}


def section_row_counts(dfs, lines):
    lines.append("## Row counts\n")
    lines.append("| File | Rows |")
    lines.append("| --- | ---: |")
    for name in SOURCES:
        lines.append(f"| {name} | {len(dfs[name]):,} |")
    lines.append("")


def section_country(dfs, lines):
    lines.append("## Country distribution\n")
    for name in SOURCES:
        counts = dfs[name]["country"].value_counts()
        lines.append(f"**{name}**")
        lines.append("")
        lines.append("| Country | Count | % |")
        lines.append("| --- | ---: | ---: |")
        total = len(dfs[name])
        for country, cnt in counts.items():
            lines.append(f"| {country} | {cnt:,} | {100*cnt/total:.1f}% |")
        lines.append("")


def section_missingness(dfs, lines):
    lines.append("## Missingness (empty after normalization)\n")
    lines.append("| File | Empty name % | Empty address % | Has postal % |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name in SOURCES:
        df = dfs[name]
        n = len(df)
        empty_name = (df["name_normalized"] == "").sum()
        empty_addr = (df["address_normalized"] == "").sum()
        has_postal = (df["address_postal"] != "").sum()
        lines.append(
            f"| {name} | {100*empty_name/n:.2f}% | {100*empty_addr/n:.2f}% | {100*has_postal/n:.2f}% |"
        )
    lines.append("")


def section_lengths(dfs, lines):
    lines.append("## Name / address length stats (token counts, core tokens)\n")
    lines.append("| File | Name tokens (mean) | Name tokens (median) | Address tokens (mean) | Address tokens (median) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name in SOURCES:
        df = dfs[name]
        name_tok_counts = df["name_core"].str.split().str.len().fillna(0)
        addr_tok_counts = df["address_core"].str.split().str.len().fillna(0)
        lines.append(
            f"| {name} | {name_tok_counts.mean():.2f} | {name_tok_counts.median():.0f} | "
            f"{addr_tok_counts.mean():.2f} | {addr_tok_counts.median():.0f} |"
        )
    lines.append("")


def section_duplicates(dfs, lines):
    lines.append("## Duplicate normalized names/addresses (within-file)\n")
    lines.append("Businesses sharing an identical normalized name/address within the same file — "
                  "indicates how much generic/chain naming exists (relevant to blocking precision).\n")
    lines.append("| File | Rows | Distinct names | Distinct addresses | Top repeated name (count) |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for name in SOURCES:
        df = dfs[name]
        nonempty_names = df[df["name_core"] != ""]["name_core"]
        vc = nonempty_names.value_counts()
        top = vc.index[0] if len(vc) else ""
        top_count = vc.iloc[0] if len(vc) else 0
        n_distinct_names = df["name_core"].nunique()
        n_distinct_addr = df["address_core"].nunique()
        lines.append(
            f"| {name} | {len(df):,} | {n_distinct_names:,} | {n_distinct_addr:,} | "
            f"`{top}` ({top_count}) |"
        )
    lines.append("")


def section_top_tokens(dfs, lines):
    lines.append("## Most frequent core-name tokens (train, S2+S3 combined)\n")
    lines.append("These are candidates to prune from token-based blocking (too common to be discriminative).\n")
    combined = pd.concat([dfs["train_source2"]["name_core"], dfs["train_source3"]["name_core"]])
    tokens = combined.str.split().explode()
    tokens = tokens[tokens.notna() & (tokens != "")]
    top = tokens.value_counts().head(25)
    lines.append("| Token | Count |")
    lines.append("| --- | ---: |")
    for tok, cnt in top.items():
        lines.append(f"| {tok} | {cnt:,} |")
    lines.append("")


def section_ground_truth(lines):
    lines.append("## Ground truth (train) match statistics\n")
    gt = pd.read_csv(f"{RAW}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    n = len(gt)
    is_singleton = gt["matched_entity_ids"] == ""
    n_singleton = is_singleton.sum()
    match_counts = gt.loc[~is_singleton, "matched_entity_ids"].str.count(",") + 1

    lines.append(f"- Total Source-1 entities: {n:,}")
    lines.append(f"- Singletons (no match): {n_singleton:,} ({100*n_singleton/n:.1f}%)")
    lines.append(f"- Avg matches per non-singleton: {match_counts.mean():.2f}")
    lines.append(f"- Median matches per non-singleton: {match_counts.median():.0f}")
    lines.append("")
    lines.append("| # matches | Count |")
    lines.append("| ---: | ---: |")
    vc = match_counts.value_counts().sort_index()
    for k, cnt in vc.items():
        lines.append(f"| {k} | {cnt:,} |")
    lines.append(f"| 0 (singleton) | {n_singleton:,} |")
    lines.append("")

    matched = gt[~is_singleton].copy()
    has_s2 = matched["matched_entity_ids"].str.contains("S2-")
    has_s3 = matched["matched_entity_ids"].str.contains("S3-")
    lines.append("### Match composition (non-singleton S1 entities)\n")
    lines.append(f"- Has at least one S2 match: {has_s2.sum():,} ({100*has_s2.mean():.1f}%)")
    lines.append(f"- Has at least one S3 match: {has_s3.sum():,} ({100*has_s3.mean():.1f}%)")
    lines.append(f"- Has both S2 and S3 matches: {(has_s2 & has_s3).sum():,} ({100*(has_s2 & has_s3).mean():.1f}%)")
    lines.append(f"- S2-only: {(has_s2 & ~has_s3).sum():,} | S3-only: {(~has_s2 & has_s3).sum():,}")
    lines.append("")
    return gt


def section_noise_examples(dfs, gt, lines, n_examples=12):
    lines.append("## Sample true-match pairs (illustrating noise patterns)\n")
    s1 = dfs["train_source1"].set_index("entity_id")
    s2 = dfs["train_source2"].set_index("entity_id")
    s3 = dfs["train_source3"].set_index("entity_id")

    matched = gt[gt["matched_entity_ids"] != ""]
    rng = random.Random(42)
    sample_rows = rng.sample(range(len(matched)), min(n_examples, len(matched)))

    lines.append("| S1 name | S1 address | Match name | Match address |")
    lines.append("| --- | --- | --- | --- |")
    shown = 0
    for idx in sample_rows:
        row = matched.iloc[idx]
        s1_id = row["source1_entity_id"]
        if s1_id not in s1.index:
            continue
        mids = row["matched_entity_ids"].split(",")
        mid = mids[0]
        table = s2 if mid.startswith("S2-") else s3
        if mid not in table.index:
            continue
        s1_row = s1.loc[s1_id]
        m_row = table.loc[mid]
        s1_name = str(s1_row["name_normalized"])[:40].replace("|", " ")
        s1_addr = str(s1_row["address_normalized"])[:50].replace("|", " ")
        m_name = str(m_row["name_normalized"])[:40].replace("|", " ")
        m_addr = str(m_row["address_normalized"])[:50].replace("|", " ")
        lines.append(f"| {s1_name} | {s1_addr} | {m_name} | {m_addr} |")
        shown += 1
    lines.append("")
    lines.append(f"({shown} example pairs shown, sampled with fixed seed)\n")


def main():
    t0 = time.time()
    print("loading processed parquet files...", flush=True)
    dfs = load_all()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    lines = ["# Basic EDA Report — Amazon ML Challenge 2026 (Business Entity Resolution)\n"]
    lines.append(f"_Generated by `src/eda_report.py` — {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")

    section_row_counts(dfs, lines)
    print("row counts done", flush=True)

    section_country(dfs, lines)
    print("country dist done", flush=True)

    section_missingness(dfs, lines)
    print("missingness done", flush=True)

    section_lengths(dfs, lines)
    print("lengths done", flush=True)

    section_duplicates(dfs, lines)
    print("duplicates done", flush=True)

    section_top_tokens(dfs, lines)
    print("top tokens done", flush=True)

    gt = section_ground_truth(lines)
    print("ground truth stats done", flush=True)

    section_noise_examples(dfs, gt, lines)
    print("noise examples done", flush=True)

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {OUT_PATH} ({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
