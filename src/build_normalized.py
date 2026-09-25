"""Preprocess raw source TSVs into normalized parquet files for blocking/features.

Run once per source file; output cached in data/processed/ so downstream
blocking/feature/training scripts don't re-pay tokenization cost.
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_address, normalize_name

RAW_DIR = "data/student_resource/dataset"
OUT_DIR = "data/processed"

SOURCES = {
    "train_source1": f"{RAW_DIR}/train/train_source1.tsv",
    "train_source2": f"{RAW_DIR}/train/train_source2.tsv",
    "train_source3": f"{RAW_DIR}/train/train_source3.tsv",
    "test_source1": f"{RAW_DIR}/test/test_source1.tsv",
    "test_source2": f"{RAW_DIR}/test/test_source2.tsv",
    "test_source3": f"{RAW_DIR}/test/test_source3.tsv",
}


def build_one(name, path):
    out_path = f"{OUT_DIR}/{name}.parquet"
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return

    t0 = time.time()
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    print(f"[{name}] read {len(df)} rows in {time.time()-t0:.1f}s")

    t0 = time.time()
    names = df["business_name"].apply(normalize_name)
    addrs = df["business_address"].apply(normalize_address)
    print(f"[{name}] normalized in {time.time()-t0:.1f}s")

    out = pd.DataFrame({
        "entity_id": df["entity_id"],
        "country": df["country"],
        "name_normalized": [d["normalized"] for d in names],
        "name_core": [d["core"] for d in names],
        "name_compact": [d["compact"] for d in names],
        "name_first_token": [d["core_tokens"][0] if d["core_tokens"] else "" for d in names],
        "address_normalized": [d["normalized"] for d in addrs],
        "address_core": [" ".join(d["core_tokens"]) for d in addrs],
        "address_postal": [d["postal"] for d in addrs],
        "address_numbers": ["|".join(sorted(d["numbers"])) for d in addrs],
    })

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[{name}] wrote {out_path} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    only = sys.argv[1:] or list(SOURCES.keys())
    for name in only:
        build_one(name, SOURCES[name])
