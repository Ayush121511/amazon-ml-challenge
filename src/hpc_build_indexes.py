"""Build BM25 indexes ONCE per country/channel over a full target pool and
cache them to disk, so every later query job (intermediate validation, full
S1 sweep, sharded workers) loads instead of rebuilding. This is the
production "index once, query many" pattern applied across job boundaries,
not just within one process.
"""
import sys
import time

import bm25s
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")
from bm25s_blocking import char_ngrams

INDEX_ROOT = "data/processed/bm25_indexes"
CHANNELS = [("name", "name_normalized"), ("address", "address_normalized")]


def build_and_save(source, other_df):
    for country in sorted(other_df["country"].unique()):
        sub = other_df[other_df["country"] == country]
        for channel, field in CHANNELS:
            out_dir = f"{INDEX_ROOT}/{source}/{country}/{channel}"
            import os
            if os.path.exists(f"{out_dir}/params.index.json"):
                print(f"[skip] {out_dir} already cached", flush=True)
                continue
            t0 = time.time()
            tokens = [char_ngrams(t) for t in sub[field]]
            retriever = bm25s.BM25()
            retriever.index(tokens, show_progress=False)
            os.makedirs(out_dir, exist_ok=True)
            retriever.save(out_dir, show_progress=False)
            np.save(f"{out_dir}/entity_ids.npy", sub["entity_id"].to_numpy())
            print(f"[built] {source}/{country}/{channel}: {len(sub):,} rows in {time.time()-t0:.1f}s -> {out_dir}",
                  flush=True)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "s2"
    t0 = time.time()
    other = pd.read_parquet(f"data/processed/train_source{source[1]}.parquet")
    print(f"loaded train_source{source[1]} = {len(other):,} rows in {time.time()-t0:.1f}s", flush=True)
    build_and_save(source, other)
    print("done", flush=True)


if __name__ == "__main__":
    main()
