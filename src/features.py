"""Pairwise similarity features between a Source-1 entity and a candidate
Source-2/3 record. Built on merged columns (not per-row python objects) so
it scales to tens of millions of candidate pairs; rapidfuzz does ~1.3M
string-pair scores/sec so an explicit loop over aligned numpy arrays is fine.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

FEATURE_COLUMNS = [
    "country_match",
    "name_compact_exact",
    "name_fuzz_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_core_jaccard",
    "address_fuzz_ratio",
    "address_token_sort_ratio",
    "address_numbers_jaccard",
    "postal_match",
    "postal_both_present",
    "first_token_match",
]


def _jaccard(a, b):
    """Token-set Jaccard for two space-separated strings; empty-vs-empty -> 0."""
    sa = set(a.split()) if a else set()
    sb = set(b.split()) if b else set()
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _numbers_jaccard(a, b):
    sa = set(a.split("|")) if a else set()
    sb = set(b.split("|")) if b else set()
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_pair_table(cand_df, s1_df, other_df):
    """Merge candidate pairs with s1/other normalized fields, prefixed s1_/o_."""
    s1_cols = s1_df.add_prefix("s1_").rename(columns={"s1_entity_id": "source1_entity_id"})
    o_cols = other_df.add_prefix("o_").rename(columns={"o_entity_id": "candidate_entity_id"})
    merged = cand_df.merge(s1_cols, on="source1_entity_id", how="left")
    merged = merged.merge(o_cols, on="candidate_entity_id", how="left")
    return merged


def compute_features(pair_table):
    """Given a merged pair table (see build_pair_table), return a features DataFrame."""
    n = len(pair_table)
    s1_name = pair_table["s1_name_normalized"].to_numpy()
    o_name = pair_table["o_name_normalized"].to_numpy()
    s1_addr = pair_table["s1_address_normalized"].to_numpy()
    o_addr = pair_table["o_address_normalized"].to_numpy()

    name_fuzz = np.empty(n, dtype=np.float32)
    name_tsr = np.empty(n, dtype=np.float32)
    name_partial = np.empty(n, dtype=np.float32)
    addr_fuzz = np.empty(n, dtype=np.float32)
    addr_tsr = np.empty(n, dtype=np.float32)
    for i in range(n):
        a, b = s1_name[i], o_name[i]
        name_fuzz[i] = fuzz.ratio(a, b)
        name_tsr[i] = fuzz.token_sort_ratio(a, b)
        name_partial[i] = fuzz.partial_ratio(a, b)
        addr_fuzz[i] = fuzz.ratio(s1_addr[i], o_addr[i])
        addr_tsr[i] = fuzz.token_sort_ratio(s1_addr[i], o_addr[i])

    name_core_jaccard = np.array(
        [_jaccard(a, b) for a, b in zip(pair_table["s1_name_core"], pair_table["o_name_core"])],
        dtype=np.float32,
    )
    addr_num_jaccard = np.array(
        [_numbers_jaccard(a, b) for a, b in zip(pair_table["s1_address_numbers"], pair_table["o_address_numbers"])],
        dtype=np.float32,
    )

    s1_postal = pair_table["s1_address_postal"]
    o_postal = pair_table["o_address_postal"]
    postal_both_present = (s1_postal != "") & (o_postal != "")
    postal_match = postal_both_present & (s1_postal == o_postal)

    feats = pd.DataFrame({
        "country_match": (pair_table["s1_country"] == pair_table["o_country"]).astype(np.int8),
        "name_compact_exact": (
            (pair_table["s1_name_compact"] != "") & (pair_table["s1_name_compact"] == pair_table["o_name_compact"])
        ).astype(np.int8),
        "name_fuzz_ratio": name_fuzz / 100.0,
        "name_token_sort_ratio": name_tsr / 100.0,
        "name_partial_ratio": name_partial / 100.0,
        "name_core_jaccard": name_core_jaccard,
        "address_fuzz_ratio": addr_fuzz / 100.0,
        "address_token_sort_ratio": addr_tsr / 100.0,
        "address_numbers_jaccard": addr_num_jaccard,
        "postal_match": postal_match.astype(np.int8),
        "postal_both_present": postal_both_present.astype(np.int8),
        "first_token_match": (
            (pair_table["s1_name_first_token"] != "")
            & (pair_table["s1_name_first_token"] == pair_table["o_name_first_token"])
        ).astype(np.int8),
    })
    return feats
