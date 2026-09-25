"""Candidate generation (blocking) for Source-1 entities against Source-2/3.

Union of several cheap, high-recall blocking keys, each implemented as a
hash join (pandas merge) rather than an all-pairs comparison so it scales to
millions of rows. Precision is not the goal here — the matching model (see
features.py / train_baseline.py) narrows candidates down; blocking only
needs to keep the true match reachable.

Every join below goes through `_bounded_merge_candidates`, which caps the
per-key freq_s1 * freq_other product before merging. A flat frequency
threshold on one side isn't enough: a key appearing a few thousand times on
*each* side of the join still produces millions of pairs on its own, and a
handful of such keys can balloon a plain merge to hundreds of GB (hit this
twice in practice — once on name tokens, once on first-name-word — so every
blocking key is capped uniformly now, not just the one that broke first).
"""
import pandas as pd

MAX_KEY_FREQ = 200          # a key this common on either side isn't discriminative on its own
MAX_PAIR_PRODUCT = 40_000   # hard cap on freq_s1 * freq_other per key, bounds the merge blowup


def _bounded_merge_candidates(s1, other, key_cols, require_nonempty_col, label):
    s1f = s1[s1[require_nonempty_col] != ""][["entity_id"] + key_cols]
    otherf = other[other[require_nonempty_col] != ""][["entity_id"] + key_cols]

    freq_s1 = s1f.groupby(key_cols).size()
    freq_other = otherf.groupby(key_cols).size()
    common = freq_s1.index.intersection(freq_other.index)
    fs1, fo = freq_s1.loc[common], freq_other.loc[common]
    product = fs1 * fo
    keep = product[(fs1 <= MAX_KEY_FREQ) & (fo <= MAX_KEY_FREQ) & (product <= MAX_PAIR_PRODUCT)].index

    print(f"  [{label}] {len(common)} shared keys, keeping {len(keep)} after cap "
          f"(dropped {len(common) - len(keep)} overly-generic keys, expected pairs ~= {int(product.loc[keep].sum()):,})",
          flush=True)

    keep_df = keep.to_frame(index=False) if isinstance(keep, pd.MultiIndex) else pd.DataFrame({key_cols[0]: keep})
    s1f = s1f.merge(keep_df, on=key_cols)
    otherf = otherf.merge(keep_df, on=key_cols)

    m = s1f.merge(otherf, on=key_cols, suffixes=("_s1", ""))
    m = m[["entity_id_s1", "entity_id"]].rename(
        columns={"entity_id_s1": "source1_entity_id", "entity_id": "candidate_entity_id"}
    )
    return m.drop_duplicates()


def compact_name_candidates(s1, other):
    # Empirically, 0 of 7.6M true matched pairs in train cross a country
    # boundary (verified against train_ground_truth.tsv) — country is an
    # open string set (France included in test), not a hardcoded filter, but
    # partitioning blocking joins on it is a safe, recall-neutral speedup.
    return _bounded_merge_candidates(s1, other, ["name_compact", "country"], "name_compact", "compact_name")


def postal_country_candidates(s1, other):
    return _bounded_merge_candidates(s1, other, ["address_postal", "country"], "address_postal", "postal_country")


def first_token_country_candidates(s1, other):
    return _bounded_merge_candidates(s1, other, ["name_first_token", "country"], "name_first_token", "first_token_country")


def _explode_tokens(df):
    tok = df[["entity_id", "country", "name_core"]].copy()
    tok["token"] = tok["name_core"].str.split()
    tok = tok.explode("token")
    tok = tok[tok["token"].notna() & (tok["token"] != "")]
    return tok[["entity_id", "country", "token"]]


def token_candidates(s1, other):
    """Blocking on shared core-name tokens, partitioned by country."""
    s1_tok = _explode_tokens(s1)
    other_tok = _explode_tokens(other)
    return _bounded_merge_candidates(s1_tok, other_tok, ["token", "country"], "token", "token_candidates")


def generate_candidates(s1, other, use_tokens=True):
    """Union of all blocking strategies between s1 and one other source (S2 or S3)."""
    parts = [
        compact_name_candidates(s1, other),
        postal_country_candidates(s1, other),
        first_token_country_candidates(s1, other),
    ]
    if use_tokens:
        parts.append(token_candidates(s1, other))
    all_cands = pd.concat(parts, ignore_index=True).drop_duplicates()
    return all_cands


def candidates_to_map(df):
    """Collapse a (source1_entity_id, candidate_entity_id) frame into {s1_id: set(cand_ids)}."""
    return df.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
