"""Splink-style dry-run: estimate how many candidate pairs a blocking key
would generate BEFORE actually merging on it.

This is the guardrail we didn't have when two blocking keys (first_token,
then a raw token frequency cutoff) blew past a 64GB job budget to 200GB+ on
HPC. Call `precheck` on any key (an exact-match key, a bucket id from LSH,
anything) before committing to a merge — it costs a groupby + a join on the
distinct key values, never materializes the full pair list, and tells you
whether the merge you're about to run is safe.
"""
import pandas as pd

DEFAULT_MAX_TOTAL_PAIRS = 200_000_000   # bail point for the whole key
DEFAULT_MAX_SINGLE_KEY_PAIRS = 500_000  # bail point for any one key value


def precheck(s1, other, key_cols, require_nonempty_col=None,
             max_total_pairs=DEFAULT_MAX_TOTAL_PAIRS,
             max_single_key_pairs=DEFAULT_MAX_SINGLE_KEY_PAIRS,
             label=""):
    """Estimate the pair count a merge on `key_cols` would produce.

    Returns a report dict with the total expected pairs, the worst offending
    keys, and a `safe` flag. Never builds the actual candidate pair list —
    only value_counts + a join on distinct keys, so this is cheap even when
    the merge itself would not be.
    """
    s1f, otherf = s1, other
    if require_nonempty_col is not None:
        s1f = s1[s1[require_nonempty_col] != ""]
        otherf = other[other[require_nonempty_col] != ""]

    freq_s1 = s1f.groupby(key_cols).size()
    freq_other = otherf.groupby(key_cols).size()
    common = freq_s1.index.intersection(freq_other.index)
    fs1, fo = freq_s1.loc[common], freq_other.loc[common]
    product = fs1 * fo

    total_pairs = int(product.sum())
    worst = product.sort_values(ascending=False).head(10)
    n_over_single_cap = int((product > max_single_key_pairs).sum())

    report = {
        "label": label,
        "n_shared_keys": int(len(common)),
        "total_expected_pairs": total_pairs,
        "n_keys_over_single_cap": n_over_single_cap,
        "worst_keys": worst.to_dict(),
        "safe": total_pairs <= max_total_pairs and n_over_single_cap == 0,
    }
    return report


def safe_keys(s1, other, key_cols, max_single_key_pairs=DEFAULT_MAX_SINGLE_KEY_PAIRS,
              max_key_freq=None):
    """Return the subset of shared key values safe to merge on.

    Unlike `precheck`, this is meant to actually be used to filter before a
    merge: prune only the offending key values (a handful out of possibly
    hundreds of thousands), not abandon the whole key/table because of them.
    """
    freq_s1 = s1.groupby(key_cols).size()
    freq_other = other.groupby(key_cols).size()
    common = freq_s1.index.intersection(freq_other.index)
    fs1, fo = freq_s1.loc[common], freq_other.loc[common]
    product = fs1 * fo
    mask = product <= max_single_key_pairs
    if max_key_freq is not None:
        mask &= (fs1 <= max_key_freq) & (fo <= max_key_freq)
    return product[mask].index


def print_report(report):
    tag = "OK" if report["safe"] else "UNSAFE"
    print(f"[precheck:{tag}] {report['label']}: {report['n_shared_keys']:,} shared keys, "
          f"~{report['total_expected_pairs']:,} expected pairs, "
          f"{report['n_keys_over_single_cap']} key(s) over the single-key cap", flush=True)
    if not report["safe"] and report["worst_keys"]:
        print(f"  worst offending keys: {report['worst_keys']}", flush=True)
