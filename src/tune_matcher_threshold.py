"""Step 1: sweep the matcher's decision-rule floor globally and per-country on
the dev split only (never touch holdout), using Vishal's own v4 model,
features, and decide()/macro_f05 logic exactly as in submission_pipeline.py --
this is pure re-scoring, no retraining, so it's cheap.
"""
import gzip
import hashlib
import json
import time
from collections import defaultdict

import numpy as np
import xgboost as xgb

ART = "data/vishal_v4_artifacts"
BEST_ITERATION = 732


def f05(truth, predicted):
    if not truth:
        return 1.0 if not predicted else 0.0
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    return 5.0 * tp / (5 * tp + 4 * fp + fn)


def split_of(ref_id):
    h = int(hashlib.sha256(ref_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else "dev" if h < 85 else "holdout"


def decide(ref_idx, prob, threshold, n_refs):
    """Exact port of submission_pipeline.decide(rule='expected', ...)."""
    order = np.lexsort((-prob, ref_idx))
    result = np.zeros(len(prob), dtype=bool)
    bounds = np.searchsorted(ref_idx[order], np.arange(n_refs + 1))
    for i in range(n_refs):
        idx = order[bounds[i]:bounds[i + 1]]
        if not len(idx):
            continue
        p = prob[idx].astype(np.float64)
        expected_size = p.sum()
        k = np.arange(1, len(p) + 1)
        value = 1.25 * np.cumsum(p) / (0.25 * expected_size + k)
        best = int(np.argmax(value))
        empty = float(np.prod(1 - np.clip(p, 0, 1 - 1e-9)))
        if value[best] > empty and p[0] >= threshold:
            result[idx[:best + 1]] = True
    return result


def macro_score(ref_names, ref_idx, targets, mask, gold):
    pred = defaultdict(set)
    for i, t in zip(ref_idx[mask], targets[mask]):
        pred[i].add(t)
    return float(np.mean([f05(set(gold.get(name, [])), pred.get(i, set())) for i, name in enumerate(ref_names)]))


def main():
    t0 = time.time()
    print("loading pairs.tsv.gz ...", flush=True)
    ref_ids, target_ids = [], []
    with gzip.open(f"{ART}/pairs/pairs.tsv.gz", "rt") as f:
        for line in f:
            r, t = line.rstrip("\n").split("\t")
            ref_ids.append(r)
            target_ids.append(t)
    n = len(ref_ids)
    print(f"loaded {n:,} pairs in {time.time()-t0:.1f}s", flush=True)

    print("loading gold/countries ...", flush=True)
    gold = json.load(open(f"{ART}/pairs/gold.json"))
    countries = json.load(open(f"{ART}/pairs/countries.json"))

    print("computing dev-split mask ...", flush=True)
    unique_refs = sorted(set(ref_ids))
    ref_split = {r: split_of(r) for r in unique_refs}
    ref_country = countries

    ref_ids_arr = np.array(ref_ids)
    dev_row_mask = np.array([ref_split[r] == "dev" for r in ref_ids], dtype=bool)
    print(f"dev rows: {dev_row_mask.sum():,} / {n:,}", flush=True)

    dev_refs = sorted({r for r in unique_refs if ref_split[r] == "dev"})
    ref_to_idx = {r: i for i, r in enumerate(dev_refs)}
    print(f"dev references: {len(dev_refs):,}", flush=True)

    print("loading x.npy (mmap) and scoring dev rows ...", flush=True)
    x = np.load(f"{ART}/pairs/x.npy", mmap_mode="r")
    print(f"x shape: {x.shape}", flush=True)

    dev_row_idx = np.nonzero(dev_row_mask)[0]
    x_dev = np.asarray(x[dev_row_idx])  # materialize just the dev subset
    print(f"x_dev shape: {x_dev.shape}, loaded in {time.time()-t0:.1f}s total", flush=True)

    booster = xgb.Booster()
    booster.load_model(f"{ART}/model_xgb/model.json")
    t1 = time.time()
    prob_dev = booster.inplace_predict(x_dev, iteration_range=(0, BEST_ITERATION + 1))
    prob_dev = np.asarray(prob_dev, dtype=np.float64)
    print(f"scored {len(prob_dev):,} dev pairs in {time.time()-t1:.1f}s", flush=True)

    dev_ref_ids = ref_ids_arr[dev_row_idx]
    dev_target_ids = np.array(target_ids)[dev_row_idx]
    dev_ref_idx = np.array([ref_to_idx[r] for r in dev_ref_ids])
    dev_ref_names = dev_refs
    dev_country_of_ref = np.array([ref_country[r] for r in dev_refs])

    grid = np.round(np.arange(0.05, 0.96, 0.05), 3)

    print("\n--- GLOBAL floor sweep ---", flush=True)
    best_global = (None, -1)
    for t in grid:
        mask = decide(dev_ref_idx, prob_dev, t, len(dev_refs))
        score = macro_score(dev_ref_names, dev_ref_idx, dev_target_ids, mask, gold)
        print(f"  floor={t:.2f} macro_f05={score:.6f}", flush=True)
        if score > best_global[1]:
            best_global = (t, score)
    print(f"BEST GLOBAL: floor={best_global[0]}, dev macro_f05={best_global[1]:.6f}", flush=True)

    def score_over_refs(mask, ref_index_subset):
        """macro F0.5 restricted to a subset of reference indices (by index into dev_refs)."""
        pred = defaultdict(set)
        for ri, t in zip(dev_ref_idx[mask], dev_target_ids[mask]):
            pred[ri].add(t)
        return float(np.mean([f05(set(gold.get(dev_ref_names[ri], [])), pred.get(ri, set()))
                               for ri in ref_index_subset]))

    print("\n--- PER-COUNTRY floor sweep ---", flush=True)
    best_per_country = {}
    for country in sorted(set(dev_country_of_ref)):
        country_ref_indices = np.nonzero(dev_country_of_ref == country)[0]
        country_ref_set = set(country_ref_indices.tolist())
        row_mask_country = np.isin(dev_ref_idx, country_ref_indices)
        sub_ref_idx = dev_ref_idx[row_mask_country]
        sub_prob = prob_dev[row_mask_country]
        row_positions = np.nonzero(row_mask_country)[0]

        best = (None, -1)
        for t in grid:
            sub_mask = decide(sub_ref_idx, sub_prob, t, len(dev_refs))
            full_mask = np.zeros(len(prob_dev), dtype=bool)
            full_mask[row_positions[sub_mask]] = True
            score = score_over_refs(full_mask, country_ref_indices)
            if score > best[1]:
                best = (t, score)
        print(f"  {country}: best floor={best[0]}, dev macro_f05={best[1]:.6f} ({len(country_ref_indices):,} refs)",
              flush=True)
        best_per_country[country] = best

    print("\n--- COMBINED: per-country best floors applied together ---", flush=True)
    combined_mask = np.zeros(len(prob_dev), dtype=bool)
    for country, (t, _) in best_per_country.items():
        country_ref_indices = np.nonzero(dev_country_of_ref == country)[0]
        row_mask_country = np.isin(dev_ref_idx, country_ref_indices)
        sub_mask = decide(dev_ref_idx[row_mask_country], prob_dev[row_mask_country], t, len(dev_refs))
        row_positions = np.nonzero(row_mask_country)[0]
        combined_mask[row_positions[sub_mask]] = True
    combined_score = macro_score(dev_ref_names, dev_ref_idx, dev_target_ids, combined_mask, gold)
    print(f"COMBINED per-country floors: dev macro_f05={combined_score:.6f}", flush=True)
    print(f"\nvs single global floor 0.75 (Vishal's reported dev macro_f05): 0.9755574731989121")
    print(f"vs best global floor found here: {best_global[1]:.6f}")


if __name__ == "__main__":
    main()
