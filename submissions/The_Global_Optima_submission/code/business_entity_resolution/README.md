# Business Entity Resolution: The Global Optima

This folder holds the code that produced `output/matching_results.tsv` (public F0.5 **0.990273**) and
`output/candidate_pairs.tsv`. The method is described in `Documentation_template.md` at the root of the package, and
`ARCHITECTURE.md` is a one-page map of the system.

- **Stage 1 · Blocking:** retrieval (lexical TF-IDF, reverse search, dense e5), pruning to the top 10 per business,
  and two recall extensions: a rescue of ranks 11–20 (only those a light matcher finds plausible, accepted only when
  all three cross-encoders are near-certain) and address-based rule channels. This gives `output/candidate_pairs.tsv`,
  which is exactly the set the matcher decides on.
- **Stage 2 · Pair scoring:** feature matcher + three cross-encoders, fused into one probability.
- **Stage 3 · Decision model:** list-level XGBoost with ownership context, a 5-seed average for US/India, and a
  France-adapted version for France.
- **Post-processing:** exclusivity and label-backed France rules.

| File | sha256 |
|---|---|
| `output/matching_results.tsv` | `a63a626cf1fd11cb8d0ea462aeb36878bff94d293f66ef023f925a6c6c8f7537` |
| `output/candidate_pairs.tsv` | `58bd076058761dfef31d8a6c8e888a1e6553a1ec31784c0c3bceb64df98f408f` |

`MANIFEST.sha256` lists a SHA-256 checksum for every file in the package, written when the package was built. It lets
a reviewer confirm that the output files are exactly the ones uploaded to the leaderboard and that no file changed
after the build. To check, run from the package root: `sha256sum -c code/business_entity_resolution/MANIFEST.sha256`
(on macOS: `shasum -a 256 -c …`).

## Names used in the docs and in the code

| Docs | Code |
|---|---|
| Stage 1 pruning | stage-1 (`prune_v1`) |
| Stage 2 feature matcher | v5 matcher |
| Stage 2 cross-encoders | chunk2, full21, decoyw2 |
| Stage 2 fusion | L2 late fusion |
| Stage 3 decision model | stage-3b; final model T6; 5-seed average = T6 bag |
| Stage 3 for France | FRrobust, FRtrained_w4 |
| Same-address France rules | census (v2) |
| France post-processing rules | U6–U19 (named after the submission that added them) |

## Environment

- Python 3.11.16. Install with `pip install -r requirements.txt`, then install the CUDA 11.8 build of torch 2.4.1.
- Hardware: IIT Delhi HPC cluster, PBS scheduler, NVIDIA A100-SXM4 80 GB. Each job used 1–2 GPUs and at most 160 GB RAM.
  The GPU steps are the cross-encoders, dense retrieval and GPU XGBoost; everything else runs on CPU.
- Pretrained checkpoints, downloaded once. The pipeline makes no internet calls.
  - `microsoft/mdeberta-v3-base` (MIT): cross-encoders.
  - `intfloat/multilingual-e5-small` (MIT): dense retrieval and reverse competition.
  - `Qwen2.5-7B-Instruct` (Apache-2.0) with our LoRA adapter: confirming French decoy drops only.
- No external data. Everything is trained on the provided training files only.

## Layout

```
src/
  core/                 main pipeline module: candidate generation, feature matcher and decision rule
                        (submission_pipeline), and the F0.5 metric
  stage1_blocking/      preprocessing, lexical index, reverse search, dense retrieval, top-10 pruner,
                        recall extensions (rescue of ranks 11-20, address-based rule channel)
  stage2_pair_scoring/  cross-encoder pairs, training and test scoring, transfer check, fusion
  stage3_decision/      decision model: features, training, test prediction, ownership and rival-copy features,
                        5-seed average
  stage3_france/        France-robust and France-trained decision model, French pseudo-labels, same-address rules
  postprocessing/       exclusivity prune, merge by country, label-backed France rules, LLM judge
    synthetic_alias/    France synthetic-alias channels and their training-label checks (see its README)
  analysis/             the analyses behind the reported results, merged into 5 files by theme
                        (see analysis/README.md)
  tests/                unit tests for preprocessing, the posting index and the pipeline
```

Scripts import each other by module name, so put every `src/` folder on the path first:
`export PYTHONPATH=$(find src -type d | tr '\n' ':')`. `analysis/` holds the experiments and
audits behind the reported results; the pipeline itself uses only the stage folders and `core/`.

**Paths.** The scripts ran from three team accounts on one cluster (`aib262467`, `aib262015`, `aib262140`). They contain
absolute paths of the form `/home/scai/mtech/<account>/scratch/amazon_ml_2026/...`.

To reproduce on one machine, create one working directory with this structure (`data/dataset/{train,test}` = the
organizer files, and `artifacts/` for outputs), then point those constants and the `ER_SCRATCH` / `ER_HOME` variables at
it.

## End-to-end run order

All paths below are relative to `src/`.

### Stage 1 · Blocking: builds `candidate_pairs.tsv`

1. **Preprocess** (normalise, fold, transliterate Indic scripts, phonetic keys):
   `stage1_blocking/preprocess.py --split all` → `artifacts/preprocessed_v2/{train,test}`.
2. **Lexical posting index**, hashed TF-IDF per country, with four routes: name, address, number + word anchor, and
   combined. `stage1_blocking/posting_index.py` → `posting_index_v1/{train,test}_index_c`.
3. **Reverse search** (each record queries a Source 1 index): `stage1_blocking/reverse_index.py` → `{split}_ref_index`,
   `{split}_reverse_t`.
4. **Dense retrieval.** Fine-tune e5-small, then embed and search exactly in both directions:
   `stage1_blocking/dense_retrieval.py finetune | embed | search` → `artifacts/dense_v1`.
5. **Pair features and the feature matcher**, trained on 100k training businesses with a fixed train/dev/holdout split:
   `core/submission_pipeline.py pairs`, then `train --backend xgboost` → `matcher_v4/model_xgb`.
6. **Pruner** (retrieval signals only, out-of-fold): `stage1_blocking/prune_analysis.py` → `prune_v1/` (stage1.json,
   mask_top10.npy, model_top10).
7. **Test candidates and feature-matcher probabilities:**
   `core/submission_pipeline.py predict --variants stage1_blocking/variants_top10.json --shard i/2` (two shards) →
   `submission_v5_top10/candidate_pairs.tsv` (the top 10) plus per-pair scores. A re-run with `PROB_FLOOR=0`
   reproduced the saved scores exactly.
8. **Recall extensions** (appended to the candidate file):
   - ranks 11–20 that the top-20 feature matcher scores at p ≥ 0.02: `stage1_blocking/rescue_11_20.py prep` (step 25);
   - address-based rule channels for France: `stage1_blocking/u9_build.py` (uses `postprocessing/pattern_audit.py`), then the channels of
     steps 27e–27g.

### Stage 2 · Pair scoring

9. **Training sample pack** (feature-matcher out-of-fold probability, top-10 mask): `stage2_pair_scoring/make_matcher_pack.py`.
10. **Cross-encoder training pairs.** All training businesses outside the 100k sample, with gold links plus e5
    neighbours: `stage2_pair_scoring/fulltrain_pairs.py`.
11. **Cross-encoders** (mDeBERTa-v3-base, `name | address`, max 128 tokens). Each writes `model/` and
    `ce_logit_sample.npy`.

    | Member | Script |
    |---|---|
    | chunk2 | `stage2_pair_scoring/train_chunk.py`, `CHUNK=2` (trained on the third team account) |
    | mdeberta_full | `stage2_pair_scoring/train_ddp.py` |
    | full21 | `stage2_pair_scoring/train_ddp.py`, `TRAIN_PAIRS=22000000` |
    | decoyw2 | `stage2_pair_scoring/train_ddp_w.py`, `DECOY_W=2 SEED=2027` |

12. **Test scoring** of the top-10 candidates: `stage2_pair_scoring/score_ce_test.py --shard i/n` → `ce_<name>_shard<i>of<n>.npz`.
13. **Transfer check.** A new cross-encoder must score test like the sample on pairs the other models agree on:
    `stage2_pair_scoring/ce_shift_check.py`.
14. **Fusion:** stage-1 out-of-fold probabilities, then late fusion: `stage2_pair_scoring/fusion_stage1.py`, then
    `stage2_pair_scoring/late_full_train.py`.

### Stage 3 · Decision model, US and India

15. **Base model.** List context, global name and address frequency, reverse competition over the full Source 1
    table, and house numbers.
    - Training: `stage3_decision/boundary_exp.py` + `stage3_decision/boundary_exp2.py` (all three cross-encoder logits) → `boundary_models_t4`.
    - Test features and prediction: `stage3_decision/stage3b_predict.py` → `submission_T4_s3b` (keeps `features.npy`).
16. **Ownership features:** `stage3_decision/ownership_features.py sample | eval | test` → `ownership_v1/`.
17. **Rival-copy features:** `stage3_decision/w2_rival_copies.py sample | test` and `stage3_decision/w2_eval.py` → `w2_v1/`.
18. **Final model (T6).**
    - Training: `stage3_decision/boundary_exp2.py` with `OWN_FILE` = ownership + rival-copy features and `CACHE_FROM` = step 15 →
      `fusion/boundary_models_t6`.
    - Test prediction: `stage3_decision/stage3b_repredict.py` on step 15's saved test features + ownership + rival-copy features
      → `submission_T6_own`.
    - The decision is the dev-chosen expected-F0.5 rule with global record exclusivity (`finalize` in
      `core/submission_pipeline.py`).
19. **5-seed average** (T6 bag): `stage3_decision/t6_bag.py eval` uses T6's exact inputs, adds 4 more seeds and averages them
    with T6; the rule (expected F0.5, floor 0.68) is chosen on dev, and the average is used only because it beats T6
    on both dev and holdout. Then `t6_bag.py test`.

### Stage 3 · France (no training labels; French businesses only)

20. **France-robust model.** Drops the inputs that shift in France: `stage3_france/france_robust.py` → `submission_FRrobust`,
    then the same-address rules: `stage3_france/france_census.py`.
21. **French pseudo-labels** (same-address rules + confident model predictions): `stage3_france/fr_labels.py` → `fr_labels/`.
22. **France-trained model** (`is_france` input; confident negatives subsampled and re-weighted):
    `stage3_france/france_trained.py` with `EXTREME_NEG_WEIGHT=4` → `submission_FRtrained_w4/expected_0.64`.

### Post-processing

23. **Exclusivity prune, then same-address rules** (France): `postprocessing/exclusivity_prune.py` (`LATE` = the step-22
    `late_scores.npz`), then `CENSUS_V2=1 stage3_france/france_census.py` → `submission_FRw4_xprune_census/expected_0.64`.
24. **Merge by country.** T6 rows for all countries (`submission_UP2_T6_FRw4/final`), then the
    France rows replaced by `postprocessing/merge_by_country.py` (`BASE_TSV` = that file, `OVERRIDE_TSV` = the step-23 file,
    `COUNTRIES=France`). The US/India rows are then replaced by the step-19
    5-seed average in the same way.
25. **Rescue of ranks 11–20** (the Stage 1 recall extension; it runs here because it needs the final decisions):
    - `stage1_blocking/rescue_11_20.py prep`: ranks 11–20 with top-20 feature-matcher p ≥ 0.02, scored by all three cross-encoders
      (`stage2_pair_scoring/score_ce_test.py`).
    - `eval` chooses the rule on dev; `apply` runs on test. A pair is accepted when the feature matcher is ≥ 0.1,
      all three cross-encoders are ≥ 0.9, and no other business holds the record.
26. **Address-evidence rules** (France):
    - `postprocessing/nonus_fix.py` with `TAU=0.9` adds a rejected candidate at the business's house number whose name is invented
      or only garbled (no word from any Source 1 name of the country), when T6 ≥ 0.9 and the record is unclaimed.
    - `postprocessing/restore_fix.py <earlier upload> <current> <out>` re-adds same-address copies that an earlier version linked
      and the current pipeline dropped (earlier model p ≥ 0.85, same house number, non-blank address, extra words
      garbled or fillers, record unclaimed).
27. **Label-backed France rules**, applied in this order; adds only when the record is unclaimed. Each rule's pattern
    was measured on the full US/India training labels first.
    - a. Invented single-word aliases at the business's exact address, accepted by the 5-seed average:
      `postprocessing/u8_build.py`. Train: 98.7–100% true.
    - b. Rule-based candidate channel (invented word or initials at a unique own address; the new pairs are appended
      to `candidate_pairs.tsv`): `stage1_blocking/u9_build.py`. Train: 97.5% / 100% true.
    - c. Blank records whose exact name belongs to one French business; aliases in the rank 11–20 pool; business-type
      word replaced by a filler word: `postprocessing/u10_build.py`.
    - d. Invented word + alias suffix (Sys / Labs / One / .com) at a unique own address; "+ Services / Center" are
      decoys and excluded: `postprocessing/u11_build.py`, `postprocessing/inv2.py`. Train: 98.6–99.7% true.
    - e. Invented aliases the France model rejected, at most two invented names at the address: `postprocessing/u12_build.py`,
      `postprocessing/inv_struct.py`. Train: 97.1–99.3% true.
    - f. Synthetic-alias channels (`postprocessing/synthetic_alias/`, see its README): records named with the generator's invented
      syllables, at a full address that belongs to exactly one business (train 99.76% true) or to the accepted copies
      of exactly one business. `merge_synthetic_rules.py` writes the link list; then
      `postprocessing/apply_alias_links.py <base> <links> <out> unique_s1_address`, and the same with `all` on its output.
    - g. Blank records whose core name belongs to one business only, and records without a house number on the
      business's street with a copy-type name: `postprocessing/recall2.py`, `postprocessing/alias_union.py` (France only). Train: 94–100% true.
    - h. Decoy drops: accepted links that add an affiliate word (Holding, Distribution, International…; 0% true on
      train) or that the LoRA judge confirms as decoys: `postprocessing/lora_judge.py`, `postprocessing/llm_france.py`.
    - i. Legal-form swaps: dropped at another house number (train 19–26% true), restored on another street at the
      same number (train 90% true): `postprocessing/legalswap.py`.

    This gives **`output/matching_results.tsv`** and **`output/candidate_pairs.tsv`**.
28. **Validation** with the organizer validator:
    ```
    python utils/validate_submission.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
    ```

## Evaluation protocol

100,000 training businesses (50k US, 50k India) are split by sha256 of the business ID into train, dev (15%) and
holdout (15%). Models train on the train split, rules and thresholds are chosen on dev, and holdout is scored once per
configuration. The embedding model and all cross-encoders are trained only on businesses outside this sample.
US/India holdout macro F0.5: **0.99128**. Public leaderboard: **0.990273**.
