# Analysis

Experiments, audits and checks behind the results reported in the documentation. The pipeline itself does not
use them. Each file groups related scripts as commands: `python <file>.py <command> [args]`.

## `blocking_experiments.py`: Blocking and retrieval experiments

| Command | What it checks |
|---|---|
| `benchmark_capped` | Test capped global retrieval against saved original results; never replace baseline. |
| `blocking_cutoff_cost` | How many true links each candidate-rank cutoff keeps. |
| `blocking_recall` | Forward candidate recall for several route combinations of one index, on fixed references. |
| `export_previous_samples` | Writes the earlier diagnostic reference samples, regenerated from their seeds, as ID lists. |
| `native_loss` | Share of India holdout loss (T6) on records whose name is (mostly) in a native Indian script, split by stage. |
| `prepare_saved_pairs` | Prepare labeled development pairs from saved retrieval; never inject missing gold. |
| `script_recall` | Blocking recall by script: does Tamil / Telugu / Malayalam / ... text lose true links at the candidate stage? |
| `sparse_fullpool` | Streaming full-pool name/address TF-IDF top-k benchmark for fresh references. |
| `v5raw_align` | Align the raw (unfloored) v5 test scores of the two predict shards to candidate_pairs.tsv expanded order |
| `web_leet_channel` | Two deterministic candidate channels for copies that character retrieval misses (US + India), found by reading the |

## `model_experiments.py`: Pair-scoring and decision-model experiments

| Command | What it checks |
|---|---|
| `analyze_matcher_loss` | Where does macro F0.5 go? Splits holdout loss into blocking (gold never a candidate) |
| `calibrate_probs` | Probability calibration of the feature matcher, with the decision rule re-chosen on dev. |
| `ce_report` | Holdout report for the cross-encoders: scores and F0.5 per model. |
| `cut_and_infer` | Production candidate cut for upload 2: the reduced candidate set is exactly what the final models decide on. |
| `decision_tweaks` | Two cheap tweaks on the S4 setup (big-data mdeberta fold-A logits + v5 probability, decoy w=3, L2 on fold B): |
| `decoy_weight_check` | Decoy reweighting for the late-fusion matcher (L2: XGBoost on [v5 top-10 probability, mdeberta logit]). |
| `exclusivity_norm` | Exclusivity-consistent re-decision: every S2/S3 record belongs to at most one S1, so its probabilities over the S1 |
| `fusion_predict` | Test inference for the cross-encoder fusion matcher. |
| `gap_diag` | Where do the remaining points go? Per-country diagnostics for the final push (27 Sep afternoon). |
| `late_fusion_check` | Late fusion of a cross-encoder with the feature-matcher probability, checked on dev and holdout. |
| `multi_ce_eval` | Which cross-encoders add information on top of each other? Late-fusion comparison on one fixed protocol. |
| `overcommit_check` | Exclusivity-consistent probabilities: a record belongs to at most one S1, so its probabilities over S1s sum to <= 1. |
| `pilot_matcher` | Pilot: entity-grouped train/dev/holdout, fixed LightGBM, dev-only threshold. |
| `score_decomp` | Where is the leaderboard score lost: France vs US+India, without labels for test. |
| `seg_calib` | Calibration of T6 by error segment on dev and holdout: is stage-3b under-confident where it misses real copies? |
| `shap_misclassified` | SHAP feature attributions for the feature matcher's misclassified holdout pairs. |
| `shift_audit` | Train-sample -> test transfer audit of every stage-3b input (the listwise lesson: holdout does not see test shift). |
| `shift_ens_check` | Two last-day checks on the stage-3b decision. |
| `sibling_noise_check` | Do true aliases share their address noise with the S1's other copies? Train check (dev + holdout). |
| `t1_empty_check` | Why does T1 leave 16.9k references empty that S5 matched with near-certainty? |
| `t5_shift_audit` | Transfer audit for T5 (the shift_audit method, extended to T5): on pairs v5 and chunk2 agree on, compare |
| `train_listwise` | Listwise cross-encoder: one reference + its whole candidate list in one sequence, one logit per candidate. |

## `france_audits.py`: France audits and France-model reports

| Command | What it checks |
|---|---|
| `affix_check` | Is 'S1 name + one extra word' a true match or a generated decoy? Measured with train labels, by added word. |
| `census_review` | Manual-review sheet for the France census fixes: sampled restored / rejected pairs with each S1's full candidate list. |
| `colocated_check` | Co-located unrelated names: alias or same-building distractor? True rate by how many such records share the S1's address. |
| `cross_source_check` | Cross-source link completion, sized before building: rejected candidates that duplicate an accepted sibling. |
| `delta_audit` | Hand-check sheet for the France score change between the uploaded leaderboard file and a new submission. |
| `filler_test` | Label test for the France-trained pattern 'first word kept, type word dropped, filler added, same address'. |
| `france_anatomy` | France rejection anatomy: where does France reject pairs the v5 matcher believes, and are such pairs usually true? |
| `france_audit` | Manual France audit sheet for a submission: French references with every top-10 candidate (+ accepted rescue links), |
| `france_block_check` | France blocking check: exact-key copies (same core name, same house number, same street words) missing from the |
| `france_cases` | France case file for manual reading: businesses where the decision is shakiest, one block per S1. |
| `france_check` | Label-free per-country look at the S5 test scores: is France (no training data) less certain than US/India? |
| `france_diff_cases` | Read what a new France submission changed: random France links added / removed vs a base, with the S1's accepted |
| `france_eda` | France deep dive on the S5 test scores (France has no training data, so everything here is label-free). |
| `france_noise` | France diagnosis (label-free): which name words make the cross-encoder reject pairs the v5 model is sure about? |
| `france_numdiff` | France pairs by (name relation, house-number/street relation), accepted vs rejected in a submission. |
| `france_patterns` | Size four France patterns found by reading case files, on every France pair (label-free counts + p quantiles). |
| `france_patterns2` | Size four France patterns found by reading case files, on every France pair (label-free counts + p quantiles). |
| `france_pseudo` | France self-training set from T1's test predictions: only pairs that every decider is very sure about. |
| `france_rates` | Per-country, label-free: links per reference, and how often each kind of candidate is accepted (T5 + census): |
| `france_review` | Readable French case dump for manual review (label-free): T5 + census decisions vs S5, with T5 probabilities. |
| `france_robust_report` | France-robust vs T5 on France: links per reference, empty share, added/removed links (pre-census vs T5 raw, and |
| `france_trained_report` | France-trained vs France-robust (both with census) on France: links per reference, empty share, added/removed links, |
| `france_veto` | France AND-veto: a France match survives only if both the base submission and the self-trained view accept it. |
| `gen2_eda` | Matcher EDA for the S5 generation: is the remaining loss in the ranking or in the stop decision? |
| `gen4_eda` | Gen-4 error EDA of T4's stage-3b (holdout 0.99037), holdout side (US/India, labels). |
| `swap_france` | Ablation build: U5 with its France rows replaced by another France version, keeping U5's France rescue links. |
| `train_fr_aug` | French-noise augmentation fine-tune of a pair cross-encoder (France has no training data). |
| `typo_street_check` | Train check for census v2's typo path: is a street-word typo copy noise or a decoy signal in US/India labels? |
| `u8_check` | Independent check of U8 against U7: per-country diff, structural checks, expected gain, hand-check sample. |

## `rule_evidence.py`: Error sweeps and training-label evidence for the rules

| Command | What it checks |
|---|---|
| `alias_base` | Training-label rates for alias records at a business's address (true / other business / no business). |
| `alias_check_france_audit` | Patterns found by the manual France audit, measured on test (per country) and on the labelled US/India holdout. |
| `alias_check_typo_street` | Train check (alias variant) for census v2's typo path: is a street-word typo copy noise or a decoy signal in US/India labels? |
| `alias_cond` | Alias rule, conditioned on the model: among US/India dev + holdout top-10 pairs (error_sweep frames, T6 decisions), |
| `alias_link` | Alias link-up: a sibling record "X dba/fka/aka <S1 name>" names X as an alias of that S1, so another candidate record |
| `alias_rate` | Train-label check for the U6/U7 France restores: true rate of invented-name aliases (no shared name word) and of |
| `alias_test` | Counts and samples of alias records at a business's address in test. |
| `alias_union_test` | New links from the accepted-copy-address alias rule beyond an earlier file, with samples. |
| `blank_eda` | Blank-address candidates under the S5 decision: how much loss, and is any of it recoverable? |
| `blank_expert` | Fixes for blank-address misses on top of stage-3b (cached features from boundary_exp2.py). |
| `blind_sample` | Blind judge set: 80 holdout pairs from T6's decision band (0.15 < p < 0.85), with the context an LLM prompt would |
| `empty_guard` | Empty-guard check on T6 dev/holdout: when the expected-F0.5 rule leaves an S1 empty, add its top candidate if p >= x. |
| `error_sweep_report` | Tables from error_sweep.py outputs: loss per stage and pattern cell, train (actual, dev + holdout) vs test (expected), |
| `excl_empty` | Exclusivity finalize on test: when a record is claimed by several S1s, the highest p keeps it and the rest re-decide. |
| `nocand_usin` | US/India copy-type records that retrieval never returned: counts and a trial file. |
| `pattern_cond` | Pattern cells (pattern_audit.py definitions) on the US/India dev + holdout top-10 pairs, split by T6's decision D: |
| `rowsweep` | Every-row check of U5 + the textual-decidability ceiling for any reader (LLM or human). |
| `u5_check` | Pre-upload audit of U5 against U2: per-country link diff, exclusivity (a record in at most one S1 row), |
| `usin_patterns` | Two generator patterns behind T6's non-blank US/India errors, measured on labelled train pairs (dev + holdout). |
| `usin_read` | Every non-blank US/India holdout error of T6, for reading one by one, plus the blocking misses. |

## `llm_experiments.py`: LLM judge experiments

| Command | What it checks |
|---|---|
| `delta_merge` | Apply the LLM-judge changes to the best file as a delta, keeping every other rule's links. |
| `llm_apply` | Test path of the LLM judge (only run if llm_probe.py MODE=eval gains on dev and holdout). |
| `llm_ctx` | LLM judge v2: the same context stage-3b has + the data's rules + worked examples (US/India band, labelled). |
| `llm_lexicon` | Qwen2.5-7B-Instruct labels single French words from business names: A business type/activity, B generic filler, |

## `support/`

Helper modules that the commands above and the unit tests import: `retrieval_experiment`, `blocking_v2`, `diagnose_retrieval`, `audit_sparse_misses`, `posting_index_benchmark`, `selective_index_benchmark`, `sparse_retrieval_benchmark`, `gen4_test_side`, `usin_errors`, `blank_disagree`, `error_sweep`, `indic_translit`, `native_channel`.
