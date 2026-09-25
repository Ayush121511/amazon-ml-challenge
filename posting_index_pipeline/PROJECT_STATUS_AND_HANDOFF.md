# Amazon ML Challenge 2026: project status and new-chat handoff

**Last refreshed: 25 September 2026.** Read
`CHALLENGE_DATA_AND_RULES.md` first. Workspace is `E:\Amazon_ML_2026`
on Windows. Padum project is at `~/scratch/amazon_ml_2026` under user
`aib262140`. This file records confirmed work; a queued/running PBS job must
be checked live rather than inferred from these notes.

## Confirmed completed work

1. Audited all supplied TSVs: train contains 2,206,821 Source 1 references
   and 10,320,219 Source 2/3 targets; test contains 1,732,544 references
   and 9,969,589 targets. Ground truth has 7,638,365 links and 123,247
   no-match references. Detailed audit: `research/dataset_audit.json`.
2. Set up VS Code Remote-SSH and a Padum Conda environment at
   `~/scratch/amazon_ml_2026/.conda-er`; Python 3.11 and CPU packages
   worked. The IITD proxy was used only for package downloads. Credentials
   are not in this workspace or handoff.
3. Preprocessed **all three training sources** using version 2. Output is
   `artifacts/preprocessed_v2/train/` on Padum, 12,527,040 rows total.
   Original fields/IDs were preserved. Extra normalized and Latin accent-folded
   fields preserve Indic combining marks. Manifests and row counts passed.
   Test files were later preprocessed in the `aib262467` account (see
   "Posting index workstream" below).
4. Implemented the official macro F0.5 scorer, including no-match cases.
5. Built a SQLite FTS5 index of the complete training target pool on Padum.
   Candidate retrieval used same-country name words, address words, and
   name 3-grams, with rank fusion. The run was stopped at **850 of 2,000**
   sampled references because queries were too slow. On this ordered partial
   sample, link recall was **95.36% at top 50**, **96.82% in the union**
   of three routes. Of 2,929 true links, 43 were found but lost at the
   top-50 cutoff and 93 were missed by every route. This is a biased partial
   development diagnostic, **not** a final validation score.
6. Tried capping global SQLite search results at 300/1,000. It did not solve
   the speed issue: 22/36 and 17/36 timed-out queries respectively. Do not
   scale that implementation.
7. Created `research/pairs_dev_v1/` from saved retrieval unions: **850
   references, 114,612 pairs, 2,836 positive, 111,776 negative**. The 93
   unretrieved true links are recorded separately, not inserted into the
   candidate set.
8. Trained a **local pilot LightGBM matcher** with 33 similarity and retrieval
   features. Connected references sharing a candidate or true target were
   grouped across splits. The groups produced train/dev/holdout sizes of
   **758/49/43 references**. The threshold was selected only on dev;
   macro F0.5 was **0.8967 dev / 0.9086 holdout**. The 43-reference holdout
   is too small and selected from the retrieval prefix; this is **not a
   leaderboard estimate**. Model and report are under
   `artifacts/pilot_matcher_local_v1/`. Do not repeatedly tune against that
   holdout.
9. Implemented a new bounded inverted-list blocking prototype in
   `business_entity_resolution/src/blocking_v2.py`. It selects **400 fresh
   references**, excluding all old 2,000 sample IDs. It creates bounded
   country+name-word, country+address-word, and country+name-4-gram posting
   lists from the full training target pool. It drops blocks larger than 200,
   ranks candidates, and reports recall plus total runtime and memory.
   Full Padum benchmark job `1065215.pbshpc` completed: 400 references,
   1,379 true links, top-50 recall **47.72%**, union recall **49.17%**,
   126 empty shortlists. It dropped **3,810/4,392** requested keys due to
   the 200-posting cap. Index/sample scan took 239.7s, candidate mapping
   plus evaluation 63.6s; peak process RSS about 31.6 MiB. This fast method
   loses too many matches and must **not** replace the old retrieval.
   Full result is visible in the user's Padum job log; a local copy of
   `artifacts/blocking_v2/report.json` is not yet present.

Latest verification: **19 unit tests passed** after adding blocking v2.
The small smoke dataset lacks almost all true targets, so its zero recall is
not a meaningful quality result. Never quote that as full-pool recall.

## Where to find things

| Item | Location in workspace |
| --- | --- |
| Organizer rules | `6ab10eb3b23ba_student_resource/student_resource/README.md` |
| Dataset audit and early EDA | `research/REVIEW_SUMMARY.md`, `research/dataset_audit.json` |
| Preprocessing | `business_entity_resolution/src/preprocess.py` |
| Slow baseline retrieval | `business_entity_resolution/src/retrieval_experiment.py` |
| Saved retrieval reports and examples | `research/partial_report.json`, `research/retrieval_misses.json` |
| Pilot pair data | `research/pairs_dev_v1/` |
| Pilot matcher and report | `business_entity_resolution/src/pilot_matcher.py`, `artifacts/pilot_matcher_local_v1/report.json` |
| New blocking experiment | `business_entity_resolution/src/blocking_v2.py`, `padum/blocking_v2.pbs`, `business_entity_resolution/BLOCKING_V2.md` |
| Technical notes | `business_entity_resolution/PREPROCESSING.md`, `business_entity_resolution/RETRIEVAL.md`, `research/BLOCKING_HANDOFF.md` |

## Latest blocking result and current next step

The same-sample 5,000-posting-cap diagnostic **completed** as job
`1065219.pbshpc`. It searched all 10,320,219 targets for 400 references:

| Cap | Top-50 link recall | Union recall | Empty shortlists | Dropped keys |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 47.72% | 49.17% | 126/400 | 3,810/4,392 |
| 5,000 | 73.68% | 79.04% | 3/400 | 2,169/4,392 |

The higher-cap job took about 237.7s for indexing/sampling, 1.25s to query
the 400 references, and 63.4s to map target IDs/evaluate. Peak process RSS
was 48,512 KiB. The same-sample increase shows the strict cap caused much
of the loss, but **73.68% recall at 50 remains too low**. Its small query time
is misleading: the focused index requires two full target scans, so it is not
yet suitable for millions of references. The full report is on Padum at
`artifacts/blocking_v2_cap5000/report.json`; local copy not yet downloaded.

Bulk TF-IDF benchmark implemented in
`business_entity_resolution/src/sparse_retrieval_benchmark.py`. On a local
400,000-target prefix sample and 400 fresh references, it took 111s total and
about 644MB process RSS. Name/address sparse top-k multiplication was fast;
all 73 gold links *present in that truncated target slice* were found in
the union. This conditional figure is **not full-pool recall** and must not
be compared to the 95.36% slow-baseline result.

A streaming full-pool benchmark completed as job `1065234.pbshpc` using
`business_entity_resolution/src/sparse_fullpool.py` and
`padum/sparse_fullpool.pbs`. It fits name/address character 3-5 gram TF-IDF
on an unlabeled 20k-target prefix per country, then streams all 10.32M
training targets in 50k-record chunks, keeping top100 per route for 200 fresh
references per country. It will report true full-pool top10/20/50 and union
recall, total runtime, chunk count and peak memory. It scanned all 10,320,219
targets. On the same 400-reference sample used for the posting-cap benchmark,
top-50 link recall was about **94.27%**, name/address union recall about
**96.17%**, and top-10 recall about **89.12%** (1,379 known links). Total
runtime was about **1,102s / 18.4min**; sparse multiplication took about 29s,
so parsing, vectorization, and scanning dominate. Its two routes omit exact
and numeric blocking; compare their missed-link types before committing to it.
Linux `sparse-dot-topn==1.1.5` and `psutil==7.1.1` wheels are in
`padum/wheels/` for offline installation. Local smoke run with that exact
sparse-dot-topn version and 20 unit tests passed. Full Padum report is at
`artifacts/sparse_fullpool_v1/report.json`; local copy not yet downloaded.

The prepared miss audit in
`business_entity_resolution/src/audit_sparse_misses.py` ran via
`padum/audit_sparse_misses.pbs`. It reads the
saved 400-reference candidate lists, classifies each missed gold link as
`lost_at_50` or `absent_from_routes`, scans target files once to attach the
original/normalized text for missed cases, and writes
`artifacts/sparse_miss_audit_v1/{summary.json,misses.jsonl}`. Labels are read
only for this post-retrieval audit. All 21 local tests passed after this edit.

The audit completed as job `1065268.pbshpc`: 1,300/1,379 gold links were in
the final top 50. Of 79 misses, 26 were in the name/address union but cut by
the final top-50 rank; 53 were absent from both routes. India accounts for
60 misses (14 cut, 46 absent); US for 19 (12 cut, 7 absent). There were no
cross-country missed links; 24 missed pairs had a non-ASCII character in at
least one original business name. No missed pair had an exactly equal folded
name or folded address. These counts alone do not prove a cross-script cause;
inspect individual `misses.jsonl` records before choosing a rescue route.

The individual file was downloaded to `research/sparse_misses.jsonl` and
analyzed with `research/analyze_sparse_misses.py`. The 79 missed links belong
to 57 distinct references. Among the 53 absent-from-route links, 19 have a
non-ASCII business name, 39 share at least one address number, and 19 still
have name-token Jaccard >=0.5. Examples include Latin-to-Devanagari/Tamil
name variants, web-domain names replacing legal names, shortened addresses,
and high-overlap names crowded out by similar companies. These categories
overlap; they are not additive. A number/address anchor and a cross-script
fallback are plausible, but must be evaluated without using labels to select
candidates. Existing streaming TF-IDF scans all targets per 400 references,
so the next candidate generator must reuse a persistent target index.

Use the audit to guide a persistent or otherwise reusable target retrieval structure. The
current script scans the full target pool for just 400 references and must
not be repeated thousands of times for all 1.73M test references. Benchmark
all-reference throughput before committing to it. Add exact/address-number
routes only where the audit shows they help, build a larger entity-safe
validation set, and then a complete validator-passing submission.

A first reusable-index benchmark is now implemented in
`business_entity_resolution/src/selective_index_benchmark.py` with job
`padum/selective_index_benchmark.pbs`; see
`business_entity_resolution/SELECTIVE_INDEX.md`. It queries the existing
complete `artifacts/retrieval_v1/targets.sqlite` index with selective name
AND and address-number-plus-word AND expressions, a two-second per-query
timeout, and a fresh 400-reference sample excluding the earlier samples.
It reports recall and throughput without a target rescan. All 22 local tests
passed. **Padum benchmark pending**; do not claim its recall or scalability
until the report is checked.

## FINAL (26 Sep, 05:10): validated submission files ready

- `artifacts/submission_v4_final/matching_results.tsv` (97 MB, sha256 6327cae1…, identical to
  the laptop copy) + `candidate_pairs.tsv` (6.8 GB, no empty rows). **Validator PASS** on both
  files (all rules + matches within candidates) and `--check-ids` PASS on the matching file.
- Test per-country predictions: France no-match 5.34% / 3.34 matches per reference, India
  5.66% / 3.34, US 5.64% / 3.35 (train truth: 5.6% / 3.46). No France cutoff applied.
- Disk: 100.4 GB (over the 100 GB soft limit, 7-day grace). `full_v4_s0/s1` (7.2 GB) can be
  removed after the leaderboard score is confirmed (ask the user).

## Earlier (26 Sep, about 05:00): final test files

- **Final test run done:** shards `1066141`/`1066142` scored all 1,732,544 test references
  with v4 XGBoost (GPU) at about 90-160 references/s per shard. Shard 1 merged at about 03:52.
  **`matching_results.tsv` was written and passes the organizer validator** (1,732,544 rows;
  97,110 empty = 5.6%, the same as train's no-match rate). It is at
  `artifacts/submission_v4/matching_results.tsv` and was downloaded to the laptop
  (`C:\Users\visha\hackathon\mlchallenge\matching_results.tsv`) for the leaderboard.
- The `candidate_pairs.tsv` write **hit the 110 GB hard disk quota**. With the user's OK,
  obsolete items were deleted (v1 train/test indexes, matcher_v1, the cancelled
  train_reverse, dryrun_v1_t, the partial file; about 16 GB), bringing usage to 94 GB.
  Re-merge job `1066181` (`merge_test.pbs`, 120 GB RAM, `SHARDS=a:b`) writes both files to
  `artifacts/submission_v4_final/` and runs the validator twice (both files; then
  `--check-ids` on the matching file).
- Ayush (aib262015) asked for the v4 pairs, the XGBoost model and the e5 checkpoint. Read-only
  ACL commands were given to the user to run; Claude may not grant permissions.

## Earlier (26 Sep, about 01:20): v4 = new best (holdout 0.9731)

- **Dense job `1066089` results:** fine-tune 9 min (loss 3.42 → about 0.002); embedded 24.2M
  records at about 12.5k/s; exact GPU search train + test; GPU stages done 00:12.
- **v4 pairs** (same 100k references as v2): **candidate recall 99.88%** (India 99.86, US 99.90)
  vs v2 94.64%, with only 288 candidates per reference (v2: 223). Text search top 200 with
  the combined route: 94.57%; + text reverse: 95.16%; + embeddings: 99.88%.
- **Matchers on v4 pairs:**
  - **XGBoost on the A100** (`--backend xgboost`, 255 leaves, lr 0.05, best iteration 732,
    7.5 min, job `1066129`): **dev 0.9756 / holdout 0.9731** (US 0.9749, India 0.9713),
    expected@0.75. Top gains are all embedding features (dense_rev_inv_rank, dense_score,
    dense_rev_top1, …). Loss analysis: a perfect matcher would score 0.9996; blocking loss
    0.04 pts (64 links); the rest is matcher misses (2,674 links, 1.67 pts), false positives 0.57,
    singleton 0.24.
  - LightGBM v4 (CPU) reached a similar dev logloss (0.00209, iteration 1942); its job was
    cancelled for speed before scoring.
- **XGBoost backend:** `Scorer`/`fit_matcher`/`load_scorer` in `submission_pipeline.py`;
  wheel `padum/gpu_wheels/xgboost-2.1.4-…manylinux_2_28…whl` (CUDA build, Apache-2.0,
  installed `--no-deps`).
- **Final test run (running since about 01:15):** `1066141` (builds `test_index_c`, then
  shard 0) and `1066142` (waits for the index, then shard 1). Both use `MODEL=matcher_v4/model_xgb`,
  `DENSE=artifacts/dense_v1`, REVERSE; the second shard to finish merges + validates →
  `artifacts/submission_v4/`. v2 shards were cancelled (user OK).

## Earlier (25 Sep, about 23:00): dense retrieval (embeddings) on the GPU

- **Why:** the matcher is saturated (a bigger LightGBM gave +0.0001), and lexical blocking
  gains have plateaued. Fine-tuned multilingual embeddings target cross-script and
  semantic misses. The user approved running it on Padum (AWS was considered and dropped).
- **Model:** `intfloat/multilingual-e5-small` (MIT, 118M params) in
  `models/multilingual-e5-small/` on Padum (sha256 1a55775f…).
- **Environment:** torch 2.4.1+cu118 + transformers 4.46.3 + the NVIDIA cu11 libs + triton
  as offline wheels in `padum/gpu_wheels/` (36 files, about 2.8 GB). `pip download
  --platform` skips Linux-only markers on Windows, so the NVIDIA wheels were fetched
  separately with `--no-deps`. `dense_chain.pbs` installs them into `.conda-er` if torch
  is missing.
- **Code:** `src/dense_retrieval.py`. Stages:
  - `finetune`: symmetric InfoNCE, in-batch negatives within one country, 400k references
    excluding the matcher's 100k sample + all earlier samples, 4,000 steps × 256.
  - `embed`: fp16, original name | address text, targets in index ordinal order.
  - `search`: exact GPU matmul top-k per country, forward top 50 + reverse top 10.
  Pipeline: `--dense` adds `Dense`/`merge_dense` candidates and embedding features
  (emb_cos, gap, rank, dense ranks). 53 tests pass.
- **Running since 23:01:** `1066089` on scai03 (**A100-SXM4-80GB**, driver 595 / CUDA 13.2, 8 CPUs,
  64 GB). The first attempt `1066087` failed at install: huggingface-hub 0.36 needs `hf-xet`
  (a Linux-only marker skipped on Windows). It was replaced by huggingface-hub 0.26.5 and
  verified with `pip install --dry-run`. v2 shard 0 was cancelled (the user's decision);
  shard 1 `1066045` keeps running.
- **Job:** `padum/dense_chain.pbs` (GPU stages → **matcher v4** = v2 + combined route +
  dense, same 100k references → holdout → `test_index_c`). It is submitted when a slot
  frees: the user chose to cancel v2 shard 0 at that point (shard 1 `1066045` continues).

## Earlier (25 Sep, about 22:00)

- **Route check done** (`1065959`, 20k of v2's references, 68,948 links): forward recall
  at top 200 was v2 routes **93.79%** (US 96.84, India 90.74); + phonetic 93.67% (worse:
  sound-alike candidates crowd out real ones, **dropped**); **+ combined name+address
  94.56%** (US 97.85, India 91.26; top 50 +1.05, top 100 +1.08); all routes 94.42%.
  Report: `artifacts/route_check_v1/report.json`.
- **Bigger LightGBM** (`1065993`, 127 leaves, stopped at 1,237 rounds): holdout
  **0.9477** vs v2's 0.9476. No gain; the matcher is saturated with the current features.
- **Decision:** v3 = v2 + combined route (no phonetic, v2's LightGBM settings). Expected
  gain is modest (+0.3-0.6), since reverse search already recovers some of the same links.
- **scai_q allows at most 2 jobs per user, queued + running** (a 3rd qsub is rejected).
  `qsub -v` splits on commas, so the v3 job now takes `ROUTES=name:address:anchor:combined`.
- 22:00: shard 0 was resubmitted with 2 CPUs as **`1066053` (running on scai02)**, since only
  2-CPU slots were free; shard 1 `1066045` (4 CPUs) is still queued.
- **Originally queued:** the v2 full test run as 2 shards (`1066044` `full_v2_s0`, `1066045`
  `full_v2_s1`, 4 CPUs each) → **the shard that finishes second runs merge + validator**
  → `artifacts/submission_v2/`, a safe submission file. v3 is submitted when a slot frees:
  `qsub -v ROUTES=name:address:anchor:combined padum/v3_train_chain.pbs`.

## Earlier (25 Sep, about 20:05): the user delegated "do whatever is best, give final best results"

- **Running:** route check `1065959` (scai02, 2 CPUs); recall results about 21:00.
- **Queued:** `1065993` `padum/train_big.pbs` retrains v2 K=200 on its saved pairs with
  127 leaves, lr 0.06, up to 4,000 rounds, early stopping 100 →
  `artifacts/matcher_v2/model_k200_big` + loss analysis (same holdout as v2).
- **Ready, not submitted:** `padum/v3_train_chain.pbs` reuses `train_index_c`
  (from the route check), builds pairs on the same 100k references
  (`ROUTES=` chooses routes; the model remembers them and predict defaults to
  them), trains K=200 (`LGB_ARGS=` for the winning LightGBM settings), runs
  the loss analysis, then builds `test_index_c`.
- **New:** `--country-threshold` (e.g. `France=0.55`) in predict/merge applies a
  per-country decision cutoff at merge time without re-scoring. It is for
  France's under-matching (dry run: 12.2% no-match vs about 5.6% in train). The
  merge report now lists each country's best-probability quantiles. 52 tests pass.
- **Plan:** best of {v2, v2-big, v3} on the same holdout → full test run (2
  shards + merge + validator) → France cutoff decided from the test
  statistics → submission file.

## Shared on GitHub

Team repo: https://github.com/Ayush121511/amazon-ml-challenge (public; default
branch `eda`; teammate branch `ayush-progress`). This workstream is on branch
**`vishal-progress`** (v2 commit `a18a71e`, 25 Sep; **v4 commit `9b74a5c`, 26 Sep: embeddings + XGBoost GPU, holdout 0.9731**), entirely inside
`posting_index_pipeline/`: pipeline code, 42 tests, Padum job scripts, docs and
aggregate result reports (`results/v1`, `results/v2`, test dry run). No data,
artifacts or per-record miss files. Superseded experiments were left out. To
update it, commit the changed files into that folder on the branch; it is not
merged into `eda`.

## Posting index workstream (aib262467, parallel)

Details: `POSTING_INDEX_HANDOFF.md` and `business_entity_resolution/POSTING_INDEX.md`.
Artifacts are in `aib262467`'s `~/scratch/amazon_ml_2026` and are not yet
shared with `aib262140`.

- Version-2 preprocessing of **train and test**, all 24,229,173 rows, done
  (job `1065324`).
- A persistent hashed TF-IDF posting index (name 3-5 grams, address 3-5
  grams, address-number+word anchors) was built once over the full 10.32M
  train targets: 5.6 min, 5.44 GB on disk (job `1065330`).
- 20k fresh references (IDs in `research/posting_index_v1_reference_ids.txt`):
  R@50 90.75%, union 93.92%. Queries run at 418 refs/s, so 1.73M test
  references would take about 70 min (job `1065333`).
- **Same-sample check** against sparse_fullpool's 400 references (job
  `1065364`): R@50 91.59% vs 94.27% and union 94.27% vs 96.17%. The posting
  index is about 340x faster. Most of the gap is India similar names crowded
  out. Combined with sparse, only 2.25% of links are missed.
- **Query sweep** (job `1065375`, 2,000 retrieval_v1 references as a tuning
  set): changing `--top-terms` from 32 to 128 adds only +0.2 points. Raising
  per-route `--top-k` from 100 to 200 lifts union recall from 93.7% to 95.2%
  (US 97.8%, India 92.6%) at about 465 union candidates. Chosen: top-terms 64,
  top-k 200. The remaining gap likely comes from the common-term cutoff, which
  needs a rebuild.
- **Test index built** (job `1065374`, 5 min 45 s, 5.17 GB): 9,969,589 targets,
  matching the official count (France 1,434,993, India 4,717,565, US
  3,817,031).
- **Friend's bundle reviewed** (`team_progress_2026-09-25.zip`, extracted to
  `friend_bundle/`). Their scaled streaming sparse run reached R@50 93.49% on
  10k references, but costs about 300 s of multiplication per 10k references.
  That would be about 15 h for the full test set, against about 70 min for the
  posting index. So the posting index is the submission blocker; their LightGBM
  pilot matcher design was reused.
- **End-to-end pipeline** `business_entity_resolution/src/submission_pipeline.py`
  (stages `pairs` / `train` / `predict`; tests in
  `tests/test_submission_pipeline.py`, 40 local tests pass). It keeps the fused
  top-K candidates, computes about 40 label-free features (route scores/ranks/gaps,
  rapidfuzz name/address/number/postal similarities, missing and script flags),
  and trains LightGBM on an entity-grouped 70/15/15 split. The decision rule
  (threshold vs expected-F0.5 per reference) is chosen on dev; holdout is
  scored once. `predict` writes `matching_results.tsv` and
  `candidate_pairs.tsv` for every reference. rapidfuzz 3.14.6 and lightgbm
  4.7.0 (both MIT) were installed from `padum/wheels/`.
- **Pairs done** (job `1065413`, 7 min 54 s): 60,000 fresh train references
  (30k per country). Fused-candidate link recall: R@50 91.2%, R@100 92.6%,
  R@200 93.8% (India 87.2/89.1/90.7%, US 95.2/96.1/96.8%). Its train stage
  crashed on a gold-lookup bug in dev scoring; it is fixed and covered by new
  tests.
- **Matcher v1 trained** (job `1065464`): macro F0.5 dev/holdout **0.925/0.926**
  (K=100) and **0.926/0.929** (K=200). Rule: expected-F0.5 per reference, floor
  0.60/0.65. Estimated test score about 0.90-0.92, because France is unseen.
  The public leaderboard (25 Sep afternoon) has #1 0.984 and #12 0.973, so the
  main gap is recall: only 92.6-93.8% of true links are ever candidates.
- **Ground-truth structure** (train): **no target is linked to more than one
  reference** (0 of 7,638,365), and 74% of all targets are linked to some
  reference (mean 3.46 per reference, 5.6% of references have none).
- **v2 techniques implemented** (tests: 45 pass locally):
  1. *Reverse blocking* `src/reverse_index.py`: posting index over Source 1
     (`posting_index.py --sources 1`), and every target searches it and keeps
     its top-10 references. Candidates = forward top-K ∪ targets that ranked this
     reference. This rescues look-alike and chain-store targets crowded out of
     a reference's own list.
  2. *Mutual-rank features*: rev_rank, rev_top1, rev_rrf, reverse route
     scores and in_forward. They are computed against all references, so
     sample training stays valid.
  3. *Target exclusivity at predict*: each target goes to its most probable
     reference only (`--no-exclusive` disables it).
  4. A legal-suffix-stripped name similarity (`name_core`).
  5. `predict --sample N`, a random trial subset (per the user: never run the
     full test first), plus per-country no-match/matches/candidates
     diagnostics.
  6. *Offline Indic→Latin transliteration* `src/transliterate.py`, driven by
     Unicode character names (9 Brahmic scripts, schwa deletion, pra/li →
     pvt/ltd). It is applied inside `posting_index.raw_counts` at build and
     query time, so target ordinals are unchanged, and it adds name/address
     transliteration similarity features. On the 1,392 cross-script misses,
     median name token_set similarity rose from 10 to 76.
- The v1 test trial and the non-transliterated reverse job were
  **cancelled**; they were superseded (`train_reverse/` is a partial, unused
  output).
- **scai_q was full** (all GPU slots taken; PBS estimated the 8-CPU train chain would start at 18:50). `1065533` was resubmitted as **`1065571` with 4 CPUs** (scripts now use `$NCPUS`) and started at about 15:03 on scai03.
- `1065571` failed its unit tests. Exclusivity let two identical references
  both keep a target on an exact probability tie. It was fixed with exactly one
  winner per target (ties go to the lowest reference index) and resubmitted as
  **`1065595`** (49/49 tests pass in the job).
- **v1 loss analysis** (8,999 holdout references,
  `artifacts/matcher_v1/model_k200/loss_analysis/`): actual 0.929, and a
  perfect matcher over the same candidates would score 0.975. So **blocking
  costs about 2.5 pts and the matcher about 4.6 pts**. The loss by kind:
  blocking-only 2.15, true candidate rejected 2.04, mixed 1.57,
  false-positive-only 0.72, singleton false match 0.60. Links: the matcher
  rejected 2,399 true candidates, made 564 false matches, and blocking missed
  1,949 of 31,168. The matcher is too conservative. Planned v3: second-stage
  group features, such as a candidate's similarity to the reference's most
  confident match (S2/S3 duplicates of one entity) and its probability rank
  within the reference.
- **Group (anchor) features added** (`group_features` in `submission_pipeline.py`,
  50 tests pass). Each reference's top-2 candidates by a label-free text score
  (name_translit + address token_set + 0.5 × number overlap) become anchors,
  and every candidate gets its similarity to each anchor (transliterated
  name, address, numbers), its proxy rank and gap, and a strong-candidate
  count. This targets the v1 loss of rejected true candidates (S2/S3
  duplicates of one business). It was uploaded before `1065595` reached its
  pairs stage, so **matcher v2 includes these features**.
- **Test v2 blocking done** (`1065534`, 37 min, 4.0 GB peak): `test_index_t`,
  `test_ref_index`, and `test_reverse_t`, where each of 9,969,589 test targets
  keeps its top-10 references (India 47.1M, US 38.2M, France 14.3M pairs).
- **Dry run** `1065636`: prediction path on 100k random test references (v1
  model, `test_index_t`, no reverse), which checks runtime, memory,
  exclusivity and France rates. It is not a submission.
- **Dry run result** (`1065636`, 15.5 min, 7.5 GB, exit 0): 100k test
  references. No-match rate / mean matches were US 5.6% / 3.20, India
  7.4% / 2.93, **France 12.2% / 2.78** (train ground truth: 5.6% / 3.46).
  France looks under-matched, and v1 is conservative overall. Exclusivity
  blocked 637 above-threshold duplicate claims. Speed was about 150
  references/s, so a full test run is about 3.2 h in one job.
- **Sharded prediction:** `predict --shard i/n` scores every n-th reference
  and saves `scores.npz` + `candidates.tsv`. `merge` applies global
  exclusivity + the decision rule and writes the TSVs. A test checks that
  2 shards + merge equal a single run byte for byte (50 tests pass). Jobs:
  `predict_test.pbs` (`SHARD=i/n`, now 32 GB) and `merge_test.pbs` (merge +
  validator). Windows CRLF line endings had broken `predict_test.pbs`; all
  `padum/*.pbs` are now LF and pass `bash -n` on Padum.
- **v2 pairs** (100k refs, 22.3M pairs, 345,756 gold links, 20 min): candidate
  recall at top 100 / top 200 / top 200 + reverse = 92.6 / 93.8 / **94.6%**
  (India 89.1 / 90.7 / 91.7; US 96.1 / 96.9 / 97.6). Reverse search added 2,966
  true links (+0.86 pts at top 200), less than hoped. Transliteration inside
  the index did not raise India forward recall (90.72% vs v1 90.71%).
- **v2 K=100 result: dev 0.9476, holdout 0.9450** (v1: 0.926), US 0.963 and
  India 0.927 (15,113 holdout references). Rule: expected@0.70. Best iteration
  1,999/2,000, so more rounds would still help. Top gain: proxy, proxy_rank
  (group features), rev_inv_rank (reverse), anchor1_address_numbers, rrf,
  transliterated similarities.
- **v3 prepared, NOT started (the user asked to be asked first):** a phonetic
  name route (`transliterate.phonetic`: transliterate → drop legal words →
  sound rules such as soft c/g, x→ks, ph→f → drop non-initial vowels). On the
  1,392 cross-script misses the median token_set rose to 100, 619 are exactly
  equal and 1,217 score ≥80. It is added as a 4th index route + `name_phonetic`
  features (50 tests pass locally, not uploaded). `padum/v3_train_chain.pbs`
  rebuilds `*_index_p` and uses the same 100k references as v2.
- **v2 final (K=200): dev 0.9513, holdout 0.9476** (US 0.9641, India 0.9315),
  expected@0.65, **current best model** (`artifacts/matcher_v2/model_k200`).
  Loss analysis: a perfect matcher over the same candidates would score 0.977.
  Lost points: blocking-only 2.10, rejected true candidate 1.38, mixed 0.74,
  false-positive-only 0.58, singleton 0.42. Links: 2,875 of 51,978 missed by
  blocking, 2,603 rejected by the matcher, 629 false matches. **Blocking is now
  the #1 loss.**
- Route check `1065934` waited about 45 min for 4 CPUs and was resubmitted with 2 CPUs as
  **`1065959`, running on scai02 since 19:38** (about 45-50 min).
- **Route check** (originally `1065934`, `padum/route_check.pbs`;
  experiment `1065895` is on hold so this goes first). It adds a **combined
  name+address route** (`route_text`: one TF-IDF vector over name + address,
  so chain branches that tie on name are separated by address) and the
  phonetic route to `train_index_c`, then compares forward recall on 20k of
  v2's references for v2_routes / +phonetic / +combined / all
  (`src/blocking_recall.py` → `artifacts/route_check_v1/`). 52 tests pass.
  **Rule agreed with the user: if recall does not clearly improve, keep v2
  K=200.**
- **Decision (user):** judge changes by train holdout F0.5 and run the test
  set only once, with the final model. The 100k test trial `1065793` was
  cancelled.
- (Cancelled by the user's decision, 25 Sep about 19:55: low value, and it would take a job slot.) Learning-curve experiments (`padum/train_experiments.pbs`, new
  `train --train-refs/--num-leaves/--learning-rate/--min-data-in-leaf/--early-stopping`):
  `1065895` (queued, waiting for a GPU slot) trains on 25k of v2's training
  references with the same dev/holdout. Variant A uses v2's settings; variant B
  uses 127 leaves, lr 0.08, up to 4,000 rounds. Output:
  `artifacts/matcher_v2/exp_25000_{a,b}`.
- (Superseded) `1065793` (held until `1065595` succeeds) runs
  the v2 test trial on 100k references with model_k200 + reverse → `artifacts/trial_v2/`.
- **Running (25 Sep, from about 15:20):** `1065595` `padum/v2_train_chain.pbs`: v1 loss
  analysis → `train_index_t` → `train_reverse_t` → matcher v2 pairs (50k/country)
  → models K=100/200 (2,000 rounds) → v2 loss analysis. `1065534`
  `padum/v2_test_chain.pbs`: `test_index_t` → `test_ref_index` → `test_reverse_t`.
- Next: compare v2 holdout F0.5 with v1 (0.929), run a sampled test trial
  (`predict_test.pbs` with SAMPLE and REVERSE, pointed at the `_t` index),
  then the full test with validator, then submit.

## Work still required for a real submission

- Run the full-test `predict` with the trained matcher (the pipeline and test
  index now exist). France is unseen in training and needs explicit checks of
  its match rate.
- Larger entity-safe validation, including singletons, missed candidates,
  non-ASCII/cross-script cases, and country-transfer checks.
- Retrain/calibrate a matcher using the chosen full-scale candidate generator.
- Generate and validate both required TSV outputs,
  submit to the portal, and package reproducible code/documentation.

**There is no leaderboard submission or confirmed top-50 result yet.**

For a new chat, say: “Read `CHALLENGE_DATA_AND_RULES.md` and
`PROJECT_STATUS_AND_HANDOFF.md` in `E:\Amazon_ML_2026`. Check the latest
Padum blocking v2 job/report, then continue building a scalable entity
resolution pipeline. Preserve the existing validation boundaries.”
