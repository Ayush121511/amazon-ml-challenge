# Posting-index pipeline (vishal-progress)

A complete, label-safe entity-resolution pipeline for the challenge: blocking →
pair features → LightGBM matcher → `matching_results.tsv` + `candidate_pairs.tsv`.
It is self-contained in this folder and does not touch the repo's `src/`.

Detailed history and every number: `PROJECT_STATUS_AND_HANDOFF.md` (running
log), `POSTING_INDEX_HANDOFF.md` and `business_entity_resolution/POSTING_INDEX.md`.

## Current results (train holdout, macro F0.5)

The model is trained on 100k random train references (50k US, 50k India) and
split 70/15/15 into train/dev/holdout by reference. The decision rule is chosen
on dev, and holdout is scored once. The score is end to end: blocking misses
and no-match references count.

| Model | Holdout F0.5 | US | India | Notes |
|---|---:|---:|---:|---|
| v1 | 0.929 | – | – | forward blocking only |
| v2 (K=200) | 0.9476 | 0.964 | 0.932 | + reverse search, group/anchor features, transliteration, mutual ranks |
| **v4 (XGBoost on GPU)** | **0.9731** | **0.975** | **0.971** | + combined name+address route, **fine-tuned multilingual-e5 embeddings** (candidates + features) |

v4 raises candidate recall from 94.6% to **99.88%** with about 288 candidates per
reference. In its loss analysis, a perfect matcher over the same candidates would
score 0.9996, so blocking costs only 0.04 points; what remains is the matcher
rejecting some true candidates (1.7 pts) and false matches (0.8 pts).

## How it works

1. **Preprocessing** (`src/preprocess.py`): Unicode-normalised and folded
   name/address, numbers and postal candidates. The originals are kept.
2. **Blocking** (`src/posting_index.py`): a persistent hashed TF-IDF posting
   index built once per split and country (so France needs no special
   handling). Routes: name 3-5 grams, address 3-5 grams, address-number+word
   anchors; newer routes are a phonetic name skeleton and a combined
   name+address route. Indic scripts are transliterated to Latin by
   `src/transliterate.py` (rule-based from Unicode names, fully offline).
3. **Reverse blocking** (`src/reverse_index.py`): every target searches an index
   of Source 1 and keeps its top-10 references. A reference's candidates are its
   forward top-K ∪ the targets that ranked it.
4. **Dense retrieval** (`src/dense_retrieval.py`, GPU): `intfloat/multilingual-e5-small`
   (MIT, 118M params) is fine-tuned with in-batch negatives (same country) on train
   pairs. The matcher's evaluation references are excluded. All 24.2M records are
   embedded in fp16, and an exact GPU matmul top-k runs per country in both directions
   (reference → top 50 targets, target → top 10 references). This adds candidates and
   embedding features. On an A100: fine-tune 9 min, embed about 45 min, search about 20 min.
5. **Features + matcher** (`src/submission_pipeline.py`): about 110 label-free
   pair features (rapidfuzz similarities, route scores/ranks, reverse ranks,
   group/anchor features) and LightGBM on an entity-grouped split. The decision
   rule (expected-F0.5 per reference) is picked on dev. `--backend xgboost` trains and
   scores on the GPU (XGBoost 2.1.4 CUDA build, Apache-2.0); it is the v4 model.
6. **Prediction:** global target exclusivity (in train, a target never links to
   two references), sharded scoring + `merge`, then the organizer's validator.

## Running on Padum (IITD HPC)

The job scripts assume this folder is the project root on scratch, e.g.
`~/scratch/amazon_ml_2026`, with the dataset at
`6ab10eb3b23ba_student_resource/student_resource/dataset`
and a Python 3.11 env at `.conda-er`. `scai_q` needs `ngpus=1` per job even for
this CPU-only code.

```bash
qsub padum/preprocess_all_cpu.pbs                    # train + test preprocessing
qsub padum/v2_train_chain.pbs                        # indexes, reverse search, pairs, train v2
qsub padum/v2_test_chain.pbs                         # test indexes + reverse search
qsub padum/route_check.pbs                           # train index with the combined route
qsub padum/dense_chain.pbs                           # GPU: fine-tune + embed + search; v4 pairs
qsub -v PAIRS=artifacts/matcher_v4/pairs,OUT=artifacts/matcher_v4/model_xgb padum/train_xgb_gpu.pbs
# Final test run: 2 shards; the second to finish merges and runs the validator.
qsub -v MODEL=artifacts/matcher_v4/model_xgb,RUN=full_v4_s0,SHARD=0/2,REVERSE=1,INDEX=test_index_c,DENSE=artifacts/dense_v1,BUILD_INDEX=1,MERGE_RUN=submission_v4,OTHER=artifacts/full_v4_s1 padum/predict_test.pbs
qsub -v MODEL=artifacts/matcher_v4/model_xgb,RUN=full_v4_s1,SHARD=1/2,REVERSE=1,INDEX=test_index_c,DENSE=artifacts/dense_v1,WAIT_INDEX=1,MERGE_RUN=submission_v4,OTHER=artifacts/full_v4_s0 padum/predict_test.pbs
```

GPU packages (torch 2.4.1+cu118, transformers 4.46.3, huggingface-hub 0.26.5, the
NVIDIA cu11 libraries, triton, xgboost 2.1.4 CUDA) are installed offline from wheels;
`dense_chain.pbs` installs them if missing. On Windows, `pip download --platform`
skips Linux-only dependency markers, so fetch the `nvidia-*-cu11` wheels explicitly.
The fine-tuned embedding model (466 MB) and the XGBoost model (26 MB) are on Padum
under `artifacts/`; they are not committed.

Tests: `python -m unittest discover -s business_entity_resolution/tests` (43 tests; XGBoost + LightGBM included).
Dependencies: `requirements.txt` in this folder.

## Results files (`results/`)

| File | What it is |
|---|---|
| `v4/model_xgb_report.json` | **Best model (v4)**: decision rule, dev/holdout F0.5, per-country holdout, feature gains |
| `v4/pairs_report.json`, `v4/loss_summary.json` | v4 candidate recall (99.88%) and holdout loss split |
| `v4/dense_*.json` | Embedding fine-tune and GPU search reports |
| `route_check/report.json` | Forward recall of route combinations (combined kept, phonetic dropped) |
| `v2_big/model_report.json` | Larger LightGBM on v2 pairs (no gain: 0.9477) |
| `v2/model_k200_report.json` | v2 model: decision rule, dev/holdout F0.5, per-country holdout, feature gains |
| `v2/model_k100_report.json` | Same with the top-100 candidate cap |
| `v2/pairs_report.json` | Candidate recall on the 100k training references (forward top-K, + reverse) |
| `v2/loss_summary.json` | Holdout loss split: blocking vs matcher vs false matches, oracle F0.5 |
| `v1/*` | The same for v1, the baseline |
| `test_dryrun_v1/report.json` | 100k-reference test dry run: per-country no-match rate / matches, timing |
| `blocking_sweep_terms64_k200_report.json` | Chosen blocking settings on 2,000 references |
| `research_samples/` | Shared reference-ID lists (fixed seeds) and the same-sample comparison with the sparse TF-IDF baseline |

## Not included

This folder has no raw or preprocessed data, indexes, candidate files, models,
per-record miss files (they contain dataset text), or wheels. Earlier
experiments that were superseded (SQLite FTS search, capped inverted lists,
streaming sparse TF-IDF, the pilot matcher) are left out; the history is in
`PROJECT_STATUS_AND_HANDOFF.md`.
