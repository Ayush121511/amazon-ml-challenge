# Tapan's sparse entity resolution experiment

This directory contains the independent Padum pipeline developed for the Amazon ML Challenge 2026. It lives under `experiments/` so the team's existing `src/` implementation stays intact. The code, tests, and PBS scripts are included; datasets, generated candidates, model weights, credentials, logs, and submission TSVs are not.

## What has been built

1. Version 2 preprocessing retains original IDs and fields and adds normalized/folded names, addresses, address numbers, and postal candidates. All three train and test sources were preprocessed and their row counts/manifests checked.
2. Sparse TF-IDF retrieval uses country-specific name and address character n-grams. Target chunks are streamed, and compact NumPy arrays retain the top 100 targets per route and reference. Retrieval never reads gold labels to choose candidates.
3. A LightGBM matcher scores the deduplicated name/address route union using string, numeric/postal, missingness, and route-rank features. Its threshold is chosen on a grouped development split using the official macro F0.5 metric; holdout references and gold-linked entities are kept out of training.
4. Full-test inference caches target vectors and records once, retrieves in checkpointed reference blocks for US, France, and India, scores candidates in batches, and writes both required TSVs. Assembly and streaming checks validate complete reference coverage, valid target IDs, and predictions drawn from the candidate sets.
5. `resume_test_cache.py` repairs a cache interrupted before its manifest was written. It reuses intact chunks and safely rebuilds an interrupted final chunk. Its PBS job runs a small recovery test before touching the partial cache.

## Measured results

| Measurement | Result | Scope |
| --- | ---: | --- |
| Top-50 link recall | 93.32% | 100,000 fresh train references against all 10,320,219 train targets |
| Name/address union link recall | 95.87% | Same benchmark; up to 200 deduplicated candidates per reference |
| Retrieval benchmark time / peak RSS | 67.9 min / 2.22 GiB | Four CPU threads; not a full-test timing estimate |
| Matcher dev macro F0.5 | 0.8945 | 2,000-reference dev split from the 10,000-reference candidate set |
| Matcher holdout macro F0.5 | 0.8964 | 2,000-reference holdout; India 0.8699, US 0.9229 |

These are offline measurements, not a public leaderboard score. No France labels were available for country-specific validation. The matcher uses v1's 33 features; the richer v2 feature set is experimental and has not replaced it. See [CURRENT_STATUS.md](CURRENT_STATUS.md) for the launch state and immediate next steps.

## Code map

- `business_entity_resolution/src/preprocess.py`: version 2 preprocessing.
- `business_entity_resolution/src/sparse_fullpool.py` and `compact_topk.py`: full-pool benchmark retrieval.
- `business_entity_resolution/src/prepare_saved_pairs.py`, `pair_features.py`, `train_sparse_matcher.py`, and `metrics.py`: labeled pair preparation, features, training, and official-style macro F0.5 evaluation.
- `business_entity_resolution/src/test_pipeline.py` and `validate_final_submission.py`: full-test caching, retrieval, scoring, assembly, and final checks.
- `business_entity_resolution/src/resume_test_cache.py`: recovery from the interrupted first cache job.
- `business_entity_resolution/tests/`: synthetic and invariant tests.
- `padum/`: PBS jobs and CPU dependency list.

The full local development suite passed 37 tests before this focused snapshot was staged, including interrupted-cache recovery and a small three-country submission fixture. The included tests cover the pipeline and diagnostics in this directory. Those tests do not measure full-test runtime or leaderboard quality.

## Running environment

The PBS scripts expect this experiment's original project layout at `~/scratch/amazon_ml_2026`, a working `.conda-er` Python environment, the organizer's provided data under its original path, and the preprocessed inputs under `artifacts/preprocessed_v2/`. Copying this directory into the team repository does not itself recreate those datasets or the fitted matcher. The queue requires a GPU allocation, although this pipeline's preprocessing, cache build, sparse retrieval, and LightGBM inference currently run on CPUs.

For the active launch, use the job IDs and checks in [CURRENT_STATUS.md](CURRENT_STATUS.md). Do not resubmit the original five-job launcher. A valid submission requires `artifacts/test_prototype_v1/submission/VALIDATED.json` to report `PASS`, followed by a manual upload of `matching_results.tsv` and `candidate_pairs.tsv` through the challenge portal.
