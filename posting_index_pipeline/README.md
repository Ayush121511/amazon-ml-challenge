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
| **v2 (K=200)** | **0.9476** | 0.964 | 0.932 | + reverse search, group/anchor features, transliteration, mutual ranks |

In v2's loss analysis, a perfect matcher over the same candidates would score
0.977. Blocking recall is 94.6% (US 97.6%, India 91.7%), and blocking is the
largest remaining loss.

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
4. **Features + matcher** (`src/submission_pipeline.py`): about 90 label-free
   pair features (rapidfuzz similarities, route scores/ranks, reverse ranks,
   group/anchor features) and LightGBM on an entity-grouped split. The decision
   rule (expected-F0.5 per reference) is picked on dev.
5. **Prediction:** global target exclusivity (in train, a target never links to
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
qsub -v MODEL=artifacts/matcher_v2/model_k200,RUN=full_v2_s0,SHARD=0/2,REVERSE=1 padum/predict_test.pbs
qsub -v MODEL=artifacts/matcher_v2/model_k200,RUN=full_v2_s1,SHARD=1/2,REVERSE=1 padum/predict_test.pbs
qsub -v MODEL=artifacts/matcher_v2/model_k200,RUN=submission_v2,SHARDS="artifacts/full_v2_s0 artifacts/full_v2_s1" padum/merge_test.pbs
```

Tests: `python -m unittest discover -s business_entity_resolution/tests` (42 tests).
Dependencies: `requirements.txt` in this folder.

## Results files (`results/`)

| File | What it is |
|---|---|
| `v2/model_k200_report.json` | Best model: decision rule, dev/holdout F0.5, per-country holdout, feature gains |
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
