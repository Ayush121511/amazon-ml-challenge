# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [TODO]
**Team Members:** Ayush Srivastava, Vishal [TODO surname], Tapan [TODO surname]
**Submission Date:** 28 September 2026

> Draft, 26 Sep evening. Matcher sections (2.1, 4, 5) are written from measured results; blocker (3) is
> Vishal's; validation protocol and count model are Tapan's. Final numbers get updated after the last submission.
> Must be trimmed to 1–2 pages.

---

## 1. Executive Summary

We match each Source-1 business to its Source-2/3 records with a three-stage pipeline: (1) multi-route retrieval
(TF-IDF, reverse search, fine-tuned multilingual-e5 embeddings) reaching 99.88% recall, (2) a retrieval-only
XGBoost pruner that keeps about 10 candidates per reference at 99.3% recall, and (3) a matcher that stacks
fine-tuned **cross-encoders** (mdeberta-v3-base and others reading both records jointly) on a 103-feature XGBoost.
A per-reference expected-F0.5 decision rule with global target exclusivity turns probabilities into match lists.
The cross-encoder lifts holdout macro F0.5 from 0.973 to 0.985 [TODO final].

---

## 2. Methodology

### 2.1 Problem Analysis

- **Metric shape.** F0.5 is macro-averaged per reference and weights precision twice; one false link on a true
  singleton scores 0. References with a single true link are the most fragile group (holdout F0.5 0.911 vs 0.98
  for 5+ links).
- **Where the loss is.** On the top-10 candidate set the matcher's *ranking* is nearly right: choosing the best
  top-k per reference from the same probabilities would score 0.9946 vs 0.9728 achieved. Most loss comes from
  stopping one link early (1,981 of 15,113 holdout references return one link too few).
- **Missed links** are mostly weaker sibling copies of an already-accepted record (96% of misses) whose address is
  empty on one side (45% vs 1.7% of accepted true links) or whose house numbers disagree (24% vs 6%), often through
  truncation (2758 → 275).
- **False links** are confident look-alikes (median probability 0.89): 38% are records of a different, near-duplicate
  Source-1 business; 62% belong to no Source-1 record at all (distractors). Test has more distractors per reference
  than train (5.75 vs 4.68).
- **Calibration** of the XGBoost matcher is already excellent (ECE 0.002); threshold, per-country and class-weight
  tuning each moved holdout by less than 0.0005.
- **Data quirks:** fragmented Devanagari spacing in India names, generic name tokens (services, center, india),
  postal codes present in only 5–8% of addresses, and France appearing only in test.

### 2.2 Solution Strategy

**Approach Type:** Blocking + pruning + stacked classifier (hybrid lexical, dense and cross-encoder).
**Core Innovation:** Fine-tuned cross-encoders stacked on a feature-based XGBoost, trained out-of-fold, with a
late-fusion path that reuses the saved first-stage probabilities, so the text model can be added to test inference
without recomputing features.

---

## 3. Candidate Generation (Blocking) — [TODO Vishal]

- **Retrieval routes:** [TF-IDF name/address/anchor/combined, reverse index, fine-tuned multilingual-e5 dense
  forward top-50 / reverse top-10] — about 290 candidates per reference, 99.88% recall on sampled train references.
- **Stage-1 pruning:** XGBoost on retrieval signals only, out-of-fold on train; top-10 set keeps 99.32% recall.
- **Candidate pairs generated (test):** [TODO]
- **How true matches were not lost:** [TODO]

---

## 4. Matching Model

**Features used (first stage, 103):**
- Name: folded/normalized/core/transliterated/phonetic token-sort, token-set, partial, Jaro-Winkler, length
  ratios, first-token equality, empty flags.
- Address: folded and transliterated token-set, number-token overlap, postal candidates, empty flags.
- Retrieval and embedding: route scores, ranks and gaps, reciprocal-rank fusion, reverse ranks, e5 cosine and
  ranks (forward and reverse), anchor/proxy features.

**Cross-encoders (second stage).** Each candidate pair is serialized as `name | address` for the reference and
the candidate and read jointly by a fine-tuned transformer (binary cross-entropy, bf16, max 128 tokens):

| Backbone (license) | Alone, holdout | Stacked on XGBoost, holdout |
|---|---|---|
| none | – | 0.9707 |
| multilingual-e5-small, team fine-tune (MIT) | 0.9717 | 0.9819 |
| bge-reranker-v2-m3 (Apache-2.0) | 0.9760 | 0.9831 |
| **mdeberta-v3-base (MIT)** | **0.9784** | **0.9847** |
| Qwen3-Reranker-0.6B (Apache-2.0) | [TODO] | [TODO] |

(Controlled comparison: cross-encoder trained on half of the train references, XGBoost on the other half, decision
rule chosen on dev, holdout scored once.) All models are MIT/Apache-2.0 and at most 0.6B parameters.

**Fusion.** [TODO final: 2-fold out-of-fold cross-encoder scores for every train pair, XGBoost on the 103 features +
cross-encoder scores + set context (rank, gaps, sums within the reference) + sibling features comparing a candidate
with the reference's confident matches.] For test inference we also use a late-fusion model (small XGBoost on the
first-stage probability and the cross-encoder logit), which matched feature-level fusion within 0.0002 holdout.

**Model type:** XGBoost (GPU) stacked on fine-tuned transformer cross-encoders.
**Threshold selection method:** per reference, choose the top-k that maximizes expected F0.5 under the predicted
probabilities, predicting no match when that is more likely, with a probability floor chosen on dev. Each target is
assigned to at most one reference (its most probable). [TODO: count model if adopted — Tapan.]

---

## 5. Results & Error Analysis

- **F0.5 (macro), dev / holdout:** v4 0.9756 / 0.9731; stacked mdeberta 0.9868 / 0.9847; [TODO final fusion].
  Public leaderboard: v4 0.964, [TODO].
- **Common false positives:** same address but a different business (shared buildings), same name with a truncated
  or different house number (branches), and near-duplicate Source-1 references competing for one record. The
  cross-encoder halves false links on pairs that have an address (e5-small: 422 → 201 on holdout).
- **Common false negatives:** sibling copies with an empty address or conflicting house numbers. Recall on
  empty-address true pairs stays around 47–49% for all models, against 98.9% when both addresses are present.

---

## 6. Conclusion

Most of the attainable gain was in the matcher, not the blocker: once candidates were pruned to about 10 per
reference, reading both records jointly with a fine-tuned cross-encoder recovered about 1.2 holdout points, more
than any feature or threshold change. Diagnostics that split ranking error from decision error (oracle top-k,
known-count ceilings) directed effort to the decision layer and to the text model. [TODO lessons on holdout-to-test gap.]

---

## Appendix

### A. Code Artefacts
[TODO final layout] Main entry points (in `business_entity_resolution/src/`):
- `submission_pipeline.py` — retrieval, pairs, first-stage matcher, `finalize()` (exclusivity + decision + TSVs).
- `cross_encoder_exp.py` — cross-encoder fine-tuning per fold.
- `fusion_stage1.py`, `fusion_train.py` — out-of-fold stage-1, set/sibling features, fusion model.
- `score_ce_test.py`, `late_predict.py` / `fusion_predict.py` — test inference to `matching_results.tsv` and
  `candidate_pairs.tsv`.
- Diagnostics: `matcher_eda.py`, `shap_misclassified.py`, `sibling_check.py`, `late_fusion_check.py`,
  `distractor_check.py`.

### B. Additional Results
Ceiling analysis, loss decomposition, SHAP and sampled errors: [link or figures TODO].
