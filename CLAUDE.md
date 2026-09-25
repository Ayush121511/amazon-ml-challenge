# Amazon ML Challenge 2026

## Status
- Repo structure set up (see README.md)
- HPC (IIT Delhi PADUM) access done and verified: A100 80GB GPU confirmed working
  on queue `scai_q`, project `scai`. Team setup guide pushed to
  https://github.com/Ayush121511/padum-setup — HPC setup itself is DONE, don't
  redo it unless something breaks.
- Miniconda env `amlc` created on HPC (`~/scratch/miniconda3`, python 3.11);
  `numpy pandas scikit-learn lightgbm rapidfuzz pyarrow` installed there via the
  research proxy (see `references/hpc/proxy.sh`, needs your own filled-in password,
  never commit it).
- **Dataset dropped 25 Sept 2026** (same day as Round 1 start). Problem confirmed:
  **Business Entity Resolution** — match noisy business records from Source 2/3
  against the deduplicated Source 1 reference. Full problem statement mirrored at
  `data/student_resource/README.md`; original PDF at repo root.
- Currently on the **`eda` git branch**. All pipeline code (`src/`), the dataset
  (`data/`), notebooks, and HPC job scripts are **untracked/uncommitted** — nothing
  has been committed since the dataset drop. Commit deliberately when ready, not
  automatically.

## The problem, precisely
- 3 source files: `entity_id` (prefix S1-/S2-/S3-), `business_name`,
  `business_address`, `country`. Ground truth: `source1_entity_id` →
  comma-separated `matched_entity_ids` (empty = singleton, no match).
- Metric: **F_0.5** (precision weighted 2x over recall), macro-averaged per
  Source-1 entity. Singletons count — predicting empty correctly = 1.0, any false
  match on a true singleton = 0.0. This punishes over-matching hard.
- Output: `matching_results.tsv` (scored) + `candidate_pairs.tsv` (blocking
  candidates, audited not scored, final matches must be a subset of it). Local
  validator: `data/student_resource/utils/validate_submission.py`.
- Hard constraints: final model ≤8B params, MIT/Apache-2.0 license; **zero
  external lookups** (no geocoding, no registry APIs, no internet augmentation) —
  disqualification risk.
- Scale: train has 2,206,821 S1 entities (~5M each S2/S3); test has 1,732,544 S1
  entities (~5M each S2/S3). Train is US/India only; **test adds France** with zero
  training examples — country must stay an open string set, never hardcoded.

## EDA findings (see `notebooks/eda_basic.md` on the `eda` branch for the full report)
- **Singleton rate 5.6%** (123,247 of 2,206,821); match-count distribution peaks
  at 3 matches, avg 3.67 among non-singletons. 92%/93% of non-singleton entities
  have ≥1 S2/S3 match respectively, 85% have both — genuinely many-to-many.
- **Country distribution shifts between train and test**, not just "France added":
  train is US 60%/India 40%; test *excluding France* is roughly US 45%/India 55%.
- **Name similarity of true matches** (RapidFuzz `token_sort_ratio` on all 7.6M
  true pairs): mean ≈80, median ≈90, but a real low-similarity tail exists —
  ~10-13% of true matches have similarity ≤50, concentrated as a distinct cluster
  around 10-20%, not a smooth tail. ~175-243 pairs (~0.01%) have *zero* name
  similarity at all — a small number of true matches are only findable via
  address, never via name.
- **Devanagari tokenization artifact**: India-source Hindi business names have
  anomalous per-syllable spacing (e.g. the fragment `पर इव`, part of "प्राइवेट"/
  Private, appears as a "name" 1,850+ times). This breaks whitespace tokenization
  for that subset, and later turned out to be the exact root cause of a real
  MinHash/Jaccard failure mode (see below) — 7.9% of India names share this
  fragmented "Private Limited" suffix.
- **Generic name-token pollution**: most frequent core-name tokens are `center`,
  `com`, `partners`, `services`, `and`, `of`, `associates`, `india`, `care` — none
  were in the legal-suffix stopword list; token-based blocking has to actively
  guard against these (see blocking section).
- Missingness: ~3.3% of S2/S3 addresses empty; postal-code presence only 5-8%
  everywhere — postal-based blocking has low coverage on its own.

## Blocking / candidate-generation approach — progress log
Pipeline shape: normalize → block (generate per-S1-entity candidate S2/S3 ids) →
pairwise features → classifier (not built yet) → threshold tuned for F_0.5.

**Normalization** (`src/normalize.py`): lowercase, ASCII-fold accents, strip
punctuation/apostrophes, tokenize, strip a hardcoded `LEGAL_SUFFIXES` set
(ltd/pvt/llc/corp/sarl/gmbh/etc. — English/French coverage only, **no Devanagari
suffix coverage**, a known gap). Address: same cleaning + a heuristic postal-code
guess (any 5-6 digit number, last one wins — not validated, can misfire on house
numbers). `src/build_normalized.py` caches results to `data/processed/*.parquet`.

**Exact-match blockers** (`src/blocking.py`): `compact_name` (full cleaned name),
`postal_country`, `first_token_country` (first word of name), `token` (any shared
core word) — all joined AND partitioned by country (verified safe: 0 of 7.6M true
matches cross a country boundary in train).

**Two production incidents, same root cause, now fixed**: a blocking key with no
bound on `freq_s1 × freq_other` per value can blow past any memory budget — a
word appearing ~3,000× on each side alone creates 9M pairs. Hit this twice on
HPC: first in `token_candidates` (~219GB against a 64GB job, killed live), then
in `first_token_country_candidates` after patching only the first one (~243GB,
also killed live). Fix: all four blockers now go through one shared
`_bounded_merge_candidates` helper with `MAX_KEY_FREQ=200` /
`MAX_PAIR_PRODUCT=40,000` caps. Validated safe at full scale after the fix
(peaked at ~31GB doing the real 2.2M×5M join). **Lesson applied going forward**:
`src/blocking_precheck.py` — a Splink-inspired dry-run (`count_comparisons`-style)
that estimates a key's pair count via groupby before ever merging, so a future
blowup gets caught before it happens, not mid-run.

**Full-scale blocking recall (train, after the fix)**:

| Strategy | Recall vs S2 | Recall vs S3 | Entities w/ ALL matches (S2/S3) |
|---|---|---|---|
| compact_name | 48.68% | 46.94% | 31.76% / 28.99% |
| postal_country | 4.77% | 4.90% | 4.15% / 4.20% |
| first_token_country | 35.52% | 33.53% | 30.60% / 27.15% |
| token | 38.70% | 38.14% | 35.08% / 34.02% |
| **UNION (4 strategies)** | **65.85%** | **65.78%** | **53.91% / 52.79%** |

Avg ~98 candidates/entity. Honest caveat: full-scale recall is meaningfully
*lower* than a 3%-stratified dev sample predicted (dev sample said ~77% union
recall) — full-scale data has far more generic-key collisions that the safety
caps correctly have to prune, at a real recall cost. ~34% of true matches are
currently structurally unreachable by blocking alone.

**Dev-sample methodology** (`src/make_dev_sample.py` →
`data/processed/dev_sample/`): a 3%-stratified (by country × singleton-status)
sample of train S1, with *every* true S2/S3 match preserved (never invents false
negatives) plus an independent 3% random background slice of S2/S3 to keep
collision-rate structure realistic. Built specifically so blocking/feature
experiments iterate in seconds locally instead of a 20-40 min HPC round-trip —
use this for any new blocking/feature idea before running at full scale, but
remember it *underestimates* generic-key collision risk (proven twice now), so
memory-safety claims still need a full-scale check.

**TF-IDF (character n-gram) + SimHash (Random Projection LSH)** — explored as a
complementary blocking channel, not a replacement:
- Rationale: exact-match blocking needs hand-tuned frequency caps because cost is
  driven by the data's own skew. TF-IDF's IDF term automatically down-weights
  common n-grams, and SimHash's cost is set by *parameters we choose* (bits per
  table, number of tables), not by data skew — directly targets the failure mode
  that caused both HPC incidents.
- Dev-sample result: SimHash alone (6 tables × 16 bits) gets 52.18%/51.10% recall
  at only ~28-32 candidates/entity — beats every individual exact-match blocker
  per-candidate-spent. **Combined with the exact-match union**: 76.90%→**78.47%**
  (S2), 75.99%→**78.15%** (S3) — a real, if modest, improvement.
  Implementation: `src/simhash_blocking.py`, `src/dev_simhash_check.py`.
- **Full-scale run in progress** (`src/run_simhash_full.py`,
  `references/hpc/simhash_job.sh`) — hypothesis: SimHash's recall shouldn't
  degrade at full scale the way the exact-match blockers' did, since its cost
  isn't governed by data skew. Check job history / re-run for current numbers.
- **Hyperparameter sweep in progress** (`src/tune_simhash.py`) over bits-per-table
  × num-tables on the dev sample — fewer bits (e.g. 10 vs 16) traded ~1200%+ more
  candidates for meaningfully higher recall (59%+ vs 52%) in early results; full
  sweep results TBD, check output before picking final params.

**MinHash + LSH (Jaccard similarity)** — tried once (`src/minhash_lsh_blocking.py`,
`src/dev_minhash_check.py`), **not adopted**: 7 of 8 LSH bands got flagged UNSAFE
by the precheck and skipped, because plain Jaccard/MinHash has no IDF-equivalent
— it weighs every shingle equally, so the Devanagari "Private Limited" fragment
(shared by 7.9% of India names) created a genuine, enormous similarity cluster
that TF-IDF's weighting avoids entirely. Concrete, data-grounded reason to prefer
TF-IDF+SimHash over MinHash+LSH on this specific dataset.

**TF-IDF top-N (exact cosine, `sparse_dot_topn`)** — a second way to spend the
same TF-IDF vectors, safe *by construction* rather than by a post-hoc frequency
cap: `sp_matmul_topn` never returns more than `top_n` matches per S1 row, so
output size is hard-bounded at `n_s1_rows * top_n` regardless of data skew — no
precheck needed the way the hash-bucket approaches (SimHash, MinHash) require.
Trades exactness of the LSH approximation away (real cosine scores, not
hash-bucket collisions) for that hard bound. Implementation:
`src/tfidf_topn_blocking.py`, dev check `src/dev_tfidf_topn_check.py`,
full-scale runner `src/run_tfidf_topn_full.py` /
`references/hpc/tfidf_topn_job.sh` (top_n=75). Full-scale numbers: check job
history / re-run — not yet logged here.

**Production candidate-generation pipeline** (`src/generate_candidates.py`,
`references/hpc/generate_candidates_job.sh`) — until now, every blocking
channel above was only ever measured in isolation by its own dev/full-scale
script; nothing actually produced the `candidate_pairs.tsv` the submission
format requires. This script is the real union: exact-match (`blocking.py`) ∪
TF-IDF top-N (`tfidf_topn_blocking.py`, top_n=75 default), deduped per
S2/S3 source, then grouped per S1 entity into the exact
`source1_entity_id\tcandidate_entity_ids` (comma-joined S2+S3, empty string
for zero-candidate entities) format — includes every S1 entity, not just ones
with hits. `--eval` (train only) prints per-channel recall next to the
combined recall so the lift from adding TF-IDF top-N on top of exact-match is
visible in one run instead of cross-referencing separate logs. SimHash is
deliberately *not* included in this union yet — its full-scale recall was
still unconfirmed, and it's a probabilistic approximation of the same cosine
signal TF-IDF top-N computes exactly; add it as a third channel once
full-scale SimHash numbers justify the extra candidates. Only smoke-tested
against synthetic fixtures so far (no dataset access in this environment) —
run `--eval` on the real dev sample / full scale on HPC before trusting the
combined recall numbers. Fixed a real gap while at it: `requirements.txt` was
missing `sparse_dot_topn`, `scipy`, `pyarrow`, and `rapidfuzz`, all of which
production blocking/EDA code already imports — would have failed on any fresh
environment built strictly from that file (the actual submission
requirement).

**Not yet built**: entity-level train/val split for leakage-safe evaluation,
pairwise feature computation at scale (`src/features.py` exists, untested at
scale), the actual classifier (LightGBM planned) + F_0.5-tuned threshold,
submission validation. Test-set candidate generation is now wired up
(`generate_candidates.py --split test`) but untested against the real test
data (France country coverage, in particular, still unverified end-to-end).

## HPC operational notes learned the hard way
- `scai_q` is the **only** queue our `scai` project has quota on — `standard`/
  `high` explicitly show `max_queued=[p:scai=0]`. Every job must request
  `ngpus>=1` even for pure-CPU work (queue policy, not our choice) — harmless,
  just don't rely on the GPU.
- 4 dedicated nodes: `scai01`-`scai03` (32 cpu/8 gpu/~1TB mem each), `scai04` (72
  cpu/4 gpu/~500GB mem). Check live availability (`pbsnodes -a`) before sizing a
  job — availability shifts constantly with teammates'/other users' jobs.
- `qsub`/`qstat` aren't on the default `$PATH` — the `default` symlink under
  `/opt/pbs/` is broken; use `export PATH=/opt/pbs/2024.1.5/bin:...` explicitly.
- Always add `sys.stdout.reconfigure(line_buffering=True)` to any script run via
  `qsub` with output piped to a file — otherwise output sits in a buffer and you
  can't tell a slow-but-fine job apart from a stuck one until it exits.

## Timeline
- Round 1: 25 Sept 2026 - 28 Sept 2026 (dataset + problem statement drop Day 1)
- Submission: 1-2 page approach doc + code/notebook, zipped
- Grand Finale: 7 Oct 2026 (top 10 teams, live virtual)

## Repo layout
- `data/raw/` - original dataset, untouched (gitignored)
- `data/processed/` - cleaned/feature-engineered data (gitignored)
- `notebooks/` - EDA and experiments
- `src/` - reusable pipeline code (preprocessing, features, models, eval)
- `models/` - saved model artifacts (gitignored)
- `submissions/` - generated submission files (gitignored)
- `references/hpc/` - HPC setup docs/scripts (personal copy, has real Kerberos ID)

## Workflow once dataset drops
1. Drop dataset in `data/raw/`
2. EDA in `notebooks/`
3. Move stable logic into `src/`
4. Track leaderboard metric explicitly, optimize for it directly
5. Get a working baseline submitted early, then iterate
6. For GPU training: push code to HPC, run via `references/hpc/batchjob.sh`
   (already configured for `scai`/`scai_q`)

## Conventions
- No comments unless explaining non-obvious WHY
- Don't over-engineer — 72hr time box, working baseline beats clever architecture

## Prep notes (pre-dataset drop, written 24 Sept 2026)

### Past problem types (pattern: text-only → multimodal)
- 2021: browse node classification, text → ~9919 classes, subset accuracy
- 2023: product length regression (tabular), metric max(0,100*(1-MAPE))
- 2024: entity extraction from product images (weight/volume/voltage/dims), F1. Winner
  (KhadgaA repo): Qwen2-VL-7B-Instruct AWQ, VQA-style prompts ("What is the {entity}?"),
  no heavy OCR pipeline — let VLM read image directly. LLaMA-Factory + QLoRA 8bit,
  batch 8, lr 5e-5, grad accum 8. Heavy manual data curation (fix bad labels, NA for
  missing, ranges → max or NA) mattered more than model choice.
- 2025: price prediction from catalog text + image, SMAPE, external price lookup banned.
  Top solutions: multimodal fusion (CLIP/SigLIP2 image embeddings + BERT/DeBERTa/DistilBERT
  or FLAN-T5 text embeddings) → LightGBM/XGBoost/CatBoost/MLP ensemble on fused features.
  One rank-80/23k team used Unsloth for fast LLM fine-tune + FAISS similarity search for
  inference, validated on 75k local holdout, stressed "data understanding > architecture."
- Takeaway: whatever 2026 problem is, expect text+image multimodal, likely regression or
  extraction. Build baseline that ignores hardest modality first, submit, then add fusion.

### Execution pattern that wins
1. Baseline in first few hours: simplest model that produces valid submission format.
   Submit early even if bad score — validates pipeline end-to-end, avoids last-hour format bugs.
2. EDA next: missing values, label noise, entity/unit distributions, image quality.
   Manual data cleaning consistently outweighs model sophistication in these past challenges.
3. Iterate: swap in stronger encoders / fine-tune, then ensemble at the end.
4. Reserve last few hours for: ensembling, submission format validation, writing the
   1-2 page approach doc — don't leave doc writing to the last 30 min.
5. Track leaderboard metric exactly (SMAPE/MAPE/F1/accuracy) — optimize loss to match it,
   not a proxy (e.g. use MAE/SMAPE loss not plain MSE if metric is SMAPE).

### GPU (A100 80GB, scai_q) optimization checklist
- Use bf16 mixed precision (more stable than fp16 on A100) via `torch.autocast` /
  `torch.amp` — big speedup, minimal accuracy loss.
- `torch.set_float32_matmul_precision('high')` for any fp32 matmuls (TF32 tensor cores).
- Prefer LoRA/QLoRA over full fine-tune for any LLM/VLM — fits in 80GB with room for
  large batch, trains much faster, standard for these challenges (see 2024/2025 winners).
- Unsloth for LLM fine-tuning if applicable — 2025 top team used it for training speed.
- Max out batch size to saturate 80GB before adding gradient accumulation.
- Gradient checkpointing only if OOM — trades compute for memory, don't use unless needed.
- Use HPC batch job (`references/hpc/batchjob.sh`) for any run >~30min so it survives
  disconnects; keep quick iteration/debug loops local or in short interactive sessions.

### Time-box plan (72hr window, adapt once actual dataset size/deadline known)
- Hr 0-4: read problem statement, get dataset into `data/raw/`, minimal EDA, decide metric.
- Hr 4-10: dumbest-possible baseline → valid submission file → submit.
- Hr 10-30: real EDA + data cleaning, pick modality strategy (text-only vs multimodal),
  first real model trained on HPC.
- Hr 30-55: iterate on model/features, start ensembling candidates.
- Hr 55-65: final ensemble, hyperparameter polish, multiple submission attempts.
- Hr 65-72: freeze code, write approach doc, package submission zip, buffer for HPC queue
  delays / bugs.
