# Amazon ML Challenge 2026 — Progress Summary

_Business Entity Resolution. Last updated 25 Sept 2026._

## The problem, precisely

- **Task:** for each Source-1 business, find every matching business in Source 2 and Source 3 (zero, one, or many matches).
- **Metric:** macro **F_0.5** (precision weighted 2x over recall), averaged per Source-1 entity, singletons included. Predicting "no match" correctly on a true singleton scores 1.0; any false match on one scores 0.0. Over-merging is punished hard.
- **Scale:** train 2,206,821 Source-1 entities (~5M each S2/S3); test 1,732,544 entities (~5M each S2/S3). Train is US/India only; test adds France with zero training examples.
- **Output:** `matching_results.tsv` (scored) + `candidate_pairs.tsv` (blocking candidates, audited not scored — final matches must be a subset of it). Local validator: `student_resource/utils/validate_submission.py`.
- **Hard constraints:** final model ≤8B params, MIT/Apache-2.0 license; zero external lookups (no geocoding, no registry APIs) — disqualification risk.

## EDA highlights

- Singleton rate 5.6% (123,247 of 2.2M); avg 3.67 matches among non-singletons; 85% of non-singleton entities have matches in *both* S2 and S3 — genuinely many-to-many.
- Country distribution shifts between train (US 60%/India 40%) and test (US 45%/India 55%/France 15%), not just "France added."
- Name similarity of true matches: mean `token_sort_ratio` ≈80, but ~10-13% of true matches have similarity ≤50, and ~0.01% have *zero* name similarity — findable only via address.
- Devanagari tokenization artifact: 7.9% of India names share a fragmented "Private Limited" pattern (`पर इव...`) that breaks whitespace tokenization and later broke plain Jaccard/MinHash.
- Abbreviations: `ltd`/`limited` is the most common name abbreviation (222,911 true pairs) and the worst case for character n-grams — 0% shared 3/4-grams, because it's a "skeleton" abbreviation (drops middle letters), not a prefix truncation like `dr`→`drive`.

## Pipeline evolution

### 1. Exact-match blocking (`src/blocking.py`)
Four strategies (`compact_name`, `postal_country`, `first_token_country`, `token`), country-partitioned (verified safe: 0 of 7.6M true matches cross a country boundary). **Two production memory incidents** — an unbounded blocking key blew a 64GB job to ~219GB then ~243GB, both killed live on HPC — fixed with a shared `_bounded_merge_candidates` helper (`MAX_KEY_FREQ=200`, `MAX_PAIR_PRODUCT=40,000`) plus a Splink-style pre-check (`blocking_precheck.py`) that estimates pair counts before merging.

**Full-scale union recall:** 65.85%/65.78% (S2/S3), 53.91%/52.79% entities fully recovered, ~98 candidates/entity.

### 2. SimHash / MinHash exploration
- **SimHash (TF-IDF + Random Projection LSH):** dev sample 52%/51% alone; combined with exact-match union, 78.47%/78.15%. Failed completely at full scale (0% recall — buckets too large) until fixed to prune per-bucket instead of per-table; even fixed, aggregate pair count across "safe" buckets still hit ~200M/table — retired in favor of TF-IDF top-N.
- **MinHash + LSH (Jaccard):** rejected — 7/8 bands unsafe at full scale, because plain Jaccard has no IDF-equivalent and the Devanagari suffix fragment created a real, enormous similarity cluster.

### 3. TF-IDF cosine + `sparse_dot_topn`
Safe by construction (hard top-N cap per row). Dev sample at N=75: 84.62%/86.63% alone. An A/B test disproved stripping legal suffixes before vectorizing — made recall slightly worse (IDF already handles it; stripping shortens already-short names, worsening top-N competition). **Full-scale run stuck 70+ minutes with zero output** — memory stayed safe, but compute time is unbounded (cost driven by n-gram co-occurrence density, not row count). A `threshold` parameter was tested as a speedup and empirically does *not* help.

### 4. Teammate's parallel workstream (critical cross-check)
A teammate (separate Padum account) independently arrived at the same core technique and validated it at real scale:
- Streams the **target** pool in 50k-row chunks (not one monolithic multiply), fits TF-IDF **separately per country** on a 20k-row prefix sample, runs **name + address routes independently, fused by reciprocal rank fusion**.
- **10,000 references against the full 10,320,219-target pool: 93.49% top-50 recall, 95.94% union recall, 24.7 min total** (Padum job `1065336`).
- Their SQLite FTS5 and exact-match posting-list attempts both failed to scale (too slow / too lossy) before landing on this.

### 5. BM25 (`bm25s`) — beats TF-IDF on quality
Researched BM25 vs TF-IDF: term-frequency saturation + explicit length normalization are the two mechanical improvements. `bm25s` is a pure NumPy/SciPy library, no server, claims Elasticsearch-comparable speed.

**Dev sample, name-only, top-75:** BM25 86.25%/88.62% recall vs TF-IDF's 84.62%/86.63% — a real, consistent gain at the same candidate budget, though ~7x slower per-call at this scale in our first (naive) implementation.

### 6. Name+address RRF fusion — best result by far
Adopted the teammate's fusion strategy (name + address routes, reciprocal rank fusion, `1/(60+rank)`) on top of BM25 instead of TF-IDF.

- **First attempt** (chunked target + fusion): **98.80% recall, 97.93% entities fully recovered** on dev sample S2 — far exceeding the teammate's ~96%. But slow: 681.6s, because it rebuilt a full BM25 index per target chunk (backwards from the standard "index once, query many" production pattern).
- **Fixed architecture** (index built **once** per country, only the *query* side batched — confirmed correct via research on how BM25/Lucene/Elasticsearch actually scale): **98.81% recall, 97.95% entities fully recovered, in 261.0s — a 2.6x speedup with recall preserved** (in fact marginally better, since there's no per-chunk top-k merge approximation anymore).

**This is currently our best validated result: 98.8% recall via BM25 name+address fusion, index-once/query-batched.**

## The open question: does this survive full scale?

Dev sample is ~3% of S1 (66,205 vs 2,206,821) and ~5% of the target pool (258,447 vs ~5,034,616). Naive bilinear extrapolation (33x more queries × 19.5x bigger corpus ≈ 650x more work) would put a full S2 run at **~47 hours** — not viable as a single sequential run, even with the index-once fix.

**Researched fix for this specific gap** (Salesforce's Identity Resolution case study, 50M→2B records): the remaining lever isn't a smarter single-machine trick, it's **sharding across independent workers** — each worker builds its own local index over its own shard of data, fully parallel, zero cross-node contention. We have this available for free via multiple simultaneous PBS jobs across `scai01`-`scai04`.

**Also researched but not yet applied:** prefix filtering (PPJoin-style) — a mathematically provable way to skip candidate pairs before computing anything for them, unlike our `threshold` parameter which was empirically shown not to help.

## Scaling architecture debate + intermediate-scale validation attempt (this session)

Discussed a GPU-ANN (FAISS) retrieval channel as a third candidate-generation route alongside BM25, run in parallel on the A100s. **Decision: rejected for now** — BM25 fusion already gets 98.8% recall on the dev sample, so recall isn't the bottleneck, throughput is; standing up a whole new embedding/FAISS pipeline would burn scarce 72hr-window time validating a component that doesn't address the actual gap. `bm25s` itself has no GPU path (pure NumPy/SciPy) — confirmed via research, so forcing GPU onto the existing BM25 stage buys nothing. GPU is better saved for the downstream classifier stage (LightGBM GPU / optional reranker), not blocking.

Agreed direction instead: **aggressive CPU parallelism via sharding** — split the S1 query pool into N shards, one PBS job per shard, each building its own local BM25 index and running independently across `scai01`-`scai04` (the Salesforce-pattern lever already identified). Not yet implemented.

**Attempted intermediate-scale validation** (20k S1 queries vs FULL S2 pool of 5,034,616, the teammate's own benchmark methodology) to get a real scaling curve before committing to a shard count:
- **HPC check first** (per explicit instruction: try HPC before falling back local): all 4 `scai` nodes were fully CPU-busy at check time (`ncpus assigned == pcpus` on every node; `scai_q` showed 15 running/3 queued/6 held jobs). GPU: only 2 free on scai01, 1 on scai02, 0 elsewhere. No free slot to run immediately — would have queued behind other jobs (likely the teammate's).
- **Fell back to local** (16GB Mac, 10 cores) with process-level parallelism (one OS process per BM25 channel — name/address — via `fork` to avoid macOS's memory-doubling `spawn` default). **Failed**: running 2 full-corpus (5M-row / 2M-row per-country) BM25 index builds concurrently exceeded local RAM — swap went from near-zero to 14GB used, CPU time nearly stalled (5s of progress per 20s wall-clock, i.e. thrashing not computing), had to kill it before it finished even the smaller India shard.
- **Root cause identified**: process-level (fork) parallelism duplicates the full target-corpus BM25 index build per process — fine at dev-sample scale, not at full-corpus scale on a 16GB machine. The correct local lever is `bm25s`'s own internal thread-level parallelism (`n_threads=-1` in `retrieve()`), which shares one in-memory index across threads instead of duplicating it — i.e. run channels **sequentially** locally (as the already-validated `chunked_bm25.build_candidates_fused` does), not as separate OS processes.
- **Status: intermediate-scale validation not yet completed.** Next attempt should either (a) rerun locally with sequential channels (lower peak memory, proven pattern) at a smaller query count first, or (b) wait for an HPC slot where node-level memory (500GB-1TB per node) makes the process-per-channel approach safe. `src/validate_intermediate_scale.py` currently contains the process-parallel version that OOM'd locally — needs the sequential rewrite before reuse.

## Immediate next steps

1. Rewrite `validate_intermediate_scale.py` to run name/address channels sequentially (not as parallel OS processes) and re-attempt the intermediate-scale validation (20k S1 queries vs full S2 pool), either locally or on HPC once a slot frees up.
2. Once real per-query timing is known, shard the full run across multiple PBS jobs/nodes (Salesforce pattern: N shards of the S1 query pool, one job per shard, independent local BM25 index per job), merging per-shard top-K results afterward.
3. **Coordinate with the teammate** — we've independently converged on the same core technique from two directions, and their jobs are likely part of what's currently occupying the `scai_q` nodes; avoid continuing to duplicate work.
4. Still entirely unbuilt on both sides: entity-level train/val split, the actual classifier (LightGBM planned) + F_0.5-tuned threshold, test-set generation, and a first leaderboard submission. Given the 72-hour window, getting *any* valid submission in banks pipeline-correctness before further tuning.

## Repo

Code and history: https://github.com/Ayush121511/amazon-ml-challenge (public; Kerberos ID scrubbed from all history via `git filter-repo` before making it public). Default branch `eda`.

## Key files

| File | What it does |
| --- | --- |
| `src/normalize.py` | Text cleaning, legal-suffix stripping, postal-code heuristic |
| `src/blocking.py`, `src/blocking_precheck.py` | Exact-match blocking + Splink-style dry-run pair-count estimator |
| `src/simhash_blocking.py`, `src/minhash_lsh_blocking.py` | SimHash and MinHash+LSH experiments (SimHash partially useful, MinHash rejected) |
| `src/tfidf_topn_blocking.py` | TF-IDF cosine + `sparse_dot_topn`, safe-by-construction top-N blocker |
| `src/bm25s_blocking.py` | BM25 (single channel) via `bm25s` |
| `src/chunked_bm25.py` | **Current best**: BM25 name+address fusion, index-once/query-batched |
| `src/make_dev_sample.py` → `data/processed/dev_sample/` | 3%-stratified dev sample, every true match preserved, for fast local iteration |
| `notebooks/eda_basic.md` | Full EDA report |
| `references/hpc/*.sh` | PBS job scripts (Kerberos ID placeholder, fill in your own) |
