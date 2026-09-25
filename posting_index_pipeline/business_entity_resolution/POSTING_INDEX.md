# Persistent posting index (candidate generator v3)

Owner: aib262467 (teammate workstream from `TEAMMATE_HANDOFF.md`).
Code: `src/posting_index.py` (build + search), `src/posting_index_benchmark.py`
(fresh sample, throughput, recall, miss audit), `tests/test_posting_index.py`.
Padum jobs: `padum/posting_index_build.pbs`, `padum/posting_index_query.pbs`.
Outputs go to `artifacts/posting_index_v1/` only, in the aib262467 Padum
account, so nothing overlaps the other workstreams' artifacts.

## Why a different index

The sparse TF-IDF benchmark had good recall (94.27% at 50) but re-parsed and
re-vectorised all 10.32M targets for every 400 references. The SQLite FTS
path was slow per query, and bounded blocking lost too many links. This index
keeps the TF-IDF signal but builds it **once per split** and saves it to disk.
Queries then touch only the posting lists of their own terms.

## Method

- **Three routes**, all over the preprocessed v2 folded fields:
  - `name`: character 3-5 grams (word-bounded) of `name_folded`.
  - `address`: character 3-5 grams of `address_folded`.
  - `anchor`: address-number + neighbouring-word keys, for example
    `630|terrace`. Leading zeros are removed (`0684` and `684` agree), and a
    number is paired with the nearest word of at least 3 letters on each side
    (within two tokens). A bare number is never a key.
- **Hashed terms** (`HashingVectorizer`, 2^22 buckets, MurmurHash3). There is
  no fitted vocabulary, so France and any other unseen country need no
  special handling. Each country gets its own index, so matching stays within
  a country, and the country set comes from the data rather than a fixed list.
- **Label-free weighting**: IDF comes from the target pool of that split and
  country. Vectors are L2-normalised over all of their terms, then very common
  terms (document frequency above `max(1000, 0.5% of the country pool)`) are
  left out of the postings. A query keeps its 32 highest-weighted remaining
  terms, so its cost depends on posting lengths, not on the pool size.
- **Storage**: per country and route, a term-major CSR matrix saved as `.npy`
  files (`indptr`, `indices`, `data`), plus `idf`/`keep` arrays. `ids.npy`
  maps each posting ordinal to its entity ID, and `targets.tsv.gz` holds the
  original and folded text in the same ordinal order, so the matcher can
  recover both.
- **Search**: sparse top-k multiplication (`sparse_dot_topn`) per route
  (top 100), then reciprocal-rank fusion (k=60) into a ranked top 50. All
  three per-route lists, with their cosine scores, are kept for union recall
  and as matcher features.
- **Build**: two streaming passes over Source 2 and Source 3, parallel over
  50k-row chunks. Pass 1 counts document frequencies and writes the text
  table; pass 2 writes weighted chunks, which are then transposed into
  postings.

## Evaluation protocol

- The fresh sample is 10,000 references per country (US and India), drawn by
  reservoir sampling with seed `posting-index-v1`. It excludes every earlier
  diagnostic sample, all regenerated from their fixed seeds: `retrieval_v1`
  (2,000), `blocking_v2` / `sparse_fullpool` (400) and `selective_index_v1`
  (400). The exact IDs are in `reference_ids.txt`, with a SHA-256 in the
  report, so other methods can be compared on the same set.
- Candidates for every reference are written to `candidates.jsonl` **before**
  ground truth is opened. Labels are used only for recall and the miss audit.
- The report gives recall at 10/20/50, pre-trim union recall, per-route
  recall, singleton top-score distributions, and a miss audit by country,
  stage and reason. Script mismatch (the two names share no alphabetic script)
  is reported separately from non-ASCII text in the same script.
- Throughput is measured end to end: index load, one pass over Source 1, and
  query time, plus the extrapolation to 1,732,544 test references.

## Reproduce

```bash
cd ~/scratch/amazon_ml_2026
qsub padum/preprocess_all_cpu.pbs                           # train + test v2 preprocessing
qsub -v SPLIT=train padum/posting_index_build.pbs           # -> artifacts/posting_index_v1/train_index
qsub -v SPLIT=train padum/posting_index_query.pbs           # -> artifacts/posting_index_v1/train_query_10000/report.json
qsub -v SPLIT=test  padum/posting_index_build.pbs           # test pool incl. France
qsub -v SPLIT=test,PER_COUNTRY=10000 padum/posting_index_query.pbs   # test throughput, no labels
```

Local tests: `python -m unittest discover -s business_entity_resolution/tests`.

## Results (train split, 25 September 2026)

Local copies: `research/posting_index_v1_train_report.json`,
`research/posting_index_v1_train_index_manifest.json`,
`research/posting_index_v1_misses.jsonl`, and the shared fresh ID list
`research/posting_index_v1_reference_ids.txt` (20,000 IDs, SHA-256
`1da8079e…c53f5`). The regenerated earlier samples had the expected sizes
(2,000 / 400 / 400).

**Build** (job `1065330`, 8 CPUs, 48 GB requested, node amdepyc): 10,320,219
targets (US 6,186,873, India 4,133,346). 338 s wall in total: pass 1 165 s,
pass 2 112 s, finalize 61 s. The index is 5.44 GB on disk. Peak RSS was
4.1 GB in the main process and 0.8 GB in the largest worker.

| Country | Route | Postings nnz | Max df | Terms indexed | Terms dropped as common |
| --- | --- | ---: | ---: | ---: | ---: |
| US | name | 138.8M | 30,934 | 1,437,618 | 1,896 |
| US | address | 164.0M | 30,934 | 586,626 | 2,196 |
| US | anchor | 8.1M | 30,934 | 2,208,959 | 0 |
| India | name | 85.5M | 20,666 | 1,029,119 | 1,876 |
| India | address | 129.7M | 20,666 | 722,198 | 3,532 |
| India | anchor | 8.8M | 20,666 | 1,123,380 | 5 |

**Throughput** (job `1065333`, 8 threads on scai03): 20,000 references
against the full 10.32M pool took 160.6 s end to end:
- 87.3 s regenerating the earlier samples (only needed for exclusion)
- 14.7 s loading references
- 10.7 s loading both indexes
- 47.9 s of queries, which is **418 references/s**

Peak RSS was 4.6 GB. The naive extrapolation to 1,732,544 test references is
about **70 minutes** (4,172 s). This still needs to be confirmed on a real
test index.

**Recall** on the fresh sample (18,853 positive references, 1,147
singletons, 68,914 gold links):

| Group | R@10 | R@20 | R@50 | Union (pre-trim) | name@100 | address@100 | anchor@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 84.22% | 88.03% | 90.75% | 93.92% | 54.39% | 82.83% | 66.26% |
| US | 88.93% | 92.51% | 94.78% | 96.81% | 61.36% | 87.67% | 71.64% |
| India | 79.51% | 83.55% | 86.71% | 91.04% | 47.42% | 77.99% | 60.89% |

On average a reference gets 235 union candidates, and no shortlist is empty.
Source 1 names in this sample were all ASCII, so the `non_ascii_name` group is
empty. Non-Latin script appears only on the target side.

These numbers come from a different, 50x larger sample than the 400-reference
sparse benchmark (94.27% R@50, 96.17% union). They are **not** a same-sample
comparison, and this sample is harder in India. Re-running on the 400 sparse
IDs is the next check.

**Singletons**: retrieval scores do not separate them. The median top-1
cosine is 0.64 for singletons and 0.67 for positive references on the name
route, and 0.49 vs 0.56 on the address route. The anchor route separates
them slightly better (median 0.82 vs 1.00). The matcher must decide no-match
from pair features, not from retrieval score alone.

**Miss audit**: 6,377 gold links were outside the top 50.
- By stage: 2,189 were in the union but lost at the top-50 trim, and 4,188
  were absent from every route.
- By country: India 4,580, US 1,797.
- By primary reason (priority order, sums to the total): similar name crowded
  out 2,624, target address missing 1,475, **name script mismatch 1,392 (all
  India)**, address anchor only 373, web-domain name 354, low text similarity
  159.
- Overlapping flags: shares an address number 3,553; name token Jaccard ≥ 0.5
  3,958; non-ASCII text in the same script 392.
- No miss had its target in another country's pool.

## Same-sample check against sparse_fullpool (job `1065364`)

The 400 `sparse_fullpool` references were regenerated from their seed by
`src/export_previous_samples.py` and queried with `--reference-ids`
(`padum/posting_index_same_sample.pbs`, 1 min 8 s wall, 3.8 GB peak RSS).
The regenerated sample is consistent with the original: it has the same 1,379
gold links, contains all 57 references in `research/sparse_misses.jsonl`, and
reproduces sparse top-50 recall of 94.27% when that recall is recomputed from
the miss file. The local comparison is `research/compare_same_sample.py`, with
outputs in `research/posting_index_v1_same_sample/`.

| Method | R@10 | R@20 | R@50 | Union (pre-trim) |
| --- | ---: | ---: | ---: | ---: |
| sparse_fullpool (streaming, fitted vocabulary) | 89.12% | – | **94.27%** | **96.17%** |
| posting index v1 | 85.21% | 88.69% | 91.59% | 94.27% |
| posting index v1, US | 89.43% | 93.54% | 95.74% | 97.21% |
| posting index v1, India | 81.09% | 83.95% | 87.54% | 91.40% |

**On the same sample, the posting index is 2.7 points behind at 50 and 1.9
points behind on the union.** In exchange, it is about 340x faster.

Link-level results, top 50: 1,236 links were found by both methods, 27 only
by the posting index, 64 only by sparse, and 52 by neither. Of the 64 found
only by sparse, 38 are *similar name crowded out* (29 of them in India),
10 are *target address missing*, and 8 are India cross-script. On the union,
48 were found only by sparse (25 crowded out, 12 address missing, 7
cross-script), and 22 only by the posting index. The union of both methods
misses just 31/1,379 links (97.75%), so the two are complementary.

Interpretation: the posting index's main loss is ranking among similar names,
not a missing route. The likely causes are the 32-term query cap and dropping
common terms (df > 0.5%), both of which remove signal that sparse's full query
vectors keep. This makes next step 4 (tune `top-terms` / `max-df-fraction`)
the top quality priority, ahead of transliteration.

## Known limits and next steps

1. **Same-sample check: done** (section above). It is 2.7 points worse at 50
   and 1.9 points worse on the union than sparse, mostly because India similar
   names are crowded out. Speed: End to end, excluding the one-time 5.6-minute build, it
   takes 0.008 s per reference versus 2.76 s for the streaming sparse scan
   (about 340x faster).
2. **Lost at 50** (2,189 links, 3.2 points): the union is already 93.9%.
   Keeping the union rather than a fused top 50, or learning the fusion,
   recovers them.
3. **India name route is weak** (47% at 100) and 1,392 misses are cross-script
   (Latin reference, Devanagari/Tamil target). A rule-based Indic-to-Latin
   transliteration route (offline, no external lookup) is the clearest gap.
4. **Crowded-out similar names** (2,624): tune `top-terms` (32), per-route
   `top-k` (100) and `max-df-fraction` (0.005). Also consider a name route
   that down-weights legal suffixes.
5. **Missing target address** (1,475): these can match only through the name
   route, so it matters most for them.
6. **Test split**: build `test_index` (it includes France) and run the test
   throughput job. France is covered by unit tests only so far.

Resource notes: `scai_q` counts held (dependency) jobs toward the 2-job
per-user limit, so a third chained `qsub` is rejected. Every job must request
a GPU even for CPU work.
