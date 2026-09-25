# Handoff: persistent posting index workstream (aib262467)

Snapshot: 25 September 2026. This covers the task in `TEAMMATE_HANDOFF.md`:
build a reusable target index and candidate generator. Full method, numbers
and reproduction commands are in `business_entity_resolution/POSTING_INDEX.md`.

## Done

1. **Padum setup in account `aib262467`** (separate from `aib262140`):
   `~/scratch/amazon_ml_2026` mirrors your layout. `.conda-er` is a symlink
   to a Python 3.11 env with numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1,
   sparse-dot-topn 1.1.5 and psutil 7.1.1. The student_resource `dataset`
   folder is a symlink to the official TSVs. Your PBS scripts run unchanged.
2. **Version-2 preprocessing of train and test** (job `1065324`, your
   `preprocess.py`, new `padum/preprocess_all_cpu.pbs`): all 24,229,173 rows,
   and manifests pass. **Test preprocessing now exists**, in this account.
3. **New candidate generator**:
   - `src/posting_index.py`: hashed TF-IDF postings built once per split and
     country, with name 3-5 grams, address 3-5 grams, and address-number+word
     anchors.
   - `src/posting_index_benchmark.py`: fresh sample, throughput, recall and
     miss audit.
   - `tests/test_posting_index.py`: 12 new tests. The full suite passes
     locally and on Padum (**34 tests**).
4. **Train results** on a fresh 20,000-reference sample (10k US, 10k India).
   It excludes all 2,800 earlier diagnostic IDs, regenerated from their seeds.
   - Build: 5.6 min, 5.44 GB on disk, 4.1 GB peak RSS.
   - Query: 418 references/s on 8 threads, 160.6 s end to end for 20k
     references. The extrapolation to 1.73M test references is about
     70 min.
   - R@10 / R@20 / R@50: 84.22% / 88.03% / **90.75%**.
   - Union recall: **93.92%**. US 94.78% R@50, India 86.71% R@50.
   - Miss audit: 6,377 missed links. Of these, 2,189 were lost at the top-50
     trim and 4,188 were absent from every route. **1,392 are true
     cross-script name mismatches, all in India**, counted separately from
     same-script non-ASCII.
5. Shared artifacts in `research/`:
   - `posting_index_v1_reference_ids.txt`: fresh sample IDs. Use these to
     compare methods.
   - `posting_index_v1_train_report.json`
   - `posting_index_v1_train_index_manifest.json`
   - `posting_index_v1_misses.jsonl`: 6,377 misses with text and reason
     flags.

## Caveats

- The 20k fresh-sample numbers are not directly comparable with the sparse
  94.27%. The same-sample comparison is in next step 1: the posting index is
  2.7 points behind at 50.
- Retrieval scores do not separate singletons from positives. No-match
  decisions must come from the matcher.
- France is tested only by unit tests; the test index has not been built yet.
- Artifacts live in `aib262467` scratch. To read them from `aib262140`, the
  owner has to grant access (for example `setfacl -R -m u:aib262140:rX
  ~/scratch/amazon_ml_2026/artifacts/posting_index_v1` plus `x` on parent
  directories). This has not been done.

## Next steps, in priority order

1. ~~Same-sample check on the 400 `sparse_fullpool` IDs~~ **Done, job
   `1065364`.** R@50 91.59% vs sparse 94.27%; union 94.27% vs 96.17%.
   64 links were found only by sparse and 27 only by posting; 38 of the 64 are
   similar names crowded out (29 India). The union of both misses only 2.25%.
   So step 4 below is now the top quality priority. See the "Same-sample check"
   section in `POSTING_INDEX.md`. The earlier sample ID lists are in
   `research/posting_index_v1_same_sample/`.
2. **In progress:** the matcher sees the fused top K (100 or 200) of the
   route union, not just the top 50. Job `1065413` measures recall at K and
   trains both variants.
3. Add an offline Indic-to-Latin transliteration name route for the 1,392
   cross-script misses (rule-based; no external lookup).
4. **Partly done** (sweep job `1065375`): `--top-terms` barely matters
   (+0.2 points), while `--top-k` 200 lifts union recall from 93.7% to 95.2%.
   Next: a rebuild with a higher `--max-df-fraction`.
5. **Done:** test index built (job `1065374`), 9,969,589 targets including
   France.
6. **In progress:** matcher and submission via `src/submission_pipeline.py`
   (see `PROJECT_STATUS_AND_HANDOFF.md`).

`scai_q` limits: 2 jobs per user, and held dependency jobs count toward that.
Every job needs `ngpus=1` even for CPU work.
