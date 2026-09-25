# Amazon ML Challenge 2026: dataset and problem

This file is a quick introduction for a teammate or a new chat. The organizer's
authoritative rules are in
`6ab10eb3b23ba_student_resource/student_resource/README.md`. If they differ from
this summary, follow the organizer README.

## What the model must do

This is **business entity resolution**. Source 1 is a deduplicated list of
reference businesses. For *every* Source 1 record, find **all** records in
Source 2 and Source 3 referring to the same real-world business. A reference
may have zero, one, or multiple matches, including several within one target
source. Do not force one-to-one matching.

Records have no shared identifier across sources. Names and addresses can vary
through spelling, abbreviations, missing fields, punctuation, language/script,
and formatting. There are no images in the released source files; OCR is not
needed for this dataset.

## Files and sizes

Root: `6ab10eb3b23ba_student_resource/student_resource/dataset/`.
All input files are **tab-separated**, despite commas inside addresses and
match lists. Always set the tab delimiter explicitly.

| File | Rows | Purpose |
| --- | ---: | --- |
| `train/train_source1.tsv` | 2,206,821 | Labeled reference businesses |
| `train/train_source2.tsv` | 5,034,616 | Training target records |
| `train/train_source3.tsv` | 5,285,603 | Training target records |
| `train/train_ground_truth.tsv` | 2,206,821 | Correct match lists for Source 1 |
| `test/test_source1.tsv` | 1,732,544 | References requiring predictions |
| `test/test_source2.tsv` | 4,887,273 | Test target records |
| `test/test_source3.tsv` | 5,082,316 | Test target records |

Each source row has `entity_id`, `business_name`, `business_address`, `country`.
Prefixes S1-/S2-/S3- identify the source. The ground-truth columns are
`source1_entity_id` and `matched_entity_ids`; the latter is a comma-separated
list or empty for a business with no match. Training covers US and India.
Test also contains France: 259,452 French Source 1 references. Do not make
the final pipeline specific to only the training countries.

Full training ground truth contains 7,638,365 positive links and 123,247
Source 1 records with no matches (about 5.6%). Target addresses can be blank.
The train/test counts above were measured by streaming the supplied files;
see `research/dataset_audit.json` and `research/REVIEW_SUMMARY.md`.

## Score and outputs

The score is **macro F0.5**: compute F0.5 separately for every Source 1
business, then average equally over references. F0.5 favors precision; an
incorrect link is especially costly. A no-match business scores 1 when the
predicted match list is empty and 0 when any match is predicted. Local
implementation: `business_entity_resolution/src/metrics.py`.

The portal scores `matching_results.tsv`, with columns
`source1_entity_id<TAB>matched_entity_ids`. It must have exactly one row per
test Source 1 ID; an empty list means no match. The final package also requires
`candidate_pairs.tsv`, with columns
`source1_entity_id<TAB>candidate_entity_ids`. That file must be the exact
candidate set fed to the final matcher, and every final matched ID must be
among that reference's candidate IDs. Neither list may contain duplicate IDs,
and listed target IDs must exist in test Source 2 or Source 3. Use the provided
`utils/validate_submission.py` before submission.

The final zip must also include a runnable end-to-end code copy, pinned
dependencies, and a completed `Documentation_template.md`. During the
challenge, the public leaderboard scores only a subset of test records;
final ranking uses the private remainder. A public score is not a final score.

## Constraints

- External lookup of business identities, business registrations, geocoding,
  entity-resolution services, and external data augmentation are prohibited.
- The organizer specifies an MIT/Apache-2.0 licensed final model of at most
  8 billion parameters. Verify a checkpoint's license before adding it.
- The user-provided three-person team uses IITD Padum project `scai`.
  The available `scai_q` queue requires at least one GPU per job, even for
  CPU-only experiments. Never run heavy work on a login node.

See `PROJECT_STATUS_AND_HANDOFF.md` for what has actually been implemented,
measured, and still needs work. The linked Claude EDA artifact was inaccessible
(HTTP 403), so its contents are not incorporated here.
