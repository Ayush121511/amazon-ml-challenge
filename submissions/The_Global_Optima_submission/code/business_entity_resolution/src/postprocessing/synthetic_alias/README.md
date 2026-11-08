# France synthetic-alias channels

Two label-free rule channels that link an invented alias record to a French business. The record's name must be one
token built entirely from the generator's fixed syllables (`arc`, `halo`, `nyla`, `zeta`, …), optionally followed by
Labs, One, Sys or Co.

| Script | What it does |
|---|---|
| `invented_morpheme_probe.py` | The fixed syllable grammar (`parse`). |
| `france_morpheme_sorted_address.py` | **unique_s1_address:** the record's full address (lower-cased, accents and punctuation removed, tokens sorted; at least one digit and four tokens) equals the address of exactly one business. |
| `selected_copy_alias_probe.py` | **accepted_copy_address:** the record's full address equals the address of records already accepted for exactly one business. |
| `merge_synthetic_rules.py` | Union of the two channels. A record claimed by two businesses is dropped. Writes `proposed_synthetic_alias_links_safe.tsv` (included). |
| `train_morpheme_sorted_address.py` | Precision of the unique-address rule on the full training labels: 99.76%. |
| `train_accepted_copy_alias.py` | Precision of the accepted-copy rule on the full training labels, with true copies as anchors: 99.7–99.9%. |

**Inputs.** `france_test.sqlite3` is a table `records(entity_id, source, name, address)` holding the French test
records. The accepted links come from the base submission, and records already linked by the earlier France rules are
skipped.

**Merge.** `src/postprocessing/apply_alias_links.py` adds the links to the submission (README step 27f): first
`unique_s1_address`, then `all`. A link is added only when no business holds the record, and the record is appended to
`candidate_pairs.tsv` when it is not already a candidate.
