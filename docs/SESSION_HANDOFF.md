# Session handoff: Amazon ML Challenge 2026, Business Entity Resolution (27 Sep 2026, ~15:30 IST)

Read this first in a new session. Decision log: `docs/SUBMISSION_TRACKER.md`.

## 1. Where things stand right now
- **Deadlines (27 Sep, IST):** code freeze and methodology doc **20:00** (extended from 18:00); last leaderboard upload by 22:30; hard deadline midnight. Five uploads per day, and today is the last day.
- **Public scores so far:** S5 0.983415 → T2 0.988085 → H2 0.989067 → **U1 0.989635 (best, uploaded ~13:00)**.
- **U2 is staged:** `submissions/U2_T6_FRw4_xprune/` (matching sha `9881548f…`, candidates `716e89aa…`). HPC copy is `aib262015:~/scratch/amazon_ml_2026/artifacts/submission_U2_final/`. Expected ~0.9899.
  - Contents: T6 for US/India, plus France = FRtrained_w4 → exclusivity prune → census v2.
  - **Check with Ayush whether it was uploaded, and its score.**
- **U3 = last upload.** A fallback is staged at `artifacts/submission_U3_fallback/` (sha `4e5fbf7a…`): T6 + U1's France, no prune. It is byte-identical to Vishal's `submission_UP2_T6_FRw4/final`.
  - **Rule:** if U2 ≥ 0.9896, U2 is final and U3 stays unused, unless a validated gain exists. If U2 < 0.9896, upload the fallback.
- **Running:** job 1068144 (`er_webleet`) validates two extra US/India candidate channels on train labels (`web_leet_channel.py`). Outputs: `artifacts/{web,leet}_train.json`.
  - **web:** handle/domain names built from the S1 name ("aesociety.com" = A(chintya) E(ducational) + Society).
  - **leet:** "b1ack" → black.
  - **Use only if** the new candidates (outside the top 10) are mostly true **and** F0.5 gains on both dev and holdout. If so, run `MODE=test` and add the pairs to `candidate_pairs.tsv` plus the accepted links (records unowned). Otherwise drop.
  - **15:45 result: both rejected.** web: dev 9 new candidates, 0 true, F0.5 −0.00017. leet: dev −0.00006, holdout 7 new (2 true) −0.00006. Job killed.
  - The native-script channel was validated and **rejected**: new candidates were 0/152 true on dev and 6/152 on holdout, F0.5 −0.0017.
- **Code zip:** Tapan (reproducibility dry-run, docstrings). **Doc:** Vishal. Ayush's frozen source tarball: `artifacts/freeze_ayush_20260927.tar.gz` (131 files, `MANIFEST.md5`, `FREEZE_NOTES.txt`). Rebuild it with `business_entity_resolution/padum_freeze_job.sh` after adding scripts.
- **`candidate_pairs.tsv` = the top-10 file.** README lines 102–106 require the exact set the model ran inference on. The t = 0.001 cut (6.61 candidates per S1, holdout loss 0.000003) is a documented experiment only, **not** submitted.

## 1b. Error sweep (27 Sep 16:20): full tables in `docs/ERROR_SWEEP_REPORT.md`
Scripts: `error_sweep.py` (MODE=train/test), `error_sweep_report.py`, `empty_guard.py`, `excl_empty.py`; outputs `artifacts/sweep/` on HPC.
- **Loss budget of U2, public score units:**
  - US/India blocking 0.0018 (top-10 prune 0.0014, retrieval 0.0004).
  - US/India decision 0.0056 on holdout vs 0.0042 test estimate: blank-address missed links 0.0042 / 0.0028, non-blank missed links 0.0009, wrong links 0.0006.
  - France 0.0034 (model's own view).
  - Sum 0.0094–0.0108, i.e. public 0.9892–0.9906, which brackets U1's 0.9896.
- **Label-free estimator is trustworthy:** holdout in-list expected loss 0.70 vs actual 0.67.
- **Blank ties are calibrated coin flips, not fixable:** p 0.245 vs true rate 0.243. Mostly names shared by 6+ S1s.
- **France in-list loss is 4× US/India** (2.24 vs 0.53 points per S1).
- **Dead this session:**
  - Per-cell recalibration: +0.00002.
  - Empty-guard: −0.0004 to −0.0008 on dev.
  - Exclusivity "don't empty the loser": ceiling +0.00027, expected ~0 (contests are p 0.89 vs 0.99).
  - Web and leet channels.
- **Conclusion: no remaining validated lever above ~+0.0001.** The final choice is U2 vs the U3 fallback, decided by U2's public score.

## 1c. U5 UPLOADED 17:23 → public **0.989868** (U1 0.989635, +0.000233). U2 was NOT uploaded. +2 gift uploads (4 left after U5? confirm).
- 17:50: user decided NOT to upload U6 (measurement only). U5 stays the base, prune stays in. Alias link-up dead (0 train flags). LLM probe prepared: `llm_probe.py` (probe on dev/holdout band, auto-submitted by watcher once Qwen2.5-7B-Instruct is in ~/scratch/models/qwen2.5-7b-instruct), `llm_apply.py` (test path, France from U5).
- **18:00 every-row check (`rowsweep.py`):**
  - On train, a record whose normalized name + address matches exactly one S1 is that S1's record 100% of the time (117,886 strict / 228,805 loose, 0 exceptions).
  - U5 already links every such test copy: US 44,803/44,803 and India 26,010/26,010. France differs on only 2–10 records, which are same-business twins or city-only addresses. **Nothing to fix.**
- **Text-decidability ceiling:** 39% of train S1s have an exact-name twin. Blank address + twin = 72% of the uncertain band and 58% of holdout error loss; no reader can decide these.
- **LLM blind test:** Claude judged 79 holdout pairs from T6's band (0.15 < p < 0.85) blind (`scratchpad/blind`).
  - T6: AUC 0.787, log loss 0.548. Claude: AUC 0.652, log loss 0.646. Averaged: 0.791 / 0.556 (log loss worse).
  - An LLM judge adds ~0. **LLM route stopped.** `llm_probe.py` / `llm_apply.py` are left ready but unused.
- **18:40 LLM judge, measured (Qwen2.5-7B-Instruct, zero-shot yes/no logit; `llm_probe.py`, `llm_france.py`, outputs in `artifacts/llm/`):**
  - US/India uncertain band (15,895 labelled pairs): LLM AUC 0.54–0.55 vs T6 0.88–0.89.
  - Combining the LLM with T6: dev −0.00017, holdout −0.00013.
  - No LLM threshold reaches 92% precision for rescue or ≤5% true for veto, so **0 France flips**; U7 is byte-identical to U5.
  - On France, the LLM agrees with U5 on average: rejected pairs mean −3.9, accepted +2.3.
  - Model download needed the standard proxy (proxy62); the research proxy login failed after the first session.
- **FINAL = U5 (0.989868).** No further uploads planned.
- **19:15 update, supersedes "FINAL = U5":**
  - **U7 (Vishal):** U5 + 3,566 France links, expected ~0.9900. Vishal's 0.9908 used the wrong anchor: 0.989635 was U1, not H2v2; he confirmed.
  - **U8 (ours):** U7 + 1,193 France invented-alias links at a unique exact address that the T6 bag accepts (p ≥ 0.68). Expected ~0.99005.
    - Files: `artifacts/submission_U8_alias_fr/U8_alias_fr.zip`, sha `237c0446…`; matching `8bc5ed56…`. Validator PASS.
  - **Train-label evidence:** an invented single word at the S1's exact address (only S1 there) is 98.7% true (45,989 cases); 99.8–100% where T6 accepts. The ~14 T6-rejected US/India cases are a wash, so no US/India adds.
  - **LLM v2** (context + rules + 6 examples) on the US/India band: AUC 0.50 vs T6 0.88, holdout −0.00006.
  - **LLM on France:** it says "No" to 72.5% of the 4,677 U7/U8 alias links (median −9.9), i.e. anti-correlated with truth. LLM route closed.
  - **Upload plan:** U7 → U8 → re-upload the best in the last slot. Tell Vishal the final file.
- **19:35 deep pattern audit (`pattern_audit.py`, `pattern_cond.py`):**
  - Copy recipes at a unique own address, full-train true rate: exact 100%, initials 100%, same words 98.8%, handle 98.4%, invented 97.5%, typo 95.9%; blank + unique exact name 93.8%.
  - Shifted-number decoys (+1…+21 with an added or swapped word): 0–5% true.
  - **US/India:** within every pattern, T6's decisions on dev + holdout are ~perfect (accepted 98–100% true, rejected ~0%). No US/India rule fixes.
  - **France:** the T6 bag accepts 7,771 links that U8 rejects, but these are mostly type-word swaps (decoys), other addresses, or blank twins. Skipped.
- **U9 = U8 + 865 France rule-channel links** (`u9_build.py`):
  - 514 invented aliases + 240 initials at a unique own address that were **not in candidates** (754 pairs appended to `candidate_pairs.tsv`), plus 111 initials already in candidates.
  - Validator PASS. Zip sha `b1ba44a6…`; matching `20d3e3a1…`; candidates `47d819ae…`.
  - Location: `submissions/U9_rule_channel/` and `artifacts/submission_U9_rule_channel/`. Expected ~+0.00004 over U8.
  - India/US excluded: native-script names get misread as invented words.
- **French decoy check:** of own-address type → type swaps (the decoy recipe), France links **0** and rejects 4,876. There are no decoy FPs for an LLM judge to veto.
- **Shared with Tapan:** U7/U8 diff lists in `artifacts/u7u8_diffs/`.
- **U10 = U9 + 925 France links** (`u10_build.py`; `submissions/U10_small_steps/`, zip sha `2269ef6d…`, matching `f8a08f5f…`; candidates = U9's `47d819ae…`; validator PASS).
  - Adds: 469 blank records with a unique exact France name, T6-bag accepted (US/India analog 98.1% true); 23 own-address aliases, T6-bag accepted; 284 aliases only in the rescue pool (97.5%); 149 own-address type → filler swaps ("Comite" → "Groupe"; France links 5,106 of these already). Alias words must not appear in any France S1 name.
  - **Rejected rules:**
    - Exact name at a shifted number: US adds spread evenly over the decoy shift set, so they are test decoys.
    - India exact name at the own address: a native-script type word was being stripped.
    - Handles: mixed.
  - **Totals:** U10 vs U5 is +6,400 France links, 0 removed. Expected ≈ U9 + 0.00003 ≈ 0.99012.
- (unused) U6 = U5 without France prune (`artifacts/submission_U6_U5_noprune/U6_U5_noprune.zip`, sha `f4ba599c…`, matching `a672d6cf…`, validator PASS; France +3,496/−56 vs U5, US/India identical). Keep whichever of U5/U6 scores higher.
- **What it is:** Vishal's `/scratch/scai/mtech/aib262467/final_u5/final_u5.zip`, sha256 `f7826b1f…`. It is U2 plus:
  - a 5-seed T6 bag at expected@0.68 (holdout +0.000065);
  - +2,862 rescue links from ranks 11–20 (holdout +0.000064);
  - `candidate_pairs.tsv` = top 10 + 113,340 scored pairs.
- **Audit PASS** (`u5_check.py`, `artifacts/u5_check/report.txt`):
  - validator `--check-ids` passes; S5 sanity passes;
  - no record appears in two S1 rows; matches ⊆ candidates;
  - France is additions only. France rescues are mostly initials/domain/typo copies at the exact house number (1,164 of 1,380), with 8 decoy shifts.
- **Expected public:** ~0.9900 (range 0.9896–0.9903).
- **Plan:** upload U5 and tell Vishal the sha. If it scores < 0.9896, use the last slot for "U5 without France prune" (ask Vishal to stage it).
- **Local copy is broken:** `submissions/U5_final/final_u5.zip` is a 3.4 MB partial download. Re-download it before uploading.

## 2. Team, rules, preferences
- **Team and accounts:** Ayush (the user; the only one who uploads), HPC `aib262015`. Vishal `aib262467`: v5 blocker, T4–T6, FRtrained, doc; his env `…/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python`. Tapan `aib262140`: chunk2 CE, France self-trained CEs, code zip.
- **User preferences:**
  - Terse (minimum tokens).
  - No AWS or AWS MCP.
  - Never compute on the login node.
  - No Monitor polling. Short waits only; the user dislikes long blocking waits.
  - One change per upload where possible.
  - Validate before every upload: validator `--check-ids`, S5 sanity (`scratchpad/sub_compare.py`), transfer audit.
  - Pseudo-labels only when very confident.
- **HPC:**
  - `export PATH=/opt/pbs/2024.1.5/bin:$PATH`; queue `scai_q`, project `scai`, and every job needs `ngpus=1`.
  - **Per-user limit ~2 queued/running jobs.** To parallelise, run several processes inside one job (see `padum_webleet_job.sh`).
  - SSH to a compute node only works while you have a job on it.
  - Share files with `setfacl -m u:aib262140:r,u:aib262467:r`.
- **Fair play:** the France census method and the AND-veto idea came from findings shared by another participant (team SSM, IIT Delhi, public 0.990282). They **must be disclosed in the doc**. The user decided to take nothing further from them; in particular, do not copy their learned per-word statistic.

## 3. Final pipeline
1. **Blocking (v5, Vishal):** character TF-IDF plus fine-tuned multilingual-e5 dense retrieval, within each country (~290 candidates, 99.88% recall), then an XGBoost pruner keeping the **top 10** (99.32% recall; perfect-matcher ceiling 0.998).
2. **Pair scores:** v5 probability; mDeBERTa cross-encoders (chunk2 [Tapan], mdeberta_full, full21, decoyw2); L2 late fusion (XGBoost, decoys up-weighted ×3).
3. **Stage-3b** (XGBoost over each S1's list; T6 = 89 features). Features: list context, reverse competition against the full S1 table, house number, number shift, rare-word address keys, ownership (26), W2 rival (8). Decision: expected-F0.5 top-k (expected@0.74 for T6), then global exclusivity (`late_predict.finalize`).
   - Holdout: T5 0.99109, **T6 0.99122** (dev 0.99228).
4. **France** (no labels):
   - **FRtrained_w4:** stage-3b with an `is_france` input, trained on `fr_labels.py` pseudo-labels (census rules plus extreme-agreement pairs) with extreme negatives reweighted ×4, on a robust feature set that drops 9 France-shifted features. Applied to France only.
   - **Exclusivity prune** (`exclusivity_prune.py`, ours): a record belongs to one S1, so its probabilities over S1s must sum ≤ 1. France violates this 12× more than US/India. Drop a link when the sum > 1.2, nothing breaks the tie (blank address or a different number/street), p/sum < 0.77, and the S1 keeps another link. That's −3,858 links.
   - **Census v2** (`france_census.py`, `CENSUS_V2=1`): same-address restore/reject, plus our own paths (street-name typo, type-word drop in town + type names, initials).
   - **Final assembly:** `merge_by_country.py`.

## 4. Key EDA insights (evidence-backed)
- **Public ≈ 0.85 × US/India (≈ holdout) + 0.15 × France** (France is 259,452 of 1,732,544 S1). Implied France F0.5: S5 0.9675 → T2 0.9766 → H2 0.9776 → U1 0.9814.
- **T6 holdout loss, 0.878 points:**
  - blocking 0.206;
  - **blank-address errors ~0.54**: ambiguous same-name ties; W2, gen-3/4 EDA and blank-expert all failed to separate them;
  - non-blank errors ~0.13, fragmented into 30+ cells, none bigger than 0.015.
- **Non-blank US/India patterns:**
  - Invented-name aliases (syllable vocabulary Brix/Tavo/Lyra/Evo…) are the biggest group. They're ~50% true at the margin, T6 is calibrated there, and sibling-address-noise doesn't separate them.
  - Digit-drop house numbers are true copies (2118→118: 80–99% true); small upward shifts are decoys (1–3%). T6 mostly handles this already.
  - The native-script channel was rejected (see §1).
- **France generator facts:**
  - Decoys use a uniform shift set **+1, 2, 3, 4, 5, 7, 9, 11, 13, 21** on the S1's street (same name + legal-form swap, often SNC). The model rejects 89% of them.
  - Same-number different-street and type-word swaps are decoys.
  - Filler words (& Fils, Groupe, Développement, France, Cie) at the same address are copies.
  - France's lower link count (3.22 vs 3.38) comes from correctly rejected shifted decoys, not under-linking.
- **Train-label checks:**
  - Same name + typo'd street at the same number: 100% true.
  - Co-located unrelated names at p ≥ 0.9: ~100% true.
  - Cross-source sibling completion: 0–15% true.
  - No row-order or ID leakage.
  - Label shift ruled out: test mean p equals the holdout true rate.
  - France blocking: 99.988% of exact-key copies are in the candidates.
- **Transfer audit (KS on sure pairs, train vs test):** US/India ≈ 0.03, France 0.19, with address features up to 0.67. The listwise CE failed this audit (it emptied 16,850 S1 on test) and was dropped.

## 5. Tried and dropped
- Listwise CE: transfer failure.
- AND-veto with a T1b-based self view: 0.4% of links, and the wrong base model.
- Self-training round 2: converged.
- Stage-3b ensembles: dev prefers T6.
- Co-located-name pruning: those names are true in train.
- Restoring same-name shifted-number records: they're the decoy family.
- Cross-source completion.
- The learned per-word census statistic: the other team's method, declined.
- Native-script channel.
- Post-inference candidate cut: the README forbids it.

## 6. Useful scripts (`posting_index_pipeline/business_entity_resolution/src/`)
- **Pipeline:** `boundary_exp.py`, `boundary_exp2.py`, `stage3b_predict.py`, `stage3b_repredict.py`, `late_predict.py`, `distractor_check.py`, `sibling_check.py` (original module; `late_predict` imports `prefix_match`/`num_jacc`; never overwrite it), `fr_labels.py`, `france_census.py`, `exclusivity_prune.py`, `merge_by_country.py`.
- **Analysis:** `france_cases.py`, `france_patterns*.py`, `france_numdiff.py`, `france_diff_cases.py`, `colocated_check.py`, `overcommit_check.py`, `france_block_check.py`, `cross_source_check.py`, `shift_ens_check.py`, `usin_errors.py`, `usin_read.py`, `usin_patterns.py`, `seg_calib.py`, `sibling_noise_check.py`, `typo_street_check.py`, `alias_check.py`, `indic_translit.py`, `native_channel.py`, `web_leet_channel.py`.
- **Job scripts:** HPC `business_entity_resolution/padum_*.sh`; local `posting_index_pipeline/padum/`.

## 7. Artifacts (claude.ai pages)
Ayush's:
- [France Census Fixes](https://claude.ai/artifact/5G5pjK4JT4q9fajka4QztM) (27 Sep): the France same-address census, explained with worked cases.
- [Gen-3 Error Anatomy](https://claude.ai/artifact/FBozAaWfP6UNWc8JySecMG) (26 Sep): stage-3b loss split, blank-address ties, voting table.
- [Matcher Loss Atlas](https://claude.ai/artifact/F3K4Z54xTVogazSNwK81xy) (26 Sep): earlier matcher loss decomposition.
- [Entity Resolution Status](https://claude.ai/artifact/FxiEZ2WW5CehUjjQkBLajD) (25 Sep): early project status.
- [Abbreviation Patterns](https://claude.ai/artifact/BJD2arhveKyJCw6n81ABJR) (25 Sep): name abbreviation EDA.
- [Match & Similarity Stats](https://claude.ai/artifact/JHVdE1sabXJAfYfUiyERqu) (24 Sep): first EDA (match counts, name similarity).

Shared by teammates:
- [Entity Resolution EDA](https://claude.ai/artifact/2HqPwtsJZnB6kzTDeh2S2o)
- [Gen-4 Error Anatomy](https://claude.ai/artifact/Tr8fwRgS1HKtAK3DJbw54Z) (Vishal): the holdout side is exhausted, blank ties can't be separated, France is the lever.
- [T5 Ownership Report](https://claude.ai/artifact/LQWtUXTcJzBHi8VrarBuv2) (Vishal): ownership features behind T5.

## 8. Next steps
1. Get U2's public score from Ayush, then choose the final (U2, or the U3 fallback).
2. Read `artifacts/{web,leet}_train.json`. Ship a channel only if both dev and holdout gain; that would become U3 on top of the final base.
3. Support Tapan's zip dry-run (the matching file should reproduce byte for byte) and Vishal's doc: approach, blocking and the cut experiment, the France pipeline, experiments, and the disclosure.
4. Update `docs/SUBMISSION_TRACKER.md` with the final upload and score.
