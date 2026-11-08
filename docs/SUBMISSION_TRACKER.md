# Submission tracker — Amazon ML Challenge 2026

**Deadline:** 27 Sep 2026, midnight IST. **Submissions left:** 3 today. Best public: **0.989067 (H2)**.
**Last updated:** 26 Sep, 23:55

## Submissions

| # | Target time | What it is | What it tests | Expected public | Status |
|---|---|---|---|---|---|
| S1 | 26 Sep 18:45 | v5 top-10 blocker + v5 XGBoost matcher | Baseline for the new candidate set | 0.963–0.967 | **Done: 0.964** (same as v4, so the gap is in the matcher) |
| S2 | 26 Sep ~21:10 | L2 late fusion: v5 probability + mdeberta fold-A cross-encoder, `expected@0.80` | Does the cross-encoder (+1.2 holdout) show on test, and does it narrow the 0.9 gap? | 0.974–0.978 | **Ready, validated PASS** → `submissions/S2_mdeberta_L2/` (sha256 5eefe33f…) |
| S3 | 26 Sep ~21:40 | S2 + decoy weighting w=2 (Tapan's model), `expected@0.72` | Friend's decoy idea, one variable changed vs S2 | S2 ± 0.002 | **Done: 0.98112** (holdout 0.9847, gap 0.36 vs 0.9 before) |
| S4 | 26 Sep ~22:15 | **Big-data mdeberta** (150k extra train refs, 5× data) L2 late fusion **+ decoy w=3** (user choice; rule from dev) | More training data | +0.1 over S2 | **Done: 0.982855** (holdout 0.9857, gap 0.29) |
| S5 | 26 Sep 23:41 | mdeberta on 4.5M full-train pairs, L2 late fusion, decoy w=3, `threshold@0.68` | Full-train data | +0.05 | **Done: 0.983415** (holdout 0.98622, gap 0.28) |
| T1 | 27 Sep ~10:00 | **mdeberta trained on all 21.2M full-train pairs** (2-GPU, ~5 h overnight) + late fusion, decoy w=3 | Scaling the S3→S4→S5 lever to all data | ~0.984–0.986 | Vishal slot A overnight |
| T2 | 27 Sep ~12:00 | **Multi-cross-encoder late fusion**: mdeberta-full + Qwen3-0.6B (~6 h, ~6M pairs) + listwise + chunk models, decoy w=3, members kept only if they add on dev | Ensemble of complementary text models | T1 + 0.05–0.15 | Qwen on Tapan slot A; listwise on Ayush slot A |
| T3 | 27 Sep ~15:00 | T2 + feature-level fusion (103 test features) + set context | +0.08–0.11 holdout measured for feature-level + set/sibling | T2 + ~0.05–0.1 | Needs Vishal's test feature dump |
| T4 | 27 Sep ~19:00 | Best so far + one targeted fix from the production error audit (contextual/listwise or competition features) | Residual error class | TBD | Audit on the T2 model |
| T5 | 27 Sep by 22:30 | **Final**: best public-validated configuration | Safe final | — | — |
## Pending tasks by person

| Person | Now | Next | Deliverable for the doc |
|---|---|---|---|
| **Ayush** | Upload S2 then S3; full-train pair builder running (`1066865`); S4 auto-starts when 4 shards land | S3, S4; evaluate Qwen from saved scores; decide full-train run; T1 fusion test path | Matcher section (cross-encoders, fusion, EDA) |
| **Vishal** | S4 test scoring, big-data model, shards 0–1/4 | Test feature dump for top-10 (enables 129-feature fusion); if full-train run goes: train chunks 0–1; bge/e5 test scoring overnight | Blocker section; push pruning code to GitHub |
| **Tapan** | S4 test scoring, big-data model, shards 2–3/4 | Count model on `fusion/models/all_decoyw3/prob_all_rows.npy` (and `all`); if full-train run goes: train chunks 2–3; test scoring overnight | Validation protocol, count model, experiment log |

## Final approach (26 Sep 23:00 sweep)

**What works (measured on the leaderboard):** cross-encoder reading both records + late fusion with the v5 probability
(S1 0.964 → S3 0.9811), decoy weighting (test ~42% decoys vs 26% train), and more cross-encoder training data
(S3 → S4 +0.0017, gap 0.36 → 0.29).

**What we stop:** threshold/floor tuning, calibration, per-country rules, address dropout, count decoder (−0.0005 dev),
label-shift EM (test positive rate 0.333 vs 0.344: negligible), top-20 expansion (+0.0001), pseudo-labels.

**Overnight (26→27 Sep):**
| Slot | Job | Ends | Output |
|---|---|---|---|
| Vishal A | mdeberta on all 21.2M pairs, 2 GPUs, time-budgeted 330 min | ~05:15 | `mdeberta_full21/` model + sample logits |
| Vishal B | Test feature dump (top-10, 103 features) | ~02:00 | enables feature-level fusion on test |
| Tapan A | Qwen3-0.6B, 2 GPUs, short prompt, 360 min | ~06:30 | `qwen_big/` model + sample logits |
| Tapan B | Test scoring of chunk models 2 and 3 | ~04:00 | `test_ce_chunks/` |
| Ayush A | Listwise mdeberta (reference + all candidates in one sequence), 2 GPUs, 180 min, then sample + test lists | ~04:00 | `listwise/` |
| Ayush B | Multi-cross-encoder late-fusion evaluation on dev; then test scoring | — | member selection |

**Morning:** test-score mdeberta-full21 (4 GPUs, ~30 min) and Qwen (6 GPUs, ~70 min), then T1/T2.
Every member enters the fusion only if it improves **dev**; each final config is checked once on holdout.

## Key measured results (dev / holdout)

| Model | Dev | Holdout |
|---|---|---|
| v5 top-10 XGBoost matcher | 0.9754 | 0.9728 |
| L2 late fusion + mdeberta (S2) | 0.9862 | 0.9846 |
| L2 + decoy w=2 (S3) | 0.9861 | 0.9847 |
| Fusion: all cross-encoders + set/sibling + decoy w=2 | 0.9882 | 0.9858 |
| Fusion: same, decoy w=3 | 0.9881 | 0.9861 |
| L2 + big-data mdeberta (150k extra refs), decoy w=3 (S4) | 0.9873 | 0.9857 |
| L2 + Qwen3-0.6B (35k refs) | 0.9873 | 0.9852 |
| Feature-level + Qwen + set/sibling (35k refs) | 0.9882 | 0.9865 |

## Decisions log
- 27 Sep 23:21: **FINAL = U19, public 0.990273** (rank ~44).
  - Path from U5: U10 0.990102 → U13 0.990193 → U14 0.990225 → U16d 0.990237 → U18 0.99026 → U19 0.990273.
  - All gains came from France label-backed rules, Tapan's alias channels and France decoy drops.
  - Dropped after measurement: US/India never-retrieved adds (negative on the leaderboard); LoRA LLM judge on US/India (AUC 0.79 vs T6 0.88, holdout −0.00015).
  - Package: `artifacts/final_U19/` + `~/scratch/final_docs_U19/` (team names still to fill).
- 27 Sep 18:45: **FINAL = U5 (0.989868).**
  - Checked after U5, all dropped: every-row exact-copy check (U5 already links 100% of US/India exact copies); Claude blind judge (AUC 0.65 vs T6 0.79); Qwen2.5-7B judge (AUC 0.54 vs 0.88, holdout −0.00013, 0 France flips).
  - Remaining loss is same-name twins with blank addresses (58% of holdout error loss), invented aliases, and France without labels.
- 27 Sep 17:23: **U5 = 0.989868 public** (Vishal: T6 5-seed bag @0.68 US/India + rescue ranks 11-20 + U2 France [FRw4 + exclusivity prune + census v2]). +0.000233 over U1; expected ~0.9900. Next: U6 = U5 without France prune (measures the prune alone).
- 27 Sep 06:55: **T2 = 0.988085 public** (+0.00467 over S5), holdout 0.99011 -> gap 0.20 (S5 0.28). Stage-3b + census transfer. 4 submissions left today.
- 27 Sep 09:14: **H2 = 0.989067 public** (T5 + census, +0.00098 over T2; predicted 0.9889). Implied France F0.5 ~0.9776 (T2 0.9766), US/India ~0.991. Top 10 needs >= 0.9903 now, ~0.991 at close. 3 submissions left today.
- 27 Sep 10:15: **Own France patterns from manual reading** (france_cases.py, france_patterns.py): street-typo filler copies (6.6k rejected), type-word drop + filler in town+type names (0.9k), initials at the exact address (0.8k). Census v2 (CENSUS_V2=1, off by default; v1 reproduces H2 byte for byte) restores +7,174. 40/40 sampled restores read as true; train check: same-name + typo street is 100% true (US 947, India 472), so a typo is noise, not a decoy. Staged submissions/H3_H2_censusv2 (sha a078078a, validator PASS, France 3.251 links/ref vs H2 3.224, US/India = H2).
- 27 Sep ~13:00: **U1 = 0.989635 public** (T5 US/India + FRtrained_w4 France + census v2; France F ~0.9776 -> ~0.9814). Ruled out after U1: label shift, stage-3b ensembles (dev picks T6), shifted-number restores (decoy shift set +1..+21), co-located name removal (p>=0.9 is ~100% true in train), round-2 self-training (converged), France blocking misses (99.988% exact copies in candidates), cross-source completion (408 pairs, 0-15% true in train), row-order/ID leakage (none).
- 27 Sep ~14:05: **Exclusivity prune (own idea)**: a record belongs to one S1, so its p over S1s must sum <= 1; France accepted links on over-committed records (sum > 1.2) are 1.1% vs 0.1% US/India. exclusivity_prune.py drops 3,858 France links (3,077 blank coin-flip ties + 781 at another number/street) when p/S < 0.77 and the S1 keeps another link; same-street links kept. Staged **U2** = T6 US/India + FRtrained_w4 France + prune + census v2: submissions/U2_T6_FRw4_xprune (sha 9881548f, validator PASS, top-10 candidates 716e89aa).
- 27 Sep 06:50: **T2 ready** (T1b + census fix incl. A/C, identical-name exemption, B only when a name word is dropped; built on Vishal's slot): validator PASS locally (100,976 empty); vs S5 empty 5.83%, France links/ref 3.206 (S5 3.180), France empty 6.17% (S5 6.12%), newly emptied at S5 p>=0.99: 205. Census: restored 16,887, rejected 5,271 (B still skips 24,894 drop-edited restores; conservative vs other team's 32k). sha256 cd21f80baafb26cf. User decision: upload T2, not T1b.
- 27 Sep 06:30: **T1b ready** (stage-3b chunk2 + W1 + French features, holdout 0.99011): validator PASS; vs S5 empty 5.84% (S5 5.80%), newly emptied at S5 p>=0.99 only 109 (T1: 11,012), France links/ref 3.161 (S5 3.180). Transfer audit PASS: stage-3b output identical at p1/p10/p50/p90 on sure pairs, sample vs test. T2 (T1b + census): restored 14,055 / rejected 5,271, France links/ref 3.195; weak-name rule B skipped 28,077 restores (too strict for template names). Plan: upload T1b first (measures stage-3b transfer), then T2.
- 27 Sep 06:20: T4 retrain (chunk2 + full21 + decoyw2 + house-number shift): dev 0.99191 / holdout 0.99037 (T1b model 0.99138 / 0.99011); T4 test path running. Tapan's chunk2 French self-training done (06:10). T3 = T1b -> France AND-veto (self-trained view = T1b stage-3b with France chunk2 scores from Tapan's model) -> census fix; auto-chained (france_veto.py, t3self_job.sh, t3_final_job.sh). Vishal uploading hybrid (new CEs + ownership US/India, S5 France) to measure US/India transfer.
- 27 Sep 05:45: census tightenings from manual review integrated (france_census.py): A skip restores when the S1 has no candidate at another house number; B shared rarest core word must be in <= 50 French S1 names (city names are not identity); C space-separated compound numbers parsed ('108 4 Rue' = number 108, sub-number 4). Uploaded before T2 auto-run. Vishal's ownership eval: S5 L2 + ownership + addr + raw holdout 0.98905 (+0.0028); main ownership block file not yet in ownership_v1/.
- 27 Sep 05:50: census fix aligned with the other team's rules (restore: strict same address, first + rarest core word kept, only filler added, drops allowed; reject: accepted record at the S1's house number adding a type word): on T1 34,276 restored / 7,552 rejected (theirs 32,439 / 11,655), PASS. Manual review page: https://claude.ai/artifact/5G5pjK4JT4q9fajka4QztM (3 likely failure patterns A/B/C proposed).
- 27 Sep 05:50: Vishal's transfer check: full21 and decoyw2 pass (agreed-pair p10 sample->test 12.25->12.19 and 11.81->11.81). T4 retrain started (boundary_models_t4): chunk2 + full21 + decoyw2 + house-number shift features (num_shift, shift_up_1_13, num2_eq).
- 27 Sep 05:20: friend's top-10 write-up (0.99028): adopted (1) France same-address census fixes (restore filler-word records: & Fils/Associés/Cie/Services/Groupe/Développement/France; reject type-word swaps) -> france_census.py; (2) self-training only as an AND-veto; (3) number-shift direction features (later). Our French synthetic fine-tune encoded wrong France decoy rules -> fr_aug/fr_self not used; fr_self job killed; Tapan's chunk2 self-training set to AUG_P=0.
- 27 Sep 05:20: chunk2-only stage-3b + W1 + French features (boundary_models_c2_w1): dev 0.99138 / holdout 0.99011 -> T1b test path running. Census fix tested on T1: 15,332 restored, 5,098 rejected, validator PASS; auto-applied to T1b -> T2.
- 27 Sep 04:50: **T1 (stage-3b chunk2+listwise) NOT submitted.** Validator PASS, but vs S5 it empties 16,850 refs S5 matched at median p=1.0 (T1 empty 6.7% vs S5 5.8%). Cause: listwise CE does not transfer to test lists — on pairs with v5>=0.99 & chunk2>=9 its logit p10 is 5.5 on test vs 12.1 on the sample (0.0% below 1 on sample); dropped S5 matches are obvious true matches (same name+address) with listwise ~0 amid decoy families. Listwise removed from the submission path; retraining stage-3b on chunk2 only + W1 + French features (boundary_models_c2_w1) -> T1b. W1 with listwise: 0.99120 vs 0.99096 (small). France listwise self-training dropped (DDP crash, listwise out); pairwise self-training rerun.
- 27 Sep 05:30: France self-training approved (rules allow self-training on test; user: only when VERY confident). france_pseudo.py: positive = stage-3b >=0.995 & v5 >=0.9 & chunk2 >=4 & listwise >=4 & in T1 final matches; negative = all four at the opposite extreme; listwise uses only fully-confident France lists. Fine-tunes listwise (from our listwise) and pairwise (from French-noise model); fusion/stage-3b stay train-only. France-only re-scoring + US/India no-regression check (multi_ce_eval_frself.json). Auto-queued after T1.
- 27 Sep 05:10: gen3_eda (stage-3b c2lw, holdout 0.99093): loss 0.907 pts (S5 1.378); over −69%, singleton FP −82%; remaining: 815 blank FN (same-name ties, features identical to rejected rows), 149 non-blank FN = synthetic alias names at the exact same address. W1 address features added (order-free keys: house number + rare street words; S1 entities / S2-S3 name keys per key, exact-key agreement, unit agreement, no shared name word); retrain boundary_models_c2lw_w1 queued. Page: https://claude.ai/artifact/FBozAaWfP6UNWc8JySecMG
- 27 Sep 04:45: **stage-3b with chunk2 + listwise CEs (boundary_models_c2lw): stage3+rev+num expected@0.60 dev 0.99260 / holdout 0.99093 (+0.0047 over S5)**; stage3+rev 0.99271 / 0.99096; stage3 0.99213 / 0.99026. T1 test path running (job 1067301). French-noise fine-tune restarted from S5 CE (Tapan's model unreadable, mode 600).
- 27 Sep 04:00: **listwise CE (ref + all candidates in one sequence) L2 fold-B: dev 0.99020 / holdout 0.98868 vs chunk2 0.98805 / 0.98672 (+0.0020)**; chunk2+listwise 0.99040 / 0.98888. Stage-3b now takes chunk2 + listwise (boundary_models_c2lw) -> T1 test path.
- 27 Sep 04:00: blank expert: segment cutoffs 0.98970, identity features 0.98988, blank-only model 0.98996 holdout; all below stage-3b dev -> dropped.
- 27 Sep 03:30: affix_check (train labels): 'S1 name + group/holdings/international/development/sons/distribution/...' = 0% positive even at same house number; 'center/services/partners/the/com' = 77-99%. France 'X Groupe/Développement/Et Fils' records are affiliate decoys -> CE veto correct; suffix-stripping idea dropped.
- 27 Sep 03:30: blank_disagree: 830 blank FN are NOT v5-vs-CE disagreements (strong disagreement cells hold ~20-40 FN); 241 in both-say-no cell (pos 0.6%), rest in both-uncertain cells with calibrated p. FN vs rejected profiles identical (~5 rival S1 within 0.02) -> chain ambiguity.
- 27 Sep 03:30: stage-3 features: French legal forms added to name-key STOP, 'group' removed from STOP, new decoy_word feature (EN list + FR equivalents). Chunk2 stage-3b retrain -> T1 test path (stage3b_predict.py) auto-queued; French-noise augmentation fine-tune (train_fr_aug.py, from chunk2 model) auto-queued when listwise frees GPUs.
- 27 Sep 02:23: boundary_exp2 (record-centric competition over full S1 table via e5 retriever + house numbers): **stage3+rev+num expected@0.60 dev 0.99124 / holdout 0.98985** (+0.0015 over stage3, +0.0036 over S5). rev alone 0.98957; num alone +0.00001. Top new features: name-only reverse gap, full reverse gap. Holdout errors vs S5: blank FN 1031->830, blank FP 141->34, non-blank FN 249->232, FP 103->51.
- 27 Sep 01:00: blank_eda: blank-address rows = 27% of holdout candidates; label-perfect blank decisions -> 0.99488 (+0.0087) vs non-blank 0.98954 (+0.0033). Blank band negatives: other entity same name 828, other entity 527, decoy 43. S1 name multiplicity 1 -> pos 0.885 vs p 0.661 (under-confident); 3+ -> over-confident. Sibling similarity: no extra signal. Ref lacking accepted same-source record: pos 0.589 vs p 0.486.
- 27 Sep 01:00: france_check (test, label-free): France expected F0.5 0.9731 vs India 0.9887 / US 0.9870; CE rejects 7.4% of France v5>=0.9 pairs vs 0.74-0.85%. Driver: French suffix added (43.5% vs 2.6%) + core word dropped, same house number -> CE wrong on French name noise. Est. +0.0013 public if fixed.
- 27 Sep 01:00: multi_ce_eval (fold-B protocol): chunk2 0.98672 / chunk3 0.98670 holdout vs mdebfull45 0.98630; pairs/all add <= +0.0001. CE ensembling saturated.
- 27 Sep 00:30: gen2_eda on S5: loss 1.38 pts = stop decision 1.04 (early 0.50, late 0.31, empty 0.16, singleton 0.07), blocking 0.20, ranking 0.13. Probabilities calibrated; best-cut oracle 0.99674.
- 27 Sep 00:30: boundary_exp: **stage-3 XGBoost on 26 list/global-name features, threshold@0.66: dev 0.99034 / holdout 0.98839 (+0.0022 over S5)**. Cut transformer 0.98942 / 0.98782. Top new feature: S1 name multiplicity. Candidate for T-submission (needs test path).
- 26 Sep 23:55: decision_tweaks (exact expected-F0.5 DP, 9 cheap flags) on the S4 setup: no holdout gain (best +0.00002). Dropped. Running gen2_eda (ranking vs boundary loss split for S5).

- 26 Sep 18:45 — S1 = 0.964 equals v4: blocker is not the source of the holdout-to-public gap.
- 26 Sep 19:00 — Late fusion ≈ feature-level (−0.0002 holdout): no test feature rerun needed for S2–S4.
- 26 Sep 19:05 — Denser distractor pools (top-20) do not hurt L2: no plain stricter-floor probe.
- 26 Sep 20:00 — Decoy implied weight 2.056 (train decoy share 26%, test ~42%): S3 uses w=2.
- 26 Sep 20:45 — First S2 run found 0 stage-1 probabilities (glob on unlistable folder); guarded and rerun. Never submit a run whose log shows stage-1 "found for" far below ~7.9M of 17.3M pairs.
- 26 Sep 20:45 — Full-train run on mdeberta, not Qwen: Qwen is 6× slower (~14 h on 4 GPUs) and unmeasured.
- 26 Sep 20:50 — 150k extra train refs lift L2 holdout 0.98459 → 0.98577: more data helps; full-train mdeberta run approved.
- 26 Sep 20:50 — Vishal's test scores drop pairs < 0.02 by design, so ~45% coverage is normal; guard checks both files load and ≥25% found.
- 26 Sep 21:00 — S3 (mdeberta + decoy w=2) = 0.98112 public, +1.7 over S1; gap 0.36. S4 switched to big-data model with decoy w=2.
- 26 Sep 21:40 — S4 decoy weight set to 3 (user decision); w=3 late model trained on the big-data cross-encoder, rule chosen on dev.
- 26 Sep 21:35 — Qwen3-0.6B beats mdeberta at equal data (L2 dev 0.98726 vs 0.98624, holdout 0.98516 vs 0.98459): candidate for overnight multi-GPU training.
- 26 Sep 22:27 — S4 (big-data mdeberta 1.86M pairs + decoy w=3) = 0.982855, +0.0017 over S3: more CE training data transfers to test.
