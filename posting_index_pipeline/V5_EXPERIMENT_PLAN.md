# Tapan v5 experiments: start from the submitted v4

The reported platform score is 0.964. The saved US/India holdout score is 0.9730633.
No improvement has yet been measured with the changes below. France has no labeled
validation. Its similar prediction counts do not establish matching accuracy.

## Verified findings and limits

- `merge_reverse` previously dropped combined/phonetic route scores. The new code
  passes these through when present and uses zeros for older caches. Existing v4
  model feature lists remain compatible. It does not change the candidate union.
- `reverse_index.py` stores **fused rank**, not per-route rank. There is no cached
  `rank_combined` to expose. Do not fabricate one from fused rank or the retained top 10.
- The current source supports combined scores, but v4 reuses `train_reverse_t` and
  `train_ref_index` created earlier. Only the cache preflight can establish whether
  the actual artifacts contain useful nonzero combined scores.
- `dense_*.npz` stores reverse top-k scores for each target. Target best score and
  top1-versus-top2 margins can be computed without a new embedding search. Adding
  them to the existing training pairs still requires an ID-aligned feature pass.
- E5 training shuffles positive pairs without carrying entity IDs into the batch.
  If a reference appears twice, the diagonal-only contrastive loss treats its other
  positive target as a negative. The collision frequency and resulting score impact
  have not been measured. Fixing this is not a demonstrated path to a large gain.

## First job: unchanged candidates and feature matrix

`v5_weighting_dev.pbs` runs three controlled XGBoost configurations with the v4
hyperparameters, using the exact existing pair arrays:

| Run | Pair training weight before mean-one normalization |
| --- | --- |
| weight_0 | 1 (baseline rerun) |
| weight_0.5 | 1 / sqrt(candidate count for this reference) |
| weight_1 | 1 / candidate count for this reference |

The full weighting gives equal total training weight to each reference. It does
not make logloss identical to macro F0.5 and could underweight difficult crowded
references; it is an experiment. Dev early stopping keeps the original unweighted
logloss so that only training weights change.

All three use a 0.01 dev cutoff grid for threshold/expected rules. The grid groups
and sorts once and reuses prefix counts, preserving the original decision logic.
`--dev-only` prevents holdout prediction/scoring. Compare dev results against the
rerun baseline; do not choose a winner by evaluating three holdout variants.

The newly exposed reverse scores are **not** retroactively added to saved `x.npy`.
This first job tests weights only. Next, use the preflight result to choose an
aligned extra-feature pass or abandon empty route fields. Do not rebuild all
indexes and embeddings just to enable feature columns.

## Run on Tapan's account with shared read access

GitHub contains code and aggregate reports, not the 13 GB pair matrix or fitted
models/caches. Vishal must grant `aib262140` read access and directory traversal.
The minimum files for weighting are under:

```
/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/matcher_v4/pairs/
  x.npy, y.npy, pairs.tsv.gz, gold.json, countries.json, report.json
```

Read access to the model, reverse/reference-index manifests and dense cache folders
is also useful for diagnostics. Shared inputs are never edited. Outputs go to
Tapan's `~/scratch/amazon_ml_2026/artifacts/tapan_v5_weighting/`; no 13 GB copy is made.

Deploy the bundled `tapan_v5` folder under `~/scratch/amazon_ml_2026/`, keeping it
separate from Tapan's older pipeline. Then on the Padum login node:

```bash
cd ~/scratch/amazon_ml_2026
bash tapan_v5/padum/check_v5_inputs.sh
qstat -u aib262140
```

If inputs and packages are available and a job slot is free:

```bash
qsub tapan_v5/padum/v5_weighting_dev.pbs
```

`V4_ROOT`, `V5_CODE`, and `V5_PYTHON` can override the defaults via `qsub -v`.
The default Python is Tapan's `.conda-er/bin/python`; it must have working CUDA
PyTorch and XGBoost 2.1.4. The job verifies GPU visibility and stops on missing
inputs/packages; it does not silently install software or fall back to CPU.
If packages are missing, first arrange read access to Vishal's offline wheels
and prepare the environment on a compute node. Check quota before installing.

The job requests the same 8 CPUs / 96 GB class as the original trainer and one GPU.
The old 7.5-minute measurement is model training, not a guarantee for data loading,
three variants, preflight, or queue waiting. The walltime request is three hours.
It writes three small models/reports, not another full feature matrix.

## Selection and next phases

1. Compare dev scores and chosen rules; inspect the cache preflight. Keep baseline
   if weighting has no stable gain. Small gains need a paired per-reference check.
2. Add useful cached reverse/dense context features using an aligned sidecar to
   avoid a second large feature matrix. Verify identical candidate IDs and train/test
   feature definitions. Compare the feature change separately, then combine wins.
3. If these changes do not close enough of the gap, build a second-stage matcher
   using out-of-fold first-stage scores and candidate/anchor agreement. Reserve the
   E5 rerun or an uncertain-pair cross-encoder for errors that justify their cost.
4. Freeze one configuration on dev; evaluate holdout once. If it regresses, stop
   and diagnose rather than iterate against holdout. Benchmark France transfer
   separately; there is no labeled French score in the current artifacts.
5. Run full test inference only for a justified winner, validate with the organizer
   CLI, package on Padum, and submit. A new platform improvement is not guaranteed.
