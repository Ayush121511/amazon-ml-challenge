#!/bin/bash
#PBS -N er_t5own
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=4:ngpus=1:mem=200gb
#PBS -l walltime=01:30:00
#PBS -j oe
# T5 = T4's stage-3b inputs + Vishal's ownership blocks (26 cols). Training reuses T4's cached features; the test
# prediction reuses T4's saved test features and only appends the ownership test columns.
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
S=business_entity_resolution/src
W=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts
O=$W/ownership_v1
CE_FILE=/home/scai/mtech/aib262140/scratch/amazon_ml_2026/ftchunk2/ce_logit_sample.npy \
  CE2_FILE="$W/mdeberta_full21/ce_logit_sample.npy;$W/mdeberta_decoyw2/ce_logit_sample.npy" \
  OWN_FILE="$O/own_sample.npy;$O/own_addr_sample.npy;$O/own_raw_sample.npy" \
  CACHE_FROM=artifacts/fusion/boundary_models_t4 BOUNDARY_OUT=boundary_models_t5 TAG=_t5 $PY -u $S/boundary_exp2.py
BASE_TEST=artifacts/submission_T4_s3b MODEL_DIR=artifacts/fusion/boundary_models_t5 \
  OWN_TEST="$O/own_test.npy;$O/own_addr_test.npy;$O/own_raw_test.npy" OUT=artifacts/submission_T5_own $PY -u $S/stage3b_repredict.py
for d in artifacts/submission_T5_own/*/; do
  $PY data/utils/validate_submission.py --matching "$d/matching_results.tsv" --candidate "$d/candidate_pairs.tsv" \
    --test-dir data/dataset/test --check-ids
done
