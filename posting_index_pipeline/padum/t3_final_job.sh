#!/bin/bash
#PBS -N er_t3fin
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=1:mem=96gb
#PBS -l walltime=01:00:00
#PBS -j oe
# T3 = T1b -> France AND-veto with the self-trained view -> France census fix -> validator.
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
V=/home/scai/mtech/aib262467/scratch/amazon_ml_2026
PY="$V/.conda-er/bin/python"
BASE_DIR=artifacts/submission_T1b_s3b_c2w1 SELF_DIR=artifacts/submission_T3self OUT_DIR=artifacts/submission_T3_veto \
  $PY -u business_entity_resolution/src/france_veto.py
SUB_DIR=artifacts/submission_T3_veto OUT_DIR=artifacts/submission_T3 $PY -u business_entity_resolution/src/france_census.py
for d in artifacts/submission_T3/*/; do
  echo "=== VALIDATE $d"
  $PY data/utils/validate_submission.py --matching "$d/matching_results.tsv" --candidate "$d/candidate_pairs.tsv" \
    --test-dir data/dataset/test --check-ids
done
