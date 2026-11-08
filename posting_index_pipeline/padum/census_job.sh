#!/bin/bash
#PBS -N er_census
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=1:mem=96gb
#PBS -l walltime=01:00:00
#PBS -j oe
# qsub -v SUB_DIR=<submission folder with late_scores.npz>,OUT_DIR=<new folder> padum/census_job.sh
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
V=/home/scai/mtech/aib262467/scratch/amazon_ml_2026
SUB_DIR=$SUB_DIR OUT_DIR=$OUT_DIR "$V/.conda-er/bin/python" -u business_entity_resolution/src/france_census.py
for d in "$OUT_DIR"/*/; do
  echo "=== VALIDATE $d"
  "$V/.conda-er/bin/python" data/utils/validate_submission.py --matching "$d/matching_results.tsv" \
    --candidate "$d/candidate_pairs.tsv" --test-dir data/dataset/test --check-ids
done
