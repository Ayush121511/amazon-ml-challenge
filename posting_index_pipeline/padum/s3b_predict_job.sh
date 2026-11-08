#!/bin/bash
#PBS -N er_s3b_pred
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=8:ngpus=1:mem=300gb
#PBS -l walltime=04:00:00
#PBS -j oe
# qsub -v MODEL_DIR=...,OUT=...,CE_TEST='a.npz;b.npz' padum/s3b_predict_job.sh
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
V=/home/scai/mtech/aib262467/scratch/amazon_ml_2026
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false NPROC=8
MODEL_DIR=$MODEL_DIR OUT=$OUT CE_TEST="${CE_TEST:-}" CE2_TEST="${CE2_TEST:-}" OWN_TEST="${OWN_TEST:-}" "$V/.conda-er/bin/python" -u business_entity_resolution/src/stage3b_predict.py
for d in "$OUT"/*/; do
  [ -f "$d/matching_results.tsv" ] || continue
  echo "=== VALIDATE $d"
  "$V/.conda-er/bin/python" data/utils/validate_submission.py --matching "$d/matching_results.tsv" \
    --candidate "$d/candidate_pairs.tsv" --test-dir data/dataset/test --check-ids
done
