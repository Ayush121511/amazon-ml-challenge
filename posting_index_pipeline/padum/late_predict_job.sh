#!/bin/bash
#PBS -N er_latepred
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=4:ngpus=1:mem=160gb
#PBS -l walltime=03:00:00
#PBS -j oe
# qsub -v CAND=...,S1=<glob scores.npz>,CEG=<glob ce npz>,LATE=<late model dir>,OUT=<dir> padum/late_predict_job.sh
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
V=/home/scai/mtech/aib262467/scratch/amazon_ml_2026
"$V/.conda-er/bin/python" -u business_entity_resolution/src/late_predict.py --candidates "$CAND" --stage1-scores "$S1" \
  --ce-scores "$CEG" --late-model "$LATE" --output "$OUT" ${RULE:+--rule "$RULE"}
for d in "$OUT"/*/; do
  [ -f "$d/matching_results.tsv" ] || continue
  echo "=== VALIDATE $d"
  "$V/.conda-er/bin/python" data/utils/validate_submission.py --matching "$d/matching_results.tsv" \
    --candidate "$d/candidate_pairs.tsv" --test-dir data/dataset/test --check-ids
done
