#!/bin/bash
#PBS -N er_testce
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=4:ngpus=1:mem=96gb
#PBS -l walltime=04:00:00
#PBS -j oe
# qsub -v CAND=<candidate_pairs.tsv>,MODEL=<hf dir>,NAME=<tag>,SHARD=0/2 padum/test_ce_job.sh
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
V=/home/scai/mtech/aib262467/scratch/amazon_ml_2026
nvidia-smi -L
"$V/.conda-er/bin/python" -u business_entity_resolution/src/score_ce_test.py --candidates "$CAND" --model "$MODEL" \
  --name "$NAME" --out artifacts/test_ce --shard "$SHARD" --max-len 128 --batch 1024
