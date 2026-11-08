#!/bin/bash
#PBS -N er_listw
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=2:mem=200gb
#PBS -l walltime=06:00:00
#PBS -j oe
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
V=/home/scai/mtech/aib262467/scratch/amazon_ml_2026
nvidia-smi -L
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
MAX_MIN=${MAX_MIN:-180} TEST_CAND=$V/artifacts/submission_v5_top10/candidate_pairs.tsv OUT_DIR=$HOME/scratch/amazon_ml_2026/artifacts/listwise \
  $V/.conda-er/bin/python -m torch.distributed.run --nproc_per_node=2 business_entity_resolution/src/train_listwise.py
