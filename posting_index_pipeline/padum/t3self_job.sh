#!/bin/bash
#PBS -N er_t3self
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=8:ngpus=1:mem=300gb
#PBS -l walltime=03:00:00
#PBS -j oe
# Self-trained view for the France AND-veto: T1b's stage-3b with France pairs' chunk2 scores replaced by Tapan's
# self-trained chunk2 (later CE_TEST files override earlier ones). Plus the US/India no-regression check of that model.
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
V=/home/scai/mtech/aib262467/scratch/amazon_ml_2026
PY="$V/.conda-er/bin/python"
T=/home/scai/mtech/aib262140/scratch/amazon_ml_2026
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false NPROC=8
MODEL_DIR=artifacts/fusion/boundary_models_c2_w1 OUT=artifacts/submission_T3self \
  CE_TEST="$T/test_ce_chunks/ce_mdeb_chunk2_shard0of1.npz;$T/ftchunk2_fr/test_france/ce_c2fr_shard0of2.npz;$T/ftchunk2_fr/test_france/ce_c2fr_shard1of2.npz" \
  $PY -u business_entity_resolution/src/stage3b_predict.py
