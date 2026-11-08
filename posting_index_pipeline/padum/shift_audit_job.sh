#!/bin/bash
#PBS -N er_shift
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=1:mem=96gb
#PBS -l walltime=00:45:00
#PBS -j oe
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
M=artifacts/fusion/boundary_models_c2_w1
CACHE=$M/exp2_cache.npz META=$M/stage3b_meta.json P3_SAMPLE=$M/stage3b_prob.npy TEST_DIR=artifacts/submission_T1b_s3b_c2w1 \
  /home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python -u business_entity_resolution/src/shift_audit.py
