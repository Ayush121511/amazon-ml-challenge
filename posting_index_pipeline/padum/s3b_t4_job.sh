#!/bin/bash
#PBS -N er_s3b_t4
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=4:ngpus=1:mem=96gb
#PBS -l walltime=01:30:00
#PBS -j oe
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
CE_FILE=/home/scai/mtech/aib262140/scratch/amazon_ml_2026/ftchunk2/ce_logit_sample.npy CE2_FILE='/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/mdeberta_full21/ce_logit_sample.npy;/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/mdeberta_decoyw2/ce_logit_sample.npy' BOUNDARY_OUT=boundary_models_t4 TAG=_t4 \
  /home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python -u business_entity_resolution/src/boundary_exp2.py
