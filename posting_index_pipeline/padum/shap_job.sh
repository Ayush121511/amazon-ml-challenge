#!/bin/bash
#PBS -N er_shap
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=4:ngpus=1:mem=48gb
#PBS -l walltime=00:40:00
#PBS -j oe
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
V=/home/scai/mtech/aib262467/scratch/amazon_ml_2026
"$V/.conda-er/bin/python" -u business_entity_resolution/src/shap_misclassified.py "$HOME/scratch/vishal_v4_artifacts" 4 \
  $V/artifacts/prune_v1/model_top10 $V/artifacts/prune_v1/mask_top10.npy
