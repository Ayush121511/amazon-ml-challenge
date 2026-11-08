#!/bin/bash
#PBS -N er_frlab
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=1:mem=96gb
#PBS -l walltime=01:00:00
#PBS -j oe
# France pseudo-labels (census + extreme pairs) for the France-trained stage-3b, on Vishal's robust run's pair order.
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
V=/home/scai/mtech/aib262467/scratch/amazon_ml_2026
SUB_DIR=$V/artifacts/submission_FRrobust FINAL_DIR=$V/artifacts/submission_FRrobust_census OUT=artifacts/fr_labels \
  $V/.conda-er/bin/python -u business_entity_resolution/src/fr_labels.py
chmod -R a+rX artifacts/fr_labels
