#!/bin/bash
#PBS -N er_calib
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=00:30:00
#PBS -j oe
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
PY="$HOME/scratch/miniconda3/envs/amlc/bin/python"
"$PY" -u business_entity_resolution/src/calibrate_probs.py "$HOME/scratch/vishal_v4_artifacts" 8
