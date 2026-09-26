#!/bin/bash
#PBS -N er_cutoff
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=1:mem=32gb
#PBS -l walltime=00:30:00
#PBS -j oe
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
PY="$HOME/scratch/miniconda3/envs/amlc/bin/python"
"$PY" -u business_entity_resolution/src/blocking_cutoff_cost.py "$HOME/scratch/vishal_v4_artifacts"
