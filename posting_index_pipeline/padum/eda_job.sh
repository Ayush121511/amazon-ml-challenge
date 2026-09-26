#!/bin/bash
#PBS -N er_eda
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=01:30:00
#PBS -j oe
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
"$HOME/scratch/miniconda3/envs/amlc/bin/python" -u business_entity_resolution/src/matcher_eda.py "$SET" 4
