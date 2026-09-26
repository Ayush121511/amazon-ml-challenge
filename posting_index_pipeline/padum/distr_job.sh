#!/bin/bash
#PBS -N er_distr
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=1:mem=64gb
#PBS -l walltime=01:00:00
#PBS -j oe
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
V=/home/scai/mtech/aib262467/scratch/amazon_ml_2026
nvidia-smi -L
"$V/.conda-er/bin/python" -u business_entity_resolution/src/distractor_check.py
