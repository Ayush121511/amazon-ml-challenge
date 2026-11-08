#!/bin/bash
#PBS -N er_bexpert
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=4:ngpus=1:mem=96gb
#PBS -l walltime=01:30:00
#PBS -j oe
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
$PY -u business_entity_resolution/src/blank_expert.py
