#!/bin/bash
#PBS -N er_t1chk
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=1:mem=96gb
#PBS -l walltime=00:45:00
#PBS -j oe
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python -u business_entity_resolution/src/t1_empty_check.py
