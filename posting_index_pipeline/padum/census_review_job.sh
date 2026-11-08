#!/bin/bash
#PBS -N er_cenrev
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=1:mem=64gb
#PBS -l walltime=00:40:00
#PBS -j oe
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
BASE_DIR=$BASE_DIR FIX_DIR=$FIX_DIR /home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python -u business_entity_resolution/src/census_review.py
