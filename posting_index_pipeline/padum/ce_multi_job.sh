#!/bin/bash
#PBS -N er_ce
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=1:mem=64gb
#PBS -l walltime=04:00:00
#PBS -j oe
# CONFIGS="tag@base_dir@epochs@lr@addr_drop+..."
set -uo pipefail
cd "$HOME/scratch/amazon_ml_2026"
V=/home/scai/mtech/aib262467/scratch/amazon_ml_2026
nvidia-smi -L
IFS='+' read -ra RUNS <<< "$CONFIGS"
for run in "${RUNS[@]}"; do
  IFS='@' read -r TAG BASE EP LR AD <<< "$run"
  echo "=== RUN $TAG base=$BASE epochs=$EP lr=$LR addr_drop=$AD"
  TAG=$TAG LR=$LR ADDR_DROP=$AD "$V/.conda-er/bin/python" -u business_entity_resolution/src/cross_encoder_exp.py "$BASE" mask_top10.npy "$EP"
done
