#!/bin/bash
#PBS -N er_fusion
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=3:ngpus=1:mem=96gb
#PBS -l walltime=08:00:00
#PBS -j oe
# STEPS="stage1+ce:tag@base@epochs@lr@drop@fold+...+fuse:tag1,tag2"
set -uo pipefail
cd "$HOME/scratch/amazon_ml_2026"
V=/home/scai/mtech/aib262467/scratch/amazon_ml_2026
PY="$V/.conda-er/bin/python"
SRC=business_entity_resolution/src
nvidia-smi -L
IFS='+' read -ra ST <<< "$STEPS"
for s in "${ST[@]}"; do
  echo "=== STEP $s $(date +%T)"
  case "$s" in
    stage1) $PY -u $SRC/fusion_stage1.py ;;
    report) $PY -u $SRC/ce_report.py ;;
    wait1) until [ -f artifacts/fusion/stage1.npz ]; do sleep 60; done ;;
    waitf:*) until [ -f "${s#waitf:}" ]; do sleep 60; done ;;
    mv:*) IFS='@' read -r A B <<< "${s#mv:}"; [ -f "$A" ] && [ ! -f "$B" ] && cp "$A" "$B" ;;
    ce:*) IFS='@' read -r TAG BASE EP LR AD FOLD <<< "${s#ce:}"
          TAG=$TAG LR=$LR ADDR_DROP=$AD FOLD=$FOLD EVAL=0 $PY -u $SRC/cross_encoder_exp.py "$BASE" mask_top10.npy "$EP" ;;
    fuse:*) TAGS="$(echo ${s#fuse:} | tr % ,)" $PY -u $SRC/fusion_train.py ;;
  esac
done
