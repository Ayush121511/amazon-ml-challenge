#!/bin/bash
#PBS -N er_ddp
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=8:ngpus=4:mem=320gb
#PBS -l walltime=02:30:00
#PBS -j oe
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
V=/home/scai/mtech/aib262467/scratch/amazon_ml_2026
PY="$V/.conda-er/bin/python"
nvidia-smi -L
until [ -f artifacts/fulltrain_pairs/README.txt ]; do sleep 30; done
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
OUT=artifacts/cross_encoder_v1/mdeberta_full
MAX_MIN=${MAX_MIN:-75} TRAIN_PAIRS=${TRAIN_PAIRS:-12000000} OUT_DIR=$OUT $PY -m torch.distributed.run --nproc_per_node=4 \
  business_entity_resolution/src/train_ddp.py
TAG=mdeberta_full CE_SAMPLE=$OUT/ce_logit_sample.npy $PY -u business_entity_resolution/src/late_full_train.py
