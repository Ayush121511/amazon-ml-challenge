#!/bin/bash
#PBS -N er_ctxA
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=2:mem=100gb
#PBS -l walltime=01:00:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
L=$HOME/scratch/amazon_ml_2026/artifacts/llm
export A=$HOME/scratch/amazon_ml_2026/artifacts/sweep HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OUT=$L/ctx_llm
IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"
k=0; for i in 0 1; do CUDA_VISIBLE_DEVICES=${G[$k]} SHARD=$i NSHARD=4 $PY -u llm_ctx.py > $L/ctx_$i.log 2>&1 & k=$((k+1)); done
wait
for k in $(seq 1 90); do [ -f $L/ctx_llm_2.npy ] && [ -f $L/ctx_llm_3.npy ] && break; sleep 20; done
MODE=eval PAIRS=$L/train_band.npz OUT=$L/ctx_llm $PY -u llm_probe.py > $L/ctx_eval.log 2>&1
