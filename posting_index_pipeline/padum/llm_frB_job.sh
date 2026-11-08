#!/bin/bash
#PBS -N er_llmfrB
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=2:mem=120gb
#PBS -l walltime=01:00:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
L=$HOME/scratch/amazon_ml_2026/artifacts/llm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
T="$L/train_band.npz|train|$L/train_llm;$L/fr_pairs.npz|test|$L/fr_llm"
until [ -f $L/pairs.ready ]; do sleep 20; done
IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"
for i in 2 3; do CUDA_VISIBLE_DEVICES=${G[$((i-2))]} MODE=score SHARD=$i NSHARD=4 TASKS="$T" $PY -u llm_probe.py > $L/score_$i.log 2>&1 & done
wait
