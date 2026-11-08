#!/bin/bash
#PBS -N er_llmprobe
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=3:ngpus=3:mem=120gb
#PBS -l walltime=01:30:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
S=$HOME/scratch/amazon_ml_2026/artifacts/sweep
L=$HOME/scratch/amazon_ml_2026/artifacts/llm
mkdir -p $L
export A=$S PAIRS=$L/train_band.npz OUT=$L/train_llm SPLIT=train HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODE=pairs_train $PY -u llm_probe.py > $L/probe.log 2>&1
IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"
N=${#G[@]}
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${G[$i]} MODE=score SHARD=$i NSHARD=$N $PY -u llm_probe.py > $L/score_$i.log 2>&1 &
done
wait
MODE=eval $PY -u llm_probe.py >> $L/probe.log 2>&1
