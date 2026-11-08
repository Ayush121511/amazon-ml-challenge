#!/bin/bash
#PBS -N er_loraT
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=4:ngpus=1:mem=120gb
#PBS -l walltime=01:15:00
#PBS -j oe
cd /home/scai/mtech/aib262015/scratch/amazon_ml_2026/business_entity_resolution/src
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
L=/home/scai/mtech/aib262015/scratch/amazon_ml_2026/artifacts/lora
export PYTHONPATH=/home/scai/mtech/aib262015/scratch/pylib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 A=/home/scai/mtech/aib262015/scratch/amazon_ml_2026/artifacts/sweep
MODE=pairs_test PAIRS=$L/test_usin_band.npz $PY -u llm_apply.py > $L/pairs_test.log 2>&1
touch $L/PAIRS_READY
MODE=train ADAPTER=$L/adapter DEV_PAIRS=/home/scai/mtech/aib262015/scratch/amazon_ml_2026/artifacts/llm/train_band.npz OUT=$L/devband MAX_SEC=1500 $PY -u lora_judge.py > $L/train.log 2>&1
MODE=eval PAIRS=/home/scai/mtech/aib262015/scratch/amazon_ml_2026/artifacts/llm/train_band.npz OUT=$L/devband $PY -u llm_probe.py > $L/devband_eval.log 2>&1
setfacl -R -m u:aib262140:rX,u:aib262467:rX $L
