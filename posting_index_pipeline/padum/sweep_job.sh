#!/bin/bash
#PBS -N er_sweep
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=4:ngpus=1:mem=200gb
#PBS -l walltime=01:30:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
A=$HOME/scratch/amazon_ml_2026/artifacts/sweep
mkdir -p $A
MODE=train NPROC=4 OUT=$A/train.json $PY -u error_sweep.py > $A/train.log 2>&1 &
MODE=test NPROC=4 OUT=$A/test.json $PY -u error_sweep.py > $A/test.log 2>&1 &
wait
TRAIN=$A/train.json TEST=$A/test.json OUT=$A/report.md $PY -u error_sweep_report.py > $A/report.log 2>&1
