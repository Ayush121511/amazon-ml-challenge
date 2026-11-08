#!/bin/bash
#PBS -N er_sweept
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=1:mem=160gb
#PBS -l walltime=01:00:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
A=$HOME/scratch/amazon_ml_2026/artifacts/sweep
MODE=test NPROC=4 OUT=$A/test.json $PY -u error_sweep.py > $A/test.log 2>&1
until [ -f $A/train.json ]; do sleep 20; done
TRAIN=$A/train.json TEST=$A/test.json OUT=$A/report.md $PY -u error_sweep_report.py > $A/report.log 2>&1
