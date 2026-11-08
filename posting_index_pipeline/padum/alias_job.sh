#!/bin/bash
#PBS -N er_alias
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=1:mem=96gb
#PBS -l walltime=00:30:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
S=$HOME/scratch/amazon_ml_2026/artifacts/sweep
MODE=train A=$S $PY -u alias_link.py > $S/alias_train.txt 2>&1 &
MODE=test A=$S U5_DIR=/scratch/scai/mtech/aib262467/final_u5/out $PY -u alias_link.py > $S/alias_test.txt 2>&1 &
wait
