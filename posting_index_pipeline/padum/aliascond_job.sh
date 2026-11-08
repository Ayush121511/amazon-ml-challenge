#!/bin/bash
#PBS -N er_acond
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=1:ngpus=1:mem=48gb
#PBS -l walltime=00:20:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
A=$HOME/scratch/amazon_ml_2026/artifacts/sweep /home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python -u alias_cond.py > $HOME/scratch/amazon_ml_2026/artifacts/sweep/alias_cond.txt 2>&1
