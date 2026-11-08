#!/bin/bash
#PBS -N er_rows
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=1:ngpus=1:mem=160gb
#PBS -l walltime=01:00:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
A=$HOME/scratch/amazon_ml_2026/artifacts/sweep U5_DIR=/scratch/scai/mtech/aib262467/final_u5/out /home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python -u rowsweep.py > $HOME/scratch/amazon_ml_2026/artifacts/sweep/rowsweep.txt 2>&1
