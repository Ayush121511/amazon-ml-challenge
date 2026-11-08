#!/bin/bash
#PBS -N er_native
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=1:ngpus=1:mem=48gb
#PBS -l walltime=00:15:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
A=$HOME/scratch/amazon_ml_2026/artifacts/sweep /home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python -u native_loss.py > $HOME/scratch/amazon_ml_2026/artifacts/sweep/native_loss.txt 2>&1
