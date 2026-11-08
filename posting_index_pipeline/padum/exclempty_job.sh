#!/bin/bash
#PBS -N er_exclE
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=1:ngpus=1:mem=96gb
#PBS -l walltime=00:20:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
A=$HOME/scratch/amazon_ml_2026/artifacts/sweep /home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python -u excl_empty.py > $HOME/scratch/amazon_ml_2026/artifacts/sweep/excl_empty.txt 2>&1
