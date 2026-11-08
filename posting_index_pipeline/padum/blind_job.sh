#!/bin/bash
#PBS -N er_blind
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=1:ngpus=1:mem=32gb
#PBS -l walltime=00:20:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
S=$HOME/scratch/amazon_ml_2026/artifacts/sweep
A=$S OUT_Q=$S/blind_questions.txt OUT_K=$S/blind_key.tsv /home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python -u blind_sample.py > $S/blind.log 2>&1
