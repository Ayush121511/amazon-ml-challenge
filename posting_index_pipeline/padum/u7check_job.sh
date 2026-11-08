#!/bin/bash
#PBS -N er_u7chk
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=1:ngpus=1:mem=64gb
#PBS -l walltime=00:30:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
O=$HOME/scratch/amazon_ml_2026/artifacts/u7_check.txt
{ A=$HOME/scratch/amazon_ml_2026/artifacts/sweep $PY -u alias_rate.py
  sha256sum /scratch/scai/mtech/aib262467/final_u7/*.tsv /scratch/scai/mtech/aib262467/final_u7/final_u7.zip
  U5=/scratch/scai/mtech/aib262467/final_u7 BASE=/scratch/scai/mtech/aib262467/final_u5/out $PY -u u5_check.py; } > $O 2>&1
