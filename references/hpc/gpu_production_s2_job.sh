#!/bin/sh
#PBS -N amlc_gpu_prod_s2
#PBS -P scai
#PBS -q scai_q
#PBS -m bea
#PBS -M {kerberos_id}@iitd.ac.in
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
# Production GPU BM25 name+address RRF fusion query, full S1 sweep vs full S2.
# Validated (job 1066121): GPU sparse x dense matmul reproduces bm25s CPU
# scores exactly (0/200 mismatches), 8.18x faster per channel. Estimated
# ~10-11h for the full 2.2M-query sweep on one A100 -- comfortably inside
# this job's 24h walltime, checkpointed per-country so a resubmit resumes
# cleanly if it doesn't.

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="

cd $HOME/scratch/AmazonMLChallenge

source $HOME/scratch/miniconda3/bin/activate
conda activate amlc

python src/gpu_query_production.py s2 all 75 200
