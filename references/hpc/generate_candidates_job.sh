#!/bin/sh
#PBS -N amlc_generate_candidates
#PBS -P scai
#PBS -q scai_q
#PBS -m bea
#PBS -M {kerberos_id}@iitd.ac.in
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=02:00:00
# CPU-only workload (exact-match blocking + TF-IDF top-N); ngpus=1 requested
# only because scai_q enforces a minimum of 1 GPU per job, not because we use
# one. Assumes src/run_stage1.py (preprocessing) has already produced
# data/processed/{train,test}_source{1,2,3}.parquet.

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="

cd $HOME/scratch/AmazonMLChallenge

source $HOME/scratch/miniconda3/bin/activate
conda activate amlc

python src/generate_candidates.py --split train --eval
python src/generate_candidates.py --split test
