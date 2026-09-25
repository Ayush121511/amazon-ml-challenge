#!/bin/sh
#PBS -N amlc_stage1
#PBS -P scai
#PBS -q scai_q
#PBS -m bea
#PBS -M {kerberos_id}@iitd.ac.in
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=02:00:00
# $PBS_O_WORKDIR is the directory from where the job is fired.
# CPU-only workload (preprocessing + blocking); ngpus=1 requested only
# because scai_q enforces a minimum of 1 GPU per job, not because we use one.

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="

cd $HOME/scratch/AmazonMLChallenge

source $HOME/scratch/miniconda3/bin/activate
conda activate amlc

python src/run_stage1.py
