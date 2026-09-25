#!/bin/sh
#PBS -N amlc_eda
#PBS -P scai
#PBS -q scai_q
#PBS -m bea
#PBS -M {kerberos_id}@iitd.ac.in
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=01:00:00
# CPU-only EDA job; ngpus=1 requested only because scai_q enforces a
# minimum of 1 GPU per job, not because this workload uses one.

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="

cd $HOME/scratch/AmazonMLChallenge

source $HOME/scratch/miniconda3/bin/activate
conda activate amlc

python src/eda_report.py
