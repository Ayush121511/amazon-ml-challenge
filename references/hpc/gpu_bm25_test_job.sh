#!/bin/sh
#PBS -N amlc_gpu_bm25_test
#PBS -P scai
#PBS -q scai_q
#PBS -m bea
#PBS -M {kerberos_id}@iitd.ac.in
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=00:45:00
# Small correctness + speed test: does reconstructing BM25 scoring as a
# sparse matmul on the A100 (via CuPy/cuSPARSE) reproduce bm25s's own CPU
# scores exactly, and is it actually faster at real corpus scale.

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="

cd $HOME/scratch/AmazonMLChallenge

source $HOME/scratch/miniconda3/bin/activate
conda activate amlc

python src/gpu_bm25_matmul_test.py India 2000
