#!/bin/sh
#PBS -N amlc_bm25_interm
#PBS -P scai
#PBS -q scai_q
#PBS -m bea
#PBS -M {kerberos_id}@iitd.ac.in
#PBS -l select=1:ncpus=32:ngpus=1:mem=512gb
#PBS -l walltime=04:00:00
# CPU-only workload (BM25 index build + sharded query); ngpus=1 requested only
# because scai_q enforces a minimum of 1 GPU per job, not because we use one.
# Full node requested (32 cpu, 512gb) so index-build-once + N_WORKERS=32
# sharded query stage both get real parallelism headroom -- this failed on a
# 16GB local machine (OOM from process-per-channel duplicating the full
# target-corpus index); a full scai node has 500GB-1TB, safe for this.

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="

cd $HOME/scratch/AmazonMLChallenge

source $HOME/scratch/miniconda3/bin/activate
conda activate amlc

echo "--- building/caching BM25 indexes (skips if already cached) ---"
python src/hpc_build_indexes.py s2

echo "--- sharded query: 20000 S1 refs vs full S2 pool, 32-way parallel ---"
python src/hpc_query_shards.py s2 20000 75 32
