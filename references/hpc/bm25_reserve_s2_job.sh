#!/bin/sh
#PBS -N amlc_bm25_s2
#PBS -P scai
#PBS -q scai_q
#PBS -m bea
#PBS -M {kerberos_id}@iitd.ac.in
#PBS -l select=1:ncpus=8:ngpus=1:mem=128gb
#PBS -l walltime=24:00:00
# CPU-only workload; ngpus=1 requested only because scai_q enforces a minimum
# of 1 GPU per job. ncpus=8 (not 32) chosen deliberately: smaller requests fit
# into partial free-core gaps that open up as other users' jobs finish, so
# this is more likely to actually start than a 32-cpu request that needs a
# full free node. Once running, holds the slot for up to 24h (queue max) and
# chains straight from validation into the full S1 sweep so the reservation
# isn't wasted on a short test.

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="

cd $HOME/scratch/AmazonMLChallenge

source $HOME/scratch/miniconda3/bin/activate
conda activate amlc

echo "--- building/caching BM25 indexes for S2 (skips if already cached) ---"
python src/hpc_build_indexes.py s2

echo "--- intermediate validation: 20000 S1 refs vs full S2 pool, 8-way parallel ---"
python src/hpc_query_shards.py s2 20000 75 8

echo "--- full S1 sweep vs full S2 pool, 8-way parallel ---"
python src/hpc_query_shards.py s2 all 75 8
