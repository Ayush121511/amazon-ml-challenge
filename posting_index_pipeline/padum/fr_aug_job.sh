#!/bin/bash
#PBS -N er_fraug
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=8:ngpus=2:mem=200gb
#PBS -l walltime=03:00:00
#PBS -j oe
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
OUT=artifacts/cross_encoder_v1/mdeb_fr_aug
MAX_MIN=${MAX_MIN:-60} OUT_DIR=$OUT $PY -m torch.distributed.run --nproc_per_node=2 business_entity_resolution/src/train_fr_aug.py
CAND=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/submission_v5_top10/candidate_pairs.tsv
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i $PY -u business_entity_resolution/src/score_ce_test2.py --candidates $CAND --model $OUT/model \
    --name fraug --out artifacts/test_ce_fraug --shard $i/2 --country France > artifacts/test_ce_fraug_$i.log 2>&1 &
done
MEMBERS="chunk2=/home/scai/mtech/aib262140/scratch/amazon_ml_2026/ftchunk2/ce_logit_sample.npy;fraug=$HOME/scratch/amazon_ml_2026/$OUT/ce_logit_sample.npy" \
  $PY -u business_entity_resolution/src/multi_ce_eval.py || true
wait
ls -la artifacts/test_ce_fraug/
