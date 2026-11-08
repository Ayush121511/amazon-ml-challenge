#!/bin/bash
#PBS -N er_frself
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=8:ngpus=2:mem=200gb
#PBS -l walltime=03:30:00
#PBS -j oe
# France self-training on very-confident T1 test pairs: listwise (from our listwise model) and pairwise (from the
# French-noise model), France-only test re-scoring, and a US/India holdout no-regression check.
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
S=business_entity_resolution/src
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
VCAND=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/artifacts/submission_v5_top10/candidate_pairs.tsv
BASE=artifacts/cross_encoder_v1/mdeb_fr_aug/model OUT_DIR=artifacts/cross_encoder_v1/mdeb_fr_self PSEUDO=artifacts/fr_pseudo/pairs.npz \
  MAX_MIN=30 AUG_P=0.3 $PY -m torch.distributed.run --standalone --nproc_per_node=2 $S/train_fr_aug.py
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i $PY -u $S/score_ce_test2.py --candidates $VCAND --model artifacts/cross_encoder_v1/mdeb_fr_self/model \
    --name frself --out artifacts/test_ce_frself --shard $i/2 --country France > artifacts/test_ce_frself_$i.log 2>&1 &
done
MEMBERS="chunk2=/home/scai/mtech/aib262140/scratch/amazon_ml_2026/ftchunk2/ce_logit_sample.npy;fr_self=$HOME/scratch/amazon_ml_2026/artifacts/cross_encoder_v1/mdeb_fr_self/ce_logit_sample.npy" \
  $PY -u $S/multi_ce_eval.py || true
cp artifacts/fusion/multi_ce_eval.json artifacts/fusion/multi_ce_eval_frself.json || true
wait
chmod -R a+rX artifacts/cross_encoder_v1/mdeb_fr_self artifacts/test_ce_frself artifacts/fr_pseudo
ls -la artifacts/test_ce_frself
