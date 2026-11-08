#!/bin/bash
#PBS -N er_llmfrA
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=2:mem=150gb
#PBS -l walltime=01:45:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
S=$HOME/scratch/amazon_ml_2026/artifacts/sweep
L=$HOME/scratch/amazon_ml_2026/artifacts/llm
export A=$S HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
T="$L/train_band.npz|train|$L/train_llm;$L/fr_pairs.npz|test|$L/fr_llm"
until grep -q "^downloaded" $L/download.log 2>/dev/null; do sleep 20; done
MODE=pairs_train PAIRS=$L/train_band.npz $PY -u llm_probe.py > $L/fr_run.log 2>&1
MODE=pairs_fr PAIRS=$L/fr_pairs.npz $PY -u llm_france.py >> $L/fr_run.log 2>&1
touch $L/pairs.ready
IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"
for i in 0 1; do CUDA_VISIBLE_DEVICES=${G[$i]} MODE=score SHARD=$i NSHARD=4 TASKS="$T" $PY -u llm_probe.py > $L/score_$i.log 2>&1 & done
wait
for k in $(seq 1 75); do [ -f $L/fr_llm_2.npy ] && [ -f $L/fr_llm_3.npy ] && break; sleep 20; done
if [ ! -f $L/fr_llm_2.npy ] || [ ! -f $L/fr_llm_3.npy ]; then
  for i in 2 3; do [ -f $L/fr_llm_$i.npy ] || { CUDA_VISIBLE_DEVICES=${G[$((i-2))]} MODE=score SHARD=$i NSHARD=4 TASKS="$T" $PY -u llm_probe.py > $L/score_$i.log 2>&1 & }; done
  wait
fi
MODE=eval PAIRS=$L/train_band.npz OUT=$L/train_llm $PY -u llm_probe.py >> $L/fr_run.log 2>&1
MODE=thresholds TRAIN_OUT=$L/train_llm THR=$L/thr.json $PY -u llm_france.py >> $L/fr_run.log 2>&1
O=$HOME/scratch/amazon_ml_2026/artifacts/submission_U7_llm_france
MODE=apply_fr PAIRS=$L/fr_pairs.npz OUT=$L/fr_llm THR=$L/thr.json OUT_DIR=$O $PY -u llm_france.py >> $L/fr_run.log 2>&1
{ sha256sum $O/*.tsv; $PY $HOME/scratch/amazon_ml_2026/data/utils/validate_submission.py -m $O/matching_results.tsv -c $O/candidate_pairs.tsv -t $HOME/scratch/amazon_ml_2026/data/dataset/test --check-ids 2>&1 | tail -4; } >> $L/fr_run.log 2>&1
echo ALLDONE >> $L/fr_run.log
