#!/bin/bash
#PBS -N er_loraS
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=2:mem=100gb
#PBS -l walltime=01:30:00
#PBS -j oe
# Usage (any team account): qsub -v SHARDS="2 3",NSHARD=6,OUTD=$HOME/scratch/lora_out <this file>
SRC=/home/scai/mtech/aib262015/scratch/amazon_ml_2026/business_entity_resolution/src
PY=${PYBIN:-/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python}
L=/home/scai/mtech/aib262015/scratch/amazon_ml_2026/artifacts/lora
export PYTHONPATH=/home/scai/mtech/aib262015/scratch/pylib:$SRC HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p $OUTD
until [ -f $L/adapter/DONE ] && [ -f $L/PAIRS_READY ]; do sleep 20; done
IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"; k=0
for i in $SHARDS; do
  CUDA_VISIBLE_DEVICES=${G[$k]} MODE=score SHARD=$i NSHARD=$NSHARD PAIRS=$L/test_usin_band.npz ADAPTER=$L/adapter OUT=$OUTD/test_usin SPLIT=test \
    $PY -u $SRC/lora_judge.py > $OUTD/score_$i.log 2>&1 &
  k=$((k+1))
done
wait
ls $OUTD/test_usin_*.npy >/dev/null 2>&1 || { echo 'NO OUTPUT - see score logs'; exit 1; }
setfacl -m u:aib262015:x $HOME $HOME/scratch 2>/dev/null; setfacl -R -m u:aib262015:rX $OUTD
echo DONE > $OUTD/SCORE_DONE_${SHARDS// /_}
