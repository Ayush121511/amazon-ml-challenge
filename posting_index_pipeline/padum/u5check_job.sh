#!/bin/bash
#PBS -N er_u5chk
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=1:ngpus=1:mem=64gb
#PBS -l walltime=00:30:00
#PBS -j oe
set -u
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
F=/scratch/scai/mtech/aib262467/final_u5
W=$HOME/scratch/amazon_ml_2026/artifacts/u5_check
O=$W/report.txt
rm -rf $W && mkdir -p $W/zip && cd $W/zip && unzip -q $F/final_u5.zip && cd -
{ echo "== sha256"; sha256sum $F/final_u5.zip $F/out/*.tsv; find $W/zip -type f -exec sha256sum {} \;
  M=$(find $W/zip -name matching_results.tsv); C=$(find $W/zip -name candidate_pairs.tsv)
  echo "== validator (zip contents)"; $PY $HOME/scratch/amazon_ml_2026/data/utils/validate_submission.py -m $M -c $C -t $HOME/scratch/amazon_ml_2026/data/dataset/test --check-ids 2>&1 | tail -15
  echo "== audit"; U5=$(dirname $M) $PY -u u5_check.py 2>&1; } > $O
