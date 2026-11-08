#!/bin/bash
#PBS -N er_u6
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=1:ngpus=1:mem=64gb
#PBS -l walltime=00:30:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
A=$HOME/scratch/amazon_ml_2026/artifacts
O=$A/submission_U6_U5_noprune
{ U5_DIR=/scratch/scai/mtech/aib262467/final_u5/out U2_TSV=$A/submission_U2_final/matching_results.tsv \
  FR_TSV=$A/submission_U3_fallback/matching_results.tsv OUT_DIR=$O $PY -u swap_france.py
  sha256sum $O/*.tsv
  $PY $HOME/scratch/amazon_ml_2026/data/utils/validate_submission.py -m $O/matching_results.tsv -c $O/candidate_pairs.tsv -t $HOME/scratch/amazon_ml_2026/data/dataset/test --check-ids 2>&1 | tail -6
  cd $O && rm -f U6_U5_noprune.zip && zip -q -j U6_U5_noprune.zip matching_results.tsv candidate_pairs.tsv && sha256sum U6_U5_noprune.zip
  chmod -R a+rX $O; setfacl -R -m u:aib262140:rX,u:aib262467:rX $O; } > $A/u6_build.txt 2>&1
