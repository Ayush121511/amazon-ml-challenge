#!/bin/bash
#PBS -N er_u8
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=1:ngpus=1:mem=64gb
#PBS -l walltime=00:30:00
#PBS -j oe
cd $HOME/scratch/amazon_ml_2026/business_entity_resolution/src
PY=/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python
A=$HOME/scratch/amazon_ml_2026/artifacts; O=$A/submission_U8_alias_fr
{ ADDS=$A/alias_adds.tsv OUT_DIR=$O T6_T=0.68 $PY -u u8_build.py
  sha256sum $O/*.tsv
  $PY $HOME/scratch/amazon_ml_2026/data/utils/validate_submission.py -m $O/matching_results.tsv -c $O/candidate_pairs.tsv -t $HOME/scratch/amazon_ml_2026/data/dataset/test --check-ids 2>&1 | tail -3
  cd $O && zip -q -j U8_alias_fr.zip matching_results.tsv candidate_pairs.tsv && sha256sum U8_alias_fr.zip; } > $A/u8_build.txt 2>&1
