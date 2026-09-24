#!/bin/sh
#PBS -N amlc_train
#PBS -P scai
#PBS -q standard
#PBS -m bea
#PBS -M {kerberos_id}@iitd.ac.in
#PBS -l select=1:ncpus=1:ngpus=1:centos=icelake
#PBS -l walltime=50:00:00
# $PBS_O_WORKDIR is the directory from where the job is fired.

echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="

cd $HOME/scratch/AmazonMLChallenge

# internet for pip/dataset download if needed
./references/hpc/proxy.sh &
export http_proxy="http://proxy62.iitd.ac.in:3128"
export https_proxy="http://proxy62.iitd.ac.in:3128"

source $HOME/anaconda3/bin/activate
conda activate amlc

python src/train.py
