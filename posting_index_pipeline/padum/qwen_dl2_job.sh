#!/bin/bash
#PBS -N er_qwendl2
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=1:mem=32gb
#PBS -l walltime=00:45:00
#PBS -j oe
L=$HOME/scratch/amazon_ml_2026/artifacts/llm
bash $HOME/scratch/amazon_ml_2026/business_entity_resolution/sp.sh > $L/proxy2.log 2>&1 &
PX=$!; sleep 12
export https_proxy=http://proxy62.iitd.ac.in:3128 http_proxy=http://proxy62.iitd.ac.in:3128
echo "std proxy: $(head -c 200 $L/proxy2.log | tr '\n' ' ') | $(curl -sI --max-time 20 https://huggingface.co | head -1)" >> $L/download.log
if grep -q "error in login" $L/proxy2.log; then kill $PX; exit 0; fi
/home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='/home/scai/mtech/aib262015/scratch/models/qwen2.5-7b-instruct',
                  allow_patterns=['*.json', '*.safetensors', '*.txt', 'LICENSE'], max_workers=4)
print('OK')" > $L/dl_try2.log 2>&1
grep -q "^OK" $L/dl_try2.log && echo "downloaded" >> $L/download.log || tail -c 300 $L/dl_try2.log >> $L/download.log
kill $PX
