#!/bin/bash
#PBS -N er_qwendl
#PBS -P scai
#PBS -q scai_q
#PBS -l select=1:ncpus=2:ngpus=1:mem=32gb
#PBS -l walltime=01:00:00
#PBS -j oe
L=$HOME/scratch/amazon_ml_2026/artifacts/llm; mkdir -p $L
RP=$HOME/scratch/amazon_ml_2026/business_entity_resolution/rp.sh
export https_proxy=http://10.10.88.6:3128 http_proxy=http://10.10.88.6:3128
: > $L/download.log
for a in 1 2 3 4 5 6 7 8; do
  bash $RP > $L/proxy.log 2>&1 &
  PX=$!; sleep 12
  echo "attempt $a: $(curl -sI --max-time 20 https://huggingface.co | head -1)" >> $L/download.log
  /home/scai/mtech/aib262467/scratch/amazon_ml_2026/.conda-er/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='/home/scai/mtech/aib262015/scratch/models/qwen2.5-7b-instruct',
                  allow_patterns=['*.json', '*.safetensors', '*.txt', 'LICENSE'], max_workers=4)
print('OK')" > $L/dl_try.log 2>&1
  ok=$(grep -c "^OK" $L/dl_try.log)
  kill $PX 2>/dev/null; sleep 3
  echo "attempt $a done ok=$ok size=$(du -sh /home/scai/mtech/aib262015/scratch/models/qwen2.5-7b-instruct | cut -f1)" >> $L/download.log
  if [ "$ok" = "1" ]; then echo "downloaded" >> $L/download.log; break; fi
  tail -c 300 $L/dl_try.log >> $L/download.log
  sleep 10
done
