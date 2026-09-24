# IIT Delhi HPC Setup — Amazon ML Challenge 2026

Kerberos ID: `{kerberos_id}`
Faculty supervisor: `raunakbh`
Program: mtech (PROGRAM_CODE 62)
Project name: `scai`
GPU queue: `scai_q` — NVIDIA A100 80GB PCIe, CUDA 13.2, driver 595.71.05, max 2 concurrent jobs/user

## 1. Account creation
1. Apply: https://userm.iitd.ernet.in/usermanage/hpc.html
2. Login with Kerberos ID/password
3. Faculty supervisor (uid): `raunakbh`
4. Expiry date: 15 December
5. On first login, run once:
   ```
   source /home/apps/skeleton/oneTimeHPCAccountEnvSetup.sh
   ```

## 2. Connect via VSCode
1. Bottom-left green Remote button → "Connect to Host" → "Add New SSH Host"
2. `ssh -X {kerberos_id}@hpc.iitd.ac.in`
   (official CPU login host per supercomputing.iitd.ac.in; `-X` enables X11 forwarding)
   - GPU nodes directly: `gpu.hpc.iitd.ac.in`
   - Mic/Xeon Phi nodes: `mic.hpc.iitd.ac.in`
   - `login1.hpc.iitd.ernet.in` also resolves but is likely an older/internal alias — prefer `hpc.iitd.ac.in`
3. Enter Kerberos password when prompted, then connect.

You land on a **login node** — never run compute here (30 warnings = account ban).

## 3. Internet on login/interactive node
1. Copy `proxy.sh` to HPC (VSCode drag-drop or `scp`)
2. Edit password field in the HPC copy only — do not commit real password
3. `chmod +x proxy.sh && ./proxy.sh &`
4. Export:
   ```
   export http_proxy="http://proxy62.iitd.ac.in:3128"
   export https_proxy="http://proxy62.iitd.ac.in:3128"
   ```
5. Verify: `wget google.com`

## 4. Conda setup (Miniconda, installed to scratch not home)
```
mkdir -p ~/scratch/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -u -p ~/scratch/miniconda3
rm ~/miniconda.sh
~/scratch/miniconda3/bin/conda init bash
source ~/.bashrc
```
Accept Anaconda channel ToS once (needed before first env create):
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```
Then create challenge env:
```
conda create -n amlc python=3.11 -y
conda activate amlc
pip install -r requirements.txt
```

## 5. Interactive GPU job (for dev/debug)
```
tmux new -s amlc      # survives dropped connections
qsub -I -P scai -q scai_q -lselect=1:ncpus=1:ngpus=1 -lwalltime=4:00:00
```
- `scai_q` gives NVIDIA A100 80GB — use this queue for the challenge
- Max walltime 168h, max 10 concurrent jobs total, max 2 on `scai_q`
- Verify GPU once job starts: `nvidia-smi`

## 6. Batch job (for actual training runs)
Edit `batchjob.sh`: adjust walltime, script path as needed (project already set to `scai`).
```
qsub references/hpc/batchjob.sh
qstat -T -u {kerberos_id}          # check status
qstat -ans {job_id}            # details
qdel {job_id}                  # cancel
```

## 7. Research proxy (fast dataset download, inside a job only)
```
./references/hpc/research_proxy.sh &
export https_proxy=http://10.10.88.6:3128
export http_proxy=http://10.10.88.6:3128
```
Only one session at a time cluster-wide. Compute speed unaffected, only download speed.

## 8. Storage
- Home (`$HOME`): 100GB
- Scratch (`$HOME/scratch`): 2TB — put dataset + checkpoints here, not home

## 9. Check balance
```
amgr login
amgr ls project
amgr checkbalance project -n scai
```

## Reference
Based on https://github.com/mahesh-keswani/HPC-Details
