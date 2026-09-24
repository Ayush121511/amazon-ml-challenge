# IIT Delhi HPC Setup — Amazon ML Challenge 2026

Kerberos ID: `{kerberos_id}`
Faculty supervisor: `raunakbh`
Program: mtech (PROGRAM_CODE 62)

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
2. `ssh {kerberos_id}@login1.hpc.iitd.ernet.in`
   (note: `.ernet.in` domain, not `.ac.in` — the latter doesn't resolve)
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

## 4. Conda setup
```
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
sh Anaconda3-2024.06-1-Linux-x86_64.sh
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
qsub -I -P {PROJECT_NAME} -q standard -N amlc_dev -lselect=1:ncpus=1:ngpus=1:centos=icelake -lwalltime=4:00:00
```
- NODE: icelake / skylake / haswell (default haswell)
- QUEUE: standard (default) / high / scai_q
- Max walltime 168h, max 10 concurrent jobs, max 2 on scai_q

## 6. Batch job (for actual training runs)
Edit `batchjob.sh`: set `{PROJECT_NAME}`, walltime, script path.
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
amgr checkbalance project -n {PROJECT_NAME}
```

## Reference
Based on https://github.com/mahesh-keswani/HPC-Details
