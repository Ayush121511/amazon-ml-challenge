#!/usr/bin/env bash
set -eu
echo "Host: $(hostname)"
echo "PBS job: ${PBS_JOBID:-none}"
case "$(hostname)" in
  login*) echo 'This is a login node. Request a PBS compute job first.'; exit 1 ;;
esac
if [ -z "${PBS_JOBID:-}" ]; then
  echo 'No PBS allocation detected. Stop here and request a compute job.'
  exit 1
fi
echo 'Allocated resources:'
qstat -f "$PBS_JOBID" | grep -E 'Resource_List|exec_host|exec_vnode' || true
echo 'GPU:'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi
echo 'Python:'
command -v python3 || true
python3 --version || true
echo 'Conda:'
command -v conda || true
echo 'Project storage:'
df -h "$HOME/scratch/amazon_ml_2026"
echo 'Available memory (node total; allocation is shown above):'
free -h
