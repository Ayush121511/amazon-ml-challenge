#!/bin/bash
# Run on the Padum login node. This only submits PBS jobs; it does no ML work.
set -euo pipefail
cd "$HOME/scratch/amazon_ml_2026"
matcher_job="${1:-}"
if test -f artifacts/matcher_sparse_10k/report.json && test -f artifacts/matcher_sparse_10k/model.txt; then
  matcher_job=''
elif [[ ! "$matcher_job" =~ ^[0-9]+([.][A-Za-z0-9_-]+)?$ ]]; then
  echo 'Pass the running matcher job ID, for example: bash padum/submit_test_prototype.sh 1065613.pbshpc'
  exit 1
else
  qstat "$matcher_job" >/dev/null
fi
log_path=padum/test_prototype_jobs.tsv
if test -e "$log_path"; then
  echo "A submission record already exists at $log_path. Check its job IDs before submitting again."
  exit 1
fi
record_job() {
  printf '%s\t%s\n' "$1" "$2" | tee -a "$log_path"
}
cache_job=$(qsub padum/test_cache.pbs)
record_job cache "$cache_job"
dependency="$cache_job"
if test -n "$matcher_job"; then
  dependency="$dependency:$matcher_job"
fi
inference_job=$(qsub -W "depend=afterok:$dependency" padum/test_all_countries.pbs)
record_job all_countries_and_finalize "$inference_job"
echo 'Submitted cache plus one continuation job for all countries and final validation.'
echo 'Countries run sequentially within that allocation. Final portal upload remains a separate action.'
