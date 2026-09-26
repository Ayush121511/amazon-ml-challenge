#!/bin/bash
# Lightweight pre-submit checks; no model training or feature reads on login node.
set -euo pipefail
ROOT="$HOME/scratch/amazon_ml_2026"
SHARED="${V4_ROOT:-/home/scai/mtech/aib262467/scratch/amazon_ml_2026}"
PY="${V5_PYTHON:-$ROOT/.conda-er/bin/python}"
FAILED=0
for FILE in x.npy y.npy pairs.tsv.gz gold.json countries.json report.json; do
  PATH_TO_CHECK="$SHARED/artifacts/matcher_v4/pairs/$FILE"
  if test -r "$PATH_TO_CHECK"; then echo "READABLE $FILE";
  else echo "MISSING_OR_NOT_READABLE $PATH_TO_CHECK"; FAILED=1; fi
done
if test -x "$PY"; then
  "$PY" -c "import importlib.util; missing=[n for n in ('numpy','scipy','sparse_dot_topn','rapidfuzz','xgboost','torch') if importlib.util.find_spec(n) is None]; print('Missing packages:', missing); raise SystemExit(bool(missing))" || FAILED=1
else
  echo "Python is unavailable: $PY"; FAILED=1
fi
exit "$FAILED"
