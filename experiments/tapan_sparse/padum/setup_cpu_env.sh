#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
bash padum/check_compute.sh
# Requires a working Python 3.10+ with venv. Do not install another system Python.
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Load Python 3.10+ or activate an existing suitable Conda environment first"'
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r padum/requirements-cpu.txt
mkdir -p artifacts
python -m pip freeze > artifacts/requirements-cpu.lock.txt
python -c 'import numpy, scipy, sklearn, rapidfuzz, lightgbm, pyarrow; print("CPU baseline imports OK")'
python -m unittest discover -s business_entity_resolution/tests -v
