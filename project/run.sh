#!/usr/bin/env bash
# End-to-end эксперимент одной командой: ./run.sh [--no-cv] [--set key=value ...]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
for f in train.json ytrain.csv test.json ytest.csv; do
  [ -f "data/raw/$f" ] || { echo "Нет data/raw/$f — скачайте данные соревнования в data/raw/"; exit 1; }
done
PYTHONPATH=src .venv/bin/python -m youarebot.pipeline --config configs/config.yaml "$@"
