#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="submitit_logs/benchmark_sweep"
EXPECTED_NEW_RESULTS=4
POLL_SECONDS=5
TIMEOUT_SECONDS=3600

mkdir -p "$LOG_DIR"

before_count=$(find "$LOG_DIR" -maxdepth 1 -type f -name '*_result.pkl' | wc -l)

for mode in forward forward_backward training_step; do
  for ctx in 128; do
    for warmup in 0 1 2 3 4 5; do
      echo "=== mode=$mode ctx=$ctx warmup=$warmup ==="
      PYTHONPATH=cs336-basics .venv/bin/python -m cs336_basics.benchmarking_script \
        --device cuda \
        --precision fp32 \
        --mode "$mode" \
        --warmup-steps "$warmup" \
        --measure-steps 10 \
        --batch-size 8 \
        --context-length "$ctx" \
        --vocab-size 50257 \
        --d-model 256 \
        --num-layers 4 \
        --num-heads 8 \
        --d-ff 1024 \
        --rope-theta 10000.0
    done
  done
done

echo "Waiting for ${EXPECTED_NEW_RESULTS} new result pkl files in ${LOG_DIR} ..."
start_ts=$(date +%s)
while true; do
  current_count=$(find "$LOG_DIR" -maxdepth 1 -type f -name '*_result.pkl' | wc -l)
  new_count=$((current_count - before_count))
  if (( new_count >= EXPECTED_NEW_RESULTS )); then
    echo "Detected ${new_count} new result files."
    break
  fi

  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))
  if (( elapsed > TIMEOUT_SECONDS )); then
    echo "Timed out after ${TIMEOUT_SECONDS}s while waiting for result pkl files."
    exit 1
  fi

  sleep "$POLL_SECONDS"
done

python - <<'PY'
import glob
import pickle

for p in sorted(glob.glob("submitit_logs/benchmark_sweep/*_result.pkl")):
    with open(p, "rb") as f:
        obj = pickle.load(f)
    print(f"\n=== {p} ===")
    print(type(obj))
    print(obj)
PY
