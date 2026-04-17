#!/usr/bin/env bash
set -euo pipefail

# Defaults to one mode for convenience; override MODES if needed.
# Example:
#   MODES=training CONTEXTS=128,256,512 ./run_nsys_profiles.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH="${PYTHONPATH:-cs336-basics}"
OUTPUT_DIR="${OUTPUT_DIR:-nsys_reports}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-fp32}"
CONTEXTS="${CONTEXTS:-128,256,512,1024}"
MODES="${MODES:-training}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASURE_STEPS="${MEASURE_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
D_MODEL="${D_MODEL:-256}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-8}"
D_FF="${D_FF:-1024}"

PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" -m cs336_basics.nsys_profile \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --precision "${PRECISION}" \
  --contexts "${CONTEXTS}" \
  --modes "${MODES}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --measure-steps "${MEASURE_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --d-model "${D_MODEL}" \
  --num-layers "${NUM_LAYERS}" \
  --num-heads "${NUM_HEADS}" \
  --d-ff "${D_FF}" \
  --python-backtrace-cuda \
  --run-stats

echo "Summarizing nsys reports to ${OUTPUT_DIR}/summary.csv ..."
PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" -m cs336_basics.nsys_summarize \
  --report-dir "${OUTPUT_DIR}" \
  --glob "*.nsys-rep" \
  --output-csv "${OUTPUT_DIR}/summary.csv"
