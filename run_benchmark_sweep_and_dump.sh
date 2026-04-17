#!/usr/bin/env bash
set -euo pipefail

# Convenience benchmark sweep.
# Defaults to a single mode run (training), but can be overridden.
# Example:
#   MODES="training" CONTEXTS="128 256 512" WARMUPS="0 1 2 3 4 5" ./run_benchmark_sweep_and_dump.sh

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PYTHONPATH="${PYTHONPATH:-cs336-basics}"
DEVICE="${DEVICE:-cuda}"
MODES="${MODES:-training}"
CONTEXTS="${CONTEXTS:-128}"
WARMUPS="${WARMUPS:-0 1 2 3 4 5}"
MEASURE_STEPS="${MEASURE_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VOCAB_SIZE="${VOCAB_SIZE:-50257}"
D_MODEL="${D_MODEL:-256}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-8}"
D_FF="${D_FF:-1024}"
ROPE_THETA="${ROPE_THETA:-10000.0}"

for mode in ${MODES}; do
  for ctx in ${CONTEXTS}; do
    for warmup in ${WARMUPS}; do
      echo "=== mode=$mode ctx=$ctx warmup=$warmup ==="
      PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" -m cs336_basics.benchmarking_script \
        --device "${DEVICE}" \
        --precision fp32 \
        --mode "$mode" \
        --warmup-steps "$warmup" \
        --measure-steps "${MEASURE_STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --context-length "$ctx" \
        --vocab-size "${VOCAB_SIZE}" \
        --d-model "${D_MODEL}" \
        --num-layers "${NUM_LAYERS}" \
        --num-heads "${NUM_HEADS}" \
        --d-ff "${D_FF}" \
        --rope-theta "${ROPE_THETA}"
    done
  done
done
