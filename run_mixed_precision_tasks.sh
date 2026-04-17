#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH="${PYTHONPATH:-cs336-basics}"
DEVICE="${DEVICE:-cuda}"
BENCH_MODE="${BENCH_MODE:-training}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASURE_STEPS="${MEASURE_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# 1) Accumulation precision demo
PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" -m cs336_basics.mixed_precision_tasks accumulation

# 2) Toy model dtype inspection under autocast(fp16)
PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" -m cs336_basics.mixed_precision_tasks toy_dtypes --device "${DEVICE}"

# 3) Benchmark fp32 vs bf16 autocast for one mode (override BENCH_MODE if needed).
PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" -m cs336_basics.mixed_precision_tasks benchmark_compare \
  --device "${DEVICE}" \
  --mode "${BENCH_MODE}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --measure-steps "${MEASURE_STEPS}" \
  --batch-size "${BATCH_SIZE}"
