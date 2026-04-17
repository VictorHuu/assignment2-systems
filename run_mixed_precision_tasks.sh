#!/usr/bin/env bash
set -euo pipefail

# 1) Accumulation precision demo
PYTHONPATH=cs336-basics python -m cs336_basics.mixed_precision_tasks accumulation

# 2) Toy model dtype inspection under autocast(fp16)
PYTHONPATH=cs336-basics python -m cs336_basics.mixed_precision_tasks toy_dtypes --device cuda

# 3) Benchmark fp32 vs bf16 autocast for forward and forward_backward
PYTHONPATH=cs336-basics python -m cs336_basics.mixed_precision_tasks benchmark_compare \
  --device cuda \
  --mode forward \
  --warmup-steps 5 \
  --measure-steps 10 \
  --batch-size 8

PYTHONPATH=cs336-basics python -m cs336_basics.mixed_precision_tasks benchmark_compare \
  --device cuda \
  --mode forward_backward \
  --warmup-steps 5 \
  --measure-steps 10 \
  --batch-size 8
