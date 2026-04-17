#!/usr/bin/env bash
set -euo pipefail

# Hardcoded example run for Nsight Systems profiling.
PYTHONPATH=cs336-basics python -m cs336_basics.nsys_profile \
  --output-dir nsys_reports \
  --device cuda \
  --precision fp32 \
  --contexts 128,256,512,1024 \
  --modes forward,forward_backward,training_step \
  --warmup-steps 5 \
  --measure-steps 10 \
  --batch-size 8 \
  --d-model 256 \
  --num-layers 4 \
  --num-heads 8 \
  --d-ff 1024 \
  --python-backtrace-cuda \
  --run-stats

echo "Summarizing nsys reports to nsys_reports/summary.csv ..."
PYTHONPATH=cs336-basics python -m cs336_basics.nsys_summarize \
  --report-dir nsys_reports \
  --glob "*.nsys-rep" \
  --output-csv nsys_reports/summary.csv
