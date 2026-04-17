#!/usr/bin/env bash
set -euo pipefail

# Generate PyTorch CUDA memory snapshots for the 2.7B configuration:
# d_model=2560, d_ff=10240, num_layers=32, num_heads=32.
#
# Produces snapshots for:
#   - inference (forward-only)
#   - training (forward + backward + optimizer.step)
# across context lengths 128/256/512, in both:
#   - fp32
#   - bf16 autocast (params still fp32)
#
# Usage:
#   ./run_memory_profiles_2_7b.sh
#
# Optional environment overrides:
#   PYTHON_BIN=python
#   BATCH_SIZE=8
#   WARMUP_STEPS=3
#   MEASURE_STEPS=1
#   OUT_DIR=memory_profiles_2p7b
#   CONTEXTS="128 256 512"
#   MAX_ENTRIES=1000000
#   ENABLE_NVTX=1
#   DRY_RUN=1

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH="${PYTHONPATH:-cs336-basics}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
MEASURE_STEPS="${MEASURE_STEPS:-1}"
OUT_DIR="${OUT_DIR:-memory_profiles_2p7b}"
CONTEXTS="${CONTEXTS:-128 256 512}"
MAX_ENTRIES="${MAX_ENTRIES:-1000000}"
DRY_RUN="${DRY_RUN:-0}"
ENABLE_NVTX="${ENABLE_NVTX:-1}"

DMODEL=2560
DFF=10240
NLAYERS=32
NHEADS=32
VOCAB=50257

mkdir -p "${OUT_DIR}"
SUMMARY_CSV="${OUT_DIR}/peak_memory_summary.csv"
echo "precision,mode,context_length,peak_allocated_mib,peak_reserved_mib,snapshot_path,log_path" > "${SUMMARY_CSV}"

run_one() {
  local precision_name="$1"   # fp32 | bf16
  local mode="$2"             # inference | training
  local ctx="$3"

  local autocast_dtype="none"
  if [[ "${precision_name}" == "bf16" ]]; then
    autocast_dtype="bf16"
  fi

  local base="${precision_name}_${mode}_ctx${ctx}"
  local snapshot_path="${OUT_DIR}/${base}.pickle"
  local log_path="${OUT_DIR}/${base}.log"

  local -a cmd=(
    "${PYTHON_BIN}" -m cs336_basics.benchmarking_script
    --device cuda
    --mode "${mode}"
    --precision fp32
    --autocast-dtype "${autocast_dtype}"
    --memory-profile
    --memory-snapshot-path "${snapshot_path}"
    --memory-max-entries "${MAX_ENTRIES}"
    --warmup-steps "${WARMUP_STEPS}"
    --measure-steps "${MEASURE_STEPS}"
    --batch-size "${BATCH_SIZE}"
    --context-length "${ctx}"
    --vocab-size "${VOCAB}"
    --d-model "${DMODEL}"
    --num-layers "${NLAYERS}"
    --num-heads "${NHEADS}"
    --d-ff "${DFF}"
  )
  if [[ "${ENABLE_NVTX}" == "1" ]]; then
    cmd+=(--enable-nvtx)
  fi

  echo "=== ${base} ==="
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN: PYTHONPATH=%q ' "${PYTHONPATH}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  PYTHONPATH="${PYTHONPATH}" "${cmd[@]}" | tee "${log_path}"

  local peak_line
  peak_line="$(grep "peak_allocated=" "${log_path}" | tail -n 1 || true)"
  if [[ -z "${peak_line}" ]]; then
    echo "ERROR: could not parse peak memory line from ${log_path}" >&2
    exit 1
  fi

  local peak_alloc peak_reserved
  peak_alloc="$(sed -n 's/.*peak_allocated=\([0-9.]\+\) MiB.*/\1/p' <<< "${peak_line}")"
  peak_reserved="$(sed -n 's/.*peak_reserved=\([0-9.]\+\) MiB.*/\1/p' <<< "${peak_line}")"

  echo "${precision_name},${mode},${ctx},${peak_alloc},${peak_reserved},${snapshot_path},${log_path}" >> "${SUMMARY_CSV}"
}

for precision in fp32 bf16; do
  for mode in inference training; do
    for ctx in ${CONTEXTS}; do
      run_one "${precision}" "${mode}" "${ctx}"
    done
  done
done

echo
echo "Done. Summary CSV: ${SUMMARY_CSV}"
echo "Open https://pytorch.org/memory_viz and drag/drop any generated *.pickle file."
