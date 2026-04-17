from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SummaryRow:
    report_path: str
    mode: str
    context_length: int | None
    d_model: int | None
    num_layers: int | None
    measured_step_ms: float | None
    mode_step_ms: float | None
    top_kernel_name: str | None
    top_kernel_time_pct: float | None
    top_kernel_instances: int | None
    softmax_avg_ns: float | None
    final_matmul_avg_ns: float | None
    softmax_to_final_matmul_ratio: float | None
    h2d_total_mb: float | None
    d2d_total_mb: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize key metrics from multiple .nsys-rep files.")
    parser.add_argument(
        "reports",
        nargs="*",
        help="Optional explicit .nsys-rep files. If omitted, --report-dir + --glob are used.",
    )
    parser.add_argument("--report-dir", default="nsys_reports")
    parser.add_argument("--glob", default="*.nsys-rep")
    parser.add_argument("--output-csv", default="", help="Optional output CSV path. Defaults to stdout.")
    return parser.parse_args()


def discover_reports(args: argparse.Namespace) -> list[Path]:
    if args.reports:
        return [Path(p) for p in args.reports]
    return sorted(Path(args.report_dir).glob(args.glob))


def run_nsys_stats(report: Path) -> str:
    cmd = [
        "nsys",
        "stats",
        "--report",
        "nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum",
        str(report),
    ]
    process = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return process.stdout


def parse_table_row(line: str) -> list[str]:
    return [tok for tok in re.split(r"\s{2,}", line.strip()) if tok]


def parse_section_rows(text: str, marker: str) -> list[list[str]]:
    rows: list[list[str]] = []
    in_section = False
    in_table = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if marker in line:
            in_section = True
            in_table = False
            continue
        if not in_section:
            continue
        if line.strip().startswith("Time (%)") or line.strip().startswith("Total (MB)"):
            in_table = True
            continue
        if in_table and line.strip().startswith("--------"):
            continue
        if in_table and line.strip().startswith("Processing ["):
            break
        if in_table and line.strip().startswith("** "):
            break
        if in_table and line.strip():
            tokens = parse_table_row(line)
            if tokens:
                rows.append(tokens)
    return rows


def first_matching_nvtx_avg_ns(rows: list[list[str]], ranges: list[str]) -> float | None:
    for row in rows:
        if len(row) < 10:
            continue
        range_name = row[-1]
        if range_name in ranges:
            return float(row[3])
    return None


def parse_filename_metadata(path: Path) -> dict[str, Any]:
    pattern = re.compile(r"nsys_(?P<mode>.+)_ctx(?P<ctx>\d+)_d(?P<d>\d+)_L(?P<L>\d+)\.nsys-rep$")
    match = pattern.search(path.name)
    if not match:
        return {"mode": path.stem, "context_length": None, "d_model": None, "num_layers": None}
    return {
        "mode": match.group("mode"),
        "context_length": int(match.group("ctx")),
        "d_model": int(match.group("d")),
        "num_layers": int(match.group("L")),
    }


def summarize_report(report: Path) -> SummaryRow:
    text = run_nsys_stats(report)
    meta = parse_filename_metadata(report)

    nvtx_rows = parse_section_rows(text, "** NVTX Range Summary")
    kernel_rows = parse_section_rows(text, "** CUDA GPU Kernel Summary")
    mem_size_rows = parse_section_rows(text, "** CUDA GPU MemOps Summary (by Size)")

    measured_step_ns = first_matching_nvtx_avg_ns(nvtx_rows, [":measured_step"])
    mode_step_ns = first_matching_nvtx_avg_ns(
        nvtx_rows,
        [":inference", ":forward", ":forward_step", ":training", ":training_step"],
    )
    softmax_ns = first_matching_nvtx_avg_ns(nvtx_rows, [":computing softmax"])
    final_matmul_ns = first_matching_nvtx_avg_ns(nvtx_rows, [":final matmul"])

    top_kernel_name = None
    top_kernel_pct = None
    top_kernel_instances = None
    if kernel_rows:
        top = kernel_rows[0]
        if len(top) >= 9:
            top_kernel_pct = float(top[0])
            top_kernel_instances = int(top[2])
            top_kernel_name = top[-1]

    h2d_total_mb = None
    d2d_total_mb = None
    for row in mem_size_rows:
        if len(row) < 8:
            continue
        total_mb = float(row[0])
        operation = row[-1]
        if operation == "[CUDA memcpy Host-to-Device]":
            h2d_total_mb = total_mb
        elif operation == "[CUDA memcpy Device-to-Device]":
            d2d_total_mb = total_mb

    ratio = None
    if softmax_ns and final_matmul_ns and final_matmul_ns > 0:
        ratio = softmax_ns / final_matmul_ns

    return SummaryRow(
        report_path=str(report),
        mode=meta["mode"],
        context_length=meta["context_length"],
        d_model=meta["d_model"],
        num_layers=meta["num_layers"],
        measured_step_ms=None if measured_step_ns is None else measured_step_ns / 1e6,
        mode_step_ms=None if mode_step_ns is None else mode_step_ns / 1e6,
        top_kernel_name=top_kernel_name,
        top_kernel_time_pct=top_kernel_pct,
        top_kernel_instances=top_kernel_instances,
        softmax_avg_ns=softmax_ns,
        final_matmul_avg_ns=final_matmul_ns,
        softmax_to_final_matmul_ratio=ratio,
        h2d_total_mb=h2d_total_mb,
        d2d_total_mb=d2d_total_mb,
    )


def write_csv(rows: list[SummaryRow], output_csv: str) -> None:
    fieldnames = list(SummaryRow.__annotations__.keys())
    if output_csv:
        out_f = Path(output_csv).open("w", newline="", encoding="utf-8")
        close_file = True
    else:
        import sys

        out_f = sys.stdout
        close_file = False
    try:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    finally:
        if close_file:
            out_f.close()


def main() -> None:
    args = parse_args()
    reports = discover_reports(args)
    if not reports:
        raise SystemExit("No .nsys-rep files found. Pass paths explicitly or set --report-dir/--glob.")
    rows = [summarize_report(r) for r in reports]
    rows.sort(key=lambda r: (r.mode, r.context_length or -1, r.report_path))
    write_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()
