from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from timeit import default_timer

import torch


@dataclass
class BenchmarkRow:
    d_model: int
    seq_len: int
    status: str
    forward_ms_mean: float | None
    backward_ms_mean: float | None
    memory_before_backward_mib_mean: float | None
    memory_before_backward_mib_max: float | None
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naive PyTorch scaled-dot-product attention.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-models", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=100)
    parser.add_argument("--csv-path", type=Path, default=Path("reports/pytorch_attention_benchmark_results.csv"))
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    return idx[None, :, None] >= idx[None, None, :]


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    scores = torch.where(mask, scores, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)

    if device.type != "cuda":
        raise RuntimeError("This benchmark is designed for CUDA; use --device cuda on a GPU machine.")

    torch.manual_seed(0)

    rows: list[BenchmarkRow] = []

    for d_model in args.d_models:
        for seq_len in args.seq_lens:
            print(f"Running d_model={d_model}, seq_len={seq_len} ...", flush=True)
            try:
                q = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                k = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                v = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                grad_out = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)
                mask = build_causal_mask(seq_len, device)

                # Warmup
                for _ in range(args.warmup_iters):
                    out = attention(q, k, v, mask)
                    synchronize(device)
                    out.backward(grad_out, retain_graph=False)
                    synchronize(device)
                    q.grad = None
                    k.grad = None
                    v.grad = None

                forward_times = []
                backward_times = []
                mem_before_backward = []

                for _ in range(args.measure_iters):
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)

                    t0 = default_timer()
                    out = attention(q, k, v, mask)
                    synchronize(device)
                    t1 = default_timer()
                    forward_times.append((t1 - t0) * 1000.0)

                    mem_before_backward.append(torch.cuda.memory_allocated(device) / (1024.0 ** 2))

                    t2 = default_timer()
                    out.backward(grad_out, retain_graph=False)
                    synchronize(device)
                    t3 = default_timer()
                    backward_times.append((t3 - t2) * 1000.0)

                    q.grad = None
                    k.grad = None
                    v.grad = None

                rows.append(
                    BenchmarkRow(
                        d_model=d_model,
                        seq_len=seq_len,
                        status="ok",
                        forward_ms_mean=sum(forward_times) / len(forward_times),
                        backward_ms_mean=sum(backward_times) / len(backward_times),
                        memory_before_backward_mib_mean=sum(mem_before_backward) / len(mem_before_backward),
                        memory_before_backward_mib_max=max(mem_before_backward),
                        note="",
                    )
                )
            except torch.cuda.OutOfMemoryError as exc:
                rows.append(
                    BenchmarkRow(
                        d_model=d_model,
                        seq_len=seq_len,
                        status="oom",
                        forward_ms_mean=None,
                        backward_ms_mean=None,
                        memory_before_backward_mib_mean=None,
                        memory_before_backward_mib_max=None,
                        note=str(exc).split("\n")[0],
                    )
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                rows.append(
                    BenchmarkRow(
                        d_model=d_model,
                        seq_len=seq_len,
                        status="runtime_error",
                        forward_ms_mean=None,
                        backward_ms_mean=None,
                        memory_before_backward_mib_mean=None,
                        memory_before_backward_mib_max=None,
                        note=str(exc).split("\n")[0],
                    )
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"Wrote {len(rows)} rows to {args.csv_path}")


if __name__ == "__main__":
    main()
