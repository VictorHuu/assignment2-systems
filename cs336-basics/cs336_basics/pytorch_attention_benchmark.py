from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from timeit import default_timer

import torch


@dataclass
class BenchmarkRow:
    variant: str
    d_model: int
    seq_len: int
    status: str
    forward_ms_mean: float | None
    backward_ms_mean: float | None
    memory_before_backward_mib_mean: float | None
    memory_before_backward_mib_max: float | None
    note: str


class AttentionModule(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = torch.where(mask, scores, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return probs @ v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naive PyTorch scaled-dot-product attention.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-models", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=100)
    parser.add_argument("--compare-compile", action="store_true", help="Also benchmark torch.compile(attention_module).")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Mode passed to torch.compile(..., mode=...). Used with --compare-compile.",
    )
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


def benchmark_variant(
    *,
    variant: str,
    attn_module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    mask: torch.Tensor,
    warmup_iters: int,
    measure_iters: int,
    device: torch.device,
) -> tuple[float, float, float, float]:
    for _ in range(warmup_iters):
        out = attn_module(q, k, v, mask)
        synchronize(device)
        out.backward(grad_out, retain_graph=False)
        synchronize(device)
        q.grad = None
        k.grad = None
        v.grad = None

    forward_times_ms: list[float] = []
    backward_times_ms: list[float] = []
    mem_before_backward_mib: list[float] = []

    for _ in range(measure_iters):
        t0 = default_timer()
        out = attn_module(q, k, v, mask)
        synchronize(device)
        t1 = default_timer()
        forward_times_ms.append((t1 - t0) * 1000.0)

        mem_before_backward_mib.append(torch.cuda.memory_allocated(device) / (1024.0**2))

        t2 = default_timer()
        out.backward(grad_out, retain_graph=False)
        synchronize(device)
        t3 = default_timer()
        backward_times_ms.append((t3 - t2) * 1000.0)

        q.grad = None
        k.grad = None
        v.grad = None

    return (
        sum(forward_times_ms) / len(forward_times_ms),
        sum(backward_times_ms) / len(backward_times_ms),
        sum(mem_before_backward_mib) / len(mem_before_backward_mib),
        max(mem_before_backward_mib),
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)

    if device.type != "cuda":
        raise RuntimeError("This benchmark is designed for CUDA; use --device cuda on a GPU machine.")

    if args.compare_compile and not hasattr(torch, "compile"):
        raise RuntimeError("--compare-compile requested but torch.compile is unavailable in this PyTorch build.")

    torch.manual_seed(0)

    rows: list[BenchmarkRow] = []

    for d_model in args.d_models:
        for seq_len in args.seq_lens:
            print(f"Running d_model={d_model}, seq_len={seq_len} ...", flush=True)
            try:
                mask = build_causal_mask(seq_len, device)
                base_q = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)
                base_k = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)
                base_v = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)
                base_grad_out = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)

                variants: list[tuple[str, torch.nn.Module]] = [("vanilla", AttentionModule())]
                if args.compare_compile:
                    compiled = torch.compile(AttentionModule(), mode=args.compile_mode)
                    variants.append(("compiled", compiled))

                for variant_name, module in variants:
                    q = base_q.detach().clone().requires_grad_(True)
                    k = base_k.detach().clone().requires_grad_(True)
                    v = base_v.detach().clone().requires_grad_(True)
                    grad_out = base_grad_out.detach().clone()

                    fwd_ms, bwd_ms, mem_mean, mem_max = benchmark_variant(
                        variant=variant_name,
                        attn_module=module,
                        q=q,
                        k=k,
                        v=v,
                        grad_out=grad_out,
                        mask=mask,
                        warmup_iters=args.warmup_iters,
                        measure_iters=args.measure_iters,
                        device=device,
                    )
                    rows.append(
                        BenchmarkRow(
                            variant=variant_name,
                            d_model=d_model,
                            seq_len=seq_len,
                            status="ok",
                            forward_ms_mean=fwd_ms,
                            backward_ms_mean=bwd_ms,
                            memory_before_backward_mib_mean=mem_mean,
                            memory_before_backward_mib_max=mem_max,
                            note="",
                        )
                    )

            except torch.cuda.OutOfMemoryError as exc:
                rows.append(
                    BenchmarkRow(
                        variant="vanilla",
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
                if args.compare_compile:
                    rows.append(
                        BenchmarkRow(
                            variant="compiled",
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
                        variant="vanilla",
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
                if args.compare_compile:
                    rows.append(
                        BenchmarkRow(
                            variant="compiled",
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

    if not rows:
        raise RuntimeError("No benchmark rows were produced.")

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    if args.compare_compile:
        print("\nAttention torch.compile comparison (mean ms):")
        print("| d_model | seq_len | vanilla forward | compiled forward | vanilla backward | compiled backward |")
        print("|---:|---:|---:|---:|---:|---:|")
        for d_model in args.d_models:
            for seq_len in args.seq_lens:
                vanilla = next((r for r in rows if r.variant == "vanilla" and r.d_model == d_model and r.seq_len == seq_len and r.status == "ok"), None)
                compiled = next((r for r in rows if r.variant == "compiled" and r.d_model == d_model and r.seq_len == seq_len and r.status == "ok"), None)
                if vanilla is None or compiled is None:
                    continue
                print(
                    f"| {d_model} | {seq_len} | {vanilla.forward_ms_mean:.3f} | {compiled.forward_ms_mean:.3f} "
                    f"| {vanilla.backward_ms_mean:.3f} | {compiled.backward_ms_mean:.3f} |"
                )
        print(f"compile_mode={args.compile_mode}")

    print(f"Wrote {len(rows)} rows to {args.csv_path}")


if __name__ == "__main__":
    main()