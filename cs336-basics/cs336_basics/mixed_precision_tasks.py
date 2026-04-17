from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utilities for assignment mixed-precision questions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("accumulation", help="Run the accumulation precision demo from the handout.")

    toy = sub.add_parser("toy_dtypes", help="Report tensor dtypes for ToyModel under autocast(fp16).")
    toy.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    bench = sub.add_parser("benchmark_compare", help="Compare fp32 vs bf16 autocast timings.")
    bench.add_argument("--device", default="cuda")
    bench.add_argument("--mode", choices=["inference", "training"], default="training")
    bench.add_argument("--warmup-steps", type=int, default=5)
    bench.add_argument("--measure-steps", type=int, default=10)
    bench.add_argument("--batch-size", type=int, default=8)
    bench.add_argument("--vocab-size", type=int, default=50_257)
    bench.add_argument("--sizes", default="256:4:8:1024,512:8:8:2048,768:12:12:3072")
    return parser.parse_args()


def run_accumulation_demo() -> dict[str, float]:
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    fp32_fp32 = float(s.item())

    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    fp16_fp16 = float(s.item())

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    fp32_plus_fp16 = float(s.item())

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    fp32_plus_cast_fp16 = float(s.item())

    return {
        "fp32_accum_fp32_input": fp32_fp32,
        "fp16_accum_fp16_input": fp16_fp16,
        "fp32_accum_fp16_input": fp32_plus_fp16,
        "fp32_accum_cast_fp16_input": fp32_plus_cast_fp16,
    }


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def inspect_toy_dtypes(device: str) -> dict[str, str]:
    if device != "cuda":
        raise ValueError("toy_dtypes should be run on CUDA to match mixed-precision behavior in the assignment.")
    model = ToyModel(8, 4).to(device=device, dtype=torch.float32)
    x = torch.randn(2, 8, device=device, dtype=torch.float32, requires_grad=False)
    target = torch.randn(2, 4, device=device, dtype=torch.float32)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        fc1_out = model.fc1(x)
        ln_out = model.ln(model.relu(fc1_out))
        logits = model.fc2(ln_out)
        loss = ((logits - target) ** 2).mean()

    loss.backward()
    grad_dtype = next(p.grad for p in model.parameters() if p.grad is not None).dtype

    return {
        "parameter_dtype": str(next(model.parameters()).dtype),
        "fc1_output_dtype": str(fc1_out.dtype),
        "layernorm_output_dtype": str(ln_out.dtype),
        "logits_dtype": str(logits.dtype),
        "loss_dtype": str(loss.dtype),
        "gradient_dtype": str(grad_dtype),
    }


@dataclass
class SizeConfig:
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int


def parse_sizes(spec: str) -> list[SizeConfig]:
    sizes: list[SizeConfig] = []
    for chunk in spec.split(","):
        d_model, num_layers, num_heads, d_ff = (int(x) for x in chunk.split(":"))
        sizes.append(SizeConfig(d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff))
    return sizes


def parse_mean_ms(output: str) -> float:
    match = re.search(r"mean=([0-9.]+)\s+ms", output)
    if not match:
        raise ValueError(f"Could not parse mean ms from output:\n{output}")
    return float(match.group(1))


def run_benchmark_once(args: argparse.Namespace, size: SizeConfig, autocast_dtype: str) -> float:
    cmd = [
        sys.executable,
        "-m",
        "cs336_basics.benchmarking_script",
        "--device",
        args.device,
        "--precision",
        "fp32",
        "--autocast-dtype",
        autocast_dtype,
        "--mode",
        args.mode,
        "--warmup-steps",
        str(args.warmup_steps),
        "--measure-steps",
        str(args.measure_steps),
        "--batch-size",
        str(args.batch_size),
        "--context-length",
        "1024",
        "--vocab-size",
        str(args.vocab_size),
        "--d-model",
        str(size.d_model),
        "--num-layers",
        str(size.num_layers),
        "--num-heads",
        str(size.num_heads),
        "--d-ff",
        str(size.d_ff),
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return parse_mean_ms(proc.stdout)


def run_benchmark_compare(args: argparse.Namespace) -> None:
    print("name,d_model,num_layers,num_heads,d_ff,mode,fp32_ms,bf16_autocast_ms,speedup")
    for idx, size in enumerate(parse_sizes(args.sizes)):
        fp32_ms = run_benchmark_once(args, size, "none")
        bf16_ms = run_benchmark_once(args, size, "bf16")
        speedup = fp32_ms / bf16_ms if bf16_ms > 0 else float("inf")
        print(
            f"size{idx},{size.d_model},{size.num_layers},{size.num_heads},{size.d_ff},"
            f"{args.mode},{fp32_ms:.4f},{bf16_ms:.4f},{speedup:.4f}"
        )


def main() -> None:
    args = parse_args()
    if args.cmd == "accumulation":
        result = run_accumulation_demo()
        for k, v in result.items():
            print(f"{k}={v}")
    elif args.cmd == "toy_dtypes":
        result = inspect_toy_dtypes(args.device)
        for k, v in result.items():
            print(f"{k}={v}")
    elif args.cmd == "benchmark_compare":
        run_benchmark_compare(args)
    else:
        raise ValueError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    main()
