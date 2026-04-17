from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def comma_separated_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nsight Systems profiles for benchmarking_script.")
    parser.add_argument("--output-dir", default="nsys_reports")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--contexts", type=comma_separated_ints, default=[128, 256, 512, 1024])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=50_257)
    parser.add_argument("--modes", default="inference,training")

    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--python-backtrace-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_profile(args: argparse.Namespace, mode: str, context_length: int) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = output_dir / f"nsys_{mode}_ctx{context_length}_d{args.d_model}_L{args.num_layers}"

    cmd = [
        "nsys",
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt",
        "-o",
        str(out_prefix),
    ]
    if args.python_backtrace_cuda:
        cmd.append("--python-backtrace=cuda")

    cmd.extend(
        [
            "python",
            "-m",
            "cs336_basics.benchmarking_script",
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--mode",
            mode,
            "--warmup-steps",
            str(args.warmup_steps),
            "--measure-steps",
            str(args.measure_steps),
            "--batch-size",
            str(args.batch_size),
            "--context-length",
            str(context_length),
            "--vocab-size",
            str(args.vocab_size),
            "--d-model",
            str(args.d_model),
            "--num-layers",
            str(args.num_layers),
            "--num-heads",
            str(args.num_heads),
            "--d-ff",
            str(args.d_ff),
            "--rope-theta",
            str(args.rope_theta),
            "--enable-nvtx",
            "--annotate-attention-nvtx",
        ]
    )

    print("[nsys]", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        for context_length in args.contexts:
            run_profile(args, mode, context_length)


if __name__ == "__main__":
    main()
