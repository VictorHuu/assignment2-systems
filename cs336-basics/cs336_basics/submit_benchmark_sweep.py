from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import submitit


@dataclass(frozen=True)
class BenchmarkConfig:
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    context_length: int
    batch_size: int
    precision: str
    mode: str
    warmup_steps: int


def comma_separated_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def comma_separated_strings(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a benchmarking sweep for cs336_basics.benchmarking_script using submitit/Slurm."
    )
    parser.add_argument("--log-dir", default="submitit_logs/benchmark_sweep")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to pass to benchmark script. Use 'auto' to let the benchmark pick cuda if available, else cpu.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5, help="Single warmup step value for all jobs.")
    parser.add_argument(
        "--warmup-steps-list",
        type=comma_separated_ints,
        default=[],
        help="Optional comma-separated warmup steps (e.g. 0,1,2,5). Overrides --warmup-steps when provided.",
    )
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=50_257)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--d-models", type=comma_separated_ints, default=[768])
    parser.add_argument("--num-layers", type=comma_separated_ints, default=[12])
    parser.add_argument("--num-heads", type=comma_separated_ints, default=[12])
    parser.add_argument("--d-ffs", type=comma_separated_ints, default=[3072])
    parser.add_argument("--context-lengths", type=comma_separated_ints, default=[1024])
    parser.add_argument("--batch-sizes", type=comma_separated_ints, default=[8])
    parser.add_argument("--precisions", type=comma_separated_strings, default=["fp32"])
    parser.add_argument("--modes", type=comma_separated_strings, default=["training"])

    parser.add_argument("--slurm-partition", default="")
    parser.add_argument("--slurm-account", default="")
    parser.add_argument("--slurm-qos", default="")
    parser.add_argument("--timeout-min", type=int, default=60)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem-gb", type=int, default=32)
    parser.add_argument("--omp-num-threads", type=int, default=0, help="Set OMP_NUM_THREADS for benchmark jobs.")
    parser.add_argument("--mkl-num-threads", type=int, default=0, help="Set MKL_NUM_THREADS for benchmark jobs.")
    parser.add_argument(
        "--slurm-exclusive",
        action="store_true",
        help="Request exclusive node allocation (if supported by your cluster).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> list[BenchmarkConfig]:
    product = itertools.product(
        args.d_models,
        args.num_layers,
        args.num_heads,
        args.d_ffs,
        args.context_lengths,
        args.batch_sizes,
        args.precisions,
        args.modes,
        args.warmup_steps_list if args.warmup_steps_list else [args.warmup_steps],
    )
    return [
        BenchmarkConfig(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            context_length=context_length,
            batch_size=batch_size,
            precision=precision,
            mode=mode,
            warmup_steps=warmup_steps,
        )
        for d_model, num_layers, num_heads, d_ff, context_length, batch_size, precision, mode, warmup_steps in product
    ]


def run_single(config: BenchmarkConfig, args: argparse.Namespace) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    basics_root = repo_root / "cs336-basics"

    cmd = [
        sys.executable,
        "-m",
        "cs336_basics.benchmarking_script",
        "--precision",
        config.precision,
        "--mode",
        config.mode,
        "--warmup-steps",
        str(config.warmup_steps),
        "--measure-steps",
        str(args.measure_steps),
        "--batch-size",
        str(config.batch_size),
        "--context-length",
        str(config.context_length),
        "--vocab-size",
        str(args.vocab_size),
        "--d-model",
        str(config.d_model),
        "--num-layers",
        str(config.num_layers),
        "--num-heads",
        str(config.num_heads),
        "--d-ff",
        str(config.d_ff),
        "--rope-theta",
        str(args.rope_theta),
    ]

    if args.device != "auto":
        cmd.extend(["--device", args.device])

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{basics_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(basics_root)
    )
    env["OMP_NUM_THREADS"] = str(args.omp_num_threads or args.cpus_per_task)
    env["MKL_NUM_THREADS"] = str(args.mkl_num_threads or args.cpus_per_task)

    process = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env, cwd=repo_root)
    return process.stdout.strip()


def main() -> None:
    args = parse_args()
    configs = build_configs(args)

    print(f"Prepared {len(configs)} benchmark jobs.")
    if args.dry_run:
        for idx, config in enumerate(configs):
            print(f"[{idx}] {config}")
        return

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=str(log_dir))
    executor.update_parameters(
        timeout_min=args.timeout_min,
        slurm_partition=args.slurm_partition or None,
        slurm_account=args.slurm_account or None,
        slurm_qos=args.slurm_qos or None,
        slurm_gpus_per_node=args.gpus_per_node,
        slurm_cpus_per_task=args.cpus_per_task,
        slurm_mem=f"{args.mem_gb}G",
        slurm_exclusive=args.slurm_exclusive,
    )

    jobs = [executor.submit(run_single, config, args) for config in configs]
    for config, job in zip(configs, jobs, strict=True):
        print(f"Submitted job_id={job.job_id} config={config}")


if __name__ == "__main__":
    main()
