from __future__ import annotations

import argparse
import contextlib
import statistics
from dataclasses import dataclass
from timeit import default_timer
from typing import Iterator

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end forward/backward benchmark for BasicsTransformerLM")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument(
        "--mode",
        choices=["inference", "training"],
        default="training",
        help="inference=forward-only, training=forward+backward+optimizer step.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument(
        "--enable-nvtx",
        action="store_true",
        help="Annotate warmup/measured/model regions and fine-grained step stages with NVTX.",
    )
    parser.add_argument(
        "--annotate-attention-nvtx",
        action="store_true",
        help="Wrap scaled_dot_product_attention with detailed NVTX ranges (CUDA-only).",
    )
    parser.add_argument("--optimizer-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--compare-compile",
        action="store_true",
        help=(
            "Run a 4-way comparison table for vanilla vs torch.compile with both "
            "forward-only and end-to-end (forward+backward+optimizer) timings."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Mode passed to torch.compile(..., mode=...). Used only with --compare-compile.",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Record CUDA memory allocation history and dump a snapshot pickle.",
    )
    parser.add_argument(
        "--memory-snapshot-path",
        default="memory_snapshot.pickle",
        help="Output path for torch.cuda.memory._dump_snapshot when --memory-profile is enabled.",
    )
    parser.add_argument(
        "--memory-max-entries",
        type=int,
        default=1_000_000,
        help="Maximum number of memory history entries to keep while recording.",
    )
    parser.add_argument(
        "--autocast-dtype",
        choices=["none", "fp16", "bf16"],
        default="none",
        help="Use torch.autocast for mixed precision during forward/loss computation.",
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=50_257)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    return parser.parse_args()


def get_dtype(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_autocast_context(device: torch.device, autocast_dtype: str):
    if autocast_dtype == "none":
        return contextlib.nullcontext()
    if device.type != "cuda":
        raise ValueError("--autocast-dtype requires --device=cuda.")
    dtype = torch.float16 if autocast_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@contextlib.contextmanager
def maybe_nvtx_range(enabled: bool, message: str):
    if enabled:
        with torch.cuda.nvtx.range(message):
            yield
    else:
        yield


@contextlib.contextmanager
def maybe_patch_attention_with_nvtx(enabled: bool, device: torch.device) -> Iterator[None]:
    if not enabled or device.type != "cuda":
        yield
        return

    import cs336_basics.model as model_module

    original = model_module.scaled_dot_product_attention

    def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
        with torch.cuda.nvtx.range("scaled dot product attention"):
            with torch.cuda.nvtx.range("computing attention scores"):
                d_k = K.shape[-1]
                attention_scores = torch.einsum("...qd,...kd->...qk", Q, K) / (d_k**0.5)
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))
            with torch.cuda.nvtx.range("computing softmax"):
                attention_weights = torch.softmax(attention_scores, dim=-1)
            with torch.cuda.nvtx.range("final matmul"):
                return torch.einsum("...qk,...kd->...qd", attention_weights, V)

    model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    try:
        yield
    finally:
        model_module.scaled_dot_product_attention = original


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = get_dtype(args.precision)
    torch.manual_seed(args.seed)
    autocast_enabled = args.autocast_dtype != "none"

    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 on CPU is not supported for this benchmark. Use fp32 or bf16.")
    if autocast_enabled and dtype != torch.float32:
        raise ValueError("When using --autocast-dtype, keep --precision=fp32 so parameters stay in FP32.")
    if args.memory_profile and device.type != "cuda":
        raise ValueError("--memory-profile requires CUDA because torch.cuda.memory snapshots are CUDA-only.")

    def build_model() -> BasicsTransformerLM:
        return BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
        ).to(device=device, dtype=dtype)

    base_state_dict: dict[str, torch.Tensor] | None = None
    single_run_model: BasicsTransformerLM | None = None
    if args.compare_compile:
        base_model = build_model()
        base_state_dict = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        single_run_model = build_model()

    input_tokens = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )
    labels = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )
    @dataclass
    class BenchResult:
        mean_ms: float
        median_ms: float
        std_ms: float

    def make_fresh_model() -> BasicsTransformerLM:
        if args.compare_compile:
            assert base_state_dict is not None
            model = build_model()
            model.load_state_dict(base_state_dict)
            return model
        assert single_run_model is not None
        return single_run_model

    nvtx_enabled = args.enable_nvtx and device.type == "cuda"

    @contextlib.contextmanager
    def stage_nvtx(stage_name: str):
        with maybe_nvtx_range(nvtx_enabled, stage_name):
            yield

    def run_single_benchmark(model: torch.nn.Module, mode: str) -> BenchResult:
        optimizer = AdamW(model.parameters(), lr=args.optimizer_lr) if mode == "training" else None
        model.eval()

        def inference_step() -> None:
            with stage_nvtx("forward"):
                with torch.inference_mode():
                    with get_autocast_context(device, args.autocast_dtype):
                        _ = model(input_tokens)

        def training_step() -> None:
            model.train()
            assert optimizer is not None
            with stage_nvtx("optimizer_zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            with stage_nvtx("forward"):
                with get_autocast_context(device, args.autocast_dtype):
                    logits = model(input_tokens)
            with stage_nvtx("loss"):
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            with stage_nvtx("backward"):
                loss.backward()
            with stage_nvtx("optimizer_step"):
                optimizer.step()

        step_fn_map = {"inference": inference_step, "training": training_step}
        step_fn = step_fn_map[mode]

        with maybe_patch_attention_with_nvtx(args.annotate_attention_nvtx, device):
            with maybe_nvtx_range(args.enable_nvtx and device.type == "cuda", "warmup"):
                for _ in range(args.warmup_steps):
                    with maybe_nvtx_range(args.enable_nvtx and device.type == "cuda", "warmup_step"):
                        step_fn()
                    synchronize(device)

            if args.memory_profile:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.memory._record_memory_history(max_entries=args.memory_max_entries)

            timings = []
            for _ in range(args.measure_steps):
                with maybe_nvtx_range(args.enable_nvtx and device.type == "cuda", "measured_step"):
                    start = default_timer()
                    with maybe_nvtx_range(args.enable_nvtx and device.type == "cuda", mode):
                        step_fn()
                    synchronize(device)
                    end = default_timer()
                timings.append(end - start)

            if args.memory_profile:
                torch.cuda.memory._dump_snapshot(args.memory_snapshot_path)
                torch.cuda.memory._record_memory_history(enabled=None)

        return BenchResult(
            mean_ms=statistics.mean(timings) * 1000,
            median_ms=statistics.median(timings) * 1000,
            std_ms=statistics.stdev(timings) * 1000 if len(timings) > 1 else 0.0,
        )

    print("Benchmark configuration:")
    print(
        f"device={device}, precision={args.precision}, mode={args.mode}, autocast_dtype={args.autocast_dtype}, "
        f"warmup_steps={args.warmup_steps}, measure_steps={args.measure_steps}"
    )
    print(
        f"batch_size={args.batch_size}, context_length={args.context_length}, vocab_size={args.vocab_size}, "
        f"d_model={args.d_model}, num_layers={args.num_layers}, num_heads={args.num_heads}, d_ff={args.d_ff}"
    )

    if args.compare_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build.")
        if args.memory_profile:
            raise ValueError("--memory-profile is only supported for single-run mode, not --compare-compile.")

        vanilla_forward = run_single_benchmark(make_fresh_model(), mode="inference")
        compiled_forward_model = torch.compile(make_fresh_model(), mode=args.compile_mode)
        compiled_forward = run_single_benchmark(compiled_forward_model, mode="inference")
        vanilla_e2e = run_single_benchmark(make_fresh_model(), mode="training")
        compiled_e2e_model = torch.compile(make_fresh_model(), mode=args.compile_mode)
        compiled_e2e = run_single_benchmark(compiled_e2e_model, mode="training")

        print("\nTransformer torch.compile comparison (ms):")
        print("| Benchmark | Mean (ms) | Median (ms) | Std (ms) |")
        print("|---|---:|---:|---:|")
        print(f"| vanilla forward | {vanilla_forward.mean_ms:.3f} | {vanilla_forward.median_ms:.3f} | {vanilla_forward.std_ms:.3f} |")
        print(f"| compiled forward | {compiled_forward.mean_ms:.3f} | {compiled_forward.median_ms:.3f} | {compiled_forward.std_ms:.3f} |")
        print(f"| vanilla end-to-end | {vanilla_e2e.mean_ms:.3f} | {vanilla_e2e.median_ms:.3f} | {vanilla_e2e.std_ms:.3f} |")
        print(f"| compiled end-to-end | {compiled_e2e.mean_ms:.3f} | {compiled_e2e.median_ms:.3f} | {compiled_e2e.std_ms:.3f} |")
        print(f"compile_mode={args.compile_mode}")
    else:
        result = run_single_benchmark(make_fresh_model(), mode=args.mode)
        print(f"mean={result.mean_ms:.3f} ms, median={result.median_ms:.3f} ms, std={result.std_ms:.3f} ms")
        if args.memory_profile:
            peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
            peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)
            print(
                f"memory_snapshot={args.memory_snapshot_path}, "
                f"peak_allocated={peak_allocated_mib:.3f} MiB, peak_reserved={peak_reserved_mib:.3f} MiB"
            )


if __name__ == "__main__":
    main()
