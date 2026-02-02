#!/usr/bin/env python3
"""
Training Throughput & Memory Benchmarks
========================================

This benchmark suite measures:
1. Tokens/second throughput at various batch sizes
2. GPU memory usage (allocated vs reserved)
3. Scaling efficiency across multiple GPUs
4. LoRA vs full fine-tuning comparison

Usage:
------
# Single GPU benchmark
python benchmarks/benchmark_training.py --model gpt2 --batch-sizes 1,2,4,8

# Multi-GPU scaling benchmark (requires torchrun)
torchrun --nproc_per_node=4 benchmarks/benchmark_training.py --model gpt2 --scaling-test

# Full benchmark suite
python benchmarks/benchmark_training.py --model gpt2 --full-suite
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.lora import apply_lora_to_model, LoRAConfig, count_parameters


@dataclass
class BenchmarkResult:
    """Container for benchmark metrics."""
    model_name: str
    batch_size: int
    seq_length: int
    tokens_per_second: float
    samples_per_second: float
    memory_allocated_gb: float
    memory_reserved_gb: float
    peak_memory_gb: float
    forward_time_ms: float
    backward_time_ms: float
    step_time_ms: float
    lora_enabled: bool
    lora_rank: Optional[int] = None
    world_size: int = 1
    dtype: str = "fp32"
    gradient_checkpointing: bool = False


@contextmanager
def cuda_timer():
    """Context manager for accurate CUDA timing."""
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield lambda: (end.record(), torch.cuda.synchronize(), start.elapsed_time(end))
    else:
        start_time = time.perf_counter()
        yield lambda: (time.perf_counter() - start_time) * 1000


class MemoryTracker:
    """Track GPU memory during benchmarks."""

    def __init__(self):
        self.enabled = torch.cuda.is_available()
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()

    def get_stats(self) -> Dict[str, float]:
        if not self.enabled:
            return {"allocated_gb": 0, "reserved_gb": 0, "peak_gb": 0}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
        }

    def reset(self):
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()


def create_synthetic_batch(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Create synthetic training batch."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def run_training_benchmark(
    model: torch.nn.Module,
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = "cuda",
    use_amp: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """
    Run training benchmark and return timing metrics.

    Measures:
    - Forward pass time
    - Backward pass time
    - Full step time (forward + backward + optimizer)
    - Throughput (tokens/sec, samples/sec)
    """
    model.train()

    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # Mixed precision - GradScaler only works with FP16, not BFloat16
    # BFloat16 doesn't need loss scaling due to its larger exponent range
    use_scaler = use_amp and device == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    memory = MemoryTracker()
    memory.reset()

    forward_times = []
    backward_times = []
    step_times = []

    for i in range(num_warmup + num_iterations):
        batch = create_synthetic_batch(batch_size, seq_length, vocab_size, device)
        optimizer.zero_grad(set_to_none=True)

        # Forward
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.amp.autocast("cuda", enabled=use_amp and device == "cuda", dtype=dtype):
            outputs = model(**batch)
            loss = outputs.loss

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Backward
        scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()

        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Optimizer step
        scaler.step(optimizer) if scaler.is_enabled() else optimizer.step()
        scaler.update() if scaler.is_enabled() else None

        if device == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        if i >= num_warmup:
            forward_times.append((t1 - t0) * 1000)
            backward_times.append((t2 - t1) * 1000)
            step_times.append((t3 - t0) * 1000)

    mem_stats = memory.get_stats()

    avg_step_ms = sum(step_times) / len(step_times)
    tokens_per_step = batch_size * seq_length
    tokens_per_sec = tokens_per_step / (avg_step_ms / 1000)

    return {
        "forward_time_ms": sum(forward_times) / len(forward_times),
        "backward_time_ms": sum(backward_times) / len(backward_times),
        "step_time_ms": avg_step_ms,
        "tokens_per_second": tokens_per_sec,
        "samples_per_second": batch_size / (avg_step_ms / 1000),
        **mem_stats,
    }


def benchmark_batch_sizes(
    model_name: str,
    seq_length: int = 512,
    batch_sizes: List[int] = None,
    lora_rank: int = 0,
    use_amp: bool = True,
    dtype_str: str = "fp16",
    gradient_checkpointing: bool = False,
) -> List[BenchmarkResult]:
    """Benchmark training at various batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.float32)

    # Load model
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if device == "cuda" else torch.float32,
    )

    # Apply LoRA if specified
    lora_enabled = lora_rank > 0
    if lora_enabled:
        config = LoRAConfig(r=lora_rank, alpha=lora_rank * 2)
        model = apply_lora_to_model(model, config, verbose=False)
        print(f"Applied LoRA (r={lora_rank})")

    # Gradient checkpointing
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    model.to(device)

    vocab_size = model.config.vocab_size
    param_stats = count_parameters(model)
    print(f"Total params: {param_stats['total']:,}, Trainable: {param_stats['trainable']:,}")

    results = []

    for bs in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch size: {bs}, Seq length: {seq_length}")
        print(f"{'='*60}")

        try:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            metrics = run_training_benchmark(
                model=model,
                batch_size=bs,
                seq_length=seq_length,
                vocab_size=vocab_size,
                device=device,
                use_amp=use_amp,
                dtype=dtype,
            )

            result = BenchmarkResult(
                model_name=model_name,
                batch_size=bs,
                seq_length=seq_length,
                tokens_per_second=metrics["tokens_per_second"],
                samples_per_second=metrics["samples_per_second"],
                memory_allocated_gb=metrics["allocated_gb"],
                memory_reserved_gb=metrics["reserved_gb"],
                peak_memory_gb=metrics["peak_gb"],
                forward_time_ms=metrics["forward_time_ms"],
                backward_time_ms=metrics["backward_time_ms"],
                step_time_ms=metrics["step_time_ms"],
                lora_enabled=lora_enabled,
                lora_rank=lora_rank if lora_enabled else None,
                dtype=dtype_str,
                gradient_checkpointing=gradient_checkpointing,
            )
            results.append(result)

            print(f"  Throughput: {metrics['tokens_per_second']:.0f} tokens/sec")
            print(f"  Step time:  {metrics['step_time_ms']:.1f} ms")
            print(f"  Peak mem:   {metrics['peak_gb']:.2f} GB")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at batch size {bs}")
                break
            raise

    return results


def benchmark_lora_vs_full(
    model_name: str,
    batch_size: int = 4,
    seq_length: int = 512,
    lora_ranks: List[int] = None,
) -> Dict[str, List[BenchmarkResult]]:
    """Compare LoRA fine-tuning vs full fine-tuning."""
    if lora_ranks is None:
        lora_ranks = [4, 8, 16, 32]

    print("\n" + "="*70)
    print("LoRA vs Full Fine-tuning Comparison")
    print("="*70)

    results = {"full": [], "lora": {}}

    # Full fine-tuning baseline
    print("\n--- Full Fine-tuning (no LoRA) ---")
    full_results = benchmark_batch_sizes(
        model_name,
        seq_length=seq_length,
        batch_sizes=[batch_size],
        lora_rank=0,
    )
    results["full"] = full_results

    # LoRA at various ranks
    for rank in lora_ranks:
        print(f"\n--- LoRA (r={rank}) ---")
        lora_results = benchmark_batch_sizes(
            model_name,
            seq_length=seq_length,
            batch_sizes=[batch_size],
            lora_rank=rank,
        )
        results["lora"][rank] = lora_results

    # Summary
    if results["full"]:
        full = results["full"][0]
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        print(f"{'Method':<20} {'Tokens/sec':>12} {'Memory (GB)':>12} {'Speedup':>10}")
        print("-"*70)
        print(f"{'Full fine-tune':<20} {full.tokens_per_second:>12.0f} {full.peak_memory_gb:>12.2f} {'1.00x':>10}")

        for rank, lora_res in results["lora"].items():
            if lora_res:
                res = lora_res[0]
                speedup = res.tokens_per_second / full.tokens_per_second
                mem_savings = (full.peak_memory_gb - res.peak_memory_gb) / full.peak_memory_gb * 100
                print(f"{'LoRA r=' + str(rank):<20} {res.tokens_per_second:>12.0f} {res.peak_memory_gb:>12.2f} {speedup:>9.2f}x")

    return results


def benchmark_scaling(
    model_name: str,
    batch_size: int = 4,
    seq_length: int = 512,
) -> Dict[int, BenchmarkResult]:
    """Benchmark multi-GPU scaling efficiency."""
    # Check if distributed
    if not dist.is_initialized():
        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        else:
            print("Not in distributed mode. Run with torchrun for scaling benchmark.")
            return {}

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Scaling Benchmark - {world_size} GPUs")
        print(f"{'='*60}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    config = LoRAConfig(r=8, alpha=16)
    model = apply_lora_to_model(model, config, verbose=(rank == 0))
    model.to(device)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    vocab_size = model.module.config.vocab_size

    # Benchmark
    metrics = run_training_benchmark(
        model=model,
        batch_size=batch_size,
        seq_length=seq_length,
        vocab_size=vocab_size,
        device=device,
        use_amp=True,
        dtype=torch.bfloat16,
    )

    # Aggregate across ranks
    throughput_tensor = torch.tensor([metrics["tokens_per_second"]], device=device)
    dist.all_reduce(throughput_tensor, op=dist.ReduceOp.SUM)
    total_throughput = throughput_tensor.item()

    if rank == 0:
        print(f"\nTotal throughput: {total_throughput:.0f} tokens/sec")
        print(f"Per-GPU throughput: {metrics['tokens_per_second']:.0f} tokens/sec")
        print(f"Peak memory per GPU: {metrics['peak_gb']:.2f} GB")

    result = BenchmarkResult(
        model_name=model_name,
        batch_size=batch_size,
        seq_length=seq_length,
        tokens_per_second=total_throughput,
        samples_per_second=metrics["samples_per_second"] * world_size,
        memory_allocated_gb=metrics["allocated_gb"],
        memory_reserved_gb=metrics["reserved_gb"],
        peak_memory_gb=metrics["peak_gb"],
        forward_time_ms=metrics["forward_time_ms"],
        backward_time_ms=metrics["backward_time_ms"],
        step_time_ms=metrics["step_time_ms"],
        lora_enabled=True,
        lora_rank=8,
        world_size=world_size,
        dtype="bf16",
    )

    dist.destroy_process_group()
    return {world_size: result}


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save benchmark results to JSON."""
    data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


def print_results_table(results: List[BenchmarkResult]):
    """Print results in a formatted table."""
    print("\n" + "="*100)
    print(f"{'Batch':>6} {'Seq':>5} {'Tokens/s':>10} {'Samples/s':>10} {'Step(ms)':>9} {'Peak(GB)':>9} {'LoRA':>6}")
    print("="*100)

    for r in results:
        lora_str = f"r={r.lora_rank}" if r.lora_enabled else "no"
        print(
            f"{r.batch_size:>6} "
            f"{r.seq_length:>5} "
            f"{r.tokens_per_second:>10.0f} "
            f"{r.samples_per_second:>10.2f} "
            f"{r.step_time_ms:>9.1f} "
            f"{r.peak_memory_gb:>9.2f} "
            f"{lora_str:>6}"
        )


def main():
    parser = argparse.ArgumentParser(description="Training Throughput Benchmarks")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8", help="Comma-separated batch sizes")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (0 for full fine-tuning)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--scaling-test", action="store_true", help="Run multi-GPU scaling test")
    parser.add_argument("--lora-comparison", action="store_true", help="Compare LoRA vs full fine-tuning")
    parser.add_argument("--full-suite", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    print("="*70)
    print("DDP-LoRA-Trainer Benchmark Suite")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")

    all_results = []

    if args.scaling_test:
        benchmark_scaling(args.model, batch_size=4, seq_length=args.seq_length)
        return

    if args.lora_comparison or args.full_suite:
        benchmark_lora_vs_full(args.model, batch_size=4, seq_length=args.seq_length)

    if not args.lora_comparison:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        results = benchmark_batch_sizes(
            model_name=args.model,
            seq_length=args.seq_length,
            batch_sizes=batch_sizes,
            lora_rank=args.lora_rank,
            dtype_str=args.dtype,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        all_results.extend(results)
        print_results_table(results)

    if args.output and all_results:
        save_results(all_results, args.output)


if __name__ == "__main__":
    main()
