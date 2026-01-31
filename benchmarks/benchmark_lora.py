#!/usr/bin/env python3
"""
LoRA-Specific Benchmarks
========================

This benchmark focuses on LoRA operations:
1. Forward pass overhead (LoRA vs base linear)
2. Merge/unmerge performance
3. Memory savings at various ranks
4. Parameter efficiency analysis

Usage:
------
python benchmarks/benchmark_lora.py --hidden-sizes 768,1024,2048,4096
python benchmarks/benchmark_lora.py --ranks 4,8,16,32,64 --detailed
"""

import argparse
import gc
import sys
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import json

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lora import LoRALinear, LoRAConfig, apply_lora_to_model, count_parameters


@dataclass
class LoRABenchmarkResult:
    """Results from LoRA benchmarking."""
    hidden_size: int
    rank: int
    base_forward_us: float
    lora_forward_us: float
    merged_forward_us: float
    merge_time_us: float
    unmerge_time_us: float
    forward_overhead_percent: float
    memory_base_mb: float
    memory_lora_mb: float
    memory_savings_percent: float
    trainable_params: int
    total_params: int
    trainable_percent: float


def benchmark_forward_pass(
    layer: nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 50,
    num_iterations: int = 200,
) -> float:
    """Benchmark forward pass, return average time in microseconds."""
    # Warmup
    for _ in range(num_warmup):
        _ = layer(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = layer(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return (elapsed / num_iterations) * 1e6  # microseconds


def benchmark_lora_operations(
    hidden_size: int,
    rank: int,
    batch_size: int = 32,
    seq_length: int = 512,
    device: str = "cuda",
) -> LoRABenchmarkResult:
    """
    Comprehensive benchmark of LoRA operations.

    Tests:
    1. Base nn.Linear forward
    2. LoRA forward (unmerged)
    3. LoRA forward (merged)
    4. Merge operation
    5. Unmerge operation
    """
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Create base linear layer
    base_linear = nn.Linear(hidden_size, hidden_size).to(device)

    # Create LoRA layer
    lora_linear = LoRALinear.from_linear(
        base_linear,
        r=rank,
        alpha=rank * 2,
        dropout=0.0,
    ).to(device)

    # Test input
    x = torch.randn(batch_size, seq_length, hidden_size, device=device)

    # Memory baseline
    if device == "cuda":
        torch.cuda.synchronize()
        mem_base = torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        mem_base = 0

    # Benchmark base linear
    base_time = benchmark_forward_pass(base_linear, x)

    # Benchmark LoRA (unmerged)
    lora_time = benchmark_forward_pass(lora_linear, x)

    # Merge weights
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    lora_linear.merge()
    if device == "cuda":
        torch.cuda.synchronize()
    merge_time = (time.perf_counter() - t0) * 1e6

    # Benchmark LoRA (merged)
    merged_time = benchmark_forward_pass(lora_linear, x)

    # Unmerge weights
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    lora_linear.unmerge()
    if device == "cuda":
        torch.cuda.synchronize()
    unmerge_time = (time.perf_counter() - t0) * 1e6

    # Memory with LoRA
    if device == "cuda":
        mem_lora = torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        mem_lora = 0

    # Parameter counts
    total_params = hidden_size * hidden_size  # Base weight
    lora_params = rank * hidden_size * 2  # A + B matrices
    trainable_params = lora_params

    # Calculate metrics
    forward_overhead = ((lora_time - base_time) / base_time) * 100

    return LoRABenchmarkResult(
        hidden_size=hidden_size,
        rank=rank,
        base_forward_us=base_time,
        lora_forward_us=lora_time,
        merged_forward_us=merged_time,
        merge_time_us=merge_time,
        unmerge_time_us=unmerge_time,
        forward_overhead_percent=forward_overhead,
        memory_base_mb=mem_base,
        memory_lora_mb=mem_lora,
        memory_savings_percent=100 * (1 - trainable_params / total_params),
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_percent=100 * trainable_params / total_params,
    )


def benchmark_model_lora(
    model_name: str,
    ranks: List[int],
    device: str = "cuda",
) -> Dict[int, Dict]:
    """Benchmark LoRA application to a full model."""
    from transformers import AutoModelForCausalLM

    results = {}

    for rank in ranks:
        print(f"\n--- LoRA Rank {rank} ---")

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        # Memory before LoRA
        if device == "cuda":
            mem_before = torch.cuda.memory_allocated() / (1024**3)
        else:
            mem_before = 0

        # Apply LoRA
        config = LoRAConfig(r=rank, alpha=rank * 2)
        model = apply_lora_to_model(model, config, verbose=False)
        model.to(device)

        # Memory after LoRA
        if device == "cuda":
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / (1024**3)
        else:
            mem_after = 0

        # Parameter stats
        stats = count_parameters(model)

        results[rank] = {
            "total_params": stats["total"],
            "trainable_params": stats["trainable"],
            "lora_params": stats["lora"],
            "trainable_percent": stats["trainable_percent"],
            "memory_gb": mem_after,
            "memory_increase_gb": mem_after - mem_before,
        }

        print(f"  Total params:     {stats['total']:,}")
        print(f"  Trainable params: {stats['trainable']:,} ({stats['trainable_percent']:.2f}%)")
        print(f"  Memory:           {mem_after:.2f} GB")

        del model

    return results


def run_sweep(
    hidden_sizes: List[int],
    ranks: List[int],
    device: str = "cuda",
) -> List[LoRABenchmarkResult]:
    """Run benchmark sweep over hidden sizes and ranks."""
    results = []

    for hidden in hidden_sizes:
        for rank in ranks:
            if rank >= hidden:
                continue  # Skip invalid configurations

            print(f"Benchmarking: hidden={hidden}, rank={rank}")
            result = benchmark_lora_operations(hidden, rank, device=device)
            results.append(result)

    return results


def print_sweep_results(results: List[LoRABenchmarkResult]):
    """Print benchmark results in tabular format."""
    print("\n" + "="*120)
    print(f"{'Hidden':>8} {'Rank':>6} {'Base(us)':>10} {'LoRA(us)':>10} {'Merged(us)':>10} "
          f"{'Overhead%':>10} {'Trainable%':>11} {'Params':>12}")
    print("="*120)

    for r in results:
        print(
            f"{r.hidden_size:>8} "
            f"{r.rank:>6} "
            f"{r.base_forward_us:>10.1f} "
            f"{r.lora_forward_us:>10.1f} "
            f"{r.merged_forward_us:>10.1f} "
            f"{r.forward_overhead_percent:>10.1f} "
            f"{r.trainable_percent:>10.2f}% "
            f"{r.trainable_params:>12,}"
        )


def print_merge_analysis(results: List[LoRABenchmarkResult]):
    """Print merge/unmerge performance analysis."""
    print("\n" + "="*80)
    print("Merge/Unmerge Performance Analysis")
    print("="*80)
    print(f"{'Hidden':>8} {'Rank':>6} {'Merge(us)':>12} {'Unmerge(us)':>12} {'Merged vs Unmerged':>18}")
    print("-"*80)

    for r in results:
        merged_speedup = r.lora_forward_us / r.merged_forward_us
        print(
            f"{r.hidden_size:>8} "
            f"{r.rank:>6} "
            f"{r.merge_time_us:>12.1f} "
            f"{r.unmerge_time_us:>12.1f} "
            f"{merged_speedup:>17.2f}x"
        )


def save_results(results: List[LoRABenchmarkResult], output_path: str):
    """Save results to JSON."""
    data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Operation Benchmarks")
    parser.add_argument("--hidden-sizes", type=str, default="768,1024,2048,4096",
                       help="Comma-separated hidden sizes")
    parser.add_argument("--ranks", type=str, default="4,8,16,32",
                       help="Comma-separated LoRA ranks")
    parser.add_argument("--model", type=str, help="Model name for full-model benchmark")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("LoRA Benchmark Suite")
    print("="*70)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]

    # Run sweep
    results = run_sweep(hidden_sizes, ranks, device)
    print_sweep_results(results)

    if args.detailed:
        print_merge_analysis(results)

    # Model-level benchmark
    if args.model:
        print("\n" + "="*70)
        print(f"Full Model Benchmark: {args.model}")
        print("="*70)
        model_results = benchmark_model_lora(args.model, ranks, device)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
