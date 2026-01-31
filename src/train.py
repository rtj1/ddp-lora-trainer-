"""
Distributed LLM Training with LoRA
==================================

This module provides a production-ready training loop with:
- DDP, FSDP, and DeepSpeed backends
- Gradient checkpointing for memory efficiency
- Evaluation loop with perplexity calculation
- Memory profiling and throughput tracking
- Proper checkpoint save/resume
"""

import argparse
import os
import time
from typing import Dict, Optional, Tuple
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)

try:
    import wandb
except ImportError:
    wandb = None

try:
    import deepspeed
except ImportError:
    deepspeed = None

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision
except ImportError:
    FSDP = None
    ShardingStrategy = None
    MixedPrecision = None

from .utils import (
    load_config,
    override_from_kv,
    set_seed,
    is_main_process,
    barrier,
    ThroughputMeter,
)
from .dataset import build_dataloader
from .lora_utils import apply_lora, save_lora_weights, count_parameters


# =============================================================================
# Memory Tracking
# =============================================================================

class MemoryTracker:
    """Track GPU memory usage during training."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.enabled = device == "cuda" and torch.cuda.is_available()
        self.peak_allocated = 0
        self.peak_reserved = 0

    def update(self):
        if not self.enabled:
            return
        self.peak_allocated = max(
            self.peak_allocated,
            torch.cuda.max_memory_allocated() / (1024**3),
        )
        self.peak_reserved = max(
            self.peak_reserved,
            torch.cuda.max_memory_reserved() / (1024**3),
        )

    def reset(self):
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()

    def get_stats(self) -> Dict[str, float]:
        if not self.enabled:
            return {}
        return {
            "gpu_mem_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "gpu_peak_allocated_gb": self.peak_allocated,
            "gpu_peak_reserved_gb": self.peak_reserved,
        }


# =============================================================================
# Gradient Checkpointing
# =============================================================================

def enable_gradient_checkpointing(model: torch.nn.Module, enabled: bool = True):
    """
    Enable gradient checkpointing to reduce memory usage.

    Trade-off: ~30% slower training but ~50% less memory.
    This allows training larger models or with larger batch sizes.
    """
    if not enabled:
        return

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if is_main_process():
            print("Gradient checkpointing enabled")
    else:
        if is_main_process():
            print("Warning: Model does not support gradient checkpointing")


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    eval_loader,
    device: str,
    max_batches: int = 50,
) -> Dict[str, float]:
    """
    Run evaluation and compute metrics.

    Returns:
        Dictionary with 'eval_loss' and 'perplexity'
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for batch in eval_loader:
        if num_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Count actual tokens (not padding)
        num_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens
        num_batches += 1

    model.train()

    if total_tokens == 0:
        return {"eval_loss": float("inf"), "perplexity": float("inf")}

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
    }


# =============================================================================
# Distributed Setup
# =============================================================================

def init_distributed() -> Tuple[int, int, int]:
    """Initialize distributed training if environment variables are set."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local)

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

        return rank, world, local

    return 0, 1, 0


# =============================================================================
# Model Setup
# =============================================================================

def setup_model(cfg: dict, device: str):
    """
    Load and configure the model.

    Returns:
        (model, tokenizer)
    """
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(cfg["model"].get("dtype", "bf16"), torch.bfloat16)

    # Model kwargs
    model_kwargs = {}

    # QLoRA (4-bit quantization)
    quant_cfg = cfg.get("quantization", {})
    if quant_cfg.get("load_in_4bit", False):
        compute_dtype = dtype_map.get(quant_cfg.get("compute_dtype", "bf16"), torch.bfloat16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = bnb_config
    else:
        if device == "cuda":
            model_kwargs["torch_dtype"] = dtype

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        **model_kwargs,
    )

    # Apply LoRA
    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("enabled", True):
        model = apply_lora(
            model,
            target_modules=lora_cfg.get("target_modules", []),
            r=lora_cfg.get("r", 8),
            alpha=lora_cfg.get("alpha", 16),
            dropout=lora_cfg.get("dropout", 0.05),
        )

    # Gradient checkpointing
    enable_gradient_checkpointing(
        model,
        enabled=cfg.get("training", {}).get("gradient_checkpointing", False),
    )

    return model, tokenizer


def wrap_distributed(
    model: torch.nn.Module,
    cfg: dict,
    device: str,
    local_rank: int,
    world_size: int,
) -> torch.nn.Module:
    """Wrap model for distributed training."""
    backend = cfg.get("distributed", {}).get("backend", "ddp")

    model.to(device)

    if backend == "fsdp" and FSDP is not None:
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        fsdp_cfg = cfg.get("fsdp", {})
        mp_dtype = dtype_map.get(fsdp_cfg.get("mixed_precision", "bf16"), torch.bfloat16)
        mp = MixedPrecision(
            param_dtype=mp_dtype,
            reduce_dtype=mp_dtype,
            buffer_dtype=mp_dtype,
        )
        strat_map = {
            "full": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "hybrid": ShardingStrategy.HYBRID_SHARD,
        }
        sharding = strat_map.get(
            fsdp_cfg.get("sharding_strategy", "full"),
            ShardingStrategy.FULL_SHARD,
        )
        model = FSDP(
            model,
            mixed_precision=mp,
            sharding_strategy=sharding,
            use_orig_params=True,
            device_id=local_rank if device == "cuda" else None,
        )

    elif backend == "ddp" and world_size > 1:
        if device == "cuda":
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        else:
            model = DDP(model)

    return model


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    scaler,
    step: int,
    cfg: dict,
    tokenizer,
    memory_tracker: MemoryTracker,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Get unwrapped model
    unwrapped = model.module if hasattr(model, "module") else model

    # Save training state
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler and scaler.is_enabled() else None,
        "config": cfg,
        "memory_stats": memory_tracker.get_stats(),
    }
    torch.save(state, path)

    # Save LoRA weights separately (much smaller)
    if cfg.get("checkpoint", {}).get("save_lora_only", True):
        lora_path = os.path.join(os.path.dirname(path), "lora_weights.pt")
        save_lora_weights(unwrapped, lora_path)
    else:
        full_dir = os.path.join(os.path.dirname(path), "full_model")
        unwrapped.save_pretrained(full_dir)
        tokenizer.save_pretrained(full_dir)

    if is_main_process():
        print(f"Saved checkpoint at step {step} to {path}")


def load_checkpoint(
    path: str,
    optimizer,
    scheduler,
    scaler,
) -> int:
    """Load checkpoint and return the step number."""
    ckpt = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("scaler") and scaler and scaler.is_enabled():
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0)


# =============================================================================
# W&B Logging
# =============================================================================

def init_wandb(cfg: dict) -> bool:
    """Initialize Weights & Biases if enabled."""
    if wandb is None:
        return False

    wb_cfg = cfg.get("logging", {}).get("wandb", {})
    if not wb_cfg.get("enabled", True):
        return False

    if os.environ.get("WANDB_DISABLED"):
        return False

    if not is_main_process():
        return False

    wandb.init(
        project=wb_cfg.get("project", "ddp-lora-trainer"),
        entity=wb_cfg.get("entity") or None,
        name=wb_cfg.get("run_name"),
        config=cfg,
        resume="allow",
    )
    return True


# =============================================================================
# Main Training Loop
# =============================================================================

def train(cfg: dict):
    """Main training function."""
    set_seed(cfg.get("seed", 42))

    # Distributed setup
    rank, world, local = init_distributed()
    device = "cuda" if torch.cuda.is_available() and cfg["device"].get("cuda_if_available", True) else "cpu"

    # Model and tokenizer
    model, tokenizer = setup_model(cfg, device)

    # Distributed wrapping
    model = wrap_distributed(model, cfg, device, local, world)

    # Data
    train_cfg = cfg["training"]
    train_loader, _ = build_dataloader(
        tokenizer,
        cfg["data"],
        train_cfg["batch_size_per_device"],
        rank,
        world,
    )

    # Eval loader (optional)
    eval_loader = None
    if cfg["data"].get("eval_file") or cfg["data"].get("use_hf_streaming"):
        eval_cfg = dict(cfg["data"])
        if cfg["data"].get("eval_file"):
            eval_cfg["local_file"] = cfg["data"]["eval_file"]
        eval_loader, _ = build_dataloader(
            tokenizer,
            eval_cfg,
            train_cfg["batch_size_per_device"],
            rank,
            world,
            shuffle=False,
        )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=train_cfg["lr"],
        betas=(train_cfg["adam_beta1"], train_cfg["adam_beta2"]),
        eps=train_cfg["adam_eps"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg["warmup_steps"],
        num_training_steps=train_cfg["max_steps"],
    )

    # DeepSpeed init (must be after optimizer creation)
    backend = cfg.get("distributed", {}).get("backend", "ddp")
    if backend == "deepspeed" and deepspeed is not None:
        ds_config_path = cfg.get("distributed", {}).get("deepspeed_config", "./deepspeed/zero2.json")
        import json
        with open(ds_config_path) as f:
            ds_config = json.load(f)
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=trainable_params,
            config=ds_config,
            optimizer=optimizer,
            lr_scheduler=scheduler,
        )

    # Mixed precision scaler
    use_fp16 = cfg["model"].get("dtype") == "fp16" and device == "cuda" and backend != "deepspeed"
    scaler = GradScaler(enabled=use_fp16)

    # Resume from checkpoint
    step = 0
    resume_path = cfg.get("checkpoint", {}).get("resume_from")
    if resume_path and os.path.exists(resume_path):
        if is_main_process():
            print(f"Resuming from checkpoint: {resume_path}")
        step = load_checkpoint(resume_path, optimizer, scheduler, scaler)

    # Logging
    use_wandb = init_wandb(cfg)

    # Tracking
    memory_tracker = MemoryTracker(device)
    throughput = ThroughputMeter()
    t0 = time.time()

    # Training loop
    model.train()
    running_loss = 0.0
    grad_accum_steps = train_cfg["grad_accum_steps"]

    if is_main_process():
        param_stats = count_parameters(model.module if hasattr(model, "module") else model)
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"Total parameters: {param_stats['total']:,}")
        print(f"Trainable: {param_stats['trainable']:,} ({param_stats['trainable_percent']:.2f}%)")
        print(f"Max steps: {train_cfg['max_steps']}")
        print(f"Batch size per device: {train_cfg['batch_size_per_device']}")
        print(f"Gradient accumulation: {grad_accum_steps}")
        print(f"Effective batch size: {train_cfg['batch_size_per_device'] * grad_accum_steps * world}")
        print(f"{'='*60}\n")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    amp_dtype = dtype_map.get(cfg["model"].get("dtype", "bf16"), torch.bfloat16)
    use_amp = device == "cuda" and cfg["model"].get("dtype") in ["bf16", "fp16"]

    for epoch in range(10**9):  # Run until max_steps
        for batch in train_loader:
            # Move to device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Forward pass
            with autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum_steps

            # Backward pass
            if backend == "deepspeed":
                model.backward(loss)
            elif scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()

            # Optimizer step (with gradient accumulation)
            if (step + 1) % grad_accum_steps == 0:
                # Gradient clipping
                if train_cfg.get("max_grad_norm"):
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        train_cfg["max_grad_norm"],
                    )

                # Step
                if backend == "deepspeed":
                    model.step()
                elif scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # Memory tracking
            memory_tracker.update()

            # Logging
            if step % train_cfg["log_interval"] == 0 and is_main_process():
                dt = max(time.time() - t0, 1e-8)
                num_tokens = int(attention_mask.sum().item())
                if dist.is_initialized():
                    num_tokens *= world
                throughput.update(tokens=num_tokens, dt=dt)
                t0 = time.time()

                avg_loss = running_loss * grad_accum_steps
                running_loss = 0.0

                log_dict = {
                    "step": step,
                    "loss": avg_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_per_sec": throughput.avg_toks_per_sec(),
                    **memory_tracker.get_stats(),
                }

                print(
                    f"[step {step:>6}] "
                    f"loss={avg_loss:.4f} "
                    f"lr={log_dict['lr']:.2e} "
                    f"toks/s={log_dict['tokens_per_sec']:.0f} "
                    f"mem={log_dict.get('gpu_mem_allocated_gb', 0):.1f}GB"
                )

                if use_wandb:
                    wandb.log(log_dict, step=step)

            # Evaluation
            if (
                train_cfg.get("eval_interval")
                and step > 0
                and step % train_cfg["eval_interval"] == 0
                and eval_loader is not None
            ):
                if is_main_process():
                    print(f"\nRunning evaluation at step {step}...")
                eval_results = evaluate(model, eval_loader, device)
                if is_main_process():
                    print(f"  Eval loss: {eval_results['eval_loss']:.4f}")
                    print(f"  Perplexity: {eval_results['perplexity']:.2f}\n")
                    if use_wandb:
                        wandb.log(eval_results, step=step)

            # Checkpointing
            if (
                train_cfg.get("save_interval")
                and step > 0
                and step % train_cfg["save_interval"] == 0
                and is_main_process()
            ):
                ckpt_path = os.path.join(
                    cfg["checkpoint"]["output_dir"],
                    f"checkpoint-{step}",
                    "training_state.pt",
                )
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    cfg,
                    tokenizer,
                    memory_tracker,
                )

            step += 1
            if step >= train_cfg["max_steps"]:
                break

        if step >= train_cfg["max_steps"]:
            break

    # Final save
    if is_main_process():
        print("\nTraining complete!")
        final_path = os.path.join(
            cfg["checkpoint"]["output_dir"],
            "final",
            "training_state.pt",
        )
        save_checkpoint(
            final_path,
            model,
            optimizer,
            scheduler,
            scaler,
            step,
            cfg,
            tokenizer,
            memory_tracker,
        )

        # Final memory stats
        mem_stats = memory_tracker.get_stats()
        if mem_stats:
            print(f"\nPeak GPU memory: {mem_stats.get('gpu_peak_allocated_gb', 0):.2f} GB allocated")

    # Cleanup
    if dist.is_initialized():
        barrier()
        dist.destroy_process_group()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed LoRA Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = override_from_kv(cfg, args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
