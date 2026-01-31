"""
Integration tests for training functionality.

Tests:
- Full training loop
- Gradient checkpointing
- Evaluation loop
- Memory tracking
- Checkpoint save/load
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import (
    MemoryTracker,
    enable_gradient_checkpointing,
    evaluate,
    setup_model,
    save_checkpoint,
    load_checkpoint,
)
from src.lora import apply_lora_to_model, LoRAConfig


class TestMemoryTracker:
    """Tests for MemoryTracker class."""

    def test_initialization(self):
        """Test memory tracker initializes correctly."""
        tracker = MemoryTracker("cuda" if torch.cuda.is_available() else "cpu")
        assert tracker.peak_allocated == 0
        assert tracker.peak_reserved == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_tracks_allocations(self):
        """Test memory tracker captures allocations."""
        tracker = MemoryTracker("cuda")
        tracker.reset()

        # Allocate some memory
        tensors = [torch.randn(1000, 1000, device="cuda") for _ in range(5)]
        tracker.update()

        stats = tracker.get_stats()
        assert stats["gpu_mem_allocated_gb"] > 0
        assert stats["gpu_peak_allocated_gb"] > 0

        del tensors
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_peak_tracking(self):
        """Test peak memory is tracked correctly."""
        tracker = MemoryTracker("cuda")
        tracker.reset()

        # Create and delete tensors
        t1 = torch.randn(1000, 1000, device="cuda")
        tracker.update()
        peak1 = tracker.peak_allocated

        t2 = torch.randn(2000, 2000, device="cuda")
        tracker.update()
        peak2 = tracker.peak_allocated

        del t2
        torch.cuda.empty_cache()
        tracker.update()

        # Peak should be from when t2 existed
        assert peak2 > peak1

    def test_cpu_mode(self):
        """Test tracker works on CPU (returns empty stats)."""
        tracker = MemoryTracker("cpu")
        stats = tracker.get_stats()
        assert stats == {}


class TestGradientCheckpointing:
    """Tests for gradient checkpointing."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_enables_on_supported_model(self):
        """Test gradient checkpointing enables on supported models."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        enable_gradient_checkpointing(model, enabled=True)

        # Check it's enabled
        assert model.is_gradient_checkpointing

    def test_handles_unsupported_model(self):
        """Test graceful handling of unsupported models."""
        model = nn.Linear(64, 64)

        # Should not raise
        enable_gradient_checkpointing(model, enabled=True)

    def test_disabled_mode(self):
        """Test disabled mode doesn't change model."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        initial_state = model.is_gradient_checkpointing if hasattr(model, "is_gradient_checkpointing") else False

        enable_gradient_checkpointing(model, enabled=False)

        # State should be unchanged
        current_state = model.is_gradient_checkpointing if hasattr(model, "is_gradient_checkpointing") else False
        assert current_state == initial_state


class TestEvaluate:
    """Tests for evaluation function."""

    def test_returns_correct_metrics(self, device):
        """Test evaluation returns loss and perplexity."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from torch.utils.data import DataLoader, TensorDataset

        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Create fake eval data
        input_ids = torch.randint(0, 1000, (10, 64))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        dataset = TensorDataset(input_ids, attention_mask, labels)

        def collate_fn(batch):
            ids, mask, lab = zip(*batch)
            return {
                "input_ids": torch.stack(ids),
                "attention_mask": torch.stack(mask),
                "labels": torch.stack(lab),
            }

        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        results = evaluate(model, loader, device, max_batches=3)

        assert "eval_loss" in results
        assert "perplexity" in results
        assert results["eval_loss"] >= 0
        assert results["perplexity"] >= 1.0

    def test_handles_empty_loader(self, device):
        """Test evaluation handles empty data gracefully."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        empty_loader = []

        results = evaluate(model, iter(empty_loader), device)

        assert results["eval_loss"] == float("inf")
        assert results["perplexity"] == float("inf")

    def test_respects_max_batches(self, device):
        """Test max_batches limit is respected."""
        from transformers import AutoModelForCausalLM
        from torch.utils.data import DataLoader, TensorDataset

        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

        # Create more data than max_batches
        input_ids = torch.randint(0, 1000, (20, 32))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        dataset = TensorDataset(input_ids, attention_mask, labels)

        def collate_fn(batch):
            ids, mask, lab = zip(*batch)
            return {
                "input_ids": torch.stack(ids),
                "attention_mask": torch.stack(mask),
                "labels": torch.stack(lab),
            }

        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        # Should only process 3 batches
        results = evaluate(model, loader, device, max_batches=3)

        assert results["eval_loss"] > 0


class TestCheckpointing:
    """Tests for checkpoint save/load."""

    def test_save_load_training_state(self, device):
        """Test saving and loading training state."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from torch.optim import AdamW
        from transformers import get_cosine_schedule_with_warmup
        from torch.cuda.amp import GradScaler

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        config = LoRAConfig(r=4, target_modules=["c_attn"])
        model = apply_lora_to_model(model, config, verbose=False)
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Setup optimizer and scheduler
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=1e-4)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 10, 100)
        scaler = GradScaler(enabled=False)

        # Step a few times
        for _ in range(5):
            scheduler.step()

        memory_tracker = MemoryTracker(device)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ckpt", "training_state.pt")

            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler,
                scaler,
                step=42,
                cfg={"test": True},
                tokenizer=tokenizer,
                memory_tracker=memory_tracker,
            )

            assert os.path.exists(ckpt_path)

            # Load checkpoint
            new_optimizer = AdamW(trainable_params, lr=1e-4)
            new_scheduler = get_cosine_schedule_with_warmup(new_optimizer, 10, 100)
            new_scaler = GradScaler(enabled=False)

            step = load_checkpoint(ckpt_path, new_optimizer, new_scheduler, new_scaler)

            assert step == 42

    def test_saves_lora_weights_separately(self, device):
        """Test LoRA weights are saved separately."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from torch.optim import AdamW
        from transformers import get_cosine_schedule_with_warmup
        from torch.cuda.amp import GradScaler

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        config = LoRAConfig(r=4, target_modules=["c_attn"])
        model = apply_lora_to_model(model, config, verbose=False)
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=1e-4)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 10, 100)
        scaler = GradScaler(enabled=False)
        memory_tracker = MemoryTracker(device)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ckpt", "training_state.pt")

            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler,
                scaler,
                step=10,
                cfg={"checkpoint": {"save_lora_only": True}},
                tokenizer=tokenizer,
                memory_tracker=memory_tracker,
            )

            # Check LoRA weights file exists
            lora_path = os.path.join(tmpdir, "ckpt", "lora_weights.pt")
            assert os.path.exists(lora_path)

            # LoRA weights should be small
            lora_size = os.path.getsize(lora_path)
            assert lora_size < 1024 * 1024  # Less than 1MB


class TestTrainingLoop:
    """Integration tests for training loop components."""

    def test_single_training_step(self, device):
        """Test a single training step works correctly."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        config = LoRAConfig(r=4, target_modules=["c_attn"])
        model = apply_lora_to_model(model, config, verbose=False)
        model.to(device)
        model.train()

        # Setup
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

        # Fake batch
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        # Forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        assert loss.item() > 0

        # Backward
        loss.backward()

        # Check gradients exist for LoRA params
        for name, param in model.named_parameters():
            if "lora_" in name:
                assert param.grad is not None, f"No gradient for {name}"

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

    def test_loss_decreases(self, device):
        """Test that loss decreases over training steps."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        config = LoRAConfig(r=4, target_modules=["c_attn"])
        model = apply_lora_to_model(model, config, verbose=False)
        model.to(device)
        model.train()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

        # Fixed batch for consistent comparison
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1000, (4, 32), device=device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        losses = []
        for _ in range(10):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Loss should generally decrease (may not be monotonic)
        assert losses[-1] < losses[0], "Loss should decrease over training"


class TestSetupModel:
    """Tests for setup_model function."""

    def test_loads_model_and_tokenizer(self, small_model_config, device):
        """Test model and tokenizer loading."""
        model, tokenizer = setup_model(small_model_config, device)

        assert model is not None
        assert tokenizer is not None
        assert tokenizer.pad_token is not None

    def test_applies_lora(self, small_model_config, device):
        """Test LoRA is applied when enabled."""
        small_model_config["lora"]["enabled"] = True

        model, _ = setup_model(small_model_config, device)

        # Check some params are trainable (LoRA)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        assert trainable > 0
        assert frozen > trainable  # Most params should be frozen

    def test_no_lora_when_disabled(self, small_model_config, device):
        """Test no LoRA when disabled."""
        small_model_config["lora"]["enabled"] = False

        model, _ = setup_model(small_model_config, device)

        # All params should be trainable
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        total = sum(1 for p in model.parameters())

        assert trainable == total
