"""
Unit tests for custom LoRA implementation.

Tests:
- LoRALinear initialization and forward pass
- Merge/unmerge functionality
- Parameter freezing
- Model application
- Save/load weights
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora_to_model,
    count_parameters,
    get_lora_params,
    merge_lora_weights,
    unmerge_lora_weights,
    save_lora_weights,
    load_lora_weights,
)


class TestLoRALinear:
    """Tests for LoRALinear layer."""

    def test_initialization(self):
        """Test LoRALinear initializes correctly."""
        layer = LoRALinear(in_features=64, out_features=64, r=4, alpha=8)

        # Check dimensions
        assert layer.lora_A.shape == (4, 64)
        assert layer.lora_B.shape == (64, 4)
        assert layer.weight.shape == (64, 64)

        # Check B is initialized to zero
        assert torch.allclose(layer.lora_B, torch.zeros_like(layer.lora_B))

        # Check scaling
        assert layer.scaling == 8 / 4  # alpha / r

    def test_forward_unchanged_at_init(self, device):
        """Test that forward pass equals base linear at initialization."""
        base = nn.Linear(64, 64)
        lora = LoRALinear.from_linear(base, r=4, alpha=8)

        base = base.to(device)
        lora = lora.to(device)

        x = torch.randn(2, 10, 64, device=device)

        base_out = base(x)
        lora_out = lora(x)

        # Should be identical since B is zero
        assert torch.allclose(base_out, lora_out, atol=1e-5)

    def test_forward_after_training(self, device):
        """Test forward pass changes after LoRA parameters are modified."""
        base = nn.Linear(64, 64)
        lora = LoRALinear.from_linear(base, r=4, alpha=8).to(device)

        x = torch.randn(2, 10, 64, device=device)

        # Get initial output
        out_before = lora(x).clone()

        # Modify LoRA weights
        with torch.no_grad():
            lora.lora_A.fill_(0.1)
            lora.lora_B.fill_(0.1)

        # Output should change
        out_after = lora(x)
        assert not torch.allclose(out_before, out_after)

    def test_merge_unmerge(self, device):
        """Test that merge/unmerge preserves outputs."""
        base = nn.Linear(64, 64)
        lora = LoRALinear.from_linear(base, r=4, alpha=8).to(device)

        # Train LoRA (simulate by setting weights)
        with torch.no_grad():
            lora.lora_A.normal_(0, 0.1)
            lora.lora_B.normal_(0, 0.1)

        x = torch.randn(2, 10, 64, device=device)

        # Get unmerged output
        out_unmerged = lora(x)

        # Merge and check output
        lora.merge()
        assert lora.merged
        out_merged = lora(x)
        assert torch.allclose(out_unmerged, out_merged, atol=1e-5)

        # Unmerge and check output
        lora.unmerge()
        assert not lora.merged
        out_unmerged_again = lora(x)
        assert torch.allclose(out_unmerged, out_unmerged_again, atol=1e-5)

    def test_gradient_flow(self, device):
        """Test that gradients flow through LoRA parameters."""
        lora = LoRALinear(64, 64, r=4, alpha=8).to(device)

        x = torch.randn(2, 10, 64, device=device)
        target = torch.randn(2, 10, 64, device=device)

        out = lora(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()

        # LoRA params should have gradients
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None

        # Base weight should NOT have gradients (frozen)
        assert lora.weight.grad is None

    def test_from_linear_preserves_weights(self, device):
        """Test that from_linear correctly copies weights."""
        base = nn.Linear(64, 64)
        base.weight.data.fill_(0.5)
        base.bias.data.fill_(0.1)

        lora = LoRALinear.from_linear(base, r=4)

        assert torch.allclose(lora.weight, base.weight)
        assert torch.allclose(lora.bias, base.bias)


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoRAConfig()
        assert config.r == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.05
        assert config.scaling == 2.0  # 16 / 8

    def test_default_target_modules(self):
        """Test default target modules are set."""
        config = LoRAConfig()
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules

    def test_custom_target_modules(self):
        """Test custom target modules."""
        config = LoRAConfig(target_modules=["custom_layer"])
        assert config.target_modules == ["custom_layer"]


class TestApplyLoRA:
    """Tests for apply_lora_to_model function."""

    def test_apply_to_simple_model(self, device):
        """Test LoRA application to a simple model."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        model[0]._modules_names = "q_proj"  # Hack for testing

        # Create model with named modules
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 128)
                self.relu = nn.ReLU()
                self.v_proj = nn.Linear(128, 64)

            def forward(self, x):
                return self.v_proj(self.relu(self.q_proj(x)))

        model = SimpleModel().to(device)
        config = LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj"])

        model = apply_lora_to_model(model, config, verbose=False)

        # Check LoRA was applied
        assert isinstance(model.q_proj, LoRALinear)
        assert isinstance(model.v_proj, LoRALinear)
        assert not isinstance(model.relu, LoRALinear)

    def test_parameter_freezing(self, device):
        """Test that non-LoRA parameters are frozen."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.other = nn.Linear(64, 64)

        model = SimpleModel().to(device)
        config = LoRAConfig(r=4, target_modules=["q_proj"])
        model = apply_lora_to_model(model, config, verbose=False)

        lora_params, other_params = get_lora_params(model)

        # LoRA params should require grad
        for p in lora_params:
            assert p.requires_grad

        # Other params should be frozen
        for p in other_params:
            assert not p.requires_grad

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_apply_to_gpt2(self):
        """Test LoRA application to GPT-2."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        config = LoRAConfig(r=8, alpha=16, target_modules=["c_attn", "c_proj"])

        model = apply_lora_to_model(model, config, verbose=False)

        stats = count_parameters(model)

        # Should have ~0.3% trainable params
        assert stats["trainable_percent"] < 1.0
        assert stats["trainable"] > 0
        assert stats["lora"] > 0


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_frozen_and_trainable(self):
        """Test parameter counting."""
        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = nn.Linear(64, 64)
                self.lora_A = nn.Parameter(torch.randn(4, 64))
                self.lora_B = nn.Parameter(torch.randn(64, 4))

        model = MixedModel()
        model.frozen.weight.requires_grad = False
        model.frozen.bias.requires_grad = False

        stats = count_parameters(model)

        assert stats["total"] == 64*64 + 64 + 4*64 + 64*4
        assert stats["trainable"] == 4*64 + 64*4  # Only LoRA params
        assert stats["lora"] == 4*64 + 64*4
        assert stats["frozen"] == 64*64 + 64


class TestSaveLoadLoRA:
    """Tests for save/load LoRA weights."""

    def test_save_load_weights(self, device):
        """Test saving and loading LoRA weights."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)

        model = SimpleModel().to(device)
        config = LoRAConfig(r=4, target_modules=["q_proj"])
        model = apply_lora_to_model(model, config, verbose=False)

        # Set known weights
        with torch.no_grad():
            model.q_proj.lora_A.fill_(0.5)
            model.q_proj.lora_B.fill_(0.25)

        # Save weights
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        save_lora_weights(model, path)

        # Create fresh model and load weights
        model2 = SimpleModel().to(device)
        model2 = apply_lora_to_model(model2, config, verbose=False)

        load_lora_weights(model2, path)

        # Verify weights match
        assert torch.allclose(model.q_proj.lora_A, model2.q_proj.lora_A)
        assert torch.allclose(model.q_proj.lora_B, model2.q_proj.lora_B)

        os.unlink(path)

    def test_saved_file_size(self, device):
        """Test that LoRA weights file is small."""
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(1024, 1024)  # 4MB weight

        model = LargeModel().to(device)
        config = LoRAConfig(r=8, target_modules=["q_proj"])
        model = apply_lora_to_model(model, config, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        save_lora_weights(model, path)

        # LoRA weights should be much smaller than full model
        file_size_kb = os.path.getsize(path) / 1024
        assert file_size_kb < 100  # Should be < 100KB for r=8, 1024-dim

        os.unlink(path)


class TestMergeUnmerge:
    """Tests for global merge/unmerge functions."""

    def test_merge_all_layers(self, device):
        """Test merging all LoRA layers in a model."""
        class MultiLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)

        model = MultiLayerModel().to(device)
        config = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        model = apply_lora_to_model(model, config, verbose=False)

        # Set weights
        with torch.no_grad():
            model.q_proj.lora_A.normal_(0, 0.1)
            model.q_proj.lora_B.normal_(0, 0.1)
            model.v_proj.lora_A.normal_(0, 0.1)
            model.v_proj.lora_B.normal_(0, 0.1)

        # Verify not merged initially
        assert not model.q_proj.merged
        assert not model.v_proj.merged

        # Merge all
        merge_lora_weights(model)
        assert model.q_proj.merged
        assert model.v_proj.merged

        # Unmerge all
        unmerge_lora_weights(model)
        assert not model.q_proj.merged
        assert not model.v_proj.merged
