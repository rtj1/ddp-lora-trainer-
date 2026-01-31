"""
Pytest fixtures for ddp-lora-trainer tests.
"""

import os
import sys
import tempfile

import pytest
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def device():
    """Return available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def small_model_config():
    """Configuration for a small test model."""
    return {
        "model": {
            "name": "gpt2",
            "tokenizer_name": "gpt2",
            "dtype": "fp32",
        },
        "lora": {
            "enabled": True,
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["c_attn", "c_proj"],
        },
        "data": {
            "block_size": 128,
            "collate_mode": "pad",
        },
        "training": {
            "batch_size_per_device": 2,
            "lr": 1e-4,
            "max_steps": 10,
            "warmup_steps": 2,
            "grad_accum_steps": 1,
            "log_interval": 5,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,
            "weight_decay": 0.01,
            "gradient_checkpointing": False,
        },
        "device": {
            "cuda_if_available": True,
        },
        "checkpoint": {
            "output_dir": tempfile.mkdtemp(),
            "save_lora_only": True,
        },
        "seed": 42,
    }


@pytest.fixture
def sample_texts():
    """Sample text data for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Python is a versatile programming language.",
        "Neural networks learn patterns from data.",
        "The transformer architecture revolutionized NLP.",
        "Deep learning requires large amounts of data.",
        "Gradient descent optimizes model parameters.",
        "Attention mechanisms capture long-range dependencies.",
    ]


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for fast testing."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def temp_data_file(sample_texts):
    """Create a temporary data file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for text in sample_texts:
            f.write(text + "\n")
        return f.name


@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Clean up CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
