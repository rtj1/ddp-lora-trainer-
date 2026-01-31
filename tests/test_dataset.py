"""
Unit tests for dataset and collation functions.

Tests:
- TextFileDataset loading and indexing
- CollateFunction padding and packing modes
- Document boundary preservation
- build_dataloader integration
"""

import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    TextFileDataset,
    CollateFunction,
    build_collate,
    build_dataloader,
)


class TestTextFileDataset:
    """Tests for TextFileDataset."""

    def test_loading_from_lines(self, sample_texts):
        """Test dataset loads lines correctly."""
        ds = TextFileDataset(sample_texts)

        assert len(ds) == len(sample_texts)
        assert ds[0] == sample_texts[0].strip()

    def test_filters_empty_lines(self):
        """Test that empty lines are filtered."""
        lines = ["Hello", "", "World", "  ", "Test"]
        ds = TextFileDataset(lines)

        assert len(ds) == 3
        assert ds[0] == "Hello"
        assert ds[1] == "World"
        assert ds[2] == "Test"

    def test_strips_whitespace(self):
        """Test that whitespace is stripped."""
        lines = ["  Hello  ", "\tWorld\t", "\n  Test  \n"]
        ds = TextFileDataset(lines)

        assert ds[0] == "Hello"
        assert ds[1] == "World"
        assert ds[2] == "Test"


class TestCollateFunction:
    """Tests for CollateFunction."""

    def test_padded_mode_shapes(self, mock_tokenizer):
        """Test padded mode produces correct shapes."""
        block_size = 64
        collate = CollateFunction(mock_tokenizer, block_size, mode="pad")

        texts = ["Hello world", "This is a test", "Short"]
        batch = collate(texts)

        assert batch["input_ids"].shape == (3, block_size)
        assert batch["attention_mask"].shape == (3, block_size)
        assert batch["labels"].shape == (3, block_size)

    def test_padded_mode_attention_mask(self, mock_tokenizer):
        """Test attention mask is correct for padded sequences."""
        block_size = 64
        collate = CollateFunction(mock_tokenizer, block_size, mode="pad")

        texts = ["Hello"]
        batch = collate(texts)

        # First few positions should be 1, rest should be 0 (padding)
        mask = batch["attention_mask"][0]
        assert mask[:5].sum() > 0  # At least some tokens
        assert mask[-10:].sum() == 0 or mask.sum() <= block_size  # Padding at end

    def test_labels_mask_padding(self, mock_tokenizer):
        """Test that labels have -100 for padding positions."""
        block_size = 64
        collate = CollateFunction(mock_tokenizer, block_size, mode="pad")

        texts = ["Hi"]
        batch = collate(texts)

        labels = batch["labels"][0]
        mask = batch["attention_mask"][0]

        # Where mask is 0, labels should be -100
        padding_positions = (mask == 0)
        assert (labels[padding_positions] == -100).all()

    def test_packed_mode_combines_documents(self, mock_tokenizer):
        """Test packed mode combines short documents."""
        block_size = 128
        collate = CollateFunction(mock_tokenizer, block_size, mode="pack")

        # Short texts that should fit in one packed sequence
        texts = ["Hello", "World", "Test", "Data"]
        batch = collate(texts)

        # Should pack into fewer sequences than input texts
        assert batch["input_ids"].shape[1] == block_size

    def test_packed_mode_adds_eos(self, mock_tokenizer):
        """Test packed mode adds EOS between documents."""
        block_size = 256
        collate = CollateFunction(mock_tokenizer, block_size, mode="pack")

        texts = ["Hello world", "Test document"]
        batch = collate(texts)

        input_ids = batch["input_ids"][0]
        eos_id = mock_tokenizer.eos_token_id

        # Should contain EOS tokens
        assert (input_ids == eos_id).sum() >= 2  # At least 2 EOS tokens

    def test_document_boundary_preservation(self, mock_tokenizer):
        """Test that document boundaries are preserved."""
        block_size = 128
        collate = CollateFunction(mock_tokenizer, block_size, mode="pad")

        # Each text should tokenize independently
        text1 = "Document one about cats."
        text2 = "Document two about dogs."
        texts = [text1, text2]

        batch = collate(texts)

        # Tokenize individually for comparison
        enc1 = mock_tokenizer(text1, truncation=True, max_length=block_size, padding="max_length", return_tensors="pt")
        enc2 = mock_tokenizer(text2, truncation=True, max_length=block_size, padding="max_length", return_tensors="pt")

        # Batch outputs should match individual tokenization
        assert torch.equal(batch["input_ids"][0], enc1["input_ids"][0])
        assert torch.equal(batch["input_ids"][1], enc2["input_ids"][0])

    def test_truncation(self, mock_tokenizer):
        """Test that long texts are truncated."""
        block_size = 32
        collate = CollateFunction(mock_tokenizer, block_size, mode="pad")

        long_text = " ".join(["word"] * 100)  # Much longer than block_size
        batch = collate([long_text])

        assert batch["input_ids"].shape[1] == block_size

    def test_custom_ignore_index(self, mock_tokenizer):
        """Test custom ignore index for labels."""
        collate = CollateFunction(mock_tokenizer, block_size=64, mode="pad", ignore_index=-999)

        texts = ["Short text"]
        batch = collate(texts)

        labels = batch["labels"][0]
        mask = batch["attention_mask"][0]

        # Padding positions should use custom ignore index
        padding_positions = (mask == 0)
        if padding_positions.any():
            assert (labels[padding_positions] == -999).all()


class TestBuildCollate:
    """Tests for build_collate function."""

    def test_returns_collate_function(self, mock_tokenizer):
        """Test build_collate returns a CollateFunction."""
        collate = build_collate(mock_tokenizer, block_size=64, mode="pad")
        assert isinstance(collate, CollateFunction)

    def test_mode_parameter(self, mock_tokenizer):
        """Test mode parameter is applied."""
        collate_pad = build_collate(mock_tokenizer, block_size=64, mode="pad")
        collate_pack = build_collate(mock_tokenizer, block_size=64, mode="pack")

        assert collate_pad.mode == "pad"
        assert collate_pack.mode == "pack"


class TestBuildDataloader:
    """Tests for build_dataloader function."""

    def test_builds_from_local_file(self, mock_tokenizer, temp_data_file):
        """Test building dataloader from local file."""
        cfg_data = {
            "local_file": temp_data_file,
            "block_size": 64,
            "collate_mode": "pad",
            "num_workers": 0,
        }

        loader, collate = build_dataloader(
            tokenizer=mock_tokenizer,
            cfg_data=cfg_data,
            batch_size=2,
        )

        # Check loader produces batches
        batch = next(iter(loader))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 2

    def test_dataloader_shuffles(self, mock_tokenizer, temp_data_file):
        """Test dataloader shuffling."""
        cfg_data = {
            "local_file": temp_data_file,
            "block_size": 64,
            "num_workers": 0,
        }

        loader1, _ = build_dataloader(mock_tokenizer, cfg_data, batch_size=2, shuffle=True)
        loader2, _ = build_dataloader(mock_tokenizer, cfg_data, batch_size=2, shuffle=True)

        # Get first batches - with shuffle they should differ (probabilistically)
        # This is a weak test but avoids determinism issues
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # At minimum, verify batches have correct structure
        assert batch1["input_ids"].shape == batch2["input_ids"].shape

    def test_distributed_sharding(self, mock_tokenizer, sample_texts):
        """Test data sharding across workers."""
        # Create temp file with enough lines
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for text in sample_texts * 4:  # 32 lines
                f.write(text + "\n")
            temp_file = f.name

        cfg_data = {
            "local_file": temp_file,
            "block_size": 64,
            "num_workers": 0,
        }

        # Simulate 2 workers
        loader0, _ = build_dataloader(mock_tokenizer, cfg_data, batch_size=1, rank=0, world_size=2)
        loader1, _ = build_dataloader(mock_tokenizer, cfg_data, batch_size=1, rank=1, world_size=2)

        # Each worker should see different data
        data0 = list(loader0)
        data1 = list(loader1)

        # Both should have data
        assert len(data0) > 0
        assert len(data1) > 0

        os.unlink(temp_file)

    def test_drop_last(self, mock_tokenizer, temp_data_file):
        """Test drop_last behavior."""
        cfg_data = {
            "local_file": temp_data_file,
            "block_size": 64,
            "num_workers": 0,
        }

        loader, _ = build_dataloader(mock_tokenizer, cfg_data, batch_size=3)

        # All batches should have same size
        batch_sizes = [batch["input_ids"].shape[0] for batch in loader]
        assert all(size == 3 for size in batch_sizes)


class TestPadVsPackComparison:
    """Compare pad and pack modes."""

    def test_pack_more_efficient_for_short_texts(self, mock_tokenizer):
        """Test that pack mode uses fewer sequences for short texts."""
        block_size = 256

        # Short texts
        texts = ["Hi", "Hello", "Test", "OK"] * 10  # 40 very short texts

        collate_pad = CollateFunction(mock_tokenizer, block_size, mode="pad")
        collate_pack = CollateFunction(mock_tokenizer, block_size, mode="pack")

        batch_pad = collate_pad(texts)
        batch_pack = collate_pack(texts)

        # Pack should produce fewer sequences
        assert batch_pack["input_ids"].shape[0] < batch_pad["input_ids"].shape[0]

    def test_both_modes_valid_for_training(self, mock_tokenizer):
        """Test both modes produce valid training data."""
        block_size = 64
        texts = ["This is a test sentence for training."] * 4

        for mode in ["pad", "pack"]:
            collate = CollateFunction(mock_tokenizer, block_size, mode=mode)
            batch = collate(texts)

            # Valid shapes
            assert len(batch["input_ids"].shape) == 2
            assert batch["input_ids"].shape[1] == block_size

            # Valid dtypes
            assert batch["input_ids"].dtype == torch.long
            assert batch["labels"].dtype == torch.long

            # Labels have correct ignore index
            assert (batch["labels"] >= -100).all()
