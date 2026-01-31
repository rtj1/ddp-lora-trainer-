"""
Dataset utilities for distributed LLM training.

This module provides:
- TextFileDataset: Map-style dataset from text files
- HFDStreamIterable: Streaming dataset from Hugging Face
- Proper collation that preserves document boundaries
- Support for packing multiple sequences

Key Design Decision:
--------------------
The old collate function joined all texts into one string, losing document
boundaries. The new implementation:
1. Tokenizes each document separately
2. Pads/truncates to block_size
3. Optionally packs multiple short sequences together
"""

from typing import List, Dict, Iterator, Optional, Tuple
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch
import os

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


class TextFileDataset(Dataset):
    """Map-style dataset from text file lines."""

    def __init__(self, lines: List[str]):
        self.lines = [ln.strip() for ln in lines if ln.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> str:
        return self.lines[idx]


class HFDStreamIterable(IterableDataset):
    """
    Streaming dataset from Hugging Face with shard-aware partitioning.

    Each worker (rank) sees a unique subset of the data, enabling
    efficient distributed training without data duplication.
    """

    def __init__(
        self,
        ds_name: str,
        split: str,
        text_key: str,
        rank: int,
        world_size: int,
        shuffle_buffer: int = 10000,
        seed: int = 42,
    ):
        super().__init__()
        if load_dataset is None:
            raise RuntimeError("`datasets` is not installed")

        self.ds_name = ds_name
        self.split = split
        self.text_key = text_key
        self.rank = rank
        self.world_size = world_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _build(self):
        use_auth_token = os.environ.get("HF_TOKEN", None)
        ds = load_dataset(
            self.ds_name,
            split=self.split,
            streaming=True,
            token=use_auth_token,
        )

        if self.shuffle_buffer and hasattr(ds, "shuffle"):
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)

        if hasattr(ds, "shard"):
            ds = ds.shard(
                num_shards=max(self.world_size, 1),
                index=self.rank,
                contiguous=True,
            )

        return ds

    def __iter__(self) -> Iterator[str]:
        ds = self._build()
        for ex in ds:
            if self.text_key in ex and isinstance(ex[self.text_key], str):
                yield ex[self.text_key]


class CollateFunction:
    """
    Proper collation that preserves document boundaries.

    Modes:
    ------
    - "pad": Tokenize each document, pad/truncate to block_size
    - "pack": Pack multiple documents into block_size (more efficient)

    The old implementation joined all texts with spaces, losing document
    boundaries. This implementation tokenizes each document separately.
    """

    def __init__(
        self,
        tokenizer,
        block_size: int,
        mode: str = "pad",
        ignore_index: int = -100,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mode = mode
        self.ignore_index = ignore_index

        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        if self.mode == "pack":
            return self._collate_packed(batch_texts)
        return self._collate_padded(batch_texts)

    def _collate_padded(self, batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize each document separately, then pad to block_size.

        This preserves document boundaries - each sequence in the batch
        is from a single document.
        """
        # Tokenize each text separately
        encodings = self.tokenizer(
            batch_texts,
            truncation=True,
            max_length=self.block_size,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Labels: same as input_ids, but with padding positions masked
        labels = input_ids.clone()
        labels[attention_mask == 0] = self.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _collate_packed(self, batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Pack multiple documents into sequences of exactly block_size.

        This is more efficient than padding when documents are much shorter
        than block_size. Documents are separated by EOS tokens.

        Example with block_size=10:
            Doc1: "Hello" (3 tokens)
            Doc2: "World" (3 tokens)
            Packed: [Hello, EOS, World, EOS, PAD, PAD, ...]
        """
        packed_input_ids = []
        packed_attention_mask = []
        packed_labels = []

        current_ids = []
        current_mask = []
        current_labels = []

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        for text in batch_texts:
            # Tokenize without special tokens (we add EOS manually)
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.block_size - 1,  # Leave room for EOS
            )

            # Add EOS at end
            tokens.append(eos_id)

            # Check if we can fit in current sequence
            if len(current_ids) + len(tokens) <= self.block_size:
                current_ids.extend(tokens)
                current_mask.extend([1] * len(tokens))
                current_labels.extend(tokens)
            else:
                # Pad and save current sequence
                if current_ids:
                    current_ids, current_mask, current_labels = self._pad_sequence(
                        current_ids, current_mask, current_labels
                    )
                    packed_input_ids.append(current_ids)
                    packed_attention_mask.append(current_mask)
                    packed_labels.append(current_labels)

                # Start new sequence
                current_ids = tokens
                current_mask = [1] * len(tokens)
                current_labels = tokens.copy()

        # Handle last sequence
        if current_ids:
            current_ids, current_mask, current_labels = self._pad_sequence(
                current_ids, current_mask, current_labels
            )
            packed_input_ids.append(current_ids)
            packed_attention_mask.append(current_mask)
            packed_labels.append(current_labels)

        # Stack into tensors
        return {
            "input_ids": torch.tensor(packed_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(packed_attention_mask, dtype=torch.long),
            "labels": torch.tensor(packed_labels, dtype=torch.long),
        }

    def _pad_sequence(
        self,
        ids: List[int],
        mask: List[int],
        labels: List[int],
    ) -> Tuple[List[int], List[int], List[int]]:
        """Pad a sequence to block_size."""
        pad_len = self.block_size - len(ids)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id
            ids = ids + [pad_id] * pad_len
            mask = mask + [0] * pad_len
            labels = labels + [self.ignore_index] * pad_len
        return ids, mask, labels


def build_collate(
    tokenizer,
    block_size: int,
    mode: str = "pad",
) -> CollateFunction:
    """Create a collate function with the given configuration."""
    return CollateFunction(tokenizer, block_size, mode=mode)


def build_dataloader(
    tokenizer,
    cfg_data: dict,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
) -> Tuple[DataLoader, CollateFunction]:
    """
    Build a DataLoader for training.

    Args:
        tokenizer: HuggingFace tokenizer
        cfg_data: Data configuration dict
        batch_size: Batch size per device
        rank: Process rank for distributed training
        world_size: Total number of processes
        shuffle: Whether to shuffle (for map-style datasets)

    Returns:
        (DataLoader, collate_function)
    """
    block_size = cfg_data.get("block_size", 512)
    collate_mode = cfg_data.get("collate_mode", "pad")  # "pad" or "pack"
    collate = build_collate(tokenizer, block_size, mode=collate_mode)

    # Streaming dataset (HuggingFace)
    if cfg_data.get("use_hf_streaming", False):
        if not cfg_data.get("hf_dataset"):
            raise ValueError("data.use_hf_streaming=true requires data.hf_dataset")

        text_key = cfg_data.get("text_key", "text")
        stream = HFDStreamIterable(
            ds_name=cfg_data["hf_dataset"],
            split=cfg_data.get("hf_split", "train"),
            text_key=text_key,
            rank=rank,
            world_size=world_size,
            shuffle_buffer=cfg_data.get("streaming_shuffle_buffer", 10000),
            seed=cfg_data.get("seed", 42),
        )

        loader = DataLoader(
            stream,
            batch_size=batch_size,
            shuffle=False,  # Streaming datasets handle their own shuffling
            num_workers=0,  # Must be 0 for streaming
            pin_memory=cfg_data.get("pin_memory", True),
            drop_last=True,
            collate_fn=collate,
        )
        return loader, collate

    # Map-style dataset (local file)
    with open(cfg_data["local_file"], "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Shard data across workers
    if world_size > 1:
        lines = lines[rank::world_size]

    ds = TextFileDataset(lines)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg_data.get("num_workers", 4),
        prefetch_factor=cfg_data.get("prefetch_factor", 4),
        pin_memory=cfg_data.get("pin_memory", True),
        drop_last=True,
        collate_fn=collate,
    )
    return loader, collate
