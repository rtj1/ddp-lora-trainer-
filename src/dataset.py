from typing import List, Dict, Iterator, Optional
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch
import os

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

class TextFileDataset(Dataset):
    def __init__(self, lines: List[str]):
        self.lines = [ln.strip() for ln in lines if ln.strip()]
    def __len__(self): return len(self.lines)
    def __getitem__(self, idx): return self.lines[idx]

class HFDStreamIterable(IterableDataset):
    """Wraps a Hugging Face streaming dataset and yields raw text strings.
    Shards by (world_size, rank) so each worker sees a unique subset."""
    def __init__(self, ds_name: str, split: str, text_key: str, rank: int, world_size: int, shuffle_buffer: int = 10000, seed: int = 42):
        super().__init__()
        if load_dataset is None:
            raise RuntimeError("`datasets` is not installed. Add it to requirements.txt")
        self.ds_name = ds_name
        self.split = split
        self.text_key = text_key
        self.rank = rank
        self.world_size = world_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _build(self):
        use_auth_token = os.environ.get("HF_TOKEN", None)
        ds = load_dataset(self.ds_name, split=self.split, streaming=True, token=use_auth_token)
        if self.shuffle_buffer and hasattr(ds, "shuffle"):
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        if hasattr(ds, "shard"):
            ds = ds.shard(num_shards=max(self.world_size,1), index=self.rank, contiguous=True)
        return ds

    def __iter__(self) -> Iterator[str]:
        ds = self._build()
        for ex in ds:
            if self.text_key in ex and isinstance(ex[self.text_key], str):
                yield ex[self.text_key]

def _build_collate(tokenizer, block_size: int):
    def collate(batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        joined = " ".join(batch_texts)
        toks = tokenizer(joined, return_tensors="pt", truncation=True, max_length=block_size, padding="max_length")
        input_ids = toks["input_ids"]
        attn = toks["attention_mask"]
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
    return collate

def build_dataloader(tokenizer, cfg_data: dict, batch_size: int, rank: int=0, world_size: int=1, shuffle: bool=True):
    block_size = cfg_data.get("block_size", 512)
    collate = _build_collate(tokenizer, block_size)

    if cfg_data.get("use_hf_streaming", False):
        if not cfg_data.get("hf_dataset"):
            raise ValueError("data.use_hf_streaming=true requires data.hf_dataset to be set")
        text_key = cfg_data.get("text_key", "text")
        stream = HFDStreamIterable(
            ds_name=cfg_data["hf_dataset"],
            split=cfg_data.get("hf_split", "train"),
            text_key=text_key,
            rank=rank, world_size=world_size,
            shuffle_buffer=cfg_data.get("streaming_shuffle_buffer", 10000),
            seed=cfg_data.get("seed", 42),
        )
        loader = DataLoader(
            stream,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=cfg_data.get("pin_memory", True),
            drop_last=True,
        )
        return loader, collate

    with open(cfg_data["local_file"], "r", encoding="utf-8") as f:
        lines = f.readlines()
    if world_size>1:
        lines = lines[rank::world_size]
    ds = TextFileDataset(lines)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=cfg_data.get("num_workers",4),
        prefetch_factor=cfg_data.get("prefetch_factor",4),
        pin_memory=cfg_data.get("pin_memory",True),
        drop_last=True,
    )
    return loader, collate