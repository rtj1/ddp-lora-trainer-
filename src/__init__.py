"""
DDP-LoRA-Trainer: Distributed training infrastructure for LLM fine-tuning.
"""

from .lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    apply_lora_to_model,
    count_parameters,
)
from .dataset import (
    TextFileDataset,
    HFDStreamIterable,
    CollateFunction,
    build_dataloader,
)
from .utils import (
    load_config,
    set_seed,
    is_main_process,
    ThroughputMeter,
)

__version__ = "0.1.0"
