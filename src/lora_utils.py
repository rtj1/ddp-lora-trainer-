"""
LoRA utilities - backward compatible wrapper.

This module provides backward compatibility while using our custom LoRA implementation.
For the full implementation with detailed documentation, see lora.py
"""

from .lora import (
    apply_lora,
    apply_lora_to_model,
    LoRAConfig,
    LoRALinear,
    count_parameters,
    get_lora_params,
    merge_lora_weights,
    unmerge_lora_weights,
    save_lora_weights,
    load_lora_weights,
)

__all__ = [
    "apply_lora",
    "apply_lora_to_model",
    "LoRAConfig",
    "LoRALinear",
    "count_parameters",
    "get_lora_params",
    "merge_lora_weights",
    "unmerge_lora_weights",
    "save_lora_weights",
    "load_lora_weights",
]
