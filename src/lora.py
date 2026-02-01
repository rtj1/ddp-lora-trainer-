"""
Custom LoRA (Low-Rank Adaptation) Implementation
=================================================

This module implements LoRA from scratch, providing:
- Custom LoRALinear layer with low-rank decomposition
- Automatic module replacement for transformer models
- Memory-efficient parameter counting
- Merge/unmerge functionality for inference

LoRA Paper: https://arxiv.org/abs/2106.09685

Key Insight:
------------
Instead of fine-tuning W (d×k), we learn:
    W' = W + BA
where B (d×r) and A (r×k), with r << min(d,k)

This reduces trainable parameters from d×k to r×(d+k).
For r=8 on a 4096×4096 weight: 16M → 65K params (250x reduction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

# Try to import Conv1D from transformers (used by GPT-2)
try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    r: int = 8                          # Rank of low-rank matrices
    alpha: float = 16.0                 # Scaling factor
    dropout: float = 0.05               # Dropout on LoRA path
    target_modules: List[str] = None    # Module names to adapt
    fan_in_fan_out: bool = False        # Set True for Conv1D (GPT-2 style)
    merge_weights: bool = False         # Merge at init (for inference)

    def __post_init__(self):
        if self.target_modules is None:
            # Default targets for common architectures
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # LLaMA
                "c_attn", "c_proj",                       # GPT-2
                "query", "key", "value",                  # BERT
            ]

    @property
    def scaling(self) -> float:
        """LoRA scaling factor: alpha / r"""
        return self.alpha / self.r


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Architecture:
    -------------
        Input x (batch, seq, in_features)
              │
              ├──────────────────┐
              │                  │
              ▼                  ▼
        [Frozen W]          [LoRA path]
        (out, in)           A: (r, in)
              │             B: (out, r)
              │                  │
              │             Dropout
              │                  │
              │             x @ A.T @ B.T
              │                  │
              │             × scaling
              │                  │
              └────────┬─────────┘
                       │
                       ▼
                   Output (batch, seq, out_features)

    Memory Analysis:
    ----------------
    Original: d × k parameters (all trainable)
    LoRA:     d × k (frozen) + r × (d + k) (trainable)

    For d=k=4096, r=8:
        Original: 16.7M trainable
        LoRA:     65K trainable (0.4%)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.fan_in_fan_out = fan_in_fan_out
        self.merge_weights = merge_weights
        self.merged = False

        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = None  # Will be set if original has bias

        # LoRA matrices
        # A: (r, in_features) - down-projection
        # B: (out_features, r) - up-projection
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))

        # Dropout on LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """
        Initialize LoRA parameters.

        A: Kaiming uniform (same as nn.Linear default)
        B: Zero initialization

        This ensures ΔW = BA = 0 at initialization, so the model
        starts from the pretrained weights.
        """
        # A uses Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is zero so ΔW = BA = 0 at init
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        fan_in_fan_out: bool = False,
    ) -> "LoRALinear":
        """Create LoRALinear from an existing nn.Linear layer."""
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            fan_in_fan_out=fan_in_fan_out,
        )

        # Copy frozen weights
        with torch.no_grad():
            if fan_in_fan_out:
                lora_linear.weight.copy_(linear.weight.T)
            else:
                lora_linear.weight.copy_(linear.weight)

            if linear.bias is not None:
                lora_linear.bias = nn.Parameter(linear.bias.clone(), requires_grad=False)

        return lora_linear

    def merge(self):
        """Merge LoRA weights into base weights for efficient inference."""
        if not self.merged:
            # W' = W + scaling * B @ A
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            if self.fan_in_fan_out:
                self.weight.data += delta_w.T
            else:
                self.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights (restore original base weights)."""
        if self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            if self.fan_in_fan_out:
                self.weight.data -= delta_w.T
            else:
                self.weight.data -= delta_w
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        If merged: Just use the merged weight (efficient inference)
        If not merged: Compute base + LoRA path separately (training)
        """
        if self.merged:
            # Inference mode: use merged weights
            if self.fan_in_fan_out:
                return F.linear(x, self.weight.T, self.bias)
            return F.linear(x, self.weight, self.bias)

        # Training mode: base + LoRA
        # Base path (frozen)
        if self.fan_in_fan_out:
            base_out = F.linear(x, self.weight.T, self.bias)
        else:
            base_out = F.linear(x, self.weight, self.bias)

        # LoRA path: x @ A.T @ B.T * scaling
        # Equivalent to: (x @ A.T) @ B.T = x @ (B @ A).T
        lora_out = self.lora_dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)  # (batch, seq, r)
        lora_out = F.linear(lora_out, self.lora_B)  # (batch, seq, out)
        lora_out = lora_out * self.scaling

        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"merged={self.merged}"
        )


def get_lora_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Separate model parameters into LoRA and non-LoRA groups.

    Returns:
        (lora_params, other_params)
    """
    lora_params = []
    other_params = []

    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_params.append(param)
        else:
            other_params.append(param)

    return lora_params, other_params


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters by category.

    Returns dict with:
        - total: All parameters
        - trainable: Parameters with requires_grad=True
        - lora: LoRA-specific parameters
        - frozen: Frozen base weights
    """
    total = 0
    trainable = 0
    lora = 0

    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            if "lora_" in name:
                lora += param.numel()

    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "frozen": total - trainable,
        "trainable_percent": 100.0 * trainable / total if total > 0 else 0,
    }


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply LoRA to a model by replacing target Linear layers.

    Args:
        model: The model to adapt
        config: LoRA configuration
        verbose: Print parameter statistics

    Returns:
        The modified model with LoRA layers

    Example:
        >>> config = LoRAConfig(r=8, alpha=16, target_modules=["q_proj", "v_proj"])
        >>> model = apply_lora_to_model(model, config)
    """
    # Track replaced modules
    replaced = []

    # Find target modules
    target_names = set(config.target_modules)

    def replace_module(parent: nn.Module, name: str, module: nn.Module):
        """Replace a module in the parent."""
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)

    # Iterate through all modules
    for name, module in list(model.named_modules()):
        # Check if this module should be replaced
        module_name = name.split(".")[-1]

        # Check for nn.Linear
        if isinstance(module, nn.Linear) and module_name in target_names:
            # Detect if this is GPT-2 style Conv1D stored as Linear
            fan_in_fan_out = hasattr(module, "nf") or (
                hasattr(module, "weight") and
                module.weight.shape[0] != module.out_features
            )

            # Create LoRA replacement
            lora_module = LoRALinear.from_linear(
                module,
                r=config.r,
                alpha=config.alpha,
                dropout=config.dropout,
                fan_in_fan_out=fan_in_fan_out,
            )

            # Replace in model
            replace_module(model, name, lora_module)
            replaced.append(name)

        # Check for transformers Conv1D (used by GPT-2)
        elif Conv1D is not None and isinstance(module, Conv1D) and module_name in target_names:
            # Conv1D stores weights as (in_features, out_features)
            # For fan_in_fan_out=True, we store as-is and transpose in forward pass
            in_features = module.weight.shape[0]
            out_features = module.weight.shape[1]

            # Create a LoRALinear with fan_in_fan_out=True for Conv1D
            lora_module = LoRALinear(
                in_features=in_features,
                out_features=out_features,
                r=config.r,
                alpha=config.alpha,
                dropout=config.dropout,
                fan_in_fan_out=True,  # Conv1D uses transposed weights
            )

            # Copy Conv1D weights (stored as in_features, out_features)
            # For fan_in_fan_out=True, forward does F.linear(x, weight.T)
            # So we store as (in_features, out_features) directly
            with torch.no_grad():
                lora_module.weight = nn.Parameter(
                    module.weight.clone(), requires_grad=False
                )
                # Reinitialize lora_A and lora_B with correct dimensions
                lora_module.lora_A = nn.Parameter(
                    torch.empty(config.r, in_features, device=module.weight.device, dtype=module.weight.dtype)
                )
                lora_module.lora_B = nn.Parameter(
                    torch.empty(out_features, config.r, device=module.weight.device, dtype=module.weight.dtype)
                )
                nn.init.kaiming_uniform_(lora_module.lora_A, a=math.sqrt(5))
                nn.init.zeros_(lora_module.lora_B)

                if module.bias is not None:
                    lora_module.bias = nn.Parameter(module.bias.clone(), requires_grad=False)

            # Replace in model
            replace_module(model, name, lora_module)
            replaced.append(name)

    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    if verbose:
        stats = count_parameters(model)
        print(f"\n{'='*60}")
        print("LoRA Applied Successfully")
        print(f"{'='*60}")
        print(f"Replaced modules: {len(replaced)}")
        for name in replaced[:10]:  # Show first 10
            print(f"  - {name}")
        if len(replaced) > 10:
            print(f"  ... and {len(replaced) - 10} more")
        print(f"\nParameter Statistics:")
        print(f"  Total parameters:     {stats['total']:,}")
        print(f"  Trainable parameters: {stats['trainable']:,} ({stats['trainable_percent']:.2f}%)")
        print(f"  LoRA parameters:      {stats['lora']:,}")
        print(f"  Frozen parameters:    {stats['frozen']:,}")
        print(f"{'='*60}\n")

    return model


def merge_lora_weights(model: nn.Module):
    """Merge all LoRA weights into base weights for efficient inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module):
    """Unmerge all LoRA weights (restore training mode)."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def save_lora_weights(model: nn.Module, path: str):
    """Save only the LoRA weights (much smaller than full model)."""
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.data.clone()

    torch.save(lora_state, path)

    # Calculate size
    size_mb = sum(p.numel() * p.element_size() for p in lora_state.values()) / (1024 * 1024)
    print(f"Saved LoRA weights to {path} ({size_mb:.2f} MB)")


def load_lora_weights(model: nn.Module, path: str):
    """Load LoRA weights into a model."""
    lora_state = torch.load(path, map_location="cpu")

    model_state = model.state_dict()
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"Warning: {name} not found in model")

    print(f"Loaded LoRA weights from {path}")


# =============================================================================
# Convenience wrapper for backwards compatibility
# =============================================================================

def apply_lora(
    model: nn.Module,
    target_modules: List[str] = None,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> nn.Module:
    """
    Convenience function matching the old API.

    Replaces the trivial PEFT wrapper with our custom implementation.
    """
    config = LoRAConfig(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules or [],
    )
    return apply_lora_to_model(model, config, verbose=True)
