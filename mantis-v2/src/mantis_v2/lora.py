"""Small native LoRA implementation for the pinned MantisV2 attention projections."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


class LoRAContractError(RuntimeError):
    """Raised when the pinned attention structure or adapter state is invalid."""


class LoRALinear(nn.Module):
    """Frozen linear projection plus trainable low-rank residual."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: int) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise LoRAContractError("LoRA rank and alpha must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = F.linear(F.linear(value, self.lora_A), self.lora_B)
        return cast(torch.Tensor, self.base(value) + residual * self.scaling)

    def merged(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        delta = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + delta.to(self.base.weight.dtype))
            if self.base.bias is not None and merged.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def _parent_module(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
        if not isinstance(parent, nn.Module):
            raise LoRAContractError(f"LoRA parent is not a module: {name}")
    return parent, parts[-1]


def inject_mantis_lora(backbone: nn.Module, *, rank: int, alpha: int) -> tuple[str, ...]:
    """Replace only MantisV2 attention wQKV/wO linears with LoRA wrappers."""
    targets = [
        (name, module)
        for name, module in backbone.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in {"wQKV", "wO"}
    ]
    if not targets or {name.rsplit(".", 1)[-1] for name, _ in targets} != {"wQKV", "wO"}:
        raise LoRAContractError("pinned MantisV2 attention projections are missing")
    for name, module in targets:
        parent, attribute = _parent_module(backbone, name)
        setattr(parent, attribute, LoRALinear(module, rank=rank, alpha=alpha))
    return tuple(name for name, _ in targets)


def lora_spec(mode: str) -> tuple[int, int]:
    if mode in {"lora_r8_alpha16", "lora_r8_alpha16_head_warmstart"}:
        return 8, 16
    if mode == "lora_r16_alpha32":
        return 16, 32
    raise LoRAContractError(f"unsupported LoRA mode: {mode}")


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return every trainable delta needed to reconstruct a LoRA candidate."""
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.named_parameters()
        if tensor.requires_grad
    }
    if not state or not any(name.endswith((".lora_A", ".lora_B")) for name in state):
        raise LoRAContractError("model has no LoRA adapter state")
    return state


def merged_lora_copy(model: nn.Module) -> nn.Module:
    """Return a deep copy with every LoRA wrapper folded into its base weight."""
    merged = copy.deepcopy(model)
    wrappers = [
        (name, module) for name, module in merged.named_modules() if isinstance(module, LoRALinear)
    ]
    if not wrappers:
        raise LoRAContractError("model has no LoRA modules to merge")
    for name, module in wrappers:
        parent, attribute = _parent_module(merged, name)
        setattr(parent, attribute, module.merged())
    return merged


def lora_metadata(model: nn.Module) -> Mapping[str, object]:
    wrappers = [
        (name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)
    ]
    if not wrappers:
        raise LoRAContractError("model has no LoRA modules")
    specs = {(module.rank, module.alpha) for _, module in wrappers}
    if len(specs) != 1:
        raise LoRAContractError("model mixes incompatible LoRA specifications")
    rank, alpha = next(iter(specs))
    return {"rank": rank, "alpha": alpha, "targets": [name for name, _ in wrappers]}
