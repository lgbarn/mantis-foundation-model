"""Runtime selection and deterministic seeding."""

from __future__ import annotations

import random

import numpy as np
import torch

from mantis_v2.config import RunConfig


class RuntimeContractError(RuntimeError):
    """Raised when configured hardware is unavailable."""


def select_device(config: RunConfig) -> torch.device:
    requested = config.device
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeContractError("CUDA was required but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeContractError("MPS was requested but is unavailable")
    if config.require_accelerator and device.type == "cpu":
        raise RuntimeContractError("an accelerator is required but only CPU is available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
