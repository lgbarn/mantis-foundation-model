"""Fail-closed precision policy for foundation adaptation."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch

from mantis_v2.config import Precision


class PrecisionContractError(RuntimeError):
    """Raised when a precision/device pair cannot honor its declared semantics."""


def validate_precision_device(
    precision: Precision,
    device: torch.device,
    *,
    bf16_supported: bool | None = None,
) -> None:
    """Reject unsupported BF16 before any run artifact is opened."""
    if precision == "fp32":
        return
    supported = bool(torch.cuda.is_bf16_supported()) if bf16_supported is None else bf16_supported
    if device.type != "cuda" or not supported:
        raise PrecisionContractError("BF16 requires an explicitly supported CUDA device")


def autocast_context(precision: Precision, device: torch.device) -> AbstractContextManager[None]:
    """Return the registered compute context without changing parameter dtype."""
    validate_precision_device(precision, device)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def validate_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Require every floating optimizer tensor to remain finite FP32 master state."""
    for state in optimizer.state.values():
        for value in state.values():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                continue
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise PrecisionContractError("non-finite FP32 optimizer state")
