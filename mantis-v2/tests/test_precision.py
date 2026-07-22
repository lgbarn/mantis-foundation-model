from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from mantis_v2.precision import (
    PrecisionContractError,
    autocast_context,
    validate_optimizer_state,
)


def test_bf16_autocast_preserves_finite_fp32_optimizer_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.dtype]] = []

    def fake_autocast(*, device_type: str, dtype: torch.dtype):
        calls.append((device_type, dtype))
        return nullcontext()

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch, "autocast", fake_autocast)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    with autocast_context("bf16", torch.device("cuda")):
        loss = model(torch.ones(1, 2)).square().mean()
    loss.backward()
    optimizer.step()
    validate_optimizer_state(optimizer)

    assert calls == [("cuda", torch.bfloat16)]
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(
        value.dtype == torch.float32
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )

    first_state = next(iter(optimizer.state.values()))
    first_state["exp_avg"].fill_(float("nan"))
    with pytest.raises(PrecisionContractError, match="non-finite FP32 optimizer state"):
        validate_optimizer_state(optimizer)
