from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from mantis_v2 import model as model_module
from mantis_v2.config import load_config
from mantis_v2.model import (
    MantisV2Adapter,
    ModelContractError,
    NextLegModel,
    download_verified_weights,
    nextleg_loss,
    sha256_file,
)
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def test_scratch_model_obeys_multichannel_output_contract() -> None:
    config = load_config(ROOT / "configs" / "smoke.toml")
    model = NextLegModel(
        config.model,
        config.target,
        feature_count=5,
        device=torch.device("cpu"),
    )
    context = torch.randn(2, 5, 512)
    output = model(context)
    assert output["candle"].shape == (2, 5, 1)
    assert output["leg"].shape == (2, 2)
    targets = {"candle": torch.zeros(2, 5, 1), "leg": torch.zeros(2, 2)}
    losses = nextleg_loss(output, targets["candle"], targets["leg"], config.target)
    assert losses["total"].isfinite()


def test_non_adapter_mode_freezes_unused_adapter() -> None:
    config = load_config(ROOT / "configs" / "smoke.toml")
    model = NextLegModel(config.model, config.target, 5, torch.device("cpu"))
    assert all(not parameter.requires_grad for parameter in model.adapter.parameters())
    assert any(parameter.requires_grad for parameter in model.encoder.parameters())


def test_weight_download_rejects_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs" / "nextleg.toml")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"not the pinned checkpoint")
    monkeypatch.setattr(model_module, "hf_hub_download", lambda **_: str(weights))
    bad_model = replace(config.model, weights_sha256="0" * 64)
    with pytest.raises(ModelContractError, match="weight digest mismatch"):
        download_verified_weights(bad_model)
    good_model = replace(config.model, weights_sha256=sha256_file(weights))
    assert download_verified_weights(good_model) == weights


def test_pretrained_adapter_uses_pinned_hub_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(ROOT / "configs" / "nextleg.toml")
    calls: list[tuple[str, str]] = []

    class FakeMantisV2(nn.Module):
        def __init__(self, **_: object) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def from_pretrained(self, model_name: str, *, revision: str) -> nn.Module:
            calls.append((model_name, revision))
            return self

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.zeros((len(value), 256), dtype=value.dtype, device=value.device)

    monkeypatch.setattr(model_module, "MantisV2", FakeMantisV2)
    monkeypatch.setattr(model_module, "download_verified_weights", lambda _: Path("verified"))
    adapter = MantisV2Adapter(config.model, torch.device("cpu"))
    output = adapter(torch.zeros(2, 5, 512))
    assert calls == [(config.model.hub_model, config.model.hub_revision)]
    assert output.shape == (2, 5 * 256)
