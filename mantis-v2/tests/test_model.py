from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from mantis_v2 import model as model_module
from mantis_v2.config import load_config
from mantis_v2.lora import LoRALinear, adapter_state, merge_lora_inplace, merged_lora_copy
from mantis_v2.model import (
    MantisV2Adapter,
    ModelContractError,
    NextLegModel,
    download_verified_weights,
    nextleg_loss,
    nextleg_loss_per_sample,
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

    per_sample = nextleg_loss_per_sample(output, targets["candle"], targets["leg"], config.target)
    assert per_sample["total"].shape == (2,)
    for key in ("total", "candle", "leg"):
        torch.testing.assert_close(per_sample[key].mean(), losses[key])


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


def test_transformer_finetune_freezes_only_upstream_token_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "nextleg.toml")

    class FakeMantisV2(nn.Module):
        def __init__(self, **_: object) -> None:
            super().__init__()
            self.tokgen_unit = nn.Linear(1, 1)
            self.transf_unit = nn.Linear(1, 1)
            self.prj = nn.Linear(1, 1)

        def from_pretrained(self, *_: object, **__: object) -> nn.Module:
            return self

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.zeros((len(value), 256), dtype=value.dtype, device=value.device)

    monkeypatch.setattr(model_module, "MantisV2", FakeMantisV2)
    monkeypatch.setattr(model_module, "download_verified_weights", lambda _: Path("verified"))
    model = NextLegModel(
        replace(config.model, mode="transformer_finetune"),
        config.target,
        5,
        torch.device("cpu"),
    )

    assert all(
        not parameter.requires_grad for parameter in model.encoder.backbone.tokgen_unit.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.encoder.backbone.transf_unit.parameters()
    )
    assert all(not parameter.requires_grad for parameter in model.encoder.backbone.prj.parameters())
    assert all(not parameter.requires_grad for parameter in model.adapter.parameters())
    assert all(parameter.requires_grad for parameter in model.candle_head.parameters())
    assert all(parameter.requires_grad for parameter in model.leg_head.parameters())


@pytest.mark.parametrize(
    ("mode", "rank", "alpha"),
    [
        ("lora_r8_alpha16", 8, 16),
        ("lora_r16_alpha32", 16, 32),
        ("lora_r8_alpha16_head_warmstart", 8, 16),
    ],
)
def test_lora_targets_only_attention_projections_and_keeps_heads_trainable(
    monkeypatch: pytest.MonkeyPatch, mode: str, rank: int, alpha: int
) -> None:
    config = load_config(ROOT / "configs" / "nextleg.toml")

    class FakeAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wQKV = nn.Linear(4, 12)
            self.wO = nn.Linear(4, 4)
            self.other = nn.Linear(4, 4)

    class FakeLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = FakeAttention()

    class FakeTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(), FakeLayer()])

    class FakeTransfUnit(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = FakeTransformer()

    class FakeMantisV2(nn.Module):
        def __init__(self, **_: object) -> None:
            super().__init__()
            self.tokgen_unit = nn.Linear(4, 4)
            self.transf_unit = FakeTransfUnit()
            self.prj = nn.Linear(4, 4)

        def from_pretrained(self, *_: object, **__: object) -> nn.Module:
            return self

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.zeros((len(value), 256), dtype=value.dtype, device=value.device)

    monkeypatch.setattr(model_module, "MantisV2", FakeMantisV2)
    monkeypatch.setattr(model_module, "download_verified_weights", lambda _: Path("verified"))
    model = NextLegModel(
        replace(config.model, mode=mode),
        config.target,
        5,
        torch.device("cpu"),
    )

    expected_lora = {
        f"backbone.transf_unit.transformer.layers.{layer}.attention.{projection}.{parameter}"
        for layer in range(2)
        for projection in ("wQKV", "wO")
        for parameter in ("lora_A", "lora_B")
    }
    trainable_encoder = {
        name for name, parameter in model.encoder.named_parameters() if parameter.requires_grad
    }
    if mode == "lora_r8_alpha16_head_warmstart":
        assert trainable_encoder == set()
        assert all(parameter.requires_grad for parameter in model.candle_head.parameters())
        assert all(parameter.requires_grad for parameter in model.leg_head.parameters())
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad]
        )
        output = model(torch.zeros(2, 5, 512))
        (output["candle"].sum() + output["leg"].sum()).backward()
        optimizer.step()
        after_warm = model.state_dict()
        for name, tensor in before.items():
            if name.startswith(("encoder.", "adapter.")):
                assert torch.equal(tensor, after_warm[name])
        assert any(
            not torch.equal(tensor, after_warm[name])
            for name, tensor in before.items()
            if name.startswith("candle_head.")
        )
        assert any(
            not torch.equal(tensor, after_warm[name])
            for name, tensor in before.items()
            if name.startswith("leg_head.")
        )
        before_transition = {name: tensor.clone() for name, tensor in after_warm.items()}
        model.set_adaptation_phase("lora")
        assert all(
            torch.equal(before_transition[name], tensor)
            for name, tensor in model.state_dict().items()
        )
        trainable_encoder = {
            name for name, parameter in model.encoder.named_parameters() if parameter.requires_grad
        }
    assert trainable_encoder == expected_lora
    for module in model.encoder.modules():
        if module.__class__.__name__ == "LoRALinear":
            assert module.rank == rank
            assert module.alpha == alpha
    assert all(not parameter.requires_grad for parameter in model.adapter.parameters())
    assert all(parameter.requires_grad for parameter in model.candle_head.parameters())
    assert all(parameter.requires_grad for parameter in model.leg_head.parameters())
    saved = adapter_state(model)
    assert any(name.startswith("candle_head.") for name in saved)
    assert any(name.startswith("leg_head.") for name in saved)
    assert set(saved) == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def test_lora_merge_preserves_outputs_and_removes_adapter_tensors() -> None:
    class ProjectionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wQKV = LoRALinear(nn.Linear(4, 12), rank=2, alpha=4)
            self.wO = LoRALinear(nn.Linear(12, 4), rank=2, alpha=4)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.wO(torch.tanh(self.wQKV(value)))

    model = ProjectionModel().eval()
    with torch.no_grad():
        model.wQKV.lora_B.copy_(torch.randn_like(model.wQKV.lora_B))
        model.wO.lora_B.copy_(torch.randn_like(model.wO.lora_B))
    fixture = torch.randn(3, 4)

    merged = merged_lora_copy(model).eval()

    assert torch.allclose(model(fixture), merged(fixture), atol=1e-6, rtol=1e-6)
    assert set(adapter_state(model)) == {
        "wQKV.lora_A",
        "wQKV.lora_B",
        "wO.lora_A",
        "wO.lora_B",
    }
    assert not any(isinstance(module, LoRALinear) for module in merged.modules())
    assert not any("lora_" in name for name in merged.state_dict())


def test_lora_inplace_merge_avoids_unpicklable_upstream_state() -> None:
    class ProjectionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.upstream_staticmethod = staticmethod(lambda value: value)
            self.wQKV = LoRALinear(nn.Linear(4, 12), rank=2, alpha=4)
            self.wO = LoRALinear(nn.Linear(12, 4), rank=2, alpha=4)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.wO(torch.tanh(self.wQKV(value)))

    model = ProjectionModel().eval()
    with torch.no_grad():
        model.wQKV.lora_B.copy_(torch.randn_like(model.wQKV.lora_B))
        model.wO.lora_B.copy_(torch.randn_like(model.wO.lora_B))
    fixture = torch.randn(3, 4)
    expected = model(fixture)

    with pytest.raises(TypeError, match="staticmethod"):
        copy.deepcopy(model)
    merged = merge_lora_inplace(model).eval()

    assert torch.allclose(expected, merged(fixture), atol=1e-6, rtol=1e-6)
    assert not any(isinstance(module, LoRALinear) for module in merged.modules())
