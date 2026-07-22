"""Pinned upstream adapter and local NextLeg heads."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypedDict

import torch
from huggingface_hub import hf_hub_download
from mantis.architecture import MantisV2
from torch import nn
from torch.nn import functional as F

from mantis_v2.config import ModelConfig, TargetConfig
from mantis_v2.lora import LoRALinear, inject_mantis_lora, lora_spec


class ModelContractError(RuntimeError):
    """Raised when upstream weights or model behavior violate the pinned contract."""


class NextLegOutput(TypedDict):
    candle: torch.Tensor
    leg: torch.Tensor


class LossOutput(TypedDict):
    total: torch.Tensor
    candle: torch.Tensor
    leg: torch.Tensor


class PerSampleLossOutput(TypedDict):
    total: torch.Tensor
    candle: torch.Tensor
    leg: torch.Tensor


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_verified_weights(config: ModelConfig) -> Path:
    """Fetch the immutable Hub checkpoint and verify its published digest."""
    path = Path(
        hf_hub_download(
            repo_id=config.hub_model,
            filename="model.safetensors",
            revision=config.hub_revision,
        )
    )
    actual = sha256_file(path)
    if actual != config.weights_sha256:
        raise ModelContractError(
            f"MantisV2 weight digest mismatch: expected {config.weights_sha256}, got {actual}"
        )
    return path


class MantisV2Adapter(nn.Module):
    """Keep all upstream-specific construction and channel handling in one boundary."""

    embedding_dim = 256

    def __init__(self, config: ModelConfig, device: torch.device) -> None:
        super().__init__()
        backbone = MantisV2(device=str(device), pre_training=False)
        if config.mode != "scratch":
            download_verified_weights(config)
            loaded = backbone.from_pretrained(config.hub_model, revision=config.hub_revision)
            if not isinstance(loaded, nn.Module):
                raise ModelContractError("upstream from_pretrained did not return a torch module")
            backbone = loaded
        self.backbone: nn.Module = backbone
        self.input_length = config.input_length
        self.lora_targets: tuple[str, ...] = ()
        if config.mode.startswith("lora_"):
            rank, alpha = lora_spec(config.mode)
            self.lora_targets = inject_mantis_lora(backbone, rank=rank, alpha=alpha)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3:
            raise ModelContractError("context must have shape [batch, channels, time]")
        batch, channels, length = context.shape
        if length != self.input_length or length % 32:
            raise ModelContractError(
                f"context length must be configured length {self.input_length} and divisible by 32"
            )
        univariate = context.reshape(batch * channels, 1, length)
        embedding = self.backbone(univariate)
        if not isinstance(embedding, torch.Tensor) or embedding.shape != (
            batch * channels,
            self.embedding_dim,
        ):
            raise ModelContractError("unexpected upstream embedding shape")
        return embedding.reshape(batch, channels * self.embedding_dim)

    def freeze_outside_transformer(self) -> None:
        """Freeze every upstream encoder component except its transformer."""
        transformer = getattr(self.backbone, "transf_unit", None)
        if not isinstance(transformer, nn.Module):
            raise ModelContractError("upstream MantisV2 has no compatible transformer")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in transformer.parameters():
            parameter.requires_grad = True

    def freeze_for_lora(self) -> None:
        if not self.lora_targets:
            raise ModelContractError("LoRA mode did not install attention adapters")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for module in self.backbone.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True


class NextLegModel(nn.Module):
    """MantisV2 encoder with candle and two-leg prediction heads."""

    def __init__(
        self,
        model_config: ModelConfig,
        target_config: TargetConfig,
        feature_count: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.encoder = MantisV2Adapter(model_config, device)
        embedding_dim = feature_count * self.encoder.embedding_dim
        adapter_hidden = max(embedding_dim // 4, 64)
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, adapter_hidden),
            nn.GELU(),
            nn.Linear(adapter_hidden, embedding_dim),
        )
        head_hidden = max(embedding_dim // 4, 64)
        self.candle_head = nn.Linear(embedding_dim, feature_count * len(target_config.horizons))
        self.leg_head = nn.Sequential(
            nn.Linear(embedding_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 2),
        )
        self.feature_count = feature_count
        self.horizon_count = len(target_config.horizons)
        self.mode = model_config.mode
        self._apply_freeze_policy()

    def _apply_freeze_policy(self) -> None:
        if self.mode in {"head_only", "adapter_head"}:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        elif self.mode == "transformer_finetune":
            self.encoder.freeze_outside_transformer()
        elif self.mode.startswith("lora_"):
            self.encoder.freeze_for_lora()
        if self.mode != "adapter_head":
            for parameter in self.adapter.parameters():
                parameter.requires_grad = False

    def forward(self, context: torch.Tensor) -> NextLegOutput:
        embedding = self.encoder(context)
        if self.mode == "adapter_head":
            embedding = embedding + self.adapter(embedding)
        candle = self.candle_head(embedding).reshape(
            len(context), self.feature_count, self.horizon_count
        )
        return {"candle": candle, "leg": self.leg_head(embedding)}


def nextleg_loss(
    output: NextLegOutput,
    candle_target: torch.Tensor,
    leg_target: torch.Tensor,
    target_config: TargetConfig,
) -> LossOutput:
    candle = F.mse_loss(output["candle"], candle_target)
    leg = F.smooth_l1_loss(output["leg"], leg_target)
    total = target_config.candle_loss_weight * candle + target_config.leg_loss_weight * leg
    return {"total": total, "candle": candle, "leg": leg}


def nextleg_loss_per_sample(
    output: NextLegOutput,
    candle_target: torch.Tensor,
    leg_target: torch.Tensor,
    target_config: TargetConfig,
) -> PerSampleLossOutput:
    """Return unreduced losses so evaluation can aggregate streams without bias."""
    candle = F.mse_loss(output["candle"], candle_target, reduction="none").mean(dim=(1, 2))
    leg = F.smooth_l1_loss(output["leg"], leg_target, reduction="none").mean(dim=1)
    total = target_config.candle_loss_weight * candle + target_config.leg_loss_weight * leg
    return {"total": total, "candle": candle, "leg": leg}
