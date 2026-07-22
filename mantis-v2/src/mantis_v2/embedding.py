"""Validated MantisV2 export loading and bounded embedding extraction."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from mantis.architecture import MantisV2
from safetensors.torch import load_file
from torch.nn import functional as F

from mantis_v2.downstream_config import DownstreamConfig
from mantis_v2.model import sha256_file
from mantis_v2.strategy import load_market_frame


class EmbeddingContractError(RuntimeError):
    """Raised when an export or embedding shard violates its identity contract."""


@dataclass(frozen=True)
class LoadedFoundation:
    model: torch.nn.Module
    device: torch.device
    weights_path: Path
    weights_sha256: str
    validation_evidence_path: Path
    validation_evidence_sha256: str
    embedding_dim: int
    manifest: dict[str, Any]


def _validated_evidence(manifest_path: Path, manifest: dict[str, Any]) -> tuple[Path, str]:
    validation_gate = manifest.get("validation_gate")
    if validation_gate is not None:
        if not isinstance(validation_gate, dict) or validation_gate.get("verified") is not True:
            raise EmbeddingContractError("foundation export validation gate is not verified")
        evaluation_value = validation_gate.get("evaluation")
        expected_sha = validation_gate.get("evaluation_sha256")
        if not isinstance(evaluation_value, str) or not isinstance(expected_sha, str):
            raise EmbeddingContractError("foundation validation gate lacks evaluation identity")
        evaluation_path = Path(evaluation_value)
        if not evaluation_path.is_file():
            evaluation_path = manifest_path.parent / "evaluation.json"
        if not evaluation_path.is_file() or sha256_file(evaluation_path) != expected_sha:
            raise EmbeddingContractError("foundation validation evidence digest mismatch")
        return evaluation_path, expected_sha

    legacy_path = manifest_path.parent.parent / "evaluation.json"
    try:
        legacy = json.loads(legacy_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise EmbeddingContractError(
            "foundation export has neither a validation gate nor legacy evaluation evidence"
        ) from exc
    provenance = manifest.get("provenance")
    metrics = legacy.get("metrics")
    if (
        not isinstance(provenance, dict)
        or not isinstance(metrics, dict)
        or legacy.get("split") != "validation"
        or legacy.get("config_digest") != provenance.get("config_digest")
        or legacy.get("dataset_digest") != provenance.get("dataset_digest")
        or not metrics
        or any(
            not isinstance(value, int | float) or not np.isfinite(value)
            for value in metrics.values()
        )
    ):
        raise EmbeddingContractError("legacy foundation evaluation evidence is invalid")
    return legacy_path, sha256_file(legacy_path)


def resolve_embedding_device(
    requested: Literal["cpu", "cuda", "mps"],
    *,
    cuda_available: Callable[[], bool] | None = None,
    mps_available: Callable[[], bool] | None = None,
) -> torch.device:
    """Resolve only an explicitly requested downstream embedding device."""
    cuda_available = cuda_available or torch.cuda.is_available
    mps_available = mps_available or torch.backends.mps.is_available
    if requested == "cuda":
        if not cuda_available():
            raise EmbeddingContractError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not mps_available():
            raise EmbeddingContractError("MPS was requested but is unavailable")
        return torch.device("mps")
    return torch.device("cpu")


def load_foundation(
    config: DownstreamConfig, *, device: torch.device | None = None
) -> LoadedFoundation:
    """Load only the encoder from a parity-verified validated export."""
    manifest_path = config.foundation.manifest_path
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError as exc:
        raise EmbeddingContractError(f"foundation manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise EmbeddingContractError(f"invalid foundation manifest: {manifest_path}") from exc
    parity = manifest.get("parity")
    if not isinstance(parity, dict) or parity.get("verified") is not True:
        raise EmbeddingContractError("foundation export parity is not verified")
    validation_path, validation_sha = _validated_evidence(manifest_path, manifest)
    configured_weights = manifest.get("weights")
    if not isinstance(configured_weights, str):
        raise EmbeddingContractError("foundation manifest does not identify safetensors weights")
    weights_path = Path(configured_weights)
    if not weights_path.is_file():
        weights_path = manifest_path.parent / "model.safetensors"
    if not weights_path.is_file():
        raise EmbeddingContractError(f"foundation weights not found: {weights_path}")
    actual_sha = sha256_file(weights_path)
    if actual_sha != config.foundation.weights_sha256:
        raise EmbeddingContractError("foundation safetensors digest does not match config")
    recorded_sha = manifest.get("weights_sha256")
    if recorded_sha is not None and recorded_sha != actual_sha:
        raise EmbeddingContractError("foundation safetensors digest does not match manifest")
    if device is None:
        device = resolve_embedding_device(config.run.device)
    model = MantisV2(
        return_transf_layer=config.foundation.return_transf_layer,
        output_token=config.foundation.output_token,
        device=str(device),
        pre_training=False,
    )
    state = load_file(weights_path)
    prefix = "encoder.backbone."
    encoder_state = {
        key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
    }
    if not encoder_state:
        raise EmbeddingContractError("foundation export contains no encoder tensors")
    incompatible = model.load_state_dict(encoder_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise EmbeddingContractError("foundation encoder state is incompatible")
    model = model.to(device).eval()
    with torch.inference_mode():
        fixture = torch.zeros(1, 1, config.foundation.input_length, device=device)
        output = model(fixture)
    if not isinstance(output, torch.Tensor) or output.ndim != 2:
        raise EmbeddingContractError("foundation encoder returned an invalid embedding")
    return LoadedFoundation(
        model=model,
        device=device,
        weights_path=weights_path,
        weights_sha256=actual_sha,
        validation_evidence_path=validation_path,
        validation_evidence_sha256=validation_sha,
        embedding_dim=output.shape[1],
        manifest=manifest,
    )


def _contexts(
    values: np.ndarray,
    end_indices: np.ndarray,
    config: DownstreamConfig,
) -> torch.Tensor:
    offsets = np.arange(config.data.context_bars, dtype=np.int64)
    rows = end_indices[:, None] - config.data.context_bars + 1 + offsets[None, :]
    context = values[rows].transpose(0, 2, 1).astype(np.float32, copy=False)
    mean = context.mean(axis=2, keepdims=True)
    std = np.maximum(context.std(axis=2, keepdims=True), 1e-6)
    context = np.clip((context - mean) / std, -10.0, 10.0)
    tensor = torch.from_numpy(context)
    return F.interpolate(
        tensor,
        size=config.foundation.input_length,
        mode="linear",
        align_corners=False,
    )


def iter_symbol_embeddings(
    candidates: pd.DataFrame,
    symbol: str,
    config: DownstreamConfig,
    foundation: LoadedFoundation,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Extract concatenated 1m/3m/15m channel embeddings in bounded batches."""
    if candidates.empty:
        raise EmbeddingContractError(f"candidate table is empty for {symbol}")
    streams = {
        timeframe: load_market_frame(config, symbol, timeframe)
        .loc[:, list(config.data.feature_columns)]
        .to_numpy(dtype=np.float32, copy=True)
        for timeframe in config.data.timeframes
    }
    batch_size = config.foundation.batch_size
    channels = len(config.data.feature_columns)
    for start in range(0, len(candidates), batch_size):
        stop = min(start + batch_size, len(candidates))
        timeframe_embeddings: list[torch.Tensor] = []
        with torch.inference_mode():
            for timeframe in config.data.timeframes:
                indices = candidates[f"{timeframe}_index"].iloc[start:stop].to_numpy(dtype=np.int64)
                context = _contexts(streams[timeframe], indices, config).to(foundation.device)
                batch = len(context)
                encoded = foundation.model(context.reshape(batch * channels, 1, -1))
                timeframe_embeddings.append(
                    encoded.reshape(batch, channels * foundation.embedding_dim)
                )
            combined = torch.cat(timeframe_embeddings, dim=1)
        dtype = np.float16 if config.foundation.storage_dtype == "float16" else np.float32
        result = combined.cpu().numpy().astype(dtype)
        if not np.isfinite(result).all():
            raise EmbeddingContractError(f"non-finite embeddings generated for {symbol}")
        yield start, stop, result
