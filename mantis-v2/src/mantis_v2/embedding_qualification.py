"""CPU-testable evidence gate for downstream CUDA embedding qualification."""

from __future__ import annotations

import json
import math
import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from mantis_v2.embedding_artifacts import (
    FOUR_TIMEFRAME_CONTRACT,
    EmbeddingArtifactError,
    EmbeddingIdentity,
    EmbeddingPerformance,
    compare_embedding_parity,
    scan_embedding_pairs,
    validate_embedding_identity,
)
from mantis_v2.model import sha256_file


@dataclass(frozen=True)
class EmbeddingQualificationConfig:
    purpose: Literal["matrix_scoring", "production"]
    source: dict[str, str]
    max_abs: float
    minimum_row_cosine: float
    projected_timeframes: int
    checkpoint_free_restart: bool


def load_embedding_qualification_config(
    path: str | Path,
) -> EmbeddingQualificationConfig:
    """Load the strict downstream embedding qualification policy."""
    try:
        raw = tomllib.loads(Path(path).read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EmbeddingArtifactError(f"cannot load embedding qualification config: {exc}") from exc
    if set(raw) != {"schema_version", "purpose", "timeframes", "source", "parity", "performance"}:
        raise EmbeddingArtifactError("invalid embedding qualification schema")
    if raw["schema_version"] != 1 or tuple(raw["timeframes"]) != FOUR_TIMEFRAME_CONTRACT:
        raise EmbeddingArtifactError("qualification requires the ordered four-timeframe contract")
    if raw["purpose"] not in {"matrix_scoring", "production"}:
        raise EmbeddingArtifactError("invalid embedding qualification purpose")
    source_keys = {"repository", "revision", "hub_model", "hub_revision", "weights_sha256"}
    if not isinstance(raw["source"], dict) or set(raw["source"]) != source_keys:
        raise EmbeddingArtifactError("invalid embedding qualification source pins")
    parity = raw["parity"]
    performance = raw["performance"]
    if not isinstance(parity, dict) or set(parity) != {"max_abs", "minimum_row_cosine"}:
        raise EmbeddingArtifactError("invalid embedding parity policy")
    if not isinstance(performance, dict) or set(performance) != {
        "projected_timeframes",
        "checkpoint_free_restart",
    }:
        raise EmbeddingArtifactError("invalid embedding performance policy")
    max_abs = parity["max_abs"]
    cosine = parity["minimum_row_cosine"]
    if not all(
        isinstance(value, int | float) and math.isfinite(value) for value in (max_abs, cosine)
    ):
        raise EmbeddingArtifactError("embedding parity tolerances must be finite")
    if max_abs < 0 or not 0 <= cosine <= 1:
        raise EmbeddingArtifactError("embedding parity tolerances are out of range")
    if performance != {"projected_timeframes": 4, "checkpoint_free_restart": True}:
        raise EmbeddingArtifactError(
            "embedding performance policy must require restart and four TF"
        )
    return EmbeddingQualificationConfig(
        purpose=raw["purpose"],
        source={key: str(value) for key, value in raw["source"].items()},
        max_abs=float(max_abs),
        minimum_row_cosine=float(cosine),
        projected_timeframes=4,
        checkpoint_free_restart=True,
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingArtifactError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise EmbeddingArtifactError(f"invalid {label}: {path}")
    return cast(dict[str, Any], value)


def _identity(raw: dict[str, Any]) -> EmbeddingIdentity:
    expected = {
        "export_role",
        "foundation_export_sha256",
        "producer_config_sha256",
        "corpus_sha256",
        "source_digest",
        "lock_digest",
        "timeframes",
        "feature_width",
    }
    if set(raw) != expected or not isinstance(raw["timeframes"], list):
        raise EmbeddingArtifactError("invalid embedding identity evidence")
    return EmbeddingIdentity(**{**raw, "timeframes": tuple(raw["timeframes"])})


def _validate_foundation_manifest(
    manifest_path: Path,
    identity: EmbeddingIdentity,
    config: EmbeddingQualificationConfig,
) -> None:
    if sha256_file(manifest_path) != identity.foundation_export_sha256:
        raise EmbeddingArtifactError("foundation export manifest digest mismatch")
    manifest = _load_json(manifest_path, "foundation export manifest")
    if manifest.get("export_role") != identity.export_role:
        raise EmbeddingArtifactError("foundation export role substitution")
    model = (
        manifest.get("config", {}).get("model")
        if isinstance(manifest.get("config"), dict)
        else None
    )
    if not isinstance(model, dict):
        raise EmbeddingArtifactError("foundation export lacks pinned model source")
    observed = {
        "repository": model.get("source_repository"),
        "revision": model.get("source_revision"),
        "hub_model": model.get("hub_model"),
        "hub_revision": model.get("hub_revision"),
        "weights_sha256": model.get("weights_sha256"),
    }
    if observed != config.source:
        raise EmbeddingArtifactError("foundation export source pins mismatch")


def _performance(raw: dict[str, Any]) -> EmbeddingPerformance:
    try:
        evidence = EmbeddingPerformance(**raw)
        rebuilt = EmbeddingPerformance.build(
            rows=evidence.rows,
            duration_seconds=evidence.duration_seconds,
            data_wait_seconds=evidence.data_wait_seconds,
            peak_vram_bytes=evidence.peak_vram_bytes,
            peak_rss_bytes=evidence.peak_rss_bytes,
            disk_bytes=evidence.disk_bytes,
            checkpoint_free_restart=evidence.checkpoint_free_restart,
            measured_timeframes=evidence.measured_timeframes,
            projected_timeframes=evidence.projected_timeframes,
        )
    except (TypeError, EmbeddingArtifactError) as exc:
        raise EmbeddingArtifactError("invalid embedding performance evidence") from exc
    if asdict(evidence) != asdict(rebuilt):
        raise EmbeddingArtifactError("derived embedding performance evidence mismatch")
    return evidence


def qualify_embedding_files(
    *,
    config_path: str | Path,
    identity_path: str | Path,
    foundation_manifest_path: str | Path,
    cpu_features_path: str | Path,
    cuda_features_path: str | Path,
    cpu_metadata_path: str | Path,
    cuda_metadata_path: str | Path,
    shard_directory: str | Path,
    performance_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate fixed parity, provenance, resume receipts, and resource evidence."""
    config = load_embedding_qualification_config(config_path)
    identity = _identity(_load_json(Path(identity_path), "embedding identity"))
    validate_embedding_identity(identity, purpose=config.purpose)
    _validate_foundation_manifest(Path(foundation_manifest_path), identity, config)
    pairs = scan_embedding_pairs(shard_directory, identity)
    if not pairs:
        raise EmbeddingArtifactError("qualification requires at least one committed shard pair")
    parity = compare_embedding_parity(
        np.load(cpu_features_path, allow_pickle=False),
        np.load(cuda_features_path, allow_pickle=False),
        pd.read_parquet(cpu_metadata_path),
        pd.read_parquet(cuda_metadata_path),
        max_abs_tolerance=config.max_abs,
        minimum_row_cosine=config.minimum_row_cosine,
    )
    performance = _performance(_load_json(Path(performance_path), "performance evidence"))
    result = {
        "schema_version": 1,
        "qualified": True,
        "purpose": config.purpose,
        "identity": asdict(identity),
        "source": config.source,
        "parity": parity,
        "performance": asdict(performance),
        "committed_pairs": len(pairs),
    }
    output = Path(output_path)
    if output.exists():
        raise EmbeddingArtifactError(f"qualification output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return result
