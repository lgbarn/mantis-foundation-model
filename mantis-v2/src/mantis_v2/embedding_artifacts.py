"""Fail-closed identities, publication, parity, and performance for embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from mantis_v2.model import sha256_file

FOUR_TIMEFRAME_CONTRACT = ("1min", "3min", "5min", "15min")


class EmbeddingArtifactError(RuntimeError):
    """Raised when an embedding artifact violates its durable contract."""


@dataclass(frozen=True)
class EmbeddingIdentity:
    export_role: Literal["diagnostic_candidate", "promoted"] | str
    foundation_export_sha256: str
    producer_config_sha256: str
    corpus_sha256: str
    source_digest: str
    lock_digest: str
    timeframes: tuple[str, ...]
    feature_width: int

    @property
    def digest(self) -> str:
        encoded = json.dumps(_identity_payload(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def _identity_payload(identity: EmbeddingIdentity) -> dict[str, Any]:
    payload = asdict(identity)
    payload["timeframes"] = list(identity.timeframes)
    return payload


def validate_embedding_identity(
    identity: EmbeddingIdentity,
    *,
    purpose: Literal["matrix_scoring", "production"],
) -> None:
    """Validate role separation and the exact ordered four-timeframe contract."""
    _validate_identity_shape(identity)
    if purpose == "production" and identity.export_role != "promoted":
        raise EmbeddingArtifactError("production embeddings require a promoted export")
    if purpose == "matrix_scoring" and identity.export_role != "diagnostic_candidate":
        raise EmbeddingArtifactError(
            "matrix scoring requires an explicitly labeled diagnostic_candidate export"
        )


def _validate_identity_shape(identity: EmbeddingIdentity) -> None:
    if identity.timeframes != FOUR_TIMEFRAME_CONTRACT:
        raise EmbeddingArtifactError(
            "embedding identity must use the ordered four-timeframe contract"
        )
    if identity.feature_width < 1:
        raise EmbeddingArtifactError("embedding feature width must be positive")
    digests = {
        "foundation_export_sha256": identity.foundation_export_sha256,
        "producer_config_sha256": identity.producer_config_sha256,
        "corpus_sha256": identity.corpus_sha256,
        "source_digest": identity.source_digest,
        "lock_digest": identity.lock_digest,
    }
    for name, value in digests.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise EmbeddingArtifactError(f"{name} must be a lowercase SHA-256 digest")
    if identity.export_role not in {"diagnostic_candidate", "promoted"}:
        raise EmbeddingArtifactError("embedding export role is invalid")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_record(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingArtifactError(f"invalid embedding pair receipt: {path}") from exc
    if not isinstance(value, dict):
        raise EmbeddingArtifactError(f"invalid embedding pair receipt: {path}")
    return cast(dict[str, Any], value)


def _verify_pair(
    record: dict[str, Any], identity: EmbeddingIdentity, *, expected_number: int
) -> None:
    if (
        record.get("identity") != _identity_payload(identity)
        or record.get("identity_digest") != identity.digest
    ):
        raise EmbeddingArtifactError("embedding pair identity mismatch")
    if record.get("number") != expected_number:
        raise EmbeddingArtifactError("embedding pair number mismatch")
    row_start = record.get("row_start")
    row_stop = record.get("row_stop")
    if not isinstance(row_start, int) or not isinstance(row_stop, int) or row_stop <= row_start:
        raise EmbeddingArtifactError("embedding pair row span is invalid")
    if record.get("rows") != row_stop - row_start:
        raise EmbeddingArtifactError("embedding pair row count mismatch")
    if record.get("feature_width") != identity.feature_width:
        raise EmbeddingArtifactError("embedding pair width mismatch")
    for label in ("features", "metadata"):
        file_record = record.get(label)
        if not isinstance(file_record, dict):
            raise EmbeddingArtifactError(f"embedding pair lacks {label} identity")
        path = Path(str(file_record.get("path", "")))
        if not path.is_file() or path.stat().st_size != file_record.get("size"):
            raise EmbeddingArtifactError(f"embedding pair {label} size mismatch")
        if sha256_file(path) != file_record.get("sha256"):
            raise EmbeddingArtifactError(f"embedding pair {label} digest mismatch")


def scan_embedding_pairs(
    directory: str | Path, identity: EmbeddingIdentity
) -> tuple[dict[str, Any], ...]:
    """Rehash every committed pair while leaving uncommitted partial files untouched."""
    _validate_identity_shape(identity)
    root = Path(directory)
    receipts = sorted(root.glob("pair-*.json")) if root.is_dir() else []
    records: list[dict[str, Any]] = []
    expected_row_start = 0
    for expected_number, path in enumerate(receipts):
        if path.name != f"pair-{expected_number:05d}.json":
            raise EmbeddingArtifactError("embedding pair receipt sequence has a gap")
        record = _load_record(path)
        _verify_pair(record, identity, expected_number=expected_number)
        if record["row_start"] != expected_row_start:
            raise EmbeddingArtifactError("embedding pair row span is not contiguous")
        expected_row_start = int(record["row_stop"])
        records.append(record)
    return tuple(records)


def publish_embedding_pair(
    directory: str | Path,
    number: int,
    row_start: int,
    features: np.ndarray,
    metadata: pd.DataFrame,
    identity: EmbeddingIdentity,
    *,
    after_write: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Publish feature/metadata files as one pair committed by a final receipt."""
    _validate_identity_shape(identity)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    values = np.asarray(features)
    if (
        values.ndim != 2
        or values.shape[1] != identity.feature_width
        or len(values) != len(metadata)
        or not np.isfinite(values).all()
    ):
        raise EmbeddingArtifactError("embedding pair values violate rows, width, or finiteness")
    receipt_path = root / f"pair-{number:05d}.json"
    if receipt_path.exists():
        record = _load_record(receipt_path)
        _verify_pair(record, identity, expected_number=number)
        if record["row_start"] != row_start or record["row_stop"] != row_start + len(values):
            raise EmbeddingArtifactError("embedding pair row span mismatch")
        return record

    feature_path = root / f"features-{number:05d}.npy"
    metadata_path = root / f"metadata-{number:05d}.parquet"
    if feature_path.exists():
        existing = np.load(feature_path, allow_pickle=False)
        if not np.array_equal(existing, values, equal_nan=False):
            raise EmbeddingArtifactError("partial embedding features do not match resumed values")
    else:
        _atomic_npy(feature_path, values)
        if after_write is not None:
            after_write("features")
    if metadata_path.exists():
        existing_metadata = pd.read_parquet(metadata_path)
        if not existing_metadata.equals(metadata.reset_index(drop=True)):
            raise EmbeddingArtifactError("partial embedding metadata do not match resumed values")
    else:
        _atomic_parquet(metadata_path, metadata.reset_index(drop=True))
        if after_write is not None:
            after_write("metadata")

    record = {
        "schema_version": 1,
        "number": number,
        "row_start": row_start,
        "row_stop": row_start + len(values),
        "rows": len(values),
        "feature_width": values.shape[1],
        "identity": _identity_payload(identity),
        "identity_digest": identity.digest,
        "features": _file_record(feature_path),
        "metadata": _file_record(metadata_path),
    }
    _atomic_json(receipt_path, record)
    if after_write is not None:
        after_write("receipt")
    return record


def compare_embedding_parity(
    cpu: np.ndarray,
    cuda: np.ndarray,
    cpu_metadata: pd.DataFrame,
    cuda_metadata: pd.DataFrame,
    *,
    max_abs_tolerance: float = 0.01,
    minimum_row_cosine: float = 0.999,
) -> dict[str, Any]:
    """Compare injected CPU/CUDA outputs under the downstream parity contract."""
    cpu_values = np.asarray(cpu, dtype=np.float64)
    cuda_values = np.asarray(cuda, dtype=np.float64)
    if cpu_values.shape != cuda_values.shape or cpu_values.ndim != 2:
        raise EmbeddingArtifactError("CPU/CUDA embedding shapes differ")
    if not np.isfinite(cpu_values).all() or not np.isfinite(cuda_values).all():
        raise EmbeddingArtifactError("CPU/CUDA embeddings must be finite")
    if not cpu_metadata.equals(cuda_metadata):
        raise EmbeddingArtifactError("CPU/CUDA metadata order or values differ")
    denominators = np.linalg.norm(cpu_values, axis=1) * np.linalg.norm(cuda_values, axis=1)
    if np.any(denominators == 0):
        raise EmbeddingArtifactError("CPU/CUDA embedding rows must have non-zero norms")
    row_cosines = np.sum(cpu_values * cuda_values, axis=1) / denominators
    maximum = float(np.max(np.abs(cpu_values - cuda_values), initial=0.0))
    minimum = float(np.min(row_cosines, initial=1.0))
    if maximum > max_abs_tolerance:
        raise EmbeddingArtifactError("CPU/CUDA maximum absolute difference exceeds tolerance")
    if minimum < minimum_row_cosine:
        raise EmbeddingArtifactError("CPU/CUDA row cosine falls below tolerance")
    return {
        "max_abs_difference": maximum,
        "minimum_row_cosine": minimum,
        "metadata_exact": True,
        "rows": len(cpu_values),
    }


@dataclass(frozen=True)
class EmbeddingPerformance:
    rows: int
    duration_seconds: float
    rows_per_second: float
    data_wait_seconds: float
    peak_vram_bytes: int
    peak_rss_bytes: int
    disk_bytes: int
    disk_bytes_per_row: float
    checkpoint_free_restart: bool
    measured_timeframes: int
    projected_timeframes: int
    projected_four_timeframe_disk_bytes: int

    @classmethod
    def build(
        cls,
        *,
        rows: int,
        duration_seconds: float,
        data_wait_seconds: float,
        peak_vram_bytes: int,
        peak_rss_bytes: int,
        disk_bytes: int,
        checkpoint_free_restart: bool,
        measured_timeframes: int,
        projected_timeframes: int,
    ) -> EmbeddingPerformance:
        numbers = (duration_seconds, data_wait_seconds)
        integers = (
            rows,
            peak_vram_bytes,
            peak_rss_bytes,
            disk_bytes,
            measured_timeframes,
            projected_timeframes,
        )
        if (
            rows <= 0
            or duration_seconds <= 0
            or any(not math.isfinite(value) or value < 0 for value in numbers)
            or any(value < 0 for value in integers)
            or measured_timeframes <= 0
            or projected_timeframes != 4
            or not isinstance(checkpoint_free_restart, bool)
        ):
            raise EmbeddingArtifactError("invalid embedding performance measurement")
        projection = math.ceil(disk_bytes * projected_timeframes / measured_timeframes)
        return cls(
            rows=rows,
            duration_seconds=float(duration_seconds),
            rows_per_second=rows / duration_seconds,
            data_wait_seconds=float(data_wait_seconds),
            peak_vram_bytes=peak_vram_bytes,
            peak_rss_bytes=peak_rss_bytes,
            disk_bytes=disk_bytes,
            disk_bytes_per_row=disk_bytes / rows,
            checkpoint_free_restart=checkpoint_free_restart,
            measured_timeframes=measured_timeframes,
            projected_timeframes=projected_timeframes,
            projected_four_timeframe_disk_bytes=projection,
        )
