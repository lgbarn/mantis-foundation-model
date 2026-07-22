"""Read-only proper-score probe for one frozen Trend Magic diagnostic fixture."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


class DiagnosticError(RuntimeError):
    """Raised when the frozen fixture or diagnostic probe fails closed."""


_PROBE = {
    "scaler": "StandardScaler",
    "classifier": "LogisticRegression",
    "C": 0.0001,
    "solver": "lbfgs",
    "max_iter": 1000,
    "tol": 0.0001,
    "class_weight": "balanced",
    "random_state": 0,
    "threshold": None,
    "calibrator": None,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as exc:
        raise DiagnosticError(f"diagnostic artifact is missing: {path}") from exc
    return digest.hexdigest()


def _read_json(path: str | Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"invalid {description}: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"invalid {description}: {path}")
    return value


def _atomic_json_idempotent(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise DiagnosticError(f"diagnostic result already exists with different bytes: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise DiagnosticError(
                    f"diagnostic result already exists with different bytes: {path}"
                ) from None
    finally:
        Path(temporary).unlink(missing_ok=True)


def _validated_fixture(
    path: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    manifest_path = Path(path).resolve()
    manifest = _read_json(manifest_path, "diagnostic fixture manifest")
    if manifest.get("schema_version") != 1:
        raise DiagnosticError("unsupported diagnostic fixture schema")
    core = {key: value for key, value in manifest.items() if key != "fixture_digest"}
    if manifest.get("fixture_digest") != _digest(core):
        raise DiagnosticError("diagnostic fixture digest mismatch")
    if manifest.get("label_semantics") != "fixed_3r_trend_magic":
        raise DiagnosticError("diagnostic fixture label semantics are not fixed 3R Trend Magic")
    if manifest.get("holdout_start") != "2026-01-01T00:00:00+00:00":
        raise DiagnosticError("diagnostic fixture holdout boundary changed")
    selection = manifest.get("selection")
    if selection != {
        "rng": "PCG64",
        "seed": 0,
        "fit_cap": 25000,
        "score_cap": 25000,
    }:
        raise DiagnosticError("diagnostic fixture row selection contract changed")
    rows_record = manifest.get("rows")
    if not isinstance(rows_record, dict):
        raise DiagnosticError("diagnostic fixture rows are missing")
    rows_path = Path(str(rows_record.get("path", "")))
    if not rows_path.is_absolute():
        rows_path = manifest_path.parent / rows_path
    if _sha256_file(rows_path) != rows_record.get("sha256"):
        raise DiagnosticError("diagnostic fixture rows hash mismatch")
    try:
        rows = pd.read_parquet(rows_path)
    except Exception as exc:
        raise DiagnosticError("diagnostic fixture rows are unreadable") from exc
    required = {"row_id", "decision_at", "label_end_at", "label"}
    if set(rows.columns) != required or len(rows) != rows_record.get("count"):
        raise DiagnosticError("diagnostic fixture row schema or count mismatch")
    if rows["row_id"].isna().any() or rows["row_id"].duplicated().any():
        raise DiagnosticError("diagnostic fixture row IDs are missing or duplicated")
    row_ids_digest = hashlib.sha256(
        json.dumps(rows["row_id"].astype(str).tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    if row_ids_digest != manifest.get("row_ids_sha256"):
        raise DiagnosticError("diagnostic fixture row identity mismatch")
    try:
        decision_at = pd.to_datetime(rows["decision_at"], utc=True)
        label_end_at = pd.to_datetime(rows["label_end_at"], utc=True)
    except (TypeError, ValueError) as exc:
        raise DiagnosticError("diagnostic fixture timestamps are invalid") from exc
    fit_before = pd.Timestamp(str(manifest.get("fit_before")))
    score_start = pd.Timestamp(str(manifest.get("score_start")))
    score_end = pd.Timestamp(str(manifest.get("score_end")))
    if (fit_before, score_start, score_end) != (
        pd.Timestamp("2024-01-01T00:00:00+00:00"),
        pd.Timestamp("2025-01-01T00:00:00+00:00"),
        pd.Timestamp("2026-01-01T00:00:00+00:00"),
    ):
        raise DiagnosticError("diagnostic fixture time partitions changed")
    if ((decision_at >= fit_before) & (decision_at < score_start)).any() or (
        (label_end_at >= fit_before) & (label_end_at < score_start)
    ).any():
        raise DiagnosticError("diagnostic fixture contains excluded 2024 rows")
    if (decision_at >= score_end).any() or (label_end_at >= score_end).any():
        raise DiagnosticError("diagnostic fixture accesses the sealed holdout")
    fit_mask = (decision_at < fit_before) & (label_end_at < fit_before)
    score_mask = (
        (decision_at >= score_start)
        & (decision_at < score_end)
        & (label_end_at >= score_start)
        & (label_end_at < score_end)
    )
    if not (fit_mask | score_mask).all():
        raise DiagnosticError("diagnostic fixture rows fall outside fit and score partitions")
    fit_index = np.flatnonzero(fit_mask.to_numpy())
    score_index = np.flatnonzero(score_mask.to_numpy())
    if not 0 < len(fit_index) <= 25000 or not 0 < len(score_index) <= 25000:
        raise DiagnosticError("diagnostic fixture exceeds the frozen row caps")
    labels = rows["label"].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise DiagnosticError("diagnostic fixture labels must be binary")
    for name, index in (("fit", fit_index), ("score", score_index)):
        if set(np.unique(labels[index])) != {0, 1}:
            raise DiagnosticError(f"diagnostic fixture {name} partition is missing a class")
    return manifest, rows, fit_index, score_index


def _features(
    path: str | Path, fixture: Mapping[str, Any], expected_rows: int
) -> tuple[dict[str, Any], np.ndarray]:
    manifest_path = Path(path).resolve()
    manifest = _read_json(manifest_path, "diagnostic feature manifest")
    if manifest.get("schema_version") != 1:
        raise DiagnosticError("unsupported diagnostic feature schema")
    core = {key: value for key, value in manifest.items() if key != "feature_digest"}
    if manifest.get("feature_digest") != _digest(core):
        raise DiagnosticError("diagnostic feature manifest digest mismatch")
    if manifest.get("fixture_digest") != fixture.get("fixture_digest"):
        raise DiagnosticError("diagnostic feature fixture identity mismatch")
    if manifest.get("row_ids_sha256") != fixture.get("row_ids_sha256"):
        raise DiagnosticError("diagnostic feature row identity mismatch")
    record = manifest.get("features")
    if not isinstance(record, dict):
        raise DiagnosticError("diagnostic feature artifact is missing")
    feature_path = Path(str(record.get("path", "")))
    if not feature_path.is_absolute():
        feature_path = manifest_path.parent / feature_path
    if _sha256_file(feature_path) != record.get("sha256"):
        raise DiagnosticError("diagnostic feature hash mismatch")
    try:
        values = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DiagnosticError("diagnostic features are unreadable") from exc
    if (
        values.ndim != 2
        or values.shape != (record.get("rows"), record.get("width"))
        or len(values) != expected_rows
        or values.shape[1] <= 0
    ):
        raise DiagnosticError("diagnostic feature shape mismatch")
    if not np.isfinite(values).all():
        raise DiagnosticError("diagnostic features contain non-finite values")
    return manifest, np.asarray(values, dtype=np.float64)


def _balanced_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    if (counts == 0).any():
        raise DiagnosticError("diagnostic score partition is missing a class")
    return len(labels) / (2.0 * counts[labels.astype(np.int64)])


def _calibration(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[list[dict[str, Any]], float]:
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        selected = (
            (probabilities >= lower) & (probabilities <= upper)
            if index == 9
            else (probabilities >= lower) & (probabilities < upper)
        )
        count = int(selected.sum())
        mean_probability = float(probabilities[selected].mean()) if count else None
        positive_rate = float(labels[selected].mean()) if count else None
        if count and mean_probability is not None and positive_rate is not None:
            ece += count / len(labels) * abs(mean_probability - positive_rate)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_probability": mean_probability,
                "positive_rate": positive_rate,
            }
        )
    return bins, ece


def _score(
    values: np.ndarray, labels: np.ndarray, fit: np.ndarray, score: np.ndarray
) -> dict[str, Any]:
    scaler = StandardScaler()
    fit_values = scaler.fit_transform(values[fit])
    score_values = scaler.transform(values[score])
    classifier = LogisticRegression(
        C=0.0001,
        solver="lbfgs",
        max_iter=1000,
        tol=0.0001,
        class_weight="balanced",
        random_state=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(fit_values, labels[fit])
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise DiagnosticError("diagnostic LogisticRegression did not converge")
    probabilities = classifier.predict_proba(score_values)[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise DiagnosticError("diagnostic probabilities are non-finite or out of range")
    score_labels = labels[score].astype(np.int64)
    weights = _balanced_weights(score_labels)
    weighted_brier = float(np.average((probabilities - score_labels) ** 2, weights=weights))
    weighted_log_loss = float(
        log_loss(score_labels, probabilities, sample_weight=weights, labels=[0, 1])
    )
    auc = float(roc_auc_score(score_labels, probabilities))
    calibration_bins, ece = _calibration(score_labels, probabilities)
    metrics = {
        "log_loss": weighted_log_loss,
        "brier": weighted_brier,
        "roc_auc": auc,
        "ece": ece,
        "calibration_bins": calibration_bins,
    }
    if any(not math.isfinite(value) for value in (weighted_log_loss, weighted_brier, auc, ece)):
        raise DiagnosticError("diagnostic metrics are non-finite")
    return metrics


def score_diagnostic_fixture(
    fixture_manifest: str | Path,
    candidate_manifest: str | Path,
    reference_manifest: str | Path,
    output: str | Path,
) -> Path:
    """Score two precomputed embedding sets on one immutable fixture without constructing labels."""
    fixture, rows, fit_index, score_index = _validated_fixture(fixture_manifest)
    candidate_record, candidate = _features(candidate_manifest, fixture, len(rows))
    reference_record, reference = _features(reference_manifest, fixture, len(rows))
    if candidate_record.get("seed") != reference_record.get("seed"):
        raise DiagnosticError("diagnostic candidate and reference seeds do not match")
    result_core = {
        "schema_version": 1,
        "fixture_digest": fixture["fixture_digest"],
        "row_ids_sha256": fixture["row_ids_sha256"],
        "seed": candidate_record["seed"],
        "fit_rows": len(fit_index),
        "score_rows": len(score_index),
        "probe": _PROBE,
        "candidate_id": candidate_record.get("candidate_id"),
        "reference_id": reference_record.get("candidate_id"),
        "candidate": _score(candidate, rows["label"].to_numpy(), fit_index, score_index),
        "reference": _score(reference, rows["label"].to_numpy(), fit_index, score_index),
    }
    result = {**result_core, "result_digest": _digest(result_core)}
    path = Path(output)
    _atomic_json_idempotent(path, result)
    return path
