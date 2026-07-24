"""Freeze the one strategy-owned fixture consumed by foundation diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mantis_v2.downstream_config import TrendMagicStrategyConfig, load_downstream_config
from mantis_v2.embedding import iter_symbol_embeddings, load_foundation
from mantis_v2.embedding_artifacts import FOUR_TIMEFRAME_CONTRACT
from mantis_v2.strategy import build_symbol_candidates


class FoundationFixtureError(RuntimeError):
    """Raised when the fixed diagnostic fixture cannot be frozen causally."""


def _embedding_runtime_config(config: Any) -> Any:
    """Apply an explicit executor-only override after fixture identity validation."""
    device = os.environ.get("MANTIS_V2_EMBED_DEVICE")
    if device is not None and device not in {"cpu", "cuda", "mps"}:
        raise FoundationFixtureError("MANTIS_V2_EMBED_DEVICE must be cpu, cuda, or mps")
    data_root = os.environ.get("MANTIS_V2_EMBED_DATA_ROOT")
    corpus_manifest = os.environ.get("MANTIS_V2_EMBED_CORPUS_MANIFEST")
    if bool(data_root) != bool(corpus_manifest):
        raise FoundationFixtureError(
            "MANTIS_V2_EMBED_DATA_ROOT and MANTIS_V2_EMBED_CORPUS_MANIFEST must be set together"
        )
    if corpus_manifest is not None:
        runtime_manifest = Path(corpus_manifest)
        if (
            not runtime_manifest.is_file()
            or _sha256_file(runtime_manifest) != config.data.corpus_manifest_sha256
        ):
            raise FoundationFixtureError(
                "MANTIS_V2_EMBED_CORPUS_MANIFEST does not match the configured corpus hash"
            )
    run = replace(config.run, device=device) if device is not None else config.run
    data = (
        replace(
            config.data,
            root=Path(data_root),
            corpus_manifest_path=runtime_manifest,
        )
        if data_root is not None
        else config.data
    )
    return replace(config, run=run, data=data)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _row_id(row: pd.Series) -> str:
    identity = {
        "symbol": str(row["symbol"]),
        "decision_index": int(row["decision_index"]),
        "decision_at": pd.Timestamp(row["decision_ts"]).isoformat(),
        "label_end_at": pd.Timestamp(row["label_end_ts"]).isoformat(),
    }
    return _digest(identity)


def _partition_cap(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    generator = np.random.Generator(np.random.PCG64(0))
    return np.sort(generator.choice(indices, size=maximum, replace=False))


def freeze_diagnostic_fixture(config_path: str | Path, output_root: str | Path) -> Path:
    """Construct labels once outside the scorer and publish an immutable 25k/25k fixture."""
    source_path = Path(config_path).resolve()
    config = load_downstream_config(source_path)
    if (
        not isinstance(config.strategy, TrendMagicStrategyConfig)
        or config.strategy.target_r != 3.0
        or config.data.timeframes != FOUR_TIMEFRAME_CONTRACT
        or config.data.decision_timeframe != "3min"
        or config.evaluation.allow_holdout
        or config.data.holdout_start.isoformat() != "2026-01-01T00:00:00+00:00"
    ):
        raise FoundationFixtureError(
            "fixture requires the locked four-timeframe fixed-3R Trend Magic contract"
        )
    frames: list[pd.DataFrame] = []
    for symbol_order, symbol in enumerate(config.data.symbols):
        candidates = build_symbol_candidates(config, symbol, split="pre_holdout")
        required = {
            "symbol",
            "decision_index",
            "decision_ts",
            "label_end_ts",
            "label",
            *(f"{timeframe}_index" for timeframe in FOUR_TIMEFRAME_CONTRACT),
        }
        if not required.issubset(candidates.columns) or candidates.empty:
            raise FoundationFixtureError(f"candidate schema is incomplete for {symbol}")
        selected = candidates.loc[:, sorted(required)].copy()
        selected["_symbol_order"] = symbol_order
        frames.append(selected)
    combined = pd.concat(frames, ignore_index=True)
    combined["decision_ts"] = pd.to_datetime(combined["decision_ts"], utc=True, errors="raise")
    combined["label_end_ts"] = pd.to_datetime(combined["label_end_ts"], utc=True, errors="raise")
    combined = combined.sort_values(
        ["_symbol_order", "decision_ts", "decision_index"], kind="stable"
    ).reset_index(drop=True)
    fit_boundary = pd.Timestamp("2024-01-01T00:00:00+00:00")
    score_start = pd.Timestamp("2025-01-01T00:00:00+00:00")
    holdout = pd.Timestamp("2026-01-01T00:00:00+00:00")
    fit = (combined["decision_ts"] < fit_boundary) & (combined["label_end_ts"] < fit_boundary)
    score = (
        (combined["decision_ts"] >= score_start)
        & (combined["decision_ts"] < holdout)
        & (combined["label_end_ts"] >= score_start)
        & (combined["label_end_ts"] < holdout)
    )
    fit_indices = _partition_cap(np.flatnonzero(fit.to_numpy()), 25000)
    score_indices = _partition_cap(np.flatnonzero(score.to_numpy()), 25000)
    if not len(fit_indices) or not len(score_indices):
        raise FoundationFixtureError("fixture fit and score partitions must both be non-empty")
    chosen = np.concatenate((fit_indices, score_indices))
    frozen = combined.iloc[chosen].reset_index(drop=True)
    if set(frozen["label"].unique()) != {0, 1}:
        raise FoundationFixtureError("frozen fixture is missing a label class")
    frozen["row_id"] = frozen.apply(_row_id, axis=1)
    if frozen["row_id"].duplicated().any():
        raise FoundationFixtureError("frozen fixture row identity collision")
    rows = frozen.rename(
        columns={"decision_ts": "decision_at", "label_end_ts": "label_end_at"}
    ).loc[:, ["row_id", "decision_at", "label_end_at", "label"]]
    contexts = frozen.loc[
        :,
        [
            "row_id",
            "symbol",
            *(f"{timeframe}_index" for timeframe in FOUR_TIMEFRAME_CONTRACT),
        ],
    ]
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".fixture.", dir=output))
    try:
        rows_path = temporary / "rows.parquet"
        contexts_path = temporary / "contexts.parquet"
        rows.to_parquet(rows_path, index=False)
        contexts.to_parquet(contexts_path, index=False)
        row_ids_digest = hashlib.sha256(
            json.dumps(rows["row_id"].tolist(), separators=(",", ":")).encode()
        ).hexdigest()
        corpus_path = config.data.corpus_manifest_path
        if corpus_path is None or not corpus_path.is_file():
            raise FoundationFixtureError("fixture corpus manifest is missing")
        if _sha256_file(corpus_path) != config.data.corpus_manifest_sha256:
            raise FoundationFixtureError("fixture corpus manifest hash mismatch")
        core: dict[str, Any] = {
            "schema_version": 1,
            "label_semantics": "fixed_3r_trend_magic",
            "holdout_start": "2026-01-01T00:00:00+00:00",
            "fit_before": "2024-01-01T00:00:00+00:00",
            "score_start": "2025-01-01T00:00:00+00:00",
            "score_end": "2026-01-01T00:00:00+00:00",
            "selection": {
                "rng": "PCG64",
                "seed": 0,
                "fit_cap": 25000,
                "score_cap": 25000,
            },
            "timeframes": list(FOUR_TIMEFRAME_CONTRACT),
            "symbols": list(config.data.symbols),
            "strategy_contract": config.strategy_contract,
            "source_config": {
                "path": str(source_path),
                "sha256": _sha256_file(source_path),
                "config_digest": config.digest,
            },
            "corpus_manifest": {
                "path": str(corpus_path.resolve()),
                "sha256": config.data.corpus_manifest_sha256,
            },
            "rows": {
                "path": "rows.parquet",
                "sha256": _sha256_file(rows_path),
                "count": len(rows),
                "fit_count": len(fit_indices),
                "score_count": len(score_indices),
            },
            "contexts": {
                "path": "contexts.parquet",
                "sha256": _sha256_file(contexts_path),
                "count": len(contexts),
            },
            "row_ids_sha256": row_ids_digest,
        }
        fixture_digest = _digest(core)
        manifest = {**core, "fixture_digest": fixture_digest}
        (temporary / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
        destination = output / fixture_digest
        if destination.exists():
            existing = destination / "manifest.json"
            if not existing.is_file() or existing.read_bytes() != (_canonical(manifest) + b"\n"):
                raise FoundationFixtureError("immutable fixture destination differs")
            return existing
        temporary.rename(destination)
        return destination / "manifest.json"
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _read_manifest(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise FoundationFixtureError(f"invalid {description}: {path}") from exc
    if not isinstance(value, dict):
        raise FoundationFixtureError(f"invalid {description}: {path}")
    return value


def embed_diagnostic_fixture(
    config_path: str | Path,
    fixture_manifest: str | Path,
    foundation_manifest: str | Path,
    output_root: str | Path,
) -> Path:
    """Embed a frozen fixture in its original row order and publish it atomically."""
    source_config = Path(config_path).resolve()
    fixture_path = Path(fixture_manifest).resolve()
    foundation_path = Path(foundation_manifest).resolve()
    config = load_downstream_config(source_config)
    fixture = _read_manifest(fixture_path, "diagnostic fixture manifest")
    fixture_core = {key: value for key, value in fixture.items() if key != "fixture_digest"}
    if (
        fixture.get("schema_version") != 1
        or fixture.get("fixture_digest") != _digest(fixture_core)
        or fixture.get("timeframes") != list(config.data.timeframes)
        or fixture.get("symbols") != list(config.data.symbols)
        or fixture.get("strategy_contract") != config.strategy_contract
        or fixture.get("source_config", {}).get("config_digest") != config.digest
    ):
        raise FoundationFixtureError("fixture does not match the embedding configuration")
    contexts_record = fixture.get("contexts")
    if not isinstance(contexts_record, dict):
        raise FoundationFixtureError("fixture contexts are missing")
    contexts_path = Path(str(contexts_record.get("path", "")))
    if not contexts_path.is_absolute():
        contexts_path = fixture_path.parent / contexts_path
    if _sha256_file(contexts_path) != contexts_record.get("sha256"):
        raise FoundationFixtureError("fixture contexts hash mismatch")
    try:
        contexts = pd.read_parquet(contexts_path)
    except Exception as exc:
        raise FoundationFixtureError("fixture contexts are unreadable") from exc
    expected_columns = [
        "row_id",
        "symbol",
        *(f"{timeframe}_index" for timeframe in config.data.timeframes),
    ]
    if contexts.columns.tolist() != expected_columns or len(contexts) != contexts_record.get(
        "count"
    ):
        raise FoundationFixtureError("fixture context schema or count mismatch")
    row_ids_digest = hashlib.sha256(
        json.dumps(contexts["row_id"].astype(str).tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    if row_ids_digest != fixture.get("row_ids_sha256"):
        raise FoundationFixtureError("fixture context row identity mismatch")

    export = _read_manifest(foundation_path, "foundation export manifest")
    export_role = export.get("export_role")
    weights_sha256 = export.get("weights_sha256")
    seed = export.get("config", {}).get("run", {}).get("seed")
    if (
        export_role not in {"diagnostic_candidate", "promoted"}
        or not isinstance(weights_sha256, str)
        or len(weights_sha256) != 64
        or not isinstance(seed, int)
    ):
        raise FoundationFixtureError("foundation export identity is incomplete")
    embedding_config = replace(
        _embedding_runtime_config(config),
        foundation=replace(
            config.foundation,
            manifest_path=foundation_path,
            weights_sha256=weights_sha256,
            export_role=export_role,
        ),
    )
    loaded = load_foundation(embedding_config)
    if loaded.weights_sha256 != weights_sha256:
        raise FoundationFixtureError("loaded foundation weights identity mismatch")

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".features.", dir=output))
    try:
        feature_path = temporary / "features.npy"
        features: np.memmap | None = None
        width: int | None = None
        written = np.zeros(len(contexts), dtype=bool)
        dtype = np.float16 if config.foundation.storage_dtype == "float16" else np.float32
        for symbol in config.data.symbols:
            positions = np.flatnonzero(contexts["symbol"].to_numpy() == symbol)
            if not len(positions):
                raise FoundationFixtureError(f"fixture has no contexts for {symbol}")
            candidates = contexts.iloc[positions].reset_index(drop=True)
            for start, stop, values in iter_symbol_embeddings(
                candidates, symbol, embedding_config, loaded
            ):
                if not 0 <= start < stop <= len(positions) or values.shape[0] != stop - start:
                    raise FoundationFixtureError(
                        "foundation embedder returned an invalid row range"
                    )
                if width is None:
                    if values.ndim != 2 or values.shape[1] <= 0:
                        raise FoundationFixtureError(
                            "foundation embedder returned an invalid shape"
                        )
                    width = int(values.shape[1])
                    features = np.lib.format.open_memmap(
                        feature_path, mode="w+", dtype=dtype, shape=(len(contexts), width)
                    )
                if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
                    raise FoundationFixtureError(
                        "foundation embedder returned incompatible or non-finite features"
                    )
                row_positions = positions[start:stop]
                if written[row_positions].any():
                    raise FoundationFixtureError("fixture feature rows were written more than once")
                assert features is not None
                features[row_positions] = values.astype(dtype, copy=False)
                written[row_positions] = True
        if features is None or width is None or not written.all():
            raise FoundationFixtureError("fixture feature extraction is incomplete")
        features.flush()
        del features
        foundation_sha256 = _sha256_file(foundation_path)
        core: dict[str, Any] = {
            "schema_version": 1,
            "fixture_digest": fixture["fixture_digest"],
            "row_ids_sha256": fixture["row_ids_sha256"],
            "seed": seed,
            "candidate_id": foundation_sha256,
            "foundation_manifest": {
                "path": str(foundation_path),
                "sha256": foundation_sha256,
                "weights_sha256": weights_sha256,
                "export_role": export_role,
            },
            "producer_config": {
                "path": str(source_config),
                "sha256": _sha256_file(source_config),
                "config_digest": config.digest,
            },
            "features": {
                "path": "features.npy",
                "sha256": _sha256_file(feature_path),
                "rows": len(contexts),
                "width": width,
                "dtype": np.dtype(dtype).name,
            },
        }
        feature_digest = _digest(core)
        manifest = {**core, "feature_digest": feature_digest}
        manifest_bytes = _canonical(manifest) + b"\n"
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        destination = output / feature_digest
        if destination.exists():
            existing = destination / "manifest.json"
            if not existing.is_file() or existing.read_bytes() != manifest_bytes:
                raise FoundationFixtureError("immutable feature destination differs")
            if _sha256_file(destination / "features.npy") != manifest["features"]["sha256"]:
                raise FoundationFixtureError("immutable feature artifact hash mismatch")
            return existing
        temporary.rename(destination)
        return destination / "manifest.json"
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
