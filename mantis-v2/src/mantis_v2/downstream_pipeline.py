"""Stage-oriented downstream MantisV2 trading pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mantis_v2.contamination import report_digest
from mantis_v2.downstream_config import DownstreamConfig, load_downstream_config
from mantis_v2.embedding import iter_symbol_embeddings, load_foundation
from mantis_v2.instrumentation import (
    RunInstrumentation,
    collect_resource_metrics,
    synchronize_device,
)
from mantis_v2.model import sha256_file
from mantis_v2.strategy import build_symbol_candidates, market_path
from mantis_v2.topstep import simulate_topstep
from mantis_v2.walk_forward import (
    HeadConvergenceError,
    PortableHead,
    build_folds,
    deterministic_cap,
    fit_logistic_head,
    fold_masks,
    predict_head,
    probability_metrics,
)


class DownstreamPipelineError(RuntimeError):
    """Raised when a downstream stage fails closed."""


def artifact_root(config: DownstreamConfig) -> Path:
    return config.run.artifact_root / config.run.name


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npy(values: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def _require_new(path: Path, config: DownstreamConfig) -> None:
    if path.exists() and not config.run.allow_overwrite:
        raise DownstreamPipelineError(
            f"stage output exists: {path}; choose a new run.name or set run.allow_overwrite=true"
        )


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DownstreamPipelineError(f"required file does not exist: {path}")
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)}


def _source_digest() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for path in sorted((repository_root / "mantis-v2" / "src").rglob("*.py")):
        digest.update(str(path.relative_to(repository_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _manifest_base(config: DownstreamConfig, stage: str) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[3]
    lock_path = repository_root / "uv.lock"
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "workflow_digest": config.workflow_digest,
        "source_digest": _source_digest(),
        "lock_digest": sha256_file(lock_path),
    }
    if config.strategy_contract is not None:
        manifest["strategy_contract"] = config.strategy_contract
    return manifest


def _record_stage_instrumentation(
    config: DownstreamConfig,
    stage: str,
    started: float,
    progress: dict[str, int | float],
    device: torch.device,
) -> dict[str, Any]:
    root = artifact_root(config)
    synchronize_device(device)
    telemetry: dict[str, Any] = {
        "stage": stage,
        "duration_seconds": max(time.perf_counter() - started, 0.0),
        **progress,
        **collect_resource_metrics(root, device),
    }
    path = root / "instrumentation" / f"{stage}.json"
    _atomic_json(path, telemetry)
    writer = RunInstrumentation(root)
    writer.text(
        f"stage/{stage}/metadata",
        {"stage": stage, "seed": config.run.seed, "device": str(device)},
        0,
    )
    writer.scalars(
        {
            f"stage/{stage}/completed": 1,
            f"stage/{stage}/duration_seconds": float(telemetry["duration_seconds"]),
            **{f"stage/{stage}/{key}": value for key, value in progress.items()},
            "host/rss_bytes": int(telemetry["host_rss_bytes"]),
            "filesystem/free_bytes": int(telemetry["filesystem_free_bytes"]),
        },
        int(progress.get("step", 0)),
    )
    writer.scalars(
        {
            tag: float(value)
            for tag, value in {
                "cuda/allocated_bytes": telemetry["cuda_allocated_bytes"],
                "cuda/reserved_bytes": telemetry["cuda_reserved_bytes"],
                "cuda/utilization_percent": telemetry["cuda_utilization_percent"],
            }.items()
            if value is not None
        },
        int(progress.get("step", 0)),
    )
    writer.close()
    return {"path": str(path), **telemetry}


def verify_contract(config: DownstreamConfig) -> dict[str, Any]:
    """Report the validated downstream recipe without reading data or writing artifacts."""
    return {
        "run": config.run.name,
        "workflow_digest": config.workflow_digest,
        "embedding_contract_digest": config.embedding_contract_digest,
        "strategy_contract": config.strategy_contract,
    }


def _manifest(path: Path, config: DownstreamConfig, expected_stage: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DownstreamPipelineError(
            f"missing or invalid {expected_stage} manifest: {path}"
        ) from exc
    expected = _manifest_base(config, expected_stage)
    for key in ("stage", "workflow_digest", "source_digest", "lock_digest"):
        if value.get(key) != expected[key]:
            raise DownstreamPipelineError(f"{expected_stage} manifest identity mismatch: {key}")
    if "strategy_contract" in value and value["strategy_contract"] != expected.get(
        "strategy_contract"
    ):
        raise DownstreamPipelineError(
            f"{expected_stage} manifest identity mismatch: strategy_contract"
        )
    return cast(dict[str, Any], value)


def _embedding_manifest_input(
    config: DownstreamConfig,
) -> tuple[dict[str, Any], Path, str]:
    configured = config.walk_forward.embed_manifest_path
    if configured is None:
        path = artifact_root(config) / "embed" / "manifest.json"
        manifest = _manifest(path, config, "embed")
    else:
        path = configured.resolve()
        expected_sha = config.walk_forward.embed_manifest_sha256
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise DownstreamPipelineError("reusable embed manifest digest mismatch")
        try:
            manifest = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise DownstreamPipelineError("reusable embed manifest is invalid JSON") from exc
        if manifest.get("schema_version") != 1 or manifest.get("stage") != "embed":
            raise DownstreamPipelineError("reusable embed manifest contract is invalid")
        producer_path_value = config.walk_forward.embed_producer_config_path
        if producer_path_value is None:
            raise DownstreamPipelineError("reusable embed producer config is missing")
        producer_path = producer_path_value.resolve()
        if (
            not producer_path.is_file()
            or sha256_file(producer_path) != config.walk_forward.embed_producer_config_sha256
        ):
            raise DownstreamPipelineError("reusable embed producer config digest mismatch")
        producer = load_downstream_config(producer_path)
        if "strategy_contract" in manifest and manifest["strategy_contract"] != (
            producer.strategy_contract
        ):
            raise DownstreamPipelineError(
                "reusable embed manifest identity mismatch: strategy_contract"
            )
        producer_workflow_digests = {
            producer.workflow_digest,
            producer.legacy_workflow_digest,
        }
        if manifest.get("workflow_digest") not in producer_workflow_digests:
            raise DownstreamPipelineError(
                "reusable embed manifest does not match its producer config"
            )
        if producer.embedding_contract_digest != config.embedding_contract_digest:
            raise DownstreamPipelineError("reusable embedding semantics mismatch")
        if manifest.get("foundation_weights_sha256") != config.foundation.weights_sha256:
            raise DownstreamPipelineError("reusable embeddings use different foundation weights")
        embedding_dim = manifest.get("embedding_dim_per_channel")
        if not isinstance(embedding_dim, int) or embedding_dim < 1:
            raise DownstreamPipelineError("reusable embedding dimension is invalid")
        expected_width = (
            embedding_dim * len(config.data.feature_columns) * len(config.data.timeframes)
        )
        if manifest.get("feature_width") != expected_width:
            raise DownstreamPipelineError("reusable embedding feature width mismatch")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise DownstreamPipelineError("reusable embed manifest has no shard outputs")
        if manifest.get("rows") != sum(int(output.get("rows", -1)) for output in outputs):
            raise DownstreamPipelineError("reusable embed manifest row count mismatch")
    return manifest, path, sha256_file(path)


def prepare(config: DownstreamConfig) -> dict[str, Any]:
    """Validate source data and write every-bar candidate Parquet files."""
    stage_root = artifact_root(config) / "prepare"
    manifest_path = stage_root / "manifest.json"
    _require_new(stage_root, config)
    inputs = [
        _file_identity(market_path(config, symbol, timeframe))
        for symbol in config.data.symbols
        for timeframe in config.data.timeframes
    ]
    outputs: list[dict[str, Any]] = []
    contamination_by_symbol: list[dict[str, Any]] = []
    total_rows = 0
    for symbol in config.data.symbols:
        candidates = build_symbol_candidates(config, symbol)
        contamination_by_symbol.append(
            {"symbol": symbol, **dict(candidates.attrs.get("contamination", {}))}
        )
        candidates = candidates.loc[~candidates["is_holdout"]].reset_index(drop=True)
        if candidates.empty:
            raise DownstreamPipelineError(f"no pre-holdout candidates for {symbol}")
        output = stage_root / "candidates" / f"{symbol}.parquet"
        _atomic_parquet(candidates, output)
        identity = _file_identity(output)
        identity["symbol"] = symbol
        identity["rows"] = len(candidates)
        outputs.append(identity)
        total_rows += len(candidates)
    contamination: dict[str, Any] = {
        "schema_version": 1,
        "max_relative_close_jump": config.data.max_relative_close_jump,
        "symbols": contamination_by_symbol,
    }
    contamination["digest"] = report_digest(contamination)
    contamination_path = stage_root / "contamination.json"
    _atomic_json(contamination_path, contamination)
    manifest = {
        **_manifest_base(config, "prepare"),
        "inputs": inputs,
        "outputs": outputs,
        "rows": total_rows,
        "candidate_semantics": (
            "every_eligible_3min_trend_magic_state_bar"
            if getattr(config.strategy, "kind", "supertrend") == "trend_magic"
            else "every_eligible_3min_supertrend_state_bar"
        ),
        "holdout_locked": not config.evaluation.allow_holdout,
        "contamination": _file_identity(contamination_path),
        "contamination_digest": contamination["digest"],
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def embed(config: DownstreamConfig) -> dict[str, Any]:
    """Extract bounded MantisV2 embedding shards from prepared candidates."""
    root = artifact_root(config)
    started = time.perf_counter()
    prepared = _manifest(root / "prepare" / "manifest.json", config, "prepare")
    stage_root = root / "embed"
    manifest_path = stage_root / "manifest.json"
    _require_new(stage_root, config)
    foundation = load_foundation(config)
    synchronize_device(foundation.device)
    outputs: list[dict[str, Any]] = []
    shard_number = 0
    for candidate_identity in prepared["outputs"]:
        candidate_path = Path(candidate_identity["path"])
        if sha256_file(candidate_path) != candidate_identity["sha256"]:
            raise DownstreamPipelineError(f"prepared candidate changed: {candidate_path}")
        symbol = str(candidate_identity["symbol"])
        candidates = pd.read_parquet(candidate_path)
        feature_parts: list[np.ndarray] = []
        metadata_parts: list[pd.DataFrame] = []
        buffered = 0

        for start, stop, features in iter_symbol_embeddings(candidates, symbol, config, foundation):
            feature_parts.append(features)
            metadata_parts.append(candidates.iloc[start:stop].reset_index(drop=True))
            buffered += len(features)
            if buffered >= config.foundation.shard_rows:
                shard_number = _flush_embedding_shard(
                    stage_root,
                    shard_number,
                    feature_parts,
                    metadata_parts,
                    outputs,
                )
                buffered = 0
        shard_number = _flush_embedding_shard(
            stage_root, shard_number, feature_parts, metadata_parts, outputs
        )
    manifest = {
        **_manifest_base(config, "embed"),
        "prepare_manifest_sha256": sha256_file(root / "prepare" / "manifest.json"),
        "foundation_manifest": str(config.foundation.manifest_path.resolve()),
        "foundation_manifest_sha256": sha256_file(config.foundation.manifest_path),
        "foundation_weights": str(foundation.weights_path.resolve()),
        "foundation_weights_sha256": foundation.weights_sha256,
        "foundation_validation_evidence": str(foundation.validation_evidence_path.resolve()),
        "foundation_validation_evidence_sha256": foundation.validation_evidence_sha256,
        "embedding_dim_per_channel": foundation.embedding_dim,
        "feature_width": foundation.embedding_dim
        * len(config.data.feature_columns)
        * len(config.data.timeframes),
        "outputs": outputs,
        "rows": sum(int(output["rows"]) for output in outputs),
    }
    manifest["instrumentation"] = _record_stage_instrumentation(
        config,
        "embed",
        started,
        {"step": len(outputs), "rows": manifest["rows"], "shards": len(outputs)},
        foundation.device,
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def _flush_embedding_shard(
    stage_root: Path,
    shard_number: int,
    feature_parts: list[np.ndarray],
    metadata_parts: list[pd.DataFrame],
    outputs: list[dict[str, Any]],
) -> int:
    if not feature_parts:
        return shard_number
    features = np.concatenate(feature_parts)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    feature_path = stage_root / "shards" / f"features-{shard_number:05d}.npy"
    metadata_path = stage_root / "shards" / f"metadata-{shard_number:05d}.parquet"
    _atomic_npy(features, feature_path)
    _atomic_parquet(metadata, metadata_path)
    outputs.append(
        {
            "number": shard_number,
            "rows": len(metadata),
            "features": _file_identity(feature_path),
            "metadata": _file_identity(metadata_path),
        }
    )
    feature_parts.clear()
    metadata_parts.clear()
    return shard_number + 1


def _load_embedding_metadata(manifest: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for shard in manifest["outputs"]:
        path = Path(shard["metadata"]["path"])
        if sha256_file(path) != shard["metadata"]["sha256"]:
            raise DownstreamPipelineError(f"embedding metadata changed: {path}")
        frame = pd.read_parquet(path)
        frame["_shard"] = int(shard["number"])
        frame["_row"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _verify_embedding_features(manifest: dict[str, Any]) -> None:
    for shard in manifest["outputs"]:
        path = Path(shard["features"]["path"])
        if not path.is_file() or sha256_file(path) != shard["features"]["sha256"]:
            raise DownstreamPipelineError(f"embedding features changed: {path}")


def _feature_rows(
    manifest: dict[str, Any], metadata: pd.DataFrame, indices: np.ndarray
) -> np.ndarray:
    requested = metadata.iloc[indices]
    pieces: list[np.ndarray] = []
    order: list[np.ndarray] = []
    for shard_number, group in requested.groupby("_shard", sort=True):
        shard = manifest["outputs"][int(shard_number)]
        path = Path(shard["features"]["path"])
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        pieces.append(np.asarray(values[group["_row"].to_numpy(dtype=np.int64)], dtype=np.float32))
        order.append(group.index.to_numpy())
    if not pieces:
        raise DownstreamPipelineError("fold partition contains no embedding rows")
    features = np.concatenate(pieces)
    source_order = np.concatenate(order)
    return features[np.argsort(source_order)]


def _write_head(path: Path, head: PortableHead) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            scaler_mean=head.scaler_mean,
            scaler_scale=head.scaler_scale,
            coefficients=head.coefficients,
            intercept=head.intercept,
            threshold=np.asarray([head.threshold]),
        )
    os.replace(temporary, path)


def _require_finite_primary(metrics: dict[str, float | None], partition: str) -> None:
    for name in ("weighted_log_loss", "weighted_brier"):
        value = metrics.get(name)
        if value is None or not np.isfinite(value):
            raise DownstreamPipelineError(f"non-finite {partition} primary metric: {name}")


def _walk_forward_quality_gate(
    fold_outputs: list[dict[str, Any]],
) -> dict[str, float | bool]:
    mean_weighted_log_loss = float(
        np.mean([float(fold["test_metrics"]["weighted_log_loss"]) for fold in fold_outputs])
    )
    mean_weighted_brier = float(
        np.mean([float(fold["test_metrics"]["weighted_brier"]) for fold in fold_outputs])
    )
    baseline_log_loss = float(np.log(2.0))
    baseline_brier = 0.25
    baseline_passed = (
        mean_weighted_log_loss < baseline_log_loss and mean_weighted_brier < baseline_brier
    )
    return {
        "passed": baseline_passed,
        "mean_test_weighted_log_loss": mean_weighted_log_loss,
        "balanced_constant_weighted_log_loss": baseline_log_loss,
        "mean_test_weighted_brier": mean_weighted_brier,
        "balanced_constant_weighted_brier": baseline_brier,
    }


def _read_head(path: Path) -> PortableHead:
    values = np.load(path, allow_pickle=False)
    return PortableHead(
        scaler_mean=values["scaler_mean"],
        scaler_scale=values["scaler_scale"],
        coefficients=values["coefficients"],
        intercept=values["intercept"],
        threshold=float(values["threshold"][0]),
    )


def walk_forward(config: DownstreamConfig) -> dict[str, Any]:
    """Fit train-only logistic heads and emit full out-of-sample predictions."""
    root = artifact_root(config)
    device = torch.device("cpu")
    synchronize_device(device)
    started = time.perf_counter()
    embedded, embed_manifest_path, embed_manifest_sha = _embedding_manifest_input(config)
    stage_root = root / "walk-forward"
    manifest_path = stage_root / "manifest.json"
    _require_new(stage_root, config)
    head_config_digest = config.head_config_digest(embed_manifest_sha)
    _verify_embedding_features(embedded)
    metadata = _load_embedding_metadata(embedded)
    folds = build_folds(metadata, config)
    fold_outputs: list[dict[str, Any]] = []
    for fold in folds:
        train_mask, validation_mask, test_mask = fold_masks(metadata, fold, config)
        train_indices = deterministic_cap(
            np.flatnonzero(train_mask),
            config.walk_forward.max_fit_rows,
            config.run.seed + fold.number,
        )
        validation_indices = deterministic_cap(
            np.flatnonzero(validation_mask),
            config.walk_forward.max_fit_rows,
            config.run.seed + 10_000 + fold.number,
        )
        test_indices = np.flatnonzero(test_mask)
        train_features = _feature_rows(embedded, metadata, train_indices)
        validation_features = _feature_rows(embedded, metadata, validation_indices)
        try:
            head, validation_metrics, convergence = fit_logistic_head(
                train_features,
                metadata["label"].iloc[train_indices].to_numpy(dtype=np.int8),
                validation_features,
                metadata["label"].iloc[validation_indices].to_numpy(dtype=np.int8),
                config,
            )
        except HeadConvergenceError as exc:
            _atomic_json(
                stage_root / "failure.json",
                {
                    "stage": "walk-forward",
                    "fold": asdict(fold),
                    "head_config_digest": head_config_digest,
                    "embed_manifest": str(embed_manifest_path),
                    "embed_manifest_sha256": embed_manifest_sha,
                    "embedding_contract_digest": config.embedding_contract_digest,
                    "convergence": asdict(exc.diagnostics),
                    "error": str(exc),
                },
            )
            raise
        _require_finite_primary(validation_metrics, f"fold {fold.number} validation")
        del train_features, validation_features
        head_path = stage_root / "heads" / f"fold-{fold.number:03d}.npz"
        _write_head(head_path, head)
        prediction_path = stage_root / "predictions" / f"fold-{fold.number:03d}.parquet"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = prediction_path.with_suffix(".parquet.tmp")
        writer: pq.ParquetWriter | None = None
        test_labels: list[np.ndarray] = []
        test_probability: list[np.ndarray] = []
        try:
            for start in range(0, len(test_indices), config.foundation.shard_rows):
                selected = test_indices[start : start + config.foundation.shard_rows]
                features = _feature_rows(embedded, metadata, selected)
                probability = predict_head(head, features)
                output = metadata.iloc[selected].drop(columns=["_shard", "_row"]).copy()
                output["fold"] = fold.number
                output["probability"] = probability
                output["threshold"] = head.threshold
                table = pa.Table.from_pandas(output, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema)
                writer.write_table(table)
                test_labels.append(output["label"].to_numpy(dtype=np.int8))
                test_probability.append(probability)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise DownstreamPipelineError(f"fold {fold.number} has no test rows")
        os.replace(temporary, prediction_path)
        test_metrics = probability_metrics(
            np.concatenate(test_labels), np.concatenate(test_probability)
        )
        _require_finite_primary(test_metrics, f"fold {fold.number} test")
        fold_outputs.append(
            {
                "fold": asdict(fold),
                "train_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "test_rows": len(test_indices),
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "convergence": asdict(convergence),
                "head": _file_identity(head_path),
                "predictions": _file_identity(prediction_path),
            }
        )
    quality_gate = _walk_forward_quality_gate(fold_outputs)
    convergence_gate_passed = all(bool(fold["convergence"]["converged"]) for fold in fold_outputs)
    manifest = {
        **_manifest_base(config, "walk-forward"),
        "embed_manifest": str(embed_manifest_path),
        "embed_manifest_sha256": embed_manifest_sha,
        "embed_producer_config": str(config.walk_forward.embed_producer_config_path),
        "embed_producer_config_sha256": config.walk_forward.embed_producer_config_sha256,
        "embedding_contract_digest": config.embedding_contract_digest,
        "head_config_digest": head_config_digest,
        "head_config": asdict(config.walk_forward),
        "convergence_gate_passed": convergence_gate_passed,
        "quality_gate": quality_gate,
        "primary_metrics": ["weighted_log_loss", "weighted_brier"],
        "diagnostic_metrics": ["log_loss", "brier", "roc_auc", "pr_auc"],
        "folds": fold_outputs,
    }
    manifest["instrumentation"] = _record_stage_instrumentation(
        config,
        "walk_forward",
        started,
        {
            "step": len(fold_outputs),
            "folds": len(fold_outputs),
            "validation_weighted_log_loss": float(
                np.mean([fold["validation_metrics"]["weighted_log_loss"] for fold in fold_outputs])
            ),
        },
        device,
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def simulate(config: DownstreamConfig) -> dict[str, Any]:
    """Replay walk-forward predictions under the configured Combine rules."""
    root = artifact_root(config)
    walked = _manifest(root / "walk-forward" / "manifest.json", config, "walk-forward")
    if walked.get("convergence_gate_passed") is not True:
        raise DownstreamPipelineError("walk-forward convergence gate did not pass")
    quality_gate = walked.get("quality_gate")
    if not isinstance(quality_gate, dict) or quality_gate.get("passed") is not True:
        raise DownstreamPipelineError("walk-forward quality gate did not pass")
    stage_root = root / "simulate"
    manifest_path = stage_root / "manifest.json"
    _require_new(stage_root, config)
    predictions: list[pd.DataFrame] = []
    for fold in walked["folds"]:
        path = Path(fold["predictions"]["path"])
        if sha256_file(path) != fold["predictions"]["sha256"]:
            raise DownstreamPipelineError(f"walk-forward predictions changed: {path}")
        predictions.append(pd.read_parquet(path))
    result, trades = simulate_topstep(pd.concat(predictions, ignore_index=True), config)
    trades_path = stage_root / "trades.parquet"
    _atomic_parquet(trades, trades_path)
    manifest = {
        **_manifest_base(config, "simulate"),
        "walk_forward_manifest_sha256": sha256_file(root / "walk-forward" / "manifest.json"),
        "account": asdict(config.topstep),
        "result": asdict(result),
        "trades": _file_identity(trades_path),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def run(config: DownstreamConfig) -> dict[str, Any]:
    """Run all normal pre-holdout stages in dependency order."""
    return {
        "prepare": prepare(config),
        "embed": embed(config),
        "walk_forward": walk_forward(config),
        "simulate": simulate(config),
    }


def smoke(config: DownstreamConfig) -> dict[str, Any]:
    """Exercise downstream serialization, head, metrics, and simulator without CUDA."""
    if config.run.device != "cpu" or not config.run.allow_overwrite:
        raise DownstreamPipelineError(
            "downstream smoke requires device=cpu and allow_overwrite=true"
        )
    root = artifact_root(config)
    embed_started = time.perf_counter()
    generator = np.random.default_rng(config.run.seed)
    features = generator.normal(size=(12, 4)).astype(np.float32)
    labels = np.asarray([0, 1] * 6, dtype=np.int8)
    timestamps = pd.date_range("2025-01-06T15:00:00Z", periods=12, freq="24h")
    metadata = pd.DataFrame(
        {
            "symbol": [config.data.symbols[0]] * 12,
            "decision_index": np.arange(12, dtype=np.int64) * 100,
            "decision_ts": timestamps,
            "entry_ts": timestamps,
            "label_end_ts": timestamps + timedelta(seconds=180),
            "label": labels,
            "reward_r": np.where(labels == 1, 1.0, -1.0),
            "mae_r": np.where(labels == 1, -0.1, -1.0),
            "atr": np.full(12, 1.0),
        }
    )
    prepare_path = root / "prepare" / "candidates.parquet"
    _atomic_parquet(metadata, prepare_path)
    prepare_manifest = {
        **_manifest_base(config, "prepare"),
        "outputs": [_file_identity(prepare_path)],
        "rows": len(metadata),
    }
    _atomic_json(root / "prepare" / "manifest.json", prepare_manifest)
    feature_path = root / "embed" / "features.npy"
    metadata_path = root / "embed" / "metadata.parquet"
    _atomic_npy(features, feature_path)
    _atomic_parquet(metadata, metadata_path)
    embed_manifest = {
        **_manifest_base(config, "embed"),
        "prepare_manifest_sha256": sha256_file(root / "prepare" / "manifest.json"),
        "features": _file_identity(feature_path),
        "metadata": _file_identity(metadata_path),
    }
    embed_manifest["instrumentation"] = _record_stage_instrumentation(
        config,
        "embed",
        embed_started,
        {"step": 1, "rows": len(metadata), "shards": 1},
        torch.device("cpu"),
    )
    _atomic_json(root / "embed" / "manifest.json", embed_manifest)
    walk_forward_started = time.perf_counter()
    head, validation_metrics, convergence = fit_logistic_head(
        features[:8], labels[:8], features[8:], labels[8:], config
    )
    _require_finite_primary(validation_metrics, "smoke validation")
    probability = predict_head(head, features[8:])
    test_metrics = probability_metrics(labels[8:], probability)
    _require_finite_primary(test_metrics, "smoke test")
    predictions = metadata.iloc[8:].copy()
    predictions["probability"] = probability
    predictions["threshold"] = head.threshold
    head_path = root / "walk-forward" / "head.npz"
    predictions_path = root / "walk-forward" / "predictions.parquet"
    _write_head(head_path, head)
    _atomic_parquet(predictions, predictions_path)
    walk_manifest = {
        **_manifest_base(config, "walk-forward"),
        "embed_manifest_sha256": sha256_file(root / "embed" / "manifest.json"),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "convergence_gate_passed": convergence.converged,
        "quality_gate": {"passed": True},
        "convergence": asdict(convergence),
        "head": _file_identity(head_path),
        "predictions": _file_identity(predictions_path),
    }
    walk_manifest["instrumentation"] = _record_stage_instrumentation(
        config,
        "walk_forward",
        walk_forward_started,
        {
            "step": 1,
            "folds": 1,
            "validation_weighted_log_loss": float(
                cast(float, validation_metrics["weighted_log_loss"])
            ),
        },
        torch.device("cpu"),
    )
    _atomic_json(root / "walk-forward" / "manifest.json", walk_manifest)
    result, trades = simulate_topstep(predictions, config)
    trades_path = root / "simulate" / "trades.parquet"
    _atomic_parquet(trades, trades_path)
    simulate_manifest = {
        **_manifest_base(config, "simulate"),
        "walk_forward_manifest_sha256": sha256_file(root / "walk-forward" / "manifest.json"),
        "result": asdict(result),
        "trades": _file_identity(trades_path),
    }
    _atomic_json(root / "simulate" / "manifest.json", simulate_manifest)
    for stage in ("prepare", "embed", "walk-forward", "simulate"):
        _manifest(root / stage / "manifest.json", config, stage)
    return {
        "verified": True,
        "device": "cpu",
        "rows": len(metadata),
        "manifest_digest": manifest_digest(config),
    }


def evaluate_holdout(config: DownstreamConfig, unlock: str) -> dict[str, Any]:
    """Apply the last validation-owned head once to the sealed holdout."""
    if not config.evaluation.allow_holdout or unlock != config.evaluation.holdout_unlock:
        raise DownstreamPipelineError(
            "holdout is locked; set evaluation.allow_holdout=true and pass the configured unlock"
        )
    root = artifact_root(config)
    stage_root = root / "holdout"
    manifest_path = stage_root / "manifest.json"
    _require_new(stage_root, config)
    _embedded, embed_manifest_path, embed_manifest_sha = _embedding_manifest_input(config)
    walked = _manifest(root / "walk-forward" / "manifest.json", config, "walk-forward")
    if walked.get("embed_manifest_sha256") != embed_manifest_sha:
        raise DownstreamPipelineError(
            "walk-forward manifest is not bound to the current embeddings"
        )
    if walked.get("convergence_gate_passed") is not True:
        raise DownstreamPipelineError("walk-forward convergence gate did not pass")
    quality_gate = walked.get("quality_gate")
    if not isinstance(quality_gate, dict) or quality_gate.get("passed") is not True:
        raise DownstreamPipelineError("walk-forward quality gate did not pass")
    last = walked["folds"][-1]
    head_path = Path(last["head"]["path"])
    if not head_path.is_file() or sha256_file(head_path) != last["head"]["sha256"]:
        raise DownstreamPipelineError("holdout head digest mismatch")
    head = _read_head(head_path)
    foundation = load_foundation(config)
    output_parts: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for symbol in config.data.symbols:
        candidates = build_symbol_candidates(config, symbol, split="holdout")
        candidates = candidates.loc[candidates["is_holdout"]].reset_index(drop=True)
        if candidates.empty:
            continue
        for start, stop, features in iter_symbol_embeddings(candidates, symbol, config, foundation):
            probability = predict_head(head, features.astype(np.float32))
            output = candidates.iloc[start:stop].copy()
            output["probability"] = probability
            output["threshold"] = head.threshold
            output_parts.append(output)
            labels.append(output["label"].to_numpy(dtype=np.int8))
            probabilities.append(probability)
    if not output_parts:
        raise DownstreamPipelineError("sealed holdout contains no legal candidates")
    predictions_path = stage_root / "predictions.parquet"
    _atomic_parquet(pd.concat(output_parts, ignore_index=True), predictions_path)
    metrics = probability_metrics(np.concatenate(labels), np.concatenate(probabilities))
    _require_finite_primary(metrics, "holdout")
    manifest = {
        **_manifest_base(config, "holdout"),
        "requested_config_digest": config.digest,
        "walk_forward_manifest_sha256": sha256_file(root / "walk-forward" / "manifest.json"),
        "head_sha256": last["head"]["sha256"],
        "foundation_weights_sha256": foundation.weights_sha256,
        "threshold": head.threshold,
        "metrics": metrics,
        "predictions": _file_identity(predictions_path),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def manifest_digest(config: DownstreamConfig) -> str:
    """Return a compact identity for all completed normal-stage manifests."""
    digest = hashlib.sha256()
    for stage in ("prepare", "embed", "walk-forward", "simulate"):
        digest.update((artifact_root(config) / stage / "manifest.json").read_bytes())
    return digest.hexdigest()
