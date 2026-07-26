"""Official frozen MantisV2 expected-R embedding and raw-control comparison."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from mantis.architecture import MantisV2

from mantis_v2.expected_r_screen import ExpectedRScreen, ExpectedRScreenConfig
from mantis_v2.model import sha256_file
from mantis_v2.rl_provenance import source_digest


class FrozenExpectedRError(RuntimeError):
    """Raised when the official frozen expected-R contract is violated."""


class _ComparisonProgress:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.monotonic()
        self.stage_started: dict[tuple[str, str | None, str | None], float] = {}
        self.sequence = 0
        self._lock = threading.Lock()

    def write(
        self,
        stage: str,
        *,
        fold: str | None = None,
        arm: str | None = None,
        thresholds_done: int = 0,
        thresholds_total: int = 0,
        bootstrap_done: int = 0,
        bootstrap_total: int = 0,
        terminal_state: Literal["running", "complete", "failed"] = "running",
    ) -> None:
        with self._lock:
            now = time.monotonic()
            key = (stage, fold, arm)
            stage_started = self.stage_started.setdefault(key, now)
            stage_elapsed = max(now - stage_started, 0.0)
            done, total = (
                (thresholds_done, thresholds_total)
                if thresholds_total
                else (bootstrap_done, bootstrap_total)
            )
            throughput = done / stage_elapsed if done and stage_elapsed > 0 else None
            eta = (total - done) / throughput if throughput and total >= done else None
            self.sequence += 1
            _atomic_json(
                self.path,
                {
                    "schema_version": 1,
                    "sequence": self.sequence,
                    "stage": stage,
                    "fold": fold,
                    "arm": arm,
                    "thresholds_done": thresholds_done,
                    "thresholds_total": thresholds_total,
                    "bootstrap_done": bootstrap_done,
                    "bootstrap_total": bootstrap_total,
                    "elapsed_seconds": now - self.started,
                    "throughput_per_second": throughput,
                    "eta_seconds": eta,
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "terminal_state": terminal_state,
                },
            )


_SOURCE_REVISION = "0c94f8ceb9f1d1421dd292ed917090df8c31605b"
_HUB_REVISION = "99fe0f548960e272fbfa4b82fd9b5b5956779dfd"
_WEIGHTS_SHA256 = "49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1"
_OFFICIAL_IMAGE = (
    "runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35"
)
_INITIAL_SCREEN = (
    "initial_screen",
    "2023-07-01",
    "2025-07-01",
    "2025-07-01",
    "2025-10-01",
    "2025-10-01",
    "2026-01-01",
)
_ANCHORED_FOLDS = (
    (
        "fold_1",
        "2023-07-01",
        "2024-10-01",
        "2024-10-01",
        "2025-01-01",
        "2025-01-01",
        "2025-04-01",
    ),
    (
        "fold_2",
        "2023-07-01",
        "2025-01-01",
        "2025-01-01",
        "2025-04-01",
        "2025-04-01",
        "2025-07-01",
    ),
    (
        "fold_3",
        "2023-07-01",
        "2025-04-01",
        "2025-04-01",
        "2025-07-01",
        "2025-07-01",
        "2025-10-01",
    ),
)


@dataclass(frozen=True)
class FrozenExpectedRConfig:
    """Strict official MantisV2 feature and comparison contract."""

    source_repository: str = "vfeofanov/mantis"
    source_revision: str = _SOURCE_REVISION
    hub_model: str = "paris-noah/MantisV2"
    hub_revision: str = _HUB_REVISION
    weights_sha256: str = _WEIGHTS_SHA256
    checkpoint_kind: Literal["official_frozen"] = "official_frozen"
    preprocessing: Literal["native_mantis_v2"] = "native_mantis_v2"
    return_transf_layer: Literal[2] = 2
    output_token: Literal["combined"] = "combined"
    input_length: Literal[512] = 512
    market_channels: Literal[5] = 5
    embedding_width: Literal[2560] = 2560
    context_width: Literal[3] = 3
    ridge_alpha: float = 10.0
    shard_rows: int = 4096
    batch_size: int = 256
    requested_precision: Literal["bf16", "fp32"] = "bf16"
    parity_max_abs: float = 0.01
    parity_minimum_cosine: float = 0.999
    bootstrap_replicates: int = 1000
    bootstrap_restart_probability: float = 0.2
    seed: int = 86

    def __post_init__(self) -> None:
        official = (
            self.source_repository == "vfeofanov/mantis"
            and self.source_revision == _SOURCE_REVISION
            and self.hub_model == "paris-noah/MantisV2"
            and self.hub_revision == _HUB_REVISION
            and self.weights_sha256 == _WEIGHTS_SHA256
            and self.checkpoint_kind == "official_frozen"
            and self.preprocessing == "native_mantis_v2"
            and self.return_transf_layer == 2
            and self.output_token == "combined"
            and self.input_length == 512
            and self.market_channels == 5
            and self.embedding_width == 2560
            and self.context_width == 3
        )
        if not official:
            raise FrozenExpectedRError("official frozen MantisV2 contract is required")
        if self.shard_rows < 1 or self.batch_size < 1:
            raise FrozenExpectedRError("shard_rows and batch_size must be positive")
        if self.ridge_alpha <= 0 or self.bootstrap_replicates < 1:
            raise FrozenExpectedRError("ridge and bootstrap values must be positive")
        if not 0 < self.bootstrap_restart_probability <= 1:
            raise FrozenExpectedRError("bootstrap restart probability must be in (0, 1]")

    @classmethod
    def from_json(cls, path: Path) -> FrozenExpectedRConfig:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenExpectedRError(f"invalid frozen screen config: {path}") from exc
        if not isinstance(value, dict):
            raise FrozenExpectedRError("frozen screen config must be an object")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise FrozenExpectedRError(f"unknown config keys: {', '.join(unknown)}")
        try:
            return cls(**value)
        except TypeError as exc:
            raise FrozenExpectedRError("invalid frozen screen config values") from exc

    @property
    def digest(self) -> str:
        return _json_digest(asdict(self))


class _EmbeddingModel(Protocol):
    def __call__(self, values: np.ndarray, precision: str) -> np.ndarray: ...


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _file_record(path: Path, *, portable: bool = False) -> dict[str, Any]:
    recorded_path = path.name if portable else str(path.resolve())
    return {"path": recorded_path, "size": path.stat().st_size, "sha256": sha256_file(path)}


def _record_path(record: dict[str, Any], parent: Path) -> Path:
    path = Path(str(record.get("path", "")))
    return path if path.is_absolute() else parent / path


def write_frozen_input(
    candidates: pd.DataFrame, windows: np.ndarray, context: np.ndarray, output: Path
) -> Path:
    """Atomically publish the compact, immutable candidate/window input."""
    required = {
        "row_id",
        "decision_ts",
        "feature_start_ts",
        "feature_start_index",
        "outcome_ts",
        "entry_index",
        "outcome_index",
        "average_uniqueness",
        "net_r",
    }
    if missing := sorted(required.difference(candidates.columns)):
        raise FrozenExpectedRError(f"candidate input missing columns: {', '.join(missing)}")
    values = np.asarray(windows, dtype=np.float32)
    context_values = np.asarray(context, dtype=np.float32)
    if values.shape != (len(candidates), 5, 512) or not np.isfinite(values).all():
        raise FrozenExpectedRError("windows must be finite [rows, 5, 512] FP32 values")
    if context_values.shape != (len(candidates), 3) or not np.isfinite(context_values).all():
        raise FrozenExpectedRError("context must be finite [rows, 3] FP32 values")
    if candidates["row_id"].duplicated().any():
        raise FrozenExpectedRError("candidate row identities must be unique")
    feature_starts = pd.to_datetime(candidates["feature_start_ts"], utc=True)
    decisions = pd.to_datetime(candidates["decision_ts"], utc=True)
    outcomes = pd.to_datetime(candidates["outcome_ts"], utc=True)
    if (
        (feature_starts > decisions).any()
        or (outcomes < decisions).any()
        or not np.array_equal(
            candidates["feature_start_index"].to_numpy(dtype=np.int64),
            candidates["entry_index"].to_numpy(dtype=np.int64) - 512,
        )
    ):
        raise FrozenExpectedRError("candidate feature lookback or outcome span is invalid")
    if output.exists():
        raise FrozenExpectedRError(f"frozen input already exists: {output}")
    output.mkdir(parents=True)
    candidates_path = output / "candidates.parquet"
    windows_path = output / "windows.npy"
    context_path = output / "context.npy"
    candidate_table = candidates.reset_index(drop=True).copy()
    candidate_table.attrs = {}
    candidate_table.to_parquet(candidates_path, index=False)
    with windows_path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    with context_path.open("wb") as stream:
        np.save(stream, context_values, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "rows": len(candidates),
        "row_ids_sha256": _json_digest(candidates["row_id"].tolist()),
        "candidates": _file_record(candidates_path, portable=True),
        "windows": _file_record(windows_path, portable=True),
        "context": _file_record(context_path, portable=True),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "preprocessing": "native_mantis_v2",
        "producer_source_sha256": source_digest(Path(__file__).resolve().parents[3]),
        "expected_r_config_sha256": _json_digest(asdict(ExpectedRScreenConfig())),
    }
    manifest["manifest_sha256"] = _json_digest(manifest)
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


def prepare_frozen_input(market_path: Path, output: Path) -> Path:
    """Create the paid-stage input from accepted 3-minute Parquet on the local host."""
    try:
        market = pd.read_parquet(
            market_path, columns=["datetime", "open", "high", "low", "close", "volume"]
        ).rename(columns={"datetime": "timestamp"})
    except (OSError, ValueError) as exc:
        raise FrozenExpectedRError(f"accepted market Parquet is unreadable: {market_path}") from exc
    market = market.loc[pd.to_datetime(market["timestamp"], utc=True) < "2026-01-01"]
    candidates = ExpectedRScreen().generate_candidates(market)
    raw = candidates.attrs["raw_features"]
    windows = raw[:, :-3].reshape(len(candidates), 512, 5).transpose(0, 2, 1)
    return write_frozen_input(candidates, windows, raw[:, -3:], output)


def load_frozen_embeddings(
    input_manifest_path: Path, embedding_manifest_path: Path, config: FrozenExpectedRConfig
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load only complete, digest-verified shards in exact candidate order."""
    embedder = FrozenMantisEmbedder(config)
    input_manifest, candidates, windows, context = embedder.validate_input(input_manifest_path)
    embedding_manifest = embedder._validate_final(
        embedding_manifest_path, input_manifest, len(candidates), embedding_manifest_path.parent
    )
    features: list[np.ndarray] = []
    row_ids: list[str] = []
    for number, shard in enumerate(embedding_manifest.get("shards", [])):
        if shard.get("number") != number:
            raise FrozenExpectedRError("embedding manifest shard sequence changed")
        for label in ("features", "metadata"):
            record = shard.get(label, {})
            path = _record_path(record, embedding_manifest_path.parent)
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise FrozenExpectedRError(f"embedding {label} changed")
        features.append(
            np.load(
                _record_path(shard["features"], embedding_manifest_path.parent),
                allow_pickle=False,
            )
        )
        row_ids.extend(
            pd.read_parquet(_record_path(shard["metadata"], embedding_manifest_path.parent))[
                "row_id"
            ].tolist()
        )
    if row_ids != candidates["row_id"].tolist():
        raise FrozenExpectedRError("raw and Mantis candidate rows differ")
    raw = np.concatenate(
        (np.asarray(windows).transpose(0, 2, 1).reshape(len(windows), -1), context),
        axis=1,
    )
    return candidates, raw.astype(np.float32), np.concatenate(features).astype(np.float32)


def compare_frozen_artifacts(
    input_manifest_path: Path,
    embedding_manifest_path: Path,
    output: Path,
    config: FrozenExpectedRConfig,
    *,
    comparison_device: Literal["cpu", "cuda"] = "cuda",
    cpu_exception: str | None = None,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    if comparison_device == "cpu" and not (cpu_exception and cpu_exception.strip()):
        raise FrozenExpectedRError(
            "CPU comparison requires a recorded user-approved or qualification-failure exception"
        )
    if comparison_device == "cuda" and cpu_exception:
        raise FrozenExpectedRError("CPU exception is only valid for a CPU comparison")
    effective_progress_path = progress_path or output.with_name(f"{output.stem}.progress.json")
    completed = _validated_completed_comparison(
        input_manifest_path,
        embedding_manifest_path,
        output,
        config,
        comparison_device,
        effective_progress_path,
    )
    if completed is not None:
        return completed
    if output.with_suffix(output.suffix + ".tmp").exists():
        raise FrozenExpectedRError(f"partial comparison output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = _ComparisonProgress(effective_progress_path)
    progress.path.parent.mkdir(parents=True, exist_ok=True)
    progress.write("loading")
    try:
        candidates, raw, mantis = load_frozen_embeddings(
            input_manifest_path, embedding_manifest_path, config
        )
        progress.write("loaded")
        result = compare_frozen_to_raw(
            candidates,
            raw,
            mantis,
            config,
            comparison_device=comparison_device,
            progress=progress,
        )
        result["provenance"] = _comparison_provenance(
            input_manifest_path, embedding_manifest_path, config
        )
        result["comparison_backend"]["cpu_exception"] = cpu_exception or None
        result.pop("artifact_sha256", None)
        result["artifact_sha256"] = _json_digest(result)
    except Exception:
        progress.write("failed", terminal_state="failed")
        raise
    progress.write("complete", terminal_state="complete")
    _atomic_json(output, result)
    return result


def _comparison_provenance(
    input_manifest_path: Path,
    embedding_manifest_path: Path,
    config: FrozenExpectedRConfig,
) -> dict[str, Any]:
    try:
        input_manifest = json.loads(input_manifest_path.read_text())
        embedding_manifest = json.loads(embedding_manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenExpectedRError("comparison provenance is unreadable") from exc
    if not isinstance(input_manifest, dict) or not isinstance(embedding_manifest, dict):
        raise FrozenExpectedRError("comparison provenance must be an object")
    return {
        "input_manifest_sha256": input_manifest.get("manifest_sha256"),
        "embedding_artifact_sha256": embedding_manifest.get("artifact_sha256"),
        "config_sha256": config.digest,
        "weights_sha256": config.weights_sha256,
        "precision": embedding_manifest.get("precision"),
        "parity_sha256": _json_digest(embedding_manifest.get("bf16_parity")),
    }


def _validated_completed_comparison(
    input_manifest_path: Path,
    embedding_manifest_path: Path,
    output: Path,
    config: FrozenExpectedRConfig,
    comparison_device: Literal["cpu", "cuda"],
    progress_path: Path | None = None,
) -> dict[str, Any] | None:
    if not output.exists():
        return None
    try:
        result = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenExpectedRError("completed comparison is unreadable") from exc
    if not isinstance(result, dict):
        raise FrozenExpectedRError("completed comparison must be an object")
    unsigned = dict(result)
    digest = unsigned.pop("artifact_sha256", None)
    if digest != _json_digest(unsigned):
        raise FrozenExpectedRError("completed comparison digest mismatch")
    expected = _comparison_provenance(input_manifest_path, embedding_manifest_path, config)
    backend = result.get("comparison_backend", {})
    if (
        result.get("provenance") != expected
        or result.get("config_sha256") != config.digest
        or not isinstance(backend, dict)
        or backend.get("device") != comparison_device
        or result.get("status") not in {"passed", "stopped"}
        or result.get("selected") not in {"raw", "mantis", "stop"}
    ):
        raise FrozenExpectedRError("completed comparison provenance mismatch")
    if progress_path is not None:
        try:
            progress = json.loads(progress_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenExpectedRError("completed comparison progress is unreadable") from exc
        if not isinstance(progress, dict) or progress.get("terminal_state") != "complete":
            raise FrozenExpectedRError("completed comparison progress is incomplete")
    return result


def run_paid_frozen_screen(
    config_path: Path,
    input_manifest_path: Path,
    embedding_output: Path,
    comparison_output: Path,
    progress_output: Path,
) -> dict[str, Any]:
    """Resume frozen embedding, then run or validate the immutable CUDA comparison."""
    config = FrozenExpectedRConfig.from_json(config_path)
    if comparison_output.with_suffix(comparison_output.suffix + ".tmp").exists():
        raise FrozenExpectedRError(f"partial comparison output exists: {comparison_output}")
    embedding = FrozenMantisEmbedder(config).embed(input_manifest_path, embedding_output)
    if embedding.get("status") != "complete":
        raise FrozenExpectedRError("paid embedding did not complete")
    selection = compare_frozen_artifacts(
        input_manifest_path,
        embedding_output / "manifest.json",
        comparison_output,
        config,
        comparison_device="cuda",
        progress_path=progress_output,
    )
    return {
        "embedding_status": "complete",
        "selection_status": selection["status"],
    }


_FOCUSED_CHECKS = {
    "causality_next_fill": (
        "mantis-v2/tests/test_expected_r_screen.py::test_candidate_timestamps_are_causal_and_long_trail_matches_oracle",
    ),
    "label_replay_parity": (
        "mantis-v2/tests/test_expected_r_screen.py::test_short_stop_and_target_use_next_open_and_costs",
        "mantis-v2/tests/test_expected_r_screen.py::test_active_trail_precedes_same_bar_target_contact",
    ),
    "topstep_accounting": (
        "mantis-v2/tests/test_expected_r_screen.py::test_default_costs_match_one_mnq_contract",
        "mantis-v2/tests/test_expected_r_screen.py::test_session_exit_uses_last_completed_bar_before_cutoff",
    ),
    "artifact_resume": (
        "mantis-v2/tests/test_frozen_expected_r.py::test_atomic_embedding_resume_skips_complete_shards",
        "mantis-v2/tests/test_frozen_expected_r.py::test_completed_embedding_rejects_changed_config_and_modified_shard",
    ),
}


def write_focused_check_receipt(
    output: Path,
    *,
    runner: Callable[[list[str]], int] | None = None,
) -> dict[str, Any]:
    """Run the four bounded public checks and publish their exact receipt."""
    if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
        raise FrozenExpectedRError("focused check receipt already exists")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *(node for nodes in _FOCUSED_CHECKS.values() for node in nodes),
    ]
    started = time.monotonic()
    return_code = (
        runner(command) if runner is not None else subprocess.run(command, check=False).returncode
    )
    duration = time.monotonic() - started
    if return_code != 0:
        raise FrozenExpectedRError("focused preflight checks failed")
    if duration >= 60:
        raise FrozenExpectedRError("focused preflight checks exceeded 60 seconds")
    payload = {
        "schema_version": 1,
        "checks": {name: True for name in _FOCUSED_CHECKS},
        "node_ids": {name: list(nodes) for name, nodes in _FOCUSED_CHECKS.items()},
        "duration_seconds": duration,
        "producer_source_sha256": source_digest(Path(__file__).resolve().parents[3]),
    }
    payload["receipt_sha256"] = _json_digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, payload)
    return payload


def _paid_control_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenExpectedRError("paid control config is unreadable") from exc
    required = {
        "schema_version",
        "run_id",
        "source_revision",
        "frozen_config",
        "input_manifest",
        "input_bundle_manifest",
        "dependency_lock",
        "source_archive",
        "official_bootstrap_receipt",
        "spend_ledger",
        "authorization",
        "heartbeat_token",
        "pod_paths",
        "artifacts",
        "provider",
        "runpodctl",
    }
    if not isinstance(value, dict) or set(value) != required or value.get("schema_version") != 1:
        raise FrozenExpectedRError("paid control config keys mismatch")
    nested = {
        "pod_paths": {
            "authorization",
            "dependency_lock",
            "frozen_config",
            "heartbeat_token",
            "input_bundle_manifest",
            "input_manifest",
            "official_bootstrap_receipt",
            "preflight",
            "source_archive",
            "spend_ledger",
            "workload_experiment",
        },
        "artifacts": {"controller", "backup"},
        "provider": {
            "budget_usd",
            "datacenter_id",
            "deadline_hours",
            "hourly_rate_usd",
            "ram_gb",
            "storage_usd_per_gb_hour",
            "vcpu",
            "volume_id",
            "volume_size_gb",
        },
        "runpodctl": {"version", "source_commit", "binary_sha256"},
    }
    for section, keys in nested.items():
        if not isinstance(value[section], dict) or set(value[section]) != keys:
            raise FrozenExpectedRError(f"paid control config {section} keys mismatch")
    run_id = value["run_id"]
    source_revision = value["source_revision"]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise FrozenExpectedRError("paid control run_id is invalid")
    if not isinstance(source_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise FrozenExpectedRError("paid control source_revision is invalid")
    for pod_path in value["pod_paths"].values():
        parsed = Path(str(pod_path))
        if not parsed.is_absolute() or ".." in parsed.parts:
            raise FrozenExpectedRError("paid control Pod path is invalid")
    return value


def _dual_file_record(controller: Path, pod: str) -> dict[str, Any]:
    if not controller.is_file() or not Path(pod).is_absolute():
        raise FrozenExpectedRError("paid control file path is invalid")
    return {
        "controller_path": str(controller.resolve()),
        "pod_path": pod,
        "sha256": sha256_file(controller),
        "size": controller.stat().st_size,
    }


def write_paid_planning_inputs(config_path: Path, output: Path) -> dict[str, Any]:
    """Create checks, preflight, and zero-cost RunPod planning inputs from one config."""
    control = _paid_control_config(config_path)
    output.mkdir(parents=True, exist_ok=False)
    checks_path = output / "focused-checks.json"
    checks = write_focused_check_receipt(checks_path)
    paths = control["pod_paths"]
    artifacts = control["artifacts"]
    provider = control["provider"]
    if (
        not isinstance(paths, dict)
        or not isinstance(artifacts, dict)
        or not isinstance(provider, dict)
    ):
        raise FrozenExpectedRError("paid control config sections must be objects")
    run_id = str(control["run_id"])
    pod_root = f"/workspace/mantis/runs/{run_id}"
    exact_command = shlex.join(
        [
            "uv",
            "run",
            "mantis-v2",
            "frozen-screen-paid-workload",
            "--config",
            str(paths["frozen_config"]),
            "--input",
            str(paths["input_manifest"]),
            "--embedding-output",
            f"{pod_root}/embed",
            "--comparison-output",
            f"{pod_root}/selection.json",
            "--progress-output",
            f"{pod_root}/selection.progress.json",
        ]
    )
    duration = min(
        int(float(provider["deadline_hours"]) * 3600),
        int(float(provider["budget_usd"]) / float(provider["hourly_rate_usd"]) * 3600),
        6 * 3600,
    )
    preflight_path = output / "preflight.json"
    write_paid_preflight(
        Path(str(control["input_manifest"])),
        Path(pod_root),
        preflight_path,
        exact_command=exact_command,
        hourly_rate_usd=float(provider["hourly_rate_usd"]),
        budget_usd=float(provider["budget_usd"]),
        deadline_hours=duration / 3600,
        check_duration_seconds=float(checks["duration_seconds"]),
        checks={name: True for name in _FOCUSED_CHECKS},
    )
    frozen = FrozenExpectedRConfig.from_json(Path(str(control["frozen_config"])))
    experiment_toml = output / "experiment.toml"
    experiment_toml.write_text(
        "\n".join(
            (
                "schema_version = 1",
                "",
                "[experiment]",
                f'name = "{run_id}"',
                'model_family = "mantis-v2"',
                'stage = "qualification"',
                f"seed = {frozen.seed}",
                f'definition_sha256 = "{sha256_file(preflight_path)}"',
                "sealed_holdout = false",
                "",
            )
        )
    )
    intent = {
        "schema_version": 1,
        "intent_id": run_id,
        "stage": "qualification",
        "run_name": run_id,
        "gpu_type": "NVIDIA L40S",
        "datacenter_id": provider["datacenter_id"],
        "gpu_count": 1,
        "vcpu": provider["vcpu"],
        "ram_gb": provider["ram_gb"],
        "container_disk_gb": 50,
        "image_ref": _OFFICIAL_IMAGE,
        "template_id": "runpod-torch-v280",
        "registry_auth_id": "",
        "volume_id": provider["volume_id"],
        "volume_size_gb": provider["volume_size_gb"],
        "volume_mount_path": "/workspace",
        "ports": ["22/tcp"],
        "maximum_duration_seconds": duration,
    }
    intent_path = output / "intent.json"
    _atomic_json(intent_path, intent)
    workload_experiment = {
        "evaluation": {"allow_holdout": False},
        "data": {
            "holdout_start": "2026-01-01T00:00:00+00:00",
            "corpus_manifest_sha256": sha256_file(Path(str(control["input_manifest"]))),
            "root": str(Path(str(paths["input_manifest"])).parent),
            "corpus_manifest_path": paths["input_manifest"],
        },
        "model": {
            "hub_model": frozen.hub_model,
            "hub_revision": frozen.hub_revision,
            "weights_sha256": frozen.weights_sha256,
        },
        "run": {"artifact_root": "/workspace/mantis/runs"},
    }
    workload_experiment_path = output / "workload-experiment.json"
    _atomic_json(workload_experiment_path, workload_experiment)
    result = {
        "schema_version": 1,
        "focused_checks": str(checks_path),
        "preflight": str(preflight_path),
        "experiment": str(experiment_toml),
        "intent": str(intent_path),
        "workload_experiment": str(workload_experiment_path),
        "exact_command": exact_command,
        "maximum_duration_seconds": duration,
    }
    _atomic_json(output / "planning-inputs.json", result)
    return result


def seal_paid_frozen_workload(
    config_path: Path,
    planning_root: Path,
    decision_path: Path,
    manifest_root: Path,
    pod_manifest_path: Path,
    bound_decision_path: Path,
    *,
    evaluated_at: datetime,
) -> dict[str, Any]:
    """Seal and bind the frozen workload without an operator-authored workload spec."""
    from mantis_v2.runpod_config import load_launch_authorization, load_spend_ledger
    from mantis_v2.runpod_workload import bind_workload_decision, seal_workload_manifest
    from mantis_v2.transfer_bundle import load_bundle_manifest

    control = _paid_control_config(config_path)
    planning = json.loads((planning_root / "planning-inputs.json").read_text())
    decision = json.loads(decision_path.read_text())
    if not isinstance(planning, dict) or not isinstance(decision, dict):
        raise FrozenExpectedRError("paid planning or decision input is invalid")
    if decision.get("allowed") is not True:
        raise FrozenExpectedRError("paid launch decision is not authorized")
    paths = control["pod_paths"]
    artifacts = control["artifacts"]
    provider = control["provider"]
    bundle_path = Path(str(control["input_bundle_manifest"]))
    bundle = load_bundle_manifest(bundle_path)
    expected_input_root = Path("/workspace/mantis/inputs") / bundle.bundle_digest
    input_pod_path = Path(str(paths["input_manifest"]))
    config_pod_path = Path(str(paths["frozen_config"]))
    if not input_pod_path.is_relative_to(expected_input_root) or not config_pod_path.is_relative_to(
        expected_input_root
    ):
        raise FrozenExpectedRError("frozen input paths must be inside the promoted bundle")
    authorization_path = Path(str(control["authorization"]))
    ledger_path = Path(str(control["spend_ledger"]))
    authorization = load_launch_authorization(authorization_path)
    ledger = load_spend_ledger(ledger_path)
    preflight_path = planning_root / "preflight.json"
    workload_experiment_path = planning_root / "workload-experiment.json"
    run_id = str(control["run_id"])
    pod_root = f"/workspace/mantis/runs/{run_id}"
    exact_command = str(planning["exact_command"])
    maximum_duration = int(planning["maximum_duration_seconds"])
    compute_rate = Decimal(str(decision["observed_price_usd_per_gpu_hour"]))
    storage_rate = Decimal(str(provider["storage_usd_per_gb_hour"]))
    storage_gb = int(provider["volume_size_gb"])
    startup = int(decision["startup_allowance_seconds"])
    calculated_maximum = (
        (compute_rate + storage_rate * Decimal(storage_gb))
        * Decimal(maximum_duration + startup + 120)
        / Decimal(3600)
    )
    maximum_cell = max(calculated_maximum, Decimal(str(decision["projected_spend_usd"])))
    shutdown_reserve = compute_rate / Decimal(6) + storage_rate * Decimal(storage_gb) / Decimal(
        3600
    )
    auth_record = _dual_file_record(authorization_path, str(paths["authorization"]))
    core = {
        "schema_version": 1,
        "workload_kind": "frozen_expected_r",
        "run_id": run_id,
        "source_revision": control["source_revision"],
        "dependency_lock": _dual_file_record(
            Path(str(control["dependency_lock"])), str(paths["dependency_lock"])
        ),
        "image": {
            "ref": decision["image_ref"],
            "self_check": _dual_file_record(
                Path(str(control["official_bootstrap_receipt"])),
                str(paths["official_bootstrap_receipt"]),
            ),
        },
        "bootstrap": {
            "source_archive": _dual_file_record(
                Path(str(control["source_archive"])), str(paths["source_archive"])
            ),
            "project_root": f"/workspace/mantis/runtime/{control['source_revision']}",
            "venv_path": "/opt/mantis/venv",
            "uv_version": "0.9.0",
        },
        "experiment_config": _dual_file_record(
            workload_experiment_path, str(paths["workload_experiment"])
        ),
        "matrix_plan": _dual_file_record(preflight_path, str(paths["preflight"])),
        "matrix_base_config": _dual_file_record(
            Path(str(control["frozen_config"])), str(paths["frozen_config"])
        ),
        "input_bundle": {
            "manifest": _dual_file_record(bundle_path, str(paths["input_bundle_manifest"])),
            "bundle_digest": bundle.bundle_digest,
            "incoming_root": f"/workspace/mantis/transfer/incoming/{bundle.bundle_digest}/files",
            "final_parent": "/workspace/mantis/inputs",
        },
        "dataset_manifest": _dual_file_record(
            Path(str(control["input_manifest"])), str(paths["input_manifest"])
        ),
        "spend_ledger": _dual_file_record(ledger_path, str(paths["spend_ledger"])),
        "foundation_checkpoint": {
            "repository": "paris-noah/MantisV2",
            "revision": _HUB_REVISION,
            "sha256": _WEIGHTS_SHA256,
        },
        "start_command": ["bash", "-lc", f"{exact_command} || {exact_command}"],
        "artifacts": {
            "pod": pod_root,
            "controller": str(Path(str(artifacts["controller"])).resolve()),
            "backup": str(Path(str(artifacts["backup"])).resolve()),
        },
        "resume": {"enabled": True, "same_run_only": True, "provenance_required": True},
        "monitor": {
            "tensorboard": None,
            "heartbeat": f"{pod_root}/heartbeat.json",
            "poll_seconds": 30,
            "first_heartbeat_seconds": startup,
            "miss_limit": 4,
            "token": _dual_file_record(
                Path(str(control["heartbeat_token"])), str(paths["heartbeat_token"])
            ),
        },
        "maximum_duration_seconds": maximum_duration,
        "quoted_rates": {
            "compute_usd_per_hour": str(compute_rate),
            "storage_usd_per_gb_hour": str(storage_rate),
            "storage_gb": storage_gb,
        },
        "budget_guard": {
            "stage": "qualification",
            "reconciled_spend_usd": str(ledger.actual_spend_usd),
            "unbilled_live_accrual_usd": str(ledger.reserved_spend_usd),
            "stage_reconciled_spend_usd": str(ledger.bucket_actual_spend_usd["qualification"]),
            "next_cell_maximum_usd": str(maximum_cell),
            "shutdown_reserve_usd": str(shutdown_reserve),
        },
        "authorization": {
            **auth_record,
            "expires_at": authorization.expires_at.isoformat().replace("+00:00", "Z"),
            "autopay_disabled": authorization.autopay_disabled,
            "ordinary_launch_cutoff_usd": str(authorization.ordinary_launch_cutoff_usd),
            "campaign_ceiling_usd": str(authorization.campaign_ceiling_usd),
            "recovery_authorized": authorization.recovery_authorized,
        },
        "runpodctl": dict(control["runpodctl"]),
    }
    manifest = seal_workload_manifest(core, manifest_root)
    bound = bind_workload_decision(
        manifest_path=manifest,
        decision=decision,
        pod_manifest_path=pod_manifest_path,
        output_path=bound_decision_path,
        evaluated_at=evaluated_at,
    )
    return {"manifest": str(manifest), "bound_decision": str(bound)}


def write_paid_preflight(
    input_manifest_path: Path,
    embedding_output: Path,
    receipt_path: Path,
    *,
    exact_command: str,
    hourly_rate_usd: float,
    budget_usd: float,
    deadline_hours: float,
    check_duration_seconds: float,
    checks: dict[str, bool],
) -> dict[str, Any]:
    """Seal the cheap checks and safety envelope required before one L40S Pod."""
    required_checks = {
        "causality_next_fill",
        "label_replay_parity",
        "topstep_accounting",
        "artifact_resume",
    }
    if set(checks) != required_checks or not all(checks.values()):
        raise FrozenExpectedRError("all four focused preflight checks must pass")
    if check_duration_seconds < 0 or check_duration_seconds >= 60:
        raise FrozenExpectedRError("focused preflight must finish in under 60 seconds")
    if not exact_command.strip():
        raise FrozenExpectedRError("paid preflight requires the exact command")
    if budget_usd <= 0 or budget_usd > 10:
        raise FrozenExpectedRError("frozen-screen budget must be in (0, 10]")
    if hourly_rate_usd <= 0 or deadline_hours <= 0 or deadline_hours > 6:
        raise FrozenExpectedRError("paid rate and deadline must be positive and at most 6 hours")
    if deadline_hours * hourly_rate_usd > budget_usd:
        raise FrozenExpectedRError("deadline exceeds the rate-derived spend limit")
    if embedding_output.exists():
        raise FrozenExpectedRError("embedding output identity already exists")
    if receipt_path.exists() or receipt_path.with_suffix(receipt_path.suffix + ".tmp").exists():
        raise FrozenExpectedRError("paid preflight receipt already exists")
    try:
        input_manifest = json.loads(input_manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenExpectedRError("paid preflight input manifest is unreadable") from exc
    manifest_digest = input_manifest.get("manifest_sha256")
    unsigned = dict(input_manifest)
    unsigned.pop("manifest_sha256", None)
    if manifest_digest != _json_digest(unsigned):
        raise FrozenExpectedRError("paid preflight input manifest digest mismatch")
    receipt = {
        "schema_version": 1,
        "ready": True,
        "gpu": "NVIDIA L40S",
        "gpu_count": 1,
        "input_manifest": _file_record(input_manifest_path),
        "input_manifest_sha256": manifest_digest,
        "embedding_output": str(embedding_output.resolve()),
        "exact_command": exact_command,
        "hourly_rate_usd": hourly_rate_usd,
        "budget_usd": budget_usd,
        "deadline_hours": deadline_hours,
        "health_interval_seconds": 30,
        "maximum_provenance_resume": 1,
        "termination_policy": "terminate_after_success_or_second_failure",
        "focused_checks": checks,
        "focused_check_duration_seconds": check_duration_seconds,
    }
    receipt["receipt_sha256"] = _json_digest(receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(receipt_path, receipt)
    return receipt


def validate_paid_runner_contract(
    receipt_path: Path,
    workload_manifest: dict[str, Any],
    *,
    path_role: Literal["controller", "pod"] = "controller",
) -> dict[str, Any]:
    """Bind the frozen-screen preflight to the existing paid workload supervisor."""
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenExpectedRError("paid preflight receipt is unreadable") from exc
    if not isinstance(receipt, dict):
        raise FrozenExpectedRError("paid preflight receipt must be an object")
    unsigned = dict(receipt)
    receipt_digest = unsigned.pop("receipt_sha256", None)
    if receipt_digest != _json_digest(unsigned) or receipt.get("ready") is not True:
        raise FrozenExpectedRError("paid preflight receipt digest mismatch")
    input_record = receipt.get("input_manifest", {})
    staged_input = workload_manifest.get("dataset_manifest", {})
    input_path = Path(
        str(
            staged_input.get(f"{path_role}_path", input_record.get("path", ""))
            if isinstance(staged_input, dict)
            else input_record.get("path", "")
        )
    )
    if (
        not input_path.is_file()
        or input_path.stat().st_size != input_record.get("size")
        or sha256_file(input_path) != input_record.get("sha256")
    ):
        raise FrozenExpectedRError("paid preflight input identity changed")
    exact_command = receipt.get("exact_command")
    if workload_manifest.get("start_command") != [
        "bash",
        "-lc",
        f"{exact_command} || {exact_command}",
    ]:
        raise FrozenExpectedRError("paid workload does not enforce exactly one safe resume")
    artifacts = workload_manifest.get("artifacts", {})
    if str(artifacts.get("pod", "")) != str(receipt.get("embedding_output", "")):
        raise FrozenExpectedRError("paid workload output identity differs from preflight")
    monitor = workload_manifest.get("monitor", {})
    resume = workload_manifest.get("resume", {})
    duration = workload_manifest.get("maximum_duration_seconds")
    rates = workload_manifest.get("quoted_rates", {})
    budget = workload_manifest.get("budget_guard", {})
    if (
        monitor.get("poll_seconds") != receipt.get("health_interval_seconds")
        or receipt.get("health_interval_seconds") != 30
        or resume != {"enabled": True, "same_run_only": True, "provenance_required": True}
        or receipt.get("maximum_provenance_resume") != 1
        or receipt.get("termination_policy") != "terminate_after_success_or_second_failure"
        or not isinstance(duration, int)
        or duration > float(receipt.get("deadline_hours", 0)) * 3600
        or float(rates.get("compute_usd_per_hour", -1)) != float(receipt.get("hourly_rate_usd", -2))
        or float(budget.get("next_cell_maximum_usd", float("inf")))
        > float(receipt.get("budget_usd", -1))
    ):
        raise FrozenExpectedRError("paid workload differs from preflight safety envelope")

    def contains(value: Any, expected: Any) -> bool:
        if value == expected:
            return True
        if isinstance(value, dict):
            return any(contains(item, expected) for item in value.values())
        if isinstance(value, list):
            return any(contains(item, expected) for item in value)
        return False

    bundle = workload_manifest.get("input_bundle", {}).get("manifest", {})
    bundle_path = Path(str(bundle.get(f"{path_role}_path", "")))
    try:
        bundle_payload = json.loads(bundle_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenExpectedRError("paid workload input bundle is unreadable") from exc
    if not contains(bundle_payload, input_record.get("sha256")):
        raise FrozenExpectedRError("paid workload does not stage the preflight input")
    return receipt


class _OfficialFrozenModel:
    def __init__(self, config: FrozenExpectedRConfig) -> None:
        if not torch.cuda.is_available():
            raise FrozenExpectedRError("official frozen embedding requires CUDA")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        downloaded = Path(
            hf_hub_download(
                repo_id=config.hub_model,
                filename="model.safetensors",
                revision=config.hub_revision,
            )
        )
        if sha256_file(downloaded) != config.weights_sha256:
            raise FrozenExpectedRError("official weights digest mismatch")
        model = MantisV2(
            return_transf_layer=2,
            output_token="combined",
            device="cuda",
            pre_training=False,
        )
        loaded = model.from_pretrained(config.hub_model, revision=config.hub_revision)
        if not isinstance(loaded, torch.nn.Module):
            raise FrozenExpectedRError("official MantisV2 load returned an invalid model")
        self.model = loaded.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def __call__(self, values: np.ndarray, precision: str) -> np.ndarray:
        tensor = torch.from_numpy(values).to("cuda")
        batch, channels, length = tensor.shape
        autocast = torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=precision == "bf16"
        )
        with torch.inference_mode(), autocast:
            embedded = self.model(tensor.reshape(batch * channels, 1, length))
        result = embedded.reshape(batch, channels * 512).float().cpu().numpy()
        if result.shape != (batch, 2560) or not np.isfinite(result).all():
            raise FrozenExpectedRError("official MantisV2 returned invalid embeddings")
        return cast(np.ndarray, result)


class FrozenMantisEmbedder:
    """Validate one input and produce provenance-bound atomic embedding shards."""

    def __init__(
        self,
        config: FrozenExpectedRConfig,
        *,
        model_factory: Callable[[FrozenExpectedRConfig], _EmbeddingModel] = _OfficialFrozenModel,
    ) -> None:
        self.config = config
        self._model_factory = model_factory

    def validate_input(
        self, manifest_path: Path
    ) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenExpectedRError("frozen input manifest is unreadable") from exc
        expected_digest = manifest.pop("manifest_sha256", None)
        if expected_digest != _json_digest(manifest):
            raise FrozenExpectedRError("input manifest digest mismatch")
        if manifest.get("preprocessing") != "native_mantis_v2":
            raise FrozenExpectedRError("custom preprocessing is prohibited")
        active_source = source_digest(Path(__file__).resolve().parents[3])
        active_expected_r_config = _json_digest(asdict(ExpectedRScreenConfig()))
        if manifest.get("producer_source_sha256") != active_source:
            raise FrozenExpectedRError("frozen input producer source mismatch")
        if manifest.get("expected_r_config_sha256") != active_expected_r_config:
            raise FrozenExpectedRError("frozen input expected-R config mismatch")
        candidate_record = manifest.get("candidates", {})
        window_record = manifest.get("windows", {})
        candidates_path = _record_path(candidate_record, manifest_path.parent)
        windows_path = _record_path(window_record, manifest_path.parent)
        context_record = manifest.get("context", {})
        context_path = _record_path(context_record, manifest_path.parent)
        if not candidates_path.is_file() or sha256_file(candidates_path) != candidate_record.get(
            "sha256"
        ):
            raise FrozenExpectedRError("candidate digest mismatch")
        if not windows_path.is_file() or sha256_file(windows_path) != window_record.get("sha256"):
            raise FrozenExpectedRError("window digest mismatch")
        if not context_path.is_file() or sha256_file(context_path) != context_record.get("sha256"):
            raise FrozenExpectedRError("context digest mismatch")
        candidates = pd.read_parquet(candidates_path)
        windows = np.load(windows_path, mmap_mode="r", allow_pickle=False)
        context = np.load(context_path, mmap_mode="r", allow_pickle=False)
        if (
            len(candidates) != manifest.get("rows")
            or list(windows.shape) != manifest.get("shape")
            or str(windows.dtype) != manifest.get("dtype")
            or context.shape != (len(candidates), 3)
            or _json_digest(candidates["row_id"].tolist()) != manifest.get("row_ids_sha256")
        ):
            raise FrozenExpectedRError("frozen input rows or shape changed")
        manifest["manifest_sha256"] = expected_digest
        return manifest, candidates, windows, context

    def embed(
        self,
        manifest_path: Path,
        output: Path,
        *,
        maximum_shards: int | None = None,
    ) -> dict[str, Any]:
        input_manifest, candidates, windows, _context = self.validate_input(manifest_path)
        output.mkdir(parents=True, exist_ok=True)
        final_path = output / "manifest.json"
        if final_path.exists():
            return self._validate_final(final_path, input_manifest, len(candidates), output)
        model = self._model_factory(self.config)
        precision, parity = self._precision(model, np.asarray(windows[: min(len(windows), 2)]))
        completed = self._completed(output, input_manifest["manifest_sha256"], precision, parity)
        start = completed[-1]["row_stop"] if completed else 0
        for written, row_start in enumerate(range(start, len(candidates), self.config.shard_rows)):
            if maximum_shards is not None and written >= maximum_shards:
                break
            row_stop = min(row_start + self.config.shard_rows, len(candidates))
            parts = []
            for batch_start in range(row_start, row_stop, self.config.batch_size):
                batch_stop = min(batch_start + self.config.batch_size, row_stop)
                parts.append(model(np.asarray(windows[batch_start:batch_stop]), precision))
            values = np.concatenate(parts)
            number = len(completed)
            feature_path = output / f"features-{number:05d}.npy"
            metadata_path = output / f"metadata-{number:05d}.parquet"
            with feature_path.with_suffix(".npy.tmp").open("wb") as stream:
                np.save(stream, values.astype(np.float32), allow_pickle=False)
            os.replace(feature_path.with_suffix(".npy.tmp"), feature_path)
            candidates.iloc[row_start:row_stop][["row_id"]].to_parquet(
                metadata_path.with_suffix(".parquet.tmp"), index=False
            )
            os.replace(metadata_path.with_suffix(".parquet.tmp"), metadata_path)
            receipt = {
                "schema_version": 1,
                "number": number,
                "row_start": row_start,
                "row_stop": row_stop,
                "input_manifest_sha256": input_manifest["manifest_sha256"],
                "config_sha256": self.config.digest,
                "weights_sha256": self.config.weights_sha256,
                "precision": precision,
                "parity_sha256": _json_digest(parity),
                "features": _file_record(feature_path, portable=True),
                "metadata": _file_record(metadata_path, portable=True),
                "status": "complete",
            }
            receipt["artifact_sha256"] = _json_digest(receipt)
            _atomic_json(output / f"shard-{number:05d}.json", receipt)
            completed.append(receipt)
        result = {
            "schema_version": 1,
            "status": (
                "complete"
                if completed and completed[-1]["row_stop"] == len(candidates)
                else "partial"
            ),
            "rows": completed[-1]["row_stop"] if completed else 0,
            "feature_width": self.config.embedding_width,
            "dtype": "float32",
            "precision": precision,
            "bf16_parity": parity,
            "input_manifest_sha256": input_manifest["manifest_sha256"],
            "config_sha256": self.config.digest,
            "weights_sha256": self.config.weights_sha256,
            "shards": completed,
        }
        if result["status"] == "complete":
            result["artifact_sha256"] = _json_digest(result)
            _atomic_json(final_path, result)
        return result

    def _completed(
        self, output: Path, input_digest: str, precision: str, parity: dict[str, Any]
    ) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        row_start = 0
        for number, path in enumerate(sorted(output.glob("shard-*.json"))):
            if path.name != f"shard-{number:05d}.json":
                raise FrozenExpectedRError("embedding shard receipt sequence has a gap")
            receipt = json.loads(path.read_text())
            if (
                receipt.get("number") != number
                or receipt.get("row_start") != row_start
                or receipt.get("input_manifest_sha256") != input_digest
                or receipt.get("config_sha256") != self.config.digest
                or receipt.get("weights_sha256") != self.config.weights_sha256
                or receipt.get("precision") != precision
                or receipt.get("parity_sha256") != _json_digest(parity)
                or receipt.get("status") != "complete"
            ):
                raise FrozenExpectedRError("stale embedding shard receipt")
            unsigned = dict(receipt)
            artifact_digest = unsigned.pop("artifact_sha256", None)
            if artifact_digest != _json_digest(unsigned):
                raise FrozenExpectedRError("embedding shard artifact digest mismatch")
            for label in ("features", "metadata"):
                record = receipt.get(label, {})
                target = _record_path(record, output)
                if not target.is_file() or sha256_file(target) != record.get("sha256"):
                    raise FrozenExpectedRError(f"embedding {label} changed")
            row_start = int(receipt["row_stop"])
            completed.append(receipt)
        return completed

    def _validate_final(
        self,
        final_path: Path,
        input_manifest: dict[str, Any],
        expected_rows: int,
        output: Path,
    ) -> dict[str, Any]:
        try:
            final = json.loads(final_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenExpectedRError("embedding manifest is unreadable") from exc
        if not isinstance(final, dict):
            raise FrozenExpectedRError("embedding manifest must be an object")
        unsigned = dict(final)
        artifact_digest = unsigned.pop("artifact_sha256", None)
        if artifact_digest != _json_digest(unsigned):
            raise FrozenExpectedRError("embedding manifest artifact digest mismatch")
        precision = final.get("precision")
        parity = final.get("bf16_parity")
        if (
            final.get("status") != "complete"
            or final.get("rows") != expected_rows
            or final.get("input_manifest_sha256") != input_manifest["manifest_sha256"]
            or final.get("config_sha256") != self.config.digest
            or final.get("weights_sha256") != self.config.weights_sha256
            or precision not in {"bf16", "fp32"}
            or not isinstance(parity, dict)
            or (self.config.requested_precision == "fp32" and precision != "fp32")
            or (precision == "bf16" and parity.get("passed") is not True)
        ):
            raise FrozenExpectedRError("stale embedding manifest")
        completed = self._completed(output, input_manifest["manifest_sha256"], precision, parity)
        if (
            final.get("shards") != completed
            or not completed
            or completed[-1]["row_stop"] != expected_rows
        ):
            raise FrozenExpectedRError("embedding manifest references stale shards")
        return cast(dict[str, Any], final)

    def _precision(self, model: _EmbeddingModel, fixture: np.ndarray) -> tuple[str, dict[str, Any]]:
        if self.config.requested_precision == "fp32":
            return "fp32", {"passed": False, "reason": "fp32_requested"}
        fp32 = model(fixture, "fp32")
        bf16 = model(fixture, "bf16")
        maximum = float(np.max(np.abs(fp32 - bf16), initial=0.0))
        denominators = np.linalg.norm(fp32, axis=1) * np.linalg.norm(bf16, axis=1)
        cosine = np.divide(
            np.sum(fp32 * bf16, axis=1),
            denominators,
            out=np.zeros(len(fp32)),
            where=denominators > 0,
        )
        minimum = float(np.min(cosine, initial=1.0))
        passed = (
            maximum <= self.config.parity_max_abs and minimum >= self.config.parity_minimum_cosine
        )
        return ("bf16" if passed else "fp32"), {
            "passed": passed,
            "max_abs_difference": maximum,
            "minimum_row_cosine": minimum,
        }


def compare_frozen_to_raw(
    candidates: pd.DataFrame,
    raw_features: np.ndarray,
    mantis_features: np.ndarray,
    config: FrozenExpectedRConfig,
    *,
    maximum_workers: int = 2,
    comparison_device: Literal["cpu", "cuda"] = "cpu",
    progress: _ComparisonProgress | None = None,
) -> dict[str, Any]:
    """Compare both ridge arms on identical rows under the initial date gate."""
    raw = np.asarray(raw_features, dtype=np.float32)
    mantis = np.asarray(mantis_features, dtype=np.float32)
    if len(raw) != len(candidates) or len(mantis) != len(candidates):
        raise FrozenExpectedRError("raw and Mantis must use identical candidate rows")
    if (
        raw.ndim != 2
        or mantis.ndim != 2
        or not np.isfinite(raw).all()
        or not np.isfinite(mantis).all()
    ):
        raise FrozenExpectedRError("raw and Mantis features must be finite matrices")
    if maximum_workers < 1 or maximum_workers > 2:
        raise FrozenExpectedRError("comparison workers must be in [1, 2]")
    if comparison_device == "cuda" and not torch.cuda.is_available():
        raise FrozenExpectedRError("CUDA comparison requested but CUDA is unavailable")
    context = raw[:, -config.context_width :]
    combined = np.concatenate((mantis, context), axis=1)
    initial = _fit_fold(
        _INITIAL_SCREEN, candidates, raw, combined, config, comparison_device, progress
    )
    initial_result, initial_raw_passed, initial_mantis_passed, _, _ = initial
    if not initial_raw_passed and not initial_mantis_passed:
        result = {
            "schema_version": 1,
            "config_sha256": config.digest,
            "row_ids_sha256": _json_digest(candidates["row_id"].tolist()),
            "comparison_backend": _comparison_backend(comparison_device),
            "initial_screen": initial_result,
            "folds": [],
            "promotion": {
                "required_wins": 2,
                "raw_wins": 0,
                "mantis_wins": 0,
                "raw_pooled_selected_expectancy_interval_95": None,
                "mantis_pooled_selected_expectancy_interval_95": None,
                "selected_expectancy_interval_95": None,
            },
            "selected": "stop",
            "status": "stopped",
        }
        result["artifact_sha256"] = _json_digest(result)
        return result
    if comparison_device == "cuda":
        completed = [
            _fit_fold(fold, candidates, raw, combined, config, comparison_device, progress)
            for fold in _ANCHORED_FOLDS
        ]
    else:
        with ThreadPoolExecutor(max_workers=maximum_workers) as executor:
            completed = list(
                executor.map(
                    lambda fold: _fit_fold(
                        fold,
                        candidates,
                        raw,
                        combined,
                        config,
                        comparison_device,
                        progress,
                    ),
                    _ANCHORED_FOLDS,
                )
            )
    fold_results = [item[0] for item in completed]
    raw_wins = 0
    mantis_wins = 0
    selected_outcomes: dict[str, list[tuple[str, float]]] = {"raw": [], "mantis": []}
    for _fold_result, raw_passed, mantis_passed, raw_observations, mantis_observations in completed:
        raw_wins += int(raw_passed)
        mantis_wins += int(mantis_passed)
        selected_outcomes["raw"].extend(raw_observations)
        selected_outcomes["mantis"].extend(mantis_observations)
    mantis_interval = _stationary_expectancy_interval(selected_outcomes["mantis"], config)
    raw_interval = _stationary_expectancy_interval(selected_outcomes["raw"], config)
    if mantis_wins >= 2 and mantis_interval is not None and mantis_interval[0] > 0:
        selected = "mantis"
        selected_interval = mantis_interval
    elif raw_wins >= 2 and raw_interval is not None and raw_interval[0] > 0:
        selected = "raw"
        selected_interval = raw_interval
    else:
        selected = "stop"
        selected_interval = None
    result = {
        "schema_version": 1,
        "config_sha256": config.digest,
        "row_ids_sha256": _json_digest(candidates["row_id"].tolist()),
        "comparison_backend": _comparison_backend(comparison_device),
        "initial_screen": initial_result,
        "folds": fold_results,
        "promotion": {
            "required_wins": 2,
            "raw_wins": raw_wins,
            "mantis_wins": mantis_wins,
            "raw_pooled_selected_expectancy_interval_95": raw_interval,
            "mantis_pooled_selected_expectancy_interval_95": mantis_interval,
            "selected_expectancy_interval_95": selected_interval,
        },
        "selected": selected,
        "status": "passed" if selected != "stop" else "stopped",
    }
    result["artifact_sha256"] = _json_digest(result)
    return result


def _fit_fold(
    fold: tuple[str, str, str, str, str, str, str],
    candidates: pd.DataFrame,
    raw: np.ndarray,
    combined: np.ndarray,
    config: FrozenExpectedRConfig,
    comparison_device: Literal["cpu", "cuda"],
    progress: _ComparisonProgress | None = None,
) -> tuple[
    dict[str, Any],
    bool,
    bool,
    list[tuple[str, float]],
    list[tuple[str, float]],
]:
    name, train_start, train_end, validation_start, validation_end, test_start, test_end = fold
    print(f"frozen-screen-compare fold={name} state=started", flush=True)
    if progress is not None:
        progress.write("fold_started", fold=name)
    screen = ExpectedRScreen(
        ExpectedRScreenConfig(
            ridge_alpha=config.ridge_alpha,
            bootstrap_replicates=config.bootstrap_replicates,
            bootstrap_restart_probability=config.bootstrap_restart_probability,
            seed=config.seed,
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=test_end,
        )
    )
    raw_result = _fit_arm(screen, candidates, raw, comparison_device, f"{name}:raw", progress)
    mantis_result = _fit_arm(
        screen, candidates, combined, comparison_device, f"{name}:mantis", progress
    )
    raw_expectancy = raw_result["test"]["selected_expectancy"]
    mantis_expectancy = mantis_result["test"]["selected_expectancy"]
    gate = {
        "mse_beats_raw": mantis_result["test"]["mse"] < raw_result["test"]["mse"],
        "expectancy_beats_raw": bool(
            raw_expectancy is not None
            and mantis_expectancy is not None
            and mantis_expectancy > raw_expectancy
        ),
    }
    mantis_passed = bool(mantis_result["gate"]["passed"] and all(gate.values()))
    raw_passed = bool(raw_result["gate"]["passed"])
    raw_observations = list(
        zip(
            raw_result.pop("selected_session_keys"),
            raw_result.pop("selected_realized_net_r"),
            strict=True,
        )
    )
    mantis_observations = list(
        zip(
            mantis_result.pop("selected_session_keys"),
            mantis_result.pop("selected_realized_net_r"),
            strict=True,
        )
    )
    result = {
        "name": name,
        "raw": raw_result,
        "mantis": mantis_result,
        "mantis_vs_raw_gate": {
            "passed": mantis_passed,
            **gate,
            "paired_stationary_day_block_intervals_95": _paired_model_intervals(
                screen, candidates, raw_result, mantis_result
            ),
        },
    }
    print(f"frozen-screen-compare fold={name} state=complete", flush=True)
    if progress is not None:
        progress.write("fold_complete", fold=name)
    return result, raw_passed, mantis_passed, raw_observations, mantis_observations


def _fit_arm(
    screen: ExpectedRScreen,
    candidates: pd.DataFrame,
    features: np.ndarray,
    comparison_device: Literal["cpu", "cuda"],
    progress_label: str,
    progress: _ComparisonProgress | None = None,
) -> dict[str, Any]:
    print(f"frozen-screen-compare arm={progress_label} state=started", flush=True)
    fold, _separator, arm = progress_label.partition(":")
    if progress is not None:
        progress.write("arm_started", fold=fold, arm=arm)
    owned = candidates.copy()
    owned.attrs["raw_features"] = features
    owned.attrs["data_sha256"] = _json_digest(owned["row_id"].tolist())
    predictor = (
        None
        if comparison_device == "cpu"
        else lambda scaled, train, targets, weights, alpha: _cuda_ridge_predict(
            scaled,
            train,
            targets,
            weights,
            alpha,
            progress_label=progress_label,
        )
    )
    threshold_selector = (
        None
        if comparison_device == "cpu"
        else lambda rows, scores, desired: cuda_threshold(
            rows,
            scores,
            desired,
            progress_label=progress_label,
            progress=progress,
        )
    )
    interval_evaluator = (
        None
        if comparison_device == "cpu"
        else lambda rows, outcomes, predictions, selected, training_mean: (
            _cuda_paired_intervals(
                screen,
                rows,
                outcomes,
                predictions,
                selected,
                training_mean,
                progress_label=progress_label,
                progress=progress,
            )
        )
    )
    result = screen.fit(
        owned,
        ridge_predictor=predictor,
        threshold_selector=threshold_selector,
        interval_evaluator=interval_evaluator,
        stage_reporter=(
            None if progress is None else lambda stage: progress.write(stage, fold=fold, arm=arm)
        ),
    )
    test = screen._split_mask(candidates, screen.config.test_start, screen.config.test_end)
    scores = np.asarray([row["prediction"] for row in result["rows"]["values"]], dtype=np.float64)[
        test
    ]
    selected = screen._executed_mask(
        candidates.loc[test], scores, float(result["threshold"]["value"])
    )
    result["selected_realized_net_r"] = (
        candidates.loc[test, "net_r"].to_numpy(dtype=np.float64)[selected].tolist()
    )
    result["selected_session_keys"] = (
        screen._session_keys(candidates.loc[test, "decision_ts"])[selected].astype(str).tolist()
    )
    print(f"frozen-screen-compare arm={progress_label} state=complete", flush=True)
    if progress is not None:
        progress.write("arm_complete", fold=fold, arm=arm)
    return result


def _comparison_backend(device: Literal["cpu", "cuda"]) -> dict[str, Any]:
    if device == "cpu":
        return {"device": "cpu", "ridge_solver": "sklearn_lsqr"}
    return {
        "device": "cuda",
        "ridge_solver": "torch_weighted_fp64_cholesky",
        "threshold_solver": "torch_parallel_recurrence",
        "bootstrap_solver": "torch_weighted_fp64",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(),
    }


def _cuda_ridge_predict(
    features: np.ndarray,
    train: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    *,
    progress_label: str,
) -> np.ndarray:
    if not torch.cuda.is_available():
        raise FrozenExpectedRError("CUDA comparison requested but CUDA is unavailable")
    print(f"frozen-screen-compare arm={progress_label} state=cuda_transfer", flush=True)
    device = torch.device("cuda")
    with torch.inference_mode():
        x_train = torch.as_tensor(features[train], dtype=torch.float64, device=device)
        y_train = torch.as_tensor(targets[train], dtype=torch.float64, device=device)
        train_weight = torch.as_tensor(weights[train], dtype=torch.float64, device=device)
        weight_sum = train_weight.sum()
        x_mean = (x_train * train_weight[:, None]).sum(dim=0) / weight_sum
        y_mean = (y_train * train_weight).sum() / weight_sum
        centered_x = x_train - x_mean
        centered_y = y_train - y_mean
        print(f"frozen-screen-compare arm={progress_label} state=cuda_gram", flush=True)
        gram = centered_x.T @ (centered_x * train_weight[:, None])
        gram.diagonal().add_(alpha)
        right_hand_side = centered_x.T @ (centered_y * train_weight)
        print(f"frozen-screen-compare arm={progress_label} state=cuda_solve", flush=True)
        factor = torch.linalg.cholesky(gram)
        coefficients = torch.cholesky_solve(right_hand_side[:, None], factor).squeeze(1)
        predictions = np.empty(len(features), dtype=np.float64)
        print(f"frozen-screen-compare arm={progress_label} state=cuda_predict", flush=True)
        for start in range(0, len(features), 4096):
            stop = min(start + 4096, len(features))
            chunk = torch.as_tensor(features[start:stop], dtype=torch.float64, device=device)
            values = (chunk - x_mean) @ coefficients + y_mean
            predictions[start:stop] = values.cpu().numpy()
    return predictions


def cuda_threshold(
    rows: pd.DataFrame,
    scores: np.ndarray,
    desired: int,
    *,
    progress_label: str,
    progress: _ComparisonProgress | None = None,
) -> float:
    if not torch.cuda.is_available():
        raise FrozenExpectedRError("CUDA comparison requested but CUDA is unavailable")
    print(f"frozen-screen-compare arm={progress_label} state=cuda_threshold", flush=True)
    fold, _separator, arm = progress_label.partition(":")
    threshold_total = int(np.unique(scores).size)
    if progress is not None:
        progress.write("cuda_threshold", fold=fold, arm=arm, thresholds_total=threshold_total)
    device = torch.device("cuda")
    with torch.inference_mode():
        thresholds = torch.as_tensor(np.unique(scores), dtype=torch.float64, device=device)
        score_values = torch.as_tensor(scores, dtype=torch.float64, device=device)
        entries = torch.as_tensor(rows["entry_index"].to_numpy(dtype=np.int64), device=device)
        outcomes = torch.as_tensor(rows["outcome_index"].to_numpy(dtype=np.int64), device=device)
        busy_until = torch.full_like(thresholds, -1, dtype=torch.int64)
        selected_count = torch.zeros_like(thresholds, dtype=torch.int64)
        for entry, outcome, score in zip(entries, outcomes, score_values, strict=True):
            eligible = (entry > busy_until) & (score >= thresholds)
            selected_count += eligible
            busy_until = torch.where(eligible, outcome, busy_until)
        difference = torch.abs(selected_count - desired)
        best_difference = difference.min()
        threshold = thresholds[difference == best_difference].max()
    selected_threshold = float(threshold.cpu())
    if progress is not None:
        progress.write(
            "cuda_threshold",
            fold=fold,
            arm=arm,
            thresholds_done=threshold_total,
            thresholds_total=threshold_total,
        )
    return selected_threshold


def _cuda_paired_intervals(
    screen: ExpectedRScreen,
    rows: pd.DataFrame,
    outcomes: np.ndarray,
    predictions: np.ndarray,
    selected: np.ndarray,
    training_mean: float,
    *,
    progress_label: str,
    progress: _ComparisonProgress | None = None,
) -> dict[str, list[float] | None]:
    days = screen._session_keys(rows["decision_ts"])
    unique_days, day_codes = np.unique(days, return_inverse=True)
    if len(unique_days) < 2 or not selected.any():
        return {
            "mse_improvement_over_constant": None,
            "selected_expectancy": None,
            "selected_minus_take_all": None,
        }
    if not torch.cuda.is_available():
        raise FrozenExpectedRError("CUDA comparison requested but CUDA is unavailable")
    print(f"frozen-screen-compare arm={progress_label} state=cuda_bootstrap", flush=True)
    fold, _separator, arm = progress_label.partition(":")
    if progress is not None:
        progress.write(
            "cuda_bootstrap",
            fold=fold,
            arm=arm,
            bootstrap_total=screen.config.bootstrap_replicates,
        )
    multiplicities = np.zeros(
        (screen.config.bootstrap_replicates, len(unique_days)), dtype=np.int16
    )
    rng = np.random.default_rng(screen.config.seed)
    for replicate in range(screen.config.bootstrap_replicates):
        position = int(rng.integers(len(unique_days)))
        for _ in range(len(unique_days)):
            multiplicities[replicate, position] += 1
            if rng.random() < screen.config.bootstrap_restart_probability:
                position = int(rng.integers(len(unique_days)))
            else:
                position = (position + 1) % len(unique_days)
    device = torch.device("cuda")
    with torch.inference_mode():
        day_weights = torch.as_tensor(multiplicities, dtype=torch.float64, device=device)
        row_weights = day_weights[:, torch.as_tensor(day_codes, dtype=torch.int64, device=device)]
        outcome_values = torch.as_tensor(outcomes, dtype=torch.float64, device=device)
        prediction_values = torch.as_tensor(predictions, dtype=torch.float64, device=device)
        selected_values = torch.as_tensor(selected, dtype=torch.float64, device=device)
        row_count = row_weights.sum(dim=1)
        selected_weights = row_weights * selected_values
        selected_count = selected_weights.sum(dim=1)
        selected_means = (selected_weights @ outcome_values) / selected_count
        take_all_means = (row_weights @ outcome_values) / row_count
        differences = selected_means - take_all_means
        constant_error = (training_mean - outcome_values).square()
        prediction_error = (prediction_values - outcome_values).square()
        mse_improvements = (row_weights @ (constant_error - prediction_error)) / row_count

    def interval(values: torch.Tensor) -> list[float] | None:
        array = values.cpu().numpy()
        finite = array[np.isfinite(array)]
        if not len(finite):
            return None
        return [float(value) for value in np.quantile(finite, [0.025, 0.975])]

    result = {
        "mse_improvement_over_constant": interval(mse_improvements),
        "selected_expectancy": interval(selected_means),
        "selected_minus_take_all": interval(differences),
    }
    if progress is not None:
        progress.write(
            "cuda_bootstrap",
            fold=fold,
            arm=arm,
            bootstrap_done=screen.config.bootstrap_replicates,
            bootstrap_total=screen.config.bootstrap_replicates,
        )
    return result


def _stationary_expectancy_interval(
    observations: list[tuple[str, float]], config: FrozenExpectedRConfig
) -> list[float] | None:
    if not observations:
        return None
    frame = pd.DataFrame(observations, columns=["session", "outcome"])
    sessions = frame["session"].drop_duplicates().to_numpy()
    if len(sessions) < 2:
        return None
    rng = np.random.default_rng(config.seed)
    means = np.empty(config.bootstrap_replicates)
    for replicate in range(config.bootstrap_replicates):
        sampled: list[str] = []
        position = int(rng.integers(len(sessions)))
        while len(sampled) < len(sessions):
            sampled.append(str(sessions[position]))
            if rng.random() < config.bootstrap_restart_probability:
                position = int(rng.integers(len(sessions)))
            else:
                position = (position + 1) % len(sessions)
        values = np.concatenate(
            [frame.loc[frame["session"] == session, "outcome"].to_numpy() for session in sampled]
        )
        means[replicate] = values.mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _paired_model_intervals(
    screen: ExpectedRScreen,
    candidates: pd.DataFrame,
    raw_result: dict[str, Any],
    mantis_result: dict[str, Any],
) -> dict[str, list[float] | None]:
    test = screen._split_mask(candidates, screen.config.test_start, screen.config.test_end)
    rows = candidates.loc[test]
    outcomes = rows["net_r"].to_numpy(dtype=np.float64)
    raw_scores = np.asarray(
        [row["prediction"] for row in raw_result["rows"]["values"]], dtype=np.float64
    )[test]
    mantis_scores = np.asarray(
        [row["prediction"] for row in mantis_result["rows"]["values"]], dtype=np.float64
    )[test]
    raw_selected = screen._executed_mask(rows, raw_scores, float(raw_result["threshold"]["value"]))
    mantis_selected = screen._executed_mask(
        rows, mantis_scores, float(mantis_result["threshold"]["value"])
    )
    days = screen._session_keys(rows["decision_ts"])
    unique_days = np.unique(days)
    if len(unique_days) < 2:
        return {"mse_improvement_over_raw": None, "selected_expectancy_minus_raw": None}
    rng = np.random.default_rng(screen.config.seed)
    mse_delta = np.empty(screen.config.bootstrap_replicates)
    expectancy_delta = np.empty(screen.config.bootstrap_replicates)
    for replicate in range(screen.config.bootstrap_replicates):
        sampled: list[np.datetime64] = []
        position = int(rng.integers(len(unique_days)))
        while len(sampled) < len(unique_days):
            sampled.append(unique_days[position])
            if rng.random() < screen.config.bootstrap_restart_probability:
                position = int(rng.integers(len(unique_days)))
            else:
                position = (position + 1) % len(unique_days)
        indices = np.concatenate([np.flatnonzero(days == day) for day in sampled])
        mse_delta[replicate] = np.mean((raw_scores[indices] - outcomes[indices]) ** 2) - np.mean(
            (mantis_scores[indices] - outcomes[indices]) ** 2
        )
        raw_chosen = indices[raw_selected[indices]]
        mantis_chosen = indices[mantis_selected[indices]]
        expectancy_delta[replicate] = (
            outcomes[mantis_chosen].mean() - outcomes[raw_chosen].mean()
            if len(raw_chosen) and len(mantis_chosen)
            else np.nan
        )

    def interval(values: np.ndarray) -> list[float] | None:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return None
        return [float(value) for value in np.quantile(finite, [0.025, 0.975])]

    return {
        "mse_improvement_over_raw": interval(mse_delta),
        "selected_expectancy_minus_raw": interval(expectancy_delta),
    }
