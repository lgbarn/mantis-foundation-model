"""Official frozen MantisV2 expected-R embedding and raw-control comparison."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
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


_SOURCE_REVISION = "0c94f8ceb9f1d1421dd292ed917090df8c31605b"
_HUB_REVISION = "99fe0f548960e272fbfa4b82fd9b5b5956779dfd"
_WEIGHTS_SHA256 = "49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1"
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
        "2025-07-01",
        "2025-07-01",
        "2025-10-01",
        "2025-10-01",
        "2026-01-01",
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
    input_manifest, candidates, windows, context = FrozenMantisEmbedder(config).validate_input(
        input_manifest_path
    )
    try:
        embedding_manifest = json.loads(embedding_manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenExpectedRError("embedding manifest is unreadable") from exc
    if (
        embedding_manifest.get("status") != "complete"
        or embedding_manifest.get("input_manifest_sha256") != input_manifest["manifest_sha256"]
        or embedding_manifest.get("config_sha256") != config.digest
        or embedding_manifest.get("weights_sha256") != config.weights_sha256
        or embedding_manifest.get("rows") != len(candidates)
    ):
        raise FrozenExpectedRError("stale or incomplete embedding manifest")
    features: list[np.ndarray] = []
    row_ids: list[str] = []
    for number, shard in enumerate(embedding_manifest.get("shards", [])):
        if shard.get("number") != number:
            raise FrozenExpectedRError("embedding manifest shard sequence changed")
        for label in ("features", "metadata"):
            record = shard.get(label, {})
            path = Path(str(record.get("path", "")))
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise FrozenExpectedRError(f"embedding {label} changed")
        features.append(np.load(shard["features"]["path"], allow_pickle=False))
        row_ids.extend(pd.read_parquet(shard["metadata"]["path"])["row_id"].tolist())
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
) -> dict[str, Any]:
    if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
        raise FrozenExpectedRError(f"comparison output already exists: {output}")
    candidates, raw, mantis = load_frozen_embeddings(
        input_manifest_path, embedding_manifest_path, config
    )
    result = compare_frozen_to_raw(candidates, raw, mantis, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, result)
    return result


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
            final = json.loads(final_path.read_text())
            if not isinstance(final, dict):
                raise FrozenExpectedRError("embedding manifest must be an object")
            if final.get("input_manifest_sha256") != input_manifest["manifest_sha256"]:
                raise FrozenExpectedRError("stale embedding manifest")
            return cast(dict[str, Any], final)
        completed = self._completed(output, input_manifest["manifest_sha256"])
        start = completed[-1]["row_stop"] if completed else 0
        model = self._model_factory(self.config)
        precision, parity = self._precision(model, np.asarray(windows[: min(len(windows), 2)]))
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
                "features": _file_record(feature_path),
                "metadata": _file_record(metadata_path),
            }
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

    def _completed(self, output: Path, input_digest: str) -> list[dict[str, Any]]:
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
            ):
                raise FrozenExpectedRError("stale embedding shard receipt")
            for label in ("features", "metadata"):
                record = receipt.get(label, {})
                target = Path(str(record.get("path", "")))
                if not target.is_file() or sha256_file(target) != record.get("sha256"):
                    raise FrozenExpectedRError(f"embedding {label} changed")
            row_start = int(receipt["row_stop"])
            completed.append(receipt)
        return completed

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
    maximum_workers: int = 3,
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
    if maximum_workers < 1 or maximum_workers > len(_ANCHORED_FOLDS):
        raise FrozenExpectedRError("comparison workers must be in [1, 3]")
    context = raw[:, -config.context_width :]
    combined = np.concatenate((mantis, context), axis=1)
    with ThreadPoolExecutor(max_workers=maximum_workers) as executor:
        completed = list(
            executor.map(
                lambda fold: _fit_fold(fold, candidates, raw, combined, config),
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
        "initial_screen": fold_results[-1],
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
) -> tuple[
    dict[str, Any],
    bool,
    bool,
    list[tuple[str, float]],
    list[tuple[str, float]],
]:
    name, train_start, train_end, validation_start, validation_end, test_start, test_end = fold
    print(f"frozen-screen-compare fold={name} state=started", flush=True)
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
    raw_result = _fit_arm(screen, candidates, raw)
    mantis_result = _fit_arm(screen, candidates, combined)
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
    return result, raw_passed, mantis_passed, raw_observations, mantis_observations


def _fit_arm(
    screen: ExpectedRScreen, candidates: pd.DataFrame, features: np.ndarray
) -> dict[str, Any]:
    owned = candidates.copy()
    owned.attrs["raw_features"] = features
    owned.attrs["data_sha256"] = _json_digest(owned["row_id"].tolist())
    result = screen.fit(owned)
    decisions = pd.to_datetime(candidates["decision_ts"], utc=True)
    outcomes = pd.to_datetime(candidates["outcome_ts"], utc=True)
    test = screen._date_mask(decisions, outcomes, screen.config.test_start, screen.config.test_end)
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
    decisions = pd.to_datetime(candidates["decision_ts"], utc=True)
    outcomes_at = pd.to_datetime(candidates["outcome_ts"], utc=True)
    test = screen._date_mask(
        decisions, outcomes_at, screen.config.test_start, screen.config.test_end
    )
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
