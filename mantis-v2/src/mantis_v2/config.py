"""Strict TOML configuration for the MantisV2 pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


class ConfigError(ValueError):
    """Raised when configuration is missing, unknown, or inconsistent."""


Device = Literal["auto", "cpu", "mps", "cuda"]
ModelMode = Literal[
    "scratch",
    "head_only",
    "adapter_head",
    "transformer_finetune",
    "full_finetune",
]


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    artifact_root: Path
    device: Device
    require_accelerator: bool
    allow_overwrite: bool


@dataclass(frozen=True)
class DataConfig:
    root: str
    file_format: Literal["csv", "parquet"]
    corpus_manifest_path: Path | None
    corpus_manifest_sha256: str
    symbols: tuple[str, ...]
    intervals: tuple[str, ...]
    timestamp_column: str
    feature_columns: tuple[str, ...]
    holdout_start: datetime
    validation_fraction: float
    context_lengths: tuple[int, ...]
    target_reserve: int
    max_relative_close_jump: float


@dataclass(frozen=True)
class ModelConfig:
    source_repository: str
    source_revision: str
    hub_model: str
    hub_revision: str
    weights_sha256: str
    input_length: int
    channel_strategy: Literal["independent_concat"]
    mode: ModelMode


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    checkpoint_every: int
    resume: bool
    max_steps_per_epoch: int
    validation_max_steps: int
    warmup_epochs: int
    early_stopping_patience: int

    def learning_rate_for_step(self, step: int, steps_per_epoch: int) -> float:
        """Return the upstream per-update linear-warmup and cosine-decay rate."""
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        total_steps = self.epochs * steps_per_epoch
        if not 1 <= step <= total_steps:
            raise ValueError(f"step must be in [1, {total_steps}]")
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if warmup_steps and step < warmup_steps:
            return self.learning_rate * step / warmup_steps
        if not warmup_steps:
            if total_steps == 1:
                return self.learning_rate
            progress = (step - 1) / (total_steps - 1)
            return self.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        decay_steps = total_steps - warmup_steps
        progress = (step - warmup_steps) / decay_steps
        return self.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class TargetConfig:
    kind: Literal["nextleg"]
    horizons: tuple[int, ...]
    leg_cap: int
    leg_k: int
    normalization_clamp: float
    candle_loss_weight: float
    leg_loss_weight: float
    minimum_train_anchors: int
    minimum_validation_anchors: int


@dataclass(frozen=True)
class EvaluationConfig:
    allow_holdout: bool


@dataclass(frozen=True)
class ExportConfig:
    format: Literal["safetensors"]
    verify_atol: float
    verify_rtol: float


@dataclass(frozen=True)
class PipelineConfig:
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    target: TargetConfig
    evaluation: EvaluationConfig
    export: ExportConfig

    def canonical_json(self) -> str:
        """Return a stable representation suitable for provenance hashing."""
        return json.dumps(asdict(self), default=str, sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def _section(raw: dict[str, Any], name: str, expected: set[str]) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid [{name}] section")
    unknown = set(value) - expected
    if unknown:
        raise ConfigError(f"unknown [{name}] keys: {', '.join(sorted(unknown))}")
    missing = expected - set(value)
    if missing:
        raise ConfigError(f"missing [{name}] keys: {', '.join(sorted(missing))}")
    return value


def _tuple_of[T](value: Any, expected_type: type[T], field: str) -> tuple[T, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty array")
    if any(not isinstance(item, expected_type) for item in value):
        raise ConfigError(f"{field} contains an invalid value")
    return tuple(value)


def _positive(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{field} must be an integer")
    lower = 0 if allow_zero else 1
    if value < lower:
        raise ConfigError(f"{field} must be >= {lower}")
    return value


def _number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{field} must be finite")
    if result < minimum:
        raise ConfigError(f"{field} must be >= {minimum}")
    return result


def _choice(value: Any, field: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"{field} must be one of: {', '.join(sorted(choices))}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be true or false")
    return value


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ConfigError("data.holdout_start must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError("data.holdout_start is not valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ConfigError("data.holdout_start must include a timezone")
    return parsed


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline configuration file."""
    config_path = Path(path)
    try:
        raw = tomllib.loads(config_path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc

    expected_sections = {
        "run",
        "data",
        "model",
        "training",
        "target",
        "evaluation",
        "export",
    }
    unknown_sections = set(raw) - expected_sections
    if unknown_sections:
        raise ConfigError(f"unknown sections: {', '.join(sorted(unknown_sections))}")

    run = _section(
        raw,
        "run",
        {"name", "seed", "artifact_root", "device", "require_accelerator", "allow_overwrite"},
    )
    data_expected = {
        "root",
        "file_format",
        "corpus_manifest_path",
        "corpus_manifest_sha256",
        "symbols",
        "intervals",
        "timestamp_column",
        "feature_columns",
        "holdout_start",
        "validation_fraction",
        "context_lengths",
        "target_reserve",
        "max_relative_close_jump",
    }
    data_raw = raw.get("data")
    if isinstance(data_raw, dict):
        data_raw.setdefault("max_relative_close_jump", 0.05)
        data_raw.setdefault("file_format", "csv")
        data_raw.setdefault("corpus_manifest_path", "")
        data_raw.setdefault("corpus_manifest_sha256", "")
    data = _section(raw, "data", data_expected)
    model = _section(
        raw,
        "model",
        {
            "source_repository",
            "source_revision",
            "hub_model",
            "hub_revision",
            "weights_sha256",
            "input_length",
            "channel_strategy",
            "mode",
        },
    )
    training = _section(
        raw,
        "training",
        {
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "num_workers",
            "checkpoint_every",
            "resume",
            "max_steps_per_epoch",
            "validation_max_steps",
            "warmup_epochs",
            "early_stopping_patience",
        },
    )
    target = _section(
        raw,
        "target",
        {
            "kind",
            "horizons",
            "leg_cap",
            "leg_k",
            "normalization_clamp",
            "candle_loss_weight",
            "leg_loss_weight",
            "minimum_train_anchors",
            "minimum_validation_anchors",
        },
    )
    evaluation = _section(raw, "evaluation", {"allow_holdout"})
    export = _section(raw, "export", {"format", "verify_atol", "verify_rtol"})

    device = _choice(run["device"], "run.device", {"auto", "cpu", "mps", "cuda"})
    mode = _choice(
        model["mode"],
        "model.mode",
        {
            "scratch",
            "head_only",
            "adapter_head",
            "transformer_finetune",
            "full_finetune",
        },
    )
    validation_fraction = _number(data["validation_fraction"], "data.validation_fraction")
    if not 0.0 < validation_fraction < 1.0:
        raise ConfigError("data.validation_fraction must be between 0 and 1")
    input_length = _positive(model["input_length"], "model.input_length")
    if input_length % 32:
        raise ConfigError("model.input_length must be divisible by 32")

    config = PipelineConfig(
        run=RunConfig(
            name=str(run["name"]),
            seed=_positive(run["seed"], "run.seed", allow_zero=True),
            artifact_root=Path(str(run["artifact_root"])),
            device=device,  # type: ignore[arg-type]
            require_accelerator=_boolean(run["require_accelerator"], "run.require_accelerator"),
            allow_overwrite=_boolean(run["allow_overwrite"], "run.allow_overwrite"),
        ),
        data=DataConfig(
            root=str(data["root"]),
            file_format=_choice(data["file_format"], "data.file_format", {"csv", "parquet"}),  # type: ignore[arg-type]
            corpus_manifest_path=(
                Path(str(data["corpus_manifest_path"])) if data["corpus_manifest_path"] else None
            ),
            corpus_manifest_sha256=str(data["corpus_manifest_sha256"]),
            symbols=_tuple_of(data["symbols"], str, "data.symbols"),
            intervals=_tuple_of(data["intervals"], str, "data.intervals"),
            timestamp_column=str(data["timestamp_column"]),
            feature_columns=_tuple_of(data["feature_columns"], str, "data.feature_columns"),
            holdout_start=_timestamp(data["holdout_start"]),
            validation_fraction=validation_fraction,
            context_lengths=_tuple_of(data["context_lengths"], int, "data.context_lengths"),
            target_reserve=_positive(data["target_reserve"], "data.target_reserve"),
            max_relative_close_jump=_number(
                data["max_relative_close_jump"],
                "data.max_relative_close_jump",
                minimum=1e-12,
            ),
        ),
        model=ModelConfig(
            source_repository=str(model["source_repository"]),
            source_revision=str(model["source_revision"]),
            hub_model=str(model["hub_model"]),
            hub_revision=str(model["hub_revision"]),
            weights_sha256=str(model["weights_sha256"]),
            input_length=input_length,
            channel_strategy=_choice(
                model["channel_strategy"],
                "model.channel_strategy",
                {"independent_concat"},
            ),  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
        ),
        training=TrainingConfig(
            epochs=_positive(training["epochs"], "training.epochs"),
            batch_size=_positive(training["batch_size"], "training.batch_size"),
            learning_rate=_number(
                training["learning_rate"], "training.learning_rate", minimum=1e-12
            ),
            weight_decay=_number(training["weight_decay"], "training.weight_decay"),
            num_workers=_positive(training["num_workers"], "training.num_workers", allow_zero=True),
            checkpoint_every=_positive(training["checkpoint_every"], "training.checkpoint_every"),
            resume=_boolean(training["resume"], "training.resume"),
            max_steps_per_epoch=_positive(
                training["max_steps_per_epoch"],
                "training.max_steps_per_epoch",
                allow_zero=True,
            ),
            validation_max_steps=_positive(
                training["validation_max_steps"],
                "training.validation_max_steps",
                allow_zero=True,
            ),
            warmup_epochs=_positive(
                training["warmup_epochs"],
                "training.warmup_epochs",
                allow_zero=True,
            ),
            early_stopping_patience=_positive(
                training["early_stopping_patience"],
                "training.early_stopping_patience",
                allow_zero=True,
            ),
        ),
        target=TargetConfig(
            kind=_choice(target["kind"], "target.kind", {"nextleg"}),  # type: ignore[arg-type]
            horizons=_tuple_of(target["horizons"], int, "target.horizons"),
            leg_cap=_positive(target["leg_cap"], "target.leg_cap"),
            leg_k=_positive(target["leg_k"], "target.leg_k"),
            normalization_clamp=_number(
                target["normalization_clamp"],
                "target.normalization_clamp",
                minimum=1e-12,
            ),
            candle_loss_weight=_number(
                target["candle_loss_weight"],
                "target.candle_loss_weight",
                minimum=1e-12,
            ),
            leg_loss_weight=_number(
                target["leg_loss_weight"],
                "target.leg_loss_weight",
                minimum=1e-12,
            ),
            minimum_train_anchors=_positive(
                target["minimum_train_anchors"],
                "target.minimum_train_anchors",
            ),
            minimum_validation_anchors=_positive(
                target["minimum_validation_anchors"],
                "target.minimum_validation_anchors",
            ),
        ),
        evaluation=EvaluationConfig(
            allow_holdout=_boolean(evaluation["allow_holdout"], "evaluation.allow_holdout"),
        ),
        export=ExportConfig(
            format=_choice(export["format"], "export.format", {"safetensors"}),  # type: ignore[arg-type]
            verify_atol=_number(export["verify_atol"], "export.verify_atol"),
            verify_rtol=_number(export["verify_rtol"], "export.verify_rtol"),
        ),
    )
    _validate_cross_fields(config)
    return config


def _validate_cross_fields(config: PipelineConfig) -> None:
    has_manifest_path = config.data.corpus_manifest_path is not None
    has_manifest_sha = bool(config.data.corpus_manifest_sha256)
    if config.data.file_format == "parquet":
        if not has_manifest_path or len(config.data.corpus_manifest_sha256) != 64:
            raise ConfigError(
                "Parquet data requires data.corpus_manifest_path and a full SHA-256 digest"
            )
    elif has_manifest_path or has_manifest_sha:
        raise ConfigError("CSV and synthetic data cannot bind a Parquet corpus manifest")
    if len(config.model.source_revision) != 40 or len(config.model.hub_revision) != 40:
        raise ConfigError("model source and Hub revisions must be full 40-character commits")
    if len(config.model.weights_sha256) != 64:
        raise ConfigError("model.weights_sha256 must be a full SHA-256 digest")
    if any(length <= 0 for length in config.data.context_lengths):
        raise ConfigError("data.context_lengths must contain positive integers")
    if any(horizon <= 0 for horizon in config.target.horizons):
        raise ConfigError("target.horizons must contain positive integers")
    if config.training.warmup_epochs >= config.training.epochs:
        raise ConfigError("training.warmup_epochs must be less than training.epochs")
    configured_streams = len(config.data.symbols) * len(config.data.intervals)
    validation_samples = (
        config.training.batch_size * config.training.validation_max_steps
        if config.training.validation_max_steps
        else 0
    )
    if validation_samples and validation_samples < configured_streams:
        raise ConfigError(
            "bounded validation must include at least one sample per configured stream"
        )
    expected_features = ("open", "high", "low", "close", "volume")
    if config.data.feature_columns != expected_features:
        raise ConfigError(
            "data.feature_columns must be ordered exactly as " + ", ".join(expected_features)
        )
    minimum_reserve = max(config.data.context_lengths) + 2 * config.target.leg_cap
    if config.data.target_reserve < minimum_reserve:
        raise ConfigError(
            f"data.target_reserve must cover max_context + 2 * target.leg_cap ({minimum_reserve})"
        )
