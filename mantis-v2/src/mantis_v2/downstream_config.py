"""Strict configuration for downstream Supertrend and Topstep workflows."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from mantis_v2.config import ConfigError


@dataclass(frozen=True)
class DownstreamRunConfig:
    name: str
    seed: int
    artifact_root: Path
    device: Literal["auto", "cpu", "mps"]
    allow_overwrite: bool


@dataclass(frozen=True)
class DownstreamDataConfig:
    root: Path
    symbols: tuple[str, ...]
    contract_multipliers: tuple[float, ...]
    timeframes: tuple[str, ...]
    decision_timeframe: str
    timestamp_column: str
    timestamp_semantics: Literal["bar_open", "bar_close"]
    feature_columns: tuple[str, ...]
    context_bars: int
    holdout_start: datetime


@dataclass(frozen=True)
class FoundationConfig:
    manifest_path: Path
    weights_sha256: str
    return_transf_layer: int
    output_token: Literal["cls_token", "combined"]
    preprocessing: Literal["nextleg_standardized"]
    input_length: int
    batch_size: int
    shard_rows: int
    storage_dtype: Literal["float16", "float32"]


@dataclass(frozen=True)
class StrategyConfig:
    atr_period: int
    supertrend_period: int
    supertrend_multiplier: float
    stop_atr: float
    target_r: float
    analysis_targets_r: tuple[float, ...]
    horizon_bars: int
    round_trip_cost_r: float
    cooldown_bars: int
    same_bar_policy: Literal["stop_first"]


@dataclass(frozen=True)
class WalkForwardConfig:
    train_months: int
    validation_months: int
    test_months: int
    stride_months: int
    embargo_bars: int
    threshold_percentile: float
    max_fit_rows: int
    max_iter: int
    class_weight: Literal["balanced", "none"]


@dataclass(frozen=True)
class TopstepConfig:
    starting_balance: float
    profit_target: float
    mll_distance: float
    mll_lock_balance: float
    consistency_limit: float
    consistency_comparator: Literal["strict", "inclusive"]
    max_mini_equivalents: float
    contracts: int
    minimum_trading_days: int
    session_timezone: str
    session_start_hour: int
    session_end_hour: int
    session_end_minute: int
    daily_loss_limit_enabled: bool
    daily_loss_limit: float


@dataclass(frozen=True)
class DownstreamEvaluationConfig:
    allow_holdout: bool
    holdout_unlock: str


@dataclass(frozen=True)
class DownstreamConfig:
    run: DownstreamRunConfig
    data: DownstreamDataConfig
    foundation: FoundationConfig
    strategy: StrategyConfig
    walk_forward: WalkForwardConfig
    topstep: TopstepConfig
    evaluation: DownstreamEvaluationConfig

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @property
    def workflow_digest(self) -> str:
        """Identify reusable artifacts while excluding the holdout unlock controls."""
        payload = asdict(self)
        del payload["evaluation"]
        encoded = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


_EXPECTED: dict[str, set[str]] = {
    "run": {"name", "seed", "artifact_root", "device", "allow_overwrite"},
    "data": {
        "root",
        "symbols",
        "contract_multipliers",
        "timeframes",
        "decision_timeframe",
        "timestamp_column",
        "timestamp_semantics",
        "feature_columns",
        "context_bars",
        "holdout_start",
    },
    "foundation": {
        "manifest_path",
        "weights_sha256",
        "return_transf_layer",
        "output_token",
        "preprocessing",
        "input_length",
        "batch_size",
        "shard_rows",
        "storage_dtype",
    },
    "strategy": {
        "atr_period",
        "supertrend_period",
        "supertrend_multiplier",
        "stop_atr",
        "target_r",
        "analysis_targets_r",
        "horizon_bars",
        "round_trip_cost_r",
        "cooldown_bars",
        "same_bar_policy",
    },
    "walk_forward": {
        "train_months",
        "validation_months",
        "test_months",
        "stride_months",
        "embargo_bars",
        "threshold_percentile",
        "max_fit_rows",
        "max_iter",
        "class_weight",
    },
    "topstep": {
        "starting_balance",
        "profit_target",
        "mll_distance",
        "mll_lock_balance",
        "consistency_limit",
        "consistency_comparator",
        "max_mini_equivalents",
        "contracts",
        "minimum_trading_days",
        "session_timezone",
        "session_start_hour",
        "session_end_hour",
        "session_end_minute",
        "daily_loss_limit_enabled",
        "daily_loss_limit",
    },
    "evaluation": {"allow_holdout", "holdout_unlock"},
}


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid [{name}] section")
    unknown = set(value) - _EXPECTED[name]
    missing = _EXPECTED[name] - set(value)
    if unknown:
        raise ConfigError(f"unknown [{name}] keys: {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigError(f"missing [{name}] keys: {', '.join(sorted(missing))}")
    return value


def _int(value: Any, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _float(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or float(value) < minimum:
        raise ConfigError(f"{field} must be numeric and >= {minimum}")
    return float(value)


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be true or false")
    return value


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(v, str) for v in value):
        raise ConfigError(f"{field} must be a non-empty string array")
    return tuple(value)


def _floats(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty numeric array")
    return tuple(_float(item, field, minimum=1e-12) for item in value)


def _choice[T: str](value: Any, field: str, choices: set[T]) -> T:
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"{field} must be one of: {', '.join(sorted(choices))}")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be an ISO-8601 string")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"{field} is not valid ISO-8601") from exc
    if result.tzinfo is None:
        raise ConfigError(f"{field} must include a timezone")
    return result


def _apply_overrides(raw: dict[str, Any], overrides: tuple[str, ...]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"override must be section.key=value: {override}")
        dotted, encoded = override.split("=", 1)
        parts = dotted.split(".")
        if len(parts) != 2 or parts[0] not in _EXPECTED or parts[1] not in _EXPECTED[parts[0]]:
            raise ConfigError(f"unknown override key: {dotted}")
        try:
            value = tomllib.loads(f"value = {encoded}")["value"]
        except tomllib.TOMLDecodeError:
            value = encoded
        section = raw.get(parts[0])
        if not isinstance(section, dict):
            raise ConfigError(f"missing or invalid [{parts[0]}] section")
        section[parts[1]] = value


def load_downstream_config(path: str | Path, overrides: tuple[str, ...] = ()) -> DownstreamConfig:
    """Load, override, and strictly validate a downstream TOML file."""
    config_path = Path(path)
    try:
        raw = tomllib.loads(config_path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc
    unknown_sections = set(raw) - set(_EXPECTED)
    if unknown_sections:
        raise ConfigError(f"unknown sections: {', '.join(sorted(unknown_sections))}")
    _apply_overrides(raw, overrides)
    run = _section(raw, "run")
    data = _section(raw, "data")
    foundation = _section(raw, "foundation")
    strategy = _section(raw, "strategy")
    walk = _section(raw, "walk_forward")
    topstep = _section(raw, "topstep")
    evaluation = _section(raw, "evaluation")

    config = DownstreamConfig(
        run=DownstreamRunConfig(
            name=str(run["name"]),
            seed=_int(run["seed"], "run.seed"),
            artifact_root=Path(str(run["artifact_root"])),
            device=_choice(run["device"], "run.device", {"auto", "cpu", "mps"}),
            allow_overwrite=_bool(run["allow_overwrite"], "run.allow_overwrite"),
        ),
        data=DownstreamDataConfig(
            root=Path(str(data["root"])),
            symbols=_strings(data["symbols"], "data.symbols"),
            contract_multipliers=_floats(data["contract_multipliers"], "data.contract_multipliers"),
            timeframes=_strings(data["timeframes"], "data.timeframes"),
            decision_timeframe=str(data["decision_timeframe"]),
            timestamp_column=str(data["timestamp_column"]),
            timestamp_semantics=_choice(
                data["timestamp_semantics"],
                "data.timestamp_semantics",
                {"bar_open", "bar_close"},
            ),
            feature_columns=_strings(data["feature_columns"], "data.feature_columns"),
            context_bars=_int(data["context_bars"], "data.context_bars", minimum=2),
            holdout_start=_timestamp(data["holdout_start"], "data.holdout_start"),
        ),
        foundation=FoundationConfig(
            manifest_path=Path(str(foundation["manifest_path"])),
            weights_sha256=str(foundation["weights_sha256"]),
            return_transf_layer=_int(
                foundation["return_transf_layer"],
                "foundation.return_transf_layer",
                minimum=-1,
            ),
            output_token=_choice(
                foundation["output_token"],
                "foundation.output_token",
                {"cls_token", "combined"},
            ),
            preprocessing=_choice(
                foundation["preprocessing"],
                "foundation.preprocessing",
                {"nextleg_standardized"},
            ),
            input_length=_int(foundation["input_length"], "foundation.input_length", minimum=32),
            batch_size=_int(foundation["batch_size"], "foundation.batch_size", minimum=1),
            shard_rows=_int(foundation["shard_rows"], "foundation.shard_rows", minimum=1),
            storage_dtype=_choice(
                foundation["storage_dtype"],
                "foundation.storage_dtype",
                {"float16", "float32"},
            ),
        ),
        strategy=StrategyConfig(
            atr_period=_int(strategy["atr_period"], "strategy.atr_period", minimum=1),
            supertrend_period=_int(
                strategy["supertrend_period"], "strategy.supertrend_period", minimum=1
            ),
            supertrend_multiplier=_float(
                strategy["supertrend_multiplier"],
                "strategy.supertrend_multiplier",
                minimum=1e-12,
            ),
            stop_atr=_float(strategy["stop_atr"], "strategy.stop_atr", minimum=1e-12),
            target_r=_float(strategy["target_r"], "strategy.target_r", minimum=1e-12),
            analysis_targets_r=_floats(
                strategy["analysis_targets_r"], "strategy.analysis_targets_r"
            ),
            horizon_bars=_int(strategy["horizon_bars"], "strategy.horizon_bars", minimum=1),
            round_trip_cost_r=_float(strategy["round_trip_cost_r"], "strategy.round_trip_cost_r"),
            cooldown_bars=_int(strategy["cooldown_bars"], "strategy.cooldown_bars"),
            same_bar_policy=_choice(
                strategy["same_bar_policy"], "strategy.same_bar_policy", {"stop_first"}
            ),
        ),
        walk_forward=WalkForwardConfig(
            train_months=_int(walk["train_months"], "walk_forward.train_months", minimum=1),
            validation_months=_int(
                walk["validation_months"], "walk_forward.validation_months", minimum=1
            ),
            test_months=_int(walk["test_months"], "walk_forward.test_months", minimum=1),
            stride_months=_int(walk["stride_months"], "walk_forward.stride_months", minimum=1),
            embargo_bars=_int(walk["embargo_bars"], "walk_forward.embargo_bars"),
            threshold_percentile=_float(
                walk["threshold_percentile"], "walk_forward.threshold_percentile"
            ),
            max_fit_rows=_int(walk["max_fit_rows"], "walk_forward.max_fit_rows", minimum=2),
            max_iter=_int(walk["max_iter"], "walk_forward.max_iter", minimum=1),
            class_weight=_choice(
                walk["class_weight"], "walk_forward.class_weight", {"balanced", "none"}
            ),
        ),
        topstep=TopstepConfig(
            starting_balance=_float(topstep["starting_balance"], "topstep.starting_balance"),
            profit_target=_float(topstep["profit_target"], "topstep.profit_target", minimum=1e-12),
            mll_distance=_float(topstep["mll_distance"], "topstep.mll_distance", minimum=1e-12),
            mll_lock_balance=_float(topstep["mll_lock_balance"], "topstep.mll_lock_balance"),
            consistency_limit=_float(
                topstep["consistency_limit"], "topstep.consistency_limit", minimum=1e-12
            ),
            consistency_comparator=_choice(
                topstep["consistency_comparator"],
                "topstep.consistency_comparator",
                {"strict", "inclusive"},
            ),
            max_mini_equivalents=_float(
                topstep["max_mini_equivalents"],
                "topstep.max_mini_equivalents",
                minimum=1e-12,
            ),
            contracts=_int(topstep["contracts"], "topstep.contracts", minimum=1),
            minimum_trading_days=_int(
                topstep["minimum_trading_days"], "topstep.minimum_trading_days", minimum=1
            ),
            session_timezone=str(topstep["session_timezone"]),
            session_start_hour=_int(topstep["session_start_hour"], "topstep.session_start_hour"),
            session_end_hour=_int(topstep["session_end_hour"], "topstep.session_end_hour"),
            session_end_minute=_int(topstep["session_end_minute"], "topstep.session_end_minute"),
            daily_loss_limit_enabled=_bool(
                topstep["daily_loss_limit_enabled"], "topstep.daily_loss_limit_enabled"
            ),
            daily_loss_limit=_float(topstep["daily_loss_limit"], "topstep.daily_loss_limit"),
        ),
        evaluation=DownstreamEvaluationConfig(
            allow_holdout=_bool(evaluation["allow_holdout"], "evaluation.allow_holdout"),
            holdout_unlock=str(evaluation["holdout_unlock"]),
        ),
    )
    _validate(config)
    return config


def _validate(config: DownstreamConfig) -> None:
    if config.data.timeframes != ("1min", "3min", "15min"):
        raise ConfigError("data.timeframes must be ordered exactly as 1min, 3min, 15min")
    if config.data.decision_timeframe != "3min":
        raise ConfigError("data.decision_timeframe must be 3min")
    if len(config.data.contract_multipliers) != len(config.data.symbols):
        raise ConfigError("data.contract_multipliers must align one-for-one with data.symbols")
    if config.data.feature_columns != ("open", "high", "low", "close", "volume"):
        raise ConfigError("data.feature_columns must be ordered open, high, low, close, volume")
    if config.foundation.input_length % 32:
        raise ConfigError("foundation.input_length must be divisible by 32")
    if len(config.foundation.weights_sha256) != 64:
        raise ConfigError("foundation.weights_sha256 must be a full SHA-256 digest")
    if config.foundation.return_transf_layer > 5:
        raise ConfigError("foundation.return_transf_layer must be -1 or a layer index from 0 to 5")
    if config.walk_forward.threshold_percentile > 100:
        raise ConfigError("walk_forward.threshold_percentile must be <= 100")
    if not 0 < config.topstep.consistency_limit <= 1:
        raise ConfigError("topstep.consistency_limit must be in (0, 1]")
    if not 0 <= config.topstep.session_start_hour <= 23:
        raise ConfigError("topstep.session_start_hour must be in [0, 23]")
    if not 0 <= config.topstep.session_end_hour <= 23:
        raise ConfigError("topstep.session_end_hour must be in [0, 23]")
    if not 0 <= config.topstep.session_end_minute <= 59:
        raise ConfigError("topstep.session_end_minute must be in [0, 59]")
    if config.topstep.contracts > config.topstep.max_mini_equivalents:
        raise ConfigError("topstep.contracts exceeds topstep.max_mini_equivalents")
    if config.topstep.daily_loss_limit_enabled and config.topstep.daily_loss_limit <= 0:
        raise ConfigError("topstep.daily_loss_limit must be positive when enabled")
