"""Strict configuration for downstream strategy and Topstep workflows."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from mantis_v2.config import ConfigError

_TREND_MAGIC_CONTRACT_VERSION = "trend_magic_fixed_3r_v1"


@dataclass(frozen=True)
class DownstreamRunConfig:
    name: str
    seed: int
    artifact_root: Path
    device: Literal["cpu", "cuda", "mps"]
    allow_overwrite: bool


@dataclass(frozen=True)
class DownstreamDataConfig:
    root: Path
    file_format: Literal["csv", "parquet"]
    corpus_manifest_path: Path | None
    corpus_manifest_sha256: str
    symbols: tuple[str, ...]
    contract_multipliers: tuple[float, ...]
    timeframes: tuple[str, ...]
    decision_timeframe: str
    timestamp_column: str
    timestamp_semantics: Literal["bar_open", "bar_close"]
    feature_columns: tuple[str, ...]
    context_bars: int
    holdout_start: datetime
    max_relative_close_jump: float


@dataclass(frozen=True)
class FoundationConfig:
    manifest_path: Path
    weights_sha256: str
    export_role: Literal["diagnostic_candidate", "promoted"]
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
class TrendMagicStrategyConfig:
    kind: Literal["trend_magic"]
    atr_period: int
    cci_period: int
    trend_magic_atr_period: int
    trend_magic_multiplier: float
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
    solver: Literal["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
    regularization_c: float
    tolerance: float
    convergence_policy: Literal["fail", "record"]
    embed_manifest_path: Path | None
    embed_manifest_sha256: str
    embed_producer_config_path: Path | None
    embed_producer_config_sha256: str
    head_kind: Literal["logistic", "supervised_experts"] = "logistic"


@dataclass(frozen=True)
class SupervisedHeadConfig:
    hidden_width: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    early_stop_bars: int
    risk_penalty: float
    target_trades_per_symbol_day: float
    device: Literal["cpu", "mps", "cuda"]


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
    strategy: StrategyConfig | TrendMagicStrategyConfig
    walk_forward: WalkForwardConfig
    topstep: TopstepConfig
    evaluation: DownstreamEvaluationConfig
    supervised_head: SupervisedHeadConfig | None = None

    def _identity_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.walk_forward.head_kind == "logistic":
            payload["walk_forward"].pop("head_kind")
        if self.supervised_head is None:
            payload.pop("supervised_head")
        return payload

    def canonical_json(self) -> str:
        return json.dumps(
            self._identity_payload(), default=str, sort_keys=True, separators=(",", ":")
        )

    @property
    def strategy_contract(self) -> dict[str, Any] | None:
        """Return the named strategy recipe without changing legacy artifact identity."""
        if not isinstance(self.strategy, TrendMagicStrategyConfig):
            return None
        return {
            "version": _TREND_MAGIC_CONTRACT_VERSION,
            "input_timeframes": list(self.data.timeframes),
            "decision_timeframe": "3min",
            "candidate_rule": "every_eligible_closed_3min_state_bar",
            "direction_owner": "trend_magic",
            "entry_rule": "next_eligible_bar_open",
            "risk_rule": "0.5_atr_20",
            "primary_label_rule": "strict_3r_target_before_1r_stop",
            "analysis_targets_r": [2.0, 3.0, 4.0, 6.0],
            "horizon_rule": "120_bars_session_bounded",
            "session": "17:00-15:10 America/Chicago",
            "round_trip_cost_r": 0.03,
            "same_bar_policy": "stop_first",
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @property
    def workflow_digest(self) -> str:
        """Identify reusable artifacts while excluding the holdout unlock controls."""
        payload = self._identity_payload()
        del payload["evaluation"]
        encoded = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    @property
    def legacy_workflow_digest(self) -> str:
        """Reproduce the schema-v1 digest used before reusable head inputs."""
        payload = self._identity_payload()
        del payload["evaluation"]
        for key in (
            "solver",
            "regularization_c",
            "tolerance",
            "convergence_policy",
            "embed_manifest_path",
            "embed_manifest_sha256",
            "embed_producer_config_path",
            "embed_producer_config_sha256",
        ):
            del payload["walk_forward"][key]
        encoded = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    @property
    def embedding_contract_digest(self) -> str:
        """Identify data, label, and encoder semantics that produced embeddings."""
        payload = {
            "data": asdict(self.data),
            "foundation": asdict(self.foundation),
            "strategy": asdict(self.strategy),
        }
        encoded = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def head_config_digest(self, embed_manifest_sha256: str) -> str:
        """Identify a head fit independently from the reusable embeddings."""
        walk = asdict(self.walk_forward)
        if self.walk_forward.head_kind == "logistic":
            walk.pop("head_kind")
        walk.pop("embed_manifest_path")
        walk.pop("embed_manifest_sha256")
        walk.pop("embed_producer_config_path")
        walk.pop("embed_producer_config_sha256")
        payload = {
            "seed": self.run.seed,
            "walk_forward": walk,
            "embed_manifest_sha256": embed_manifest_sha256,
            "embedding_contract_digest": self.embedding_contract_digest,
        }
        if self.supervised_head is not None:
            payload["supervised_head"] = asdict(self.supervised_head)
        encoded = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


_EXPECTED: dict[str, set[str]] = {
    "run": {"name", "seed", "artifact_root", "device", "allow_overwrite"},
    "data": {
        "root",
        "file_format",
        "corpus_manifest_path",
        "corpus_manifest_sha256",
        "symbols",
        "contract_multipliers",
        "timeframes",
        "decision_timeframe",
        "timestamp_column",
        "timestamp_semantics",
        "feature_columns",
        "context_bars",
        "holdout_start",
        "max_relative_close_jump",
    },
    "foundation": {
        "manifest_path",
        "weights_sha256",
        "export_role",
        "return_transf_layer",
        "output_token",
        "preprocessing",
        "input_length",
        "batch_size",
        "shard_rows",
        "storage_dtype",
    },
    "strategy": {
        "kind",
        "atr_period",
        "supertrend_period",
        "supertrend_multiplier",
        "cci_period",
        "trend_magic_atr_period",
        "trend_magic_multiplier",
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
        "solver",
        "regularization_c",
        "tolerance",
        "convergence_policy",
        "embed_manifest_path",
        "embed_manifest_sha256",
        "embed_producer_config_path",
        "embed_producer_config_sha256",
        "head_kind",
    },
    "supervised_head": {
        "hidden_width",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "patience",
        "early_stop_bars",
        "risk_penalty",
        "target_trades_per_symbol_day",
        "device",
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

_OPTIONAL: dict[str, set[str]] = {
    "data": {
        "max_relative_close_jump",
        "file_format",
        "corpus_manifest_path",
        "corpus_manifest_sha256",
    },
    "walk_forward": {
        "solver",
        "regularization_c",
        "tolerance",
        "convergence_policy",
        "embed_manifest_path",
        "embed_manifest_sha256",
        "embed_producer_config_path",
        "embed_producer_config_sha256",
        "head_kind",
    },
}


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid [{name}] section")
    expected = _EXPECTED[name]
    if name == "strategy":
        kind = value.get("kind", "supertrend")
        if kind == "trend_magic":
            expected = {
                "kind",
                "atr_period",
                "cci_period",
                "trend_magic_atr_period",
                "trend_magic_multiplier",
                "stop_atr",
                "target_r",
                "analysis_targets_r",
                "horizon_bars",
                "round_trip_cost_r",
                "cooldown_bars",
                "same_bar_policy",
            }
        elif kind == "supertrend":
            expected = {
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
            }
            if "kind" in value:
                expected.add("kind")
        else:
            raise ConfigError("strategy.kind must be one of: supertrend, trend_magic")
    unknown = set(value) - expected
    missing = expected - _OPTIONAL.get(name, set()) - set(value)
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
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"{field} must be numeric and >= {minimum}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{field} must be finite")
    if result < minimum:
        raise ConfigError(f"{field} must be numeric and >= {minimum}")
    return result


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
    supervised = _section(raw, "supervised_head") if "supervised_head" in raw else None

    config = DownstreamConfig(
        run=DownstreamRunConfig(
            name=str(run["name"]),
            seed=_int(run["seed"], "run.seed"),
            artifact_root=Path(str(run["artifact_root"])),
            device=_choice(run["device"], "run.device", {"cpu", "cuda", "mps"}),
            allow_overwrite=_bool(run["allow_overwrite"], "run.allow_overwrite"),
        ),
        data=DownstreamDataConfig(
            root=Path(str(data["root"])),
            file_format=_choice(
                data.get("file_format", "csv"),
                "data.file_format",
                {"csv", "parquet"},
            ),
            corpus_manifest_path=(
                Path(str(data["corpus_manifest_path"]))
                if data.get("corpus_manifest_path")
                else None
            ),
            corpus_manifest_sha256=str(data.get("corpus_manifest_sha256", "")),
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
            max_relative_close_jump=_float(
                data.get("max_relative_close_jump", 0.05),
                "data.max_relative_close_jump",
                minimum=1e-12,
            ),
        ),
        foundation=FoundationConfig(
            manifest_path=Path(str(foundation["manifest_path"])),
            weights_sha256=str(foundation["weights_sha256"]),
            export_role=_choice(
                foundation["export_role"],
                "foundation.export_role",
                {"diagnostic_candidate", "promoted"},
            ),
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
        strategy=_strategy_config(strategy),
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
            solver=_choice(
                walk.get("solver", "lbfgs"),
                "walk_forward.solver",
                {"lbfgs", "liblinear", "newton-cg", "sag", "saga"},
            ),
            regularization_c=_float(
                walk.get("regularization_c", 1.0),
                "walk_forward.regularization_c",
                minimum=1e-12,
            ),
            tolerance=_float(
                walk.get("tolerance", 1e-4),
                "walk_forward.tolerance",
                minimum=1e-12,
            ),
            convergence_policy=_choice(
                walk.get("convergence_policy", "fail"),
                "walk_forward.convergence_policy",
                {"fail", "record"},
            ),
            embed_manifest_path=(
                Path(str(walk["embed_manifest_path"])) if walk.get("embed_manifest_path") else None
            ),
            embed_manifest_sha256=str(walk.get("embed_manifest_sha256", "")),
            embed_producer_config_path=(
                _resolve_config_path(walk["embed_producer_config_path"], config_path.parent)
                if walk.get("embed_producer_config_path")
                else None
            ),
            embed_producer_config_sha256=str(walk.get("embed_producer_config_sha256", "")),
            head_kind=_choice(
                walk.get("head_kind", "logistic"),
                "walk_forward.head_kind",
                {"logistic", "supervised_experts"},
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
        supervised_head=(
            SupervisedHeadConfig(
                hidden_width=_int(
                    supervised["hidden_width"], "supervised_head.hidden_width", minimum=1
                ),
                epochs=_int(supervised["epochs"], "supervised_head.epochs", minimum=1),
                batch_size=_int(supervised["batch_size"], "supervised_head.batch_size", minimum=1),
                learning_rate=_float(
                    supervised["learning_rate"],
                    "supervised_head.learning_rate",
                    minimum=1e-12,
                ),
                weight_decay=_float(supervised["weight_decay"], "supervised_head.weight_decay"),
                patience=_int(supervised["patience"], "supervised_head.patience", minimum=1),
                early_stop_bars=_int(
                    supervised["early_stop_bars"],
                    "supervised_head.early_stop_bars",
                    minimum=1,
                ),
                risk_penalty=_float(supervised["risk_penalty"], "supervised_head.risk_penalty"),
                target_trades_per_symbol_day=_float(
                    supervised["target_trades_per_symbol_day"],
                    "supervised_head.target_trades_per_symbol_day",
                    minimum=1e-12,
                ),
                device=_choice(
                    supervised["device"],
                    "supervised_head.device",
                    {"cpu", "mps", "cuda"},
                ),
            )
            if supervised is not None
            else None
        ),
    )
    _validate(config)
    return config


def _resolve_config_path(value: Any, config_directory: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_directory / path).resolve()


def _strategy_config(raw: dict[str, Any]) -> StrategyConfig | TrendMagicStrategyConfig:
    atr_period = _int(raw["atr_period"], "strategy.atr_period", minimum=1)
    stop_atr = _float(raw["stop_atr"], "strategy.stop_atr", minimum=1e-12)
    target_r = _float(raw["target_r"], "strategy.target_r", minimum=1e-12)
    analysis_targets_r = _floats(raw["analysis_targets_r"], "strategy.analysis_targets_r")
    horizon_bars = _int(raw["horizon_bars"], "strategy.horizon_bars", minimum=1)
    round_trip_cost_r = _float(raw["round_trip_cost_r"], "strategy.round_trip_cost_r")
    cooldown_bars = _int(raw["cooldown_bars"], "strategy.cooldown_bars")
    same_bar_policy: Literal["stop_first"] = _choice(
        raw["same_bar_policy"], "strategy.same_bar_policy", {"stop_first"}
    )
    if raw.get("kind", "supertrend") == "trend_magic":
        return TrendMagicStrategyConfig(
            kind="trend_magic",
            atr_period=atr_period,
            cci_period=_int(raw["cci_period"], "strategy.cci_period", minimum=2),
            trend_magic_atr_period=_int(
                raw["trend_magic_atr_period"],
                "strategy.trend_magic_atr_period",
                minimum=1,
            ),
            trend_magic_multiplier=_float(
                raw["trend_magic_multiplier"],
                "strategy.trend_magic_multiplier",
                minimum=1e-12,
            ),
            stop_atr=stop_atr,
            target_r=target_r,
            analysis_targets_r=analysis_targets_r,
            horizon_bars=horizon_bars,
            round_trip_cost_r=round_trip_cost_r,
            cooldown_bars=cooldown_bars,
            same_bar_policy=same_bar_policy,
        )
    return StrategyConfig(
        atr_period=atr_period,
        supertrend_period=_int(raw["supertrend_period"], "strategy.supertrend_period", minimum=1),
        supertrend_multiplier=_float(
            raw["supertrend_multiplier"],
            "strategy.supertrend_multiplier",
            minimum=1e-12,
        ),
        stop_atr=stop_atr,
        target_r=target_r,
        analysis_targets_r=analysis_targets_r,
        horizon_bars=horizon_bars,
        round_trip_cost_r=round_trip_cost_r,
        cooldown_bars=cooldown_bars,
        same_bar_policy=same_bar_policy,
    )


def _validate(config: DownstreamConfig) -> None:
    if config.walk_forward.head_kind == "supervised_experts":
        if config.supervised_head is None:
            raise ConfigError(
                "walk_forward.head_kind=supervised_experts requires [supervised_head]"
            )
        if not 0.0 <= config.supervised_head.risk_penalty <= 1.0:
            raise ConfigError("supervised_head.risk_penalty must be in [0, 1]")
    elif config.supervised_head is not None:
        raise ConfigError("[supervised_head] requires walk_forward.head_kind=supervised_experts")
    has_manifest_path = config.data.corpus_manifest_path is not None
    has_manifest_sha = bool(config.data.corpus_manifest_sha256)
    if config.data.file_format == "parquet":
        if not has_manifest_path or len(config.data.corpus_manifest_sha256) != 64:
            raise ConfigError(
                "Parquet data requires data.corpus_manifest_path and a full SHA-256 digest"
            )
    elif has_manifest_path or has_manifest_sha:
        raise ConfigError("CSV and synthetic data cannot bind a Parquet corpus manifest")
    supported_timeframes = {
        ("1min", "3min", "15min"),
        ("1min", "3min", "5min", "15min"),
    }
    if config.data.timeframes not in supported_timeframes:
        raise ConfigError(
            "data.timeframes must use an ordered registered three- or four-timeframe recipe"
        )
    if config.data.decision_timeframe != "3min":
        raise ConfigError("data.decision_timeframe must be 3min")
    if isinstance(config.strategy, TrendMagicStrategyConfig):
        recipe_values = (
            ("data.timestamp_semantics", config.data.timestamp_semantics, "bar_open"),
            ("strategy.atr_period", config.strategy.atr_period, 20),
            ("strategy.cci_period", config.strategy.cci_period, 20),
            (
                "strategy.trend_magic_atr_period",
                config.strategy.trend_magic_atr_period,
                5,
            ),
            (
                "strategy.trend_magic_multiplier",
                config.strategy.trend_magic_multiplier,
                1.0,
            ),
            ("strategy.stop_atr", config.strategy.stop_atr, 0.5),
            ("strategy.target_r", config.strategy.target_r, 3.0),
            (
                "strategy.analysis_targets_r",
                config.strategy.analysis_targets_r,
                (2.0, 3.0, 4.0, 6.0),
            ),
            ("strategy.horizon_bars", config.strategy.horizon_bars, 120),
            (
                "strategy.round_trip_cost_r",
                config.strategy.round_trip_cost_r,
                0.03,
            ),
            ("strategy.same_bar_policy", config.strategy.same_bar_policy, "stop_first"),
            ("topstep.session_timezone", config.topstep.session_timezone, "America/Chicago"),
            ("topstep.session_start_hour", config.topstep.session_start_hour, 17),
            ("topstep.session_end_hour", config.topstep.session_end_hour, 15),
            ("topstep.session_end_minute", config.topstep.session_end_minute, 10),
        )
        for field, actual, expected in recipe_values:
            if actual != expected:
                raise ConfigError(
                    f"{field} must be {expected!r} for {_TREND_MAGIC_CONTRACT_VERSION}"
                )
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
    has_embed_path = config.walk_forward.embed_manifest_path is not None
    has_embed_sha = bool(config.walk_forward.embed_manifest_sha256)
    has_producer_path = config.walk_forward.embed_producer_config_path is not None
    has_producer_sha = bool(config.walk_forward.embed_producer_config_sha256)
    if len({has_embed_path, has_embed_sha, has_producer_path, has_producer_sha}) != 1:
        raise ConfigError(
            "walk_forward reusable embed manifest and producer config paths and SHA-256 "
            "values must be set together"
        )
    if has_embed_sha and len(config.walk_forward.embed_manifest_sha256) != 64:
        raise ConfigError("walk_forward.embed_manifest_sha256 must be a full SHA-256 digest")
    if has_producer_sha and len(config.walk_forward.embed_producer_config_sha256) != 64:
        raise ConfigError("walk_forward.embed_producer_config_sha256 must be a full SHA-256 digest")
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
