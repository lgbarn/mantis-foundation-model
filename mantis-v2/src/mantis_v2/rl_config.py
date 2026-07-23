"""Strict configuration for the provenance-locked RL entry experiment."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from mantis_v2.config import ConfigError


@dataclass(frozen=True)
class RlUpstreamConfig:
    source_digest: str
    lock_digest: str
    rule_contract_path: Path
    rule_contract_sha256: str
    downstream_config_path: Path
    downstream_config_sha256: str
    corpus_manifest_path: Path
    corpus_manifest_sha256: str
    embedding_manifest_path: Path
    embedding_manifest_sha256: str
    foundation_manifest_path: Path
    foundation_manifest_sha256: str
    foundation_weights_path: Path
    foundation_weights_sha256: str


@dataclass(frozen=True)
class RlRunConfig:
    name: str
    profile: Literal["smoke", "production"]
    seed: int
    device: Literal["cpu"]
    artifact_root: Path


@dataclass(frozen=True)
class RlPolicyConfig:
    role: Literal["entry"]
    algorithm: Literal["maskable_ppo"]
    actions: tuple[str, ...]
    ticker_conditioning: bool
    critic: Literal["ticker_specific_heads"]
    embedding_projection_dim: int


@dataclass(frozen=True)
class RlEpisodeConfig:
    ticker_mode: Literal["one_per_episode"]
    ticker_sampling: Literal["uniform"]
    account_start: float
    timeout_trading_days: int
    randomize_starting_cushion: bool


@dataclass(frozen=True)
class RlExecutionConfig:
    adverse_slippage_ticks_per_side: float
    stress_adverse_slippage_ticks_per_side: float
    fee_schedule: Literal["topstepx-2026-07-20"]
    cost_booking: Literal["account_simulator_only"]


@dataclass(frozen=True)
class RlFeeConfig:
    snapshot: Literal["topstepx-2026-07-20"]
    es: float
    mes: float
    nq: float
    mnq: float
    rty: float
    m2k: float
    ym: float
    mym: float
    gc: float
    mgc: float
    cl: float
    mcl: float
    zb: float


@dataclass(frozen=True)
class RlExitConfig:
    policy: Literal["two_r_trailing_v0"]
    initial_stop_r: float
    activation_r: float
    giveback_r: float
    horizon_bars: int
    same_bar_policy: Literal["prior_stop_first"]


@dataclass(frozen=True)
class RlSizingConfig:
    actor_controls_size: bool
    episode_profiles: tuple[str, ...]
    profile_sampling: Literal["uniform_when_supported"]
    mini_quantity: int
    micro_quantity: int
    mini_only_instruments: tuple[str, ...]


@dataclass(frozen=True)
class RlTopstepConfig:
    rule_snapshot: Literal["topstep-100k-2026-07-20"]
    daily_loss_limit_enabled: bool
    daily_loss_limit_dollars: float
    daily_loss_limit_action: Literal["flatten_cancel_lockout"]
    daily_loss_limit_terminal: bool


@dataclass(frozen=True)
class RlRewardConfig:
    kind: Literal["terminal_pass_with_blow_cost"]
    gamma: float
    potential_shaping: bool


@dataclass(frozen=True)
class RlConstraintConfig:
    kind: Literal["episodic_blow_lagrangian"]
    cost_limit: float
    cost_gamma: float
    lambda_init: float
    lambda_lr: float
    lambda_max: float
    minimum_cushion_role: Literal["observation_metric_only"]


@dataclass(frozen=True)
class RlTrainingConfig:
    development_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    serving_seed: int
    vector_environments: int
    ppo_epochs: int
    minibatch_size: int
    smoke_timesteps: int
    search_timesteps_per_seed: int
    search_seeds: int
    maximum_search_trials: int
    development_timesteps_per_seed: int
    confirmation_timesteps_per_seed: int
    maximum_timesteps_per_seed: int


@dataclass(frozen=True)
class RlEvaluationConfig:
    market_uncertainty: Literal["synchronized_adjacent_week_moving_block_bootstrap"]
    market_block_length_weeks: int
    sealed_holdout_start: datetime
    minimum_raw_pass_rate: float
    minimum_seed_raw_pass_rate: float
    minimum_pass_rate_lcb_95: float
    maximum_observed_blows: int
    maximum_blow_rate_ucb_95: float
    minimum_chronological_attempts: int


@dataclass(frozen=True)
class RlConfig:
    schema_version: int
    upstream: RlUpstreamConfig
    run: RlRunConfig
    policy: RlPolicyConfig
    episode: RlEpisodeConfig
    execution: RlExecutionConfig
    fees: RlFeeConfig
    exit: RlExitConfig
    sizing: RlSizingConfig
    topstep: RlTopstepConfig
    reward: RlRewardConfig
    constraint: RlConstraintConfig
    training: RlTrainingConfig
    evaluation: RlEvaluationConfig

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @property
    def rule_digest(self) -> str:
        payload = {
            "downstream_config_sha256": self.upstream.downstream_config_sha256,
            "rule_contract_sha256": self.upstream.rule_contract_sha256,
            "episode": asdict(self.episode),
            "exit": asdict(self.exit),
            "sizing": asdict(self.sizing),
            "topstep": asdict(self.topstep),
        }
        return _digest(payload)

    @property
    def fee_digest(self) -> str:
        return _digest({"execution": asdict(self.execution), "fees": asdict(self.fees)})


_EXPECTED: dict[str, set[str]] = {
    "upstream": {
        "source_digest",
        "lock_digest",
        "rule_contract_path",
        "rule_contract_sha256",
        "downstream_config_path",
        "downstream_config_sha256",
        "corpus_manifest_path",
        "corpus_manifest_sha256",
        "embedding_manifest_path",
        "embedding_manifest_sha256",
        "foundation_manifest_path",
        "foundation_manifest_sha256",
        "foundation_weights_path",
        "foundation_weights_sha256",
    },
    "run": {"name", "profile", "seed", "device", "artifact_root"},
    "policy": {
        "role",
        "algorithm",
        "actions",
        "ticker_conditioning",
        "critic",
        "embedding_projection_dim",
    },
    "episode": {
        "ticker_mode",
        "ticker_sampling",
        "account_start",
        "timeout_trading_days",
        "randomize_starting_cushion",
    },
    "execution": {
        "adverse_slippage_ticks_per_side",
        "stress_adverse_slippage_ticks_per_side",
        "fee_schedule",
        "cost_booking",
    },
    "fees": {
        "snapshot",
        "es",
        "mes",
        "nq",
        "mnq",
        "rty",
        "m2k",
        "ym",
        "mym",
        "gc",
        "mgc",
        "cl",
        "mcl",
        "zb",
    },
    "exit": {
        "policy",
        "initial_stop_r",
        "activation_r",
        "giveback_r",
        "horizon_bars",
        "same_bar_policy",
    },
    "sizing": {
        "actor_controls_size",
        "episode_profiles",
        "profile_sampling",
        "mini_quantity",
        "micro_quantity",
        "mini_only_instruments",
    },
    "topstep": {
        "rule_snapshot",
        "daily_loss_limit_enabled",
        "daily_loss_limit_dollars",
        "daily_loss_limit_action",
        "daily_loss_limit_terminal",
    },
    "reward": {"kind", "gamma", "potential_shaping"},
    "constraint": {
        "kind",
        "cost_limit",
        "cost_gamma",
        "lambda_init",
        "lambda_lr",
        "lambda_max",
        "minimum_cushion_role",
    },
    "training": {
        "development_seeds",
        "confirmation_seeds",
        "serving_seed",
        "vector_environments",
        "ppo_epochs",
        "minibatch_size",
        "smoke_timesteps",
        "search_timesteps_per_seed",
        "search_seeds",
        "maximum_search_trials",
        "development_timesteps_per_seed",
        "confirmation_timesteps_per_seed",
        "maximum_timesteps_per_seed",
    },
    "evaluation": {
        "market_uncertainty",
        "market_block_length_weeks",
        "sealed_holdout_start",
        "minimum_raw_pass_rate",
        "minimum_seed_raw_pass_rate",
        "minimum_pass_rate_lcb_95",
        "maximum_observed_blows",
        "maximum_blow_rate_ucb_95",
        "minimum_chronological_attempts",
    },
}


def _digest(value: Any) -> str:
    encoded = json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid [rl.{name}] section")
    unknown = set(value) - _EXPECTED[name]
    missing = _EXPECTED[name] - set(value)
    if unknown:
        raise ConfigError(f"unknown [rl.{name}] keys: {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigError(f"missing [rl.{name}] keys: {', '.join(sorted(missing))}")
    return value


def _int(value: Any, field: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _float(value: Any, field: str, minimum: float = 0.0) -> float:
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


def _ints(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty integer array")
    return tuple(_int(item, field) for item in value)


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


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ConfigError(f"{field} must be a full SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ConfigError(f"{field} must be a full SHA-256 digest") from exc
    return value.lower()


def _run_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None
    ):
        raise ConfigError("rl.run.name must be a portable identifier")
    return value


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path string")
    return Path(value)


def load_rl_config(path: str | Path) -> RlConfig:
    """Load and validate the complete entry-only RL configuration."""
    config_path = Path(path)
    try:
        raw = tomllib.loads(config_path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc
    unknown = set(raw) - {"schema_version", "upstream", "rl"}
    if unknown:
        raise ConfigError(f"unknown sections: {', '.join(sorted(unknown))}")
    schema_version = raw.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ConfigError("schema_version must be integer 1")
    upstream = raw.get("upstream")
    if not isinstance(upstream, dict):
        raise ConfigError("missing or invalid [upstream] section")
    unknown_upstream = set(upstream) - _EXPECTED["upstream"]
    missing_upstream = _EXPECTED["upstream"] - set(upstream)
    if unknown_upstream:
        raise ConfigError(f"unknown [upstream] keys: {', '.join(sorted(unknown_upstream))}")
    if missing_upstream:
        raise ConfigError(f"missing [upstream] keys: {', '.join(sorted(missing_upstream))}")
    rl = raw.get("rl")
    if not isinstance(rl, dict):
        raise ConfigError("missing or invalid [rl] groups")
    expected_groups = set(_EXPECTED) - {"upstream"}
    unknown_groups = set(rl) - expected_groups
    missing_groups = expected_groups - set(rl)
    if unknown_groups:
        raise ConfigError(f"unknown [rl] groups: {', '.join(sorted(unknown_groups))}")
    if missing_groups:
        raise ConfigError(f"missing [rl] groups: {', '.join(sorted(missing_groups))}")

    sections = {name: _section(rl, name) for name in expected_groups}
    run = sections["run"]
    policy = sections["policy"]
    episode = sections["episode"]
    execution = sections["execution"]
    fees = sections["fees"]
    exit_config = sections["exit"]
    sizing = sections["sizing"]
    topstep = sections["topstep"]
    reward = sections["reward"]
    constraint = sections["constraint"]
    training = sections["training"]
    evaluation = sections["evaluation"]
    config = RlConfig(
        schema_version=1,
        upstream=RlUpstreamConfig(
            source_digest=_sha(upstream["source_digest"], "upstream.source_digest"),
            lock_digest=_sha(upstream["lock_digest"], "upstream.lock_digest"),
            rule_contract_path=_path(upstream["rule_contract_path"], "upstream.rule_contract_path"),
            rule_contract_sha256=_sha(
                upstream["rule_contract_sha256"], "upstream.rule_contract_sha256"
            ),
            downstream_config_path=_path(
                upstream["downstream_config_path"], "upstream.downstream_config_path"
            ),
            downstream_config_sha256=_sha(
                upstream["downstream_config_sha256"], "upstream.downstream_config_sha256"
            ),
            corpus_manifest_path=_path(
                upstream["corpus_manifest_path"], "upstream.corpus_manifest_path"
            ),
            corpus_manifest_sha256=_sha(
                upstream["corpus_manifest_sha256"], "upstream.corpus_manifest_sha256"
            ),
            embedding_manifest_path=_path(
                upstream["embedding_manifest_path"], "upstream.embedding_manifest_path"
            ),
            embedding_manifest_sha256=_sha(
                upstream["embedding_manifest_sha256"], "upstream.embedding_manifest_sha256"
            ),
            foundation_manifest_path=_path(
                upstream["foundation_manifest_path"], "upstream.foundation_manifest_path"
            ),
            foundation_manifest_sha256=_sha(
                upstream["foundation_manifest_sha256"], "upstream.foundation_manifest_sha256"
            ),
            foundation_weights_path=_path(
                upstream["foundation_weights_path"], "upstream.foundation_weights_path"
            ),
            foundation_weights_sha256=_sha(
                upstream["foundation_weights_sha256"], "upstream.foundation_weights_sha256"
            ),
        ),
        run=RlRunConfig(
            name=_run_name(run["name"]),
            profile=_choice(run["profile"], "rl.run.profile", {"smoke", "production"}),
            seed=_int(run["seed"], "rl.run.seed"),
            device=_choice(run["device"], "rl.run.device", {"cpu"}),
            artifact_root=_path(run["artifact_root"], "rl.run.artifact_root"),
        ),
        policy=RlPolicyConfig(
            role=_choice(policy["role"], "rl.policy.role", {"entry"}),
            algorithm=_choice(policy["algorithm"], "rl.policy.algorithm", {"maskable_ppo"}),
            actions=_strings(policy["actions"], "rl.policy.actions"),
            ticker_conditioning=_bool(
                policy["ticker_conditioning"], "rl.policy.ticker_conditioning"
            ),
            critic=_choice(policy["critic"], "rl.policy.critic", {"ticker_specific_heads"}),
            embedding_projection_dim=_int(
                policy["embedding_projection_dim"],
                "rl.policy.embedding_projection_dim",
                1,
            ),
        ),
        episode=RlEpisodeConfig(
            ticker_mode=_choice(
                episode["ticker_mode"], "rl.episode.ticker_mode", {"one_per_episode"}
            ),
            ticker_sampling=_choice(
                episode["ticker_sampling"], "rl.episode.ticker_sampling", {"uniform"}
            ),
            account_start=_float(episode["account_start"], "rl.episode.account_start", 1.0),
            timeout_trading_days=_int(
                episode["timeout_trading_days"], "rl.episode.timeout_trading_days", 1
            ),
            randomize_starting_cushion=_bool(
                episode["randomize_starting_cushion"],
                "rl.episode.randomize_starting_cushion",
            ),
        ),
        execution=RlExecutionConfig(
            adverse_slippage_ticks_per_side=_float(
                execution["adverse_slippage_ticks_per_side"],
                "rl.execution.adverse_slippage_ticks_per_side",
            ),
            stress_adverse_slippage_ticks_per_side=_float(
                execution["stress_adverse_slippage_ticks_per_side"],
                "rl.execution.stress_adverse_slippage_ticks_per_side",
            ),
            fee_schedule=_choice(
                execution["fee_schedule"],
                "rl.execution.fee_schedule",
                {"topstepx-2026-07-20"},
            ),
            cost_booking=_choice(
                execution["cost_booking"],
                "rl.execution.cost_booking",
                {"account_simulator_only"},
            ),
        ),
        fees=RlFeeConfig(
            snapshot=_choice(fees["snapshot"], "rl.fees.snapshot", {"topstepx-2026-07-20"}),
            es=_float(fees["es"], "rl.fees.es", 0.01),
            mes=_float(fees["mes"], "rl.fees.mes", 0.01),
            nq=_float(fees["nq"], "rl.fees.nq", 0.01),
            mnq=_float(fees["mnq"], "rl.fees.mnq", 0.01),
            rty=_float(fees["rty"], "rl.fees.rty", 0.01),
            m2k=_float(fees["m2k"], "rl.fees.m2k", 0.01),
            ym=_float(fees["ym"], "rl.fees.ym", 0.01),
            mym=_float(fees["mym"], "rl.fees.mym", 0.01),
            gc=_float(fees["gc"], "rl.fees.gc", 0.01),
            mgc=_float(fees["mgc"], "rl.fees.mgc", 0.01),
            cl=_float(fees["cl"], "rl.fees.cl", 0.01),
            mcl=_float(fees["mcl"], "rl.fees.mcl", 0.01),
            zb=_float(fees["zb"], "rl.fees.zb", 0.01),
        ),
        exit=RlExitConfig(
            policy=_choice(exit_config["policy"], "rl.exit.policy", {"two_r_trailing_v0"}),
            initial_stop_r=_float(exit_config["initial_stop_r"], "rl.exit.initial_stop_r"),
            activation_r=_float(exit_config["activation_r"], "rl.exit.activation_r"),
            giveback_r=_float(exit_config["giveback_r"], "rl.exit.giveback_r"),
            horizon_bars=_int(exit_config["horizon_bars"], "rl.exit.horizon_bars", 1),
            same_bar_policy=_choice(
                exit_config["same_bar_policy"],
                "rl.exit.same_bar_policy",
                {"prior_stop_first"},
            ),
        ),
        sizing=RlSizingConfig(
            actor_controls_size=_bool(
                sizing["actor_controls_size"], "rl.sizing.actor_controls_size"
            ),
            episode_profiles=_strings(sizing["episode_profiles"], "rl.sizing.episode_profiles"),
            profile_sampling=_choice(
                sizing["profile_sampling"],
                "rl.sizing.profile_sampling",
                {"uniform_when_supported"},
            ),
            mini_quantity=_int(sizing["mini_quantity"], "rl.sizing.mini_quantity", 1),
            micro_quantity=_int(sizing["micro_quantity"], "rl.sizing.micro_quantity", 1),
            mini_only_instruments=_strings(
                sizing["mini_only_instruments"], "rl.sizing.mini_only_instruments"
            ),
        ),
        topstep=RlTopstepConfig(
            rule_snapshot=_choice(
                topstep["rule_snapshot"],
                "rl.topstep.rule_snapshot",
                {"topstep-100k-2026-07-20"},
            ),
            daily_loss_limit_enabled=_bool(
                topstep["daily_loss_limit_enabled"],
                "rl.topstep.daily_loss_limit_enabled",
            ),
            daily_loss_limit_dollars=_float(
                topstep["daily_loss_limit_dollars"],
                "rl.topstep.daily_loss_limit_dollars",
            ),
            daily_loss_limit_action=_choice(
                topstep["daily_loss_limit_action"],
                "rl.topstep.daily_loss_limit_action",
                {"flatten_cancel_lockout"},
            ),
            daily_loss_limit_terminal=_bool(
                topstep["daily_loss_limit_terminal"],
                "rl.topstep.daily_loss_limit_terminal",
            ),
        ),
        reward=RlRewardConfig(
            kind=_choice(
                reward["kind"],
                "rl.reward.kind",
                {"terminal_pass_with_blow_cost"},
            ),
            gamma=_float(reward["gamma"], "rl.reward.gamma"),
            potential_shaping=_bool(reward["potential_shaping"], "rl.reward.potential_shaping"),
        ),
        constraint=RlConstraintConfig(
            kind=_choice(
                constraint["kind"],
                "rl.constraint.kind",
                {"episodic_blow_lagrangian"},
            ),
            cost_limit=_float(constraint["cost_limit"], "rl.constraint.cost_limit"),
            cost_gamma=_float(constraint["cost_gamma"], "rl.constraint.cost_gamma"),
            lambda_init=_float(constraint["lambda_init"], "rl.constraint.lambda_init"),
            lambda_lr=_float(constraint["lambda_lr"], "rl.constraint.lambda_lr"),
            lambda_max=_float(constraint["lambda_max"], "rl.constraint.lambda_max"),
            minimum_cushion_role=_choice(
                constraint["minimum_cushion_role"],
                "rl.constraint.minimum_cushion_role",
                {"observation_metric_only"},
            ),
        ),
        training=RlTrainingConfig(
            development_seeds=_ints(training["development_seeds"], "rl.training.development_seeds"),
            confirmation_seeds=_ints(
                training["confirmation_seeds"], "rl.training.confirmation_seeds"
            ),
            serving_seed=_int(training["serving_seed"], "rl.training.serving_seed"),
            vector_environments=_int(
                training["vector_environments"], "rl.training.vector_environments", 1
            ),
            ppo_epochs=_int(training["ppo_epochs"], "rl.training.ppo_epochs", 2),
            minibatch_size=_int(training["minibatch_size"], "rl.training.minibatch_size", 1),
            smoke_timesteps=_int(training["smoke_timesteps"], "rl.training.smoke_timesteps", 1),
            search_timesteps_per_seed=_int(
                training["search_timesteps_per_seed"],
                "rl.training.search_timesteps_per_seed",
                1,
            ),
            search_seeds=_int(training["search_seeds"], "rl.training.search_seeds", 1),
            maximum_search_trials=_int(
                training["maximum_search_trials"],
                "rl.training.maximum_search_trials",
                1,
            ),
            development_timesteps_per_seed=_int(
                training["development_timesteps_per_seed"],
                "rl.training.development_timesteps_per_seed",
                1,
            ),
            confirmation_timesteps_per_seed=_int(
                training["confirmation_timesteps_per_seed"],
                "rl.training.confirmation_timesteps_per_seed",
                1,
            ),
            maximum_timesteps_per_seed=_int(
                training["maximum_timesteps_per_seed"],
                "rl.training.maximum_timesteps_per_seed",
                1,
            ),
        ),
        evaluation=RlEvaluationConfig(
            market_uncertainty=_choice(
                evaluation["market_uncertainty"],
                "rl.evaluation.market_uncertainty",
                {"synchronized_adjacent_week_moving_block_bootstrap"},
            ),
            market_block_length_weeks=_int(
                evaluation["market_block_length_weeks"],
                "rl.evaluation.market_block_length_weeks",
                1,
            ),
            sealed_holdout_start=_timestamp(
                evaluation["sealed_holdout_start"],
                "rl.evaluation.sealed_holdout_start",
            ),
            minimum_raw_pass_rate=_float(
                evaluation["minimum_raw_pass_rate"],
                "rl.evaluation.minimum_raw_pass_rate",
            ),
            minimum_seed_raw_pass_rate=_float(
                evaluation["minimum_seed_raw_pass_rate"],
                "rl.evaluation.minimum_seed_raw_pass_rate",
            ),
            minimum_pass_rate_lcb_95=_float(
                evaluation["minimum_pass_rate_lcb_95"],
                "rl.evaluation.minimum_pass_rate_lcb_95",
            ),
            maximum_observed_blows=_int(
                evaluation["maximum_observed_blows"],
                "rl.evaluation.maximum_observed_blows",
            ),
            maximum_blow_rate_ucb_95=_float(
                evaluation["maximum_blow_rate_ucb_95"],
                "rl.evaluation.maximum_blow_rate_ucb_95",
            ),
            minimum_chronological_attempts=_int(
                evaluation["minimum_chronological_attempts"],
                "rl.evaluation.minimum_chronological_attempts",
                1,
            ),
        ),
    )
    _validate(config)
    return config


def _validate(config: RlConfig) -> None:
    locked: tuple[tuple[bool, str], ...] = (
        (config.policy.actions == ("skip", "enter"), "rl.policy.actions must be skip, enter"),
        (config.policy.ticker_conditioning, "rl.policy.ticker_conditioning must be true"),
        (
            config.policy.embedding_projection_dim == 256,
            "rl.policy.embedding_projection_dim must be 256",
        ),
        (config.episode.account_start == 100000.0, "rl.episode.account_start must be 100000"),
        (
            config.episode.timeout_trading_days == 20,
            "rl.episode.timeout_trading_days must be 20",
        ),
        (
            not config.episode.randomize_starting_cushion,
            "rl.episode.randomize_starting_cushion must be false",
        ),
        (
            config.execution.adverse_slippage_ticks_per_side == 1.0,
            "rl.execution.adverse_slippage_ticks_per_side must be 1",
        ),
        (
            config.execution.stress_adverse_slippage_ticks_per_side == 2.0,
            "rl.execution.stress_adverse_slippage_ticks_per_side must be 2",
        ),
        (
            asdict(config.fees)
            == {
                "snapshot": "topstepx-2026-07-20",
                "es": 3.78,
                "mes": 1.22,
                "nq": 3.78,
                "mnq": 1.22,
                "rty": 3.78,
                "m2k": 1.22,
                "ym": 3.78,
                "mym": 1.22,
                "gc": 4.32,
                "mgc": 1.92,
                "cl": 4.02,
                "mcl": 1.52,
                "zb": 2.76,
            },
            "rl.fees must match the pinned TopstepX 2026-07-20 round-turn snapshot",
        ),
        (config.exit.initial_stop_r == 1.0, "rl.exit.initial_stop_r must be 1"),
        (config.exit.activation_r == 2.0, "rl.exit.activation_r must be 2"),
        (config.exit.giveback_r == 0.75, "rl.exit.giveback_r must be 0.75"),
        (config.exit.horizon_bars == 120, "rl.exit.horizon_bars must be 120"),
        (
            not config.sizing.actor_controls_size,
            "rl.sizing.actor_controls_size must be false",
        ),
        (
            config.sizing.episode_profiles == ("one_mini", "ten_micros"),
            "rl.sizing.episode_profiles must be one_mini, ten_micros",
        ),
        (config.sizing.mini_quantity == 1, "rl.sizing.mini_quantity must be 1"),
        (config.sizing.micro_quantity == 10, "rl.sizing.micro_quantity must be 10"),
        (
            config.sizing.mini_only_instruments == ("ZB",),
            "rl.sizing.mini_only_instruments must contain only ZB",
        ),
        (
            config.topstep.daily_loss_limit_enabled,
            "rl.topstep.daily_loss_limit_enabled must be true",
        ),
        (
            config.topstep.daily_loss_limit_dollars == 2000.0,
            "rl.topstep.daily_loss_limit_dollars must be 2000",
        ),
        (
            not config.topstep.daily_loss_limit_terminal,
            "rl.topstep.daily_loss_limit_terminal must be false",
        ),
        (config.reward.gamma == 1.0, "rl.reward.gamma must be 1"),
        (
            not config.reward.potential_shaping,
            "rl.reward.potential_shaping must be false",
        ),
        (config.constraint.cost_limit == 0.01, "rl.constraint.cost_limit must be 0.01"),
        (config.constraint.cost_gamma == 1.0, "rl.constraint.cost_gamma must be 1"),
        (config.constraint.lambda_init == 1.0, "rl.constraint.lambda_init must be 1"),
        (config.constraint.lambda_lr == 0.01, "rl.constraint.lambda_lr must be 0.01"),
        (config.constraint.lambda_max == 100.0, "rl.constraint.lambda_max must be 100"),
        (
            config.evaluation.maximum_observed_blows == 0,
            "rl.evaluation.maximum_observed_blows must be 0",
        ),
    )
    for condition, message in locked:
        if not condition:
            raise ConfigError(message)
    if config.run.seed not in config.training.development_seeds:
        raise ConfigError("rl.run.seed must appear in rl.training.development_seeds")
    if config.training.serving_seed not in config.training.development_seeds:
        raise ConfigError("rl.training.serving_seed must appear in development_seeds")
    if config.training.serving_seed not in config.training.confirmation_seeds:
        raise ConfigError("rl.training.serving_seed must appear in confirmation_seeds")
    if len(set(config.training.development_seeds)) != len(config.training.development_seeds):
        raise ConfigError("rl.training.development_seeds must be unique")
    if len(set(config.training.confirmation_seeds)) != len(config.training.confirmation_seeds):
        raise ConfigError("rl.training.confirmation_seeds must be unique")
    if config.training.maximum_search_trials > 30:
        raise ConfigError("rl.training.maximum_search_trials must be <= 30")
    if not (
        config.training.search_timesteps_per_seed
        <= config.training.development_timesteps_per_seed
        <= config.training.confirmation_timesteps_per_seed
        <= config.training.maximum_timesteps_per_seed
    ):
        raise ConfigError("rl.training timestep budgets must be nondecreasing")
    for field, value in (
        ("minimum_raw_pass_rate", config.evaluation.minimum_raw_pass_rate),
        ("minimum_seed_raw_pass_rate", config.evaluation.minimum_seed_raw_pass_rate),
        ("minimum_pass_rate_lcb_95", config.evaluation.minimum_pass_rate_lcb_95),
        ("maximum_blow_rate_ucb_95", config.evaluation.maximum_blow_rate_ucb_95),
    ):
        if value > 1:
            raise ConfigError(f"rl.evaluation.{field} must be <= 1")
    accepted_gates = {
        "market_uncertainty": "synchronized_adjacent_week_moving_block_bootstrap",
        "market_block_length_weeks": 2,
        "sealed_holdout_start": datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        "minimum_raw_pass_rate": 0.60,
        "minimum_seed_raw_pass_rate": 0.50,
        "minimum_pass_rate_lcb_95": 0.50,
        "maximum_observed_blows": 0,
        "maximum_blow_rate_ucb_95": 0.01,
        "minimum_chronological_attempts": 300,
    }
    if asdict(config.evaluation) != accepted_gates:
        raise ConfigError("rl.evaluation promotion gates must match the accepted contract")
    accepted_production_training: dict[str, object] = {
        "development_seeds": (42, 43, 44, 45, 46),
        "confirmation_seeds": (42, 43, 44, 45, 46, 47, 48, 49, 50, 51),
        "serving_seed": 42,
        "vector_environments": 7,
        "ppo_epochs": 4,
        "minibatch_size": 512,
        "smoke_timesteps": 50_000,
        "search_timesteps_per_seed": 500_000,
        "search_seeds": 3,
        "maximum_search_trials": 30,
        "development_timesteps_per_seed": 2_000_000,
        "confirmation_timesteps_per_seed": 5_000_000,
        "maximum_timesteps_per_seed": 10_000_000,
    }
    if (
        config.run.profile == "production"
        and asdict(config.training) != accepted_production_training
    ):
        raise ConfigError(
            "rl.training production training settings must match the accepted contract"
        )
