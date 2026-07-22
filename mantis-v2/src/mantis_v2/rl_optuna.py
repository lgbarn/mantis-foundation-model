"""Persistent validation-owned Optuna search for the entry policy."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import statistics
import tempfile
import tomllib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
import optuna
import torch
from optuna.trial import TrialState

from mantis_v2.rl_config import RlConfig
from mantis_v2.rl_environment import TopstepEntryEnvironment
from mantis_v2.rl_policy import PROFILES, TICKERS, EntryActorCritic, PolicyVariant
from mantis_v2.rl_training import (
    PpoHyperparameters,
    ProductionTrainingError,
    load_training_episodes,
    train_entry_policy,
)
from mantis_v2.rl_validation import load_episode_manifest


@dataclass(frozen=True)
class SearchDistribution:
    """One immutable accepted search dimension."""

    kind: str
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[float | int, ...] = ()
    log: bool = False
    unit: str | None = None


SEARCH_SPACE = MappingProxyType(
    {
        "learning_rate": SearchDistribution("float", 1e-4, 1e-3, log=True),
        "rollout_length": SearchDistribution(
            "categorical", choices=(14, 28, 56), unit="complete_episodes"
        ),
        "batch_size": SearchDistribution("categorical", choices=(256, 512, 1024)),
        "gae_lambda": SearchDistribution("float", 0.90, 0.99),
        "clip_range": SearchDistribution("categorical", choices=(0.1, 0.2, 0.3)),
        "entropy_coefficient": SearchDistribution("float", 1e-4, 0.03, log=True),
        "value_loss_coefficient": SearchDistribution("categorical", choices=(0.25, 0.5, 1.0)),
        "max_grad_norm": SearchDistribution("categorical", choices=(0.3, 0.5, 1.0)),
        "hidden_width": SearchDistribution("categorical", choices=(64, 128, 256)),
    }
)

_TRAINING_SEED_NAMESPACE = "mantis-v2-topstep-100k-optuna-v1"
_PROPOSAL_SEED_NAMESPACE = "mantis-v2-topstep-100k-optuna-proposal-v1"
_SEED_METHOD = "sha256-u31-v1"
_OPTUNA_VERSION = "4.9.0"
_SAMPLER_SETTINGS = MappingProxyType(
    {
        "n_startup_trials": 10,
        "n_ei_candidates": 24,
        "multivariate": False,
        "group": False,
        "constant_liar": False,
    }
)
_PRUNER_SETTINGS = MappingProxyType(
    {
        "n_startup_trials": 10,
        "n_warmup_steps": 0,
        "interval_steps": 1,
        "n_min_trials": 5,
        "minimum_completed_validation_seeds_before_check": 2,
    }
)
_TUNING_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "rl-optuna-v1.toml"


@dataclass(frozen=True)
class TrialIdentity:
    """Deterministic compute identity assigned before a trial is attempted."""

    study_name: str
    trial_number: int
    proposal_seed: int
    seeds: tuple[int, ...]
    timesteps_per_seed: int

    @property
    def total_timesteps(self) -> int:
        return len(self.seeds) * self.timesteps_per_seed


class OptunaSearchError(RuntimeError):
    """Raised when search evidence cannot prove the accepted contract."""


@dataclass(frozen=True)
class SeedValidationOutcome:
    """Validation-only account outcomes for one trial seed."""

    partition: str
    seed: int
    attempts: int
    passes: int
    blows: int
    median_pass_days: float | None
    completed_timesteps: int | None = None
    timestep_overshoot: int | None = None
    checkpoint_sha256: str | None = None
    training_manifest_sha256: str | None = None
    validation_manifest_sha256: str | None = None
    validation_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.partition != "validation":
            raise OptunaSearchError("Optuna evidence must be owned by validation")
        if (
            self.attempts < 1
            or self.passes < 0
            or self.blows < 0
            or self.passes + self.blows > self.attempts
            or (self.passes == 0 and self.median_pass_days is not None)
            or (
                self.passes > 0
                and (
                    self.median_pass_days is None
                    or not math.isfinite(self.median_pass_days)
                    or self.median_pass_days < 0.0
                )
            )
        ):
            raise OptunaSearchError("validation outcome counts are invalid")

    def require_production_evidence(self) -> None:
        digests = (
            self.checkpoint_sha256,
            self.training_manifest_sha256,
            self.validation_manifest_sha256,
            self.validation_artifact_sha256,
        )
        if (
            self.completed_timesteps is None
            or self.completed_timesteps < 500_000
            or self.timestep_overshoot is None
            or self.timestep_overshoot < 0
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in digests
            )
        ):
            raise OptunaSearchError("trial seed lacks checkpoint-bound production evidence")


class PrunedTrial(Exception):
    """Validation-owned decision to stop a trial while preserving partial evidence."""

    def __init__(self, reason: str, partial_validation: tuple[SeedValidationOutcome, ...]) -> None:
        if not reason or not 2 <= len(partial_validation) <= 3:
            raise OptunaSearchError("pruned trial evidence is incomplete")
        if len({outcome.seed for outcome in partial_validation}) != len(partial_validation):
            raise OptunaSearchError("pruned trial evidence repeats a seed")
        super().__init__(reason)
        self.reason = reason
        self.partial_validation = partial_validation


@dataclass(frozen=True)
class TrialEvaluation:
    """Complete three-seed validation evidence for one attempted identity."""

    trial_number: int
    outcomes: tuple[SeedValidationOutcome, ...]

    def __post_init__(self) -> None:
        if self.trial_number < 0 or len(self.outcomes) != 3:
            raise OptunaSearchError("trial evaluation requires exactly three seeds")
        if len({outcome.seed for outcome in self.outcomes}) != 3:
            raise OptunaSearchError("trial evaluation seeds must be unique")

    @property
    def aggregate_attempts(self) -> int:
        return sum(outcome.attempts for outcome in self.outcomes)

    @property
    def aggregate_passes(self) -> int:
        return sum(outcome.passes for outcome in self.outcomes)

    @property
    def aggregate_blows(self) -> int:
        return sum(outcome.blows for outcome in self.outcomes)

    @property
    def feasible(self) -> bool:
        return self.aggregate_blows == 0

    @property
    def pass_rate_lcb_95(self) -> float:
        return _pass_rate_lcb_95(self.aggregate_passes, self.aggregate_attempts)

    @property
    def median_pass_days(self) -> float:
        days = [
            outcome.median_pass_days
            for outcome in self.outcomes
            if outcome.median_pass_days is not None
        ]
        return float(statistics.median(days)) if days else math.inf


def select_winner(evaluations: tuple[TrialEvaluation, ...]) -> TrialEvaluation:
    """Select deterministically from feasible validation evidence only."""
    feasible = [evaluation for evaluation in evaluations if evaluation.feasible]
    if not feasible:
        raise OptunaSearchError("study has no feasible validation trial")
    return min(
        feasible,
        key=lambda item: (-item.pass_rate_lcb_95, item.median_pass_days, item.trial_number),
    )


def _pass_rate_lcb_95(successes: int, attempts: int) -> float:
    z = 1.6448536269514722
    rate = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = (rate + z * z / (2.0 * attempts)) / denominator
    radius = (
        z * math.sqrt(rate * (1.0 - rate) / attempts + z * z / (4.0 * attempts**2)) / denominator
    )
    return max(0.0, center - radius)


@dataclass(frozen=True)
class StudyManifests:
    """Verified training/validation inputs without any test or holdout path."""

    training: Path
    validation: Path
    fold: int
    training_sha256: str
    validation_sha256: str


@dataclass(frozen=True)
class TrialRequest:
    """All and only the inputs reachable by one search evaluation."""

    identity: TrialIdentity
    parameters: Mapping[str, float | int]
    variant: PolicyVariant
    training_manifest: Path
    validation_manifest: Path
    fold: int
    artifact_directory: Path
    completed_validation: tuple[SeedValidationOutcome, ...]
    report_validation: Callable[[SeedValidationOutcome], None]


def _manifest_identity(
    path: Path, config: RlConfig, required_partition: str
) -> tuple[int, datetime, datetime, str]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OptunaSearchError("study episode manifest is invalid") from exc
    if not isinstance(raw, dict):
        raise OptunaSearchError("study episode manifest is invalid")
    partition = raw.get("partition")
    identities = raw.get("identities")
    configured = identities.get("config") if isinstance(identities, dict) else None
    if (
        raw.get("schema_version") != 1
        or raw.get("stage") != "rl-episode-schedule"
        or not isinstance(partition, dict)
        or partition.get("name") != required_partition
    ):
        raise OptunaSearchError(f"study requires a declared {required_partition} partition")
    if not isinstance(configured, dict) or configured.get("sha256") != config.digest:
        raise OptunaSearchError("study manifest config identity mismatch")
    fold = raw.get("fold")
    if type(fold) is not int or fold < 0:
        raise OptunaSearchError("study manifest fold is invalid")
    try:
        start = datetime.fromisoformat(cast(str, partition["start"]))
        end = datetime.fromisoformat(cast(str, partition["end"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise OptunaSearchError("study manifest partition end is invalid") from exc
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise OptunaSearchError("study manifest partition bounds are invalid")
    return fold, start, end, hashlib.sha256(path.read_bytes()).hexdigest()


def validate_study_manifests(config: RlConfig, training: Path, validation: Path) -> StudyManifests:
    """Fail before study creation unless only same-fold train/validation are reachable."""
    training_fold, _training_start, training_end, training_sha256 = _manifest_identity(
        training, config, "training"
    )
    validation_fold, validation_start, validation_end, validation_sha256 = _manifest_identity(
        validation, config, "validation"
    )
    if training_fold != validation_fold:
        raise OptunaSearchError("study manifests must own the same fold")
    holdout_start = config.evaluation.sealed_holdout_start
    if training_end >= holdout_start or validation_end >= holdout_start:
        raise OptunaSearchError("study manifests reach the sealed holdout")
    if training_end >= validation_start:
        raise OptunaSearchError("study training and validation partitions overlap")
    return StudyManifests(
        training.resolve(),
        validation.resolve(),
        training_fold,
        training_sha256,
        validation_sha256,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _search_space_payload() -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads(
            json.dumps({name: asdict(distribution) for name, distribution in SEARCH_SPACE.items()})
        ),
    )


def _study_contract(
    config: RlConfig,
    manifests: StudyManifests,
    study_name: str,
    variant: PolicyVariant,
    require_production_evidence: bool,
) -> dict[str, object]:
    try:
        tuning_bytes = _TUNING_CONFIG_PATH.read_bytes()
        tuning = tomllib.loads(tuning_bytes.decode())["rl"]["tuning"]
    except (OSError, KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise OptunaSearchError("immutable Optuna v1 TOML is invalid") from exc
    expected_search_space = {}
    for name, distribution in SEARCH_SPACE.items():
        item: dict[str, object] = {"distribution": distribution.kind}
        if distribution.low is not None:
            item["low"] = distribution.low
        if distribution.high is not None:
            item["high"] = distribution.high
        if distribution.choices:
            item["choices"] = list(distribution.choices)
        if distribution.kind == "float":
            item["log"] = distribution.log
        if distribution.unit is not None:
            item["unit"] = distribution.unit
        expected_search_space[name] = item
    if (
        tuning.get("schema_version") != 1
        or tuning.get("direction") != "maximize"
        or tuning.get("maximum_trials") != 30
        or tuning.get("timesteps_per_seed") != 500_000
        or tuning.get("seeds_per_trial") != 3
        or tuning.get("n_jobs") != 1
        or tuning.get("sampler")
        != {
            "name": "TPESampler",
            "optuna_version": _OPTUNA_VERSION,
            **dict(_SAMPLER_SETTINGS),
            "proposal_seed_namespace": _PROPOSAL_SEED_NAMESPACE,
            "proposal_seed_method": _SEED_METHOD,
        }
        or tuning.get("pruner")
        != {
            "name": "MedianPruner",
            **dict(_PRUNER_SETTINGS),
        }
        or tuning.get("training_seeds")
        != {
            "namespace": _TRAINING_SEED_NAMESPACE,
            "method": _SEED_METHOD,
            "indices": [0, 1, 2],
        }
        or tuning.get("search_space") != expected_search_space
    ):
        raise OptunaSearchError("immutable Optuna v1 TOML differs from the accepted contract")
    payload = {
        "schema_version": 1,
        "study_name": study_name,
        "variant": variant.value,
        "config_sha256": config.digest,
        "training_manifest_sha256": manifests.training_sha256,
        "validation_manifest_sha256": manifests.validation_sha256,
        "fold": manifests.fold,
        "search_space": _search_space_payload(),
        "search_space_sha256": _canonical_sha256(_search_space_payload()),
        "tuning_config_sha256": hashlib.sha256(tuning_bytes).hexdigest(),
        "suggestion_order": list(SEARCH_SPACE),
        "sampler": "TPESampler",
        "sampler_settings": dict(_SAMPLER_SETTINGS),
        "pruner": "MedianPruner",
        "pruner_settings": dict(_PRUNER_SETTINGS),
        "optuna_version": _OPTUNA_VERSION,
        "training_seed_namespace": _TRAINING_SEED_NAMESPACE,
        "proposal_seed_namespace": _PROPOSAL_SEED_NAMESPACE,
        "seed_method": _SEED_METHOD,
        "maximum_trials": config.training.maximum_search_trials,
        "search_seeds": config.training.search_seeds,
        "timesteps_per_seed": config.training.search_timesteps_per_seed,
        "direction": "maximize",
        "production_evaluator": require_production_evidence,
        "sealed_holdout_accessed": False,
    }
    return {**payload, "identity_sha256": _canonical_sha256(payload)}


def _atomic_json_no_overwrite(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(dict(payload), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise OptunaSearchError(f"immutable artifact already exists: {path}") from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_idempotent(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        if _load_trial_ledger(path) != dict(payload):
            raise OptunaSearchError(f"immutable artifact mismatch: {path}")
        return
    _atomic_json_no_overwrite(path, payload)


def _storage_url(database: Path) -> str:
    return f"sqlite:///{database.resolve()}"


def _suggest_parameters(trial: optuna.Trial) -> dict[str, float | int]:
    parameters: dict[str, float | int] = {}
    for name, distribution in SEARCH_SPACE.items():
        if distribution.kind == "categorical":
            parameters[name] = cast(
                float | int, trial.suggest_categorical(name, distribution.choices)
            )
        elif distribution.kind == "float":
            parameters[name] = trial.suggest_float(
                name,
                cast(float, distribution.low),
                cast(float, distribution.high),
                log=distribution.log,
            )
        else:
            raise OptunaSearchError(f"unsupported search distribution: {name}")
    return parameters


def _constraints(frozen_trial: optuna.trial.FrozenTrial) -> tuple[float]:
    blows = frozen_trial.user_attrs.get("validation_blow_count")
    return (float(blows),) if isinstance(blows, int | float) else (0.0,)


def _sampler(seed: int) -> optuna.samplers.TPESampler:
    if optuna.__version__ != _OPTUNA_VERSION:
        raise OptunaSearchError(
            f"Optuna version mismatch: expected {_OPTUNA_VERSION}, got {optuna.__version__}"
        )
    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=_SAMPLER_SETTINGS["n_startup_trials"],
        n_ei_candidates=_SAMPLER_SETTINGS["n_ei_candidates"],
        multivariate=False,
        group=False,
        constant_liar=False,
        constraints_func=_constraints,
    )


def _pruner() -> optuna.pruners.MedianPruner:
    return optuna.pruners.MedianPruner(
        n_startup_trials=_PRUNER_SETTINGS["n_startup_trials"],
        n_warmup_steps=_PRUNER_SETTINGS["n_warmup_steps"],
        interval_steps=_PRUNER_SETTINGS["interval_steps"],
        n_min_trials=_PRUNER_SETTINGS["n_min_trials"],
    )


def _evaluation_payload(evaluation: TrialEvaluation) -> dict[str, object]:
    return {
        "trial_number": evaluation.trial_number,
        "outcomes": [asdict(outcome) for outcome in evaluation.outcomes],
        "feasible": evaluation.feasible,
        "aggregate_attempts": evaluation.aggregate_attempts,
        "aggregate_passes": evaluation.aggregate_passes,
        "aggregate_blows": evaluation.aggregate_blows,
        "pass_rate_lcb_95": evaluation.pass_rate_lcb_95,
        "median_pass_days": (
            None if math.isinf(evaluation.median_pass_days) else evaluation.median_pass_days
        ),
    }


def _seed_validation_outcome(payload: Mapping[str, object]) -> SeedValidationOutcome:
    def required_int(name: str) -> int:
        value = payload.get(name)
        if type(value) is not int:
            raise OptunaSearchError(f"validation outcome {name} is invalid")
        return value

    def optional_int(name: str) -> int | None:
        value = payload.get(name)
        if value is None:
            return None
        if type(value) is not int:
            raise OptunaSearchError(f"validation outcome {name} is invalid")
        return value

    def optional_string(name: str) -> str | None:
        value = payload.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise OptunaSearchError(f"validation outcome {name} is invalid")
        return value

    partition = payload.get("partition")
    median_pass_days = payload.get("median_pass_days")
    if not isinstance(partition, str) or (
        median_pass_days is not None
        and (isinstance(median_pass_days, bool) or not isinstance(median_pass_days, int | float))
    ):
        raise OptunaSearchError("validation outcome scalar fields are invalid")
    return SeedValidationOutcome(
        partition=partition,
        seed=required_int("seed"),
        attempts=required_int("attempts"),
        passes=required_int("passes"),
        blows=required_int("blows"),
        median_pass_days=(None if median_pass_days is None else float(median_pass_days)),
        completed_timesteps=optional_int("completed_timesteps"),
        timestep_overshoot=optional_int("timestep_overshoot"),
        checkpoint_sha256=optional_string("checkpoint_sha256"),
        training_manifest_sha256=optional_string("training_manifest_sha256"),
        validation_manifest_sha256=optional_string("validation_manifest_sha256"),
        validation_artifact_sha256=optional_string("validation_artifact_sha256"),
    )


def _load_evaluation(payload: Mapping[str, object]) -> TrialEvaluation:
    try:
        raw_outcomes = payload["outcomes"]
        if not isinstance(raw_outcomes, list) or not all(
            isinstance(item, dict) for item in raw_outcomes
        ):
            raise OptunaSearchError("trial ledger outcomes are invalid")
        outcomes = tuple(_seed_validation_outcome(item) for item in raw_outcomes)
        return TrialEvaluation(int(cast(int, payload["trial_number"])), outcomes)
    except (KeyError, TypeError, ValueError, OptunaSearchError) as exc:
        raise OptunaSearchError("trial ledger evaluation is invalid") from exc


def _load_trial_ledger(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OptunaSearchError("trial ledger is invalid") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise OptunaSearchError("trial ledger is invalid")
    return cast(dict[str, object], raw)


def _tell_from_ledger(
    study: optuna.Study, trial: optuna.Trial, ledger: Mapping[str, object]
) -> None:
    if ledger.get("state") == "complete":
        evaluation_raw = ledger.get("evaluation")
        if not isinstance(evaluation_raw, dict):
            raise OptunaSearchError("complete trial ledger lacks validation evidence")
        evaluation = _load_evaluation(evaluation_raw)
        value = evaluation.pass_rate_lcb_95 if evaluation.feasible else -1.0
        median_days = evaluation.median_pass_days
        trial.set_user_attr("validation_blow_count", evaluation.aggregate_blows)
        trial.set_user_attr(
            "validation_median_pass_days", None if math.isinf(median_days) else median_days
        )
        study.tell(trial, values=value)
    elif ledger.get("state") == "failed":
        study.tell(trial, state=TrialState.FAIL)
    elif ledger.get("state") == "pruned":
        study.tell(trial, state=TrialState.PRUNED)
    else:
        raise OptunaSearchError("running trial ledger state is invalid")


def _completed_evaluations(
    output: Path,
    study: optuna.Study,
    contract: Mapping[str, object],
    config: RlConfig,
    study_name: str,
) -> tuple[TrialEvaluation, ...]:
    evaluations = []
    for frozen in study.trials:
        trial_number = frozen.number
        ledger = _load_trial_ledger(output / "ledger" / f"trial-{trial_number:04d}.json")
        expected_identity = asdict(derive_trial_identity(config, study_name, trial_number))
        expected_state = {
            TrialState.COMPLETE: "complete",
            TrialState.PRUNED: "pruned",
            TrialState.FAIL: "failed",
        }.get(frozen.state)
        if (
            expected_state is None
            or ledger.get("study_identity_sha256") != contract["identity_sha256"]
            or ledger.get("trial_number") != trial_number
            or ledger.get("identity")
            != cast(dict[str, object], json.loads(json.dumps(expected_identity)))
            or ledger.get("parameters") != frozen.params
            or ledger.get("parameters_sha256") != _canonical_sha256(frozen.params)
            or ledger.get("state") != expected_state
        ):
            raise OptunaSearchError("trial ledger does not match persistent Optuna state")
        if ledger.get("state") != "complete":
            continue
        evaluation = ledger.get("evaluation")
        if not isinstance(evaluation, dict):
            raise OptunaSearchError("complete trial ledger lacks validation evidence")
        loaded_evaluation = _load_evaluation(evaluation)
        expected_seeds = derive_trial_identity(config, study_name, trial_number).seeds
        expected_value = loaded_evaluation.pass_rate_lcb_95 if loaded_evaluation.feasible else -1.0
        median_days = loaded_evaluation.median_pass_days
        expected_median_days = None if math.isinf(median_days) else median_days
        if (
            loaded_evaluation.trial_number != trial_number
            or tuple(outcome.seed for outcome in loaded_evaluation.outcomes) != expected_seeds
            or frozen.value != expected_value
            or frozen.user_attrs.get("validation_blow_count") != loaded_evaluation.aggregate_blows
            or frozen.user_attrs.get("validation_median_pass_days") != expected_median_days
        ):
            raise OptunaSearchError("trial evaluation does not match persistent Optuna state")
        evaluations.append(loaded_evaluation)
    return tuple(evaluations)


def _completed_search_seed(
    path: Path, request: TrialRequest, seed_index: int
) -> dict[str, object] | None:
    manifest_path = path / "manifest.json"
    metrics_path = path / "metrics.json"
    if not manifest_path.exists() and not metrics_path.exists():
        return None
    if metrics_path.exists() and not manifest_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise OptunaSearchError("recoverable search seed metrics are invalid") from exc
        identities = metrics.get("identities") if isinstance(metrics, dict) else None
        if (
            not isinstance(metrics, dict)
            or metrics.get("status") != "complete"
            or not isinstance(identities, dict)
            or identities.get("seed") != request.identity.seeds[seed_index]
            or identities.get("search_trial_number") != request.identity.trial_number
            or identities.get("search_seed_index") != seed_index
            or identities.get("ppo_hyperparameters") != dict(request.parameters)
        ):
            raise OptunaSearchError("recoverable search seed provenance mismatch")
        return None
    if manifest_path.exists() and not metrics_path.exists():
        raise OptunaSearchError("completed search seed manifest lacks metrics")
    try:
        manifest = json.loads(manifest_path.read_text())
        metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OptunaSearchError("completed search seed artifact is invalid") from exc
    identities = metrics.get("identities") if isinstance(metrics, dict) else None
    if (
        not isinstance(manifest, dict)
        or not isinstance(metrics, dict)
        or metrics.get("status") != "complete"
        or not isinstance(identities, dict)
        or identities.get("seed") != request.identity.seeds[seed_index]
        or identities.get("search_trial_number") != request.identity.trial_number
        or identities.get("search_seed_index") != seed_index
        or identities.get("ppo_hyperparameters") != dict(request.parameters)
        or manifest.get("metrics_sha256") != hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    ):
        raise OptunaSearchError("completed search seed provenance mismatch")
    return cast(dict[str, object], metrics)


def _load_search_model(
    request: TrialRequest,
    run_output: Path,
    seed_result: Mapping[str, object],
    observation_width: int,
) -> EntryActorCritic:
    try:
        state = json.loads((run_output / "state.json").read_text())
        checkpoint = run_output / cast(str, state["checkpoint"]) / "checkpoint.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("identities") != seed_result.get(
            "identities"
        ):
            raise OptunaSearchError("search checkpoint provenance mismatch")
        model = EntryActorCritic(
            observation_width,
            request.variant,
            hidden_width=int(request.parameters["hidden_width"]),
        )
        model.load_state_dict(cast(dict[str, object], payload["model"]))
        model.eval()
        return model
    except (KeyError, OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, OptunaSearchError):
            raise
        raise OptunaSearchError("search checkpoint is invalid") from exc


def _evaluate_search_seed(
    config: RlConfig,
    request: TrialRequest,
    run_output: Path,
    seed_result: Mapping[str, object],
    seed: int,
) -> SeedValidationOutcome:
    loaded = load_episode_manifest(config, request.validation_manifest)
    if loaded.partition != "validation" or loaded.fold != request.fold:
        raise OptunaSearchError("search evaluator requires validation episodes")
    if not loaded.episodes:
        raise OptunaSearchError("search validation schedule is empty")
    first_environment = TopstepEntryEnvironment(config, loaded.episodes[0])
    first_observation, _ = first_environment.reset(seed=seed)
    model = _load_search_model(request, run_output, seed_result, len(first_observation.vector))
    passes = 0
    blows = 0
    pass_days: list[float] = []
    for episode in loaded.episodes:
        environment = TopstepEntryEnvironment(config, episode)
        observation, _ = environment.reset(seed=seed)
        while True:
            mask = environment.action_mask()
            action = 0
            if bool(mask[1]):
                with torch.no_grad():
                    logits, _values = model(
                        torch.from_numpy(observation.vector.copy()).unsqueeze(0),
                        torch.tensor([TICKERS.index(episode.ticker)]),
                        torch.tensor([PROFILES.index(episode.profile)]),
                    )
                    masked = logits.masked_fill(
                        ~torch.from_numpy(mask).unsqueeze(0),
                        torch.finfo(logits.dtype).min,
                    )
                    action = int(masked.argmax(dim=1).item())
            observation, _reward, terminated, truncated, info = environment.step(action)
            if terminated or truncated:
                status = str(info["status"])
                passes += int(status == "PASS")
                blows += int(status == "BLOW")
                if status == "PASS":
                    trading_days = environment.account_state["trading_days"]
                    if type(trading_days) is not int:
                        raise OptunaSearchError("validation trading-day count is invalid")
                    pass_days.append(float(trading_days))
                break
    median_days = float(np.median(pass_days)) if pass_days else None
    return SeedValidationOutcome(
        "validation", seed, len(loaded.episodes), passes, blows, median_days
    )


def execute_optuna_trial(config: RlConfig, request: TrialRequest) -> TrialEvaluation:
    """Train and score one trial without exposing test or sealed-holdout inputs."""
    hyperparameters = PpoHyperparameters.from_search(request.parameters)
    outcomes = list(request.completed_validation)
    for seed_index in range(len(outcomes), 3):
        seed = request.identity.seeds[seed_index]
        run_output = request.artifact_directory / f"seed-{seed_index}"
        completed = _completed_search_seed(run_output, request, seed_index)
        if completed is None:
            try:
                completed = train_entry_policy(
                    config,
                    request.training_manifest,
                    run_output,
                    seed=seed,
                    variant=request.variant,
                    target_timesteps=request.identity.timesteps_per_seed,
                    resume=run_output.exists(),
                    hyperparameters=hyperparameters,
                    search_trial_number=request.identity.trial_number,
                    search_seed_index=seed_index,
                )
            except ProductionTrainingError as exc:
                raise OptunaSearchError("search seed training failed") from exc
        if completed.get("status") != "complete":
            raise OptunaSearchError("search seed did not reach the 500K lower bound")
        outcome = _evaluate_search_seed(config, request, run_output, completed, seed)
        completed_timesteps = completed.get("completed_timesteps")
        overshoot = completed.get("timestep_overshoot")
        rollouts = completed.get("rollouts")
        rollout_decisions = rollouts.get("policy_decisions") if isinstance(rollouts, dict) else None
        try:
            state = json.loads((run_output / "state.json").read_text())
            checkpoint = run_output / cast(str, state["checkpoint"]) / "checkpoint.pt"
        except (KeyError, OSError, json.JSONDecodeError, TypeError) as exc:
            raise OptunaSearchError("search checkpoint pointer is invalid") from exc
        if (
            type(completed_timesteps) is not int
            or completed_timesteps < request.identity.timesteps_per_seed
            or type(overshoot) is not int
            or overshoot != completed_timesteps - request.identity.timesteps_per_seed
            or type(rollout_decisions) is not int
            or not 0 <= overshoot < rollout_decisions
            or not checkpoint.is_file()
        ):
            raise OptunaSearchError("search seed timestep or overshoot evidence is invalid")
        validation_core = {
            "schema_version": 1,
            "stage": "rl-optuna-seed-validation",
            "study_name": request.identity.study_name,
            "trial_number": request.identity.trial_number,
            "seed_index": seed_index,
            "seed": seed,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training_manifest_sha256": hashlib.sha256(
                (run_output / "manifest.json").read_bytes()
            ).hexdigest(),
            "validation_manifest_sha256": hashlib.sha256(
                request.validation_manifest.read_bytes()
            ).hexdigest(),
            "completed_timesteps": completed_timesteps,
            "timestep_overshoot": overshoot,
            "outcome": asdict(outcome),
            "sealed_holdout_accessed": False,
        }
        validation_path = request.artifact_directory / "validation" / f"seed-{seed_index}.json"
        _atomic_json_idempotent(validation_path, validation_core)
        outcome = replace(
            outcome,
            completed_timesteps=completed_timesteps,
            timestep_overshoot=overshoot,
            checkpoint_sha256=cast(str, validation_core["checkpoint_sha256"]),
            training_manifest_sha256=cast(str, validation_core["training_manifest_sha256"]),
            validation_manifest_sha256=cast(str, validation_core["validation_manifest_sha256"]),
            validation_artifact_sha256=hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        )
        outcomes.append(outcome)
        request.report_validation(outcome)
    return TrialEvaluation(request.identity.trial_number, tuple(outcomes))


def _load_validation_intermediates(
    output: Path, identity: TrialIdentity, contract: Mapping[str, object]
) -> tuple[SeedValidationOutcome, ...]:
    directory = output / "intermediates" / f"trial-{identity.trial_number:04d}"
    outcomes = []
    for seed_index in range(3):
        path = directory / f"seed-{seed_index}.json"
        if not path.exists():
            break
        raw = _load_trial_ledger(path)
        outcome_raw = raw.get("outcome")
        if (
            raw.get("stage") != "rl-optuna-validation-intermediate"
            or raw.get("study_identity_sha256") != contract["identity_sha256"]
            or raw.get("trial_number") != identity.trial_number
            or raw.get("seed_index") != seed_index
            or not isinstance(outcome_raw, dict)
        ):
            raise OptunaSearchError("validation intermediate provenance mismatch")
        outcome = _seed_validation_outcome(outcome_raw)
        if outcome.seed != identity.seeds[seed_index]:
            raise OptunaSearchError("validation intermediate seed mismatch")
        outcomes.append(outcome)
    return tuple(outcomes)


def _finish_running_trial(
    study: optuna.Study,
    trial: optuna.Trial,
    identity: TrialIdentity,
    parameters: dict[str, float | int],
    *,
    manifests: StudyManifests,
    selected_variant: PolicyVariant,
    contract: Mapping[str, object],
    output: Path,
    evaluator: Callable[[TrialRequest], TrialEvaluation],
) -> None:
    plan_path = output / "plans" / f"trial-{trial.number:04d}.json"
    plan = cast(
        dict[str, object],
        json.loads(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": "rl-optuna-trial-plan",
                    "study_identity_sha256": contract["identity_sha256"],
                    "trial_number": trial.number,
                    "identity": asdict(identity),
                    "parameters": parameters,
                    "parameters_sha256": _canonical_sha256(parameters),
                    "sealed_holdout_accessed": False,
                }
            )
        ),
    )
    if plan_path.exists():
        if _load_trial_ledger(plan_path) != plan:
            raise OptunaSearchError("immutable trial plan mismatch")
    else:
        _atomic_json_no_overwrite(plan_path, plan)
    ledger_path = output / "ledger" / f"trial-{trial.number:04d}.json"
    if ledger_path.exists():
        _tell_from_ledger(study, trial, _load_trial_ledger(ledger_path))
        return
    reported_validation = list(_load_validation_intermediates(output, identity, contract))

    def report_validation(outcome: SeedValidationOutcome) -> None:
        index = len(reported_validation)
        if index >= 3 or outcome.seed != identity.seeds[index]:
            raise OptunaSearchError("validation intermediate identity mismatch")
        _atomic_json_no_overwrite(
            output / "intermediates" / f"trial-{trial.number:04d}" / f"seed-{index}.json",
            {
                "schema_version": 1,
                "stage": "rl-optuna-validation-intermediate",
                "study_identity_sha256": contract["identity_sha256"],
                "trial_number": trial.number,
                "seed_index": index,
                "outcome": asdict(outcome),
                "sealed_holdout_accessed": False,
            },
        )
        reported_validation.append(outcome)
        cumulative_passes = sum(item.passes for item in reported_validation)
        cumulative_attempts = sum(item.attempts for item in reported_validation)
        cumulative_blows = sum(item.blows for item in reported_validation)
        trial.set_user_attr("validation_blow_count", cumulative_blows)
        trial.report(_pass_rate_lcb_95(cumulative_passes, cumulative_attempts), step=index)
        if len(reported_validation) >= 2 and trial.should_prune():
            raise PrunedTrial("validation MedianPruner decision", tuple(reported_validation))

    request = TrialRequest(
        identity,
        MappingProxyType(parameters),
        selected_variant,
        manifests.training,
        manifests.validation,
        manifests.fold,
        output / "trials" / f"trial-{trial.number:04d}",
        tuple(reported_validation),
        report_validation,
    )
    try:
        evaluation = evaluator(request)
        if (
            evaluation.trial_number != trial.number
            or tuple(outcome.seed for outcome in evaluation.outcomes) != identity.seeds
        ):
            raise OptunaSearchError("trial evaluation identity mismatch")
        if contract.get("production_evaluator") is True:
            for outcome in evaluation.outcomes:
                outcome.require_production_evidence()
        for outcome in evaluation.outcomes[len(reported_validation) :]:
            report_validation(outcome)
        trial.set_user_attr("validation_blow_count", evaluation.aggregate_blows)
        ledger = {
            **plan,
            "stage": "rl-optuna-trial",
            "state": "complete",
            "evaluation": _evaluation_payload(evaluation),
        }
        _atomic_json_no_overwrite(ledger_path, ledger)
        _tell_from_ledger(study, trial, ledger)
    except PrunedTrial as exc:
        if (
            tuple(outcome.seed for outcome in exc.partial_validation)
            != identity.seeds[: len(exc.partial_validation)]
        ):
            raise OptunaSearchError("pruning evidence identity mismatch") from exc
        ledger = {
            **plan,
            "stage": "rl-optuna-trial",
            "state": "pruned",
            "pruning_reason": exc.reason,
            "partial_validation": [asdict(outcome) for outcome in exc.partial_validation],
        }
        _atomic_json_no_overwrite(ledger_path, ledger)
        study.tell(trial, state=TrialState.PRUNED)
    except Exception as exc:
        if not ledger_path.exists():
            ledger = {
                **plan,
                "stage": "rl-optuna-trial",
                "state": "failed",
                "failure_type": type(exc).__name__,
                "failure": str(exc),
            }
            _atomic_json_no_overwrite(ledger_path, ledger)
            study.tell(trial, state=TrialState.FAIL)
        raise


def _run_optuna_study_locked(
    config: RlConfig,
    training_manifest: Path,
    validation_manifest: Path,
    output: Path,
    *,
    study_name: str,
    variant: PolicyVariant | str,
    evaluator: Callable[[TrialRequest], TrialEvaluation],
    maximum_trials_this_run: int | None = None,
    require_production_evidence: bool = False,
) -> dict[str, object]:
    """Run or resume the immutable 30-attempt validation search."""
    if maximum_trials_this_run is not None and maximum_trials_this_run < 1:
        raise OptunaSearchError("invocation trial budget must be positive")
    if config.training.maximum_search_trials != 30:
        raise OptunaSearchError("Optuna search requires the accepted 30-trial ceiling")
    selected_variant = PolicyVariant(variant)
    manifests = validate_study_manifests(config, training_manifest, validation_manifest)
    contract = _study_contract(
        config, manifests, study_name, selected_variant, require_production_evidence
    )
    database = output / "study.sqlite3"
    storage = _storage_url(database)
    resumed_trial = False
    if database.exists():
        study = optuna.load_study(study_name=study_name, storage=storage)
        if study.user_attrs != contract:
            raise OptunaSearchError("persistent study identity mismatch")
        attempted = len(study.trials)
        if attempted >= config.training.maximum_search_trials:
            raise OptunaSearchError("persistent study already reached the 30-trial ceiling")
        running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
        if len(running) > 1:
            raise OptunaSearchError("persistent study has multiple RUNNING trials")
        if running:
            frozen = running[0]
            identity = derive_trial_identity(config, study_name, frozen.number)
            study = optuna.load_study(
                study_name=study_name,
                storage=storage,
                sampler=_sampler(identity.proposal_seed),
                pruner=_pruner(),
            )
            trial = optuna.Trial(study, frozen._trial_id)
            parameters = _suggest_parameters(trial)
            if trial.number != frozen.number:
                raise OptunaSearchError("RUNNING trial identity changed during resume")
            trial.set_user_attr("trial_identity", asdict(identity))
            trial.set_user_attr("parameters_sha256", _canonical_sha256(parameters))
            _finish_running_trial(
                study,
                trial,
                identity,
                parameters,
                manifests=manifests,
                selected_variant=selected_variant,
                contract=contract,
                output=output,
                evaluator=evaluator,
            )
            resumed_trial = True
    else:
        output.mkdir(parents=True, exist_ok=False)
        sampler = _sampler(derive_trial_identity(config, study_name, 0).proposal_seed)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            direction="maximize",
            pruner=_pruner(),
        )
        for key, value in contract.items():
            study.set_user_attr(key, value)
    attempted = len(study.trials)
    ceiling = config.training.maximum_search_trials
    remaining = ceiling - attempted
    invocation_trials = (
        remaining if maximum_trials_this_run is None else min(remaining, maximum_trials_this_run)
    )
    if resumed_trial and maximum_trials_this_run is not None:
        invocation_trials = max(0, invocation_trials - 1)
    for _ in range(invocation_trials):
        next_number = len(study.trials)
        identity = derive_trial_identity(config, study_name, next_number)
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=_sampler(identity.proposal_seed),
            pruner=_pruner(),
        )
        trial = study.ask()
        if trial.number != next_number:
            raise OptunaSearchError("persistent study trial sequence is not contiguous")
        parameters = _suggest_parameters(trial)
        trial.set_user_attr("trial_identity", asdict(identity))
        trial.set_user_attr("parameters_sha256", _canonical_sha256(parameters))
        _finish_running_trial(
            study,
            trial,
            identity,
            parameters,
            manifests=manifests,
            selected_variant=selected_variant,
            contract=contract,
            output=output,
            evaluator=evaluator,
        )
    attempted = len(study.trials)
    result: dict[str, object] = {
        "stage": "rl-optuna-search",
        "status": "complete" if attempted == ceiling else "incomplete",
        "study_name": study_name,
        "study_identity_sha256": contract["identity_sha256"],
        "attempted_trials": attempted,
        "maximum_trials": ceiling,
        "sealed_holdout_accessed": False,
    }
    if attempted == ceiling:
        winner = select_winner(_completed_evaluations(output, study, contract, config, study_name))
        winning_ledger = _load_trial_ledger(
            output / "ledger" / f"trial-{winner.trial_number:04d}.json"
        )
        winner_payload = {
            "schema_version": 1,
            "stage": "rl-optuna-winner",
            "study_name": study_name,
            "study_identity_sha256": contract["identity_sha256"],
            "trial_number": winner.trial_number,
            "parameters": winning_ledger["parameters"],
            "parameters_sha256": winning_ledger["parameters_sha256"],
            "validation": _evaluation_payload(winner),
            "sealed_holdout_accessed": False,
        }
        _atomic_json_no_overwrite(output / "winner.json", winner_payload)
        result["winner"] = winner_payload
    return result


@contextmanager
def _exclusive_study_lock(output: Path) -> Iterator[None]:
    lock_path = output.parent / f".{output.name}.study.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def run_optuna_study(
    config: RlConfig,
    training_manifest: Path,
    validation_manifest: Path,
    output: Path,
    *,
    study_name: str,
    variant: PolicyVariant | str,
    evaluator: Callable[[TrialRequest], TrialEvaluation],
    maximum_trials_this_run: int | None = None,
    require_production_evidence: bool = False,
) -> dict[str, object]:
    """Hold one process lock while running or resuming the persistent study."""
    validate_study_manifests(config, training_manifest, validation_manifest)
    with _exclusive_study_lock(output):
        return _run_optuna_study_locked(
            config,
            training_manifest,
            validation_manifest,
            output,
            study_name=study_name,
            variant=variant,
            evaluator=evaluator,
            maximum_trials_this_run=maximum_trials_this_run,
            require_production_evidence=require_production_evidence,
        )


def run_production_optuna_study(
    config: RlConfig,
    training_manifest: Path,
    validation_manifest: Path,
    output: Path,
    *,
    study_name: str,
    variant: PolicyVariant | str,
) -> dict[str, object]:
    """Run the complete production study through the concrete train/validate objective."""
    training = load_training_episodes(config, training_manifest)
    validation = load_episode_manifest(config, validation_manifest)
    if (
        training.partition != "training"
        or validation.partition != "validation"
        or training.fold != validation.fold
        or not training.episodes
        or not validation.episodes
    ):
        raise OptunaSearchError("production study manifests are not same-fold train/validation")
    training_end = max(episode.bars[-1].timestamp for episode in training.episodes)
    validation_start = min(episode.bars[0].timestamp for episode in validation.episodes)
    validation_end = max(episode.bars[-1].timestamp for episode in validation.episodes)
    if training_end >= config.evaluation.sealed_holdout_start or validation_end >= (
        config.evaluation.sealed_holdout_start
    ):
        raise OptunaSearchError("production study episodes reach the sealed holdout")
    if training_end >= validation_start:
        raise OptunaSearchError("production study train/validation schedules overlap")
    return run_optuna_study(
        config,
        training_manifest,
        validation_manifest,
        output,
        study_name=study_name,
        variant=variant,
        evaluator=lambda request: execute_optuna_trial(config, request),
        require_production_evidence=True,
    )


def _derived_seed(namespace: str, trial_number: int, index: int) -> int:
    payload = f"{namespace}:{trial_number}:{index}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return 1 + value % 2_147_483_646


def _all_training_seeds() -> tuple[int, ...]:
    seeds = tuple(
        _derived_seed(_TRAINING_SEED_NAMESPACE, trial_number, index)
        for trial_number in range(30)
        for index in range(3)
    )
    if len(set(seeds)) != 90:
        raise OptunaSearchError("Optuna v1 derived training seeds collide")
    return seeds


def derive_trial_identity(config: RlConfig, study_name: str, trial_number: int) -> TrialIdentity:
    """Derive the fixed production search budget and independent trial seeds."""
    if not study_name or not 0 <= trial_number < 30:
        raise ValueError("study name and trial number must identify one attempted trial")
    if config.training.search_seeds != 3 or config.training.search_timesteps_per_seed != 500_000:
        raise ValueError("Optuna search requires three seeds at 500000 steps each")
    all_seeds = _all_training_seeds()
    seeds = all_seeds[trial_number * 3 : trial_number * 3 + 3]
    proposal_seed = _derived_seed(_PROPOSAL_SEED_NAMESPACE, trial_number, 0)
    return TrialIdentity(
        study_name,
        trial_number,
        proposal_seed,
        seeds,
        config.training.search_timesteps_per_seed,
    )


__all__ = [
    "SEARCH_SPACE",
    "OptunaSearchError",
    "PrunedTrial",
    "SearchDistribution",
    "SeedValidationOutcome",
    "StudyManifests",
    "TrialEvaluation",
    "TrialIdentity",
    "TrialRequest",
    "derive_trial_identity",
    "execute_optuna_trial",
    "run_optuna_study",
    "run_production_optuna_study",
    "select_winner",
    "validate_study_manifests",
]
