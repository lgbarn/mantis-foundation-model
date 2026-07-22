"""Fail-closed architecture qualification and seed confirmation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch

from mantis_v2.rl_config import RlConfig
from mantis_v2.rl_environment import TopstepEntryEnvironment
from mantis_v2.rl_optuna import validate_study_manifests
from mantis_v2.rl_policy import PROFILES, TICKERS, EntryActorCritic, PolicyVariant
from mantis_v2.rl_provenance import verify_rl_runtime
from mantis_v2.rl_training import (
    PpoHyperparameters,
    _canonical_digest,
    _completed_timesteps,
    train_entry_policy,
)
from mantis_v2.rl_validation import LoadedEpisodes, load_episode_manifest

_CANDIDATE = PolicyVariant.SHARED_TICKER_VALUE
_ABLATIONS = frozenset(variant.value for variant in PolicyVariant)
_MARKET_BOOTSTRAP_REPLICATES = 100_000
_SEED_SUMMARY_REPLICATES = 5_000
_MINIMUM_MARKET_BLOCKS = 20
_WEEK_PATTERN = re.compile(r"^[0-9]{4}-W(?:0[1-9]|[1-4][0-9]|5[0-3])$")
_RAW_ROW_FIELDS = frozenset(
    {
        "variant",
        "fold",
        "seed",
        "ticker",
        "profile",
        "regime",
        "calendar_block",
        "episode_id",
        "outcome",
        "finite",
        "action_collapsed",
    }
)


def _calendar_regime(calendar_block: str) -> str:
    year, week = calendar_block.split("-W", 1)
    month = datetime.fromisocalendar(int(year), int(week), 1).month
    return f"calendar-quarter-{(month - 1) // 3 + 1}"


class ConfirmationError(ValueError):
    """Raised when architecture or seed evidence cannot support the fixed claim."""


@dataclass(frozen=True)
class ConfirmationRequest:
    """One immutable seed-training request passed to an execution adapter."""

    phase: str
    fold: int
    training_manifest_sha256: str
    validation_manifest_sha256: str
    seed: int
    timesteps: int
    variant: str
    parameters: Mapping[str, object]
    candidate_sha256: str
    output: Path
    resume: bool
    parent_artifact_sha256: str | None
    required_milestone_timesteps: int


ConfirmationRunner = Callable[[ConfirmationRequest], Mapping[str, object]]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfirmationError(f"{description} is unreadable") from exc
    if not isinstance(value, dict):
        raise ConfirmationError(f"{description} must be a JSON object")
    return cast(dict[str, object], value)


def _atomic_no_overwrite(path: Path, payload: object) -> None:
    data = _canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if path.read_bytes() != data:
                raise ConfirmationError(
                    f"refusing to overwrite immutable artifact: {path}"
                ) from exc
        else:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _validated_winner(path: Path) -> tuple[dict[str, object], str]:
    winner = _load_object(path, "Optuna winner")
    parameters = winner.get("parameters")
    if (
        winner.get("schema_version") != 1
        or winner.get("stage") != "rl-optuna-winner"
        or winner.get("sealed_holdout_accessed") is not False
        or not isinstance(parameters, dict)
        or not parameters
        or not isinstance(winner.get("parameters_sha256"), str)
        or winner.get("parameters_sha256")
        != hashlib.sha256(_canonical_bytes(parameters)).hexdigest()
    ):
        raise ConfirmationError("Optuna winner contract is invalid")
    return winner, _sha256(path)


def freeze_architecture_plan(
    config: RlConfig,
    winner_path: Path,
    training_manifest: Path | Sequence[Path],
    validation_manifest: Path | Sequence[Path],
    output: Path,
    *,
    created_at: str,
    runtime_identities: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Publish the immutable statistical and schedule plan before outcomes exist."""
    winner, winner_sha256 = _validated_winner(winner_path)
    training_paths = (
        [training_manifest] if isinstance(training_manifest, Path) else list(training_manifest)
    )
    validation_paths = (
        [validation_manifest]
        if isinstance(validation_manifest, Path)
        else list(validation_manifest)
    )
    if not training_paths or len(training_paths) != len(validation_paths):
        raise ConfirmationError("architecture plan requires matched manifest pairs")
    manifest_pairs = []
    folds: set[int] = set()
    for training_path, validation_path in zip(training_paths, validation_paths, strict=True):
        manifests = validate_study_manifests(config, training_path, validation_path)
        if manifests.fold in folds:
            raise ConfirmationError("architecture plan contains a duplicate fold")
        folds.add(manifests.fold)
        manifest_pairs.append(
            {
                "fold": manifests.fold,
                "training_manifest_path": str(training_path.resolve()),
                "training_manifest_sha256": manifests.training_sha256,
                "validation_manifest_path": str(validation_path.resolve()),
                "validation_manifest_sha256": manifests.validation_sha256,
            }
        )
    manifest_pairs.sort(key=lambda pair: cast(int, pair["fold"]))
    runtime = dict(runtime_identities or verify_rl_runtime(config))
    source = runtime.get("source")
    lock = runtime.get("lock")
    if (
        not isinstance(source, dict)
        or source.get("dirty") is not False
        or source.get("sha256") != config.upstream.source_digest
        or not isinstance(source.get("revision"), str)
        or not isinstance(lock, dict)
        or lock.get("sha256") != config.upstream.lock_digest
    ):
        raise ConfirmationError("architecture plan runtime provenance mismatch")
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "rl-architecture-plan-v1",
        "created_at": created_at,
        "config_sha256": config.digest,
        "winner_path": str(winner_path.resolve()),
        "winner_sha256": winner_sha256,
        "winner_study_identity_sha256": winner["study_identity_sha256"],
        "source": source,
        "dependency_lock": lock,
        "manifest_pairs": manifest_pairs,
        "training_manifest_sha256": hashlib.sha256(
            _canonical_bytes([pair["training_manifest_sha256"] for pair in manifest_pairs])
        ).hexdigest(),
        "validation_manifest_sha256": hashlib.sha256(
            _canonical_bytes([pair["validation_manifest_sha256"] for pair in manifest_pairs])
        ).hexdigest(),
        "folds": sorted(folds),
        "variants": sorted(_ABLATIONS),
        "seeds": list(config.training.development_seeds),
        "candidate": _CANDIDATE.value,
        "pair_key": ["fold", "calendar_block", "episode_id", "ticker", "profile", "seed"],
        "regime_assignment": "exchange_calendar_quarter_v1",
        "bootstrap": {
            "replicates": _MARKET_BOOTSTRAP_REPLICATES,
            "rng": "numpy.PCG64",
            "quantile": 0.05,
            "quantile_method": "inverted_cdf",
            "minimum_complete_blocks": _MINIMUM_MARKET_BLOCKS,
        },
        "test_outcomes_accessed": False,
        "sealed_holdout_accessed": False,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    path = output / f"architecture-plan-{digest}.json"
    _atomic_no_overwrite(path, payload)
    return {"plan_path": str(path), "plan_sha256": digest, **payload}


def _validated_plan(
    config: RlConfig, winner_sha256: str, evidence: Mapping[str, object]
) -> tuple[dict[str, object], str]:
    plan_path_value = evidence.get("plan_path")
    if not isinstance(plan_path_value, str):
        raise ConfirmationError("architecture evidence is missing its preregistered plan")
    plan_path = Path(plan_path_value)
    plan = _load_object(plan_path, "architecture plan")
    digest = _sha256(plan_path)
    if (
        plan_path.name != f"architecture-plan-{digest}.json"
        or evidence.get("plan_sha256") != digest
        or plan.get("stage") != "rl-architecture-plan-v1"
        or plan.get("config_sha256") != config.digest
        or plan.get("winner_sha256") != winner_sha256
        or plan.get("candidate") != _CANDIDATE.value
        or plan.get("variants") != sorted(_ABLATIONS)
        or plan.get("seeds") != list(config.training.development_seeds)
        or plan.get("test_outcomes_accessed") is not False
        or plan.get("sealed_holdout_accessed") is not False
    ):
        raise ConfirmationError("architecture plan provenance mismatch")
    pairs = _validated_plan_pairs(config, plan)
    training_digest = hashlib.sha256(
        _canonical_bytes([pair["training_manifest_sha256"] for pair in pairs])
    ).hexdigest()
    validation_digest = hashlib.sha256(
        _canonical_bytes([pair["validation_manifest_sha256"] for pair in pairs])
    ).hexdigest()
    if (
        training_digest != plan.get("training_manifest_sha256")
        or validation_digest != plan.get("validation_manifest_sha256")
        or [pair["fold"] for pair in pairs] != plan.get("folds")
    ):
        raise ConfirmationError("architecture plan schedule provenance mismatch")
    return plan, digest


def _validated_plan_pairs(config: RlConfig, plan: Mapping[str, object]) -> list[dict[str, object]]:
    raw_pairs = plan.get("manifest_pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ConfirmationError("architecture plan manifest pairs are missing")
    pairs: list[dict[str, object]] = []
    prior_fold = -1
    for raw in raw_pairs:
        if not isinstance(raw, dict) or type(raw.get("fold")) is not int:
            raise ConfirmationError("architecture plan manifest pair is invalid")
        fold = cast(int, raw["fold"])
        training_path = Path(cast(str, raw["training_manifest_path"]))
        validation_path = Path(cast(str, raw["validation_manifest_path"]))
        manifests = validate_study_manifests(config, training_path, validation_path)
        if (
            fold <= prior_fold
            or manifests.fold != fold
            or manifests.training_sha256 != raw.get("training_manifest_sha256")
            or manifests.validation_sha256 != raw.get("validation_manifest_sha256")
        ):
            raise ConfirmationError("architecture plan manifest pair provenance mismatch")
        prior_fold = fold
        pairs.append(cast(dict[str, object], raw))
    return pairs


def _validated_training_artifact(
    config: RlConfig,
    manifest_path: Path,
    bundle_path: Path,
    *,
    fold: int,
    seed: int,
    variant: str,
    schedule_sha256: str,
    target_timesteps: int,
    hyperparameters: Mapping[str, object],
    campaign_phase: str | None = None,
    parent_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    """Recursively validate a published trainer manifest, bundle, and checkpoint."""
    manifest = _load_object(manifest_path, "training manifest")
    bundle = _load_object(bundle_path, "training checkpoint bundle")
    checkpoint_path = bundle_path.parent / "checkpoint.pt"
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise ConfirmationError("training checkpoint payload is unreadable") from exc
    identities = manifest.get("identities")
    required_payload_fields = {
        "model",
        "optimizer",
        "normalizers",
        "constraint_controller",
        "rng",
        "metrics",
    }
    if (
        manifest.get("schema_version") != 2
        or manifest.get("stage") != "rl-train"
        or manifest.get("status") != "complete"
        or manifest.get("finite_gradients") is not True
        or manifest.get("deterministic_reload_actions") is not True
        or manifest.get("fold") != fold
        or manifest.get("seed") != seed
        or manifest.get("variant") != variant
        or manifest.get("minimum_target_timesteps") != target_timesteps
        or not isinstance(manifest.get("completed_timesteps"), int)
        or cast(int, manifest["completed_timesteps"]) < target_timesteps
        or manifest.get("ppo_hyperparameters") != dict(hyperparameters)
        or not isinstance(identities, dict)
        or identities.get("completion_mode") != "production_timesteps"
        or identities.get("target_timesteps") != target_timesteps
        or identities.get("target_updates") is not None
        or identities.get("config_sha256") != config.digest
        or identities.get("schedule_sha256") != schedule_sha256
        or identities.get("fold") != fold
        or identities.get("seed") != seed
        or identities.get("variant") != variant
        or identities.get("ppo_hyperparameters") != dict(hyperparameters)
        or identities.get("campaign_phase") != campaign_phase
        or identities.get("parent_checkpoint_sha256") != parent_checkpoint_sha256
        or identities.get("partition") != "training"
        or identities.get("source_sha256") != config.upstream.source_digest
        or identities.get("dependency_lock_sha256") != config.upstream.lock_digest
        or bundle.get("schema_version") != 3
        or bundle.get("identities") != identities
        or bundle.get("checkpoint_sha256") != _sha256(checkpoint_path)
        or not isinstance(payload, dict)
        or payload.get("schema_version") != 3
        or payload.get("identities") != identities
        or payload.get("update") != bundle.get("update")
        or payload.get("episode_cursor") != bundle.get("episode_cursor")
        or not required_payload_fields.issubset(payload)
    ):
        raise ConfirmationError("training artifact provenance mismatch")
    metrics = payload.get("metrics")
    if not isinstance(metrics, list):
        raise ConfirmationError("training checkpoint timestep metrics are invalid")
    try:
        actual_timesteps = _completed_timesteps(cast(list[dict[str, object]], metrics))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ConfirmationError("training checkpoint timestep metrics are invalid") from exc
    if (
        manifest.get("metrics") != metrics
        or manifest.get("completed_timesteps") != actual_timesteps
        or manifest.get("timestep_overshoot") != actual_timesteps - target_timesteps
    ):
        raise ConfirmationError("training artifact timestep provenance mismatch")
    model_state = payload.get("model")
    critic_input = (
        model_state.get("critic_trunk.0.weight") if isinstance(model_state, dict) else None
    )
    if (
        not isinstance(critic_input, torch.Tensor)
        or critic_input.ndim != 2
        or critic_input.shape[1] <= 10
    ):
        raise ConfirmationError("training checkpoint model schema mismatch")
    observation_width = int(critic_input.shape[1]) - 10
    try:
        model = EntryActorCritic(
            observation_width,
            variant,
            hidden_width=cast(int, hyperparameters["hidden_width"]),
        )
        model.load_state_dict(cast(dict[str, object], model_state), strict=True)
        model.eval()
        with torch.no_grad():
            logits, values = model(
                torch.zeros((1, observation_width)),
                torch.zeros(1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )
    except (KeyError, RuntimeError, ValueError, TypeError) as exc:
        raise ConfirmationError("training checkpoint model schema mismatch") from exc
    if not _inference_is_finite(logits.numpy(), values.numpy()):
        raise ConfirmationError("training checkpoint model is non-finite")
    return cast(dict[str, object], payload)


def _completed_training_artifact(
    config: RlConfig,
    run_output: Path,
    *,
    fold: int,
    seed: int,
    variant: str,
    schedule_sha256: str,
    target_timesteps: int,
    hyperparameters: Mapping[str, object],
    campaign_phase: str | None = None,
    parent_checkpoint_sha256: str | None = None,
) -> tuple[dict[str, object], dict[str, object], Path]:
    manifest_path = run_output / "manifest.json"
    manifest = _load_object(manifest_path, "training manifest")
    state = _load_object(run_output / "state.json", "training state")
    pointer = state.get("checkpoint")
    if not isinstance(pointer, str) or not re.fullmatch(r"checkpoints/update-[0-9]{6}", pointer):
        raise ConfirmationError("training checkpoint pointer is invalid")
    bundle_path = run_output / pointer / "bundle.json"
    payload = _validated_training_artifact(
        config,
        manifest_path,
        bundle_path,
        fold=fold,
        seed=seed,
        variant=variant,
        schedule_sha256=schedule_sha256,
        target_timesteps=target_timesteps,
        hyperparameters=hyperparameters,
        campaign_phase=campaign_phase,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )
    if (
        state.get("schema_version") != 2
        or state.get("status") != "complete"
        or state.get("identities") != manifest.get("identities")
        or state.get("completed_timesteps") != manifest.get("completed_timesteps")
        or state.get("minimum_target_timesteps") != target_timesteps
        or state.get("completed_updates") != payload.get("update")
        or state.get("episode_cursor") != payload.get("episode_cursor")
    ):
        raise ConfirmationError("completed training state provenance mismatch")
    return manifest, payload, bundle_path


def _inference_is_finite(*values: object) -> bool:
    return all(bool(np.isfinite(np.asarray(value)).all()) for value in values)


def _policy_replay_diagnostics(
    model: EntryActorCritic,
    probes: Sequence[tuple[np.ndarray, np.ndarray, int, int]],
) -> dict[str, object]:
    if not probes:
        raise ConfirmationError("policy replay has no policy probes")
    observations = torch.from_numpy(np.stack([probe[0] for probe in probes]))
    masks = torch.from_numpy(np.stack([probe[1] for probe in probes]))
    tickers = torch.tensor([probe[2] for probe in probes], dtype=torch.long)
    profiles = torch.tensor([probe[3] for probe in probes], dtype=torch.long)
    if masks.dtype is not torch.bool or not bool(masks.any(dim=1).all()):
        raise ConfirmationError("policy replay action mask is invalid")
    with torch.no_grad():
        logits, values = model(observations, tickers, profiles)
        masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(masked_logits, dim=1)
        actions = probabilities.argmax(dim=1)
        entropies = -(
            probabilities * probabilities.clamp_min(torch.finfo(logits.dtype).tiny).log()
        ).sum(dim=1)
    if not _inference_is_finite(logits.numpy(), values.numpy(), probabilities.numpy()):
        raise ConfirmationError("policy replay diagnostics are non-finite")
    enter_rate = float((actions == 1).float().mean())
    mean_entropy = float(entropies.mean())
    action_count = len(set(actions.tolist()))
    return {
        "probe_count": len(probes),
        "enter_rate": enter_rate,
        "mean_entropy": mean_entropy,
        "deterministic_action_count": action_count,
        "reject_all_detected": enter_rate == 0.0,
        "take_all_detected": enter_rate == 1.0,
        "action_collapse_detected": action_count == 1 or mean_entropy < 0.05,
    }


def _deterministic_policy_replay(
    config: RlConfig,
    loaded: LoadedEpisodes,
    raw_episodes: Sequence[object],
    model: EntryActorCritic,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, int]]:
    if len(raw_episodes) != len(loaded.episodes):
        raise ConfirmationError("policy replay episode identities are invalid")
    rows: list[dict[str, object]] = []
    probes: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    outcomes = {"PASS": 0, "BLOW": 0, "TIMEOUT": 0}
    for index, (episode, identity) in enumerate(zip(loaded.episodes, raw_episodes, strict=True)):
        environment = TopstepEntryEnvironment(config, episode)
        observation, _ = environment.reset(seed=seed)
        probes.append(
            (
                observation.vector.copy(),
                environment.action_mask().copy(),
                TICKERS.index(episode.ticker),
                PROFILES.index(episode.profile),
            )
        )
        while True:
            if not _inference_is_finite(observation.vector):
                raise ConfirmationError("policy replay observation is non-finite")
            mask = environment.action_mask()
            action = 0
            if bool(mask[1]):
                with torch.no_grad():
                    logits, values = model(
                        torch.from_numpy(observation.vector.copy()).unsqueeze(0),
                        torch.tensor([TICKERS.index(episode.ticker)]),
                        torch.tensor([PROFILES.index(episode.profile)]),
                    )
                    if not _inference_is_finite(logits.numpy(), values.numpy()):
                        raise ConfirmationError("policy replay inference is non-finite")
                    action = int(
                        logits.masked_fill(
                            ~torch.from_numpy(mask).unsqueeze(0),
                            torch.finfo(logits.dtype).min,
                        )
                        .argmax(dim=1)
                        .item()
                    )
            observation, reward, terminated, truncated, info = environment.step(action)
            if not _inference_is_finite(reward, observation.vector):
                raise ConfirmationError("policy replay transition is non-finite")
            if terminated or truncated:
                outcome = str(info["status"])
                if outcome not in outcomes:
                    raise ConfirmationError("policy replay outcome is invalid")
                outcomes[outcome] += 1
                break
        iso = episode.bars[0].timestamp.isocalendar()
        number = identity.get("number") if isinstance(identity, dict) else index
        rows.append(
            {
                "fold": loaded.fold,
                "seed": seed,
                "ticker": episode.ticker,
                "profile": episode.profile,
                "regime": _calendar_regime(f"{iso.year}-W{iso.week:02d}"),
                "calendar_block": f"{iso.year}-W{iso.week:02d}",
                "episode_id": str(number),
                "outcome": outcome,
                "finite": True,
                "action_collapsed": False,
            }
        )
    diagnostics = _policy_replay_diagnostics(model, probes)
    for row in rows:
        row["action_collapsed"] = diagnostics["action_collapse_detected"]
    return rows, diagnostics, outcomes


def _replay_architecture_rows(
    config: RlConfig,
    report: Mapping[str, object],
    checkpoint_payload: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Deterministically reproduce evaluator rows from checkpoint plus frozen validation input."""
    validation_path = Path(cast(str, report["validation_manifest_path"]))
    loaded = load_episode_manifest(config, validation_path)
    raw_validation = _load_object(validation_path, "validation schedule")
    raw_episodes = raw_validation.get("episodes")
    model_state = checkpoint_payload.get("model")
    critic_input = (
        model_state.get("critic_trunk.0.weight") if isinstance(model_state, dict) else None
    )
    if (
        not isinstance(raw_episodes, list)
        or len(raw_episodes) != len(loaded.episodes)
        or not isinstance(critic_input, torch.Tensor)
    ):
        raise ConfirmationError("architecture replay inputs are invalid")
    identities = checkpoint_payload.get("identities")
    hyperparameters = (
        identities.get("ppo_hyperparameters") if isinstance(identities, dict) else None
    )
    if not isinstance(hyperparameters, dict):
        raise ConfirmationError("architecture replay hyperparameters are invalid")
    model = EntryActorCritic(
        int(critic_input.shape[1]) - 10,
        cast(str, report["variant"]),
        hidden_width=cast(int, hyperparameters["hidden_width"]),
    )
    model.load_state_dict(cast(dict[str, object], model_state), strict=True)
    model.eval()
    base_rows, diagnostics, _outcomes = _deterministic_policy_replay(
        config, loaded, raw_episodes, model, cast(int, report["seed"])
    )
    return [{"variant": report["variant"], **row} for row in base_rows], diagnostics


def _validated_architecture_replay(
    config: RlConfig,
    report: Mapping[str, object],
    checkpoint_payload: Mapping[str, object],
    report_rows: object,
) -> list[dict[str, object]]:
    replayed_rows, replayed_diagnostics = _replay_architecture_rows(
        config, report, checkpoint_payload
    )
    if replayed_rows != report_rows or replayed_diagnostics != report.get("policy_diagnostics"):
        raise ConfirmationError("architecture validation replay mismatch")
    return replayed_rows


def _record_architecture_failure(
    ledger_path: Path,
    attempt_number: int,
    fold: int,
    variant: PolicyVariant,
    seed: int,
    phase: str,
    error: Exception,
) -> None:
    _atomic_no_overwrite(
        ledger_path,
        {
            "schema_version": 1,
            "stage": "rl-architecture-ablation-attempt",
            "status": "failed",
            "attempt_number": attempt_number,
            "fold": fold,
            "variant": variant.value,
            "seed": seed,
            "failure": {
                "phase": phase,
                "type": type(error).__name__,
                "message": str(error),
            },
            "test_accessed": False,
            "sealed_holdout_accessed": False,
        },
    )


def _evaluate_architecture_attempt(
    config: RlConfig,
    loaded: LoadedEpisodes,
    raw_episodes: list[object],
    model: EntryActorCritic,
    variant: PolicyVariant,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    base_rows, diagnostics, _outcomes = _deterministic_policy_replay(
        config, loaded, raw_episodes, model, seed
    )
    return [{"variant": variant.value, **row} for row in base_rows], diagnostics


def run_architecture_ablation(
    config: RlConfig,
    plan_path: Path,
    output: Path,
    *,
    bounded_target_updates: int | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Train all preregistered variants and derive raw outcomes from validation replay."""
    plan = _load_object(plan_path, "architecture plan")
    plan_sha256 = _sha256(plan_path)
    if plan_path.name != f"architecture-plan-{plan_sha256}.json":
        raise ConfirmationError("architecture plan content address mismatch")
    winner_path = Path(cast(str, plan["winner_path"]))
    winner, winner_sha256 = _validated_winner(winner_path)
    evidence_stub = {"plan_path": str(plan_path), "plan_sha256": plan_sha256}
    _validated_plan(config, winner_sha256, evidence_stub)
    pairs = _validated_plan_pairs(config, plan)
    parameters = cast(dict[str, object], winner["parameters"])
    hyperparameters = PpoHyperparameters.from_search(parameters)
    references = []
    attempt_number = 0
    for pair in pairs:
        fold = cast(int, pair["fold"])
        training_path = Path(cast(str, pair["training_manifest_path"]))
        validation_path = Path(cast(str, pair["validation_manifest_path"]))
        loaded = load_episode_manifest(config, validation_path)
        raw_validation = _load_object(validation_path, "validation schedule")
        raw_episodes = raw_validation.get("episodes")
        if not isinstance(raw_episodes, list) or len(raw_episodes) != len(loaded.episodes):
            raise ConfirmationError("validation episode identities are invalid")
        for variant in PolicyVariant:
            for seed in config.training.development_seeds:
                attempt_number += 1
                ledger_path = output / "ledger" / f"attempt-{attempt_number:04d}.json"
                if resume and ledger_path.is_file():
                    ledger_record = _load_object(ledger_path, "architecture attempt")
                    if ledger_record.get("status") == "failed":
                        failure = ledger_record.get("failure")
                        detail = (
                            failure.get("message")
                            if isinstance(failure, dict)
                            else "unknown failure"
                        )
                        raise ConfirmationError(
                            f"architecture attempt {attempt_number} previously failed: {detail}"
                        )
                    reference = ledger_record.get("report")
                    if (
                        ledger_record.get("schema_version") != 1
                        or ledger_record.get("stage") != "rl-architecture-ablation-attempt"
                        or ledger_record.get("status") != "complete"
                        or ledger_record.get("attempt_number") != attempt_number
                        or ledger_record.get("fold") != fold
                        or ledger_record.get("variant") != variant.value
                        or ledger_record.get("seed") != seed
                        or ledger_record.get("test_accessed") is not False
                        or ledger_record.get("sealed_holdout_accessed") is not False
                        or not isinstance(reference, dict)
                        or _sha256(Path(cast(str, reference["path"]))) != reference.get("sha256")
                    ):
                        raise ConfirmationError("architecture resume ledger mismatch")
                    references.append(reference)
                    continue
                run_output = output / "runs" / f"fold-{fold}" / variant.value / f"seed-{seed}"
                if (run_output / "manifest.json").is_file():
                    if not resume:
                        raise ConfirmationError("architecture run exists; resume is required")
                    trained = _load_object(run_output / "manifest.json", "architecture training")
                else:
                    try:
                        trained = train_entry_policy(
                            config,
                            training_path,
                            run_output,
                            seed=seed,
                            variant=variant,
                            target_updates=bounded_target_updates,
                            target_timesteps=(
                                None
                                if bounded_target_updates is not None
                                else config.training.development_timesteps_per_seed
                            ),
                            hyperparameters=hyperparameters,
                            resume=resume and run_output.exists(),
                        )
                    except Exception as error:
                        _record_architecture_failure(
                            ledger_path, attempt_number, fold, variant, seed, "training", error
                        )
                        raise
                if (
                    trained.get("status") != "complete"
                    or trained.get("finite_gradients") is not True
                ):
                    failure = ConfirmationError("architecture training did not complete safely")
                    _record_architecture_failure(
                        ledger_path, attempt_number, fold, variant, seed, "training", failure
                    )
                    raise failure
                try:
                    trained, checkpoint_payload, checkpoint_bundle_path = (
                        _completed_training_artifact(
                            config,
                            run_output,
                            fold=fold,
                            seed=seed,
                            variant=variant.value,
                            schedule_sha256=cast(str, pair["training_manifest_sha256"]),
                            target_timesteps=config.training.development_timesteps_per_seed,
                            hyperparameters=asdict(hyperparameters),
                        )
                    )
                    model_state = cast(dict[str, object], checkpoint_payload["model"])
                    critic_input = cast(torch.Tensor, model_state["critic_trunk.0.weight"])
                    model = EntryActorCritic(
                        int(critic_input.shape[1]) - 10,
                        variant,
                        hidden_width=hyperparameters.hidden_width,
                    )
                    model.load_state_dict(model_state, strict=True)
                    model.eval()
                    checkpoint_bundle_sha256 = _sha256(checkpoint_bundle_path)
                except Exception as error:
                    _record_architecture_failure(
                        ledger_path, attempt_number, fold, variant, seed, "checkpoint", error
                    )
                    raise
                try:
                    rows, diagnostics = _evaluate_architecture_attempt(
                        config,
                        loaded,
                        raw_episodes,
                        model,
                        variant,
                        seed,
                    )
                except Exception as error:
                    _record_architecture_failure(
                        ledger_path, attempt_number, fold, variant, seed, "evaluation", error
                    )
                    raise
                report = {
                    "schema_version": 1,
                    "stage": "rl-architecture-validation-report",
                    "plan_sha256": plan_sha256,
                    "variant": variant.value,
                    "seed": seed,
                    "fold": loaded.fold,
                    "training_manifest_path": str(run_output / "manifest.json"),
                    "training_manifest_sha256": _sha256(run_output / "manifest.json"),
                    "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
                    "checkpoint_bundle_path": str(checkpoint_bundle_path),
                    "checkpoint_sha256": _sha256(checkpoint_bundle_path.parent / "checkpoint.pt"),
                    "validation_manifest_path": str(validation_path),
                    "validation_manifest_sha256": _sha256(validation_path),
                    "evaluator_code_path": str(Path(__file__).resolve()),
                    "evaluator_code_sha256": _sha256(Path(__file__)),
                    "partition": "validation",
                    "rows": rows,
                    "learning_curve": trained["metrics"],
                    "policy_diagnostics": diagnostics,
                    "test_accessed": False,
                    "sealed_holdout_accessed": False,
                }
                digest = hashlib.sha256(_canonical_bytes(report)).hexdigest()
                report_path = output / "reports" / f"report-{digest}.json"
                _atomic_no_overwrite(report_path, report)
                reference = {"path": str(report_path), "sha256": digest}
                _atomic_no_overwrite(
                    ledger_path,
                    {
                        "schema_version": 1,
                        "stage": "rl-architecture-ablation-attempt",
                        "status": "complete",
                        "attempt_number": attempt_number,
                        "fold": fold,
                        "variant": variant.value,
                        "seed": seed,
                        "report": reference,
                        "test_accessed": False,
                        "sealed_holdout_accessed": False,
                    },
                )
                references.append(reference)
    result = {
        "schema_version": 1,
        "stage": "rl-architecture-ablation-evidence",
        "partition": "validation",
        "config_sha256": config.digest,
        "optuna_winner_sha256": winner_sha256,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "created_at": plan["created_at"],
        "folds": plan["folds"],
        "schedule_sha256": plan["validation_manifest_sha256"],
        "training_schedule_sha256": plan["training_manifest_sha256"],
        "validation_schedule_sha256": plan["validation_manifest_sha256"],
        "reports": references,
        "sealed_holdout_accessed": False,
    }
    _validated_rows(config, result, winner_sha256, plan)
    _atomic_no_overwrite(output / "ablation.json", result)
    return result


def _validated_rows(
    config: RlConfig,
    evidence: Mapping[str, object],
    winner_sha256: str,
    plan: Mapping[str, object],
) -> tuple[list[dict[str, object]], str]:
    rows = evidence.get("rows")
    reports = evidence.get("reports")
    if reports is not None:
        if not isinstance(reports, list) or not reports:
            raise ConfirmationError("architecture evidence report references are invalid")
        derived_rows: list[object] = []
        report_owners: set[tuple[str, int, int]] = set()
        winner = _load_object(Path(cast(str, plan["winner_path"])), "Optuna winner")
        expected_hyperparameters = asdict(
            PpoHyperparameters.from_search(cast(dict[str, object], winner["parameters"]))
        )
        plan_pairs = {cast(int, pair["fold"]): pair for pair in _validated_plan_pairs(config, plan)}
        for reference in reports:
            if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
                raise ConfirmationError("architecture evidence report reference is invalid")
            report_path = Path(cast(str, reference["path"]))
            if _sha256(report_path) != reference.get("sha256"):
                raise ConfirmationError("architecture evidence report digest mismatch")
            report = _load_object(report_path, "architecture validation report")
            report_rows = report.get("rows")
            owner = (report.get("variant"), report.get("seed"), report.get("fold"))
            if (
                not isinstance(owner[0], str)
                or type(owner[1]) is not int
                or type(owner[2]) is not int
            ):
                raise ConfirmationError("architecture validation report owner is invalid")
            pair = plan_pairs.get(owner[2])
            if pair is None:
                raise ConfirmationError("architecture validation report fold is not frozen")
            checkpoint_bundle_path = Path(cast(str, report["checkpoint_bundle_path"]))
            checkpoint_payload = _validated_training_artifact(
                config,
                Path(cast(str, report["training_manifest_path"])),
                checkpoint_bundle_path,
                fold=cast(int, report["fold"]),
                seed=cast(int, report["seed"]),
                variant=cast(str, report["variant"]),
                schedule_sha256=cast(str, pair["training_manifest_sha256"]),
                target_timesteps=config.training.development_timesteps_per_seed,
                hyperparameters=expected_hyperparameters,
            )
            checkpoint_path = checkpoint_bundle_path.parent / "checkpoint.pt"
            if (
                report.get("schema_version") != 1
                or report.get("stage") != "rl-architecture-validation-report"
                or report.get("partition") != "validation"
                or report.get("test_accessed") is not False
                or report.get("sealed_holdout_accessed") is not False
                or report.get("plan_sha256") != evidence.get("plan_sha256")
                or report.get("validation_manifest_sha256")
                != pair.get("validation_manifest_sha256")
                or report.get("validation_manifest_path") != pair.get("validation_manifest_path")
                or _sha256(Path(cast(str, report["validation_manifest_path"])))
                != report.get("validation_manifest_sha256")
                or _sha256(Path(cast(str, report["training_manifest_path"])))
                != report.get("training_manifest_sha256")
                or _sha256(Path(cast(str, report["checkpoint_bundle_path"])))
                != report.get("checkpoint_bundle_sha256")
                or _sha256(Path(cast(str, report["evaluator_code_path"])))
                != report.get("evaluator_code_sha256")
                or report.get("evaluator_code_sha256") != _sha256(Path(__file__))
                or report.get("checkpoint_sha256") != _sha256(checkpoint_path)
                or checkpoint_payload.get("model") is None
                or not isinstance(report.get("learning_curve"), list)
                or not isinstance(report.get("policy_diagnostics"), dict)
                or not isinstance(report_rows, list)
                or not isinstance(owner[0], str)
                or type(owner[1]) is not int
                or type(owner[2]) is not int
                or cast(tuple[str, int, int], owner) in report_owners
                or any(
                    not isinstance(row, dict)
                    or row.get("variant") != owner[0]
                    or row.get("seed") != owner[1]
                    or row.get("fold") != owner[2]
                    for row in report_rows
                )
            ):
                raise ConfirmationError("architecture validation report provenance mismatch")
            if report.get("learning_curve") != checkpoint_payload.get("metrics"):
                raise ConfirmationError("architecture learning curve checkpoint mismatch")
            _validated_architecture_replay(config, report, checkpoint_payload, report_rows)
            report_owners.add(cast(tuple[str, int, int], owner))
            derived_rows.extend(report_rows)
        expected_owners = {
            (variant, seed, fold)
            for variant in _ABLATIONS
            for seed in config.training.development_seeds
            for fold in cast(list[int], plan["folds"])
        }
        if report_owners != expected_owners:
            raise ConfirmationError("architecture validation reports are incomplete")
        rows = derived_rows
    else:
        raise ConfirmationError("architecture evidence requires evaluator report artifacts")
    schedule = evidence.get("schedule_sha256")
    folds = evidence.get("folds")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("stage") != "rl-architecture-ablation-evidence"
        or evidence.get("partition") != "validation"
        or evidence.get("config_sha256") != config.digest
        or evidence.get("optuna_winner_sha256") != winner_sha256
        or evidence.get("sealed_holdout_accessed") is not False
        or not isinstance(schedule, str)
        or len(schedule) != 64
        or not isinstance(evidence.get("created_at"), str)
        or not isinstance(folds, list)
        or not folds
        or any(type(fold) is not int for fold in folds)
        or any(
            not isinstance(evidence.get(name), str) or len(cast(str, evidence.get(name))) != 64
            for name in ("training_schedule_sha256", "validation_schedule_sha256")
        )
        or not isinstance(rows, list)
        or not rows
    ):
        raise ConfirmationError("architecture evidence must be validation-only")
    validated: list[dict[str, object]] = []
    keys_by_variant: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    seeds_by_variant: dict[str, set[int]] = defaultdict(set)
    for raw in rows:
        if not isinstance(raw, dict):
            raise ConfirmationError("architecture evidence row is invalid")
        if set(raw) != _RAW_ROW_FIELDS:
            raise ConfirmationError("architecture evidence row schema is invalid")
        variant = raw.get("variant")
        fold = raw.get("fold")
        seed = raw.get("seed")
        outcome = raw.get("outcome")
        dimensions = tuple(
            raw.get(name)
            for name in ("ticker", "profile", "regime", "calendar_block", "episode_id")
        )
        if (
            variant not in _ABLATIONS
            or type(fold) is not int
            or type(seed) is not int
            or any(not isinstance(value, str) or not value for value in dimensions)
            or not _WEEK_PATTERN.fullmatch(cast(str, dimensions[3]))
            or dimensions[2] != _calendar_regime(cast(str, dimensions[3]))
            or outcome not in {"PASS", "TIMEOUT", "BLOW"}
            or raw.get("finite") is not True
            or raw.get("action_collapsed") is not False
            or outcome == "BLOW"
        ):
            raise ConfirmationError("architecture evidence contains a failed or invalid run")
        key = (fold, dimensions[3], dimensions[4], dimensions[0], dimensions[1], seed)
        if key in keys_by_variant[cast(str, variant)]:
            raise ConfirmationError("architecture evidence contains duplicate comparisons")
        keys_by_variant[cast(str, variant)].add(key)
        seeds_by_variant[cast(str, variant)].add(seed)
        validated.append(cast(dict[str, object], raw))
    if set(keys_by_variant) != _ABLATIONS:
        raise ConfirmationError("all three preregistered architectures are required")
    reference = keys_by_variant[PolicyVariant.INDEPENDENT_ACTOR.value]
    if any(keys != reference for keys in keys_by_variant.values()):
        raise ConfirmationError("architecture evidence is not identically scheduled")
    required_seeds = set(config.training.development_seeds)
    if any(seeds != required_seeds for seeds in seeds_by_variant.values()):
        raise ConfirmationError("architecture evidence is missing a required development seed")
    expected_contracts = {
        (ticker, profile)
        for ticker in TICKERS
        for profile in (("one_mini",) if ticker == "ZB" else PROFILES)
    }
    observed_contracts = {
        (cast(str, row["ticker"]), cast(str, row["profile"])) for row in validated
    }
    if observed_contracts != expected_contracts:
        raise ConfirmationError("architecture evidence is missing a ticker/profile contract")
    if sorted({cast(int, row["fold"]) for row in validated}) != sorted(folds):
        raise ConfirmationError("architecture evidence fold identity mismatch")
    if (
        folds != plan.get("folds")
        or evidence.get("training_schedule_sha256") != plan.get("training_manifest_sha256")
        or evidence.get("validation_schedule_sha256") != plan.get("validation_manifest_sha256")
        or schedule != plan.get("validation_manifest_sha256")
    ):
        raise ConfirmationError("architecture evidence does not match preregistered schedules")
    return validated, schedule


def _paired_effects(
    candidate: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
    seeds: Sequence[int],
) -> tuple[np.ndarray, dict[str, float], list[str]]:
    dimensions = ("fold", "calendar_block", "episode_id", "ticker", "profile", "seed")
    candidate_by_key = {tuple(row[name] for name in dimensions): row for row in candidate}
    baseline_by_key = {tuple(row[name] for name in dimensions): row for row in baseline}
    if not candidate_by_key or set(candidate_by_key) != set(baseline_by_key):
        raise ConfirmationError("paired pass-rate comparison is non-estimable")
    block_names = sorted({cast(str, key[1]) for key in candidate_by_key})
    if len(block_names) < _MINIMUM_MARKET_BLOCKS:
        raise ConfirmationError("non_estimable_insufficient_blocks")
    differences: dict[tuple[int, str], list[float]] = defaultdict(list)
    for key, row in candidate_by_key.items():
        differences[(cast(int, key[5]), cast(str, key[1]))].append(
            float(row["outcome"] == "PASS") - float(baseline_by_key[key]["outcome"] == "PASS")
        )
    if any((seed, block) not in differences for seed in seeds for block in block_names):
        raise ConfirmationError("non_estimable_missing_block_seed_attempt")
    block_effects = np.asarray(
        [np.mean([np.mean(differences[(seed, block)]) for seed in seeds]) for block in block_names],
        dtype=np.float64,
    )
    seed_effects = {
        str(seed): float(
            np.mean(
                [
                    value
                    for (owner, _block), values in differences.items()
                    if owner == seed
                    for value in values
                ]
            )
        )
        for seed in seeds
    }
    return block_effects, seed_effects, block_names


def _exact_seed_analysis(seed_effects: Mapping[str, float]) -> dict[str, object]:
    effects = np.asarray([seed_effects[str(seed)] for seed in (42, 43, 44, 45, 46)])
    assignments = np.asarray(
        [[1.0 if mask & (1 << index) else -1.0 for index in range(5)] for mask in range(32)]
    )
    distribution = (assignments * effects).mean(axis=1)
    if not np.isfinite(distribution).all() or float(np.ptp(distribution)) == 0.0:
        raise ConfirmationError("non_estimable_degenerate_seed_sign_distribution")
    observed = float(effects.mean())
    upper_null = float(np.quantile(distribution, 0.95, method="inverted_cdf"))
    return {
        "method": "exact_paired_sign_assignments_v1",
        "assignments": 32,
        "seed_effects": dict(seed_effects),
        "point_difference": observed,
        "lower_bound_95": observed - upper_null,
        "assignment_matrix_sha256": hashlib.sha256(assignments.tobytes()).hexdigest(),
    }


def _synchronized_bootstrap(
    block_effects: Sequence[np.ndarray], block_count: int, seed: int
) -> tuple[list[np.ndarray], str]:
    generator = np.random.Generator(np.random.PCG64(seed))
    outputs = [np.empty(_MARKET_BOOTSTRAP_REPLICATES, dtype=np.float64) for _ in block_effects]
    index_digest = hashlib.sha256()
    chunk_size = 2_048
    for start in range(0, _MARKET_BOOTSTRAP_REPLICATES, chunk_size):
        stop = min(start + chunk_size, _MARKET_BOOTSTRAP_REPLICATES)
        indices = generator.integers(
            0, block_count, size=(stop - start, block_count), dtype=np.uint32
        )
        index_digest.update(indices.tobytes())
        for output, effects in zip(outputs, block_effects, strict=True):
            output[start:stop] = effects[indices].mean(axis=1)
    return outputs, index_digest.hexdigest()


def _qualification_gates(
    rows: Sequence[Mapping[str, object]], freeze_payload_sha256: str
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidate = [row for row in rows if row["variant"] == _CANDIDATE.value]
    baseline = [row for row in rows if row["variant"] == PolicyVariant.INDEPENDENT_ACTOR.value]
    groups: list[tuple[str, str | None, str | None]] = [("pooled", None, None)]
    groups.extend(
        ("ticker_profile", cast(str, ticker), cast(str, profile))
        for ticker, profile in sorted({(row["ticker"], row["profile"]) for row in candidate})
    )
    view_effects = []
    for scope, ticker, profile in groups:
        selected_candidate = candidate
        selected_baseline = baseline
        if ticker is not None:
            selected_candidate = [
                row for row in candidate if row["ticker"] == ticker and row["profile"] == profile
            ]
            selected_baseline = [
                row for row in baseline if row["ticker"] == ticker and row["profile"] == profile
            ]
        blocks, seed_effects, block_names = _paired_effects(
            selected_candidate, selected_baseline, (42, 43, 44, 45, 46)
        )
        view_effects.append(
            (scope, ticker, profile, blocks, seed_effects, block_names, len(selected_candidate))
        )
    ordered_blocks = view_effects[0][5]
    if any(value[5] != ordered_blocks for value in view_effects):
        raise ConfirmationError("required views do not share synchronized calendar blocks")
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(f"{freeze_payload_sha256}:market-bootstrap-v1".encode()).digest()[:8],
        "big",
    )
    distributions, index_sha256 = _synchronized_bootstrap(
        [value[3] for value in view_effects], len(ordered_blocks), bootstrap_seed
    )
    gates: list[dict[str, object]] = []
    for (
        scope,
        ticker,
        profile,
        blocks,
        seed_effects,
        block_names,
        matched_attempts,
    ), bootstrap in zip(view_effects, distributions, strict=True):
        point = float(blocks.mean())
        if not np.isfinite(bootstrap).all() or float(np.ptp(bootstrap)) == 0.0:
            raise ConfirmationError("non_estimable_degenerate_market_bootstrap")
        lower = float(np.quantile(bootstrap, 0.05, method="inverted_cdf"))
        gate: dict[str, object] = {
            "scope": scope,
            "ticker": ticker,
            "profile": profile,
            "point_difference": point,
            "paired_lcb_95": lower,
            "matched_attempts": matched_attempts,
            "complete_calendar_blocks": len(block_names),
            "seed_uncertainty": _exact_seed_analysis(seed_effects),
            "passed": point >= 0.0 and lower >= 0.0,
        }
        gates.append(gate)
    if not gates or not all(cast(bool, gate["passed"]) for gate in gates):
        raise ConfirmationError("preregistered architecture failed the negative-transfer gate")
    bootstrap_contract = {
        "method": "synchronized_calendar_week_bootstrap_v1",
        "rng": "numpy.PCG64",
        "rng_seed": bootstrap_seed,
        "replicates": _MARKET_BOOTSTRAP_REPLICATES,
        "quantile": 0.05,
        "quantile_method": "inverted_cdf",
        "block_definition": "fixed_exchange_calendar_week",
        "ordered_source_block_ids": ordered_blocks,
        "index_matrix_sha256": index_sha256,
        "interval_scope": "pointwise_not_simultaneous_familywise",
    }
    return gates, bootstrap_contract


def _architecture_summary(
    rows: Sequence[Mapping[str, object]], evidence: Mapping[str, object]
) -> dict[str, object]:
    candidate = [row for row in rows if row["variant"] == _CANDIDATE.value]
    baseline = [row for row in rows if row["variant"] == PolicyVariant.INDEPENDENT_ACTOR.value]
    seed_rates = {
        seed: float(np.mean([row["outcome"] == "PASS" for row in candidate if row["seed"] == seed]))
        for seed in (42, 43, 44, 45, 46)
    }
    rates = np.asarray(list(seed_rates.values()), dtype=np.float64)
    ordered = np.sort(rates)
    iqm = float(ordered[1:4].mean())
    generator = np.random.default_rng(0)
    sampled = rates[generator.integers(0, 5, size=(_SEED_SUMMARY_REPLICATES, 5))].mean(axis=1)
    key_fields = ("fold", "calendar_block", "episode_id", "ticker", "profile", "seed")
    baseline_by_key = {tuple(row[field] for field in key_fields): row for row in baseline}
    regime_results = []
    for regime in sorted({cast(str, row["regime"]) for row in candidate}):
        differences = [
            float(row["outcome"] == "PASS")
            - float(baseline_by_key[tuple(row[field] for field in key_fields)]["outcome"] == "PASS")
            for row in candidate
            if row["regime"] == regime
        ]
        regime_results.append(
            {
                "regime": regime,
                "matched_attempts": len(differences),
                "point_difference": float(np.mean(differences)),
            }
        )
    curves = []
    for reference in cast(list[dict[str, object]], evidence["reports"]):
        report = _load_object(Path(cast(str, reference["path"])), "architecture report")
        curves.append(
            {
                "variant": report["variant"],
                "seed": report["seed"],
                "learning_curve_sha256": hashlib.sha256(
                    _canonical_bytes(report["learning_curve"])
                ).hexdigest(),
                "policy_diagnostics": report["policy_diagnostics"],
            }
        )
    worst_seed = min(seed_rates, key=lambda seed: (seed_rates[seed], seed))
    return {
        "all_seed_pass_rates": {str(seed): rate for seed, rate in seed_rates.items()},
        "median_pass_rate": float(np.median(rates)),
        "interquartile_mean_pass_rate": iqm,
        "worst_seed": worst_seed,
        "worst_seed_pass_rate": seed_rates[worst_seed],
        "seed_mean_pass_rate_interval_95": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "regime_results": regime_results,
        "collapse_diagnostics_and_learning_curves": curves,
    }


def qualify_architecture(
    config: RlConfig, winner_path: Path, evidence_path: Path, output: Path
) -> dict[str, object]:
    """Freeze the preregistered candidate after validation-only ablation gates."""
    winner, winner_sha256 = _validated_winner(winner_path)
    evidence = _load_object(evidence_path, "architecture evidence")
    plan, plan_sha256 = _validated_plan(config, winner_sha256, evidence)
    rows, schedule_sha256 = _validated_rows(config, evidence, winner_sha256, plan)
    gates, bootstrap_contract = _qualification_gates(rows, plan_sha256)
    repository = Path(__file__).resolve().parents[3]
    spec_path = repository / "docs" / "research" / "2026-07-20-mantisv2-topstep-rl-entry-spec.md"
    statistical_contract = {
        "pair_key": ["fold", "calendar_block", "episode_id", "ticker", "profile", "seed"],
        "development_seeds": list(config.training.development_seeds),
        "required_cells": [
            {"ticker": ticker, "profile": profile}
            for ticker in TICKERS
            for profile in (("one_mini",) if ticker == "ZB" else PROFILES)
        ],
        "minimum_complete_calendar_blocks": _MINIMUM_MARKET_BLOCKS,
        "market_bootstrap_replicates": _MARKET_BOOTSTRAP_REPLICATES,
        "market_rng": "numpy.PCG64",
        "market_rng_derivation": (
            "first_u64_be_sha256(candidate_freeze_payload_sha256 + ':market-bootstrap-v1')"
        ),
        "one_sided_quantile": 0.05,
        "quantile_method": "inverted_cdf",
        "seed_method": "enumerate_all_2_pow_5_paired_sign_assignments",
        "tie_rule": "valid_nondegenerate_point_and_lcb_greater_than_or_equal_to_zero",
        "interval_scope": "pointwise_not_simultaneous_familywise",
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "candidate-freeze-v1",
        "created_at": evidence["created_at"],
        "status": "qualified",
        "selected_variant": _CANDIDATE.value,
        "ablation_variants": sorted(_ABLATIONS),
        "config_sha256": config.digest,
        "accepted_spec": {
            "path": str(spec_path.relative_to(repository)),
            "sha256": _sha256(spec_path),
        },
        "provenance": {
            "source_sha256": config.upstream.source_digest,
            "dependency_lock_sha256": config.upstream.lock_digest,
            "rule_contract_sha256": config.upstream.rule_contract_sha256,
            "downstream_config_sha256": config.upstream.downstream_config_sha256,
            "corpus_manifest_sha256": config.upstream.corpus_manifest_sha256,
            "embedding_manifest_sha256": config.upstream.embedding_manifest_sha256,
            "foundation_manifest_sha256": config.upstream.foundation_manifest_sha256,
            "foundation_weights_sha256": config.upstream.foundation_weights_sha256,
        },
        "optuna_winner_sha256": winner_sha256,
        "optuna_winner_path": str(winner_path.resolve()),
        "optuna_winner": winner,
        "optuna_parameters": winner["parameters"],
        "optuna_validation_sha256": hashlib.sha256(
            _canonical_bytes(winner["validation"])
        ).hexdigest(),
        "architecture_evidence_sha256": _sha256(evidence_path),
        "qualification_report_sha256": _sha256(evidence_path),
        "qualification_report_path": str(evidence_path.resolve()),
        "folds": evidence["folds"],
        "training_schedule_sha256": evidence["training_schedule_sha256"],
        "validation_schedule_sha256": evidence["validation_schedule_sha256"],
        "schedule_sha256": schedule_sha256,
        "experiment_contract": _jsonable(
            {
                "policy": asdict(config.policy),
                "reward": asdict(config.reward),
                "constraint": asdict(config.constraint),
                "execution": asdict(config.execution),
                "fees": asdict(config.fees),
                "exit": asdict(config.exit),
                "sizing": asdict(config.sizing),
            }
        ),
        "action_schema": list(config.policy.actions),
        "observation_schema": {
            "ticker_conditioned": config.policy.ticker_conditioning,
            "embedding_projection_dim": config.policy.embedding_projection_dim,
        },
        "statistical_contract": statistical_contract,
        "statistical_contract_sha256": hashlib.sha256(
            _canonical_bytes(statistical_contract)
        ).hexdigest(),
        "statistical_code_sha256": _sha256(Path(__file__)),
        "statistical_code_path": str(Path(__file__).resolve()),
        "gates": gates,
        "architecture_summary": _architecture_summary(rows, evidence),
        "market_bootstrap": bootstrap_contract,
        "architecture_plan_path": evidence["plan_path"],
        "architecture_plan_sha256": plan_sha256,
        "candidate_freeze_payload_sha256": plan_sha256,
        "development": {
            "seeds": list(config.training.development_seeds),
            "timesteps_per_seed": config.training.development_timesteps_per_seed,
        },
        "confirmation": {
            "seeds": list(config.training.confirmation_seeds),
            "timesteps_per_seed": config.training.confirmation_timesteps_per_seed,
        },
        "maximum_timesteps_per_seed": config.training.maximum_timesteps_per_seed,
        "serving_seed": config.training.serving_seed,
        "serving_seed_frozen_before_test": True,
        "test_accessed": False,
        "sealed_holdout_accessed": False,
        "quality_claim": False,
    }
    data = _canonical_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    candidate_path = output / f"candidate-freeze-{digest}.json"
    _atomic_no_overwrite(candidate_path, payload)
    return {"candidate_path": str(candidate_path), "candidate_sha256": digest, **payload}


def _validated_candidate(config: RlConfig, path: Path) -> tuple[dict[str, object], str]:
    candidate = _load_object(path, "architecture candidate")
    digest = _sha256(path)
    if (
        path.name != f"candidate-freeze-{digest}.json"
        or candidate.get("stage") != "candidate-freeze-v1"
        or candidate.get("status") != "qualified"
        or candidate.get("selected_variant") != _CANDIDATE.value
        or candidate.get("config_sha256") != config.digest
        or candidate.get("test_accessed") is not False
        or candidate.get("sealed_holdout_accessed") is not False
        or candidate.get("serving_seed") != config.training.serving_seed
    ):
        raise ConfirmationError("architecture candidate provenance mismatch")
    for path_field, digest_field in (
        ("optuna_winner_path", "optuna_winner_sha256"),
        ("qualification_report_path", "qualification_report_sha256"),
        ("architecture_plan_path", "architecture_plan_sha256"),
        ("statistical_code_path", "statistical_code_sha256"),
    ):
        reference = candidate.get(path_field)
        expected = candidate.get(digest_field)
        if (
            not isinstance(reference, str)
            or not isinstance(expected, str)
            or _sha256(Path(reference)) != expected
        ):
            raise ConfirmationError("architecture candidate referenced artifact mismatch")
    accepted_spec = candidate.get("accepted_spec")
    repository = Path(__file__).resolve().parents[3]
    if (
        not isinstance(accepted_spec, dict)
        or not isinstance(accepted_spec.get("path"), str)
        or _sha256(repository / cast(str, accepted_spec["path"])) != accepted_spec.get("sha256")
    ):
        raise ConfirmationError("architecture candidate accepted spec mismatch")
    plan = _load_object(Path(cast(str, candidate["architecture_plan_path"])), "architecture plan")
    _validated_plan_pairs(config, plan)
    if _sha256(Path(cast(str, plan["winner_path"]))) != plan.get("winner_sha256"):
        raise ConfirmationError("architecture candidate plan winner mismatch")
    evidence = _load_object(
        Path(cast(str, candidate["qualification_report_path"])),
        "candidate qualification report",
    )
    winner_path = Path(cast(str, candidate["optuna_winner_path"]))
    winner_sha256 = _sha256(winner_path)
    _validated_rows(config, evidence, winner_sha256, plan)
    with tempfile.TemporaryDirectory() as directory:
        recomputed = qualify_architecture(
            config,
            winner_path,
            Path(cast(str, candidate["qualification_report_path"])),
            Path(directory),
        )
    recomputed.pop("candidate_path")
    recomputed.pop("candidate_sha256")
    if recomputed != candidate:
        raise ConfirmationError("architecture candidate decision recomputation mismatch")
    return candidate, digest


def _result_passed(result: Mapping[str, object]) -> bool:
    pass_rate = result.get("pass_rate")
    valid = (
        result.get("status") == "complete"
        and result.get("finite") is True
        and result.get("action_collapsed") is False
        and result.get("blows") == 0
        and result.get("all_gates_passed") is True
        and isinstance(pass_rate, int | float)
        and math.isfinite(float(pass_rate))
        and 0.0 <= float(pass_rate) <= 1.0
        and isinstance(result.get("artifact_sha256"), str)
        and all(
            isinstance(result.get(name), str) and len(cast(str, result.get(name))) == 64
            for name in (
                "training_manifest_sha256",
                "checkpoint_bundle_sha256",
                "validation_report_sha256",
            )
        )
    )
    if not valid:
        return False
    for prefix in ("training_manifest", "checkpoint_bundle", "validation_report"):
        path = result.get(f"{prefix}_path")
        digest = result.get(f"{prefix}_sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            return False
        try:
            if _sha256(Path(path)) != digest:
                return False
        except OSError:
            return False
    try:
        manifest = _load_object(
            Path(cast(str, result["training_manifest_path"])), "training manifest"
        )
        bundle_path = Path(cast(str, result["checkpoint_bundle_path"]))
        bundle = _load_object(bundle_path, "checkpoint bundle")
        checkpoint_path = bundle_path.parent / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        report = _load_object(
            Path(cast(str, result["validation_report_path"])), "validation report"
        )
    except (
        ConfirmationError,
        OSError,
        RuntimeError,
        ValueError,
        IndexError,
        pickle.UnpicklingError,
    ):
        return False
    if not isinstance(checkpoint, dict):
        return False
    checkpoint_metrics = checkpoint.get("metrics")
    if not isinstance(checkpoint_metrics, list):
        return False
    try:
        actual_timesteps = _completed_timesteps(cast(list[dict[str, object]], checkpoint_metrics))
    except (RuntimeError, TypeError, ValueError):
        return False
    endpoint = {
        "model_sha256": _canonical_digest(checkpoint.get("model")),
        "optimizer_sha256": _canonical_digest(checkpoint.get("optimizer")),
        "rng_sha256": _canonical_digest(checkpoint.get("rng")),
        "normalizer_sha256": _canonical_digest(checkpoint.get("normalizers")),
        "controller_sha256": _canonical_digest(checkpoint.get("constraint_controller")),
        "metrics_sha256": _canonical_digest(checkpoint.get("metrics")),
    }
    rows = report.get("rows")
    return bool(
        manifest.get("schema_version") == 2
        and manifest.get("stage") == "rl-train"
        and manifest.get("status") == "complete"
        and manifest.get("sealed_holdout_accessed") is False
        and type(result.get("completed_timesteps")) is int
        and manifest.get("completed_timesteps") == result.get("completed_timesteps")
        and actual_timesteps == result.get("completed_timesteps")
        and type(manifest.get("minimum_target_timesteps")) is int
        and manifest.get("timestep_overshoot")
        == cast(int, manifest["completed_timesteps"])
        - cast(int, manifest["minimum_target_timesteps"])
        and manifest.get("metrics") == checkpoint.get("metrics")
        and bundle.get("schema_version") == 3
        and bundle.get("checkpoint_sha256") == _sha256(checkpoint_path)
        and bundle.get("checkpoint_sha256") == result.get("artifact_sha256")
        and checkpoint.get("schema_version") == 3
        and checkpoint.get("identities") == bundle.get("identities") == manifest.get("identities")
        and checkpoint.get("update") == bundle.get("update")
        and checkpoint.get("episode_cursor") == bundle.get("episode_cursor")
        and result.get("endpoint_bundle") == endpoint
        and report.get("schema_version") == 1
        and report.get("stage") == "rl-campaign-validation"
        and report.get("partition") == "validation"
        and report.get("checkpoint_sha256") == result.get("artifact_sha256")
        and report.get("test_accessed") is False
        and report.get("sealed_holdout_accessed") is False
        and isinstance(rows, list)
        and all(
            isinstance(row, dict)
            and row.get("finite") is True
            and row.get("action_collapsed") is False
            for row in rows
        )
    )


def _lineage_passed(request: ConfirmationRequest, result: Mapping[str, object]) -> bool:
    manifest_path = result.get("training_manifest_path")
    if not isinstance(manifest_path, str):
        return False
    try:
        manifest = _load_object(Path(manifest_path), "training manifest")
    except ConfirmationError:
        return False
    identities = manifest.get("identities")
    completed_timesteps = result.get("completed_timesteps")
    if (
        not isinstance(identities, dict)
        or type(completed_timesteps) is not int
        or completed_timesteps < request.timesteps
        or manifest.get("completed_timesteps") != completed_timesteps
        or manifest.get("minimum_target_timesteps") != request.timesteps
        or manifest.get("timestep_overshoot") != completed_timesteps - request.timesteps
        or identities.get("target_timesteps") != request.timesteps
        or identities.get("campaign_phase") != request.phase
        or identities.get("parent_checkpoint_sha256") != request.parent_artifact_sha256
    ):
        return False
    if request.phase == "development":
        return request.parent_artifact_sha256 is None
    if request.phase == "confirmation":
        milestone = result.get("milestone_2m_sha256")
        if not isinstance(milestone, str) or len(milestone) != 64:
            return False
        if request.parent_artifact_sha256 is not None:
            return (
                milestone == request.parent_artifact_sha256
                and result.get("lineage_parent_sha256") == request.parent_artifact_sha256
            )
        return result.get("lineage_parent_sha256") is None
    if request.phase == "fresh_5m_reference":
        return request.parent_artifact_sha256 is None
    return result.get("lineage_parent_sha256") == request.parent_artifact_sha256


def _seed_aggregate(
    attempts: Sequence[Mapping[str, object]], *, phase: str = "confirmation"
) -> dict[str, object]:
    fold_values = [
        (
            cast(int, cast(dict[str, object], attempt["request"])["seed"]),
            cast(float, cast(dict[str, object], attempt["result"])["pass_rate"]),
        )
        for attempt in attempts
        if cast(dict[str, object], attempt["request"])["phase"] == phase
    ]
    if not fold_values:
        raise ConfirmationError("confirmation seed evidence is missing")
    by_seed: dict[int, list[float]] = defaultdict(list)
    for seed, rate in fold_values:
        by_seed[seed].append(rate)
    values = [(seed, float(np.mean(seed_rates))) for seed, seed_rates in sorted(by_seed.items())]
    rates = np.asarray([rate for _seed, rate in values], dtype=np.float64)
    ordered = np.sort(rates)
    lower = len(ordered) * 0.25
    upper = len(ordered) * 0.75
    weighted = sum(
        max(0.0, min(index + 1.0, upper) - max(float(index), lower)) * float(rate)
        for index, rate in enumerate(ordered)
    )
    generator = np.random.default_rng(0)
    sampled = rates[
        generator.integers(0, len(rates), size=(_SEED_SUMMARY_REPLICATES, len(rates)))
    ].mean(axis=1)
    worst_seed, worst_rate = min(values, key=lambda value: (value[1], value[0]))
    selected = [
        attempt
        for attempt in attempts
        if cast(dict[str, object], attempt["request"])["phase"] == phase
    ]
    grouped: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    per_seed: list[dict[str, object]] = []
    for attempt in selected:
        request = cast(dict[str, object], attempt["request"])
        result = cast(dict[str, object], attempt["result"])
        report = _load_object(
            Path(cast(str, result["validation_report_path"])), "validation report"
        )
        report_rows = cast(list[dict[str, object]], report["rows"])
        per_seed.append(
            {
                "fold": request["fold"],
                "seed": request["seed"],
                "pass_rate": result["pass_rate"],
                "learning_curve_improving": result.get("learning_curve_improving"),
                "policy_diagnostics": {
                    "finite": result["finite"],
                    "action_collapsed": result["action_collapsed"],
                    "blows": result["blows"],
                },
            }
        )
        for row in report_rows:
            grouped[
                (cast(str, row["regime"]), cast(str, row["ticker"]), cast(str, row["profile"]))
            ].append(row["outcome"] == "PASS")
    return {
        "phase": phase,
        "reported_seeds": [seed for seed, _rate_value in values],
        "pass_rates": [rate for _seed, rate in values],
        "median_pass_rate": float(np.median(rates)),
        "interquartile_mean_pass_rate": weighted / (upper - lower),
        "worst_seed": worst_seed,
        "worst_seed_pass_rate": worst_rate,
        "seed_mean_pass_rate_interval_95": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "per_seed": per_seed,
        "regime_cell_summaries": [
            {
                "regime": regime,
                "ticker": ticker,
                "profile": profile,
                "attempts": len(outcomes),
                "pass_rate": sum(outcomes) / len(outcomes),
            }
            for (regime, ticker, profile), outcomes in sorted(grouped.items())
        ],
    }


def _validated_progress_attempts(
    config: RlConfig,
    candidate: Mapping[str, object],
    progress_path: Path,
    candidate_sha256: str,
) -> list[dict[str, object]]:
    progress = _load_object(progress_path, "campaign progress")
    progress_sha256 = _sha256(progress_path)
    references = progress.get("attempt_ledger_sha256")
    if (
        progress_path.name != f"campaign-progress-{progress_sha256}.json"
        or progress.get("schema_version") != 1
        or progress.get("stage") != "rl-seed-campaign-progress-v1"
        or progress.get("status") != "awaiting_validation_budget_decision"
        or progress.get("candidate_sha256") != candidate_sha256
        or progress.get("test_accessed") is not False
        or progress.get("sealed_holdout_accessed") is not False
        or not isinstance(references, list)
        or not references
    ):
        raise ConfirmationError("campaign progress provenance mismatch")
    attempts: list[dict[str, object]] = []
    expected_paths = [
        Path(cast(str, reference.get("path")))
        for reference in references
        if isinstance(reference, dict) and isinstance(reference.get("path"), str)
    ]
    plan = _load_object(Path(cast(str, candidate["architecture_plan_path"])), "candidate plan")
    pairs = _validated_plan_pairs(config, plan)
    pairs_by_fold = {cast(int, pair["fold"]): pair for pair in pairs}
    expected_schedule = [
        (phase, fold, seed, timesteps)
        for phase, seeds, timesteps in (
            (
                "development",
                config.training.development_seeds,
                config.training.development_timesteps_per_seed,
            ),
            (
                "confirmation",
                config.training.confirmation_seeds,
                config.training.confirmation_timesteps_per_seed,
            ),
            (
                "fresh_5m_reference",
                config.training.development_seeds,
                config.training.confirmation_timesteps_per_seed,
            ),
        )
        for fold in sorted(pairs_by_fold)
        for seed in seeds
    ]
    if (
        len(expected_paths) != len(references)
        or len(expected_paths) != len(expected_schedule)
        or any(path.parent != progress_path.parent / "ledger" for path in expected_paths)
    ):
        raise ConfirmationError("campaign progress ledger set mismatch")
    development_artifacts: dict[tuple[int, int], str] = {}
    confirmation_results: dict[tuple[int, int], Mapping[str, object]] = {}
    fresh_results: dict[tuple[int, int], Mapping[str, object]] = {}
    for index, (reference, expected_path) in enumerate(
        zip(references, expected_paths, strict=True), start=1
    ):
        if (
            not isinstance(reference, dict)
            or reference.get("path") != str(expected_path)
            or reference.get("sha256") != _sha256(expected_path)
            or expected_path.name != f"attempt-{index:04d}.json"
        ):
            raise ConfirmationError("campaign progress ledger reference mismatch")
        attempt = _load_object(expected_path, "campaign attempt")
        request = attempt.get("request")
        result = attempt.get("result")
        expected_phase, expected_fold, expected_seed, expected_timesteps = expected_schedule[
            index - 1
        ]
        pair = pairs_by_fold[expected_fold]
        expected_parent = (
            development_artifacts.get((expected_fold, expected_seed))
            if expected_phase == "confirmation"
            else None
        )
        if (
            attempt.get("schema_version") != 1
            or attempt.get("stage") != "rl-seed-confirmation-attempt"
            or attempt.get("attempt_number") != index
            or attempt.get("candidate_sha256") != candidate_sha256
            or attempt.get("test_accessed") is not False
            or attempt.get("sealed_holdout_accessed") is not False
            or not isinstance(request, dict)
            or not isinstance(result, dict)
            or not isinstance(request.get("phase"), str)
            or type(request.get("fold")) is not int
            or type(request.get("seed")) is not int
            or request.get("phase") != expected_phase
            or request.get("fold") != expected_fold
            or request.get("seed") != expected_seed
            or request.get("timesteps") != expected_timesteps
            or request.get("variant") != _CANDIDATE.value
            or request.get("parameters") != candidate.get("optuna_parameters")
            or request.get("candidate_sha256") != candidate_sha256
            or request.get("training_manifest_sha256") != pair.get("training_manifest_sha256")
            or request.get("validation_manifest_sha256") != pair.get("validation_manifest_sha256")
            or request.get("parent_artifact_sha256") != expected_parent
            or request.get("required_milestone_timesteps")
            != config.training.development_timesteps_per_seed
            or type(request.get("resume")) is not bool
            or request.get("output")
            != str(
                progress_path.parent
                / "runs"
                / f"fold-{request['fold']}"
                / cast(str, request["phase"])
                / f"seed-{request['seed']}"
            )
            or not _result_passed(result)
        ):
            raise ConfirmationError("campaign progress attempt provenance mismatch")
        request_object = ConfirmationRequest(
            phase=expected_phase,
            fold=expected_fold,
            training_manifest_sha256=cast(str, pair["training_manifest_sha256"]),
            validation_manifest_sha256=cast(str, pair["validation_manifest_sha256"]),
            seed=expected_seed,
            timesteps=expected_timesteps,
            variant=_CANDIDATE.value,
            parameters=cast(Mapping[str, object], candidate["optuna_parameters"]),
            candidate_sha256=candidate_sha256,
            output=Path(cast(str, request["output"])),
            resume=cast(bool, request["resume"]),
            parent_artifact_sha256=expected_parent,
            required_milestone_timesteps=config.training.development_timesteps_per_seed,
        )
        if not _lineage_passed(request_object, result):
            raise ConfirmationError("campaign progress lineage mismatch")
        key = (expected_fold, expected_seed)
        if expected_phase == "development":
            development_artifacts[key] = cast(str, result["artifact_sha256"])
        elif expected_phase == "confirmation":
            confirmation_results[key] = result
        else:
            fresh_results[key] = result
        attempts.append(attempt)
    endpoint_fields = {
        "model_sha256",
        "optimizer_sha256",
        "rng_sha256",
        "normalizer_sha256",
        "controller_sha256",
        "metrics_sha256",
    }
    for key in fresh_results:
        continued = confirmation_results.get(key, {}).get("endpoint_bundle")
        fresh = fresh_results[key].get("endpoint_bundle")
        if (
            not isinstance(continued, dict)
            or set(continued) != endpoint_fields
            or fresh != continued
        ):
            raise ConfirmationError("campaign progress fresh endpoint parity mismatch")
    return attempts


def _budget_rows_from_progress(
    config: RlConfig,
    candidate: Mapping[str, object],
    progress_path: Path,
    candidate_sha256: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    attempts = _validated_progress_attempts(config, candidate, progress_path, candidate_sha256)
    rows: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    for attempt in attempts:
        request = cast(dict[str, object], attempt["request"])
        result = cast(dict[str, object], attempt["result"])
        if request.get("phase") != "confirmation":
            continue
        for milestone, path_field in (
            ("2m", "milestone_validation_report_path"),
            ("5m", "validation_report_path"),
        ):
            digest_field = path_field.replace("_path", "_sha256")
            path_value = result.get(path_field)
            digest_value = result.get(digest_field)
            if (
                not isinstance(path_value, str)
                or not isinstance(digest_value, str)
                or _sha256(Path(path_value)) != digest_value
            ):
                raise ConfirmationError("campaign budget report reference mismatch")
            report = _load_object(Path(path_value), "campaign budget validation report")
            report_rows = report.get("rows")
            expected_checkpoint = (
                result.get("milestone_2m_sha256")
                if milestone == "2m"
                else result.get("artifact_sha256")
            )
            if milestone == "2m":
                milestone_checkpoint_path = result.get("milestone_checkpoint_path")
                if (
                    not isinstance(milestone_checkpoint_path, str)
                    or result.get("milestone_checkpoint_sha256") != expected_checkpoint
                    or _sha256(Path(milestone_checkpoint_path)) != expected_checkpoint
                ):
                    raise ConfirmationError("campaign milestone checkpoint provenance mismatch")
            if (
                report.get("schema_version") != 1
                or report.get("stage") != "rl-campaign-validation"
                or report.get("partition") != "validation"
                or report.get("seed") != request.get("seed")
                or report.get("fold") != request.get("fold")
                or report.get("validation_manifest_sha256")
                != request.get("validation_manifest_sha256")
                or report.get("checkpoint_sha256") != expected_checkpoint
                or report.get("finite") is not True
                or report.get("action_collapsed") is not False
                or report.get("test_accessed") is not False
                or report.get("sealed_holdout_accessed") is not False
                or not isinstance(report_rows, list)
            ):
                raise ConfirmationError("campaign budget validation provenance mismatch")
            references.append(
                {
                    "attempt_number": attempt["attempt_number"],
                    "milestone": milestone,
                    "path": path_value,
                    "sha256": digest_value,
                }
            )
            rows.extend(
                {"milestone": milestone, **cast(dict[str, object], row)} for row in report_rows
            )
    if not rows:
        raise ConfirmationError("campaign budget reports are missing")
    return rows, references


def decide_continuation(
    config: RlConfig,
    candidate_path: Path,
    evidence_path: Path,
    output: Path,
) -> dict[str, object]:
    """Freeze the validation-only common-budget decision before any 10M update."""
    candidate, candidate_sha256 = _validated_candidate(config, candidate_path)
    evidence = _load_object(evidence_path, "budget comparison evidence")
    report_references = evidence.get("reports")
    progress_path_value = evidence.get("campaign_progress_path")
    progress_sha256 = evidence.get("campaign_progress_sha256")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("stage") != "rl-budget-comparison-evidence"
        or evidence.get("partition") != "validation"
        or evidence.get("candidate_sha256") != candidate_sha256
        or evidence.get("test_accessed") is not False
        or evidence.get("sealed_holdout_accessed") is not False
        or not isinstance(report_references, list)
        or not report_references
        or not isinstance(progress_path_value, str)
        or not isinstance(progress_sha256, str)
        or _sha256(Path(progress_path_value)) != progress_sha256
    ):
        raise ConfirmationError("budget comparison must be validation-only")
    rows, derived_references = _budget_rows_from_progress(
        config, candidate, Path(progress_path_value), candidate_sha256
    )
    if report_references != derived_references:
        raise ConfirmationError("budget comparison reports differ from campaign ledger")
    by_milestone: dict[str, list[dict[str, object]]] = defaultdict(list)
    keys: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    dimensions = ("fold", "seed", "ticker", "profile", "regime", "calendar_block", "episode_id")
    for raw in rows:
        if not isinstance(raw, dict):
            raise ConfirmationError("budget comparison row is invalid")
        milestone = raw.get("milestone")
        key = tuple(raw.get(name) for name in dimensions)
        if (
            milestone not in {"2m", "5m"}
            or type(raw.get("fold")) is not int
            or raw.get("seed") not in config.training.confirmation_seeds
            or any(not isinstance(value, str) or not value for value in key[2:])
            or not _WEEK_PATTERN.fullmatch(cast(str, raw.get("calendar_block")))
            or raw.get("regime") != _calendar_regime(cast(str, raw.get("calendar_block")))
            or raw.get("outcome") not in {"PASS", "TIMEOUT", "BLOW"}
            or raw.get("outcome") == "BLOW"
            or raw.get("finite") is not True
            or raw.get("action_collapsed") is not False
            or key in keys[milestone]
        ):
            raise ConfirmationError("budget comparison row failed a safety gate")
        keys[milestone].add(key)
        by_milestone[milestone].append(raw)
    if set(by_milestone) != {"2m", "5m"} or keys["2m"] != keys["5m"]:
        raise ConfirmationError("budget comparison episode pairing mismatch")
    observed_cells = {(row["ticker"], row["profile"]) for row in by_milestone["5m"]}
    required_cells = {
        (ticker, profile)
        for ticker in TICKERS
        for profile in (("one_mini",) if ticker == "ZB" else PROFILES)
    }
    if observed_cells != required_cells:
        raise ConfirmationError("budget comparison is missing a required cell")
    groups: list[tuple[str, str | None, str | None]] = [("pooled", None, None)]
    groups.extend(("ticker_profile", ticker, profile) for ticker, profile in sorted(required_cells))
    views = []
    for scope, ticker, profile in groups:
        later = by_milestone["5m"]
        earlier = by_milestone["2m"]
        if ticker is not None:
            later = [row for row in later if row["ticker"] == ticker and row["profile"] == profile]
            earlier = [
                row for row in earlier if row["ticker"] == ticker and row["profile"] == profile
            ]
        blocks, _seed_effects, block_names = _paired_effects(
            later, earlier, config.training.confirmation_seeds
        )
        views.append((scope, ticker, profile, blocks, block_names))
    ordered_blocks = views[0][4]
    if any(view[4] != ordered_blocks for view in views):
        raise ConfirmationError("budget views do not share synchronized blocks")
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(
            f"{candidate['candidate_freeze_payload_sha256']}:market-bootstrap-v1".encode()
        ).digest()[:8],
        "big",
    )
    distributions, index_sha256 = _synchronized_bootstrap(
        [view[3] for view in views], len(ordered_blocks), bootstrap_seed
    )
    comparisons = []
    estimable = True
    for (scope, ticker, profile, blocks, block_names), bootstrap in zip(
        views, distributions, strict=True
    ):
        nondegenerate = bool(np.isfinite(bootstrap).all() and float(np.ptp(bootstrap)) > 0.0)
        point = float(blocks.mean())
        lower = (
            float(np.quantile(bootstrap, 0.05, method="inverted_cdf")) if nondegenerate else None
        )
        estimable = estimable and nondegenerate
        comparisons.append(
            {
                "scope": scope,
                "ticker": ticker,
                "profile": profile,
                "point_improvement": point,
                "paired_lcb_95": lower,
                "complete_calendar_blocks": len(block_names),
                "estimable": nondegenerate,
            }
        )
    pooled = comparisons[0]
    authorized = bool(
        estimable
        and cast(float, pooled["point_improvement"]) > 0.0
        and cast(float, pooled["paired_lcb_95"]) >= 0.0
        and all(
            cast(float, comparison["point_improvement"]) >= 0.0 for comparison in comparisons[1:]
        )
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "rl-continuation-decision-v1",
        "status": "authorized" if authorized else "stopped_at_5m",
        "candidate_sha256": candidate_sha256,
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": _sha256(evidence_path),
        "campaign_progress_path": progress_path_value,
        "campaign_progress_sha256": progress_sha256,
        "comparisons": comparisons,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": _MARKET_BOOTSTRAP_REPLICATES,
        "bootstrap_index_matrix_sha256": index_sha256,
        "final_timesteps": (
            config.training.maximum_timesteps_per_seed
            if authorized
            else config.training.confirmation_timesteps_per_seed
        ),
        "test_accessed": False,
        "sealed_holdout_accessed": False,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    decision_path = output / f"continuation-decision-{digest}.json"
    _atomic_no_overwrite(decision_path, payload)
    return {"decision_path": str(decision_path), **payload}


def run_seed_confirmation(
    config: RlConfig,
    candidate_path: Path,
    output: Path,
    *,
    runner: ConfirmationRunner,
    resume: bool = False,
    continuation_timesteps: int | None = None,
    continuation_decision_path: Path | None = None,
    finalize: bool = True,
) -> dict[str, object]:
    """Run and ledger the exact development and confirmation seed schedules."""
    candidate, candidate_sha256 = _validated_candidate(config, candidate_path)
    candidate_plan = _load_object(
        Path(cast(str, candidate["architecture_plan_path"])), "candidate architecture plan"
    )
    candidate_pairs = _validated_plan_pairs(config, candidate_plan)
    folds = [cast(int, pair["fold"]) for pair in candidate_pairs]
    pairs_by_fold = {cast(int, pair["fold"]): pair for pair in candidate_pairs}
    if continuation_timesteps is not None and not (
        config.training.confirmation_timesteps_per_seed
        < continuation_timesteps
        <= config.training.maximum_timesteps_per_seed
    ):
        raise ConfirmationError("continuation must exceed confirmation and not exceed maximum")
    if continuation_decision_path is not None:
        decision = _load_object(continuation_decision_path, "continuation decision")
        decision_sha256 = _sha256(continuation_decision_path)
        expected_status = "authorized" if continuation_timesteps is not None else "stopped_at_5m"
        expected_timesteps = (
            continuation_timesteps or config.training.confirmation_timesteps_per_seed
        )
        if (
            continuation_decision_path.name != f"continuation-decision-{decision_sha256}.json"
            or decision.get("stage") != "rl-continuation-decision-v1"
            or decision.get("status") != expected_status
            or decision.get("candidate_sha256") != candidate_sha256
            or decision.get("final_timesteps") != expected_timesteps
        ):
            raise ConfirmationError("continuation decision provenance mismatch")
        evidence_reference = decision.get("evidence_path")
        if not isinstance(evidence_reference, str) or _sha256(
            Path(evidence_reference)
        ) != decision.get("evidence_sha256"):
            raise ConfirmationError("continuation decision evidence mismatch")
        progress_reference = decision.get("campaign_progress_path")
        if (
            not isinstance(progress_reference, str)
            or Path(progress_reference).resolve().parent != output.resolve()
            or _sha256(Path(progress_reference)) != decision.get("campaign_progress_sha256")
        ):
            raise ConfirmationError("continuation decision progress mismatch")
        progress = _load_object(Path(progress_reference), "campaign progress")
        progress_ledgers = progress.get("attempt_ledger_sha256")
        if (
            progress.get("stage") != "rl-seed-campaign-progress-v1"
            or progress.get("candidate_sha256") != candidate_sha256
            or not isinstance(progress_ledgers, list)
            or any(
                not isinstance(reference, dict)
                or _sha256(Path(cast(str, reference["path"]))) != reference.get("sha256")
                for reference in progress_ledgers
            )
        ):
            raise ConfirmationError("campaign progress provenance mismatch")
        with tempfile.TemporaryDirectory() as directory:
            recomputed = decide_continuation(
                config, candidate_path, Path(evidence_reference), Path(directory)
            )
        recomputed.pop("decision_path")
        if recomputed != decision:
            raise ConfirmationError("continuation decision recomputation mismatch")
    else:
        if continuation_timesteps is not None:
            raise ConfirmationError("continuation requires an immutable authorized decision")
        decision_sha256 = None
    serving_files = sorted(output.glob("serving-freeze-*.json")) if output.is_dir() else []
    if resume and serving_files:
        if len(serving_files) != 1:
            raise ConfirmationError("multiple serving freezes exist")
        serving = _load_object(serving_files[0], "serving freeze")
        if (
            serving_files[0].name != f"serving-freeze-{_sha256(serving_files[0])}.json"
            or serving.get("candidate_sha256") != candidate_sha256
            or serving.get("stage") != "serving-freeze-v1"
            or serving.get("schema_version") != 1
            or serving.get("status") != "complete"
            or serving.get("continuation_decision_sha256") != decision_sha256
            or serving.get("test_accessed") is not False
            or serving.get("sealed_holdout_accessed") is not False
        ):
            raise ConfirmationError("confirmation resume provenance mismatch")
        ledger_references = serving.get("attempt_ledger_sha256")
        if not isinstance(ledger_references, list) or any(
            not isinstance(reference, dict)
            or not isinstance(reference.get("path"), str)
            or _sha256(Path(cast(str, reference["path"]))) != reference.get("sha256")
            for reference in ledger_references
        ):
            raise ConfirmationError("serving freeze ledger provenance mismatch")
        seed_artifacts = serving.get("seed_artifacts")
        if not isinstance(seed_artifacts, list):
            raise ConfirmationError("serving freeze seed artifacts are missing")
        for artifact in seed_artifacts:
            if not isinstance(artifact, dict):
                raise ConfirmationError("serving freeze seed artifact is invalid")
            for prefix in ("training_manifest", "checkpoint_bundle", "validation_report"):
                if _sha256(Path(cast(str, artifact[f"{prefix}_path"]))) != artifact.get(
                    f"{prefix}_sha256"
                ):
                    raise ConfirmationError("serving freeze seed artifact mismatch")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ConfirmationError("confirmation output already exists; resume is required")
    phases: list[tuple[str, Sequence[int], int]] = [
        (
            "development",
            config.training.development_seeds,
            config.training.development_timesteps_per_seed,
        ),
        (
            "confirmation",
            config.training.confirmation_seeds,
            config.training.confirmation_timesteps_per_seed,
        ),
        (
            "fresh_5m_reference",
            config.training.development_seeds,
            config.training.confirmation_timesteps_per_seed,
        ),
    ]
    if continuation_timesteps is not None:
        phases.append(("continuation", config.training.confirmation_seeds, continuation_timesteps))
    expected_attempts = [
        (phase, fold, seed, timesteps)
        for phase, seeds, timesteps in phases
        for fold in folds
        for seed in seeds
    ]
    attempts: list[dict[str, object]] = []
    ledger = output / "ledger"
    completed_keys: set[tuple[str, int, int]] = set()
    development_artifacts: dict[tuple[int, int], str] = {}
    confirmation_artifacts: dict[tuple[int, int], str] = {}
    if resume and ledger.is_dir():
        ledger_paths = sorted(ledger.glob("attempt-*.json"))
        if len(ledger_paths) > len(expected_attempts):
            raise ConfirmationError("confirmation ledger has unexpected attempts")
        for index, path in enumerate(ledger_paths):
            recorded = _load_object(path, "confirmation attempt")
            if (
                path.name != f"attempt-{index + 1:04d}.json"
                or recorded.get("schema_version") != 1
                or recorded.get("stage") != "rl-seed-confirmation-attempt"
                or recorded.get("attempt_number") != index + 1
                or recorded.get("candidate_sha256") != candidate_sha256
                or recorded.get("test_accessed") is not False
                or recorded.get("sealed_holdout_accessed") is not False
            ):
                raise ConfirmationError("confirmation attempt provenance mismatch")
            request = recorded.get("request")
            result = recorded.get("result")
            if not isinstance(request, dict) or not isinstance(result, dict):
                raise ConfirmationError("a recorded seed attempt failed; candidate is ended")
            expected_phase, expected_fold, expected_seed, expected_timesteps = expected_attempts[
                index
            ]
            if (
                request.get("phase") != expected_phase
                or request.get("fold") != expected_fold
                or request.get("seed") != expected_seed
                or request.get("training_manifest_sha256")
                != pairs_by_fold[expected_fold]["training_manifest_sha256"]
                or request.get("validation_manifest_sha256")
                != pairs_by_fold[expected_fold]["validation_manifest_sha256"]
                or request.get("timesteps") != expected_timesteps
                or request.get("variant") != _CANDIDATE.value
                or request.get("candidate_sha256") != candidate_sha256
                or request.get("parameters") != candidate["optuna_parameters"]
                or request.get("required_milestone_timesteps")
                != config.training.development_timesteps_per_seed
                or request.get("output")
                != str(
                    output
                    / "runs"
                    / f"fold-{expected_fold}"
                    / expected_phase
                    / f"seed-{expected_seed}"
                )
                or type(request.get("resume")) is not bool
            ):
                raise ConfirmationError("confirmation ledger schedule mismatch")
            expected_parent = (
                development_artifacts.get((expected_fold, expected_seed))
                if expected_phase == "confirmation"
                else confirmation_artifacts.get((expected_fold, expected_seed))
                if expected_phase == "continuation"
                else None
            )
            recorded_request = ConfirmationRequest(
                phase=expected_phase,
                fold=expected_fold,
                training_manifest_sha256=cast(
                    str, pairs_by_fold[expected_fold]["training_manifest_sha256"]
                ),
                validation_manifest_sha256=cast(
                    str, pairs_by_fold[expected_fold]["validation_manifest_sha256"]
                ),
                seed=expected_seed,
                timesteps=expected_timesteps,
                variant=_CANDIDATE.value,
                parameters=cast(Mapping[str, object], candidate["optuna_parameters"]),
                candidate_sha256=candidate_sha256,
                output=Path(cast(str, request["output"])),
                resume=cast(bool, request["resume"]),
                parent_artifact_sha256=expected_parent,
                required_milestone_timesteps=config.training.development_timesteps_per_seed,
            )
            if request.get("parent_artifact_sha256") != expected_parent or not _lineage_passed(
                recorded_request, result
            ):
                raise ConfirmationError("confirmation resume lineage mismatch")
            if not _result_passed(result):
                raise ConfirmationError("a recorded seed attempt failed; candidate is ended")
            completed_keys.add(
                (
                    cast(str, request["phase"]),
                    cast(int, request["fold"]),
                    cast(int, request["seed"]),
                )
            )
            if request["phase"] == "development":
                development_artifacts[(expected_fold, cast(int, request["seed"]))] = cast(
                    str, result["artifact_sha256"]
                )
            elif request["phase"] == "confirmation":
                confirmation_artifacts[(expected_fold, cast(int, request["seed"]))] = cast(
                    str, result["artifact_sha256"]
                )
            attempts.append(recorded)
    for phase, seeds, timesteps in phases:
        for fold in folds:
            for seed in seeds:
                if (phase, fold, seed) in completed_keys:
                    continue
                request = ConfirmationRequest(
                    phase=phase,
                    fold=fold,
                    training_manifest_sha256=cast(
                        str, pairs_by_fold[fold]["training_manifest_sha256"]
                    ),
                    validation_manifest_sha256=cast(
                        str, pairs_by_fold[fold]["validation_manifest_sha256"]
                    ),
                    seed=seed,
                    timesteps=timesteps,
                    variant=_CANDIDATE.value,
                    parameters=cast(Mapping[str, object], candidate["optuna_parameters"]),
                    candidate_sha256=candidate_sha256,
                    output=output / "runs" / f"fold-{fold}" / phase / f"seed-{seed}",
                    resume=resume,
                    parent_artifact_sha256=(
                        development_artifacts.get((fold, seed))
                        if phase == "confirmation"
                        else confirmation_artifacts.get((fold, seed))
                        if phase == "continuation"
                        else None
                    ),
                    required_milestone_timesteps=(config.training.development_timesteps_per_seed),
                )
                try:
                    raw_result = runner(request)
                    result = dict(raw_result)
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                record = {
                    "schema_version": 1,
                    "stage": "rl-seed-confirmation-attempt",
                    "attempt_number": len(attempts) + 1,
                    "candidate_sha256": candidate_sha256,
                    "request": {**asdict(request), "output": str(request.output)},
                    "result": result,
                    "test_accessed": False,
                    "sealed_holdout_accessed": False,
                }
                _atomic_no_overwrite(ledger / f"attempt-{len(attempts) + 1:04d}.json", record)
                attempts.append(record)
                if not _result_passed(result) or not _lineage_passed(request, result):
                    raise ConfirmationError(f"required {phase} fold {fold} seed {seed} failed")
                if phase == "development":
                    development_artifacts[(fold, seed)] = cast(str, result["artifact_sha256"])
                elif phase == "confirmation":
                    confirmation_artifacts[(fold, seed)] = cast(str, result["artifact_sha256"])
    confirmation_results = {
        (
            cast(int, cast(dict[str, object], attempt["request"])["fold"]),
            cast(int, cast(dict[str, object], attempt["request"])["seed"]),
        ): cast(dict[str, object], attempt["result"])
        for attempt in attempts
        if cast(dict[str, object], attempt["request"])["phase"] == "confirmation"
    }
    fresh_results = {
        (
            cast(int, cast(dict[str, object], attempt["request"])["fold"]),
            cast(int, cast(dict[str, object], attempt["request"])["seed"]),
        ): cast(dict[str, object], attempt["result"])
        for attempt in attempts
        if cast(dict[str, object], attempt["request"])["phase"] == "fresh_5m_reference"
    }
    endpoint_fields = {
        "model_sha256",
        "optimizer_sha256",
        "rng_sha256",
        "normalizer_sha256",
        "controller_sha256",
        "metrics_sha256",
    }
    for fold in folds:
        for seed in config.training.development_seeds:
            continued = confirmation_results.get((fold, seed), {}).get("endpoint_bundle")
            fresh = fresh_results.get((fold, seed), {}).get("endpoint_bundle")
            if (
                not isinstance(continued, dict)
                or not isinstance(fresh, dict)
                or set(continued) != endpoint_fields
                or continued != fresh
            ):
                raise ConfirmationError(
                    f"fresh and continued 5M endpoint mismatch for fold {fold} seed {seed}"
                )
    ledger_files = sorted(ledger.glob("attempt-*.json"))
    if not finalize:
        progress_payload: dict[str, object] = {
            "schema_version": 1,
            "stage": "rl-seed-campaign-progress-v1",
            "status": "awaiting_validation_budget_decision",
            "candidate_sha256": candidate_sha256,
            "attempt_ledger_sha256": [
                {"path": str(path), "sha256": _sha256(path)} for path in ledger_files
            ],
            "test_accessed": False,
            "sealed_holdout_accessed": False,
        }
        progress_sha256 = hashlib.sha256(_canonical_bytes(progress_payload)).hexdigest()
        progress_path = output / f"campaign-progress-{progress_sha256}.json"
        _atomic_no_overwrite(progress_path, progress_payload)
        return {"progress_path": str(progress_path), **progress_payload}
    serving_payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "serving-freeze-v1",
        "status": "complete",
        "candidate_sha256": candidate_sha256,
        "candidate_path": str(candidate_path),
        "selected_variant": _CANDIDATE.value,
        "development_seeds": list(config.training.development_seeds),
        "confirmation_seeds": list(config.training.confirmation_seeds),
        "serving_seed": config.training.serving_seed,
        "serving_seed_frozen_before_test": True,
        "attempts": len(attempts),
        "confirmation_statistics": _seed_aggregate(
            attempts,
            phase="continuation" if continuation_timesteps is not None else "confirmation",
        ),
        "final_timesteps": continuation_timesteps
        or config.training.confirmation_timesteps_per_seed,
        "continuation_decision_sha256": decision_sha256,
        "attempt_ledger_sha256": [
            {"path": str(path), "sha256": _sha256(path)} for path in ledger_files
        ],
        "seed_artifacts": [
            {
                "phase": cast(dict[str, object], attempt["request"])["phase"],
                "fold": cast(dict[str, object], attempt["request"])["fold"],
                "seed": cast(dict[str, object], attempt["request"])["seed"],
                "artifact_sha256": cast(dict[str, object], attempt["result"])["artifact_sha256"],
                "training_manifest_sha256": cast(dict[str, object], attempt["result"])[
                    "training_manifest_sha256"
                ],
                "training_manifest_path": cast(dict[str, object], attempt["result"])[
                    "training_manifest_path"
                ],
                "checkpoint_bundle_sha256": cast(dict[str, object], attempt["result"])[
                    "checkpoint_bundle_sha256"
                ],
                "checkpoint_bundle_path": cast(dict[str, object], attempt["result"])[
                    "checkpoint_bundle_path"
                ],
                "validation_report_sha256": cast(dict[str, object], attempt["result"])[
                    "validation_report_sha256"
                ],
                "validation_report_path": cast(dict[str, object], attempt["result"])[
                    "validation_report_path"
                ],
                "endpoint_bundle": cast(dict[str, object], attempt["result"])["endpoint_bundle"],
            }
            for attempt in attempts
            if cast(dict[str, object], attempt["request"])["phase"]
            == ("continuation" if continuation_timesteps is not None else "confirmation")
        ],
        "test_accessed": False,
        "sealed_holdout_accessed": False,
        "quality_claim": False,
    }
    serving_data = _canonical_bytes(serving_payload)
    serving_sha256 = hashlib.sha256(serving_data).hexdigest()
    serving_path = output / f"serving-freeze-{serving_sha256}.json"
    if resume and serving_files:
        existing = _load_object(serving_files[0], "serving freeze")
        if existing != serving_payload or serving_files[0] != serving_path:
            raise ConfirmationError("existing serving freeze decision or artifacts mismatch")
        return {"serving_freeze_path": str(serving_files[0]), **existing}
    _atomic_no_overwrite(serving_path, serving_payload)
    return {"serving_freeze_path": str(serving_path), **serving_payload}


def _production_campaign_runner(
    config: RlConfig,
    manifest_pairs: Sequence[Mapping[str, object]],
    campaign_root: Path,
) -> ConfirmationRunner:
    contexts: dict[int, tuple[Path, Path, LoadedEpisodes, list[object]]] = {}
    for pair in manifest_pairs:
        fold = cast(int, pair["fold"])
        training_path = Path(cast(str, pair["training_manifest_path"]))
        validation_path = Path(cast(str, pair["validation_manifest_path"]))
        loaded = load_episode_manifest(config, validation_path)
        raw_validation = _load_object(validation_path, "campaign validation schedule")
        raw_episodes = raw_validation.get("episodes")
        if not isinstance(raw_episodes, list) or len(raw_episodes) != len(loaded.episodes):
            raise ConfirmationError("campaign validation episode identities are invalid")
        contexts[fold] = (training_path, validation_path, loaded, raw_episodes)

    def evaluate(
        checkpoint: Path,
        request: ConfirmationRequest,
        hyperparameters: PpoHyperparameters,
        phase: str,
        report_path: Path,
    ) -> tuple[dict[str, int], Path, dict[str, object]]:
        _training_manifest, validation_manifest, loaded, raw_episodes = contexts[request.fold]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ConfirmationError("campaign checkpoint payload is invalid")
        model_state = payload.get("model")
        critic_input = (
            model_state.get("critic_trunk.0.weight") if isinstance(model_state, dict) else None
        )
        if not isinstance(critic_input, torch.Tensor):
            raise ConfirmationError("campaign checkpoint model schema is invalid")
        model = EntryActorCritic(
            int(critic_input.shape[1]) - 10,
            request.variant,
            hidden_width=hyperparameters.hidden_width,
        )
        model.load_state_dict(cast(dict[str, object], model_state), strict=True)
        model.eval()
        rows, diagnostics, outcomes = _deterministic_policy_replay(
            config, loaded, raw_episodes, model, request.seed
        )
        action_collapsed = diagnostics["action_collapse_detected"] is True
        validation_payload = {
            "schema_version": 1,
            "stage": "rl-campaign-validation",
            "partition": "validation",
            "fold": request.fold,
            "seed": request.seed,
            "phase": phase,
            "checkpoint_sha256": _sha256(checkpoint),
            "validation_manifest_sha256": _sha256(validation_manifest),
            "outcomes": outcomes,
            "rows": rows,
            "policy_diagnostics": diagnostics,
            "finite": True,
            "action_collapsed": action_collapsed,
            "test_accessed": False,
            "sealed_holdout_accessed": False,
        }
        _atomic_no_overwrite(report_path, validation_payload)
        return outcomes, report_path, diagnostics

    def runner(request: ConfirmationRequest) -> Mapping[str, object]:
        training_manifest, _validation_manifest, _loaded, _raw_episodes = contexts[request.fold]
        if (
            _sha256(training_manifest) != request.training_manifest_sha256
            or _sha256(_validation_manifest) != request.validation_manifest_sha256
        ):
            raise ConfirmationError("campaign request schedule identity mismatch")
        parent_checkpoint: Path | None = None
        parent_validation_report: Path | None = None
        if request.parent_artifact_sha256 is not None:
            parent_phase = "development" if request.phase == "confirmation" else "confirmation"
            parent_root = (
                campaign_root
                / "runs"
                / f"fold-{request.fold}"
                / parent_phase
                / f"seed-{request.seed}"
            )
            parent_state = _load_object(parent_root / "state.json", "parent training state")
            parent_checkpoint = (
                parent_root / cast(str, parent_state["checkpoint"]) / "checkpoint.pt"
            )
            parent_validation_report = parent_root / "validation.json"
            if _sha256(parent_checkpoint) != request.parent_artifact_sha256:
                raise ConfirmationError("campaign parent checkpoint digest mismatch")
        hyperparameters = PpoHyperparameters.from_search(request.parameters)
        training_output_manifest = request.output / "manifest.json"
        if not training_output_manifest.is_file():
            train_entry_policy(
                config,
                training_manifest,
                request.output,
                seed=request.seed,
                variant=request.variant,
                target_timesteps=request.timesteps,
                hyperparameters=hyperparameters,
                campaign_phase=request.phase,
                parent_checkpoint=parent_checkpoint,
                resume=request.resume and request.output.exists(),
            )
        trained, payload, bundle = _completed_training_artifact(
            config,
            request.output,
            fold=request.fold,
            seed=request.seed,
            variant=request.variant,
            schedule_sha256=request.training_manifest_sha256,
            target_timesteps=request.timesteps,
            hyperparameters=asdict(hyperparameters),
            campaign_phase=request.phase,
            parent_checkpoint_sha256=request.parent_artifact_sha256,
        )
        if trained.get("status") != "complete":
            raise ConfirmationError("campaign training is incomplete")
        state = _load_object(request.output / "state.json", "campaign training state")
        checkpoint = request.output / cast(str, state["checkpoint"]) / "checkpoint.pt"
        validation_report = request.output / "validation.json"
        outcomes, validation_report, diagnostics = evaluate(
            checkpoint,
            request,
            hyperparameters,
            request.phase,
            validation_report,
        )
        collapsed = diagnostics["action_collapse_detected"] is True
        endpoint_bundle = {
            "model_sha256": _canonical_digest(payload["model"]),
            "optimizer_sha256": _canonical_digest(payload["optimizer"]),
            "rng_sha256": _canonical_digest(payload["rng"]),
            "normalizer_sha256": _canonical_digest(payload["normalizers"]),
            "controller_sha256": _canonical_digest(payload["constraint_controller"]),
            "metrics_sha256": _canonical_digest(payload["metrics"]),
        }
        milestone_sha256 = request.parent_artifact_sha256
        milestone_checkpoint_path = parent_checkpoint
        milestone_validation_report: Path | None = None
        if request.phase == "confirmation" and request.parent_artifact_sha256 is not None:
            milestone_validation_report = (
                campaign_root
                / "runs"
                / f"fold-{request.fold}"
                / "development"
                / f"seed-{request.seed}"
                / "validation.json"
            )
        if request.phase == "confirmation" and milestone_sha256 is None:
            for milestone_checkpoint in sorted(
                request.output.glob("checkpoints/update-*/checkpoint.pt")
            ):
                milestone_payload = torch.load(
                    milestone_checkpoint, map_location="cpu", weights_only=True
                )
                milestone_metrics = (
                    milestone_payload.get("metrics")
                    if isinstance(milestone_payload, dict)
                    else None
                )
                if (
                    isinstance(milestone_metrics, list)
                    and _completed_timesteps(cast(list[dict[str, object]], milestone_metrics))
                    >= request.required_milestone_timesteps
                ):
                    milestone_sha256 = _sha256(milestone_checkpoint)
                    milestone_checkpoint_path = milestone_checkpoint
                    milestone_validation_report = request.output / "milestone-2m-validation.json"
                    evaluate(
                        milestone_checkpoint,
                        request,
                        hyperparameters,
                        "milestone_2m",
                        milestone_validation_report,
                    )
                    break
            if milestone_sha256 is None:
                raise ConfirmationError("fresh confirmation run is missing its 2M milestone")
        result: dict[str, object] = {
            "status": "complete",
            "finite": trained.get("finite_gradients") is True,
            "action_collapsed": collapsed is True,
            "blows": outcomes["BLOW"],
            "all_gates_passed": outcomes["BLOW"] == 0 and collapsed is False,
            "pass_rate": outcomes["PASS"] / sum(outcomes.values()),
            "learning_curve_improving": None,
            "artifact_sha256": _sha256(checkpoint),
            "completed_timesteps": trained["completed_timesteps"],
            "milestone_2m_sha256": milestone_sha256 or _sha256(checkpoint),
            "lineage_parent_sha256": request.parent_artifact_sha256,
            "endpoint_bundle": endpoint_bundle,
            "training_manifest_path": str(training_output_manifest),
            "training_manifest_sha256": _sha256(training_output_manifest),
            "checkpoint_bundle_path": str(bundle),
            "checkpoint_bundle_sha256": _sha256(bundle),
            "validation_report_path": str(validation_report),
            "validation_report_sha256": _sha256(validation_report),
        }
        if milestone_validation_report is not None:
            if milestone_checkpoint_path is None:
                raise ConfirmationError("campaign milestone checkpoint is missing")
            result["milestone_checkpoint_path"] = str(milestone_checkpoint_path)
            result["milestone_checkpoint_sha256"] = _sha256(milestone_checkpoint_path)
            result["milestone_validation_report_path"] = str(milestone_validation_report)
            result["milestone_validation_report_sha256"] = _sha256(milestone_validation_report)
            milestone_report = _load_object(
                milestone_validation_report, "campaign milestone validation report"
            )
            milestone_outcomes = milestone_report.get("outcomes")
            if not isinstance(milestone_outcomes, dict):
                raise ConfirmationError("campaign milestone outcomes are invalid")
            milestone_total = sum(cast(int, value) for value in milestone_outcomes.values())
            result["learning_curve_improving"] = (
                outcomes["PASS"] / sum(outcomes.values())
                >= cast(int, milestone_outcomes["PASS"]) / milestone_total
            )
        elif request.phase == "continuation" and parent_validation_report is not None:
            parent_report = _load_object(
                parent_validation_report, "campaign parent validation report"
            )
            parent_outcomes = parent_report.get("outcomes")
            if not isinstance(parent_outcomes, dict):
                raise ConfirmationError("campaign parent outcomes are invalid")
            parent_total = sum(cast(int, value) for value in parent_outcomes.values())
            result["learning_curve_improving"] = (
                outcomes["PASS"] / sum(outcomes.values())
                >= cast(int, parent_outcomes["PASS"]) / parent_total
            )
        return result

    return runner


def run_production_seed_campaign(
    config: RlConfig,
    candidate_path: Path,
    training_manifest: Path | Sequence[Path],
    validation_manifest: Path | Sequence[Path],
    output: Path,
    *,
    resume: bool = False,
    continuation_decision_path: Path | None = None,
) -> dict[str, object]:
    """Run the real trainer and validation replay for the frozen seed campaign."""
    candidate, _candidate_sha256 = _validated_candidate(config, candidate_path)
    plan = _load_object(
        Path(cast(str, candidate["architecture_plan_path"])), "candidate architecture plan"
    )
    frozen_pairs = _validated_plan_pairs(config, plan)
    training_paths = (
        [training_manifest] if isinstance(training_manifest, Path) else list(training_manifest)
    )
    validation_paths = (
        [validation_manifest]
        if isinstance(validation_manifest, Path)
        else list(validation_manifest)
    )
    if not training_paths or len(training_paths) != len(validation_paths):
        raise ConfirmationError("seed campaign requires matched manifest pairs")
    supplied_pairs = []
    for training_path, validation_path in zip(training_paths, validation_paths, strict=True):
        manifests = validate_study_manifests(config, training_path, validation_path)
        supplied_pairs.append(
            {
                "fold": manifests.fold,
                "training_manifest_path": str(training_path.resolve()),
                "training_manifest_sha256": manifests.training_sha256,
                "validation_manifest_path": str(validation_path.resolve()),
                "validation_manifest_sha256": manifests.validation_sha256,
            }
        )
    supplied_pairs.sort(key=lambda pair: cast(int, pair["fold"]))
    if supplied_pairs != frozen_pairs:
        raise ConfirmationError("seed campaign schedules differ from candidate freeze")
    decision_status = None
    if continuation_decision_path is not None:
        decision_status = _load_object(continuation_decision_path, "continuation decision").get(
            "status"
        )
        if decision_status not in {"authorized", "stopped_at_5m"}:
            raise ConfirmationError("continuation decision status is invalid")
    continuation = (
        config.training.maximum_timesteps_per_seed if decision_status == "authorized" else None
    )
    result = run_seed_confirmation(
        config,
        candidate_path,
        output,
        runner=_production_campaign_runner(config, frozen_pairs, output),
        resume=resume,
        continuation_timesteps=continuation,
        continuation_decision_path=continuation_decision_path,
        finalize=continuation_decision_path is not None,
    )
    if continuation_decision_path is None:
        _candidate, candidate_sha256 = _validated_candidate(config, candidate_path)
        _rows, report_references = _budget_rows_from_progress(
            config,
            _candidate,
            Path(cast(str, result["progress_path"])),
            candidate_sha256,
        )
        evidence_payload = {
            "schema_version": 1,
            "stage": "rl-budget-comparison-evidence",
            "partition": "validation",
            "candidate_sha256": candidate_sha256,
            "campaign_progress_path": result["progress_path"],
            "campaign_progress_sha256": hashlib.sha256(
                Path(cast(str, result["progress_path"])).read_bytes()
            ).hexdigest(),
            "reports": report_references,
            "test_accessed": False,
            "sealed_holdout_accessed": False,
        }
        evidence_sha256 = hashlib.sha256(_canonical_bytes(evidence_payload)).hexdigest()
        evidence_path = output / f"budget-comparison-evidence-{evidence_sha256}.json"
        _atomic_no_overwrite(evidence_path, evidence_payload)
        result["budget_evidence_path"] = str(evidence_path)
        result["budget_evidence_sha256"] = evidence_sha256
    return result


__all__ = [
    "ConfirmationError",
    "ConfirmationRequest",
    "decide_continuation",
    "freeze_architecture_plan",
    "qualify_architecture",
    "run_architecture_ablation",
    "run_production_seed_campaign",
    "run_seed_confirmation",
]
