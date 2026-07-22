"""Resumable CPU masked PPO training for the production entry-only policy."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.distributions import Categorical

from mantis_v2.rl_config import RlConfig
from mantis_v2.rl_environment import (
    EnvironmentContractError,
    EnvironmentEpisode,
    TopstepEntryEnvironment,
)
from mantis_v2.rl_policy import (
    PROFILES,
    TICKERS,
    EntryActorCritic,
    PolicyVariant,
    ReturnNormalizers,
)
from mantis_v2.rl_provenance import sha256_file, verify_rl_runtime
from mantis_v2.rl_validation import load_episode_manifest

_POLICY_SCHEMA = "entry-policy-v3"
_LEARNING_RATE = 3e-4
_CLIP_RANGE = 0.2
_VALUE_COEFFICIENT = 0.5
_ENTROPY_COEFFICIENT = 0.01
_MAX_GRAD_NORM = 0.5


@dataclass(frozen=True)
class PpoHyperparameters:
    """Optimizer/model knobs accepted by the immutable Optuna v1 contract."""

    learning_rate: float = _LEARNING_RATE
    rollout_length: int | None = None
    batch_size: int = 512
    gae_lambda: float = 1.0
    clip_range: float = _CLIP_RANGE
    entropy_coefficient: float = _ENTROPY_COEFFICIENT
    value_loss_coefficient: float = _VALUE_COEFFICIENT
    max_grad_norm: float = _MAX_GRAD_NORM
    hidden_width: int = 64

    def __post_init__(self) -> None:
        if self.rollout_length is None:
            if (
                self.learning_rate <= 0.0
                or self.batch_size < 1
                or not 0.0 <= self.gae_lambda <= 1.0
                or self.clip_range <= 0.0
                or self.entropy_coefficient < 0.0
                or self.value_loss_coefficient <= 0.0
                or self.max_grad_norm <= 0.0
                or self.hidden_width < 1
            ):
                raise ProductionTrainingError("PPO hyperparameters are invalid")
            return
        accepted = (
            1e-4 <= self.learning_rate <= 1e-3
            and self.rollout_length in {14, 28, 56}
            and self.batch_size in {256, 512, 1024}
            and 0.90 <= self.gae_lambda <= 0.99
            and self.clip_range in {0.1, 0.2, 0.3}
            and 1e-4 <= self.entropy_coefficient <= 0.03
            and self.value_loss_coefficient in {0.25, 0.5, 1.0}
            and self.max_grad_norm in {0.3, 0.5, 1.0}
            and self.hidden_width in {64, 128, 256}
        )
        if not accepted:
            raise ProductionTrainingError("PPO search parameter values are outside v1")

    @classmethod
    def from_search(cls, raw: Mapping[str, object]) -> PpoHyperparameters:
        expected = {
            "learning_rate",
            "rollout_length",
            "batch_size",
            "gae_lambda",
            "clip_range",
            "entropy_coefficient",
            "value_loss_coefficient",
            "max_grad_norm",
            "hidden_width",
        }
        if set(raw) != expected:
            raise ProductionTrainingError("PPO search parameter keys are invalid")
        if any(type(value) is bool for value in raw.values()):
            raise ProductionTrainingError("PPO search parameter values are invalid")
        try:
            result = cls(
                learning_rate=float(cast(float, raw["learning_rate"])),
                rollout_length=int(cast(int, raw["rollout_length"])),
                batch_size=int(cast(int, raw["batch_size"])),
                gae_lambda=float(cast(float, raw["gae_lambda"])),
                clip_range=float(cast(float, raw["clip_range"])),
                entropy_coefficient=float(cast(float, raw["entropy_coefficient"])),
                value_loss_coefficient=float(cast(float, raw["value_loss_coefficient"])),
                max_grad_norm=float(cast(float, raw["max_grad_norm"])),
                hidden_width=int(cast(int, raw["hidden_width"])),
            )
        except (TypeError, ValueError) as exc:
            raise ProductionTrainingError("PPO search parameter values are invalid") from exc
        return result


class ProductionTrainingError(RuntimeError):
    """Raised when production entry training cannot prove its contract."""


class TrainingControl(StrEnum):
    """Preregistered learning controls run through the production trainer seam."""

    NORMAL = "normal"
    ZERO_REWARD = "zero_reward"
    SHUFFLED_REWARD = "shuffled_reward"
    MASK_DISABLED = "mask_disabled"


@dataclass(frozen=True)
class TrainingEpisodes:
    episodes: tuple[EnvironmentEpisode, ...]
    manifest_sha256: str
    config_sha256: str
    artifact_sha256: str
    partition: str
    fold: int
    schedule_seed: int
    collection_sha256: str = ""
    runtime_identities: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _Transition:
    observation: np.ndarray
    ticker: int
    profile: int
    action: int
    old_log_probability: float
    old_value: float
    old_cost_value: float
    mask: np.ndarray
    reward: float
    cost: float
    terminal: bool


@dataclass
class ConstraintController:
    """Projected episodic Lagrange multiplier with durable raw cost statistics."""

    kind: str
    cost_limit: float
    cost_gamma: float
    lambda_init: float
    lambda_lr: float
    lambda_max: float
    minimum_cushion_role: str
    lambda_value: float
    update_count: int = 0
    episode_count: int = 0
    cost_sum: float = 0.0
    saturation_count: int = 0

    @classmethod
    def from_config(cls, config: RlConfig) -> ConstraintController:
        constraint = config.constraint
        return cls(
            kind=constraint.kind,
            cost_limit=constraint.cost_limit,
            cost_gamma=constraint.cost_gamma,
            lambda_init=constraint.lambda_init,
            lambda_lr=constraint.lambda_lr,
            lambda_max=constraint.lambda_max,
            minimum_cushion_role=constraint.minimum_cushion_role,
            lambda_value=constraint.lambda_init,
        )

    def update(self, episode_costs: Sequence[float]) -> dict[str, float | int | bool]:
        if not episode_costs or any(cost not in {0.0, 1.0} for cost in episode_costs):
            raise ProductionTrainingError("dual update requires raw binary episode costs")
        mean_cost = float(np.mean(episode_costs))
        before = self.lambda_value
        self.lambda_value = min(
            self.lambda_max,
            max(0.0, before + self.lambda_lr * (mean_cost - self.cost_limit)),
        )
        self.update_count += 1
        self.episode_count += len(episode_costs)
        self.cost_sum += float(sum(episode_costs))
        saturated = self.lambda_value == self.lambda_max and mean_cost > self.cost_limit
        self.saturation_count += int(saturated)
        return {
            "lambda_before": before,
            "lambda_after": self.lambda_value,
            "mean_episode_cost": mean_cost,
            "cost_limit": self.cost_limit,
            "constraint_violation": mean_cost - self.cost_limit,
            "dual_update_count": self.update_count,
            "dual_saturated": saturated,
            "dual_saturation_count": self.saturation_count,
        }

    def state_dict(self) -> dict[str, float | int | str]:
        return {
            "kind": self.kind,
            "cost_limit": self.cost_limit,
            "cost_gamma": self.cost_gamma,
            "lambda_init": self.lambda_init,
            "lambda_lr": self.lambda_lr,
            "lambda_max": self.lambda_max,
            "minimum_cushion_role": self.minimum_cushion_role,
            "lambda_value": self.lambda_value,
            "update_count": self.update_count,
            "episode_count": self.episode_count,
            "cost_sum": self.cost_sum,
            "saturation_count": self.saturation_count,
        }

    def load_state_dict(self, raw: Mapping[str, object]) -> None:
        expected = self.state_dict()
        if set(raw) != set(expected):
            raise ProductionTrainingError("constraint controller state is invalid")
        for key in (
            "kind",
            "cost_limit",
            "cost_gamma",
            "lambda_init",
            "lambda_lr",
            "lambda_max",
            "minimum_cushion_role",
        ):
            if raw[key] != expected[key]:
                raise ProductionTrainingError("constraint controller config mismatch")
        try:
            self.lambda_value = float(cast(float, raw["lambda_value"]))
            self.update_count = int(cast(int, raw["update_count"]))
            self.episode_count = int(cast(int, raw["episode_count"]))
            self.cost_sum = float(cast(float, raw["cost_sum"]))
            self.saturation_count = int(cast(int, raw["saturation_count"]))
        except (TypeError, ValueError) as exc:
            raise ProductionTrainingError("constraint controller state is invalid") from exc
        if not 0.0 <= self.lambda_value <= self.lambda_max:
            raise ProductionTrainingError("constraint controller lambda is out of bounds")


def _episode_collection_digest(episodes: Sequence[EnvironmentEpisode]) -> str:
    payload = []
    for episode in episodes:
        candidates = sum(bar.candidate is not None for bar in episode.bars)
        payload.append(
            {
                "ticker": episode.ticker,
                "profile": episode.profile,
                "bars": len(episode.bars),
                "candidates": candidates,
                "start": episode.bars[0].timestamp.isoformat(),
                "end": episode.bars[-1].timestamp.isoformat(),
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_training_episodes(
    config: RlConfig,
    manifest: Path,
    repository_root: Path | None = None,
) -> TrainingEpisodes:
    """Verify runtime provenance and load the complete declared schedule."""
    root = repository_root.resolve() if repository_root else Path(__file__).resolve().parents[3]
    runtime_identities = verify_rl_runtime(config, root)
    try:
        raw = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionTrainingError("training schedule manifest is invalid") from exc
    loaded = load_episode_manifest(config, manifest, root)
    schedule_seed = raw.get("seed") if isinstance(raw, dict) else None
    raw_episodes = raw.get("episodes") if isinstance(raw, dict) else None
    if type(schedule_seed) is not int:
        raise ProductionTrainingError("training schedule seed is invalid")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != len(loaded.episodes):
        raise ProductionTrainingError("training schedule episode collection mismatch")
    manifest_sha256 = sha256_file(manifest)
    if loaded.manifest_sha256 != manifest_sha256:
        raise ProductionTrainingError("training schedule content digest mismatch")
    return TrainingEpisodes(
        loaded.episodes,
        manifest_sha256,
        config.digest,
        config.upstream.embedding_manifest_sha256,
        loaded.partition,
        loaded.fold,
        schedule_seed,
        _episode_collection_digest(loaded.episodes),
        runtime_identities,
    )


def _validate_contract(config: RlConfig, episodes: TrainingEpisodes) -> None:
    if config.run.profile != "production" or config.run.device != "cpu":
        raise ProductionTrainingError("RL production training requires the CPU production profile")
    if episodes.partition != "training":
        raise ProductionTrainingError("RL trainer accepts only a declared training partition")
    if not episodes.episodes:
        raise ProductionTrainingError("training partition contains no episodes")
    if episodes.config_sha256 != config.digest:
        raise ProductionTrainingError("training schedule config identity mismatch")
    if episodes.artifact_sha256 != config.upstream.embedding_manifest_sha256:
        raise ProductionTrainingError("training schedule artifact identity mismatch")
    expected_collection = _episode_collection_digest(episodes.episodes)
    if episodes.collection_sha256 and episodes.collection_sha256 != expected_collection:
        raise ProductionTrainingError("training schedule episode collection mismatch")
    if config.reward.gamma != 1.0 or config.reward.potential_shaping:
        raise ProductionTrainingError("training reward must be unshaped with gamma 1")
    if config.policy.actions != ("skip", "enter") or config.sizing.actor_controls_size:
        raise ProductionTrainingError("entry actor authority must remain skip/enter only")
    expected_cells = {
        (ticker, profile)
        for ticker in TICKERS
        for profile in (("one_mini",) if ticker == "ZB" else PROFILES)
    }
    cell_counts = Counter((episode.ticker, episode.profile) for episode in episodes.episodes)
    ticker_counts = Counter(episode.ticker for episode in episodes.episodes)
    cells_balanced = (
        bool(cell_counts) and max(cell_counts.values()) - min(cell_counts.values()) <= 1
    )
    tickers_balanced = (
        bool(ticker_counts) and max(ticker_counts.values()) - min(ticker_counts.values()) <= 1
    )
    profiles_balanced = all(
        abs(cell_counts[(ticker, "one_mini")] - cell_counts[(ticker, "ten_micros")]) <= 1
        for ticker in TICKERS
        if ticker != "ZB"
    )
    if (
        set(cell_counts) != expected_cells
        or not profiles_balanced
        or not (cells_balanced or tickers_balanced)
    ):
        raise ProductionTrainingError("training schedule ticker/profile matrix is incomplete")


def _identities(
    config: RlConfig,
    episodes: TrainingEpisodes,
    seed: int,
    variant: PolicyVariant,
    target_updates: int | None,
    training_control: TrainingControl,
    *,
    hyperparameters: PpoHyperparameters | None = None,
    target_timesteps: int | None = None,
    search_trial_number: int | None = None,
    search_seed_index: int | None = None,
    campaign_phase: str | None = None,
    parent_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    source = episodes.runtime_identities.get("source", {})
    lock = episodes.runtime_identities.get("lock", {})
    if not isinstance(source, Mapping) or not isinstance(lock, Mapping):
        raise ProductionTrainingError("runtime source or lock identity is invalid")
    selected_hyperparameters = hyperparameters or PpoHyperparameters(
        batch_size=config.training.minibatch_size
    )
    minimum_timesteps = target_timesteps or config.training.development_timesteps_per_seed
    return {
        "policy_schema": _POLICY_SCHEMA,
        "completion_mode": "production_timesteps" if target_updates is None else "bounded_updates",
        "target_updates": target_updates,
        "target_timesteps": minimum_timesteps if target_updates is None else None,
        "search_trial_number": search_trial_number,
        "search_seed_index": search_seed_index,
        "campaign_phase": campaign_phase,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "ppo_hyperparameters": asdict(selected_hyperparameters),
        "training_control": training_control.value,
        "variant": variant.value,
        "fold": episodes.fold,
        "seed": seed,
        "schedule_seed": episodes.schedule_seed,
        "schedule_sha256": episodes.manifest_sha256,
        "episode_collection_sha256": episodes.collection_sha256
        or _episode_collection_digest(episodes.episodes),
        "episode_count": len(episodes.episodes),
        "partition": episodes.partition,
        "config_sha256": config.digest,
        "source_revision": source.get("revision", "test-fixture"),
        "source_dirty": source.get("dirty", False),
        "source_sha256": source.get("sha256", config.upstream.source_digest),
        "dependency_lock_sha256": lock.get("sha256", config.upstream.lock_digest),
        "artifact_sha256": episodes.artifact_sha256,
        "corpus_sha256": config.upstream.corpus_manifest_sha256,
        "foundation_manifest_sha256": config.upstream.foundation_manifest_sha256,
        "foundation_weights_sha256": config.upstream.foundation_weights_sha256,
        "reward_sha256": hashlib.sha256(
            json.dumps(asdict(config.reward), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "constraint_sha256": hashlib.sha256(
            json.dumps(asdict(config.constraint), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rule_sha256": config.rule_digest,
        "fee_sha256": config.fee_digest,
    }


def derive_search_training_seed(trial_number: int, seed_index: int) -> int:
    """Return the accepted SHA256-U31 Optuna training seed."""
    if not 0 <= trial_number < 30 or seed_index not in {0, 1, 2}:
        raise ProductionTrainingError("search trial or seed index is invalid")
    message = f"mantis-v2-topstep-100k-optuna-v1:{trial_number}:{seed_index}".encode()
    value = int.from_bytes(hashlib.sha256(message).digest()[:8], "big")
    return 1 + value % 2_147_483_646


def _masked_distribution(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    if masks.dtype is not torch.bool or masks.shape != logits.shape:
        raise ProductionTrainingError("policy action mask shape or type is invalid")
    if not bool(masks.any(dim=1).all()):
        raise ProductionTrainingError("policy action mask rejects every action")
    masked = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
    return Categorical(logits=masked)


def _terminal_signal(status: str) -> tuple[float, float]:
    if status == "PASS":
        return 1.0, 0.0
    if status == "BLOW":
        return 0.0, 1.0
    if status == "TIMEOUT":
        return 0.0, 0.0
    raise ProductionTrainingError(f"unsupported terminal status: {status}")


def _rollout(
    config: RlConfig,
    model: EntryActorCritic,
    episode: EnvironmentEpisode,
    training_control: TrainingControl = TrainingControl.NORMAL,
) -> tuple[list[_Transition], dict[str, object]]:
    transitions: list[_Transition] = []
    environment = TopstepEntryEnvironment(config, episode)
    observation, _ = environment.reset()
    terminated = truncated = False
    final_status = "ACTIVE"
    bars_replayed = 0
    initial_account = environment.account_state
    minimum_mll_cushion = float(cast(float, initial_account["equity"])) - float(
        cast(float, initial_account["mll_floor"])
    )
    while not (terminated or truncated):
        ticker = TICKERS.index(episode.ticker)
        profile = PROFILES.index(episode.profile)
        environment_mask = environment.action_mask().copy()
        mask = (
            np.ones(2, dtype=np.bool_)
            if training_control is TrainingControl.MASK_DISABLED
            else environment_mask
        )
        if bool(mask[1]):
            vector = torch.from_numpy(observation.vector.copy()).unsqueeze(0)
            ticker_tensor = torch.tensor([ticker], dtype=torch.long)
            profile_tensor = torch.tensor([profile], dtype=torch.long)
            with torch.no_grad():
                logits, values, cost_values = model.forward_with_cost(
                    vector, ticker_tensor, profile_tensor
                )
                distribution = _masked_distribution(logits, torch.from_numpy(mask).unsqueeze(0))
                action_tensor = distribution.sample()  # type: ignore[no-untyped-call]
                log_probability = float(
                    distribution.log_prob(action_tensor).item()  # type: ignore[no-untyped-call]
                )
            action = int(action_tensor.item())
            if not environment_mask[action]:
                raise ProductionTrainingError(
                    "mask-disabled control produced an illegal environment action"
                )
            transitions.append(
                _Transition(
                    observation.vector.copy(),
                    ticker,
                    profile,
                    action,
                    log_probability,
                    float(values.item()),
                    float(cost_values.item()),
                    mask,
                    0.0,
                    0.0,
                    False,
                )
            )
        else:
            action = 0
        try:
            observation, _reward, terminated, truncated, info = environment.step(action)
        except EnvironmentContractError as exc:
            raise ProductionTrainingError("policy submitted an illegal environment action") from exc
        final_status = str(info["status"])
        bars_replayed += 1
        account = environment.account_state
        minimum_mll_cushion = min(
            minimum_mll_cushion,
            float(cast(float, account["equity"])) - float(cast(float, account["mll_floor"])),
        )
    if not transitions:
        raise ProductionTrainingError("training episode contains no legal entry decisions")
    reward, cost = _terminal_signal(final_status)
    last = transitions[-1]
    transitions[-1] = _Transition(
        last.observation,
        last.ticker,
        last.profile,
        last.action,
        last.old_log_probability,
        last.old_value,
        last.old_cost_value,
        last.mask,
        reward,
        cost,
        True,
    )
    return transitions, {
        "ticker": episode.ticker,
        "profile": episode.profile,
        "outcome": final_status,
        "bars_replayed": bars_replayed,
        "policy_decisions": len(transitions),
        "minimum_mll_cushion": minimum_mll_cushion,
    }


def _apply_training_control(
    episode_transitions: Sequence[list[_Transition]],
    training_control: TrainingControl,
) -> list[_Transition]:
    if training_control in {TrainingControl.NORMAL, TrainingControl.MASK_DISABLED}:
        return [transition for episode in episode_transitions for transition in episode]
    terminal_rewards = [episode[-1].reward for episode in episode_transitions]
    if training_control is TrainingControl.ZERO_REWARD:
        assigned_rewards = [0.0 for _ in terminal_rewards]
    else:
        assigned_rewards = terminal_rewards[1:] + terminal_rewards[:1]
    controlled: list[_Transition] = []
    for episode, reward in zip(episode_transitions, assigned_rewards, strict=True):
        controlled.extend((*episode[:-1], replace(episode[-1], reward=reward)))
    return controlled


def _reward_returns(transitions: Sequence[_Transition]) -> np.ndarray:
    values = np.zeros(len(transitions), dtype=np.float32)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        if transitions[index].terminal:
            running = transitions[index].reward
        values[index] = running
    return values


def _cost_returns(transitions: Sequence[_Transition]) -> np.ndarray:
    values = np.zeros(len(transitions), dtype=np.float32)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        if transitions[index].terminal:
            running = transitions[index].cost
        values[index] = running
    return values


def _generalized_advantages(
    transitions: Sequence[_Transition],
    *,
    signal: str,
    old_value: str,
    gae_lambda: float,
) -> np.ndarray:
    """Compute gamma-one GAE while resetting recursion at every complete episode."""
    if signal not in {"reward", "cost"} or old_value not in {
        "old_value",
        "old_cost_value",
    }:
        raise ProductionTrainingError("GAE signal or value field is invalid")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ProductionTrainingError("GAE lambda must be between zero and one")
    advantages = np.zeros(len(transitions), dtype=np.float32)
    next_advantage = 0.0
    next_value = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        terminal = transition.terminal
        continuation = 0.0 if terminal else 1.0
        current_value = float(getattr(transition, old_value))
        delta = float(getattr(transition, signal)) + continuation * next_value - current_value
        next_advantage = delta + continuation * gae_lambda * next_advantage
        advantages[index] = next_advantage
        next_value = current_value
        if terminal:
            next_advantage = delta
            next_value = current_value
    return advantages


def _normalized_reward_advantages(
    transitions: Sequence[_Transition],
    normalizers: ReturnNormalizers,
    gae_lambda: float,
) -> np.ndarray:
    """Compute reward GAE wholly in each ticker's frozen normalized-return basis."""
    if not 0.0 <= gae_lambda <= 1.0:
        raise ProductionTrainingError("GAE lambda must be between zero and one")
    advantages = np.zeros(len(transitions), dtype=np.float32)
    next_advantage = 0.0
    next_value = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        continuation = 0.0 if transition.terminal else 1.0
        reward = normalizers.reward_signal(
            TICKERS[transition.ticker], transition.reward, terminal=transition.terminal
        )
        delta = reward + continuation * next_value - transition.old_value
        next_advantage = delta + continuation * gae_lambda * next_advantage
        advantages[index] = next_advantage
        next_value = transition.old_value
        if transition.terminal:
            next_advantage = delta
            next_value = transition.old_value
    return advantages


def _search_rollout_episodes(
    episodes: Sequence[EnvironmentEpisode],
    rollout_length: int,
    *,
    seed: int,
    update: int,
) -> tuple[EnvironmentEpisode, ...]:
    """Build complete canonical 14-episode ticker/profile supercycles."""
    if rollout_length not in {14, 28, 56} or update < 0:
        raise ProductionTrainingError("search rollout length or update is invalid")
    grouped: dict[tuple[str, str], list[EnvironmentEpisode]] = {}
    for episode in episodes:
        grouped.setdefault((episode.ticker, episode.profile), []).append(episode)
    supercycles = rollout_length // 14
    required = {
        (ticker, profile): supercycles
        for ticker in TICKERS
        if ticker != "ZB"
        for profile in PROFILES
    }
    required[("ZB", "one_mini")] = 2 * supercycles
    selected: dict[tuple[str, str], list[EnvironmentEpisode]] = {}
    for key, count in required.items():
        available = grouped.get(key, [])
        if len(available) < count:
            raise ProductionTrainingError(
                "search rollout lacks complete ticker/profile supercycles"
            )
        start = (update * count) % len(available)
        selected[key] = [available[(start + index) % len(available)] for index in range(count)]
        if len({id(episode) for episode in selected[key]}) != count:
            raise ProductionTrainingError("search rollout repeats an episode")
    cursors = {key: 0 for key in selected}
    generator = random.Random((seed << 16) ^ update)
    rollout: list[EnvironmentEpisode] = []
    for _supercycle in range(supercycles):
        assignments = {}
        for ticker in TICKERS:
            if ticker == "ZB":
                continue
            profiles = list(PROFILES)
            generator.shuffle(profiles)
            assignments[ticker] = profiles
        for wave in range(2):
            for ticker in TICKERS:
                profile = "one_mini" if ticker == "ZB" else assignments[ticker][wave]
                key = (ticker, profile)
                rollout.append(selected[key][cursors[key]])
                cursors[key] += 1
    return tuple(rollout)


def _actor_weights(tickers: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    weights = torch.zeros(len(tickers), dtype=torch.float32)
    totals: dict[str, float] = {}
    present = [index for index in range(len(TICKERS)) if bool((tickers == index).any())]
    for ticker_index in present:
        owned = tickers == ticker_index
        weights[owned] = 1.0 / (len(present) * int(owned.sum()))
        totals[TICKERS[ticker_index]] = float(weights[owned].sum())
    return weights, totals


def _balanced_minibatches(
    tickers: torch.Tensor, minibatch_size: int
) -> tuple[list[torch.Tensor], dict[str, int]]:
    present = [index for index in range(len(TICKERS)) if bool((tickers == index).any())]
    if not present:
        raise ProductionTrainingError("PPO rollout has no ticker transitions")
    per_ticker = max(1, minibatch_size // len(present))
    owned = {index: torch.nonzero(tickers == index, as_tuple=False).flatten() for index in present}
    batch_count = max(int(np.ceil(len(indices) / per_ticker)) for indices in owned.values())
    shuffled = {index: indices[torch.randperm(len(indices))] for index, indices in owned.items()}
    batches = []
    for batch in range(batch_count):
        selected = []
        for index in present:
            indices = shuffled[index]
            positions = torch.arange(batch * per_ticker, (batch + 1) * per_ticker, dtype=torch.long)
            selected.append(indices[positions.remainder(len(indices))])
        combined = torch.cat(selected)
        batches.append(combined[torch.randperm(len(combined))])
    samples = {TICKERS[index]: batch_count * per_ticker for index in present}
    return batches, samples


def _lagrangian_actor_advantages(
    reward_advantages: torch.Tensor,
    cost_advantages: torch.Tensor,
    dual_lambda: float,
) -> torch.Tensor:
    if dual_lambda < 0.0 or reward_advantages.shape != cost_advantages.shape:
        raise ProductionTrainingError("invalid constrained actor advantages")
    return reward_advantages - dual_lambda * cost_advantages


def _reward_targets_and_advantages(
    normalizers: ReturnNormalizers,
    tickers: torch.Tensor,
    raw_returns: np.ndarray,
    old_values: torch.Tensor,
    raw_actor_advantages: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep rollout values and actor returns in one frozen normalization basis."""
    critic_targets = torch.empty(len(raw_returns), dtype=torch.float32)
    actor_advantages = torch.empty(len(raw_returns), dtype=torch.float32)
    ticker_array = tickers.numpy()
    for ticker_index, ticker in enumerate(TICKERS):
        owned = np.flatnonzero(ticker_array == ticker_index)
        if not len(owned):
            continue
        owned_tensor = torch.from_numpy(owned)
        owned_returns = torch.as_tensor(raw_returns[owned], dtype=torch.float32)
        rollout_basis_returns = normalizers.normalize(ticker, owned_returns)
        owned_advantages = (
            rollout_basis_returns - old_values[owned_tensor]
            if raw_actor_advantages is None
            else torch.as_tensor(raw_actor_advantages[owned], dtype=torch.float32)
        )
        owned_advantages = owned_advantages - owned_advantages.mean()
        reward_std = owned_advantages.std(unbiased=False)
        if len(owned_advantages) > 1 and float(reward_std) > 1e-8:
            owned_advantages = owned_advantages / reward_std
        actor_advantages[owned_tensor] = owned_advantages
        normalizers.update(ticker, raw_returns[owned])
        critic_targets[owned_tensor] = normalizers.normalize(ticker, owned_returns)
    return critic_targets, actor_advantages


def _ppo_update(
    model: EntryActorCritic,
    optimizer: torch.optim.Optimizer,
    normalizers: ReturnNormalizers,
    controller: ConstraintController,
    transitions: Sequence[_Transition],
    *,
    episode_costs: Sequence[float],
    epochs: int,
    minibatch_size: int,
    gae_lambda: float = 1.0,
    clip_range: float = _CLIP_RANGE,
    entropy_coefficient: float = _ENTROPY_COEFFICIENT,
    value_loss_coefficient: float = _VALUE_COEFFICIENT,
    max_grad_norm: float = _MAX_GRAD_NORM,
) -> dict[str, object]:
    if epochs < 2 or minibatch_size < 1:
        raise ProductionTrainingError("PPO requires at least two epochs and a positive minibatch")
    observations = torch.from_numpy(np.stack([item.observation for item in transitions]))
    tickers = torch.tensor([item.ticker for item in transitions], dtype=torch.long)
    profiles = torch.tensor([item.profile for item in transitions], dtype=torch.long)
    actions = torch.tensor([item.action for item in transitions], dtype=torch.long)
    old_log_probabilities = torch.tensor(
        [item.old_log_probability for item in transitions], dtype=torch.float32
    )
    old_values = torch.tensor([item.old_value for item in transitions], dtype=torch.float32)
    masks = torch.from_numpy(np.stack([item.mask for item in transitions]))
    raw_reward_returns = _reward_returns(transitions)
    raw_cost_returns = _cost_returns(transitions)
    reward_gae = _normalized_reward_advantages(transitions, normalizers, gae_lambda)
    cost_gae = _generalized_advantages(
        transitions, signal="cost", old_value="old_cost_value", gae_lambda=gae_lambda
    )
    normalized_reward_returns, reward_advantages = _reward_targets_and_advantages(
        normalizers, tickers, raw_reward_returns, old_values, reward_gae
    )
    cost_returns = torch.from_numpy(raw_cost_returns)
    cost_advantages = torch.empty(len(transitions), dtype=torch.float32)
    for ticker_index in range(len(TICKERS)):
        owned = np.flatnonzero(np.asarray([item.ticker for item in transitions]) == ticker_index)
        if len(owned):
            owned_tensor = torch.from_numpy(owned)
            owned_cost_advantages = torch.as_tensor(cost_gae[owned], dtype=torch.float32)
            owned_cost_advantages = owned_cost_advantages - owned_cost_advantages.mean()
            cost_std = owned_cost_advantages.std(unbiased=False)
            if len(owned_cost_advantages) > 1 and float(cost_std) > 1e-8:
                owned_cost_advantages = owned_cost_advantages / cost_std
            cost_advantages[owned_tensor] = owned_cost_advantages
    dual_lambda = controller.lambda_value
    advantages = _lagrangian_actor_advantages(reward_advantages, cost_advantages, dual_lambda)
    _actor_weight_values, actor_weight_totals = _actor_weights(tickers)
    losses: list[float] = []
    actor_losses: list[float] = []
    value_losses: list[float] = []
    cost_value_losses: list[float] = []
    entropies: list[float] = []
    ratios: list[torch.Tensor] = []
    clipped: list[torch.Tensor] = []
    balanced_samples: Counter[str] = Counter()
    for _epoch in range(epochs):
        minibatches, epoch_samples = _balanced_minibatches(tickers, minibatch_size)
        balanced_samples.update(epoch_samples)
        for index in minibatches:
            logits, values, cost_values = model.forward_with_cost(
                observations[index], tickers[index], profiles[index]
            )
            distribution = _masked_distribution(logits, masks[index])
            current = distribution.log_prob(actions[index])  # type: ignore[no-untyped-call]
            ratio = torch.exp(current - old_log_probabilities[index])
            unclipped = ratio * advantages[index]
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            surrogate = torch.minimum(unclipped, clipped_ratio * advantages[index])
            actor_loss = -surrogate.mean()
            value_loss = (values - normalized_reward_returns[index]).square().mean()
            cost_value_loss = (cost_values - cost_returns[index]).square().mean()
            entropy = distribution.entropy().mean()  # type: ignore[no-untyped-call]
            loss = (
                actor_loss
                + value_loss_coefficient * value_loss
                + value_loss_coefficient * cost_value_loss
                - entropy_coefficient * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            if not bool(torch.isfinite(loss)) or not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                raise ProductionTrainingError("policy update produced non-finite gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                raise ProductionTrainingError("policy update produced non-finite parameters")
            losses.append(float(loss.detach()))
            actor_losses.append(float(actor_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            cost_value_losses.append(float(cost_value_loss.detach()))
            entropies.append(float(entropy.detach()))
            ratios.append(ratio.detach())
            clipped.append(((ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)).detach())
    all_ratios = torch.cat(ratios)
    all_clipped = torch.cat(clipped)
    constraint_metrics = controller.update(episode_costs)
    return {
        "loss": float(np.mean(losses)),
        "actor_loss": float(np.sum(actor_losses) / epochs),
        "value_loss": float(np.mean(value_losses)),
        "cost_value_loss": float(np.mean(cost_value_losses)),
        "entropy": float(np.mean(entropies)),
        "finite_gradients": True,
        "ppo_epochs": epochs,
        "minibatch_size": minibatch_size,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "entropy_coefficient": entropy_coefficient,
        "value_loss_coefficient": value_loss_coefficient,
        "max_grad_norm": max_grad_norm,
        "ratio_non_unit_fraction": float((torch.abs(all_ratios - 1.0) > 1e-6).float().mean()),
        "ratio_max_deviation": float(torch.abs(all_ratios - 1.0).max()),
        "clip_fraction": float(all_clipped.float().mean()),
        "actor_weight_totals": actor_weight_totals,
        "balanced_actor_samples": dict(sorted(balanced_samples.items())),
        "lambda_used": dual_lambda,
        "reward_advantage_mean": float(reward_advantages.mean()),
        "cost_advantage_mean": float(cost_advantages.mean()),
        "cost_advantage_std": float(cost_advantages.std(unbiased=False)),
        "cost_target_min": float(cost_returns.min()),
        "cost_target_max": float(cost_returns.max()),
        **constraint_metrics,
    }


def _canonical_digest(value: object) -> str:
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, list | tuple):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _transition_audit(transitions: Sequence[_Transition]) -> dict[str, object]:
    return {
        "rewards": [item.reward for item in transitions],
        "costs": [item.cost for item in transitions],
        "terminals": [item.terminal for item in transitions],
        "nonterminal_reward_sum": sum(item.reward for item in transitions if not item.terminal),
        "nonterminal_cost_sum": sum(item.cost for item in transitions if not item.terminal),
    }


def _policy_diagnostics(
    model: EntryActorCritic,
    episode_transitions: Sequence[list[_Transition]],
    episode_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    probes = [episode[0] for episode in episode_transitions]
    observations = torch.from_numpy(np.stack([item.observation for item in probes]))
    tickers = torch.tensor([item.ticker for item in probes], dtype=torch.long)
    profiles = torch.tensor([item.profile for item in probes], dtype=torch.long)
    masks = torch.from_numpy(np.stack([item.mask for item in probes]))
    with torch.no_grad():
        logits, _ = model(observations, tickers, profiles)
        distribution = _masked_distribution(logits, masks)
        enter_probabilities = distribution.probs[:, 1]
        actions = distribution.probs.argmax(dim=1)
        entropies = distribution.entropy()  # type: ignore[no-untyped-call]
    by_outcome: dict[str, list[float]] = {}
    for probability, metrics in zip(enter_probabilities, episode_metrics, strict=True):
        by_outcome.setdefault(str(metrics["outcome"]), []).append(float(probability))
    means = {name: float(np.mean(values)) for name, values in sorted(by_outcome.items())}
    enter_rate = float((actions == 1).float().mean())
    mean_entropy = float(entropies.mean())
    action_count = len(set(actions.tolist()))
    pass_probability = means.get("PASS")
    blow_probability = means.get("BLOW")
    return {
        "enter_rate": enter_rate,
        "mean_entropy": mean_entropy,
        "deterministic_action_count": action_count,
        "mean_enter_probability_by_outcome": means,
        "pass_minus_blow_enter_probability": (
            pass_probability - blow_probability
            if pass_probability is not None and blow_probability is not None
            else None
        ),
        "reject_all_detected": enter_rate == 0.0,
        "take_all_detected": enter_rate == 1.0,
        "action_collapse_detected": action_count == 1 or mean_entropy < 0.05,
        "invalid_actions_observed": 0,
        "episode_enter_probabilities": [float(value) for value in enter_probabilities],
    }


def _checkpoint_payload(
    model: EntryActorCritic,
    optimizer: torch.optim.Optimizer,
    normalizers: ReturnNormalizers,
    controller: ConstraintController,
    identities: Mapping[str, object],
    update: int,
    metrics: Sequence[Mapping[str, object]],
    evidence: Mapping[str, object],
) -> dict[str, object]:
    numpy_state = cast(tuple[str, np.ndarray[Any, Any], int, int, float], np.random.get_state())
    return {
        "schema_version": 3,
        "identities": dict(identities),
        "update": update,
        "episode_cursor": 0,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizers": normalizers.state_dict(),
        "constraint_controller": controller.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": numpy_state[0],
                "keys": torch.from_numpy(numpy_state[1].copy()),
                "position": numpy_state[2],
                "has_gauss": numpy_state[3],
                "cached_gaussian": numpy_state[4],
            },
            "torch": torch.get_rng_state(),
        },
        "metrics": list(metrics),
        "last_evidence": dict(evidence),
    }


def _publish_checkpoint(output: Path, update: int, payload: Mapping[str, object]) -> str:
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    target = checkpoints / f"update-{update:06d}"
    if target.exists():
        raise ProductionTrainingError("immutable checkpoint update collision")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=checkpoints))
    try:
        checkpoint = temporary / "checkpoint.pt"
        torch.save(dict(payload), checkpoint)
        bundle = {
            "schema_version": 3,
            "update": update,
            "identities": payload["identities"],
            "episode_cursor": payload["episode_cursor"],
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        _atomic_json(temporary / "bundle.json", bundle)
        os.replace(temporary, target)
        descriptor = os.open(checkpoints, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(target.relative_to(output))


def _load_bundle(path: Path, identities: Mapping[str, object]) -> dict[str, object]:
    try:
        bundle = json.loads((path / "bundle.json").read_text())
        checkpoint = path / "checkpoint.pt"
        if bundle.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise ProductionTrainingError("resume checkpoint identity mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise ProductionTrainingError("resume checkpoint bundle is invalid") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 3
        or bundle.get("schema_version") != 3
        or payload.get("identities") != dict(identities)
        or bundle.get("identities") != dict(identities)
        or bundle.get("update") != payload.get("update")
        or bundle.get("episode_cursor") != payload.get("episode_cursor")
    ):
        raise ProductionTrainingError("resume provenance mismatch")
    return cast(dict[str, object], payload)


def _recover_checkpoint(output: Path, identities: Mapping[str, object]) -> dict[str, object]:
    state_path = output / "state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ProductionTrainingError("resume state is invalid") from exc
        if not isinstance(state, dict) or state.get("identities") != dict(identities):
            raise ProductionTrainingError("resume provenance mismatch")
    checkpoints = output / "checkpoints"
    candidates = sorted(checkpoints.glob("update-*")) if checkpoints.is_dir() else []
    if not candidates:
        raise ProductionTrainingError("resume has no immutable checkpoint bundle")
    payloads = [_load_bundle(path, identities) for path in candidates]
    updates = [payload.get("update") for payload in payloads]
    first_update = cast(int, updates[0])
    expected_first = 1 if identities.get("parent_checkpoint_sha256") is None else first_update
    if updates != list(range(expected_first, expected_first + len(payloads))):
        raise ProductionTrainingError("resume checkpoint sequence is not contiguous")
    return payloads[-1]


def _restore_checkpoint(
    payload: Mapping[str, object],
    model: EntryActorCritic,
    optimizer: torch.optim.Optimizer,
    normalizers: ReturnNormalizers,
    controller: ConstraintController,
) -> tuple[int, list[dict[str, object]], dict[str, object]]:
    try:
        model.load_state_dict(cast(dict[str, Any], payload["model"]))
        optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer"]))
        normalizers.load_state_dict(cast(dict[str, object], payload["normalizers"]))
        controller.load_state_dict(cast(dict[str, object], payload["constraint_controller"]))
        rng = cast(dict[str, Any], payload["rng"])
        random.setstate(rng["python"])
        numpy_state = rng["numpy"]
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                numpy_state["keys"].numpy(),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(rng["torch"])
        metrics = cast(list[dict[str, object]], payload["metrics"])
        evidence = cast(dict[str, object], payload["last_evidence"])
        update = int(cast(int, payload["update"]))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ProductionTrainingError("resume runtime state is invalid") from exc
    return update, metrics, evidence


def _completed_timesteps(metrics: Sequence[Mapping[str, object]]) -> int:
    total = 0
    for metric in metrics:
        rollouts = metric.get("rollouts")
        decisions = rollouts.get("policy_decisions") if isinstance(rollouts, dict) else None
        if type(decisions) is not int or decisions < 1:
            raise ProductionTrainingError("checkpoint rollout timestep metrics are invalid")
        total += decisions
    return total


def _state_payload(
    identities: Mapping[str, object],
    completed: int,
    target_updates: int | None,
    completed_timesteps: int,
    minimum_timesteps: int,
    checkpoint: str,
    complete: bool,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": "complete" if complete else "incomplete",
        "completed_updates": completed,
        "target_updates": target_updates,
        "completed_timesteps": completed_timesteps,
        "minimum_target_timesteps": minimum_timesteps,
        "checkpoint": checkpoint,
        "episode_cursor": 0,
        "identities": dict(identities),
    }


def _train_loaded_policy(
    config: RlConfig,
    episodes: TrainingEpisodes,
    output: Path,
    *,
    seed: int,
    variant: PolicyVariant | str = PolicyVariant.SHARED_TICKER_VALUE,
    target_updates: int | None = None,
    target_timesteps: int | None = None,
    maximum_updates_this_run: int | None = None,
    resume: bool = False,
    publish: bool = True,
    training_control: TrainingControl | str = TrainingControl.NORMAL,
    hyperparameters: PpoHyperparameters | None = None,
    search_trial_number: int | None = None,
    search_seed_index: int | None = None,
    campaign_phase: str | None = None,
    parent_checkpoint: Path | None = None,
) -> dict[str, object]:
    """Internal seam for an already verified, complete schedule collection."""
    _validate_contract(config, episodes)
    selected_variant = PolicyVariant(variant)
    selected_control = TrainingControl(training_control)
    selected_hyperparameters = hyperparameters or PpoHyperparameters(
        batch_size=config.training.minibatch_size
    )
    search_authorized = search_trial_number is not None or search_seed_index is not None
    if search_authorized:
        if (
            search_trial_number is None
            or search_seed_index is None
            or seed != derive_search_training_seed(search_trial_number, search_seed_index)
            or target_timesteps != config.training.search_timesteps_per_seed
            or selected_hyperparameters.rollout_length not in {14, 28, 56}
        ):
            raise ProductionTrainingError("search seed or budget authorization mismatch")
    elif campaign_phase is None and seed not in config.training.development_seeds:
        raise ProductionTrainingError("seed is not declared in development_seeds")
    elif campaign_phase is not None:
        expected: tuple[Sequence[int], int, bool]
        if campaign_phase == "development":
            expected = (
                config.training.development_seeds,
                config.training.development_timesteps_per_seed,
                False,
            )
        elif campaign_phase == "confirmation":
            expected = (
                config.training.confirmation_seeds,
                config.training.confirmation_timesteps_per_seed,
                seed in config.training.development_seeds,
            )
        elif campaign_phase == "fresh_5m_reference":
            expected = (
                config.training.development_seeds,
                config.training.confirmation_timesteps_per_seed,
                False,
            )
        elif campaign_phase == "continuation":
            expected = (
                config.training.confirmation_seeds,
                config.training.maximum_timesteps_per_seed,
                True,
            )
        else:
            raise ProductionTrainingError("unknown seed campaign phase")
        seeds, required_timesteps, requires_parent = expected
        if (
            seed not in seeds
            or (target_updates is None and target_timesteps != required_timesteps)
            or (target_updates is not None and target_timesteps is not None)
            or (parent_checkpoint is not None) is not requires_parent
        ):
            raise ProductionTrainingError("seed campaign authorization mismatch")
    if target_updates is not None and target_timesteps is not None:
        raise ProductionTrainingError("training target must use updates or timesteps, not both")
    if target_updates is not None and target_updates < 1:
        raise ProductionTrainingError("target updates must be positive")
    if target_timesteps is not None and target_timesteps < 1:
        raise ProductionTrainingError("target timesteps must be positive")
    if maximum_updates_this_run is not None and maximum_updates_this_run < 1:
        raise ProductionTrainingError("invocation update budget must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    first_environment = TopstepEntryEnvironment(config, episodes.episodes[0])
    first_observation, _ = first_environment.reset()
    model = EntryActorCritic(
        len(first_observation.vector),
        selected_variant,
        hidden_width=selected_hyperparameters.hidden_width,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=selected_hyperparameters.learning_rate)
    normalizers = ReturnNormalizers(TICKERS)
    controller = ConstraintController.from_config(config)
    parent_checkpoint_sha256 = sha256_file(parent_checkpoint) if parent_checkpoint else None
    identities = _identities(
        config,
        episodes,
        seed,
        selected_variant,
        target_updates,
        selected_control,
        hyperparameters=selected_hyperparameters,
        target_timesteps=target_timesteps,
        search_trial_number=search_trial_number,
        search_seed_index=search_seed_index,
        campaign_phase=campaign_phase,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )
    completed = 0
    metrics: list[dict[str, object]] = []
    last_evidence: dict[str, object] = {}
    if publish:
        if resume:
            if not output.is_dir() or (output / "manifest.json").exists():
                raise ProductionTrainingError("resume requires an unpublished run directory")
            payload = _recover_checkpoint(output, identities)
            completed, metrics, last_evidence = _restore_checkpoint(
                payload, model, optimizer, normalizers, controller
            )
        else:
            try:
                output.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise ProductionTrainingError("training run identity already exists") from exc
    elif resume:
        raise ProductionTrainingError("in-memory training cannot resume")
    if parent_checkpoint is not None and not resume:
        try:
            parent_payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(parent_payload, dict) or not isinstance(
                parent_payload.get("identities"), dict
            ):
                raise ProductionTrainingError("parent checkpoint payload is invalid")
            parent_identities = dict(cast(dict[str, object], parent_payload["identities"]))
            child_fixed = dict(identities)
            for key in (
                "target_updates",
                "target_timesteps",
                "campaign_phase",
                "parent_checkpoint_sha256",
            ):
                parent_identities.pop(key, None)
                child_fixed.pop(key, None)
            if parent_identities != child_fixed:
                raise ProductionTrainingError("parent checkpoint lineage mismatch")
            completed, metrics, last_evidence = _restore_checkpoint(
                parent_payload, model, optimizer, normalizers, controller
            )
            inherited_payload = _checkpoint_payload(
                model,
                optimizer,
                normalizers,
                controller,
                identities,
                completed,
                metrics,
                last_evidence,
            )
            _publish_checkpoint(output, completed, inherited_payload)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if isinstance(exc, ProductionTrainingError):
                raise
            raise ProductionTrainingError("parent checkpoint is invalid") from exc
    completed_timesteps = _completed_timesteps(metrics) if metrics else 0

    def is_complete() -> bool:
        if target_updates is not None:
            return completed >= target_updates
        minimum = target_timesteps or config.training.development_timesteps_per_seed
        return completed_timesteps >= minimum

    if publish and resume:
        pointer = f"checkpoints/update-{completed:06d}"
        _atomic_json(
            output / "state.json",
            _state_payload(
                identities,
                completed,
                target_updates,
                completed_timesteps,
                target_timesteps or config.training.development_timesteps_per_seed,
                pointer,
                is_complete(),
            ),
        )
    updates_this_run = 0
    while not is_complete() and (
        maximum_updates_this_run is None or updates_this_run < maximum_updates_this_run
    ):
        ticker_counts: Counter[str] = Counter()
        profile_counts: Counter[str] = Counter()
        outcomes: Counter[str] = Counter()
        episode_metrics: list[dict[str, object]] = []
        episode_transition_groups: list[list[_Transition]] = []
        rollout_episodes = (
            episodes.episodes
            if selected_hyperparameters.rollout_length is None
            else _search_rollout_episodes(
                episodes.episodes,
                selected_hyperparameters.rollout_length,
                seed=seed,
                update=completed,
            )
        )
        for episode in rollout_episodes:
            transitions, episode_rollout = _rollout(config, model, episode, selected_control)
            episode_transition_groups.append(transitions)
            ticker_counts[episode.ticker] += 1
            profile_counts[episode.profile] += 1
            outcomes[str(episode_rollout["outcome"])] += 1
            episode_metrics.append(episode_rollout)
        all_transitions = _apply_training_control(episode_transition_groups, selected_control)
        update_metrics = _ppo_update(
            model,
            optimizer,
            normalizers,
            controller,
            all_transitions,
            episode_costs=[item.cost for item in all_transitions if item.terminal],
            epochs=config.training.ppo_epochs,
            minibatch_size=selected_hyperparameters.batch_size,
            gae_lambda=selected_hyperparameters.gae_lambda,
            clip_range=selected_hyperparameters.clip_range,
            entropy_coefficient=selected_hyperparameters.entropy_coefficient,
            value_loss_coefficient=selected_hyperparameters.value_loss_coefficient,
            max_grad_norm=selected_hyperparameters.max_grad_norm,
        )
        decision_counts = Counter(TICKERS[item.ticker] for item in all_transitions)
        rollouts: dict[str, object] = {
            "ticker_counts": dict(sorted(ticker_counts.items())),
            "profile_counts": dict(sorted(profile_counts.items())),
            "ticker_max_minus_min": max(ticker_counts.values()) - min(ticker_counts.values()),
            "transition_counts": dict(sorted(decision_counts.items())),
            "outcomes": dict(sorted(outcomes.items())),
            "bars_replayed": sum(cast(int, item["bars_replayed"]) for item in episode_metrics),
            "policy_decisions": len(all_transitions),
            "episodes": episode_metrics,
        }
        policy_diagnostics = _policy_diagnostics(model, episode_transition_groups, episode_metrics)
        last_evidence = {
            "rollouts": rollouts,
            "transition_audit": _transition_audit(all_transitions),
            "policy_diagnostics": policy_diagnostics,
        }
        completed += 1
        updates_this_run += 1
        metrics.append({"update": completed, **update_metrics, "rollouts": rollouts})
        completed_timesteps = _completed_timesteps(metrics)
        if publish:
            payload = _checkpoint_payload(
                model,
                optimizer,
                normalizers,
                controller,
                identities,
                completed,
                metrics,
                last_evidence,
            )
            pointer = _publish_checkpoint(output, completed, payload)
            _atomic_json(
                output / "state.json",
                _state_payload(
                    identities,
                    completed,
                    target_updates,
                    completed_timesteps,
                    target_timesteps or config.training.development_timesteps_per_seed,
                    pointer,
                    is_complete(),
                ),
            )
    status = "complete" if is_complete() else "incomplete"
    if completed and not last_evidence:
        raise ProductionTrainingError("checkpoint is missing last rollout evidence")
    result: dict[str, object] = {
        "stage": "rl-train",
        "algorithm": "episodic_masked_ppo_lagrangian",
        "status": status,
        "fold": episodes.fold,
        "seed": seed,
        "variant": selected_variant.value,
        "training_control": selected_control.value,
        "completed_updates": completed,
        "target_updates": target_updates,
        "completed_timesteps": completed_timesteps,
        "minimum_target_timesteps": target_timesteps
        or config.training.development_timesteps_per_seed,
        "timestep_overshoot": (
            max(
                0,
                completed_timesteps
                - (target_timesteps or config.training.development_timesteps_per_seed),
            )
            if target_updates is None
            else None
        ),
        "ppo_hyperparameters": asdict(selected_hyperparameters),
        "identities": identities,
        "gamma": config.reward.gamma,
        "reward_shaping": config.reward.potential_shaping,
        **last_evidence,
        "finite_gradients": bool(metrics)
        and all(bool(item["finite_gradients"]) for item in metrics),
        "policy_sha256": _canonical_digest(model.state_dict()),
        "optimizer_sha256": _canonical_digest(optimizer.state_dict()),
        "normalizers": normalizers.state_dict(),
        "constraint": asdict(config.constraint),
        "constraint_controller": controller.state_dict(),
        "metrics": metrics,
    }
    if publish and status == "complete":
        reloaded = EntryActorCritic(
            len(first_observation.vector),
            selected_variant,
            hidden_width=selected_hyperparameters.hidden_width,
        )
        checkpoint_payload = _recover_checkpoint(output, identities)
        reloaded.load_state_dict(cast(dict[str, Any], checkpoint_payload["model"]))
        with torch.no_grad():
            vector = torch.from_numpy(first_observation.vector.copy()).unsqueeze(0)
            ticker = torch.tensor([TICKERS.index(episodes.episodes[0].ticker)])
            profile = torch.tensor([PROFILES.index(episodes.episodes[0].profile)])
            first_actions = model(vector, ticker, profile)[0].argmax(dim=1)
            loaded_actions = reloaded(vector, ticker, profile)[0].argmax(dim=1)
        result["deterministic_reload_actions"] = bool(torch.equal(first_actions, loaded_actions))
        _atomic_json(output / "metrics.json", result)
        manifest = {
            "schema_version": 2,
            **result,
            "metrics_sha256": sha256_file(output / "metrics.json"),
            "sealed_holdout_accessed": False,
            "quality_claim": False,
        }
        _atomic_json(output / "manifest.json", manifest)
    return result


def train_entry_policy(
    config: RlConfig,
    training_manifest: Path,
    output: Path,
    *,
    seed: int,
    variant: PolicyVariant | str = PolicyVariant.SHARED_TICKER_VALUE,
    target_updates: int | None = None,
    target_timesteps: int | None = None,
    maximum_updates_this_run: int | None = None,
    resume: bool = False,
    repository_root: Path | None = None,
    training_control: TrainingControl | str = TrainingControl.NORMAL,
    publish: bool = True,
    hyperparameters: PpoHyperparameters | None = None,
    search_trial_number: int | None = None,
    search_seed_index: int | None = None,
    campaign_phase: str | None = None,
    parent_checkpoint: Path | None = None,
) -> dict[str, object]:
    """Load the declared schedule and train one fold/seed through the public seam."""
    episodes = load_training_episodes(config, training_manifest, repository_root)
    return _train_loaded_policy(
        config,
        episodes,
        output,
        seed=seed,
        variant=variant,
        target_updates=target_updates,
        target_timesteps=target_timesteps,
        maximum_updates_this_run=maximum_updates_this_run,
        resume=resume,
        publish=publish,
        training_control=training_control,
        hyperparameters=hyperparameters,
        search_trial_number=search_trial_number,
        search_seed_index=search_seed_index,
        campaign_phase=campaign_phase,
        parent_checkpoint=parent_checkpoint,
    )


def _completed_seed(
    config: RlConfig,
    episodes: TrainingEpisodes,
    run_output: Path,
    seed: int,
    variant: PolicyVariant,
    target_updates: int | None,
) -> dict[str, object]:
    try:
        manifest = json.loads((run_output / "manifest.json").read_text())
        result = json.loads((run_output / "metrics.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionTrainingError("completed seed artifact is invalid") from exc
    expected = _identities(config, episodes, seed, variant, target_updates, TrainingControl.NORMAL)
    if (
        not isinstance(manifest, dict)
        or not isinstance(result, dict)
        or result.get("status") != "complete"
        or result.get("identities") != expected
        or manifest.get("metrics_sha256") != sha256_file(run_output / "metrics.json")
    ):
        raise ProductionTrainingError("completed seed provenance mismatch")
    return cast(dict[str, object], result)


def _wilson_bounds(successes: int, attempts: int) -> tuple[float, float]:
    if attempts < 1 or not 0 <= successes <= attempts:
        raise ProductionTrainingError("seed outcome counts are invalid")
    z = 1.6448536269514722
    rate = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = (rate + z * z / (2.0 * attempts)) / denominator
    radius = (
        z
        * float(np.sqrt(rate * (1.0 - rate) / attempts + z * z / (4.0 * attempts**2)))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _interquartile_mean(values: Sequence[float]) -> float:
    if not values:
        raise ProductionTrainingError("seed pass rates are empty")
    ordered = sorted(float(value) for value in values)
    lower = len(ordered) * 0.25
    upper = len(ordered) * 0.75
    weighted = 0.0
    for index, value in enumerate(ordered):
        weight = max(0.0, min(index + 1.0, upper) - max(float(index), lower))
        weighted += weight * value
    return weighted / (upper - lower)


def _bootstrap_mean_interval(values: Sequence[float]) -> list[float]:
    if not values:
        raise ProductionTrainingError("seed pass rates are empty")
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return [float(array[0]), float(array[0])]
    generator = np.random.default_rng(0)
    indices = generator.integers(0, len(array), size=(5_000, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return [float(low), float(high)]


def _complete_seed_statistics(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not runs or any(run.get("status") != "complete" for run in runs):
        raise ProductionTrainingError("seed statistics require complete runs")
    seed_metrics: list[dict[str, object]] = []
    for run in runs:
        seed = run.get("seed")
        metrics = run.get("metrics")
        if type(seed) is not int or not isinstance(metrics, list) or not metrics:
            raise ProductionTrainingError("seed run metrics are invalid")
        outcomes: Counter[str] = Counter()
        for update_metrics in metrics:
            rollouts = (
                update_metrics.get("rollouts") if isinstance(update_metrics, Mapping) else None
            )
            update_outcomes = rollouts.get("outcomes") if isinstance(rollouts, Mapping) else None
            if not isinstance(update_outcomes, Mapping):
                raise ProductionTrainingError("seed run outcome metrics are invalid")
            for outcome in ("PASS", "BLOW", "TIMEOUT"):
                count = update_outcomes.get(outcome, 0)
                if type(count) is not int or count < 0:
                    raise ProductionTrainingError("seed run outcome metrics are invalid")
                outcomes[outcome] += count
            if set(update_outcomes) - {"PASS", "BLOW", "TIMEOUT"}:
                raise ProductionTrainingError("seed run outcome metrics are invalid")
        episode_count = sum(outcomes.values())
        constraint_state = run.get("constraint_controller")
        controller_episodes = (
            constraint_state.get("episode_count") if isinstance(constraint_state, Mapping) else None
        )
        cost_sum = (
            constraint_state.get("cost_sum") if isinstance(constraint_state, Mapping) else None
        )
        if (
            episode_count < 1
            or controller_episodes != episode_count
            or not isinstance(cost_sum, int | float)
            or not float(cost_sum).is_integer()
            or int(cost_sum) != outcomes["BLOW"]
        ):
            raise ProductionTrainingError("seed run outcome metrics are invalid")
        pass_count = outcomes["PASS"]
        blow_count = outcomes["BLOW"]
        pass_lower, _ = _wilson_bounds(pass_count, episode_count)
        _, blow_upper = _wilson_bounds(blow_count, episode_count)
        seed_metrics.append(
            {
                "seed": seed,
                "pass_count": pass_count,
                "blow_count": blow_count,
                "timeout_count": outcomes["TIMEOUT"],
                "episode_count": episode_count,
                "pass_rate": pass_count / episode_count,
                "blow_rate": blow_count / episode_count,
                "pass_rate_lcb_95": pass_lower,
                "blow_rate_ucb_95": blow_upper,
            }
        )
    pass_rates = [cast(float, item["pass_rate"]) for item in seed_metrics]
    blow_rates = [cast(float, item["blow_rate"]) for item in seed_metrics]
    aggregate_pass_count = sum(cast(int, item["pass_count"]) for item in seed_metrics)
    aggregate_blow_count = sum(cast(int, item["blow_count"]) for item in seed_metrics)
    aggregate_episode_count = sum(cast(int, item["episode_count"]) for item in seed_metrics)
    aggregate_pass_lower, _ = _wilson_bounds(aggregate_pass_count, aggregate_episode_count)
    _, aggregate_blow_upper = _wilson_bounds(aggregate_blow_count, aggregate_episode_count)
    worst = min(seed_metrics, key=lambda item: (cast(float, item["pass_rate"]), item["seed"]))
    worst_constraint = max(
        seed_metrics, key=lambda item: (cast(float, item["blow_rate"]), -cast(int, item["seed"]))
    )
    return {
        "seed_metrics": seed_metrics,
        "worst_seed": worst["seed"],
        "worst_constraint_seed": worst_constraint["seed"],
        "median_pass_rate": float(np.median(pass_rates)),
        "interquartile_mean_pass_rate": _interquartile_mean(pass_rates),
        "seed_mean_pass_rate_interval_95": _bootstrap_mean_interval(pass_rates),
        "aggregate_pass_count": aggregate_pass_count,
        "aggregate_blow_count": aggregate_blow_count,
        "aggregate_episode_count": aggregate_episode_count,
        "aggregate_pass_rate": aggregate_pass_count / aggregate_episode_count,
        "aggregate_blow_rate": aggregate_blow_count / aggregate_episode_count,
        "aggregate_pass_rate_lcb_95": aggregate_pass_lower,
        "aggregate_blow_rate_ucb_95": aggregate_blow_upper,
        "maximum_seed_blow_rate": max(blow_rates),
    }


def _seed_failure_path(output: Path, fold: int, seed: int) -> Path:
    directory = output / f"fold-{fold:02d}" / "failures" / f"seed-{seed}"
    attempts = sorted(directory.glob("attempt-*.json")) if directory.is_dir() else []
    return directory / f"attempt-{len(attempts) + 1:04d}.json"


def _failure_identities(config: RlConfig, episodes: TrainingEpisodes) -> dict[str, object]:
    source_value = episodes.runtime_identities.get("source", {})
    lock_value = episodes.runtime_identities.get("lock", {})
    source = source_value if isinstance(source_value, Mapping) else {}
    lock = lock_value if isinstance(lock_value, Mapping) else {}
    return {
        "config_sha256": config.digest,
        "schedule_sha256": episodes.manifest_sha256,
        "episode_collection_sha256": episodes.collection_sha256
        or _episode_collection_digest(episodes.episodes),
        "source_revision": source.get("revision", "test-fixture"),
        "source_dirty": source.get("dirty", False),
        "source_sha256": source.get("sha256", config.upstream.source_digest),
        "dependency_lock_sha256": lock.get("sha256", config.upstream.lock_digest),
        "artifact_sha256": episodes.artifact_sha256,
    }


def _train_policy_seeds_loaded(
    config: RlConfig,
    episodes: TrainingEpisodes,
    output: Path,
    *,
    seeds: Sequence[int] | None = None,
    variant: PolicyVariant | str = PolicyVariant.SHARED_TICKER_VALUE,
    target_updates: int | None = None,
    maximum_updates_this_run: int | None = None,
    resume: bool = False,
) -> dict[str, object]:
    selected = tuple(config.training.development_seeds if seeds is None else seeds)
    if not selected or len(set(selected)) != len(selected):
        raise ProductionTrainingError("training seeds must be a non-empty unique sequence")
    runs: list[dict[str, object]] = []
    selected_variant = PolicyVariant(variant)
    for seed in selected:
        run_output = output / f"fold-{episodes.fold:02d}" / f"seed-{seed}"
        try:
            if resume and (run_output / "manifest.json").is_file():
                run = _completed_seed(
                    config, episodes, run_output, seed, selected_variant, target_updates
                )
            else:
                run = _train_loaded_policy(
                    config,
                    episodes,
                    run_output,
                    seed=seed,
                    variant=selected_variant,
                    target_updates=target_updates,
                    maximum_updates_this_run=maximum_updates_this_run,
                    resume=resume and run_output.exists(),
                )
            runs.append(run)
        except Exception as exc:
            completed = []
            for completed_run in runs:
                completed_seed = completed_run.get("seed")
                if type(completed_seed) is not int or completed_run.get("status") != "complete":
                    continue
                manifest_path = (
                    output
                    / f"fold-{episodes.fold:02d}"
                    / f"seed-{completed_seed}"
                    / "manifest.json"
                )
                if manifest_path.is_file():
                    completed.append(
                        {
                            "seed": completed_seed,
                            "manifest_sha256": sha256_file(manifest_path),
                        }
                    )
            _atomic_json(
                _seed_failure_path(output, episodes.fold, seed),
                {
                    "schema_version": 1,
                    "stage": "rl-train-seeds",
                    "fold": episodes.fold,
                    "failed_seed": seed,
                    "variant": selected_variant.value,
                    "config_sha256": config.digest,
                    "schedule_sha256": episodes.manifest_sha256,
                    "identities": _failure_identities(config, episodes),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "completed_seed_manifests": completed,
                    "sealed_holdout_accessed": False,
                    "quality_claim": False,
                },
            )
            raise
    base: dict[str, object] = {
        "stage": "rl-train-seeds",
        "variant": selected_variant.value,
        "fold": episodes.fold,
        "completion_mode": (
            "production_timesteps" if target_updates is None else "bounded_updates"
        ),
        "target_updates": target_updates,
        "target_timesteps": (
            config.training.development_timesteps_per_seed if target_updates is None else None
        ),
        "reported_seeds": list(selected),
        "runs": runs,
        "all_finite_gradients": all(bool(run["finite_gradients"]) for run in runs),
        "sealed_holdout_accessed": False,
        "quality_claim": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    incomplete = [run.get("seed") for run in runs if run.get("status") != "complete"]
    if incomplete:
        progress = {
            **base,
            "status": "incomplete",
            "completed_seeds": [run.get("seed") for run in runs if run.get("status") == "complete"],
            "incomplete_seeds": incomplete,
        }
        _atomic_json(output / "seed-progress.json", progress)
        return progress
    aggregate = {**base, "status": "complete", **_complete_seed_statistics(runs)}
    _atomic_json(output / "seed-summary.json", aggregate)
    return aggregate


def train_policy_seeds(
    config: RlConfig,
    training_manifest: Path,
    output: Path,
    *,
    seeds: Sequence[int] | None = None,
    variant: PolicyVariant | str = PolicyVariant.SHARED_TICKER_VALUE,
    target_updates: int | None = None,
    maximum_updates_this_run: int | None = None,
    resume: bool = False,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Load one declared schedule and train every selected seed from it."""
    episodes = load_training_episodes(config, training_manifest, repository_root)
    return _train_policy_seeds_loaded(
        config,
        episodes,
        output,
        seeds=seeds,
        variant=variant,
        target_updates=target_updates,
        maximum_updates_this_run=maximum_updates_this_run,
        resume=resume,
    )


__all__ = [
    "PolicyVariant",
    "PpoHyperparameters",
    "ProductionTrainingError",
    "TrainingControl",
    "TrainingEpisodes",
    "derive_search_training_seed",
    "load_training_episodes",
    "train_entry_policy",
    "train_policy_seeds",
]
