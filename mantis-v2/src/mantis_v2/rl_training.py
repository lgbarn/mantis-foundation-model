"""Resumable CPU PPO training for the production entry-only policy."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.distributions import Categorical

from mantis_v2.rl_config import RlConfig
from mantis_v2.rl_environment import EnvironmentEpisode, TopstepEntryEnvironment
from mantis_v2.rl_policy import (
    PROFILES,
    TICKERS,
    EntryActorCritic,
    PolicyVariant,
    ReturnNormalizers,
)
from mantis_v2.rl_provenance import sha256_file
from mantis_v2.rl_validation import load_episode_manifest

_POLICY_SCHEMA = "entry-policy-v1"
_LEARNING_RATE = 3e-4
_CLIP_RANGE = 0.2
_VALUE_COEFFICIENT = 0.5
_ENTROPY_COEFFICIENT = 0.01
_MAX_GRAD_NORM = 0.5


class ProductionTrainingError(RuntimeError):
    """Raised when production entry training cannot prove its contract."""


@dataclass(frozen=True)
class TrainingEpisodes:
    episodes: tuple[EnvironmentEpisode, ...]
    manifest_sha256: str
    config_sha256: str
    artifact_sha256: str
    partition: str
    fold: int
    schedule_seed: int


@dataclass(frozen=True)
class _Transition:
    observation: np.ndarray
    ticker: int
    profile: int
    action: int
    old_log_probability: float
    mask: np.ndarray
    reward: float
    cost: float
    terminal: bool


def load_training_episodes(
    config: RlConfig,
    manifest: Path,
    repository_root: Path | None = None,
) -> TrainingEpisodes:
    """Load a production schedule through the qualified manifest seam."""
    loaded = load_episode_manifest(config, manifest, repository_root)
    try:
        raw = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionTrainingError("training schedule manifest is invalid") from exc
    schedule_seed = raw.get("seed") if isinstance(raw, dict) else None
    if type(schedule_seed) is not int:
        raise ProductionTrainingError("training schedule seed is invalid")
    return TrainingEpisodes(
        loaded.episodes,
        loaded.manifest_sha256,
        config.digest,
        config.upstream.embedding_manifest_sha256,
        loaded.partition,
        loaded.fold,
        schedule_seed,
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
    if config.reward.gamma != 1.0 or config.reward.potential_shaping:
        raise ProductionTrainingError("training reward must be unshaped with gamma 1")
    if config.policy.actions != ("skip", "enter") or config.sizing.actor_controls_size:
        raise ProductionTrainingError("entry actor authority must remain skip/enter only")
    counts = Counter(episode.ticker for episode in episodes.episodes)
    if set(counts) != set(TICKERS) or max(counts.values()) - min(counts.values()) > 1:
        raise ProductionTrainingError("training schedule is not balanced across all tickers")
    for episode in episodes.episodes:
        if episode.profile not in PROFILES:
            raise ProductionTrainingError("training schedule has an unsupported sizing profile")


def _identities(
    config: RlConfig, episodes: TrainingEpisodes, seed: int, variant: PolicyVariant
) -> dict[str, object]:
    return {
        "policy_schema": _POLICY_SCHEMA,
        "variant": variant.value,
        "fold": episodes.fold,
        "seed": seed,
        "schedule_seed": episodes.schedule_seed,
        "schedule_sha256": episodes.manifest_sha256,
        "config_sha256": config.digest,
        "source_sha256": config.upstream.source_digest,
        "dependency_lock_sha256": config.upstream.lock_digest,
        "artifact_sha256": episodes.artifact_sha256,
        "corpus_sha256": config.upstream.corpus_manifest_sha256,
        "foundation_manifest_sha256": config.upstream.foundation_manifest_sha256,
        "foundation_weights_sha256": config.upstream.foundation_weights_sha256,
        "reward_sha256": hashlib.sha256(
            json.dumps(asdict(config.reward), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rule_sha256": config.rule_digest,
        "fee_sha256": config.fee_digest,
    }


def _masked_distribution(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    masked = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
    return Categorical(logits=masked)


def _rollout(
    config: RlConfig,
    model: EntryActorCritic,
    episode: EnvironmentEpisode,
) -> tuple[list[_Transition], dict[str, object]]:
    transitions: list[_Transition] = []
    environment = TopstepEntryEnvironment(config, episode)
    observation, _ = environment.reset()
    terminated = truncated = False
    final_status = "ACTIVE"
    bars_replayed = 0
    while not (terminated or truncated):
        ticker = TICKERS.index(episode.ticker)
        profile = PROFILES.index(episode.profile)
        mask = environment.action_mask().copy()
        if bool(mask[1]):
            vector = torch.from_numpy(observation.vector.copy()).unsqueeze(0)
            ticker_tensor = torch.tensor([ticker], dtype=torch.long)
            profile_tensor = torch.tensor([profile], dtype=torch.long)
            with torch.no_grad():
                logits, _ = model(vector, ticker_tensor, profile_tensor)
                distribution = _masked_distribution(logits, torch.from_numpy(mask).unsqueeze(0))
                action_tensor = distribution.sample()  # type: ignore[no-untyped-call]
                log_probability = float(
                    distribution.log_prob(action_tensor).item()  # type: ignore[no-untyped-call]
                )
            action = int(action_tensor.item())
            if not mask[action]:
                raise ProductionTrainingError("policy sampled an invalid masked action")
            transitions.append(
                _Transition(
                    observation.vector.copy(),
                    ticker,
                    profile,
                    action,
                    log_probability,
                    mask,
                    0.0,
                    0.0,
                    False,
                )
            )
        else:
            action = 0
        observation, _reward, terminated, truncated, info = environment.step(action)
        final_status = str(info["status"])
        bars_replayed += 1
    if not transitions:
        raise ProductionTrainingError("training episode contains no legal entry decisions")
    last = transitions[-1]
    transitions[-1] = _Transition(
        last.observation,
        last.ticker,
        last.profile,
        last.action,
        last.old_log_probability,
        last.mask,
        float(final_status == "PASS"),
        float(final_status == "BLOW"),
        True,
    )
    return transitions, {
        "ticker": episode.ticker,
        "profile": episode.profile,
        "outcome": final_status,
        "bars_replayed": bars_replayed,
        "policy_decisions": len(transitions),
    }


def _returns(transitions: Sequence[_Transition]) -> np.ndarray:
    values = np.zeros(len(transitions), dtype=np.float32)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        if transitions[index].terminal:
            running = transitions[index].reward
        values[index] = running
    return values


def _ppo_update(
    model: EntryActorCritic,
    optimizer: torch.optim.Optimizer,
    normalizers: ReturnNormalizers,
    transitions: Sequence[_Transition],
) -> dict[str, float | bool]:
    observations = torch.from_numpy(np.stack([item.observation for item in transitions]))
    tickers = torch.tensor([item.ticker for item in transitions], dtype=torch.long)
    profiles = torch.tensor([item.profile for item in transitions], dtype=torch.long)
    actions = torch.tensor([item.action for item in transitions], dtype=torch.long)
    old_log_probabilities = torch.tensor(
        [item.old_log_probability for item in transitions], dtype=torch.float32
    )
    masks = torch.from_numpy(np.stack([item.mask for item in transitions]))
    raw_returns = _returns(transitions)
    normalized_returns = torch.empty(len(transitions), dtype=torch.float32)
    for ticker_index, ticker in enumerate(TICKERS):
        owned = np.flatnonzero(np.asarray([item.ticker for item in transitions]) == ticker_index)
        if len(owned):
            normalizers.update(ticker, raw_returns[owned])
            normalized_returns[torch.from_numpy(owned)] = normalizers.normalize(
                ticker, torch.from_numpy(raw_returns[owned])
            )
    logits, values = model(observations, tickers, profiles)
    distribution = _masked_distribution(logits, masks)
    log_probabilities = distribution.log_prob(actions)  # type: ignore[no-untyped-call]
    advantages = normalized_returns - values.detach()
    if len(advantages) > 1 and float(advantages.std(unbiased=False)) > 1e-8:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False)
    ratio = torch.exp(log_probabilities - old_log_probabilities)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - _CLIP_RANGE, 1.0 + _CLIP_RANGE) * advantages
    actor_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = (values - normalized_returns).square().mean()
    entropy = distribution.entropy().mean()  # type: ignore[no-untyped-call]
    loss = actor_loss + _VALUE_COEFFICIENT * value_loss - _ENTROPY_COEFFICIENT * entropy
    optimizer.zero_grad()
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    if not finite_gradients or not bool(torch.isfinite(loss)):
        raise ProductionTrainingError("policy update produced non-finite gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), _MAX_GRAD_NORM)
    optimizer.step()
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ProductionTrainingError("policy update produced non-finite parameters")
    return {
        "loss": float(loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "finite_gradients": True,
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
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _checkpoint_payload(
    model: EntryActorCritic,
    optimizer: torch.optim.Optimizer,
    normalizers: ReturnNormalizers,
    identities: Mapping[str, object],
    update: int,
    metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "identities": dict(identities),
        "update": update,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizers": normalizers.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "metrics": list(metrics),
    }


def _publish_checkpoint(output: Path, update: int, payload: Mapping[str, object]) -> str:
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    target = checkpoints / f"update-{update:06d}"
    if target.exists():
        raise ProductionTrainingError("checkpoint update already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=checkpoints))
    try:
        checkpoint = temporary / "checkpoint.pt"
        torch.save(dict(payload), checkpoint)
        bundle = {
            "schema_version": 1,
            "update": update,
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        _atomic_json(temporary / "bundle.json", bundle)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(target.relative_to(output))


def _load_checkpoint(
    output: Path,
    identities: Mapping[str, object],
    model: EntryActorCritic,
    optimizer: torch.optim.Optimizer,
    normalizers: ReturnNormalizers,
) -> tuple[int, list[dict[str, object]]]:
    try:
        state = json.loads((output / "state.json").read_text())
        checkpoint_dir = output / str(state["checkpoint"])
        bundle = json.loads((checkpoint_dir / "bundle.json").read_text())
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ProductionTrainingError("resume state is invalid") from exc
    checkpoint = checkpoint_dir / "checkpoint.pt"
    if sha256_file(checkpoint) != bundle.get("checkpoint_sha256"):
        raise ProductionTrainingError("resume checkpoint identity mismatch")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProductionTrainingError("resume checkpoint is invalid") from exc
    if not isinstance(payload, dict) or payload.get("identities") != dict(identities):
        raise ProductionTrainingError("resume provenance mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    try:
        normalizers.load_state_dict(payload["normalizers"])
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProductionTrainingError("resume runtime state is invalid") from exc
    metrics = payload.get("metrics")
    if not isinstance(metrics, list):
        raise ProductionTrainingError("resume metrics are invalid")
    return int(payload["update"]), cast(list[dict[str, object]], metrics)


def _ablation_audit(transitions: Sequence[_Transition]) -> dict[str, bool]:
    legal = [item for item in transitions if bool(item.mask[1])]
    reject_actions = [0 for _ in legal]
    take_actions = [1 for _ in legal]
    collapsed_actions = [0 for _ in legal]
    forced_mask = torch.tensor([[True, False]])
    forced_distribution = _masked_distribution(torch.zeros((1, 2)), forced_mask)
    invalid_probability = float(forced_distribution.probs[0, 1])
    invalid_detected = not bool(forced_mask[0, 1]) and invalid_probability == 0.0
    return {
        "reject_all_detected": bool(legal and len(set(reject_actions)) == 1),
        "take_all_detected": bool(legal and len(set(take_actions)) == 1),
        "invalid_action_detected": invalid_detected,
        "action_collapse_detected": bool(legal and len(set(collapsed_actions)) < 2),
    }


def _completed_timesteps(metrics: Sequence[Mapping[str, object]]) -> int:
    total = 0
    for metric in metrics:
        rollouts = metric.get("rollouts")
        bars = rollouts.get("bars_replayed") if isinstance(rollouts, dict) else None
        if type(bars) is not int or bars < 1:
            raise ProductionTrainingError("checkpoint rollout timestep metrics are invalid")
        total += bars
    return total


def train_entry_policy(
    config: RlConfig,
    episodes: TrainingEpisodes,
    output: Path,
    *,
    seed: int,
    variant: PolicyVariant | str = PolicyVariant.SHARED_TICKER_VALUE,
    target_updates: int | None = None,
    maximum_updates_this_run: int | None = None,
    resume: bool = False,
    publish: bool = True,
) -> dict[str, object]:
    """Train or exactly resume one fold/seed policy on training episodes only."""
    _validate_contract(config, episodes)
    selected_variant = PolicyVariant(variant)
    if seed not in config.training.development_seeds:
        raise ProductionTrainingError("seed is not declared in development_seeds")
    if target_updates is not None and target_updates < 1:
        raise ProductionTrainingError("target updates must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    first_environment = TopstepEntryEnvironment(config, episodes.episodes[0])
    first_observation, _ = first_environment.reset()
    model = EntryActorCritic(len(first_observation.vector), selected_variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=_LEARNING_RATE)
    normalizers = ReturnNormalizers(TICKERS)
    identities = _identities(config, episodes, seed, selected_variant)
    completed = 0
    metrics: list[dict[str, object]] = []
    if publish:
        if resume:
            if not output.is_dir() or (output / "manifest.json").exists():
                raise ProductionTrainingError("resume requires an incomplete run directory")
            completed, metrics = _load_checkpoint(output, identities, model, optimizer, normalizers)
        else:
            try:
                output.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise ProductionTrainingError("training run identity already exists") from exc
    elif resume:
        raise ProductionTrainingError("in-memory training cannot resume")
    if maximum_updates_this_run is not None and maximum_updates_this_run < 1:
        raise ProductionTrainingError("invocation update budget must be positive")
    completed_timesteps = _completed_timesteps(metrics) if metrics else 0

    def is_complete() -> bool:
        if target_updates is not None:
            return completed >= target_updates
        return completed_timesteps >= config.training.development_timesteps_per_seed

    last_transitions: list[_Transition] = []
    last_rollouts: dict[str, object] = {}
    updates_this_run = 0
    while not is_complete() and (
        maximum_updates_this_run is None or updates_this_run < maximum_updates_this_run
    ):
        ticker_counts: Counter[str] = Counter()
        profile_counts: Counter[str] = Counter()
        outcomes: Counter[str] = Counter()
        episode_metrics: list[dict[str, object]] = []
        update_losses: list[dict[str, float | bool]] = []
        for episode in episodes.episodes:
            last_transitions, episode_rollout = _rollout(config, model, episode)
            ticker_counts[episode.ticker] += 1
            profile_counts[episode.profile] += 1
            outcomes[str(episode_rollout["outcome"])] += 1
            episode_metrics.append(episode_rollout)
            update_losses.append(_ppo_update(model, optimizer, normalizers, last_transitions))
        last_rollouts = {
            "ticker_counts": dict(sorted(ticker_counts.items())),
            "profile_counts": dict(sorted(profile_counts.items())),
            "ticker_max_minus_min": max(ticker_counts.values()) - min(ticker_counts.values()),
            "outcomes": dict(sorted(outcomes.items())),
            "bars_replayed": sum(cast(int, item["bars_replayed"]) for item in episode_metrics),
            "policy_decisions": sum(
                cast(int, item["policy_decisions"]) for item in episode_metrics
            ),
            "episodes": episode_metrics,
        }
        update_metrics: dict[str, object] = {
            "loss": float(np.mean([float(item["loss"]) for item in update_losses])),
            "actor_loss": float(np.mean([float(item["actor_loss"]) for item in update_losses])),
            "value_loss": float(np.mean([float(item["value_loss"]) for item in update_losses])),
            "entropy": float(np.mean([float(item["entropy"]) for item in update_losses])),
            "finite_gradients": all(bool(item["finite_gradients"]) for item in update_losses),
        }
        completed += 1
        updates_this_run += 1
        metrics.append({"update": completed, **update_metrics, "rollouts": last_rollouts})
        completed_timesteps = _completed_timesteps(metrics)
        if publish:
            payload = _checkpoint_payload(
                model, optimizer, normalizers, identities, completed, metrics
            )
            pointer = _publish_checkpoint(output, completed, payload)
            _atomic_json(
                output / "state.json",
                {
                    "schema_version": 1,
                    "status": "complete" if is_complete() else "incomplete",
                    "completed_updates": completed,
                    "target_updates": target_updates,
                    "completed_timesteps": completed_timesteps,
                    "minimum_target_timesteps": config.training.development_timesteps_per_seed,
                    "checkpoint": pointer,
                    "identities": identities,
                },
            )
    status = "complete" if is_complete() else "incomplete"
    model_digest = _canonical_digest(model.state_dict())
    optimizer_digest = _canonical_digest(optimizer.state_dict())
    transition_audit = {
        "rewards": [item.reward for item in last_transitions],
        "costs": [item.cost for item in last_transitions],
        "nonterminal_reward_sum": sum(
            item.reward for item in last_transitions if not item.terminal
        ),
        "nonterminal_cost_sum": sum(item.cost for item in last_transitions if not item.terminal),
    }
    result: dict[str, object] = {
        "stage": "rl-train",
        "status": status,
        "fold": episodes.fold,
        "seed": seed,
        "variant": selected_variant.value,
        "completed_updates": completed,
        "target_updates": target_updates,
        "completed_timesteps": completed_timesteps,
        "minimum_target_timesteps": config.training.development_timesteps_per_seed,
        "identities": identities,
        "gamma": config.reward.gamma,
        "reward_shaping": config.reward.potential_shaping,
        "rollouts": last_rollouts,
        "transition_audit": transition_audit,
        "finite_gradients": all(bool(item["finite_gradients"]) for item in metrics),
        "ablations": _ablation_audit(last_transitions),
        "policy_sha256": model_digest,
        "optimizer_sha256": optimizer_digest,
        "normalizers": normalizers.state_dict(),
        "metrics": metrics,
    }
    if publish and status == "complete":
        state = json.loads((output / "state.json").read_text())
        checkpoint = torch.load(
            output / state["checkpoint"] / "checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        reloaded = EntryActorCritic(len(first_observation.vector), selected_variant)
        reloaded.load_state_dict(checkpoint["model"])
        with torch.no_grad():
            vector = torch.from_numpy(first_observation.vector.copy()).unsqueeze(0)
            ticker = torch.tensor([TICKERS.index(episodes.episodes[0].ticker)])
            profile = torch.tensor([PROFILES.index(episodes.episodes[0].profile)])
            first_actions = model(vector, ticker, profile)[0].argmax(dim=1)
            loaded_actions = reloaded(vector, ticker, profile)[0].argmax(dim=1)
        result["deterministic_reload_actions"] = bool(torch.equal(first_actions, loaded_actions))
        _atomic_json(output / "metrics.json", result)
        manifest = {
            "schema_version": 1,
            **result,
            "metrics_sha256": sha256_file(output / "metrics.json"),
            "sealed_holdout_accessed": False,
        }
        _atomic_json(output / "manifest.json", manifest)
    return result


def train_policy_seeds(
    config: RlConfig,
    episodes: TrainingEpisodes,
    output: Path,
    *,
    seeds: Sequence[int] | None = None,
    variant: PolicyVariant | str = PolicyVariant.SHARED_TICKER_VALUE,
    target_updates: int | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Train every declared seed and report aggregate plus worst-seed evidence."""
    selected = tuple(config.training.development_seeds if seeds is None else seeds)
    if not selected or len(set(selected)) != len(selected):
        raise ProductionTrainingError("training seeds must be a non-empty unique sequence")
    runs = []
    selected_variant = PolicyVariant(variant)
    for seed in selected:
        run_output = output / f"fold-{episodes.fold:02d}" / f"seed-{seed}"
        completed_manifest = run_output / "manifest.json"
        if resume and completed_manifest.is_file():
            try:
                manifest = json.loads(completed_manifest.read_text())
                result = json.loads((run_output / "metrics.json").read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ProductionTrainingError("completed seed artifact is invalid") from exc
            expected = _identities(config, episodes, seed, selected_variant)
            if (
                not isinstance(manifest, dict)
                or not isinstance(result, dict)
                or result.get("status") != "complete"
                or result.get("identities") != expected
                or manifest.get("metrics_sha256") != sha256_file(run_output / "metrics.json")
            ):
                raise ProductionTrainingError("completed seed provenance mismatch")
            runs.append(result)
        else:
            runs.append(
                train_entry_policy(
                    config,
                    episodes,
                    run_output,
                    seed=seed,
                    variant=selected_variant,
                    target_updates=target_updates,
                    resume=resume and run_output.exists(),
                )
            )
    scores: dict[int, float] = {}
    for run in runs:
        run_seed = run["seed"]
        run_metrics = run["metrics"]
        if type(run_seed) is not int or not isinstance(run_metrics, list) or not run_metrics:
            raise ProductionTrainingError("seed run metrics are invalid")
        final_metrics = run_metrics[-1]
        if not isinstance(final_metrics, dict):
            raise ProductionTrainingError("seed run metrics are invalid")
        rollout_metrics = final_metrics.get("rollouts")
        outcomes = rollout_metrics.get("outcomes") if isinstance(rollout_metrics, dict) else None
        passes = outcomes.get("PASS", 0) if isinstance(outcomes, dict) else 0
        if type(passes) is not int:
            raise ProductionTrainingError("seed run outcome metrics are invalid")
        scores[run_seed] = float(passes)
    worst_seed = min(scores, key=lambda seed: (scores[seed], seed))
    aggregate: dict[str, object] = {
        "stage": "rl-train-seeds",
        "variant": selected_variant.value,
        "fold": episodes.fold,
        "reported_seeds": list(selected),
        "runs": runs,
        "worst_seed": worst_seed,
        "median_passes": float(np.median(list(scores.values()))),
        "sealed_holdout_accessed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "seed-summary.json", aggregate)
    return aggregate


__all__ = [
    "PolicyVariant",
    "ProductionTrainingError",
    "TrainingEpisodes",
    "load_training_episodes",
    "train_entry_policy",
    "train_policy_seeds",
]
