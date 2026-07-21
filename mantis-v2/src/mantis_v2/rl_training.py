"""Resumable CPU masked PPO training for the production entry-only policy."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

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
from mantis_v2.rl_provenance import sha256_file, verify_rl_runtime
from mantis_v2.rl_validation import load_episode_manifest

_POLICY_SCHEMA = "entry-policy-v2"
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
    mask: np.ndarray
    reward: float
    cost: float
    terminal: bool


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
    counts = Counter(episode.ticker for episode in episodes.episodes)
    if set(counts) != set(TICKERS) or max(counts.values()) - min(counts.values()) > 1:
        raise ProductionTrainingError("training schedule is not balanced across all tickers")
    for episode in episodes.episodes:
        if episode.profile not in PROFILES:
            raise ProductionTrainingError("training schedule has an unsupported sizing profile")


def _identities(
    config: RlConfig, episodes: TrainingEpisodes, seed: int, variant: PolicyVariant
) -> dict[str, object]:
    source = episodes.runtime_identities.get("source", {})
    lock = episodes.runtime_identities.get("lock", {})
    if not isinstance(source, Mapping) or not isinstance(lock, Mapping):
        raise ProductionTrainingError("runtime source or lock identity is invalid")
    return {
        "policy_schema": _POLICY_SCHEMA,
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
        "rule_sha256": config.rule_digest,
        "fee_sha256": config.fee_digest,
    }


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
                logits, values = model(vector, ticker_tensor, profile_tensor)
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
                    float(values.item()),
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
    reward, cost = _terminal_signal(final_status)
    last = transitions[-1]
    transitions[-1] = _Transition(
        last.observation,
        last.ticker,
        last.profile,
        last.action,
        last.old_log_probability,
        last.old_value,
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
    }


def _returns(transitions: Sequence[_Transition]) -> np.ndarray:
    values = np.zeros(len(transitions), dtype=np.float32)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        if transitions[index].terminal:
            running = transitions[index].reward - transitions[index].cost
        values[index] = running
    return values


def _actor_weights(tickers: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    weights = torch.zeros(len(tickers), dtype=torch.float32)
    totals: dict[str, float] = {}
    present = [index for index in range(len(TICKERS)) if bool((tickers == index).any())]
    for ticker_index in present:
        owned = tickers == ticker_index
        weights[owned] = 1.0 / (len(present) * int(owned.sum()))
        totals[TICKERS[ticker_index]] = float(weights[owned].sum())
    return weights, totals


def _ppo_update(
    model: EntryActorCritic,
    optimizer: torch.optim.Optimizer,
    normalizers: ReturnNormalizers,
    transitions: Sequence[_Transition],
    *,
    epochs: int,
    minibatch_size: int,
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
    raw_returns = _returns(transitions)
    normalized_returns = torch.empty(len(transitions), dtype=torch.float32)
    advantages = torch.empty(len(transitions), dtype=torch.float32)
    for ticker_index, ticker in enumerate(TICKERS):
        owned = np.flatnonzero(np.asarray([item.ticker for item in transitions]) == ticker_index)
        if len(owned):
            normalizers.update(ticker, raw_returns[owned])
            owned_tensor = torch.from_numpy(owned)
            normalized_returns[owned_tensor] = normalizers.normalize(
                ticker, torch.from_numpy(raw_returns[owned])
            )
            owned_advantages = normalized_returns[owned_tensor] - old_values[owned_tensor]
            if len(owned_advantages) > 1 and float(owned_advantages.std(unbiased=False)) > 1e-8:
                owned_advantages = (
                    owned_advantages - owned_advantages.mean()
                ) / owned_advantages.std(unbiased=False)
            advantages[owned_tensor] = owned_advantages
    actor_weights, actor_weight_totals = _actor_weights(tickers)
    losses: list[float] = []
    actor_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    ratios: list[torch.Tensor] = []
    clipped: list[torch.Tensor] = []
    size = len(transitions)
    for _epoch in range(epochs):
        permutation = torch.randperm(size)
        for start in range(0, size, minibatch_size):
            index = permutation[start : start + minibatch_size]
            logits, values = model(observations[index], tickers[index], profiles[index])
            distribution = _masked_distribution(logits, masks[index])
            current = distribution.log_prob(actions[index])  # type: ignore[no-untyped-call]
            ratio = torch.exp(current - old_log_probabilities[index])
            unclipped = ratio * advantages[index]
            clipped_ratio = torch.clamp(ratio, 1.0 - _CLIP_RANGE, 1.0 + _CLIP_RANGE)
            surrogate = torch.minimum(unclipped, clipped_ratio * advantages[index])
            actor_loss = -(surrogate * actor_weights[index]).sum()
            value_loss = (values - normalized_returns[index]).square().mean()
            entropy = distribution.entropy().mean()  # type: ignore[no-untyped-call]
            loss = actor_loss + _VALUE_COEFFICIENT * value_loss - _ENTROPY_COEFFICIENT * entropy
            optimizer.zero_grad()
            loss.backward()
            if not bool(torch.isfinite(loss)) or not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                raise ProductionTrainingError("policy update produced non-finite gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), _MAX_GRAD_NORM)
            optimizer.step()
            if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                raise ProductionTrainingError("policy update produced non-finite parameters")
            losses.append(float(loss.detach()))
            actor_losses.append(float(actor_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            entropies.append(float(entropy.detach()))
            ratios.append(ratio.detach())
            clipped.append(((ratio < 1.0 - _CLIP_RANGE) | (ratio > 1.0 + _CLIP_RANGE)).detach())
    all_ratios = torch.cat(ratios)
    all_clipped = torch.cat(clipped)
    return {
        "loss": float(np.mean(losses)),
        "actor_loss": float(np.sum(actor_losses) / epochs),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "finite_gradients": True,
        "ppo_epochs": epochs,
        "minibatch_size": minibatch_size,
        "ratio_non_unit_fraction": float((torch.abs(all_ratios - 1.0) > 1e-6).float().mean()),
        "ratio_max_deviation": float(torch.abs(all_ratios - 1.0).max()),
        "clip_fraction": float(all_clipped.float().mean()),
        "actor_weight_totals": actor_weight_totals,
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


def _ablation_audit(
    model: EntryActorCritic, transitions: Sequence[_Transition]
) -> dict[str, object]:
    legal = [item for item in transitions if bool(item.mask[1])]
    if not legal:
        raise ProductionTrainingError("ablation audit requires legal entry decisions")
    observations = torch.from_numpy(np.stack([item.observation for item in legal]))
    tickers = torch.tensor([item.ticker for item in legal], dtype=torch.long)
    profiles = torch.tensor([item.profile for item in legal], dtype=torch.long)
    masks = torch.from_numpy(np.stack([item.mask for item in legal]))
    with torch.no_grad():
        logits, _ = model(observations, tickers, profiles)
        base = _masked_distribution(logits, masks)
        base_probabilities = base.probs
        reject = _masked_distribution(logits + torch.tensor([100.0, -100.0]), masks)
        take = _masked_distribution(logits + torch.tensor([-100.0, 100.0]), masks)
        invalid_masks = masks.clone()
        invalid_masks[:, 1] = False
        invalid = _masked_distribution(logits, invalid_masks)
        collapsed_observations = observations[:1].repeat(len(observations), 1)
        collapsed_tickers = tickers[:1].repeat(len(observations))
        collapsed_profiles = profiles[:1].repeat(len(observations))
        collapsed_logits, _ = model(collapsed_observations, collapsed_tickers, collapsed_profiles)
        collapsed = _masked_distribution(collapsed_logits, masks)
    reward_objective = sum(item.reward - item.cost for item in legal if item.terminal)
    perturbed_reward_objective = sum(
        item.reward + 1.0 - item.cost for item in legal if item.terminal
    )
    evidence = {
        "base_enter_probability_range": [
            float(base_probabilities[:, 1].min()),
            float(base_probabilities[:, 1].max()),
        ],
        "reject_all_actions": sorted(set(reject.probs.argmax(dim=1).tolist())),
        "take_all_actions": sorted(set(take.probs.argmax(dim=1).tolist())),
        "invalid_enter_probability_max": float(invalid.probs[:, 1].max()),
        "collapsed_action_count": len(set(collapsed.probs.argmax(dim=1).tolist())),
        "terminal_objective": reward_objective,
        "perturbed_terminal_objective": perturbed_reward_objective,
    }
    return {
        "reject_all_detected": evidence["reject_all_actions"] == [0],
        "take_all_detected": evidence["take_all_actions"] == [1],
        "invalid_action_detected": evidence["invalid_enter_probability_max"] == 0.0,
        "action_collapse_detected": evidence["collapsed_action_count"] == 1,
        "reward_perturbation_detected": reward_objective != perturbed_reward_objective,
        "evidence": evidence,
    }


def _checkpoint_payload(
    model: EntryActorCritic,
    optimizer: torch.optim.Optimizer,
    normalizers: ReturnNormalizers,
    identities: Mapping[str, object],
    update: int,
    metrics: Sequence[Mapping[str, object]],
    evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "identities": dict(identities),
        "update": update,
        "episode_cursor": 0,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizers": normalizers.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
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
            "schema_version": 2,
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
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        raise ProductionTrainingError("resume checkpoint bundle is invalid") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 2
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
    if updates != list(range(1, len(payloads) + 1)):
        raise ProductionTrainingError("resume checkpoint sequence is not contiguous")
    return payloads[-1]


def _restore_checkpoint(
    payload: Mapping[str, object],
    model: EntryActorCritic,
    optimizer: torch.optim.Optimizer,
    normalizers: ReturnNormalizers,
) -> tuple[int, list[dict[str, object]], dict[str, object]]:
    try:
        model.load_state_dict(cast(dict[str, Any], payload["model"]))
        optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer"]))
        normalizers.load_state_dict(cast(dict[str, object], payload["normalizers"]))
        rng = cast(dict[str, Any], payload["rng"])
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
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
        bars = rollouts.get("bars_replayed") if isinstance(rollouts, dict) else None
        if type(bars) is not int or bars < 1:
            raise ProductionTrainingError("checkpoint rollout timestep metrics are invalid")
        total += bars
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
    maximum_updates_this_run: int | None = None,
    resume: bool = False,
    publish: bool = True,
) -> dict[str, object]:
    """Internal seam for an already verified, complete schedule collection."""
    _validate_contract(config, episodes)
    selected_variant = PolicyVariant(variant)
    if seed not in config.training.development_seeds:
        raise ProductionTrainingError("seed is not declared in development_seeds")
    if target_updates is not None and target_updates < 1:
        raise ProductionTrainingError("target updates must be positive")
    if maximum_updates_this_run is not None and maximum_updates_this_run < 1:
        raise ProductionTrainingError("invocation update budget must be positive")
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
    last_evidence: dict[str, object] = {}
    if publish:
        if resume:
            if not output.is_dir() or (output / "manifest.json").exists():
                raise ProductionTrainingError("resume requires an unpublished run directory")
            payload = _recover_checkpoint(output, identities)
            completed, metrics, last_evidence = _restore_checkpoint(
                payload, model, optimizer, normalizers
            )
        else:
            try:
                output.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise ProductionTrainingError("training run identity already exists") from exc
    elif resume:
        raise ProductionTrainingError("in-memory training cannot resume")
    completed_timesteps = _completed_timesteps(metrics) if metrics else 0

    def is_complete() -> bool:
        if target_updates is not None:
            return completed >= target_updates
        return completed_timesteps >= config.training.development_timesteps_per_seed

    if publish and resume:
        pointer = f"checkpoints/update-{completed:06d}"
        _atomic_json(
            output / "state.json",
            _state_payload(
                identities,
                completed,
                target_updates,
                completed_timesteps,
                config.training.development_timesteps_per_seed,
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
        all_transitions: list[_Transition] = []
        for episode in episodes.episodes:
            transitions, episode_rollout = _rollout(config, model, episode)
            all_transitions.extend(transitions)
            ticker_counts[episode.ticker] += 1
            profile_counts[episode.profile] += 1
            outcomes[str(episode_rollout["outcome"])] += 1
            episode_metrics.append(episode_rollout)
        update_metrics = _ppo_update(
            model,
            optimizer,
            normalizers,
            all_transitions,
            epochs=config.training.ppo_epochs,
            minibatch_size=config.training.minibatch_size,
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
        ablations = _ablation_audit(model, all_transitions)
        last_evidence = {
            "rollouts": rollouts,
            "transition_audit": _transition_audit(all_transitions),
            "ablations": ablations,
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
                    config.training.development_timesteps_per_seed,
                    pointer,
                    is_complete(),
                ),
            )
    status = "complete" if is_complete() else "incomplete"
    if completed and not last_evidence:
        raise ProductionTrainingError("checkpoint is missing last rollout evidence")
    result: dict[str, object] = {
        "stage": "rl-train",
        "algorithm": "masked_ppo",
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
        **last_evidence,
        "finite_gradients": bool(metrics)
        and all(bool(item["finite_gradients"]) for item in metrics),
        "policy_sha256": _canonical_digest(model.state_dict()),
        "optimizer_sha256": _canonical_digest(optimizer.state_dict()),
        "normalizers": normalizers.state_dict(),
        "metrics": metrics,
    }
    if publish and status == "complete":
        reloaded = EntryActorCritic(len(first_observation.vector), selected_variant)
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
    maximum_updates_this_run: int | None = None,
    resume: bool = False,
    repository_root: Path | None = None,
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
        maximum_updates_this_run=maximum_updates_this_run,
        resume=resume,
    )


def _completed_seed(
    config: RlConfig,
    episodes: TrainingEpisodes,
    run_output: Path,
    seed: int,
    variant: PolicyVariant,
) -> dict[str, object]:
    try:
        manifest = json.loads((run_output / "manifest.json").read_text())
        result = json.loads((run_output / "metrics.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionTrainingError("completed seed artifact is invalid") from exc
    expected = _identities(config, episodes, seed, variant)
    if (
        not isinstance(manifest, dict)
        or not isinstance(result, dict)
        or result.get("status") != "complete"
        or result.get("identities") != expected
        or manifest.get("metrics_sha256") != sha256_file(run_output / "metrics.json")
    ):
        raise ProductionTrainingError("completed seed provenance mismatch")
    return cast(dict[str, object], result)


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
        if resume and (run_output / "manifest.json").is_file():
            runs.append(_completed_seed(config, episodes, run_output, seed, selected_variant))
        else:
            runs.append(
                _train_loaded_policy(
                    config,
                    episodes,
                    run_output,
                    seed=seed,
                    variant=selected_variant,
                    target_updates=target_updates,
                    maximum_updates_this_run=maximum_updates_this_run,
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
        rollouts = final_metrics.get("rollouts") if isinstance(final_metrics, dict) else None
        outcomes = rollouts.get("outcomes") if isinstance(rollouts, dict) else None
        passes = outcomes.get("PASS", 0) if isinstance(outcomes, dict) else 0
        if type(passes) is not int:
            raise ProductionTrainingError("seed run outcome metrics are invalid")
        scores[run_seed] = float(passes)
    worst_seed = min(scores, key=lambda item: (scores[item], item))
    aggregate: dict[str, object] = {
        "stage": "rl-train-seeds",
        "variant": selected_variant.value,
        "fold": episodes.fold,
        "reported_seeds": list(selected),
        "runs": runs,
        "worst_seed": worst_seed,
        "median_passes": float(np.median(list(scores.values()))),
        "all_finite_gradients": all(bool(run["finite_gradients"]) for run in runs),
        "sealed_holdout_accessed": False,
        "quality_claim": False,
    }
    output.mkdir(parents=True, exist_ok=True)
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
    "ProductionTrainingError",
    "TrainingEpisodes",
    "load_training_episodes",
    "train_entry_policy",
    "train_policy_seeds",
]
