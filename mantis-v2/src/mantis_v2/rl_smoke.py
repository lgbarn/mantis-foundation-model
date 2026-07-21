"""CPU-only MaskablePPO qualification over the production entry mechanics."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import optuna
import sb3_contrib
import stable_baselines3
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from mantis_v2.rl_config import RlConfig
from mantis_v2.rl_environment import (
    BarData,
    CandidateData,
    EnvironmentEpisode,
    TopstepEntryEnvironment,
)


class _MaskedDiscrete(spaces.Discrete[np.int64]):
    def __init__(self, mask_provider: Callable[[], np.ndarray]) -> None:
        super().__init__(2)
        self._mask_provider = mask_provider

    def sample(
        self,
        mask: np.ndarray | None = None,
        probability: np.ndarray | None = None,
    ) -> np.int64:
        active_mask = self._mask_provider().astype(np.int8) if mask is None else mask
        return np.int64(super().sample(mask=active_mask, probability=probability))


def build_synthetic_episodes(*, seed: int, count: int) -> tuple[EnvironmentEpisode, ...]:
    """Build a balanced, deterministic entry fixture with a causal learnable signal."""
    if count < 2:
        raise ValueError("synthetic episode count must be at least two")
    labels = np.arange(count, dtype=np.int8) % 2
    generator = np.random.default_rng(seed)
    generator.shuffle(labels)
    origin = datetime(2025, 1, 6, 15, 0, tzinfo=UTC)
    episodes = []
    for index, label_value in enumerate(labels.tolist()):
        label = int(label_value)
        signal = 1.0 if label else -1.0
        candidate = CandidateData(
            embedding=np.array([signal, 0.25 * signal, 0.0, 1.0], dtype=np.float32),
            direction=1,
            trend_line=99.0,
            atr=2.0,
            bars_since_direction_change=2,
            label=label,
        )
        timestamp = origin + timedelta(days=index)
        bars = (
            BarData(timestamp, 100.0, 100.25, 99.75, 100.0, 100.0, candidate),
            BarData(
                timestamp + timedelta(minutes=3),
                100.0,
                100.25,
                99.75,
                100.0,
                100.0,
            ),
            BarData(
                timestamp + timedelta(minutes=6),
                100.0,
                100.25,
                99.75,
                100.0,
                100.0,
            ),
        )
        episodes.append(EnvironmentEpisode("ES", "one_mini", bars))
    return tuple(episodes)


class GymnasiumEntryAdapter(gym.Env[np.ndarray, int]):
    """Gymnasium and MaskablePPO seam for the production entry environment."""

    def __init__(
        self,
        config: RlConfig,
        episodes: tuple[EnvironmentEpisode, ...],
        *,
        seed: int,
    ) -> None:
        super().__init__()
        self.metadata = {"render_modes": []}
        if not episodes:
            raise ValueError("Gymnasium adapter requires at least one episode")
        self._config = config
        self._episodes = episodes
        self._initial_seed = seed
        self._cursor = 0
        self._environment = TopstepEntryEnvironment(config, episodes[0])
        observation, _ = self._environment.reset(seed=seed)
        bounds = np.finfo(np.float32)
        self.observation_space = spaces.Box(
            low=bounds.min,
            high=bounds.max,
            shape=observation.vector.shape,
            dtype=np.float32,
        )
        self.action_space = cast(spaces.Space[int], _MaskedDiscrete(self.action_masks))
        self.illegal_action_count = 0
        self.schedule_trace: list[int] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._cursor = 0
            self.schedule_trace.clear()
        episode_index = self._cursor % len(self._episodes)
        self._cursor += 1
        self.schedule_trace.append(episode_index)
        self._environment = TopstepEntryEnvironment(self._config, self._episodes[episode_index])
        observation, info = self._environment.reset(
            seed=self._initial_seed if seed is None else seed
        )
        return observation.vector.copy(), {**info, "episode_index": episode_index}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        chosen = int(action)
        mask = self.action_masks()
        if chosen not in {0, 1} or not bool(mask[chosen]):
            self.illegal_action_count += 1
        label = self._environment.current_label
        observation, _, terminated, truncated, info = self._environment.step(chosen)
        reward = 0.0 if label is None else (1.0 if chosen == label else -1.0)
        return observation.vector.copy(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._environment.action_mask().copy()

    @property
    def current_label(self) -> int | None:
        return self._environment.current_label

    @property
    def episode_cursor(self) -> int:
        return self._cursor

    def restore_episode_cursor(self, cursor: int) -> None:
        """Prepare the already-reset episode to be selected by SB3's reload reset."""
        if cursor < 1:
            raise RlSmokeError("resume episode cursor is invalid")
        self._cursor = cursor - 1


class RlSmokeError(RuntimeError):
    """Raised when the bounded CPU policy smoke cannot prove its contract."""


def run_maskable_ppo_smoke(
    config: RlConfig,
    output: str | Path,
    *,
    resume: bool = False,
    maximum_steps_this_run: int | None = None,
) -> dict[str, object]:
    """Train or resume the exact bounded CPU smoke and publish durable evidence."""
    if config.run.device != "cpu" or config.run.profile != "smoke":
        raise RlSmokeError("RL smoke requires a CPU smoke configuration")
    if config.training.smoke_timesteps != 50_000:
        raise RlSmokeError("RL smoke configuration must require exactly 50000 timesteps")
    destination = Path(output)
    manifest_path = destination / "manifest.json"
    if resume:
        if manifest_path.exists():
            raise RlSmokeError("completed RL smoke cannot be resumed or overwritten")
        if not destination.is_dir():
            raise RlSmokeError("resume destination does not exist")
    else:
        try:
            destination.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise RlSmokeError("RL smoke run identity already exists") from exc

    episodes = build_synthetic_episodes(seed=config.run.seed, count=128)
    identities = _smoke_identities(config, episodes)
    state_path = destination / "state.json"
    checkpoint_path = destination / "checkpoint.zip"
    if resume:
        state = _read_json(state_path, "resume state")
        if state.get("identities") != identities:
            raise RlSmokeError("RL smoke resume provenance mismatch")
        completed_raw = state.get("completed_timesteps")
        if isinstance(completed_raw, bool) or not isinstance(completed_raw, int):
            raise RlSmokeError("RL smoke resume state is invalid")
        completed = completed_raw
        if completed < 0 or completed >= config.training.smoke_timesteps:
            raise RlSmokeError("RL smoke resume state is invalid")
        if not checkpoint_path.is_file():
            raise RlSmokeError("RL smoke resume checkpoint is missing")
        if state.get("checkpoint_sha256") != _sha256(checkpoint_path):
            raise RlSmokeError("RL smoke resume checkpoint identity mismatch")
        schedule_cursor_raw = state.get("schedule_cursor")
        if isinstance(schedule_cursor_raw, bool) or not isinstance(schedule_cursor_raw, int):
            raise RlSmokeError("RL smoke resume state is invalid")
        schedule_cursor = schedule_cursor_raw
    else:
        completed = 0
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "running",
                "completed_timesteps": 0,
                "schedule_cursor": 0,
                "identities": identities,
            },
        )

    environment = GymnasiumEntryAdapter(config, episodes, seed=config.run.seed)
    if resume:
        environment.restore_episode_cursor(schedule_cursor)
        model = MaskablePPO.load(checkpoint_path, env=environment, device="cpu")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            environment,
            seed=config.run.seed,
            device="cpu",
            learning_rate=3e-4,
            n_steps=250,
            batch_size=50,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [32, 32]},
            verbose=0,
        )
    remaining = config.training.smoke_timesteps - completed
    run_budget = (
        remaining if maximum_steps_this_run is None else min(remaining, maximum_steps_this_run)
    )
    if run_budget <= 0 or run_budget % 250:
        raise RlSmokeError("RL smoke invocation budget must be a positive multiple of 250")
    target = completed + run_budget
    while completed < target:
        chunk = min(10_000, target - completed)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        completed += chunk
        _atomic_model_save(model, checkpoint_path)
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "running" if completed < config.training.smoke_timesteps else "trained",
                "completed_timesteps": completed,
                "schedule_cursor": environment.episode_cursor,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "identities": identities,
            },
        )
    if completed < config.training.smoke_timesteps:
        return {"status": "incomplete", "timesteps": completed, "resumable": True}

    evaluation = _evaluate_policy(model, config, episodes)
    loaded = MaskablePPO.load(checkpoint_path, env=environment, device="cpu")
    loaded_evaluation = _evaluate_policy(loaded, config, episodes)
    finite_policy = all(
        torch.isfinite(parameter).all().item() for parameter in model.policy.parameters()
    )
    if not finite_policy:
        raise RlSmokeError("RL smoke policy contains NaN or Inf values")
    if environment.illegal_action_count or evaluation["illegal_action_count"]:
        raise RlSmokeError("RL smoke submitted an illegal action")
    if evaluation["mean_reward"] <= evaluation["reject_all_mean_reward"]:
        raise RlSmokeError("RL smoke did not beat reject-all")
    if evaluation["actions"] != loaded_evaluation["actions"]:
        raise RlSmokeError("reloaded deterministic actions do not match")
    schedule_reproduced = _episode_digest(episodes) == _episode_digest(
        build_synthetic_episodes(seed=config.run.seed, count=128)
    )
    if not schedule_reproduced:
        raise RlSmokeError("synthetic episode schedule did not reproduce")
    result: dict[str, object] = {
        "schema_version": 1,
        "stage": "rl-maskable-ppo-smoke",
        "status": "complete",
        "timesteps": completed,
        "policy_mean_reward": evaluation["mean_reward"],
        "reject_all_mean_reward": evaluation["reject_all_mean_reward"],
        "finite_policy": finite_policy,
        "illegal_action_count": evaluation["illegal_action_count"],
        "loaded_actions_match": True,
        "schedule_reproduced": schedule_reproduced,
        "sealed_holdout_accessed": False,
    }
    _atomic_json(destination / "metrics.json", result)
    manifest = {
        **result,
        "identities": identities,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "metrics_sha256": _sha256(destination / "metrics.json"),
        "deterministic_actions_sha256": hashlib.sha256(
            json.dumps(evaluation["actions"], separators=(",", ":")).encode()
        ).hexdigest(),
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        state_path,
        {
            "schema_version": 1,
            "status": "complete",
            "completed_timesteps": completed,
            "schedule_cursor": environment.episode_cursor,
            "identities": identities,
        },
    )
    return result


def _evaluate_policy(
    model: MaskablePPO,
    config: RlConfig,
    episodes: tuple[EnvironmentEpisode, ...],
) -> dict[str, Any]:
    environment = GymnasiumEntryAdapter(config, episodes, seed=config.run.seed)
    actions: list[int] = []
    reward_total = 0.0
    reject_total = 0.0
    for index in range(len(episodes)):
        observation, _ = environment.reset(seed=config.run.seed if index == 0 else None)
        done = False
        while not done:
            label = environment.current_label
            masks = environment.action_masks()
            action, _ = model.predict(observation, action_masks=masks, deterministic=True)
            chosen = int(action)
            observation, reward, terminated, truncated, _ = environment.step(chosen)
            if label is not None:
                actions.append(chosen)
                reward_total += reward
                reject_total += 1.0 if label == 0 else -1.0
            done = terminated or truncated
    return {
        "actions": actions,
        "mean_reward": reward_total / len(episodes),
        "reject_all_mean_reward": reject_total / len(episodes),
        "illegal_action_count": environment.illegal_action_count,
    }


def _episode_digest(episodes: tuple[EnvironmentEpisode, ...]) -> str:
    payload = [
        {
            "ticker": episode.ticker,
            "profile": episode.profile,
            "label": episode.bars[0].candidate.label if episode.bars[0].candidate else None,
            "embedding": episode.bars[0].candidate.embedding.tolist()
            if episode.bars[0].candidate
            else None,
        }
        for episode in episodes
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _smoke_identities(
    config: RlConfig, episodes: tuple[EnvironmentEpisode, ...]
) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[3]
    source = hashlib.sha256()
    for path in sorted((repository / "mantis-v2" / "src").rglob("*.py")):
        source.update(str(path.relative_to(repository)).encode())
        source.update(b"\0")
        source.update(path.read_bytes())
        source.update(b"\0")
    policy = {
        "algorithm": "MaskablePPO",
        "learning_rate": 3e-4,
        "n_steps": 250,
        "batch_size": 50,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "net_arch": [32, 32],
    }
    source_sha256 = source.hexdigest()
    lock_sha256 = _sha256(repository / "uv.lock")
    if source_sha256 != config.upstream.source_digest:
        raise RlSmokeError("RL smoke source identity mismatch")
    if lock_sha256 != config.upstream.lock_digest:
        raise RlSmokeError("RL smoke lock identity mismatch")
    return {
        "config_sha256": config.digest,
        "source_sha256": source_sha256,
        "lock_sha256": lock_sha256,
        "schedule_sha256": _episode_digest(episodes),
        "policy_sha256": hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "seed": config.run.seed,
        "dependencies": {
            "gymnasium": gym.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__,
            "optuna": optuna.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


def _atomic_model_save(model: MaskablePPO, destination: Path) -> None:
    pending = destination.with_name(f".{destination.stem}.pending.zip")
    try:
        model.save(pending)
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RlSmokeError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise RlSmokeError(f"invalid {label}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
