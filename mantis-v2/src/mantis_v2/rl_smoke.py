"""CPU-only MaskablePPO qualification over the production entry mechanics."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
from stable_baselines3.common.monitor import Monitor
from torch.distributions import Categorical

from mantis_v2.rl_config import RlConfig
from mantis_v2.rl_environment import (
    BarData,
    CandidateData,
    EnvironmentEpisode,
    TopstepEntryEnvironment,
)

_CHECKPOINT_INTERVAL = 10_000
_ROLLOUT_STEPS = 250
_TOTAL_TIMESTEPS = 50_000


class RlSmokeError(RuntimeError):
    """Raised when the bounded CPU policy smoke cannot prove its contract."""


@dataclass
class NumericAudit:
    """Fail-closed finite-value instrumentation for every learning surface."""

    observations: int = 0
    rewards: int = 0
    policy_checks: int = 0
    checked: set[str] = field(default_factory=set)

    def observation(self, value: np.ndarray) -> None:
        self._finite("observations", value)
        self.observations += 1

    def reward(self, value: float) -> None:
        self._finite("rewards", np.asarray([value], dtype=np.float64))
        self.rewards += 1

    def policy(self, model: MaskablePPO, observation: np.ndarray, mask: np.ndarray) -> None:
        for parameter in model.policy.parameters():
            self._finite("parameters", parameter)
            if parameter.grad is not None:
                self._finite("gradients", parameter.grad)
        for state in model.policy.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    self._finite("optimizer_state", value)
        tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = model.policy.get_distribution(tensor, action_masks=mask)
            logits = cast(Categorical, distribution.distribution).logits
            values = model.policy.predict_values(tensor)
        self._finite("logits", logits)
        self._finite("values", values)
        self.policy_checks += 1

    def _finite(self, name: str, value: np.ndarray | torch.Tensor) -> None:
        finite = (
            bool(torch.isfinite(value).all().item())
            if isinstance(value, torch.Tensor)
            else bool(np.isfinite(value).all())
        )
        if not finite:
            raise RlSmokeError(f"RL smoke {name} contain NaN or Inf values")
        self.checked.add(name)

    def manifest(self) -> dict[str, object]:
        required = {
            "observations",
            "rewards",
            "logits",
            "values",
            "gradients",
            "parameters",
            "optimizer_state",
        }
        missing = required - self.checked
        if missing:
            raise RlSmokeError(f"RL smoke numerical audit missing: {', '.join(sorted(missing))}")
        return {
            "finite": True,
            "surfaces": sorted(self.checked),
            "observations": self.observations,
            "rewards": self.rewards,
            "policy_checks": self.policy_checks,
        }

    def restore(self, value: object) -> None:
        if not isinstance(value, dict):
            raise RlSmokeError("RL smoke numerical audit state is invalid")
        observations = value.get("observations")
        rewards = value.get("rewards")
        policy_checks = value.get("policy_checks")
        checked = value.get("checked")
        if (
            any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in (observations, rewards, policy_checks)
            )
            or not isinstance(checked, list)
            or any(not isinstance(item, str) for item in checked)
        ):
            raise RlSmokeError("RL smoke numerical audit state is invalid")
        self.observations = cast(int, observations)
        self.rewards = cast(int, rewards)
        self.policy_checks = cast(int, policy_checks)
        self.checked = set(cast(list[str], checked))


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
            BarData(timestamp + timedelta(minutes=3), 100.0, 100.25, 99.75, 100.0, 100.0),
            BarData(timestamp + timedelta(minutes=6), 100.0, 100.25, 99.75, 100.0, 100.0),
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
        audit: NumericAudit | None = None,
        reward_transform: Callable[[float], float] | None = None,
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
        self._audit = audit or NumericAudit()
        self._reward_transform = reward_transform or (lambda value: value)
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
        vector = observation.vector.copy()
        self._audit.observation(vector)
        return vector, {**info, "episode_index": episode_index}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        chosen = int(action)
        mask = self.action_masks()
        if chosen not in {0, 1} or not bool(mask[chosen]):
            self.illegal_action_count += 1
        label = self._environment.current_label
        observation, _, terminated, truncated, info = self._environment.step(chosen)
        reward = self._reward_transform(
            0.0 if label is None else (1.0 if chosen == label else -1.0)
        )
        vector = observation.vector.copy()
        self._audit.observation(vector)
        self._audit.reward(reward)
        return vector, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._environment.action_mask().copy()

    @property
    def current_label(self) -> int | None:
        return self._environment.current_label

    @property
    def episode_cursor(self) -> int:
        return self._cursor

    def prepare_resume(self, cursor: int, trace: Sequence[int]) -> None:
        if cursor < 1 or not trace or trace[-1] != (cursor - 1) % len(self._episodes):
            raise RlSmokeError("resume episode schedule is invalid")
        self._cursor = cursor - 1
        self.schedule_trace = list(trace[:-1])

    def rng_state(self) -> dict[str, object]:
        return {
            "environment": self.np_random.bit_generator.state,
            "action_space": self.action_space.np_random.bit_generator.state,
        }

    def restore_rng_state(self, state: Mapping[str, object]) -> None:
        environment = state.get("environment")
        action_space = state.get("action_space")
        if not isinstance(environment, dict) or not isinstance(action_space, dict):
            raise RlSmokeError("resume environment RNG state is invalid")
        self.np_random.bit_generator.state = environment
        self.action_space.np_random.bit_generator.state = action_space


def run_maskable_ppo_smoke(
    config: RlConfig,
    output: str | Path,
    *,
    resume: bool = False,
    maximum_steps_this_run: int | None = None,
    stop_after_training: bool = False,
) -> dict[str, object]:
    """Train or resume the exact bounded CPU smoke and publish durable evidence."""
    _validate_smoke_config(config)
    destination = Path(output)
    state_path = destination / "state.json"
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
    audit = NumericAudit()
    adapter = GymnasiumEntryAdapter(config, episodes, seed=config.run.seed, audit=audit)
    if resume:
        state = _validated_state(state_path, identities)
        model, completed = _load_pointed_checkpoint(destination, state, adapter, identities, audit)
        status = str(state["status"])
    else:
        completed = 0
        status = "running"
        model = _new_model(adapter, config.run.seed)
        _atomic_json(state_path, _state("running", 0, None, identities))

    if status != "trained":
        remaining = _TOTAL_TIMESTEPS - completed
        budget = (
            remaining if maximum_steps_this_run is None else min(remaining, maximum_steps_this_run)
        )
        if budget <= 0 or budget % _CHECKPOINT_INTERVAL:
            raise RlSmokeError("RL smoke invocation budget must be a positive multiple of 10000")
        target = completed + budget
        while completed < target:
            model.learn(
                total_timesteps=_CHECKPOINT_INTERVAL,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            completed = model.num_timesteps
            audit.policy(model, cast(np.ndarray, model._last_obs), adapter.action_masks())
            checkpoint = _publish_checkpoint_bundle(destination, model, adapter, identities, audit)
            next_status = "trained" if completed == _TOTAL_TIMESTEPS else "running"
            _atomic_json(state_path, _state(next_status, completed, checkpoint, identities))
        if completed < _TOTAL_TIMESTEPS:
            return {
                "status": "incomplete",
                "timesteps": completed,
                "resumable": True,
                "checkpoint": str(_read_json(state_path, "state")["checkpoint"]),
            }
    if stop_after_training:
        return {"status": "trained", "timesteps": completed, "resumable": True}
    return _evaluate_and_publish(
        destination, state_path, model, adapter, config, episodes, identities, audit
    )


def _validate_smoke_config(config: RlConfig) -> None:
    if config.run.device != "cpu" or config.run.profile != "smoke":
        raise RlSmokeError("RL smoke requires a CPU smoke configuration")
    if config.training.smoke_timesteps != _TOTAL_TIMESTEPS:
        raise RlSmokeError("RL smoke configuration must require exactly 50000 timesteps")


def _new_model(environment: GymnasiumEntryAdapter, seed: int) -> MaskablePPO:
    return MaskablePPO(
        "MlpPolicy",
        environment,
        seed=seed,
        device="cpu",
        learning_rate=3e-4,
        n_steps=_ROLLOUT_STEPS,
        batch_size=50,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [32, 32]},
        verbose=0,
    )


def _publish_checkpoint_bundle(
    destination: Path,
    model: MaskablePPO,
    adapter: GymnasiumEntryAdapter,
    identities: dict[str, object],
    audit: NumericAudit,
) -> str:
    step = model.num_timesteps
    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    final = checkpoints / f"step-{step:08d}"
    pending = Path(tempfile.mkdtemp(prefix=f".step-{step:08d}.", dir=checkpoints))
    try:
        model_path = pending / "model.zip"
        model.save(model_path)
        runtime = _runtime_state(adapter, audit)
        _atomic_json(pending / "runtime.json", runtime)
        bundle = {
            "schema_version": 1,
            "step": step,
            "identities": identities,
            "model_sha256": _sha256(model_path),
            "runtime_sha256": _sha256(pending / "runtime.json"),
        }
        _atomic_json(pending / "bundle.json", bundle)
        if final.exists():
            if not _equivalent_checkpoint(final, model, runtime, identities, step):
                raise RlSmokeError("immutable RL smoke checkpoint collision")
            shutil.rmtree(pending)
        else:
            os.replace(pending, final)
        _fsync_directory(checkpoints)
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    return str(final.relative_to(destination))


def _equivalent_checkpoint(
    existing: Path,
    model: MaskablePPO,
    runtime: dict[str, object],
    identities: dict[str, object],
    step: int,
) -> bool:
    bundle = _read_json(existing / "bundle.json", "checkpoint bundle")
    existing_runtime = _read_json(existing / "runtime.json", "checkpoint runtime")
    model_path = existing / "model.zip"
    if (
        bundle.get("step") != step
        or bundle.get("identities") != identities
        or bundle.get("model_sha256") != _sha256(model_path)
        or bundle.get("runtime_sha256") != _sha256(existing / "runtime.json")
    ):
        return False
    for key in (
        "episode_cursor",
        "schedule_trace",
        "schedule_trace_sha256",
        "environment_rng",
        "python_rng",
        "numpy_rng",
        "torch_rng",
    ):
        if existing_runtime.get(key) != runtime.get(key):
            return False
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    try:
        saved = MaskablePPO.load(model_path, device="cpu")
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
    return (
        saved.num_timesteps == model.num_timesteps
        and _torch_tree_equal(saved.policy.state_dict(), model.policy.state_dict())
        and _torch_tree_equal(
            saved.policy.optimizer.state_dict(), model.policy.optimizer.state_dict()
        )
        and _array_tree_equal(saved._last_obs, model._last_obs)
    )


def _torch_tree_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and bool(torch.equal(left, right))
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_torch_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list | tuple):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(
                _torch_tree_equal(first, second) for first, second in zip(left, right, strict=True)
            )
        )
    return left == right


def _array_tree_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and bool(np.array_equal(left, right))
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_array_tree_equal(left[key], right[key]) for key in left)
        )
    return left == right


def _validated_state(path: Path, identities: dict[str, object]) -> dict[str, object]:
    state = _read_json(path, "state")
    if state.get("schema_version") != 2 or state.get("identities") != identities:
        raise RlSmokeError("RL smoke resume provenance mismatch")
    if state.get("status") not in {"running", "trained"}:
        raise RlSmokeError("RL smoke resume state is invalid")
    return state


def _load_pointed_checkpoint(
    destination: Path,
    state: dict[str, object],
    adapter: GymnasiumEntryAdapter,
    identities: dict[str, object],
    audit: NumericAudit,
) -> tuple[MaskablePPO, int]:
    completed = state.get("completed_timesteps")
    checkpoint = state.get("checkpoint")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise RlSmokeError("RL smoke resume state is invalid")
    if completed == 0 and checkpoint is None:
        seed = identities.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RlSmokeError("RL smoke resume provenance mismatch")
        return _new_model(adapter, seed), 0
    if not isinstance(checkpoint, str):
        raise RlSmokeError("RL smoke resume state is invalid")
    bundle_dir = (destination / checkpoint).resolve()
    checkpoint_root = (destination / "checkpoints").resolve()
    if checkpoint_root not in bundle_dir.parents:
        raise RlSmokeError("RL smoke checkpoint pointer escapes run directory")
    bundle = _read_json(bundle_dir / "bundle.json", "checkpoint bundle")
    runtime = _read_json(bundle_dir / "runtime.json", "checkpoint runtime")
    model_path = bundle_dir / "model.zip"
    if (
        bundle.get("schema_version") != 1
        or bundle.get("identities") != identities
        or bundle.get("step") != completed
        or bundle.get("model_sha256") != _sha256(model_path)
        or bundle.get("runtime_sha256") != _sha256(bundle_dir / "runtime.json")
    ):
        raise RlSmokeError("RL smoke checkpoint identity mismatch")
    cursor = runtime.get("episode_cursor")
    trace = runtime.get("schedule_trace")
    rng = runtime.get("environment_rng")
    if (
        isinstance(cursor, bool)
        or not isinstance(cursor, int)
        or not isinstance(trace, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in trace)
        or not isinstance(rng, dict)
    ):
        raise RlSmokeError("RL smoke checkpoint runtime is invalid")
    adapter.prepare_resume(cursor, cast(list[int], trace))
    monitor: gym.Env[np.ndarray, int] = Monitor(adapter)
    restored_observation, _ = monitor.reset()
    adapter.restore_rng_state(rng)
    audit.restore(runtime.get("numeric_audit"))
    model = MaskablePPO.load(model_path, env=monitor, device="cpu", force_reset=False)
    if model.num_timesteps != completed:
        raise RlSmokeError("RL smoke state/checkpoint timestep mismatch")
    last_observation = cast(np.ndarray, model._last_obs)
    if model._last_obs is None or not np.array_equal(last_observation[0], restored_observation):
        raise RlSmokeError("RL smoke resume observation mismatch")
    _restore_global_rng(runtime)
    return model, completed


def _runtime_state(adapter: GymnasiumEntryAdapter, audit: NumericAudit) -> dict[str, object]:
    python = random.getstate()
    numpy = cast(tuple[str, np.ndarray, int, int, float], np.random.get_state())
    return {
        "schema_version": 1,
        "episode_cursor": adapter.episode_cursor,
        "schedule_trace": adapter.schedule_trace,
        "schedule_trace_sha256": _trace_digest(adapter.schedule_trace),
        "environment_rng": adapter.rng_state(),
        "python_rng": {"version": python[0], "state": list(python[1]), "gauss": python[2]},
        "numpy_rng": {
            "name": numpy[0],
            "keys": numpy[1].tolist(),
            "position": numpy[2],
            "has_gauss": numpy[3],
            "cached_gaussian": numpy[4],
        },
        "torch_rng": torch.get_rng_state().tolist(),
        "numeric_audit": {
            "observations": audit.observations,
            "rewards": audit.rewards,
            "policy_checks": audit.policy_checks,
            "checked": sorted(audit.checked),
        },
    }


def _restore_global_rng(runtime: dict[str, object]) -> None:
    python = runtime.get("python_rng")
    numpy = runtime.get("numpy_rng")
    torch_state = runtime.get("torch_rng")
    if (
        not isinstance(python, dict)
        or not isinstance(numpy, dict)
        or not isinstance(torch_state, list)
    ):
        raise RlSmokeError("RL smoke RNG state is invalid")
    python_values = cast(dict[str, object], python)
    numpy_values = cast(dict[str, object], numpy)
    version = python_values.get("version")
    python_state = python_values.get("state")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or not isinstance(python_state, list)
    ):
        raise RlSmokeError("RL smoke RNG state is invalid")
    random.setstate(
        (
            version,
            tuple(int(v) for v in python_state),
            cast(float | None, python_values.get("gauss")),
        )
    )
    np.random.set_state(
        (
            str(numpy_values["name"]),
            np.asarray(numpy_values["keys"], dtype=np.uint32),
            int(cast(int, numpy_values["position"])),
            int(cast(int, numpy_values["has_gauss"])),
            float(cast(float, numpy_values["cached_gaussian"])),
        )
    )
    torch.set_rng_state(torch.tensor(torch_state, dtype=torch.uint8))


def _evaluate_and_publish(
    destination: Path,
    state_path: Path,
    model: MaskablePPO,
    training_adapter: GymnasiumEntryAdapter,
    config: RlConfig,
    episodes: tuple[EnvironmentEpisode, ...],
    identities: dict[str, object],
    audit: NumericAudit,
) -> dict[str, object]:
    state = _validated_state(state_path, identities)
    if state.get("status") != "trained" or model.num_timesteps != _TOTAL_TIMESTEPS:
        raise RlSmokeError("RL smoke is not trained for publication")
    native = _evaluate_policy(model, config, episodes)
    checkpoint = str(state["checkpoint"])
    bundle_dir = destination / checkpoint
    loaded = MaskablePPO.load(bundle_dir / "model.zip", device="cpu")
    loaded_result = _evaluate_policy(loaded, config, episodes)
    audit.policy(model, cast(np.ndarray, model._last_obs), training_adapter.action_masks())
    seeded_trace = _seeded_trace(config, episodes, len(training_adapter.schedule_trace))
    training_digest = _trace_digest(training_adapter.schedule_trace)
    schedule_digests = {
        "training": training_digest,
        "same_seed": _trace_digest(seeded_trace),
        "native_evaluation": native["schedule_trace_sha256"],
        "loaded_evaluation": loaded_result["schedule_trace_sha256"],
    }
    if training_digest != schedule_digests["same_seed"]:
        raise RlSmokeError("training schedule did not reproduce from seed")
    if (
        native["actions"] != loaded_result["actions"]
        or native["schedule_trace_sha256"] != loaded_result["schedule_trace_sha256"]
    ):
        raise RlSmokeError("reloaded deterministic policy does not match")
    if native["mean_reward"] <= native["reject_all_mean_reward"]:
        raise RlSmokeError("RL smoke did not beat reject-all")
    if training_adapter.illegal_action_count or native["illegal_action_count"]:
        raise RlSmokeError("RL smoke submitted an illegal action")
    numeric = audit.manifest()
    result: dict[str, object] = {
        "schema_version": 2,
        "stage": "rl-maskable-ppo-smoke",
        "status": "complete",
        "timesteps": model.num_timesteps,
        "policy_mean_reward": native["mean_reward"],
        "reject_all_mean_reward": native["reject_all_mean_reward"],
        "finite_policy": True,
        "numeric_audit": numeric,
        "illegal_action_count": native["illegal_action_count"],
        "loaded_actions_match": True,
        "schedule_reproduced": True,
        "schedule_digests": schedule_digests,
        "sealed_holdout_accessed": False,
    }
    _atomic_json(destination / "metrics.json", result)
    manifest = {
        **result,
        "identities": identities,
        "checkpoint": checkpoint,
        "checkpoint_bundle_sha256": _sha256(bundle_dir / "bundle.json"),
        "metrics_sha256": _sha256(destination / "metrics.json"),
        "deterministic_actions_sha256": hashlib.sha256(
            json.dumps(native["actions"], separators=(",", ":")).encode()
        ).hexdigest(),
    }
    _atomic_json(destination / "manifest.json", manifest)
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
            action, _ = model.predict(
                observation, action_masks=environment.action_masks(), deterministic=True
            )
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
        "schedule_trace_sha256": _trace_digest(environment.schedule_trace),
    }


def _seeded_trace(
    config: RlConfig, episodes: tuple[EnvironmentEpisode, ...], count: int
) -> list[int]:
    environment = GymnasiumEntryAdapter(config, episodes, seed=config.run.seed)
    for index in range(count):
        environment.reset(seed=config.run.seed if index == 0 else None)
    return environment.schedule_trace


def _trace_digest(trace: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(trace), separators=(",", ":")).encode()).hexdigest()


def _state(
    status: str, completed: int, checkpoint: str | None, identities: dict[str, object]
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": status,
        "completed_timesteps": completed,
        "checkpoint": checkpoint,
        "identities": identities,
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
    source_sha256 = source.hexdigest()
    lock_sha256 = _sha256(repository / "uv.lock")
    if source_sha256 != config.upstream.source_digest:
        raise RlSmokeError("RL smoke source identity mismatch")
    if lock_sha256 != config.upstream.lock_digest:
        raise RlSmokeError("RL smoke lock identity mismatch")
    policy = {
        "algorithm": "MaskablePPO",
        "learning_rate": 3e-4,
        "n_steps": _ROLLOUT_STEPS,
        "batch_size": 50,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "net_arch": [32, 32],
    }
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
        _fsync_directory(path.parent)
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
