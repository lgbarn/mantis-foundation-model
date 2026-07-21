from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from mantis_v2 import cli
from mantis_v2.rl_config import load_rl_config
from mantis_v2.rl_environment import BarData, CandidateData, EnvironmentEpisode
from mantis_v2.rl_policy import EntryActorCritic, ReturnNormalizers
from mantis_v2.rl_training import (
    ConstraintController,
    PolicyVariant,
    ProductionTrainingError,
    TrainingControl,
    TrainingEpisodes,
    _actor_weights,
    _balanced_minibatches,
    _completed_timesteps,
    _cost_returns,
    _lagrangian_actor_advantages,
    _ppo_update,
    _reward_returns,
    _terminal_signal,
    _train_loaded_policy,
    _train_policy_seeds_loaded,
    _Transition,
    load_training_episodes,
    train_entry_policy,
)

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "CL", "ZB")


def _config():
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    return replace(
        config,
        run=replace(config.run, profile="production"),
        training=replace(config.training, ppo_epochs=2, minibatch_size=4096),
    )


def _candidate(signal: float, direction: int = 1) -> CandidateData:
    return CandidateData(
        embedding=np.array([signal, -signal, 0.5, 0.25], dtype=np.float32),
        direction=direction,
        trend_line=99.0,
        atr=2.0,
        bars_since_direction_change=2,
        label=int(signal > 0),
    )


def _episode(ticker: str, profile: str, signal: float = 1.0) -> EnvironmentEpisode:
    origin = datetime(2025, 1, 6, 15, 0, tzinfo=UTC)
    bars = tuple(
        BarData(
            origin + timedelta(minutes=3 * index),
            100.0,
            100.5,
            99.5,
            100.0,
            10.0,
            _candidate(signal) if index in {0, 2} else None,
        )
        for index in range(5)
    )
    return EnvironmentEpisode(ticker, profile, bars)


def _learning_episode(ticker: str, outcome: str) -> EnvironmentEpisode:
    origin = datetime(2025, 1, 6, 20, 0, tzinfo=UTC)
    signal = 1.0 if outcome == "PASS" else -1.0
    candidate = _candidate(signal)
    if outcome == "PASS":
        bars = (
            BarData(origin, 100, 100.5, 99.5, 100, 10, candidate),
            BarData(origin + timedelta(minutes=3), 100, 100.5, 99.5, 100, 10),
            BarData(origin + timedelta(hours=3), 2100, 2100, 2100, 2100, 10, candidate),
            BarData(origin + timedelta(hours=3, minutes=3), 100, 100.5, 99.5, 100, 10),
            BarData(origin + timedelta(days=1, hours=3), 2100, 2100, 2100, 2100, 10),
        )
    else:
        bars = (
            BarData(origin, 100, 100.5, 99.5, 100, 10, candidate),
            BarData(origin + timedelta(minutes=3), 100, 100.5, 99.5, 100, 10),
            BarData(origin + timedelta(minutes=6), -2000, -2000, -2000, -2000, 10),
        )
    return EnvironmentEpisode(ticker, "one_mini", bars)


def _learning_episodes() -> TrainingEpisodes:
    config = _config()
    episodes = tuple(
        episode
        for ticker in TICKERS
        for episode in (_learning_episode(ticker, "PASS"), _learning_episode(ticker, "BLOW"))
    )
    return TrainingEpisodes(
        episodes,
        manifest_sha256="f" * 64,
        config_sha256=config.digest,
        artifact_sha256=config.upstream.embedding_manifest_sha256,
        partition="training",
        fold=0,
        schedule_seed=42,
    )


def _training_episodes() -> TrainingEpisodes:
    config = _config()
    episodes = []
    for index, ticker in enumerate(TICKERS):
        episodes.append(_episode(ticker, "one_mini", 1.0 if index % 2 else -1.0))
        if ticker != "ZB":
            episodes.append(_episode(ticker, "ten_micros", -1.0 if index % 2 else 1.0))
    return TrainingEpisodes(
        tuple(episodes),
        manifest_sha256="a" * 64,
        config_sha256=config.digest,
        artifact_sha256=config.upstream.embedding_manifest_sha256,
        partition="training",
        fold=0,
        schedule_seed=42,
    )


@pytest.mark.parametrize("variant", tuple(PolicyVariant))
def test_public_policy_seam_emits_binary_actions_for_every_variant(variant: PolicyVariant) -> None:
    torch.manual_seed(5)
    model = EntryActorCritic(observation_width=42, variant=variant)
    observations = torch.zeros((4, 42), dtype=torch.float32)
    tickers = torch.tensor([0, 0, 1, 6])
    profiles = torch.tensor([0, 1, 0, 0])

    logits, values = model(observations, tickers, profiles)

    assert logits.shape == (4, 2)
    assert values.shape == (4,)
    assert model.action_names == ("skip", "enter")
    assert "direction" not in model.action_names
    assert "exit" not in model.action_names
    assert "size" not in model.action_names
    assert not torch.equal(logits[0], logits[1]), "profile conditioning has no effect"
    assert not torch.equal(logits[0], logits[2]), "ticker conditioning has no effect"

    trained = _train_loaded_policy(
        _config(),
        _training_episodes(),
        Path("unused"),
        seed=42,
        variant=variant,
        target_updates=1,
        publish=False,
    )
    assert trained["variant"] == variant.value
    assert trained["finite_gradients"] is True


def test_ticker_value_heads_and_normalizers_update_only_for_owners() -> None:
    model = EntryActorCritic(observation_width=42, variant=PolicyVariant.SHARED_TICKER_VALUE)
    normalizers = ReturnNormalizers(TICKERS)
    before = {
        ticker: tuple(parameter.detach().clone() for parameter in model.value_parameters(ticker))
        for ticker in TICKERS
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    observations = torch.ones((4, 42), dtype=torch.float32)
    tickers = torch.zeros(4, dtype=torch.long)
    profiles = torch.zeros(4, dtype=torch.long)

    logits, values = model(observations, tickers, profiles)
    loss = -torch.log_softmax(logits, dim=-1)[:, 1].mean() + (values - 1.0).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    normalizers.update("ES", np.ones(4, dtype=np.float64))

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before["ES"], model.value_parameters("ES"), strict=True)
    )
    for ticker in TICKERS[1:]:
        assert all(
            torch.equal(old, new)
            for old, new in zip(before[ticker], model.value_parameters(ticker), strict=True)
        )
        assert normalizers.count(ticker) == 0
    assert normalizers.count("ES") == 4
    assert any(parameter.grad is not None for parameter in model.actor_parameters())


def test_ticker_cost_heads_update_only_for_owners() -> None:
    model = EntryActorCritic(observation_width=42, variant=PolicyVariant.SHARED_TICKER_VALUE)
    before = {
        ticker: tuple(
            parameter.detach().clone() for parameter in model.cost_value_parameters(ticker)
        )
        for ticker in TICKERS
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    observations = torch.ones((4, 42), dtype=torch.float32)
    tickers = torch.zeros(4, dtype=torch.long)
    profiles = torch.zeros(4, dtype=torch.long)

    _logits, _values, cost_values = model.forward_with_cost(observations, tickers, profiles)
    loss = (cost_values - 1.0).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before["ES"], model.cost_value_parameters("ES"), strict=True)
    )
    for ticker in TICKERS[1:]:
        assert all(
            torch.equal(old, new)
            for old, new in zip(before[ticker], model.cost_value_parameters(ticker), strict=True)
        )


def test_constraint_controller_uses_raw_episode_costs_and_projects_bounds() -> None:
    controller = ConstraintController.from_config(_config())

    lower = controller.update([0.0, 0.0])
    assert lower["lambda_after"] < lower["lambda_before"]
    upper = controller.update([1.0, 1.0])
    assert upper["lambda_after"] > upper["lambda_before"]
    assert controller.update_count == 2
    assert controller.episode_count == 4
    assert controller.cost_sum == 2.0

    controller.lambda_value = 0.0001
    assert controller.update([0.0])["lambda_after"] == 0.0
    controller.lambda_value = controller.lambda_max - 0.001
    saturated = controller.update([1.0])
    assert saturated["lambda_after"] == controller.lambda_max
    assert saturated["dual_saturated"] is True


def test_actor_advantage_uses_exact_normalized_lagrangian_formula() -> None:
    reward = torch.tensor([1.0, -1.0])
    cost = torch.tensor([0.5, -0.5])

    combined = _lagrangian_actor_advantages(reward, cost, dual_lambda=3.0)

    assert torch.equal(combined, torch.tensor([-0.125, 0.125]))


def test_terminal_reward_and_cost_are_unshaped() -> None:
    assert _terminal_signal("PASS") == (1.0, 0.0)
    assert _terminal_signal("BLOW") == (0.0, 1.0)
    assert _terminal_signal("TIMEOUT") == (0.0, 0.0)
    terminal_fixture = [
        _Transition(
            np.zeros(4, dtype=np.float32),
            0,
            0,
            0,
            0.0,
            0.0,
            0.0,
            np.ones(2, dtype=np.bool_),
            reward,
            cost,
            True,
        )
        for reward, cost in ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
    ]
    assert _reward_returns(terminal_fixture).tolist() == [1.0, 0.0, 0.0]
    assert _cost_returns(terminal_fixture).tolist() == [0.0, 1.0, 0.0]
    result = _train_loaded_policy(
        _config(),
        _training_episodes(),
        Path("unused"),
        seed=42,
        variant=PolicyVariant.SHARED_TICKER_VALUE,
        maximum_updates_this_run=1,
        publish=False,
    )

    rewards = result["transition_audit"]["rewards"]
    costs = result["transition_audit"]["costs"]
    assert set(rewards).issubset({0.0, 1.0})
    assert set(costs).issubset({0.0, 1.0})
    assert result["transition_audit"]["nonterminal_reward_sum"] == 0.0
    assert result["transition_audit"]["nonterminal_cost_sum"] == 0.0
    assert result["gamma"] == 1.0
    assert result["reward_shaping"] is False


def test_training_rejects_nontraining_and_stale_identities(tmp_path: Path) -> None:
    episodes = _training_episodes()
    with pytest.raises(ProductionTrainingError, match="training partition"):
        _train_loaded_policy(
            _config(), replace(episodes, partition="validation"), tmp_path / "validation", seed=42
        )
    with pytest.raises(ProductionTrainingError, match="config identity mismatch"):
        _train_loaded_policy(
            _config(),
            replace(episodes, config_sha256="d" * 64),
            tmp_path / "stale",
            seed=42,
        )


def test_public_loader_binds_runtime_schedule_bytes_and_complete_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mantis_v2.rl_training as training

    config = _config()
    episodes = _training_episodes().episodes
    manifest = tmp_path / "schedule.json"
    manifest.write_text(json.dumps({"seed": 42, "episodes": [{} for _ in episodes]}))
    expected_sha = training.sha256_file(manifest)
    runtime = {
        "source": {"revision": "abc", "dirty": False, "sha256": "c" * 64},
        "lock": {"sha256": "d" * 64},
    }
    monkeypatch.setattr(training, "verify_rl_runtime", lambda *_args: runtime)
    monkeypatch.setattr(
        training,
        "load_episode_manifest",
        lambda *_args: SimpleNamespace(
            episodes=episodes,
            manifest_sha256=expected_sha,
            partition="training",
            fold=0,
        ),
    )

    loaded = load_training_episodes(config, manifest, ROOT.parent)

    assert loaded.manifest_sha256 == expected_sha
    assert loaded.runtime_identities == runtime
    assert len(loaded.episodes) == len(episodes)
    assert loaded.collection_sha256

    manifest.write_text(json.dumps({"seed": 42, "episodes": [{}]}))
    with pytest.raises(ProductionTrainingError, match="episode collection mismatch"):
        load_training_episodes(config, manifest, ROOT.parent)


@pytest.mark.parametrize("variant", tuple(PolicyVariant))
def test_public_manifest_bound_trainer_runs_every_variant(
    variant: PolicyVariant, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mantis_v2.rl_training as training

    config = _config()
    episodes = _training_episodes().episodes
    manifest = tmp_path / "schedule.json"
    manifest.write_text(json.dumps({"seed": 42, "episodes": [{} for _ in episodes]}))
    schedule_sha256 = training.sha256_file(manifest)
    monkeypatch.setattr(
        training,
        "verify_rl_runtime",
        lambda *_args: {
            "source": {"revision": "abc", "dirty": False, "sha256": "c" * 64},
            "lock": {"sha256": "d" * 64},
        },
    )
    monkeypatch.setattr(
        training,
        "load_episode_manifest",
        lambda *_args: SimpleNamespace(
            episodes=episodes,
            manifest_sha256=schedule_sha256,
            partition="training",
            fold=0,
        ),
    )

    result = train_entry_policy(
        config,
        manifest,
        tmp_path / variant.value,
        seed=42,
        variant=variant,
        target_updates=1,
        repository_root=ROOT.parent,
    )

    assert result["variant"] == variant.value
    assert result["identities"]["schedule_sha256"] == schedule_sha256
    assert result["identities"]["episode_count"] == len(episodes)
    assert result["identities"]["completion_mode"] == "bounded_updates"


def test_public_learning_controls_detect_reward_and_mask_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mantis_v2.rl_training as training

    config = replace(
        _config(),
        training=replace(_config().training, ppo_epochs=4, minibatch_size=64),
    )
    episodes = _learning_episodes().episodes
    manifest = tmp_path / "learning-schedule.json"
    manifest.write_text(json.dumps({"seed": 42, "episodes": [{} for _ in episodes]}))
    schedule_sha256 = training.sha256_file(manifest)
    monkeypatch.setattr(
        training,
        "verify_rl_runtime",
        lambda *_args: {
            "source": {"revision": "abc", "dirty": False, "sha256": "c" * 64},
            "lock": {"sha256": "d" * 64},
        },
    )
    monkeypatch.setattr(
        training,
        "load_episode_manifest",
        lambda *_args: SimpleNamespace(
            episodes=episodes,
            manifest_sha256=schedule_sha256,
            partition="training",
            fold=0,
        ),
    )

    separations = {}
    for control in (
        TrainingControl.NORMAL,
        TrainingControl.ZERO_REWARD,
        TrainingControl.SHUFFLED_REWARD,
    ):
        result = train_entry_policy(
            config,
            manifest,
            tmp_path / control.value,
            seed=42,
            target_updates=200,
            training_control=control,
            repository_root=ROOT.parent,
            publish=False,
        )
        probabilities = result["policy_diagnostics"]["episode_enter_probabilities"]
        pass_mean = float(np.mean(probabilities[0::2]))
        blow_mean = float(np.mean(probabilities[1::2]))
        separations[control] = pass_mean - blow_mean

    assert separations[TrainingControl.NORMAL] > 0.10
    assert separations[TrainingControl.ZERO_REWARD] < separations[TrainingControl.NORMAL] / 2
    assert separations[TrainingControl.SHUFFLED_REWARD] < 0.0
    with pytest.raises(ProductionTrainingError, match="mask-disabled control"):
        train_entry_policy(
            config,
            manifest,
            tmp_path / "mask-disabled",
            seed=42,
            target_updates=1,
            training_control=TrainingControl.MASK_DISABLED,
            repository_root=ROOT.parent,
        )


def test_balanced_rollouts_finite_gradients_and_policy_diagnostics(tmp_path: Path) -> None:
    result = _train_loaded_policy(
        _config(), _training_episodes(), tmp_path / "run", seed=42, maximum_updates_this_run=1
    )

    assert result["finite_gradients"] is True
    assert result["completed_timesteps"] == result["rollouts"]["bars_replayed"]
    assert result["rollouts"]["ticker_max_minus_min"] <= 1
    assert result["rollouts"]["profile_counts"]["one_mini"] == 7
    assert result["rollouts"]["profile_counts"]["ten_micros"] == 6
    diagnostics = result["policy_diagnostics"]
    assert 0.0 <= diagnostics["enter_rate"] <= 1.0
    assert diagnostics["mean_entropy"] >= 0.0
    assert diagnostics["deterministic_action_count"] in {1, 2}
    assert diagnostics["invalid_actions_observed"] == 0
    assert all(
        np.isfinite(episode["minimum_mll_cushion"]) for episode in result["rollouts"]["episodes"]
    )
    actor_totals = result["metrics"][-1]["actor_weight_totals"]
    assert set(actor_totals) == set(TICKERS)
    assert max(actor_totals.values()) - min(actor_totals.values()) < 1e-7


def test_checkpoint_reload_and_interruption_resume_are_exact(tmp_path: Path) -> None:
    config = _config()
    episodes = _training_episodes()
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    full = _train_loaded_policy(config, episodes, uninterrupted, seed=42, target_updates=2)
    partial = _train_loaded_policy(
        config,
        episodes,
        resumed,
        seed=42,
        target_updates=2,
        maximum_updates_this_run=1,
    )
    assert partial["status"] == "incomplete"
    partial_state = json.loads((resumed / "state.json").read_text())
    assert partial_state["completed_timesteps"] == partial["rollouts"]["bars_replayed"]
    final = _train_loaded_policy(config, episodes, resumed, seed=42, target_updates=2, resume=True)

    assert final["status"] == "complete"
    assert final["deterministic_reload_actions"] is True
    assert final["policy_sha256"] == full["policy_sha256"]
    assert final["optimizer_sha256"] == full["optimizer_sha256"]
    assert final["normalizers"] == full["normalizers"]
    assert final["constraint_controller"] == full["constraint_controller"]
    assert final["metrics"] == full["metrics"]


def test_unconstrained_checkpoint_schema_is_rejected(tmp_path: Path) -> None:
    config = _config()
    episodes = _training_episodes()
    output = tmp_path / "old-schema"
    _train_loaded_policy(
        config,
        episodes,
        output,
        seed=42,
        target_updates=2,
        maximum_updates_this_run=1,
    )
    bundle_path = output / "checkpoints" / "update-000001" / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["schema_version"] = 2
    bundle_path.write_text(json.dumps(bundle))

    with pytest.raises(ProductionTrainingError, match="resume provenance mismatch"):
        _train_loaded_policy(config, episodes, output, seed=42, target_updates=2, resume=True)


def test_actual_replayed_steps_not_nominal_episode_lengths_drive_completion() -> None:
    metrics = [
        {"rollouts": {"bars_replayed": 4906, "nominal_episode_bars": 64096}},
        {"rollouts": {"bars_replayed": 17, "nominal_episode_bars": 64096}},
    ]

    assert _completed_timesteps(metrics) == 4923
    with pytest.raises(ProductionTrainingError, match="timestep metrics"):
        _completed_timesteps([{"rollouts": {"bars_replayed": 0}}])


def test_resume_rejects_provenance_and_all_seed_reporting_has_worst_seed(
    tmp_path: Path,
) -> None:
    config = _config()
    episodes = _training_episodes()
    output = tmp_path / "resume"
    _train_loaded_policy(
        config,
        episodes,
        output,
        seed=42,
        target_updates=2,
        maximum_updates_this_run=1,
    )
    with pytest.raises(ProductionTrainingError, match="resume provenance mismatch"):
        _train_loaded_policy(
            config,
            replace(episodes, manifest_sha256="e" * 64),
            output,
            seed=42,
            target_updates=2,
            resume=True,
        )
    for stale_runtime in (
        {"source": {"revision": "changed", "dirty": False, "sha256": "a" * 64}},
        {"lock": {"sha256": "b" * 64}},
    ):
        stale = replace(episodes, runtime_identities=stale_runtime)
        with pytest.raises(ProductionTrainingError, match="resume provenance mismatch"):
            _train_loaded_policy(
                config,
                stale,
                output,
                seed=42,
                target_updates=2,
                resume=True,
            )

    two_seed_config = replace(
        config,
        training=replace(
            config.training,
            development_seeds=(42, 43),
            confirmation_seeds=(42, 43),
        ),
    )
    two_seed_episodes = replace(episodes, config_sha256=two_seed_config.digest)
    aggregate = _train_policy_seeds_loaded(
        two_seed_config,
        two_seed_episodes,
        tmp_path / "seeds",
        seeds=(42, 43),
        target_updates=1,
    )
    assert [run["seed"] for run in aggregate["runs"]] == [42, 43]
    assert aggregate["reported_seeds"] == [42, 43]
    assert aggregate["worst_seed"] in {42, 43}
    assert "best_seed" not in aggregate
    resumed = _train_policy_seeds_loaded(
        two_seed_config,
        two_seed_episodes,
        tmp_path / "seeds",
        seeds=(42, 43),
        target_updates=1,
        resume=True,
    )
    assert resumed == aggregate
    assert aggregate["completion_mode"] == "bounded_updates"
    assert aggregate["target_updates"] == 1
    assert aggregate["target_timesteps"] is None
    with pytest.raises(ProductionTrainingError, match="completed seed provenance mismatch"):
        _train_policy_seeds_loaded(
            two_seed_config,
            two_seed_episodes,
            tmp_path / "seeds",
            seeds=(42, 43),
            target_updates=None,
            resume=True,
        )


def _policy_transitions(
    model: EntryActorCritic,
    ticker_indices: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0),
) -> list[_Transition]:
    observations = torch.linspace(-1.0, 1.0, len(ticker_indices) * 4).reshape(-1, 4)
    tickers = torch.tensor(ticker_indices, dtype=torch.long)
    profiles = torch.tensor([index % 2 for index in range(len(ticker_indices))])
    masks = torch.ones((len(ticker_indices), 2), dtype=torch.bool)
    actions = torch.tensor([index % 2 for index in range(len(ticker_indices))])
    with torch.no_grad():
        logits, values, cost_values = model.forward_with_cost(observations, tickers, profiles)
        log_probabilities = torch.distributions.Categorical(logits=logits).log_prob(actions)
    result = []
    for index, ticker in enumerate(ticker_indices):
        terminal = index == len(ticker_indices) - 1 or ticker_indices[index + 1] != ticker
        result.append(
            _Transition(
                observations[index].numpy(),
                ticker,
                int(profiles[index]),
                int(actions[index]),
                float(log_probabilities[index]),
                float(values[index]),
                float(cost_values[index]),
                masks[index].numpy(),
                float(terminal and index % 2 == 1),
                float(terminal and index % 2 == 0),
                terminal,
            )
        )
    return result


def test_masked_ppo_reuses_frozen_rollout_and_clips_changed_ratios() -> None:
    torch.manual_seed(8)
    model = EntryActorCritic(4, PolicyVariant.SHARED_TICKER_VALUE, hidden_width=8)
    transitions = _policy_transitions(model)
    frozen = [item.old_log_probability for item in transitions]
    metrics = _ppo_update(
        model,
        torch.optim.Adam(model.parameters(), lr=0.2),
        ReturnNormalizers(TICKERS),
        ConstraintController.from_config(_config()),
        transitions,
        episode_costs=[item.cost for item in transitions if item.terminal],
        epochs=6,
        minibatch_size=4,
    )

    assert [item.old_log_probability for item in transitions] == frozen
    assert metrics["ratio_non_unit_fraction"] > 0.0
    assert metrics["ratio_max_deviation"] > 0.2
    assert metrics["clip_fraction"] > 0.0
    assert metrics["ppo_epochs"] == 6


def test_cost_targets_remain_binary_and_cost_advantages_are_centered_only() -> None:
    torch.manual_seed(18)
    model = EntryActorCritic(4, PolicyVariant.SHARED_TICKER_VALUE, hidden_width=8)
    transitions = _policy_transitions(model)
    transitions = [replace(item, old_cost_value=0.0) for item in transitions]
    transitions[3] = replace(transitions[3], terminal=True, reward=1.0, cost=0.0)
    transitions[-1] = replace(transitions[-1], terminal=True, reward=0.0, cost=1.0)
    controller = ConstraintController.from_config(_config())

    metrics = _ppo_update(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        ReturnNormalizers(TICKERS),
        controller,
        transitions,
        episode_costs=[0.0, 1.0],
        epochs=2,
        minibatch_size=8,
    )

    assert metrics["cost_target_min"] == 0.0
    assert metrics["cost_target_max"] == 1.0
    assert metrics["cost_advantage_mean"] == pytest.approx(0.0)
    assert metrics["cost_advantage_std"] == pytest.approx(0.5)
    assert metrics["dual_update_count"] == 1


def test_shared_actor_transition_weights_equalize_rty_and_zb() -> None:
    tickers = torch.tensor([TICKERS.index("RTY")] * 697 + [TICKERS.index("ZB")] * 93)
    weights, totals = _actor_weights(tickers)

    assert len(weights) == 790
    assert totals["RTY"] == pytest.approx(0.5)
    assert totals["ZB"] == pytest.approx(0.5)
    assert float(weights[:697].sum()) == pytest.approx(float(weights[697:].sum()))
    sampled: Counter[str] = Counter()
    torch.manual_seed(17)
    for _epoch in range(4):
        batches, counts = _balanced_minibatches(tickers, minibatch_size=100)
        sampled.update(counts)
        assert len(batches) == 14
        for batch in batches:
            assert int((tickers[batch] == TICKERS.index("RTY")).sum()) == 50
            assert int((tickers[batch] == TICKERS.index("ZB")).sum()) == 50
    assert sampled == {"RTY": 2800, "ZB": 2800}


def test_independent_actor_update_cannot_change_foreign_actor_bytes() -> None:
    torch.manual_seed(9)
    model = EntryActorCritic(4, PolicyVariant.INDEPENDENT_ACTOR, hidden_width=8)
    foreign_before = [
        parameter.detach().clone() for parameter in model.owned_actor_parameters("NQ")
    ]
    _ppo_update(
        model,
        torch.optim.Adam(model.parameters(), lr=0.05),
        ReturnNormalizers(TICKERS),
        ConstraintController.from_config(_config()),
        _policy_transitions(model),
        episode_costs=[1.0],
        epochs=3,
        minibatch_size=8,
    )

    assert all(
        torch.equal(before, after)
        for before, after in zip(foreign_before, model.owned_actor_parameters("NQ"), strict=True)
    )


def test_orphan_bundle_and_final_publication_resume_without_retraining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mantis_v2.rl_training as training

    config = _config()
    episodes = _training_episodes()
    reference = _train_loaded_policy(
        config, episodes, tmp_path / "reference", seed=42, target_updates=1
    )
    original_atomic = training._atomic_json
    state_failed = False

    def fail_state_once(path: Path, payload: dict[str, object]) -> None:
        nonlocal state_failed
        if path.name == "state.json" and not state_failed:
            state_failed = True
            raise OSError("injected crash after bundle rename")
        original_atomic(path, payload)

    monkeypatch.setattr(training, "_atomic_json", fail_state_once)
    orphan = tmp_path / "orphan"
    with pytest.raises(OSError, match="after bundle rename"):
        _train_loaded_policy(config, episodes, orphan, seed=42, target_updates=1)
    monkeypatch.setattr(training, "_atomic_json", original_atomic)
    recovered = _train_loaded_policy(
        config, episodes, orphan, seed=42, target_updates=1, resume=True
    )
    assert recovered["policy_sha256"] == reference["policy_sha256"]
    assert recovered["optimizer_sha256"] == reference["optimizer_sha256"]
    assert recovered["metrics"] == reference["metrics"]

    metrics_failed = False

    def fail_metrics_once(path: Path, payload: dict[str, object]) -> None:
        nonlocal metrics_failed
        if path.name == "metrics.json" and not metrics_failed:
            metrics_failed = True
            raise OSError("injected crash before final publication")
        original_atomic(path, payload)

    monkeypatch.setattr(training, "_atomic_json", fail_metrics_once)
    finalizing = tmp_path / "finalizing"
    with pytest.raises(OSError, match="before final publication"):
        _train_loaded_policy(config, episodes, finalizing, seed=42, target_updates=1)
    monkeypatch.setattr(training, "_atomic_json", original_atomic)
    finalized = _train_loaded_policy(
        config, episodes, finalizing, seed=42, target_updates=1, resume=True
    )
    assert finalized == reference
    assert (finalizing / "manifest.json").is_file()


def test_cli_exposes_production_training_seam(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel: Any = object()
    monkeypatch.setattr(cli, "load_rl_config", lambda _path: sentinel)
    called: dict[str, object] = {}

    def train(config: object, manifest: Path, output: Path, **_kwargs: object) -> dict[str, str]:
        called.update(config=config, manifest=manifest, output=output)
        return {"stage": "rl-train"}

    monkeypatch.setattr(cli, "train_policy_seeds", train)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mantis-v2",
            "rl-train",
            "--config",
            "rl.toml",
            "--training-manifest",
            "training.json",
            "--output",
            "result",
            "--variant",
            "shared_ticker_value",
        ],
    )

    cli.main()

    assert json.loads(capsys.readouterr().out) == {"stage": "rl-train"}
    assert called == {
        "config": sentinel,
        "manifest": Path("training.json"),
        "output": Path("result"),
    }
