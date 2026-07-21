from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from mantis_v2 import cli
from mantis_v2.rl_config import load_rl_config
from mantis_v2.rl_environment import BarData, CandidateData, EnvironmentEpisode
from mantis_v2.rl_policy import EntryActorCritic, ReturnNormalizers
from mantis_v2.rl_training import (
    PolicyVariant,
    ProductionTrainingError,
    TrainingEpisodes,
    _completed_timesteps,
    train_entry_policy,
    train_policy_seeds,
)

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "CL", "ZB")


def _config():
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    return replace(config, run=replace(config.run, profile="production"))


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
    model = EntryActorCritic(observation_width=42, variant=variant)
    observations = torch.zeros((3, 42), dtype=torch.float32)
    tickers = torch.tensor([0, 1, 6])
    profiles = torch.tensor([0, 1, 0])

    logits, values = model(observations, tickers, profiles)

    assert logits.shape == (3, 2)
    assert values.shape == (3,)
    assert model.action_names == ("skip", "enter")
    assert "direction" not in model.action_names
    assert "exit" not in model.action_names
    assert "size" not in model.action_names


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


def test_terminal_reward_and_cost_are_unshaped() -> None:
    result = train_entry_policy(
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
        train_entry_policy(
            _config(), replace(episodes, partition="validation"), tmp_path / "validation", seed=42
        )
    with pytest.raises(ProductionTrainingError, match="config identity mismatch"):
        train_entry_policy(
            _config(),
            replace(episodes, config_sha256="d" * 64),
            tmp_path / "stale",
            seed=42,
        )


def test_balanced_rollouts_finite_gradients_and_collapse_ablations(tmp_path: Path) -> None:
    result = train_entry_policy(
        _config(), _training_episodes(), tmp_path / "run", seed=42, maximum_updates_this_run=1
    )

    assert result["finite_gradients"] is True
    assert result["completed_timesteps"] == result["rollouts"]["bars_replayed"]
    assert result["rollouts"]["ticker_max_minus_min"] <= 1
    assert result["rollouts"]["profile_counts"]["one_mini"] == 7
    assert result["rollouts"]["profile_counts"]["ten_micros"] == 6
    assert result["ablations"] == {
        "reject_all_detected": True,
        "take_all_detected": True,
        "invalid_action_detected": True,
        "action_collapse_detected": True,
    }


def test_checkpoint_reload_and_interruption_resume_are_exact(tmp_path: Path) -> None:
    config = _config()
    episodes = _training_episodes()
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    full = train_entry_policy(config, episodes, uninterrupted, seed=42, target_updates=2)
    partial = train_entry_policy(
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
    final = train_entry_policy(config, episodes, resumed, seed=42, target_updates=2, resume=True)

    assert final["status"] == "complete"
    assert final["deterministic_reload_actions"] is True
    assert final["policy_sha256"] == full["policy_sha256"]
    assert final["optimizer_sha256"] == full["optimizer_sha256"]
    assert final["normalizers"] == full["normalizers"]
    assert final["metrics"] == full["metrics"]


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
    train_entry_policy(
        config,
        episodes,
        output,
        seed=42,
        target_updates=2,
        maximum_updates_this_run=1,
    )
    with pytest.raises(ProductionTrainingError, match="resume provenance mismatch"):
        train_entry_policy(
            config,
            replace(episodes, manifest_sha256="e" * 64),
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
    aggregate = train_policy_seeds(
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
    resumed = train_policy_seeds(
        two_seed_config,
        two_seed_episodes,
        tmp_path / "seeds",
        seeds=(42, 43),
        target_updates=1,
        resume=True,
    )
    assert resumed == aggregate


def test_cli_exposes_production_training_seam(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel: Any = object()
    monkeypatch.setattr(cli, "load_rl_config", lambda _path: sentinel)
    monkeypatch.setattr(cli, "load_training_episodes", lambda *_args: sentinel)
    monkeypatch.setattr(cli, "train_policy_seeds", lambda *_args, **_kwargs: {"stage": "rl-train"})
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
