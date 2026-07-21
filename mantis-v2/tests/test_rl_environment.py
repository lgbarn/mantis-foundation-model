from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from mantis_v2 import cli
from mantis_v2.rl_baselines import (
    BaselineContractError,
    MatchedRandomPolicy,
    RejectAllPolicy,
    SupervisedRows,
    TakeAllPolicy,
    fit_supervised_baselines,
    replay_policy,
)
from mantis_v2.rl_config import load_rl_config
from mantis_v2.rl_environment import (
    BarData,
    CandidateData,
    EnvironmentContractError,
    EnvironmentEpisode,
    TopstepEntryEnvironment,
)
from mantis_v2.rl_validation import FeatureRef, LoadedEpisodes, validate_environment

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")


def _candidate(direction: int = 1, label: int = 1) -> CandidateData:
    return CandidateData(
        embedding=np.array([1.0, -2.0], dtype=np.float32),
        direction=direction,
        trend_line=99.5,
        atr=2.0,
        bars_since_direction_change=3,
        label=label,
    )


def _episode(*, profile: str = "one_mini") -> EnvironmentEpisode:
    first = datetime(2025, 1, 6, 15, 0, tzinfo=UTC)
    bars = (
        BarData(first, 100.0, 101.0, 99.0, 100.0, 10.0, _candidate()),
        BarData(first + timedelta(minutes=3), 100.5, 101.5, 100.0, 101.0, 11.0),
        BarData(first + timedelta(minutes=6), 101.0, 102.0, 100.5, 101.5, 12.0),
        BarData(first + timedelta(minutes=9), 101.5, 102.0, 101.0, 101.75, 13.0),
    )
    return EnvironmentEpisode("ES", profile, bars)


def test_reset_step_mask_and_next_open_fill_are_deterministic() -> None:
    first = TopstepEntryEnvironment(_config(), _episode())
    second = TopstepEntryEnvironment(_config(), _episode())

    first_observation, first_info = first.reset(seed=17)
    second_observation, second_info = second.reset(seed=17)

    assert np.array_equal(first_observation.vector, second_observation.vector)
    assert first_info == second_info
    assert first.action_mask().tolist() == [True, True]
    assert first_observation.schema_version == 1
    assert first_observation.action_mask.tolist() == [True, True]
    assert first.observation_schema.index("quantity") >= 0

    next_observation, reward, terminated, truncated, info = first.step(1)

    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["fill_timestamp"] == "2025-01-06T15:03:00+00:00"
    assert info["fill_price"] == pytest.approx(100.5)
    assert next_observation.positioned == 1.0
    assert first.action_mask().tolist() == [True, False]
    assert next_observation.action_mask.tolist() == [True, False]
    with pytest.raises(EnvironmentContractError, match="invalid action"):
        first.step(1)


def test_observation_prefix_cannot_see_future_bars_or_labels() -> None:
    original = _episode()
    changed_future = EnvironmentEpisode(
        original.ticker,
        original.profile,
        (
            original.bars[0],
            original.bars[1],
            BarData(
                original.bars[2].timestamp,
                10_000.0,
                20_000.0,
                1.0,
                9_000.0,
                99_000.0,
                _candidate(direction=-1, label=0),
            ),
            original.bars[3],
        ),
    )

    before, _ = TopstepEntryEnvironment(_config(), original).reset(seed=5)
    after, _ = TopstepEntryEnvironment(_config(), changed_future).reset(seed=5)

    assert np.array_equal(before.vector, after.vector)


@pytest.mark.parametrize(
    ("profile", "mini", "micro", "quantity", "fee"),
    [("one_mini", 1.0, 0.0, 1.0, 3.78), ("ten_micros", 0.0, 1.0, 10.0, 12.20)],
)
def test_observation_schema_v1_exposes_profile_economics(
    profile: str, mini: float, micro: float, quantity: float, fee: float
) -> None:
    environment = TopstepEntryEnvironment(_config(), _episode(profile=profile))
    observation, _ = environment.reset(seed=1)
    schema = environment.observation_schema

    assert observation.vector[schema.index("contract_is_mini")] == mini
    assert observation.vector[schema.index("contract_is_micro")] == micro
    assert observation.vector[schema.index("quantity")] == quantity
    assert observation.vector[schema.index("dollar_stop_risk")] == pytest.approx(50.0)
    assert observation.vector[schema.index("tick_size")] == pytest.approx(0.25)
    assert observation.vector[schema.index("aggregate_tick_value")] == pytest.approx(12.5)
    assert observation.vector[schema.index("booked_round_trip_fee")] == pytest.approx(fee)


def test_deterministic_baselines_share_mechanics_and_match_entry_oracles() -> None:
    reject = replay_policy(_config(), _episode(), RejectAllPolicy())
    take = replay_policy(_config(), _episode(), TakeAllPolicy())
    random_first = replay_policy(
        _config(),
        _episode(),
        MatchedRandomPolicy(take_count=1, legal_opportunities=1, seed=23),
    )
    random_second = replay_policy(
        _config(),
        _episode(),
        MatchedRandomPolicy(take_count=1, legal_opportunities=1, seed=23),
    )

    assert reject.accepted_trades == 0
    assert reject.actions == (0, 0, 0)
    assert take.accepted_trades == 1
    assert take.actions[0] == 1
    assert take.ending_balance == pytest.approx(100_033.72)
    assert random_first == random_second
    assert random_first.accepted_trades == 1


def test_supervised_baselines_fit_training_only_and_select_on_validation() -> None:
    training = SupervisedRows(
        np.array(
            [[-3.0, 0.0], [-2.0, 1.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]],
            dtype=np.float32,
        ),
        np.array([0, 0, 0, 1, 1, 1], dtype=np.int8),
        "training",
    )
    validation = SupervisedRows(
        np.array([[-2.5, 0.0], [-0.5, 1.0], [0.5, 0.0], [2.5, 1.0]], dtype=np.float32),
        np.array([0, 0, 1, 1], dtype=np.int8),
        "validation",
    )

    logistic, hist, evidence = fit_supervised_baselines(training, validation, seed=42)

    assert logistic.name == "historical_rejected_logistic_head"
    assert hist.name == "hist_gradient_boosting_contextual"
    assert evidence.threshold_source == "validation"
    assert evidence.training_rows == 6
    assert evidence.validation_rows == 4
    assert 0.0 <= evidence.logistic_threshold <= 1.0
    assert 0.0 <= evidence.hist_gradient_boosting_threshold <= 1.0
    with pytest.raises(BaselineContractError, match="expected training rows"):
        fit_supervised_baselines(
            SupervisedRows(training.features, training.labels, "test"), validation, seed=42
        )


def _baseline_episode(profile: str = "one_mini") -> EnvironmentEpisode:
    first = datetime(2025, 1, 6, 15, 0, tzinfo=UTC)
    bars = tuple(
        BarData(
            first + timedelta(minutes=3 * index),
            100.0 + index * 0.1,
            100.5 + index * 0.1,
            99.5 + index * 0.1,
            100.1 + index * 0.1,
            10.0 + index,
            _candidate(label=index % 2) if index < 6 else None,
        )
        for index in range(7)
    )
    return EnvironmentEpisode("ES", profile, bars)


def test_environment_validation_emits_baselines_benchmarks_and_provenance(
    tmp_path: Path,
) -> None:
    features = tmp_path / "features.npy"
    np.save(features, np.arange(16, dtype=np.float16).reshape(8, 2), allow_pickle=False)
    refs = tuple(FeatureRef(features, index, 2) for index in range(8))
    training = LoadedEpisodes(
        (_baseline_episode(), _baseline_episode("ten_micros")),
        refs,
        "a" * 64,
        "training",
    )
    validation = LoadedEpisodes(
        (_baseline_episode(), _baseline_episode("ten_micros")),
        refs,
        "b" * 64,
        "validation",
    )

    result = validate_environment(_config(), training, validation)

    assert result["stage"] == "rl-environment-validation"
    assert result["sealed_holdout_accessed"] is False
    assert result["finite_observations"] is True
    assert result["causal_prefix"] is True
    assert result["shared_action_mask"] is True
    assert result["action_mask_parity"]["mismatches"] == 0
    assert result["baseline_fit"]["threshold_source"] == "validation"
    assert {
        replay["policy"] for episode in result["baseline_replays"] for replay in episode["results"]
    } == {
        "reject_all",
        "take_all",
        "matched_random_take",
        "historical_rejected_logistic_head",
        "hist_gradient_boosting_contextual",
    }
    assert result["benchmark"]["mmap_fetch"]["samples"] == 1000
    assert result["benchmark"]["environment"]["steps"] == 10_000
    assert result["host"]["machine"]
    for episode in result["baseline_replays"]:
        by_name = {item["policy"]: item for item in episode["results"]}
        assert (
            by_name["matched_random_take"]["accepted_trades"]
            == by_name["hist_gradient_boosting_contextual"]["accepted_trades"]
        )


def test_cli_exposes_manifest_backed_environment_validation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = object()
    expected = {"stage": "rl-environment-validation"}
    monkeypatch.setattr(cli, "load_rl_config", lambda _path: sentinel)
    monkeypatch.setattr(cli, "write_environment_validation", lambda *args: expected)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mantis-v2",
            "rl-validate-environment",
            "--config",
            "rl.toml",
            "--training-manifest",
            "training.json",
            "--validation-manifest",
            "validation.json",
            "--output",
            "result.json",
        ],
    )

    cli.main()

    assert json.loads(capsys.readouterr().out) == expected


def test_discontinuity_cancels_pending_entry_and_resets_legality() -> None:
    episode = _episode()
    discontinuous = EnvironmentEpisode(
        episode.ticker,
        episode.profile,
        (
            episode.bars[0],
            BarData(
                episode.bars[1].timestamp,
                episode.bars[1].open,
                episode.bars[1].high,
                episode.bars[1].low,
                episode.bars[1].close,
                episode.bars[1].volume,
                _candidate(),
                True,
            ),
            *episode.bars[2:],
        ),
    )
    environment = TopstepEntryEnvironment(_config(), discontinuous)
    environment.reset(seed=2)

    observation, _, _, _, info = environment.step(1)

    assert info["event"] == "DISCONTINUITY_RESET"
    assert observation.positioned == 0.0
    assert environment.account_state["accepted_trades"] == 0
    assert environment.action_mask().tolist() == [True, False]


def test_prior_stop_is_checked_before_current_bar_can_tighten_trail() -> None:
    first = datetime(2025, 1, 6, 15, 0, tzinfo=UTC)
    episode = EnvironmentEpisode(
        "ES",
        "one_mini",
        (
            BarData(first, 100.0, 100.0, 100.0, 100.0, 10.0, _candidate()),
            BarData(first + timedelta(minutes=3), 100.0, 102.5, 99.5, 102.0, 10.0),
            BarData(first + timedelta(minutes=6), 102.0, 102.0, 101.5, 101.75, 10.0),
            BarData(first + timedelta(minutes=9), 101.75, 101.75, 101.0, 101.0, 10.0),
        ),
    )
    environment = TopstepEntryEnvironment(_config(), episode)
    environment.reset(seed=2)

    first_hold, _, _, _, first_info = environment.step(1)
    second_hold, _, _, _, second_info = environment.step(0)

    assert first_hold.positioned == 1.0
    assert "event" not in first_info
    assert second_hold.positioned == 0.0
    assert second_info["event"] == "STOP"


@pytest.mark.parametrize(
    ("ticker", "profile"),
    (("ES", "one_mini"), ("ES", "ten_micros"), ("NQ", "one_mini"), ("ZB", "one_mini")),
)
def test_baseline_replay_is_deterministic_across_ticker_and_profile(
    ticker: str, profile: str
) -> None:
    base = _baseline_episode(profile)
    episode = EnvironmentEpisode(ticker, profile, base.bars)

    first = replay_policy(_config(), episode, TakeAllPolicy())
    second = replay_policy(_config(), episode, TakeAllPolicy())

    assert first == second
    assert first.accepted_trades == 1


def test_mll_and_dll_boundaries_use_marked_equity_and_session_reset() -> None:
    first = datetime(2025, 1, 6, 15, 0, tzinfo=UTC)
    blow = EnvironmentEpisode(
        "NQ",
        "one_mini",
        (
            BarData(first, 100.0, 100.0, 100.0, 100.0, 1.0, _candidate()),
            BarData(first + timedelta(minutes=3), 100.0, 100.0, 100.0, 100.0, 1.0),
            BarData(first + timedelta(minutes=6), -100.0, -100.0, -100.0, -100.0, 1.0),
        ),
    )
    blow_result = replay_policy(_config(), blow, TakeAllPolicy())

    assert blow_result.status == "BLOW"

    dll = EnvironmentEpisode(
        "ES",
        "one_mini",
        (
            BarData(first, 100.0, 100.0, 100.0, 100.0, 1.0, _candidate()),
            BarData(first + timedelta(minutes=3), 100.0, 100.0, 100.0, 100.0, 1.0),
            BarData(first + timedelta(minutes=6), 55.0, 55.0, 55.0, 55.0, 1.0),
            BarData(
                datetime(2025, 1, 6, 23, 0, tzinfo=UTC),
                55.0,
                55.0,
                55.0,
                55.0,
                1.0,
                _candidate(),
            ),
            BarData(datetime(2025, 1, 6, 23, 3, tzinfo=UTC), 55.0, 55.0, 55.0, 55.0, 1.0),
        ),
    )
    environment = TopstepEntryEnvironment(_config(), dll)
    environment.reset(seed=3)
    environment.step(1)
    _, _, _, _, dll_info = environment.step(0)

    assert dll_info["event"] == "DLL_LOCKOUT"
    assert environment.account_state["entry_locked"] is True
    environment.step(0)
    assert environment.account_state["entry_locked"] is False
    assert environment.action_mask().tolist() == [True, True]
