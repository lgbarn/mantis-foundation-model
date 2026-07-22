from __future__ import annotations

import json
from concurrent.futures import Future
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from mantis_v2 import cli
from mantis_v2 import rl_validation as validation_module
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


def test_delayed_entry_order_preserves_original_candidate_payload_and_identity() -> None:
    first = datetime(2025, 1, 6, 15, 0, tzinfo=UTC)
    original = CandidateData(
        embedding=np.array([1.0, -2.0], dtype=np.float32),
        direction=1,
        trend_line=99.0,
        atr=2.0,
        bars_since_direction_change=3,
        label=1,
    )
    later = CandidateData(
        embedding=np.array([-5.0, 8.0], dtype=np.float32),
        direction=-1,
        trend_line=112.0,
        atr=8.0,
        bars_since_direction_change=0,
        label=0,
    )
    episode = EnvironmentEpisode(
        "NQ",
        "one_mini",
        (
            BarData(first, 100.0, 101.0, 99.0, 100.0, 10.0, original),
            BarData(first + timedelta(minutes=3), 110.0, 111.0, 109.0, 110.0, 11.0, later),
            BarData(first + timedelta(minutes=6), 105.0, 105.0, 105.0, 105.0, 12.0),
        ),
    )
    environment = TopstepEntryEnvironment(_config(), episode)
    environment.reset(seed=42)
    opportunity_identity = f"{first.isoformat()}:original-order"
    order = environment.capture_entry_order(opportunity_identity)

    environment.step(0)
    _observation, _reward, _terminated, _truncated, info = environment.step(1, entry_order=order)

    assert order.candidate is original
    assert order.decision_close == 100.0
    assert info["entry_opportunity_identity"] == opportunity_identity
    assert info["entry_direction"] == 1
    assert info["entry_atr"] == 2.0
    assert info["entry_risk_price"] == 1.0
    assert info["fill_price"] == 105.0
    assert info["entry_stop_price"] == 104.0
    assert environment.account_state["accepted_trades"] == 1


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


def test_gap_stress_adjusts_only_an_actual_adverse_gap_fill() -> None:
    environment = TopstepEntryEnvironment(_config(), _episode(), gap_adverse_extra_ticks=1.0)
    environment.reset(seed=17)

    _, _, _, _, info = environment.step(1)

    assert info["gap_adjusted_fill"] is True
    assert info["fill_price"] == pytest.approx(100.75)
    assert info["gap_extra_cost"] == pytest.approx(12.5)

    skipped = TopstepEntryEnvironment(_config(), _episode(), gap_adverse_extra_ticks=1.0)
    skipped.reset(seed=17)
    _, _, _, _, skipped_info = skipped.step(0)
    assert "gap_adjusted_fill" not in skipped_info


def test_same_bar_stop_and_activation_is_counted_and_resolved_stop_first() -> None:
    first = datetime(2025, 1, 6, 15, 0, tzinfo=UTC)
    episode = EnvironmentEpisode(
        "ES",
        "one_mini",
        (
            BarData(first, 100.0, 100.0, 100.0, 100.0, 10.0, _candidate()),
            BarData(first + timedelta(minutes=3), 100.0, 102.0, 99.0, 100.0, 10.0),
            BarData(first + timedelta(minutes=6), 100.0, 100.0, 100.0, 100.0, 10.0),
        ),
    )
    environment = TopstepEntryEnvironment(_config(), episode)
    environment.reset(seed=17)

    _, _, _, _, info = environment.step(1)

    assert info["event"] == "STOP"
    assert environment.account_state["ambiguity_count"] == 1


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
    assert result["independent_replay_oracles"]["passed"] is True
    assert result["independent_replay_oracles"]["case_count"] == 17
    economics = [case for case in result["independent_replay_oracles"]["cases"] if "ticker" in case]
    assert len(economics) == 13
    assert all(case["actions"] == [1, 0, 0, 0, 0] for case in economics)
    assert all(case["accepted_trades"] == 1 for case in economics)
    assert all(case["status"] == "TIMEOUT" for case in economics)
    assert all(case["checks"]["pre_activation_retrace"] for case in economics)
    assert all(case["checks"]["quantity"] for case in economics)
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
    benchmark = result["benchmark"]["environment"]
    assert benchmark["benchmark_kind"] == "aggregate_vector_environment_throughput"
    assert benchmark["workers"] == 7
    assert benchmark["warmup_steps"] == 7_000
    assert benchmark["samples"] == 7
    assert all(steps >= 14_000 for steps in benchmark["sample_steps"])
    assert all(seconds >= 2.0 for seconds in benchmark["sample_elapsed_seconds"])
    assert result["host"]["machine"]
    for episode in result["baseline_replays"]:
        by_name = {item["policy"]: item for item in episode["results"]}
        assert (
            by_name["matched_random_take"]["accepted_trades"]
            == by_name["hist_gradient_boosting_contextual"]["accepted_trades"]
        )


class _ImmediateExecutor:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.max_workers: int | None = None

    def __enter__(self) -> _ImmediateExecutor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def submit(self, _function: object, *_args: object) -> Future[int]:
        self.events.append("submit")
        future: Future[int] = Future()
        future.set_result(int(_args[-1]))
        return future


def test_collector_window_times_reset_work_and_aggregates_exactly_seven_workers() -> None:
    events: list[str] = []
    executor = _ImmediateExecutor(events)
    times = iter((0.0, 2.0))

    def clock() -> float:
        events.append("clock")
        return next(times)

    transitions, elapsed, rate = validation_module._measure_collector_window(
        executor, _config(), _baseline_episode(), 7, clock=clock
    )

    assert transitions == 14_000
    assert elapsed == 2.0
    assert rate == 7_000.0
    assert events == ["clock", *("submit" for _worker in range(7)), "clock"]


def test_throughput_benchmark_excludes_warmup_and_uses_seven_sample_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    executor = _ImmediateExecutor(events)
    rates = iter((1_000.0, 2_000.0, 3_000.0, 6_000.0, 7_000.0, 8_000.0, 9_000.0))
    monkeypatch.setattr(
        validation_module,
        "ProcessPoolExecutor",
        lambda max_workers: executor,
    )
    monkeypatch.setattr(
        validation_module,
        "_warmup_collector",
        lambda _executor, _config, _episode, worker_count: events.append("warmup") or 7_000,
    )
    monkeypatch.setattr(
        validation_module,
        "_measure_collector_window",
        lambda _executor, _config, _episode, _worker_count: (
            events.append("measure") or 14_000,
            2.0,
            next(rates),
        ),
    )

    result = validation_module._throughput_benchmark(_config(), _baseline_episode())

    assert events == ["warmup", *("measure" for _sample in range(7))]
    assert result["workers"] == 7
    assert result["warmup_steps"] == 7_000
    assert result["samples"] == 7
    assert result["steps_per_second"] == 6_000.0
    assert result["minimum_steps_per_second"] == 1_000.0
    assert result["p25_steps_per_second"] == 2_500.0
    assert result["maximum_steps_per_second"] == 9_000.0


@pytest.mark.parametrize("rate", (5_000.0, 5_001.0))
def test_environment_throughput_gate_accepts_at_least_5000(rate: float) -> None:
    validation_module._require_environment_throughput(rate)


def test_environment_throughput_gate_rejects_4999_999() -> None:
    with pytest.raises(
        validation_module.EnvironmentValidationError,
        match="below 5,000 steps/second",
    ):
        validation_module._require_environment_throughput(4_999.999)


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
