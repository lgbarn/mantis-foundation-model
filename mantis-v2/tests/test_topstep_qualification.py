from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest
from mantis_v2 import (
    MNQDecision,
    QualificationDay,
    Topstep100KRules,
    TopstepQualification,
    TopstepQualificationError,
)


def _decision(
    day: date,
    *,
    decision_bar: int = 1,
    entry_ticks: int = 80000,
    stop_ticks: int = 79980,
    exit_ticks: int = 80020,
    worst_ticks: int = 79980,
) -> MNQDecision:
    start = datetime(day.year, day.month, day.day, 15, tzinfo=UTC)
    return MNQDecision(
        decision_id=f"{day.isoformat()}-{decision_bar}",
        decision_ts=start,
        entry_ts=start + timedelta(minutes=3),
        exit_ts=start + timedelta(minutes=6),
        decision_bar=decision_bar,
        entry_bar=decision_bar + 1,
        side="long",
        entry_ticks=entry_ticks,
        stop_ticks=stop_ticks,
        exit_ticks=exit_ticks,
        worst_ticks=worst_ticks,
    )


def _day(number: int, *decisions: MNQDecision) -> QualificationDay:
    return QualificationDay(date(2026, 1, number), decisions)


def test_rules_are_versioned_and_fail_closed_on_any_changed_state() -> None:
    rules = Topstep100KRules()
    assert rules.schema_version == 1
    assert rules.snapshot == "topstep-100k-2026-07-20"
    assert rules.maximum_mnq == 10
    with pytest.raises(TopstepQualificationError, match="rules do not match"):
        TopstepQualification(replace(rules, profit_target_cents=500_000))


def test_stop_distance_sizing_includes_costs_headroom_budget_and_ramp() -> None:
    qualification = TopstepQualification(Topstep100KRules())
    # 20 ticks * $0.50 + $2.22 round-trip cost = $12.22 risk per MNQ.
    assert qualification.mnq_quantity(headroom_cents=300_000, stop_ticks=20, session=1) == 2
    assert qualification.mnq_quantity(headroom_cents=300_000, stop_ticks=20, session=6) == 5
    assert qualification.mnq_quantity(headroom_cents=300_000, stop_ticks=20, session=11) == 10
    # 5% of $1,000 is $50, so only four contracts fit after costs.
    assert qualification.mnq_quantity(headroom_cents=100_000, stop_ticks=20, session=11) == 4
    assert qualification.mnq_quantity(headroom_cents=10_000_000, stop_ticks=1, session=99) == 10


def test_replay_enforces_next_bar_session_exit_daily_stop_and_stop_breaker() -> None:
    qualification = TopstepQualification(Topstep100KRules())
    stale = replace(_decision(date(2026, 1, 2)), entry_bar=3)
    with pytest.raises(TopstepQualificationError, match="next bar"):
        qualification.replay((_day(2, stale),))

    late = replace(_decision(date(2026, 1, 2)), exit_ts=datetime(2026, 1, 2, 21, 1, tzinfo=UTC))
    with pytest.raises(TopstepQualificationError, match="force-flat"):
        qualification.replay((_day(2, late),))

    stopped_once = _decision(date(2026, 1, 2), exit_ticks=79980)
    stopped_twice = replace(
        _decision(date(2026, 1, 2), decision_bar=4, exit_ticks=79980),
        decision_ts=datetime(2026, 1, 2, 15, 9, tzinfo=UTC),
        entry_ts=datetime(2026, 1, 2, 15, 12, tzinfo=UTC),
        exit_ts=datetime(2026, 1, 2, 15, 15, tzinfo=UTC),
    )
    ignored = replace(
        _decision(date(2026, 1, 2), decision_bar=7),
        decision_ts=datetime(2026, 1, 2, 15, 18, tzinfo=UTC),
        entry_ts=datetime(2026, 1, 2, 15, 21, tzinfo=UTC),
        exit_ts=datetime(2026, 1, 2, 15, 24, tzinfo=UTC),
    )
    replay = qualification.replay((_day(2, stopped_once, stopped_twice, ignored),))
    assert len(replay.trades) == 2
    assert replay.days[0].disabled_reason == "two consecutive full-risk stops"

    large_loss = _decision(date(2026, 1, 3), stop_ticks=79980, exit_ticks=79000, worst_ticks=79000)
    replay = qualification.replay((_day(3, large_loss),))
    assert replay.days[0].net_pnl_cents == -45_000
    assert replay.days[0].disabled_reason == "daily loss stop"


def test_replay_handles_mll_buffer_trailing_lock_consistency_pass_and_failure() -> None:
    qualification = TopstepQualification(Topstep100KRules())
    buffer_touch = _decision(
        date(2026, 1, 2), stop_ticks=79980, exit_ticks=77249, worst_ticks=77249
    )
    replay = qualification.replay((_day(2, buffer_touch),))
    assert replay.status == "active"
    assert replay.days[0].disabled_reason == "minimum MLL buffer"

    buffer_touch = _decision(
        date(2026, 1, 2), stop_ticks=79980, exit_ticks=74000, worst_ticks=74000
    )
    replay = qualification.replay((_day(2, buffer_touch),))
    assert replay.status == "failed"
    assert replay.reason == "maximum loss limit breached intraday"

    wins = []
    for number in range(2, 14):
        decision = _decision(date(2026, 1, number), exit_ticks=80620, worst_ticks=80000)
        wins.append(_day(number, decision))
    replay = qualification.replay(tuple(wins))
    assert replay.status == "passed"
    assert replay.maximum_loss_floor_cents == 10_000_000
    assert replay.consistency_ratio <= 0.5
    assert replay.time_to_pass is not None


def test_stationary_bootstrap_is_reproducible_ordered_and_reports_fixed_gate() -> None:
    days = tuple(
        _day(number, _decision(date(2026, 1, number), exit_ticks=80100, worst_ticks=80000))
        for number in range(2, 16)
    )
    qualification = TopstepQualification(Topstep100KRules(), paths=5_000, seed=42)
    first = qualification.qualify(days, source_hash="a" * 64)
    second = qualification.qualify(days, source_hash="a" * 64)
    assert first == second
    assert first.path_count == 5_000
    assert set(first.outcomes) == {30, 60, 90}
    assert (
        first.outcomes[90].pass_count
        + first.outcomes[90].fail_count
        + first.outcomes[90].unresolved_count
        == 5_000
    )
    assert first.outcomes[30].median_drawdown_cents >= 0
    assert first.outcomes[60].median_consistency_ratio >= 0
    assert first.config_hash and first.source_hash == "a" * 64 and first.input_hash
    assert first.promotion_passed == (
        first.outcomes[90].pass_count / 5_000 >= 0.60
        and first.mll_failures / 5_000 <= 0.20
        and first.median_successful_time_to_pass is not None
        and first.median_successful_time_to_pass <= 60
    )
    assert first.ppo_eligible is first.promotion_passed


def test_days_must_be_complete_chronological_and_decisions_non_overlapping() -> None:
    qualification = TopstepQualification(Topstep100KRules())
    with pytest.raises(TopstepQualificationError, match="chronological"):
        qualification.qualify((_day(3), _day(2)), source_hash="b" * 64)
    overlapping = replace(
        _decision(date(2026, 1, 2), decision_bar=4),
        decision_ts=datetime(2026, 1, 2, 15, 4, tzinfo=UTC),
        entry_ts=datetime(2026, 1, 2, 15, 5, tzinfo=UTC),
        exit_ts=datetime(2026, 1, 2, 15, 8, tzinfo=UTC),
    )
    with pytest.raises(TopstepQualificationError, match="overlap"):
        qualification.replay((_day(2, _decision(date(2026, 1, 2)), overlapping),))

    invalid_worst = replace(_decision(date(2026, 1, 2)), worst_ticks=80001)
    with pytest.raises(TopstepQualificationError, match="worst price"):
        qualification.replay((_day(2, invalid_worst),))


def test_account_day_and_strategy_rth_use_separate_chicago_clocks() -> None:
    qualification = TopstepQualification(Topstep100KRules())
    evening = replace(
        _decision(date(2026, 1, 3)),
        decision_ts=datetime(2026, 1, 2, 23, 0, tzinfo=UTC),
        entry_ts=datetime(2026, 1, 2, 23, 3, tzinfo=UTC),
        exit_ts=datetime(2026, 1, 2, 23, 6, tzinfo=UTC),
    )
    assert qualification.account_session_date(evening.entry_ts) == date(2026, 1, 3)
    with pytest.raises(TopstepQualificationError, match="outside the strategy RTH"):
        qualification.replay((_day(3, evening),))

    before_open = replace(
        _decision(date(2026, 1, 2)),
        decision_ts=datetime(2026, 1, 2, 14, 23, tzinfo=UTC),
        entry_ts=datetime(2026, 1, 2, 14, 26, tzinfo=UTC),
        exit_ts=datetime(2026, 1, 2, 14, 29, tzinfo=UTC),
    )
    with pytest.raises(TopstepQualificationError, match="outside the strategy RTH"):
        qualification.replay((_day(2, before_open),))
