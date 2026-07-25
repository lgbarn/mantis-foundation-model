from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from mantis_v2.expected_r_screen import ExpectedRScreen, ExpectedRScreenConfig


def _frame(direction: int, highs: list[float], lows: list[float]) -> pd.DataFrame:
    rows = len(highs) + 2
    timestamps = pd.date_range("2025-01-02T14:30:00Z", periods=rows, freq="3min")
    close = np.full(rows, 100.0)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": np.array([100.1, 100.1, *highs]),
            "low": np.array([99.9, 99.9, *lows]),
            "close": close,
            "volume": np.ones(rows),
            "trend_magic_direction": np.array([0, direction, *([direction] * len(highs))]),
            "trend_magic_line": np.full(rows, 99.0 if direction > 0 else 101.0),
            "risk_points": np.ones(rows),
        }
    )


def test_candidate_timestamps_are_causal_and_long_trail_matches_oracle() -> None:
    # Decision at row 1, fill at row 2. Price reaches 2.5R, then gives back 0.75R.
    frame = _frame(1, [102.5, 102.0], [100.0, 101.7])
    config = ExpectedRScreenConfig(
        window_bars=2,
        round_trip_commission=1.0,
        point_value=2.0,
        timestamp_semantics="bar_close",
    )

    candidates = ExpectedRScreen(config).generate_candidates(frame)

    row = candidates.iloc[0]
    assert row["decision_ts"] == frame.iloc[1]["timestamp"]
    assert row["entry_ts"] == frame.iloc[2]["timestamp"]
    assert row["outcome_ts"] == frame.iloc[3]["timestamp"]
    assert row["gross_r"] == 1.75
    assert row["net_r"] == 1.25


def test_short_stop_and_target_use_next_open_and_costs() -> None:
    config = ExpectedRScreenConfig(
        window_bars=2,
        slippage_ticks=1,
        tick_size=0.25,
        timestamp_semantics="bar_close",
    )
    stopped = ExpectedRScreen(config).generate_candidates(_frame(-1, [101.0], [99.5])).iloc[0]
    won = (
        ExpectedRScreen(config)
        .generate_candidates(_frame(-1, [99.5, 98.0, 96.5], [98.5, 97.5, 96.0]))
        .iloc[0]
    )

    assert stopped["gross_r"] == -1.0
    assert stopped["net_r"] == -1.25
    assert won["gross_r"] == 3.0
    assert won["net_r"] == 2.75


def test_session_exit_uses_last_completed_bar_before_cutoff() -> None:
    frame = _frame(1, [100.5, 100.5], [99.5, 99.5])
    frame["timestamp"] = pd.date_range("2025-01-02T21:04:00Z", periods=4, freq="3min")
    result = ExpectedRScreen(
        ExpectedRScreenConfig(window_bars=2, timestamp_semantics="bar_close")
    ).generate_candidates(frame)

    assert result.iloc[0]["exit_reason"] == "session"
    assert result.iloc[0]["outcome_ts"] == pd.Timestamp("2025-01-02T21:10:00Z")


def test_raw_bar_open_ohlcv_derives_context_and_records_close_decision() -> None:
    close = np.linspace(100.0, 110.0, 10)
    timestamps = pd.date_range("2025-01-02T14:30:00Z", periods=10, freq="3min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(10),
        }
    )
    config = ExpectedRScreenConfig(
        window_bars=2,
        horizon_bars=1,
        atr_period=2,
        cci_period=3,
        trend_magic_atr_period=2,
    )

    candidates = ExpectedRScreen(config).generate_candidates(frame)

    first = candidates.iloc[0]
    decision_index = int(first["decision_index"])
    assert first["decision_ts"] == timestamps[decision_index] + pd.Timedelta(minutes=3)
    assert first["entry_ts"] == timestamps[decision_index + 1]
    assert first["decision_ts"] == first["entry_ts"]
    assert np.isfinite(first["risk_points"])


def test_split_mask_purges_outcomes_crossing_the_boundary() -> None:
    decisions = pd.Series(pd.to_datetime(["2025-06-30T23:00:00Z", "2025-06-30T20:00:00Z"]))
    outcomes = pd.Series(pd.to_datetime(["2025-07-01T00:00:00Z", "2025-06-30T23:00:00Z"]))

    mask = ExpectedRScreen._date_mask(decisions, outcomes, "2023-07-01", "2025-07-01")

    np.testing.assert_array_equal(mask, [False, True])


def test_trading_session_keys_keep_overnight_rows_in_one_chicago_session() -> None:
    decisions = pd.Series(
        pd.to_datetime(
            [
                "2025-01-02T23:00:00Z",
                "2025-01-03T02:00:00Z",
                "2025-01-03T14:00:00Z",
                "2025-01-03T23:00:00Z",
            ]
        )
    )

    keys = ExpectedRScreen()._session_keys(decisions)

    np.testing.assert_array_equal(
        keys,
        np.array(
            ["2025-01-03", "2025-01-03", "2025-01-03", "2025-01-04"],
            dtype="datetime64[D]",
        ),
    )


def test_fit_retains_rows_freezes_threshold_and_reports_gate(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    rows = 512 + 300
    timestamps = pd.date_range("2023-07-03T14:30:00Z", periods=rows, freq="1h")
    close = 100.0 + rng.normal(0.0, 0.1, rows).cumsum()
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(rows),
            "trend_magic_direction": np.ones(rows, dtype=np.int8),
            "trend_magic_line": close - 1.0,
            "risk_points": np.ones(rows),
        }
    )
    # Exercise the chronology without needing multi-year fixture volume.
    config = replace(
        ExpectedRScreenConfig(window_bars=512, timestamp_semantics="bar_close"),
        train_start="2023-07-24",
        train_end="2023-07-27",
        validation_start="2023-07-27",
        validation_end="2023-07-30",
        test_start="2023-07-30",
        test_end="2023-08-07",
    )
    screen = ExpectedRScreen(config)
    candidates = screen.generate_candidates(frame)
    artifact_path = tmp_path / "screen.json"
    artifact = screen.run(frame, artifact_path)

    assert len(candidates) < rows - config.window_bars
    assert artifact["threshold"]["selected_on"] == "validation"
    assert artifact["splits"]["test"]["threshold"] == artifact["threshold"]["value"]
    assert artifact["features"]["window_bars"] == 512
    assert artifact["features"]["regularization"] == config.ridge_alpha
    assert artifact["rows"]["count"] == len(candidates)
    assert artifact_path.is_file()
    assert set(artifact["gate"]) == {
        "passed",
        "mse_beats_constant",
        "expectancy_positive",
        "expectancy_beats_take_all",
    }
