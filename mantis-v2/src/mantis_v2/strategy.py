"""Causal Supertrend candidates and next-open barrier labels."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from mantis_v2.downstream_config import DownstreamConfig


class StrategyContractError(ValueError):
    """Raised when market data cannot satisfy the strategy contract."""


_TIMEFRAME_MINUTES = {"1min": 1, "3min": 3, "15min": 15}


def market_path(config: DownstreamConfig, symbol: str, timeframe: str) -> Path:
    return config.data.root / f"{symbol}_{timeframe}.csv"


def load_market_frame(config: DownstreamConfig, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load one finite, sorted OHLCV stream and add its causal close timestamp."""
    path = market_path(config, symbol, timeframe)
    if not path.is_file():
        raise StrategyContractError(f"missing configured stream: {path}")
    columns = [config.data.timestamp_column, *config.data.feature_columns]
    frame = pd.read_csv(path, usecols=columns)
    if frame.empty:
        raise StrategyContractError(f"empty stream: {path}")
    timestamp = pd.to_datetime(frame[config.data.timestamp_column], utc=True, errors="raise")
    if not timestamp.is_monotonic_increasing or timestamp.duplicated().any():
        raise StrategyContractError(f"timestamps must be sorted and unique: {path}")
    values = frame.loc[:, config.data.feature_columns].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise StrategyContractError(f"non-finite OHLCV value: {path}")
    frame = frame.loc[:, config.data.feature_columns].astype(np.float64)
    frame.insert(0, "timestamp", timestamp)
    close_timestamp = timestamp
    if config.data.timestamp_semantics == "bar_open":
        close_timestamp = timestamp + pd.Timedelta(minutes=_TIMEFRAME_MINUTES[timeframe])
    frame.insert(1, "close_timestamp", close_timestamp)
    return frame


def wilder_atr(frame: pd.DataFrame, period: int) -> np.ndarray:
    """Compute the FFM-compatible Wilder average true range."""
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    previous = np.concatenate(([close[0]], close[:-1]))
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous), np.abs(low - previous)))
    atr = np.full(len(frame), np.nan, dtype=np.float64)
    if len(frame) < period:
        return atr
    atr[period - 1] = true_range[:period].mean()
    for index in range(period, len(frame)):
        atr[index] = (atr[index - 1] * (period - 1) + true_range[index]) / period
    return atr


def supertrend_state(frame: pd.DataFrame, period: int, multiplier: float) -> np.ndarray:
    """Return +1/-1 state for every bar after Supertrend warmup."""
    atr = wilder_atr(frame, period)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    midpoint = (high + low) / 2.0
    upper = midpoint + multiplier * atr
    lower = midpoint - multiplier * atr
    final_upper = upper.copy()
    final_lower = lower.copy()
    direction = np.zeros(len(frame), dtype=np.int8)
    if len(frame) < period:
        return direction
    direction[period - 1] = 1
    for index in range(period, len(frame)):
        if upper[index] < final_upper[index - 1] or close[index - 1] > final_upper[index - 1]:
            final_upper[index] = upper[index]
        else:
            final_upper[index] = final_upper[index - 1]
        if lower[index] > final_lower[index - 1] or close[index - 1] < final_lower[index - 1]:
            final_lower[index] = lower[index]
        else:
            final_lower[index] = final_lower[index - 1]
        if direction[index - 1] > 0:
            direction[index] = -1 if close[index] < final_lower[index] else 1
        else:
            direction[index] = 1 if close[index] > final_upper[index] else -1
    return direction


def causal_context_indices(
    decision_close_ns: np.ndarray,
    context_close_ns: np.ndarray,
    context_bars: int,
) -> np.ndarray:
    """Map decisions to the rightmost fully closed context bar."""
    indices = np.searchsorted(context_close_ns, decision_close_ns, side="right") - 1
    indices[indices < context_bars - 1] = -1
    return indices.astype(np.int64)


def _label_chunk(
    frame: pd.DataFrame,
    candidate_indices: np.ndarray,
    direction: np.ndarray,
    risk: np.ndarray,
    target_r: float,
    horizon: int,
    cost_r: float,
    effective_horizons: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    entry_indices = candidate_indices + 1
    offsets = np.arange(horizon, dtype=np.int64)
    future_indices = entry_indices[:, None] + offsets[None, :]
    high = frame["high"].to_numpy(dtype=np.float64)[future_indices]
    low = frame["low"].to_numpy(dtype=np.float64)[future_indices]
    close = frame["close"].to_numpy(dtype=np.float64)
    entry = frame["open"].to_numpy(dtype=np.float64)[entry_indices]
    long = direction > 0
    stop = entry - direction * risk
    target = entry + direction * risk * target_r
    stop_touch = np.where(long[:, None], low <= stop[:, None], high >= stop[:, None])
    target_touch = np.where(long[:, None], high >= target[:, None], low <= target[:, None])
    if effective_horizons is None:
        effective_horizons = np.full(len(candidate_indices), horizon, dtype=np.int64)
    valid_path = offsets[None, :] < effective_horizons[:, None]
    stop_touch &= valid_path
    target_touch &= valid_path
    any_stop = stop_touch.any(axis=1)
    any_target = target_touch.any(axis=1)
    first_stop = np.where(any_stop, stop_touch.argmax(axis=1), horizon)
    first_target = np.where(any_target, target_touch.argmax(axis=1), horizon)
    won = any_target & (first_target < first_stop)
    stopped = any_stop & (first_stop <= first_target)
    exit_offset = np.minimum(first_stop, first_target)
    exit_offset = np.where(exit_offset == horizon, effective_horizons - 1, exit_offset)
    horizon_close = close[entry_indices + effective_horizons - 1]
    exit_price = np.where(won, target, np.where(stopped, stop, horizon_close))
    gross_r = direction * (exit_price - entry) / risk
    reward_r = gross_r - cost_r
    path_r = np.where(
        long[:, None],
        (low - entry[:, None]) / risk[:, None],
        (entry[:, None] - high) / risk[:, None],
    )
    before_exit = offsets[None, :] <= exit_offset[:, None]
    mae_r = np.where(before_exit, path_r, np.inf).min(axis=1) - cost_r
    return won.astype(np.int8), reward_r, exit_price, entry_indices + exit_offset, mae_r


def _session_horizons(
    entry_timestamps: pd.Series,
    close_timestamps: pd.Series,
    entry_indices: np.ndarray,
    config: DownstreamConfig,
) -> np.ndarray:
    local = pd.DatetimeIndex(entry_timestamps).tz_convert(config.topstep.session_timezone)
    minutes = local.hour * 60 + local.minute
    start_minutes = config.topstep.session_start_hour * 60
    end_minutes = config.topstep.session_end_hour * 60 + config.topstep.session_end_minute
    after_start = minutes >= start_minutes
    valid_entry = (minutes <= end_minutes) | after_start
    end_local = local.normalize() + timedelta(minutes=int(end_minutes))
    end_local = end_local + pd.to_timedelta(after_start.astype(np.int64), unit="D")
    end_ns = end_local.tz_convert("UTC").to_numpy(dtype="datetime64[ns]")
    close_ns = close_timestamps.to_numpy(dtype="datetime64[ns]")
    final_indices = np.searchsorted(close_ns, end_ns, side="right") - 1
    effective = np.minimum(
        config.strategy.horizon_bars,
        final_indices - entry_indices + 1,
    ).astype(np.int64)
    effective[~valid_entry] = 0
    return np.asarray(effective, dtype=np.int64)


def build_symbol_candidates(
    config: DownstreamConfig,
    symbol: str,
    split: Literal["pre_holdout", "holdout", "all"] = "pre_holdout",
) -> pd.DataFrame:
    """Create every eligible 3-minute state candidate with causal context indices."""
    frames = {
        timeframe: load_market_frame(config, symbol, timeframe)
        for timeframe in config.data.timeframes
    }
    decision = frames[config.data.decision_timeframe]
    strategy = config.strategy
    atr = wilder_atr(decision, strategy.atr_period)
    direction = supertrend_state(
        decision, strategy.supertrend_period, strategy.supertrend_multiplier
    )
    decision_ns = decision["close_timestamp"].to_numpy(dtype="datetime64[ns]")
    aligned = {
        timeframe: causal_context_indices(
            decision_ns,
            frame["close_timestamp"].to_numpy(dtype="datetime64[ns]"),
            config.data.context_bars,
        )
        for timeframe, frame in frames.items()
    }
    last_candidate = len(decision) - strategy.horizon_bars - 1
    legal = np.arange(len(decision), dtype=np.int64)
    legal = legal[
        (legal <= last_candidate)
        & np.isfinite(atr)
        & (atr > 0)
        & (direction != 0)
        & np.logical_and.reduce([indices >= 0 for indices in aligned.values()])
    ]
    holdout_start = (
        pd.Timestamp(config.data.holdout_start).tz_convert("UTC").tz_localize(None).to_datetime64()
    )
    if split == "pre_holdout":
        full_horizon_end = decision_ns[legal + strategy.horizon_bars]
        legal = legal[full_horizon_end < holdout_start]
    elif split == "holdout":
        legal = legal[decision_ns[legal] >= holdout_start]
    elif split != "all":
        raise StrategyContractError(f"unknown candidate split: {split}")
    entry_indices = legal + 1
    entry_timestamps = decision["timestamp"].iloc[entry_indices]
    if config.data.timestamp_semantics == "bar_close":
        entry_timestamps = decision["close_timestamp"].iloc[legal]
    effective_horizons = _session_horizons(
        entry_timestamps,
        decision["close_timestamp"],
        entry_indices,
        config,
    )
    valid_session = effective_horizons > 0
    legal = legal[valid_session]
    effective_horizons = effective_horizons[valid_session]
    if not len(legal):
        raise StrategyContractError(f"no legal candidates for {symbol}")
    labels: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    exit_prices: list[np.ndarray] = []
    exit_indices: list[np.ndarray] = []
    adverse: list[np.ndarray] = []
    analysis_labels: dict[float, list[np.ndarray]] = {
        target: []
        for target in strategy.analysis_targets_r
        if not np.isclose(target, strategy.target_r)
    }
    chunk_rows = 20_000
    for start in range(0, len(legal), chunk_rows):
        selected = legal[start : start + chunk_rows]
        result = _label_chunk(
            decision,
            selected,
            direction[selected],
            atr[selected] * strategy.stop_atr,
            strategy.target_r,
            strategy.horizon_bars,
            strategy.round_trip_cost_r,
            effective_horizons[start : start + chunk_rows],
        )
        labels.append(result[0])
        rewards.append(result[1])
        exit_prices.append(result[2])
        exit_indices.append(result[3])
        adverse.append(result[4])
        for target, target_labels in analysis_labels.items():
            target_result = _label_chunk(
                decision,
                selected,
                direction[selected],
                atr[selected] * strategy.stop_atr,
                target,
                strategy.horizon_bars,
                strategy.round_trip_cost_r,
                effective_horizons[start : start + chunk_rows],
            )
            target_labels.append(target_result[0])
    exit_index = np.concatenate(exit_indices)
    entry_index = legal + 1
    entry_timestamp = entry_timestamps[valid_session].to_numpy()
    output = pd.DataFrame(
        {
            "symbol": symbol,
            "decision_index": legal,
            "decision_ts": decision["close_timestamp"].iloc[legal].to_numpy(),
            "entry_ts": entry_timestamp,
            "label_end_ts": decision["close_timestamp"].iloc[exit_index].to_numpy(),
            "direction": direction[legal],
            "atr": atr[legal],
            "entry_price": decision["open"].iloc[entry_index].to_numpy(),
            "exit_price": np.concatenate(exit_prices),
            "label": np.concatenate(labels),
            "reward_r": np.concatenate(rewards),
            "mae_r": np.concatenate(adverse),
        }
    )
    for timeframe, indices in aligned.items():
        output[f"{timeframe}_index"] = indices[legal]
    output[f"label_target_{strategy.target_r:g}r"] = output["label"]
    for target, labels_for_target in analysis_labels.items():
        output[f"label_target_{target:g}r"] = np.concatenate(labels_for_target)
    output["is_holdout"] = output["decision_ts"] >= pd.Timestamp(config.data.holdout_start)
    return output
