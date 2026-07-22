"""Causal strategy candidates and next-open barrier labels."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from mantis_v2.contamination import detect_discontinuities, stream_report
from mantis_v2.corpus import CorpusRepairError, validate_corpus_binding
from mantis_v2.downstream_config import DownstreamConfig, TrendMagicStrategyConfig


class StrategyContractError(ValueError):
    """Raised when market data cannot satisfy the strategy contract."""


_TIMEFRAME_MINUTES = {"1min": 1, "3min": 3, "5min": 5, "15min": 15}


def market_path(config: DownstreamConfig, symbol: str, timeframe: str) -> Path:
    return config.data.root / f"{symbol}_{timeframe}.{config.data.file_format}"


def load_market_frame(config: DownstreamConfig, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load one finite, sorted OHLCV stream and add its causal close timestamp."""
    path = market_path(config, symbol, timeframe)
    if not path.is_file():
        raise StrategyContractError(f"missing configured stream: {path}")
    if config.data.file_format == "parquet":
        assert config.data.corpus_manifest_path is not None
        try:
            validate_corpus_binding(
                config.data.root,
                config.data.corpus_manifest_path,
                config.data.corpus_manifest_sha256,
                [path],
            )
        except CorpusRepairError as exc:
            raise StrategyContractError(str(exc)) from exc
    columns = [config.data.timestamp_column, *config.data.feature_columns]
    if config.data.file_format == "parquet":
        columns.append("quality_flag")
    frame = (
        pd.read_parquet(path, columns=columns)
        if config.data.file_format == "parquet"
        else pd.read_csv(path, usecols=columns)
    )
    if frame.empty:
        raise StrategyContractError(f"empty stream: {path}")
    timestamp = pd.to_datetime(frame[config.data.timestamp_column], utc=True, errors="raise")
    if not timestamp.is_monotonic_increasing or timestamp.duplicated().any():
        raise StrategyContractError(f"timestamps must be sorted and unique: {path}")
    values = frame.loc[:, config.data.feature_columns].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise StrategyContractError(f"non-finite OHLCV value: {path}")
    quality_flag = (
        frame["quality_flag"].to_numpy(dtype=bool)
        if config.data.file_format == "parquet"
        else np.zeros(len(frame), dtype=bool)
    )
    frame = frame.loc[:, config.data.feature_columns].astype(np.float64)
    frame.insert(0, "timestamp", timestamp)
    close_timestamp = timestamp
    if config.data.timestamp_semantics == "bar_open":
        close_timestamp = timestamp + np.timedelta64(_TIMEFRAME_MINUTES[timeframe], "m")
    frame.insert(1, "close_timestamp", close_timestamp)
    boundaries, events = detect_discontinuities(
        timestamp.to_numpy(dtype="datetime64[ns]"),
        frame[["high", "low", "close"]].to_numpy(dtype=np.float64),
        config.data.max_relative_close_jump,
    )
    boundaries |= quality_flag
    frame["_discontinuity"] = boundaries
    frame.attrs["contamination"] = stream_report(f"{symbol}_{timeframe}", events)
    return frame


def _segments(frame: pd.DataFrame) -> list[tuple[int, int]]:
    if "_discontinuity" not in frame:
        return [(0, len(frame))]
    boundaries = np.flatnonzero(frame["_discontinuity"].to_numpy(dtype=bool))
    segments: list[tuple[int, int]] = []
    start = 0
    for boundary in boundaries:
        if start < boundary:
            segments.append((start, int(boundary)))
        start = int(boundary) + 1
    if start < len(frame):
        segments.append((start, len(frame)))
    return segments


def _range_crosses_boundary(
    frame: pd.DataFrame, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    if "_discontinuity" not in frame:
        return np.zeros(len(starts), dtype=bool)
    prefix = np.concatenate(([0], np.cumsum(frame["_discontinuity"].to_numpy(dtype=np.int64))))
    return np.asarray((prefix[ends + 1] - prefix[starts]) > 0, dtype=bool)


def wilder_atr(frame: pd.DataFrame, period: int) -> np.ndarray:
    """Compute the FFM-compatible Wilder average true range."""
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    atr = np.full(len(frame), np.nan, dtype=np.float64)
    for start, end in _segments(frame):
        if end - start < period:
            continue
        segment_high = high[start:end]
        segment_low = low[start:end]
        segment_close = close[start:end]
        previous = np.concatenate(([segment_close[0]], segment_close[:-1]))
        true_range = np.maximum(
            segment_high - segment_low,
            np.maximum(np.abs(segment_high - previous), np.abs(segment_low - previous)),
        )
        atr[start + period - 1] = true_range[:period].mean()
        for index in range(start + period, end):
            local = index - start
            atr[index] = (atr[index - 1] * (period - 1) + true_range[local]) / period
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
    for start, end in _segments(frame):
        if end - start < period:
            continue
        first = start + period - 1
        direction[first] = 1
        for index in range(first + 1, end):
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


def trend_magic_state(
    frame: pd.DataFrame,
    cci_period: int,
    atr_period: int,
    multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the pinned close-CCI/SMA-TR Trend Magic line and state."""
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    line = np.full(len(frame), np.nan, dtype=np.float64)
    direction = np.zeros(len(frame), dtype=np.int8)
    for start, end in _segments(frame):
        length = end - start
        if length < max(cci_period, atr_period):
            continue
        segment_close = close[start:end]
        segment_high = high[start:end]
        segment_low = low[start:end]
        previous_close = np.concatenate(([segment_close[0]], segment_close[:-1]))
        true_range = np.maximum(
            segment_high - segment_low,
            np.maximum(
                np.abs(segment_high - previous_close),
                np.abs(segment_low - previous_close),
            ),
        )
        tr_series = pd.Series(true_range)
        range_sma = tr_series.rolling(atr_period, min_periods=atr_period).mean().to_numpy()
        close_series = pd.Series(segment_close)
        close_sma = close_series.rolling(cci_period, min_periods=cci_period).mean()
        mean_deviation = close_series.rolling(cci_period, min_periods=cci_period).apply(
            lambda values: float(np.mean(np.abs(values - values.mean()))),
            raw=True,
        )
        denominator = 0.015 * mean_deviation.to_numpy()
        cci = np.divide(
            segment_close - close_sma.to_numpy(),
            denominator,
            out=np.full(length, np.nan, dtype=np.float64),
            where=denominator > 0,
        )
        first = max(cci_period, atr_period) - 1
        for local in range(first, length):
            index = start + local
            if not np.isfinite(cci[local]) or not np.isfinite(range_sma[local]):
                continue
            bullish = cci[local] >= 0
            raw_line = (
                segment_low[local] - multiplier * range_sma[local]
                if bullish
                else segment_high[local] + multiplier * range_sma[local]
            )
            if local > first and np.isfinite(line[index - 1]):
                raw_line = (
                    max(raw_line, line[index - 1]) if bullish else min(raw_line, line[index - 1])
                )
            line[index] = raw_line
            direction[index] = 1 if bullish else -1
    return line, direction


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
    mae_r = np.where(stopped, np.maximum(mae_r, reward_r), mae_r)
    return won.astype(np.int8), reward_r, exit_price, entry_indices + exit_offset, mae_r


def _execution_trail_chunk(
    frame: pd.DataFrame,
    candidate_indices: np.ndarray,
    direction: np.ndarray,
    risk: np.ndarray,
    horizon: int,
    effective_horizons: np.ndarray,
    *,
    activation_r: float = 2.0,
    giveback_r: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay the causal prior-bar 2R/0.75R execution trail in R units."""
    entry_indices = candidate_indices + 1
    entry = frame["open"].to_numpy(dtype=np.float64)[entry_indices]
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    stop = entry - direction * risk
    favorable = entry.copy()
    active = np.ones(len(candidate_indices), dtype=bool)
    exit_price = np.full(len(candidate_indices), np.nan, dtype=np.float64)
    exit_index = np.full(len(candidate_indices), -1, dtype=np.int64)
    adverse_r = np.full(len(candidate_indices), np.inf, dtype=np.float64)
    reason = np.full(len(candidate_indices), "", dtype="<U12")

    for offset in range(horizon):
        valid = active & (offset < effective_horizons)
        if not valid.any():
            break
        indices = entry_indices + offset
        bar_high = high[indices]
        bar_low = low[indices]
        stopped = valid & np.where(direction > 0, bar_low <= stop, bar_high >= stop)
        exit_price[stopped] = stop[stopped]
        exit_index[stopped] = indices[stopped]
        reason[stopped] = np.where(
            np.isclose(stop[stopped], entry[stopped] - direction[stopped] * risk[stopped]),
            "initial_stop",
            "trail_stop",
        )
        stop_reward = direction * (stop - entry) / risk
        adverse_r[stopped] = np.minimum(adverse_r[stopped], stop_reward[stopped])
        active[stopped] = False

        surviving = valid & ~stopped
        if surviving.any():
            bar_adverse = np.where(
                direction > 0,
                (bar_low - entry) / risk,
                (entry - bar_high) / risk,
            )
            adverse_r[surviving] = np.minimum(adverse_r[surviving], bar_adverse[surviving])
            favorable[surviving] = np.where(
                direction[surviving] > 0,
                np.maximum(favorable[surviving], bar_high[surviving]),
                np.minimum(favorable[surviving], bar_low[surviving]),
            )
            progress = direction * (favorable - entry) / risk
            armed = surviving & (progress >= activation_r)
            candidate_stop = favorable - direction * giveback_r * risk
            stop[armed] = np.where(
                direction[armed] > 0,
                np.maximum(stop[armed], candidate_stop[armed]),
                np.minimum(stop[armed], candidate_stop[armed]),
            )
            horizon_exit = surviving & (offset == effective_horizons - 1)
            exit_price[horizon_exit] = close[indices[horizon_exit]]
            exit_index[horizon_exit] = indices[horizon_exit]
            reason[horizon_exit] = "horizon"
            active[horizon_exit] = False

    if active.any() or not np.isfinite(exit_price).all() or (exit_index < 0).any():
        raise StrategyContractError("execution trail did not resolve every candidate")
    reward_r = direction * (exit_price - entry) / risk
    return reward_r, exit_price, exit_index, adverse_r, reason


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
    """Create every eligible 3-minute strategy state with causal context indices."""
    frames = {
        timeframe: load_market_frame(config, symbol, timeframe)
        for timeframe in config.data.timeframes
    }
    decision = frames[config.data.decision_timeframe]
    strategy = config.strategy
    atr = wilder_atr(decision, strategy.atr_period)
    if isinstance(strategy, TrendMagicStrategyConfig):
        _, direction = trend_magic_state(
            decision,
            strategy.cci_period,
            strategy.trend_magic_atr_period,
            strategy.trend_magic_multiplier,
        )
    else:
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
    before_context_filter = len(legal)
    clean_context = np.ones(len(legal), dtype=bool)
    for timeframe, frame in frames.items():
        context_end = aligned[timeframe][legal]
        context_start = context_end - config.data.context_bars + 1
        clean_context &= ~_range_crosses_boundary(frame, context_start, context_end)
    legal = legal[clean_context]
    excluded_context = before_context_filter - len(legal)
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
    entry_indices = legal + 1
    label_end_indices = entry_indices + effective_horizons - 1
    clean_label = ~_range_crosses_boundary(decision, legal, label_end_indices)
    excluded_label = int((~clean_label).sum())
    legal = legal[clean_label]
    effective_horizons = effective_horizons[clean_label]
    if not len(legal):
        raise StrategyContractError(f"no legal candidates for {symbol}")
    labels: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    exit_prices: list[np.ndarray] = []
    exit_indices: list[np.ndarray] = []
    adverse: list[np.ndarray] = []
    execution_rewards: list[np.ndarray] = []
    execution_exit_prices: list[np.ndarray] = []
    execution_exit_indices: list[np.ndarray] = []
    execution_adverse: list[np.ndarray] = []
    execution_reasons: list[np.ndarray] = []
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
        if isinstance(strategy, TrendMagicStrategyConfig):
            execution = _execution_trail_chunk(
                decision,
                selected,
                direction[selected],
                atr[selected] * strategy.stop_atr,
                strategy.horizon_bars,
                effective_horizons[start : start + chunk_rows],
            )
            execution_rewards.append(execution[0])
            execution_exit_prices.append(execution[1])
            execution_exit_indices.append(execution[2])
            execution_adverse.append(execution[3])
            execution_reasons.append(execution[4])
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
    entry_timestamp = decision["timestamp"].iloc[entry_index].to_numpy()
    if config.data.timestamp_semantics == "bar_close":
        entry_timestamp = decision["close_timestamp"].iloc[legal].to_numpy()
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
    if isinstance(strategy, TrendMagicStrategyConfig):
        execution_index = np.concatenate(execution_exit_indices)
        output["execution_reward_r"] = np.concatenate(execution_rewards)
        output["execution_mae_r"] = np.concatenate(execution_adverse)
        output["execution_exit_price"] = np.concatenate(execution_exit_prices)
        output["execution_end_ts"] = decision["close_timestamp"].iloc[execution_index].to_numpy()
        output["execution_exit_reason"] = np.concatenate(execution_reasons)
    output["is_holdout"] = output["decision_ts"] >= pd.Timestamp(config.data.holdout_start)
    output.attrs["contamination"] = {
        "streams": [
            frame.attrs.get(
                "contamination", {"stream": f"{symbol}_{timeframe}", "count": 0, "events": []}
            )
            for timeframe, frame in frames.items()
        ],
        "excluded_context_candidates": excluded_context,
        "excluded_label_candidates": excluded_label,
    }
    return output
