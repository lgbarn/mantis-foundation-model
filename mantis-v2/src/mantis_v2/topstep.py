"""Deterministic Topstep Trading Combine account state machine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from mantis_v2.downstream_config import DownstreamConfig


class TopstepContractError(ValueError):
    """Raised when predictions or account rules are invalid."""


@dataclass(frozen=True)
class TopstepResult:
    status: str
    reason: str
    ending_balance: float
    maximum_loss_floor: float
    net_profit: float
    trading_days: int
    accepted_trades: int
    best_day_profit: float
    consistency_ratio: float


def _session_day(value: pd.Timestamp, config: DownstreamConfig) -> date | None:
    try:
        local = value.tz_convert(ZoneInfo(config.topstep.session_timezone))
    except ZoneInfoNotFoundError as exc:
        raise TopstepContractError(
            f"unknown session timezone: {config.topstep.session_timezone}"
        ) from exc
    clock = local.time().replace(tzinfo=None)
    start = time(config.topstep.session_start_hour)
    end = time(config.topstep.session_end_hour, config.topstep.session_end_minute)
    if end < clock < start:
        return None
    if clock >= start:
        following = local + timedelta(days=1)
        return date(following.year, following.month, following.day)
    return date(local.year, local.month, local.day)


def _consistency_ok(ratio: float, config: DownstreamConfig) -> bool:
    if config.topstep.consistency_comparator == "strict":
        return ratio < config.topstep.consistency_limit
    return ratio <= config.topstep.consistency_limit


def simulate_topstep(
    predictions: pd.DataFrame, config: DownstreamConfig
) -> tuple[TopstepResult, pd.DataFrame]:
    """Replay thresholded out-of-sample candidates under 100K Combine rules."""
    required = {
        "symbol",
        "decision_index",
        "decision_ts",
        "entry_ts",
        "label_end_ts",
        "probability",
        "threshold",
        "reward_r",
        "mae_r",
        "atr",
    }
    missing = required - set(predictions)
    if missing:
        raise TopstepContractError(f"missing prediction columns: {', '.join(sorted(missing))}")
    if config.topstep.contracts > config.topstep.max_mini_equivalents:
        raise TopstepContractError("configured contracts exceed the maximum mini equivalents")
    frame = predictions.copy()
    for column in ("decision_ts", "entry_ts", "label_end_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    numeric = frame[["probability", "threshold", "reward_r", "mae_r", "atr"]].to_numpy()
    if not np.isfinite(numeric).all():
        raise TopstepContractError("predictions contain non-finite simulation values")
    if ((frame["probability"] < 0) | (frame["probability"] > 1)).any():
        raise TopstepContractError("probabilities must be in [0, 1]")
    exact_stop = -(1.0 + config.strategy.round_trip_cost_r)
    stopped = np.isclose(frame["reward_r"], exact_stop, rtol=0.0, atol=1e-9)
    if (stopped & (frame["mae_r"] < frame["reward_r"] - 1e-9)).any():
        raise TopstepContractError("exact-stop MAE cannot be below the stopped outcome")
    frame = frame[frame["probability"] >= frame["threshold"]]
    frame = frame.sort_values(
        ["decision_ts", "probability", "symbol"], ascending=[True, False, True]
    )
    multipliers = dict(zip(config.data.symbols, config.data.contract_multipliers, strict=True))
    unknown = set(frame["symbol"]) - set(multipliers)
    if unknown:
        raise TopstepContractError(f"missing contract multiplier: {', '.join(sorted(unknown))}")

    balance = config.topstep.starting_balance
    loss_floor = balance - config.topstep.mll_distance
    flat_at = pd.Timestamp.min.tz_localize("UTC")
    accepted_index: dict[str, int] = {}
    day_profit: dict[date, float] = {}
    halted_days: set[date] = set()
    accepted: list[dict[str, object]] = []
    status = "active"
    reason = "evaluation window ended"
    current_day: date | None = None
    day_start_balance = balance

    def close_day(day: date) -> bool:
        nonlocal loss_floor, status, reason
        loss_floor = max(
            loss_floor,
            min(balance - config.topstep.mll_distance, config.topstep.mll_lock_balance),
        )
        profit = balance - config.topstep.starting_balance
        positive_days = [value for value in day_profit.values() if value > 0]
        best_day = max(positive_days, default=0.0)
        ratio = best_day / profit if profit > 0 else float("inf")
        if (
            profit >= config.topstep.profit_target
            and len(day_profit) >= config.topstep.minimum_trading_days
            and _consistency_ok(ratio, config)
        ):
            status = "passed"
            reason = f"profit target and consistency satisfied at end of {day.isoformat()}"
            return True
        return False

    for row in frame.itertuples(index=False):
        session_day = _session_day(row.entry_ts, config)
        exit_day = _session_day(row.label_end_ts, config)
        if session_day is None:
            continue
        if exit_day != session_day:
            raise TopstepContractError("trade crosses the configured session boundary")
        if current_day is not None and session_day != current_day:
            day_profit[current_day] = balance - day_start_balance
            if close_day(current_day):
                break
            current_day = None
        if session_day in halted_days or row.entry_ts < flat_at:
            continue
        prior = accepted_index.get(row.symbol)
        if prior is not None and row.decision_index - prior < config.strategy.cooldown_bars:
            continue
        if current_day is None:
            current_day = session_day
            day_start_balance = balance
        risk_dollars = (
            row.atr * config.strategy.stop_atr * multipliers[row.symbol] * config.topstep.contracts
        )
        low_equity = balance + row.mae_r * risk_dollars
        pnl = row.reward_r * risk_dollars
        if low_equity <= loss_floor:
            balance = low_equity
            status = "failed"
            reason = "maximum loss limit touched intraday"
            accepted.append(
                {
                    **row._asdict(),
                    "pnl_dollars": pnl,
                    "pnl_booked_ts": row.label_end_ts,
                    "balance_after": balance,
                    "account_status": status,
                }
            )
            break
        if (
            config.topstep.daily_loss_limit_enabled
            and low_equity <= day_start_balance - config.topstep.daily_loss_limit
        ):
            halted_days.add(session_day)
            pnl = day_start_balance - config.topstep.daily_loss_limit - balance
        balance_before = balance
        balance += pnl
        accepted_index[row.symbol] = row.decision_index
        flat_at = row.label_end_ts
        accepted.append(
            {
                **row._asdict(),
                "pnl_dollars": pnl,
                "pnl_booked_ts": row.label_end_ts,
                "balance_before": balance_before,
                "balance_after": balance,
                "account_status": status,
            }
        )
    if current_day is not None:
        day_profit[current_day] = balance - day_start_balance
        if status == "active":
            close_day(current_day)
    net_profit = balance - config.topstep.starting_balance
    best_day = max([value for value in day_profit.values() if value > 0], default=0.0)
    consistency = best_day / net_profit if net_profit > 0 else float("inf")
    result = TopstepResult(
        status=status,
        reason=reason,
        ending_balance=balance,
        maximum_loss_floor=loss_floor,
        net_profit=net_profit,
        trading_days=len(day_profit),
        accepted_trades=len(accepted),
        best_day_profit=best_day,
        consistency_ratio=consistency,
    )
    trades = pd.DataFrame(accepted)
    trades.attrs["summary"] = asdict(result)
    return result, trades
