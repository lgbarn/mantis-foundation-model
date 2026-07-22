"""Deterministic Topstep Trading Combine account state machine."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import asdict, dataclass
from datetime import date, time, timedelta
from pathlib import Path
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


def _explicit_execution_economics(
    config: DownstreamConfig,
) -> dict[str, dict[str, float | int | str]] | None:
    path = config.topstep.rule_contract_path
    if path is None:
        return None
    try:
        content = Path(path).read_bytes()
        rules = tomllib.loads(content.decode())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise TopstepContractError("Topstep rule contract is unreadable") from exc
    if hashlib.sha256(content).hexdigest() != config.topstep.rule_contract_sha256:
        raise TopstepContractError("Topstep rule contract digest mismatch")
    contracts = rules.get("contracts")
    account = rules.get("account")
    session = rules.get("session")
    if (
        not isinstance(contracts, dict)
        or not isinstance(account, dict)
        or not isinstance(session, dict)
    ):
        raise TopstepContractError("Topstep rule contract is incomplete")
    expected_account = {
        "starting_balance": config.topstep.starting_balance,
        "initial_mll_floor": config.topstep.starting_balance - config.topstep.mll_distance,
        "mll_distance": config.topstep.mll_distance,
        "profit_target": config.topstep.profit_target,
        "consistency_limit": config.topstep.consistency_limit,
        "minimum_trading_days": config.topstep.minimum_trading_days,
        "maximum_position_equivalence": config.topstep.max_mini_equivalents,
        "mll_lock_balance": config.topstep.mll_lock_balance,
        "mll_ratchet": "end_of_day_high_water",
        "mll_enforcement": "continuous_realized_and_unrealized",
        "overnight_holding": False,
    }
    expected_session = {
        "timezone": config.topstep.session_timezone,
        "start": f"{config.topstep.session_start_hour:02d}:00",
        "force_flat": (
            f"{config.topstep.session_end_hour:02d}:{config.topstep.session_end_minute:02d}"
        ),
    }
    if any(account.get(key) != value for key, value in expected_account.items()) or any(
        session.get(key) != value for key, value in expected_session.items()
    ):
        raise TopstepContractError("Topstep rule contract mismatch")
    maximum = config.topstep.max_mini_equivalents
    result: dict[str, dict[str, float | int | str]] = {}
    rows = zip(
        config.data.symbols,
        config.topstep.execution_contracts,
        config.topstep.contract_quantities,
        config.topstep.round_trip_fees,
        strict=True,
    )
    for symbol, contract_symbol, quantity, fee in rows:
        contract = contracts.get(contract_symbol)
        if not isinstance(contract, dict) or contract.get("underlying") != symbol:
            raise TopstepContractError(f"execution contract does not match {symbol}")
        tick_size = float(contract.get("tick_size", 0))
        tick_value = float(contract.get("tick_value", 0))
        position_units = float(contract.get("position_units", 0))
        if min(tick_size, tick_value, position_units) <= 0:
            raise TopstepContractError(f"execution contract is incomplete: {contract_symbol}")
        mini_equivalents = position_units * quantity / 10.0
        if mini_equivalents > maximum:
            raise TopstepContractError("configured contracts exceed the maximum mini equivalents")
        result[symbol] = {
            "contract": contract_symbol,
            "quantity": quantity,
            "multiplier": tick_value / tick_size * quantity,
            "fee": fee * quantity,
            "slippage": (
                2.0 * config.topstep.adverse_slippage_ticks_per_side * tick_value * quantity
            ),
        }
    return result


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
    economics = _explicit_execution_economics(config)
    if economics is None and config.topstep.contracts > config.topstep.max_mini_equivalents:
        raise TopstepContractError("configured contracts exceed the maximum mini equivalents")
    frame = predictions.copy()
    for column in ("decision_ts", "entry_ts", "label_end_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    reward_column = "execution_reward_r" if economics is not None else "reward_r"
    adverse_column = "execution_mae_r" if economics is not None else "mae_r"
    exit_column = "execution_end_ts" if economics is not None else "label_end_ts"
    execution_columns = {reward_column, adverse_column, exit_column}
    missing_execution = execution_columns - set(frame)
    if missing_execution:
        raise TopstepContractError(
            f"missing execution columns: {', '.join(sorted(missing_execution))}"
        )
    frame[exit_column] = pd.to_datetime(frame[exit_column], utc=True, errors="raise")
    numeric = frame[["probability", "threshold", reward_column, adverse_column, "atr"]].to_numpy()
    if not np.isfinite(numeric).all():
        raise TopstepContractError("predictions contain non-finite simulation values")
    if ((frame["probability"] < 0) | (frame["probability"] > 1)).any():
        raise TopstepContractError("probabilities must be in [0, 1]")
    if economics is None:
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
        execution_end = getattr(row, exit_column)
        exit_day = _session_day(execution_end, config)
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
        execution = economics.get(row.symbol) if economics is not None else None
        multiplier = (
            float(execution["multiplier"])
            if execution is not None
            else multipliers[row.symbol] * config.topstep.contracts
        )
        friction = (
            float(execution["fee"]) + float(execution["slippage"]) if execution is not None else 0.0
        )
        risk_dollars = row.atr * config.strategy.stop_atr * multiplier
        low_equity = balance + getattr(row, adverse_column) * risk_dollars - friction
        pnl = getattr(row, reward_column) * risk_dollars - friction
        if low_equity <= loss_floor:
            balance = low_equity
            status = "failed"
            reason = "maximum loss limit touched intraday"
            accepted.append(
                {
                    **row._asdict(),
                    "pnl_dollars": pnl,
                    "pnl_booked_ts": execution_end,
                    "execution_contract": execution["contract"] if execution else row.symbol,
                    "quantity": execution["quantity"] if execution else config.topstep.contracts,
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
        flat_at = execution_end
        accepted.append(
            {
                **row._asdict(),
                "pnl_dollars": pnl,
                "pnl_booked_ts": execution_end,
                "execution_contract": execution["contract"] if execution else row.symbol,
                "quantity": execution["quantity"] if execution else config.topstep.contracts,
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
