"""Pure deterministic bar-level Topstep account replay."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mantis_v2.rl_config import RlConfig

CENT = Decimal("0.01")


class RlAccountError(ValueError):
    """Raised when an account replay fixture violates its contract."""


@dataclass(frozen=True)
class _Contract:
    symbol: str
    underlying: str
    contract_class: Literal["mini", "micro"]
    tick_value: Decimal
    position_units: int
    fee: Decimal


@dataclass(frozen=True)
class _Position:
    contract: _Contract
    quantity: int
    side: Literal["long", "short"]


def _money(value: Decimal) -> str:
    return str(value.quantize(CENT, rounding=ROUND_HALF_UP))


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise RlAccountError(f"{field} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise RlAccountError(f"{field} must be a finite decimal") from exc
    if not result.is_finite():
        raise RlAccountError(f"{field} must be a finite decimal")
    return result


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RlAccountError(f"{field} must be an integer >= {minimum}")
    return value


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _resolve(path: Path, repository_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _load_contracts(
    config: RlConfig, repository_root: Path
) -> tuple[dict[str, Any], dict[str, _Contract]]:
    path = _resolve(config.upstream.rule_contract_path, repository_root)
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise RlAccountError("rule contract identity file does not exist") from exc
    if _sha256_bytes(content) != config.upstream.rule_contract_sha256:
        raise RlAccountError("rule contract identity digest mismatch")
    try:
        rules = tomllib.loads(content.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise RlAccountError("rule contract is not valid TOML") from exc
    raw_contracts = rules.get("contracts")
    if not isinstance(raw_contracts, dict):
        raise RlAccountError("rule contract is missing contracts")
    fees = {
        field: Decimal(str(getattr(config.fees, field)))
        for field in (
            "es",
            "mes",
            "nq",
            "mnq",
            "rty",
            "m2k",
            "ym",
            "mym",
            "gc",
            "mgc",
            "cl",
            "mcl",
            "zb",
        )
    }
    contracts: dict[str, _Contract] = {}
    for symbol, raw in raw_contracts.items():
        if not isinstance(raw, dict) or raw.get("contract_class") not in {"mini", "micro"}:
            raise RlAccountError(f"unsupported contract mapping: {symbol}")
        contracts[symbol] = _Contract(
            symbol=symbol,
            underlying=str(raw.get("underlying")),
            contract_class=raw["contract_class"],
            tick_value=_decimal(raw.get("tick_value"), f"contracts.{symbol}.tick_value"),
            position_units=_integer(
                raw.get("position_units"), f"contracts.{symbol}.position_units", 1
            ),
            fee=fees[symbol.lower()],
        )
    return rules, contracts


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise RlAccountError("bar.timestamp must be an ISO-8601 string")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RlAccountError("bar.timestamp is not valid ISO-8601") from exc
    if result.tzinfo is None:
        raise RlAccountError("bar.timestamp must include a timezone")
    return result.astimezone(UTC)


def _session_day(timestamp: datetime, zone: ZoneInfo, start: time) -> date:
    local = timestamp.astimezone(zone)
    clock = local.time().replace(tzinfo=None)
    if clock >= start:
        return (local.date()).fromordinal(local.date().toordinal() + 1)
    return local.date()


def _position_pnl(position: _Position, mark_ticks: Decimal, config: RlConfig) -> Decimal:
    gross = mark_ticks * position.contract.tick_value * position.quantity
    slippage = (
        Decimal(str(config.execution.adverse_slippage_ticks_per_side))
        * Decimal("2")
        * position.contract.tick_value
        * position.quantity
    )
    fees = position.contract.fee * position.quantity
    return gross - slippage - fees


def _validate_fixture(fixture: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    expected = {"schema_version", "fixture_id", "ticker", "bars"}
    if set(fixture) != expected:
        raise RlAccountError(
            "fixture must contain only schema_version, fixture_id, ticker, and bars"
        )
    if type(fixture["schema_version"]) is not int or fixture["schema_version"] != 1:
        raise RlAccountError("fixture.schema_version must be integer 1")
    if not isinstance(fixture["fixture_id"], str) or not fixture["fixture_id"]:
        raise RlAccountError("fixture.fixture_id must be a non-empty string")
    ticker = fixture["ticker"]
    if not isinstance(ticker, str) or not ticker:
        raise RlAccountError("fixture.ticker must be a non-empty string")
    bars = fixture["bars"]
    if not isinstance(bars, list) or not bars or any(not isinstance(bar, dict) for bar in bars):
        raise RlAccountError("fixture.bars must be a non-empty object array")
    return ticker, bars


def replay_account_fixture(
    config: RlConfig,
    fixture: dict[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Replay deterministic marked-equity bars under the pinned account rules."""
    ticker, bars = _validate_fixture(fixture)
    rules, contracts = _load_contracts(config, repository_root)
    supported_tickers = {contract.underlying for contract in contracts.values()}
    if ticker not in supported_tickers:
        raise RlAccountError(f"unsupported fixture ticker: {ticker}")
    account = rules["account"]
    session = rules["session"]
    try:
        zone = ZoneInfo(session["timezone"])
    except (KeyError, ZoneInfoNotFoundError) as exc:
        raise RlAccountError("rule contract session timezone is invalid") from exc
    start_clock = time.fromisoformat(session["start"])
    flat_clock = time.fromisoformat(session["force_flat"])
    starting_balance = _decimal(account["starting_balance"], "account.starting_balance")
    balance = starting_balance
    mll_floor = _decimal(account["initial_mll_floor"], "account.initial_mll_floor")
    mll_distance = _decimal(account["mll_distance"], "account.mll_distance")
    mll_lock = _decimal(account["mll_lock_balance"], "account.mll_lock_balance")
    profit_target = _decimal(account["profit_target"], "account.profit_target")
    consistency_limit = _decimal(account["consistency_limit"], "account.consistency_limit")
    minimum_days = _integer(account["minimum_trading_days"], "account.minimum_trading_days", 1)
    maximum_units = _integer(
        rules["position_equivalence"]["maximum_micros"],
        "position_equivalence.maximum_micros",
        1,
    )
    dll = Decimal(str(config.topstep.daily_loss_limit_dollars))

    position: _Position | None = None
    status = "ACTIVE"
    reason = "fixture ended"
    current_session: date | None = None
    session_start_balance = balance
    session_had_activity = False
    session_count = 0
    trading_days = 0
    day_profits: list[Decimal] = []
    entry_locked = False
    closed_session: date | None = None
    last_timestamp: datetime | None = None
    path: list[dict[str, Any]] = []

    def finish_session() -> None:
        nonlocal mll_floor, status, reason, session_count, trading_days
        session_count += 1
        profit = balance - session_start_balance
        day_profits.append(profit)
        if session_had_activity:
            trading_days += 1
        mll_floor = max(mll_floor, min(balance - mll_distance, mll_lock))
        total_profit = balance - starting_balance
        best_day = max((value for value in day_profits if value > 0), default=Decimal("0"))
        consistency = best_day / total_profit if total_profit > 0 else Decimal("Infinity")
        if (
            total_profit >= profit_target
            and trading_days >= minimum_days
            and consistency <= consistency_limit
        ):
            status = "PASS"
            reason = "profit target, consistency, and minimum trading days satisfied"
        elif session_count >= config.episode.timeout_trading_days:
            status = "TIMEOUT"
            reason = "maximum trading-day duration reached"

    def state_fields() -> dict[str, Any]:
        profits = list(day_profits)
        displayed_trading_days = trading_days
        displayed_session_days = session_count
        if current_session is not None:
            profits.append(balance - session_start_balance)
            displayed_session_days += 1
            if session_had_activity:
                displayed_trading_days += 1
        total_profit = balance - starting_balance
        best_day = max((value for value in profits if value > 0), default=Decimal("0"))
        consistency = best_day / total_profit if total_profit > 0 else None
        return {
            "trading_days": displayed_trading_days,
            "session_days": displayed_session_days,
            "best_day_profit": _money(best_day),
            "consistency_ratio": None if consistency is None else str(consistency),
        }

    for index, bar in enumerate(bars):
        allowed = {
            "timestamp",
            "action",
            "contract",
            "quantity",
            "side",
            "mark_ticks",
            "pending_orders",
            "realized_pnl",
        }
        unknown = set(bar) - allowed
        if unknown:
            raise RlAccountError(f"unknown bar keys: {', '.join(sorted(unknown))}")
        timestamp = _timestamp(bar.get("timestamp"))
        if last_timestamp is not None and timestamp <= last_timestamp:
            raise RlAccountError("bar timestamps must be strictly increasing")
        last_timestamp = timestamp
        session_day = _session_day(timestamp, zone, start_clock)
        if current_session is None and session_day == closed_session:
            raise RlAccountError("bar occurs after the session was finalized")
        if current_session is not None and session_day != current_session:
            if position is not None:
                raise RlAccountError("fixture carries a position across the session boundary")
            finish_session()
            closed_session = current_session
            current_session = None
            if status != "ACTIVE":
                break
            entry_locked = False
        if current_session is None:
            current_session = session_day
            session_start_balance = balance
            session_had_activity = False

        action = bar.get("action", "none")
        if action not in {"none", "enter", "exit"}:
            raise RlAccountError(f"unsupported bar action: {action}")
        pending_orders = _integer(bar.get("pending_orders", 0), "bar.pending_orders")
        mark_ticks = _decimal(bar.get("mark_ticks", "0"), "bar.mark_ticks")
        realized_pnl = _decimal(bar.get("realized_pnl", "0"), "bar.realized_pnl")
        balance += realized_pnl
        if realized_pnl != Decimal("0"):
            session_had_activity = True

        local_clock = timestamp.astimezone(zone).time().replace(tzinfo=None)
        entry_accepted = False
        if action == "enter":
            if position is not None:
                raise RlAccountError("cannot enter while a position is open")
            symbol = bar.get("contract")
            if not isinstance(symbol, str) or symbol not in contracts:
                raise RlAccountError(f"unsupported contract mapping: {symbol}")
            contract = contracts[symbol]
            if contract.underlying != ticker:
                raise RlAccountError(f"fixture ticker {ticker} cannot trade {symbol}")
            quantity = _integer(bar.get("quantity"), "bar.quantity", 1)
            expected_quantity = (
                config.sizing.mini_quantity
                if contract.contract_class == "mini"
                else config.sizing.micro_quantity
            )
            if quantity != expected_quantity:
                raise RlAccountError(
                    f"{symbol} quantity must match the configured {contract.contract_class} profile"
                )
            if contract.position_units * quantity > maximum_units:
                raise RlAccountError("position exceeds account equivalence limit")
            side = bar.get("side")
            if side not in {"long", "short"}:
                raise RlAccountError("bar.side must be long or short for entry")
            if not entry_locked and (local_clock >= start_clock or local_clock < flat_clock):
                position = _Position(contract, quantity, side)
                entry_accepted = True
                session_had_activity = True
        elif action == "exit":
            if position is None:
                raise RlAccountError("cannot exit while flat")
            balance += _position_pnl(position, mark_ticks, config)
            position = None
            session_had_activity = True
            mark_ticks = Decimal("0")

        marked_equity = balance
        if position is not None:
            marked_equity += _position_pnl(position, mark_ticks, config)

        event = "BAR"
        if marked_equity <= mll_floor:
            balance = marked_equity
            position = None
            pending_orders = 0
            entry_locked = True
            status = "BLOW"
            reason = "maximum loss limit touched"
            event = "MLL_BLOW"
        elif (
            config.topstep.daily_loss_limit_enabled and marked_equity <= session_start_balance - dll
        ):
            if position is not None:
                balance = marked_equity
                position = None
            pending_orders = 0
            entry_locked = True
            marked_equity = balance
            event = "DLL_LOCKOUT"
        elif flat_clock <= local_clock < start_clock:
            if position is not None:
                balance = marked_equity
                position = None
                event = "SESSION_FLATTEN"
            pending_orders = 0
            entry_locked = True
            marked_equity = balance

        path.append(
            {
                "bar_index": index,
                "timestamp": timestamp.isoformat(),
                "session_day": session_day.isoformat(),
                "event": event,
                "entry_accepted": entry_accepted,
                "entry_locked": entry_locked,
                "pending_orders": pending_orders,
                "position": None
                if position is None
                else {
                    "contract": position.contract.symbol,
                    "quantity": position.quantity,
                    "side": position.side,
                    "account_units": position.contract.position_units * position.quantity,
                },
                "balance": _money(balance),
                "marked_equity": _money(marked_equity),
                "mll_floor": _money(mll_floor),
                "status": status,
                **state_fields(),
            }
        )
        if flat_clock <= local_clock < start_clock and status == "ACTIVE":
            finish_session()
            closed_session = current_session
            current_session = None
            path[-1].update(state_fields())
            path[-1]["mll_floor"] = _money(mll_floor)
            path[-1]["status"] = status
        if status == "BLOW":
            break
        if status != "ACTIVE":
            break

    if current_session is not None and status == "ACTIVE" and position is not None:
        raise RlAccountError("fixture ends with an open position before mandatory flatten")

    net_profit = balance - starting_balance
    terminal_state = state_fields()
    return {
        "schema_version": 1,
        "fixture_id": fixture["fixture_id"],
        "ticker": ticker,
        "identities": {
            "config": config.digest,
            "rule": config.rule_digest,
            "fee": config.fee_digest,
        },
        "account_path": path,
        "terminal": {
            "status": status,
            "reason": reason,
            "balance": _money(balance),
            "marked_equity": _money(balance),
            "mll_floor": _money(mll_floor),
            "net_profit": _money(net_profit),
            **terminal_state,
        },
    }


def write_account_replay_manifest(
    config: RlConfig,
    input_path: Path,
    output_path: Path,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Replay a JSON fixture and publish one atomic no-overwrite manifest."""
    root = (repository_root or Path.cwd()).resolve()
    try:
        input_bytes = input_path.read_bytes()
        fixture = json.loads(input_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise RlAccountError("replay fixture is not valid JSON") from exc
    if not isinstance(fixture, dict):
        raise RlAccountError("replay fixture root must be an object")
    result = replay_account_fixture(config, fixture, root)
    result["identities"]["input"] = _sha256_bytes(input_bytes)
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output_path)
        except FileExistsError as exc:
            raise RlAccountError(f"replay output already exists: {output_path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result
