"""Dependency-light causal entry environment."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from mantis_v2.rl_config import RlConfig


class EnvironmentContractError(ValueError):
    """Raised when an environment input or action violates its contract."""


@dataclass(frozen=True)
class CandidateData:
    embedding: np.ndarray
    direction: int
    trend_line: float
    atr: float
    bars_since_direction_change: int
    label: int

    def __post_init__(self) -> None:
        embedding = np.asarray(self.embedding, dtype=np.float32)
        if embedding.ndim != 1 or not np.isfinite(embedding).all():
            raise EnvironmentContractError("candidate embedding must be a finite vector")
        if self.direction not in {-1, 1}:
            raise EnvironmentContractError("candidate direction must be -1 or 1")
        if self.label not in {0, 1}:
            raise EnvironmentContractError("candidate label must be 0 or 1")
        if not math.isfinite(self.trend_line) or not math.isfinite(self.atr) or self.atr <= 0:
            raise EnvironmentContractError(
                "candidate market state must be finite with positive ATR"
            )
        if self.bars_since_direction_change < 0:
            raise EnvironmentContractError("bars since direction change cannot be negative")
        embedding.setflags(write=False)
        object.__setattr__(self, "embedding", embedding)


@dataclass(frozen=True)
class BarData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    candidate: CandidateData | None = None
    discontinuity: bool = False


@dataclass(frozen=True)
class EnvironmentEpisode:
    ticker: str
    profile: str
    bars: tuple[BarData, ...]

    def __post_init__(self) -> None:
        if self.ticker not in {"ES", "NQ", "RTY", "YM", "GC", "CL", "ZB"}:
            raise EnvironmentContractError(f"unsupported episode ticker: {self.ticker}")
        if self.profile not in {"one_mini", "ten_micros"}:
            raise EnvironmentContractError(f"unsupported episode profile: {self.profile}")
        if self.ticker == "ZB" and self.profile != "one_mini":
            raise EnvironmentContractError("ZB is mini-only")
        if len(self.bars) < 2:
            raise EnvironmentContractError("episode must contain at least two bars")
        if any(
            current.timestamp <= previous.timestamp
            for previous, current in zip(self.bars, self.bars[1:], strict=False)
        ):
            raise EnvironmentContractError("episode bars must be strictly chronological")
        widths = {len(bar.candidate.embedding) for bar in self.bars if bar.candidate is not None}
        if len(widths) != 1:
            raise EnvironmentContractError("episode candidates need one embedding width")


@dataclass(frozen=True)
class EntryObservation:
    vector: np.ndarray
    positioned: float
    schema_version: int = 1
    action_mask: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise EnvironmentContractError("observation must be a finite vector")
        vector.setflags(write=False)
        object.__setattr__(self, "vector", vector)
        mask = np.asarray(self.action_mask, dtype=np.bool_)
        if mask.shape != (2,):
            raise EnvironmentContractError("observation action mask must have two actions")
        mask.setflags(write=False)
        object.__setattr__(self, "action_mask", mask)


@dataclass(frozen=True)
class EntryObservationSchemaV1:
    embedding_width: int

    _FIELDS = (
        "contract_is_mini",
        "contract_is_micro",
        "quantity",
        "dollar_stop_risk",
        "tick_size",
        "aggregate_tick_value",
        "booked_round_trip_fee",
        "best_day_profit",
        "consistency_ratio",
        "action_skip",
        "action_enter",
    )

    def index(self, name: str) -> int:
        return self.embedding_width + self._FIELDS.index(name)


@dataclass
class _Position:
    entry_price: float
    direction: int
    risk_price: float
    stop_price: float
    favorable_extreme: float
    bars_held: int = 0


_TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "CL", "ZB")
_MINI_MULTIPLIERS = {
    "ES": 50.0,
    "NQ": 20.0,
    "RTY": 50.0,
    "YM": 5.0,
    "GC": 100.0,
    "CL": 1000.0,
    "ZB": 1000.0,
}
_MICRO_SYMBOLS = {"ES": "mes", "NQ": "mnq", "RTY": "m2k", "YM": "mym", "GC": "mgc", "CL": "mcl"}


class TopstepEntryEnvironment:
    def __init__(self, config: RlConfig, episode: EnvironmentEpisode) -> None:
        self.config = config
        self.episode = episode
        self._rules = self._load_rules()
        self._zone = ZoneInfo(str(self._rules["session"]["timezone"]))
        self._session_start = time.fromisoformat(str(self._rules["session"]["start"]))
        self._session_flat = time.fromisoformat(str(self._rules["session"]["force_flat"]))
        self._embedding_width = len(
            next(bar.candidate for bar in episode.bars if bar.candidate is not None).embedding
        )
        self.observation_schema = EntryObservationSchemaV1(self._embedding_width)
        self._price_multiplier = _MINI_MULTIPLIERS[episode.ticker]
        if episode.profile == "one_mini":
            self._fee = float(getattr(config.fees, episode.ticker.lower()))
        else:
            self._fee = float(getattr(config.fees, _MICRO_SYMBOLS[episode.ticker])) * 10.0
        contracts = self._rules["contracts"]
        symbol = (
            episode.ticker
            if episode.profile == "one_mini"
            else _MICRO_SYMBOLS[episode.ticker].upper()
        )
        self._quantity = 1 if episode.profile == "one_mini" else 10
        self._tick_size = float(contracts[symbol]["tick_size"])
        self._reset_state()

    def _load_rules(self) -> dict[str, Any]:
        root = Path(__file__).resolve().parents[3]
        path = self.config.upstream.rule_contract_path
        if not path.is_absolute():
            path = root / path
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise EnvironmentContractError("rule contract does not exist") from exc
        if hashlib.sha256(content).hexdigest() != self.config.upstream.rule_contract_sha256:
            raise EnvironmentContractError("rule contract digest mismatch")
        value = tomllib.loads(content.decode())
        if not isinstance(value.get("account"), dict) or not isinstance(value.get("session"), dict):
            raise EnvironmentContractError("rule contract is incomplete")
        return value

    def _reset_state(self) -> None:
        account = self._rules["account"]
        assert isinstance(account, dict)
        self._index = 0
        self._balance = float(account["starting_balance"])
        self._mll_floor = float(account["initial_mll_floor"])
        self._session_start_balance = self._balance
        self._session = self._session_id(self.episode.bars[0].timestamp)
        self._entry_locked = False
        self._pending: CandidateData | None = None
        self._position: _Position | None = None
        self._accepted_trades = 0
        self._trading_days = 0
        self._session_activity = False
        self._best_day = 0.0
        self._terminated = False
        self._truncated = False
        self._status = "ACTIVE"

    def _session_id(self, timestamp: datetime) -> str:
        local = timestamp.astimezone(self._zone)
        session_date = local.date()
        if local.time().replace(tzinfo=None) >= self._session_start:
            session_date += timedelta(days=1)
        return session_date.isoformat()

    def _finish_session(self) -> None:
        account = self._rules["account"]
        assert isinstance(account, dict)
        profit = self._balance - self._session_start_balance
        if self._session_activity:
            self._trading_days += 1
        self._best_day = max(self._best_day, profit)
        distance = float(account["mll_distance"])
        lock = float(account["mll_lock_balance"])
        self._mll_floor = max(self._mll_floor, min(self._balance - distance, lock))
        total_profit = self._balance - float(account["starting_balance"])
        consistency = self._best_day / total_profit if total_profit > 0 else math.inf
        if (
            total_profit >= float(account["profit_target"])
            and self._trading_days >= int(account["minimum_trading_days"])
            and consistency <= float(account["consistency_limit"])
        ):
            self._status = "PASS"
            self._terminated = True

    def _net_position_pnl(self, mark: float) -> float:
        assert self._position is not None
        gross = (
            self._position.direction * (mark - self._position.entry_price) * self._price_multiplier
        )
        slippage = 2.0 * self.config.execution.adverse_slippage_ticks_per_side * self._tick_value()
        return gross - slippage - self._fee

    def _tick_value(self) -> float:
        contracts = self._rules.get("contracts")
        assert isinstance(contracts, dict)
        symbol = self.episode.ticker
        if self.episode.profile == "ten_micros":
            micro_symbols = {
                "ES": "MES",
                "NQ": "MNQ",
                "RTY": "M2K",
                "YM": "MYM",
                "GC": "MGC",
                "CL": "MCL",
            }
            symbol = micro_symbols[symbol]
            return float(contracts[symbol]["tick_value"]) * 10.0
        return float(contracts[symbol]["tick_value"])

    def _close_position(self, price: float) -> None:
        self._balance += self._net_position_pnl(price)
        self._position = None

    def _process_position(self, bar: BarData) -> str | None:
        if self._position is None:
            return None
        position = self._position
        position.bars_held += 1
        stopped = (
            bar.low <= position.stop_price
            if position.direction > 0
            else bar.high >= position.stop_price
        )
        if stopped:
            gap = (
                bar.open < position.stop_price
                if position.direction > 0
                else bar.open > position.stop_price
            )
            self._close_position(bar.open if gap else position.stop_price)
            return "STOP"
        favorable = bar.high if position.direction > 0 else bar.low
        if position.direction > 0:
            position.favorable_extreme = max(position.favorable_extreme, favorable)
            progress = position.favorable_extreme - position.entry_price
            if progress >= self.config.exit.activation_r * position.risk_price:
                position.stop_price = max(
                    position.stop_price,
                    position.favorable_extreme - self.config.exit.giveback_r * position.risk_price,
                )
        else:
            position.favorable_extreme = min(position.favorable_extreme, favorable)
            progress = position.entry_price - position.favorable_extreme
            if progress >= self.config.exit.activation_r * position.risk_price:
                position.stop_price = min(
                    position.stop_price,
                    position.favorable_extreme + self.config.exit.giveback_r * position.risk_price,
                )
        if position.bars_held >= self.config.exit.horizon_bars:
            self._close_position(bar.close)
            return "HORIZON"
        local_clock = bar.timestamp.astimezone(self._zone).time().replace(tzinfo=None)
        if self._session_flat <= local_clock < self._session_start:
            self._close_position(bar.close)
            self._entry_locked = True
            return "SESSION_FLATTEN"
        return None

    def _observation(self) -> EntryObservation:
        bar = self.episode.bars[self._index]
        candidate = bar.candidate
        embedding = (
            candidate.embedding
            if candidate is not None
            else np.zeros(self._embedding_width, dtype=np.float32)
        )
        previous_close = self.episode.bars[max(0, self._index - 1)].close
        direction = float(candidate.direction) if candidate is not None else 0.0
        separation = (
            (bar.close - candidate.trend_line) / candidate.atr if candidate is not None else 0.0
        )
        bars_since = float(candidate.bars_since_direction_change) if candidate else 0.0
        atr_fraction = candidate.atr / bar.close if candidate is not None else 0.0
        local = bar.timestamp.astimezone(self._zone)
        minute = local.hour * 60 + local.minute
        market = np.array(
            [
                direction,
                separation,
                bars_since / 120.0,
                bar.close / previous_close - 1.0,
                atr_fraction,
                math.log1p(bar.volume),
                math.sin(2.0 * math.pi * minute / 1440.0),
                math.cos(2.0 * math.pi * minute / 1440.0),
                float(candidate is not None),
                float(bar.discontinuity),
            ],
            dtype=np.float32,
        )
        identity = np.zeros(len(_TICKERS) + 2, dtype=np.float32)
        identity[_TICKERS.index(self.episode.ticker)] = 1.0
        identity[len(_TICKERS) + (self.episode.profile == "ten_micros")] = 1.0
        account = self._rules["account"]
        assert isinstance(account, dict)
        starting = float(account["starting_balance"])
        target = float(account["profit_target"])
        marked = self._balance
        if self._position is not None:
            marked += self._net_position_pnl(bar.close)
        account_values = np.array(
            [
                float(self._position is not None),
                self._balance / starting,
                marked / starting,
                (marked - self._mll_floor) / starting,
                (starting + target - self._balance) / starting,
                (self._balance - self._session_start_balance) / starting,
                max(self._best_day, 0.0) / starting,
                self._trading_days / self.config.episode.timeout_trading_days,
                self._accepted_trades / 100.0,
                (len(self.episode.bars) - self._index) / len(self.episode.bars),
                float(self._entry_locked),
            ],
            dtype=np.float32,
        )
        mask = self.action_mask().astype(np.float32)
        total_profit = self._balance - starting
        consistency = self._best_day / total_profit if total_profit > 0 else 0.0
        economics = np.array(
            [
                float(self.episode.profile == "one_mini"),
                float(self.episode.profile == "ten_micros"),
                float(self._quantity),
                (candidate.atr * 0.5 * self._price_multiplier) if candidate else 0.0,
                self._tick_size,
                self._tick_value(),
                self._fee,
                self._best_day,
                consistency,
            ],
            dtype=np.float32,
        )
        vector = np.concatenate((embedding, economics, mask, market, identity, account_values))
        return EntryObservation(
            vector, float(self._position is not None), action_mask=mask.astype(np.bool_)
        )

    def reset(self, *, seed: int | None = None) -> tuple[EntryObservation, dict[str, object]]:
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise EnvironmentContractError("seed must be an integer")
        self._reset_state()
        bar = self.episode.bars[0]
        return self._observation(), {
            "seed": seed,
            "bar_index": 0,
            "timestamp": bar.timestamp.isoformat(),
            "status": self._status,
        }

    def step(self, action: int) -> tuple[EntryObservation, float, bool, bool, dict[str, object]]:
        if self._terminated or self._truncated:
            raise EnvironmentContractError("cannot step a completed environment")
        if type(action) is not int or action not in {0, 1} or not self.action_mask()[action]:
            raise EnvironmentContractError(f"invalid action {action} for current mask")
        current = self.episode.bars[self._index]
        if action == 1:
            assert current.candidate is not None
            self._pending = current.candidate
        self._index += 1
        if self._index >= len(self.episode.bars):
            self._index = len(self.episode.bars) - 1
            self._truncated = True
            return self._observation(), 0.0, False, True, {"status": "TIMEOUT"}
        bar = self.episode.bars[self._index]
        info: dict[str, object] = {
            "bar_index": self._index,
            "timestamp": bar.timestamp.isoformat(),
            "status": self._status,
        }
        new_session = self._session_id(bar.timestamp)
        if new_session != self._session:
            if self._position is not None:
                self._close_position(bar.open)
            self._finish_session()
            self._session = new_session
            self._session_start_balance = self._balance
            self._session_activity = False
            self._entry_locked = False
        if bar.discontinuity:
            if self._position is not None:
                self._close_position(bar.open)
            self._pending = None
            info["event"] = "DISCONTINUITY_RESET"
        elif self._pending is not None:
            candidate = self._pending
            risk = candidate.atr * 0.5
            self._position = _Position(
                entry_price=bar.open,
                direction=candidate.direction,
                risk_price=risk,
                stop_price=bar.open - candidate.direction * risk,
                favorable_extreme=bar.open,
            )
            self._pending = None
            self._accepted_trades += 1
            self._session_activity = True
            info["fill_timestamp"] = bar.timestamp.isoformat()
            info["fill_price"] = bar.open
            info["booked_round_trip_fee"] = self._fee
        event = self._process_position(bar)
        if event is not None:
            info["event"] = event
        marked = self._balance
        if self._position is not None:
            marked += self._net_position_pnl(bar.close)
        if marked <= self._mll_floor:
            if self._position is not None:
                self._balance = marked
                self._position = None
            self._pending = None
            self._entry_locked = True
            self._status = "BLOW"
            self._terminated = True
            info["event"] = "MLL_BLOW"
        elif (
            self.config.topstep.daily_loss_limit_enabled
            and marked <= self._session_start_balance - self.config.topstep.daily_loss_limit_dollars
        ):
            if self._position is not None:
                self._balance = marked
                self._position = None
            self._pending = None
            self._entry_locked = True
            info["event"] = "DLL_LOCKOUT"
        if self._index == len(self.episode.bars) - 1 and not self._terminated:
            if self._position is not None:
                self._close_position(bar.close)
            self._finish_session()
            self._truncated = not self._terminated
            if self._truncated:
                self._status = "TIMEOUT"
        info["status"] = self._status
        reward = 1.0 if self._status == "PASS" else 0.0
        return self._observation(), reward, self._terminated, self._truncated, info

    def action_mask(self) -> np.ndarray:
        if self._terminated or self._truncated:
            return np.array([True, False], dtype=np.bool_)
        bar = self.episode.bars[self._index]
        local_clock = bar.timestamp.astimezone(self._zone).time().replace(tzinfo=None)
        session_open = local_clock >= self._session_start or local_clock < self._session_flat
        legal_entry = (
            self._position is None
            and self._pending is None
            and not self._entry_locked
            and not bar.discontinuity
            and bar.candidate is not None
            and session_open
        )
        return np.array([True, legal_entry], dtype=np.bool_)

    @property
    def account_state(self) -> dict[str, object]:
        """Return the causal public account state used by replay and validation."""
        bar = self.episode.bars[self._index]
        equity = self._balance
        if self._position is not None:
            equity += self._net_position_pnl(bar.close)
        starting = float(self._rules["account"]["starting_balance"])
        total_profit = self._balance - starting
        consistency = self._best_day / total_profit if total_profit > 0 else 0.0
        return {
            "balance": self._balance,
            "equity": equity,
            "mll_floor": self._mll_floor,
            "best_day_profit": self._best_day,
            "consistency_ratio": consistency,
            "accepted_trades": self._accepted_trades,
            "trading_days": self._trading_days,
            "entry_locked": self._entry_locked,
            "status": self._status,
        }

    @property
    def current_label(self) -> int | None:
        """Expose the supervised target separately from the causal observation."""
        candidate = self.episode.bars[self._index].candidate
        return None if candidate is None else candidate.label
