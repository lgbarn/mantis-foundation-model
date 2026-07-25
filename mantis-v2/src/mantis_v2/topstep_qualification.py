"""Deterministic MNQ replay and stationary day-block Topstep qualification."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo


class TopstepQualificationError(ValueError):
    """Raised when rules, decisions, or qualification inputs are invalid."""


@dataclass(frozen=True)
class Topstep100KRules:
    """Pinned Topstep 100K and MNQ execution rules, represented in cents/ticks."""

    schema_version: int = 1
    snapshot: str = "topstep-100k-2026-07-20"
    starting_balance_cents: int = 10_000_000
    profit_target_cents: int = 600_000
    mll_distance_cents: int = 300_000
    initial_mll_floor_cents: int = 9_700_000
    mll_lock_balance_cents: int = 10_000_000
    consistency_limit: float = 0.50
    minimum_trading_days: int = 2
    session_timezone: str = "America/Chicago"
    account_session_start: str = "17:00"
    account_session_end: str = "16:00"
    strategy_entry_start: str = "08:30"
    strategy_entry_end: str = "15:00"
    strategy_force_flat: str = "15:00"
    strategy_bar_minutes: int = 3
    maximum_minis: int = 10
    maximum_micros: int = 100
    maximum_mnq: int = 10
    mnq_tick_value_cents: int = 50
    mnq_round_trip_fee_cents: int = 122
    adverse_slippage_ticks_per_side: int = 1
    risk_headroom_fraction: float = 0.05
    risk_cap_cents: int = 15_000
    mll_buffer_cents: int = 25_000
    daily_stop_cents: int = 45_000
    consecutive_full_risk_stop_limit: int = 2
    mll_ratchet: str = "end_of_day_high_water"
    mll_enforcement: str = "continuous_realized_and_unrealized"


@dataclass(frozen=True)
class MNQDecision:
    """One close-formed MNQ decision and its deterministic execution outcome."""

    decision_id: str
    decision_ts: datetime
    entry_ts: datetime
    exit_ts: datetime
    decision_bar: int
    entry_bar: int
    side: Literal["long", "short"]
    entry_ticks: int
    stop_ticks: int
    exit_ticks: int
    worst_ticks: int


@dataclass(frozen=True)
class QualificationDay:
    """One complete chronological session of decisions."""

    session_date: date
    decisions: tuple[MNQDecision, ...] = ()
    operational_fault: bool = False


@dataclass(frozen=True)
class TradeReplay:
    decision_id: str
    session_number: int
    quantity: int
    pnl_cents: int
    balance_after_cents: int
    full_risk_stop: bool


@dataclass(frozen=True)
class DayReplay:
    session_date: date
    session_number: int
    net_pnl_cents: int
    ending_balance_cents: int
    mll_floor_cents: int
    disabled_reason: str | None


@dataclass(frozen=True)
class AccountReplay:
    status: Literal["active", "passed", "failed"]
    reason: str
    ending_balance_cents: int
    maximum_loss_floor_cents: int
    consistency_ratio: float
    time_to_pass: int | None
    maximum_drawdown_cents: int
    trades: tuple[TradeReplay, ...]
    days: tuple[DayReplay, ...]


@dataclass(frozen=True)
class HorizonOutcome:
    pass_count: int
    fail_count: int
    unresolved_count: int
    median_successful_time_to_pass: float | None
    median_drawdown_cents: float
    median_consistency_ratio: float


@dataclass(frozen=True)
class QualificationArtifact:
    schema_version: int
    path_count: int
    seed: int
    average_block_sessions: int
    outcomes: dict[int, HorizonOutcome]
    mll_failures: int
    median_successful_time_to_pass: float | None
    drawdown_cents: dict[str, float]
    consistency: dict[str, float]
    config_hash: str
    source_hash: str
    input_hash: str
    sample_hash: str
    promotion_passed: bool
    ppo_eligible: bool


def _hash(value: object) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(content).hexdigest()


class TopstepQualification:
    """Replay MNQ decisions and qualify them on stationary session blocks."""

    def __init__(
        self,
        rules: Topstep100KRules,
        *,
        paths: int = 5_000,
        seed: int = 42,
        average_block_sessions: int = 5,
    ) -> None:
        if rules != Topstep100KRules():
            raise TopstepQualificationError("rules do not match the pinned Topstep 100K authority")
        if paths != 5_000:
            raise TopstepQualificationError("qualification requires exactly 5,000 paths")
        if average_block_sessions != 5:
            raise TopstepQualificationError("stationary blocks must average five sessions")
        if type(seed) is not int:
            raise TopstepQualificationError("seed must be an integer")
        self.rules = rules
        self.paths = paths
        self.seed = seed
        self.average_block_sessions = average_block_sessions

    def mnq_quantity(
        self,
        *,
        headroom_cents: int,
        stop_ticks: int,
        session: int,
        operational_fault: bool = False,
    ) -> int:
        """Return deterministic risk-sized MNQ quantity for an entry."""
        if headroom_cents < 0 or stop_ticks <= 0 or session <= 0:
            raise TopstepQualificationError("invalid sizing state")
        costs = (
            2 * self.rules.adverse_slippage_ticks_per_side * self.rules.mnq_tick_value_cents
            + self.rules.mnq_round_trip_fee_cents
        )
        per_contract_risk = stop_ticks * self.rules.mnq_tick_value_cents + costs
        budget = min(
            math.floor(headroom_cents * self.rules.risk_headroom_fraction),
            self.rules.risk_cap_cents,
        )
        ramp_cap = 2 if session <= 5 else 5 if session <= 10 or operational_fault else 10
        return min(budget // per_contract_risk, ramp_cap, self.rules.maximum_mnq)

    def replay(self, days: tuple[QualificationDay, ...]) -> AccountReplay:
        """Replay complete chronological days against the pinned account rules."""
        self._validate_days(days, chronological=True)
        return self._replay(days)

    def account_session_date(self, timestamp: datetime) -> date:
        """Map an aware timestamp to its 17:00-16:00 CT Topstep account day."""
        if timestamp.tzinfo is None:
            raise TopstepQualificationError("account timestamp must include a timezone")
        local = timestamp.astimezone(ZoneInfo(self.rules.session_timezone))
        clock = local.time().replace(tzinfo=None)
        start = time.fromisoformat(self.rules.account_session_start)
        end = time.fromisoformat(self.rules.account_session_end)
        if clock >= start:
            return date.fromordinal(local.date().toordinal() + 1)
        if clock <= end:
            return local.date()
        raise TopstepQualificationError("timestamp falls outside the Topstep account session")

    def qualify(
        self, days: tuple[QualificationDay, ...], *, source_hash: str
    ) -> QualificationArtifact:
        """Run the fixed 5,000-path, 30/60/90-session qualification."""
        self._validate_days(days, chronological=True)
        if len(source_hash) != 64 or any(
            character not in "0123456789abcdef" for character in source_hash
        ):
            raise TopstepQualificationError("source_hash must be a lowercase SHA-256 digest")
        if not days:
            raise TopstepQualificationError("qualification requires at least one complete day")

        rng = random.Random(self.seed)
        horizon_replays: dict[int, list[AccountReplay]] = {30: [], 60: [], 90: []}
        failures = 0
        successful_times: list[int] = []
        drawdowns: list[int] = []
        consistencies: list[float] = []
        sampled_indices: list[int] = []
        for _ in range(self.paths):
            indices = self._stationary_indices(len(days), 90, rng)
            sampled_indices.extend(indices)
            replay = self._replay(tuple(days[index] for index in indices))
            for horizon in horizon_replays:
                horizon_replays[horizon].append(
                    replay
                    if horizon == 90
                    else self._replay(tuple(days[index] for index in indices[:horizon]))
                )
            drawdowns.append(replay.maximum_drawdown_cents)
            if math.isfinite(replay.consistency_ratio):
                consistencies.append(replay.consistency_ratio)
            if replay.status == "failed":
                failures += 1
            if replay.time_to_pass is not None:
                successful_times.append(replay.time_to_pass)
        median_time = statistics.median(successful_times) if successful_times else None
        outcomes: dict[int, HorizonOutcome] = {}
        for horizon, replays in horizon_replays.items():
            horizon_successes = [
                replay.time_to_pass for replay in replays if replay.time_to_pass is not None
            ]
            horizon_consistency = [
                replay.consistency_ratio
                for replay in replays
                if math.isfinite(replay.consistency_ratio)
            ]
            outcomes[horizon] = HorizonOutcome(
                pass_count=sum(replay.status == "passed" for replay in replays),
                fail_count=sum(replay.status == "failed" for replay in replays),
                unresolved_count=sum(replay.status == "active" for replay in replays),
                median_successful_time_to_pass=(
                    statistics.median(horizon_successes) if horizon_successes else None
                ),
                median_drawdown_cents=statistics.median(
                    replay.maximum_drawdown_cents for replay in replays
                ),
                median_consistency_ratio=(
                    statistics.median(horizon_consistency) if horizon_consistency else 0.0
                ),
            )
        promotion = (
            outcomes[90].pass_count / self.paths >= 0.60
            and failures / self.paths <= 0.20
            and median_time is not None
            and median_time <= 60
        )
        rules_payload = asdict(self.rules)
        inputs_payload = [
            {
                "session_date": day.session_date.isoformat(),
                "operational_fault": day.operational_fault,
                "decisions": [asdict(decision) for decision in day.decisions],
            }
            for day in days
        ]
        return QualificationArtifact(
            schema_version=1,
            path_count=self.paths,
            seed=self.seed,
            average_block_sessions=self.average_block_sessions,
            outcomes=outcomes,
            mll_failures=failures,
            median_successful_time_to_pass=median_time,
            drawdown_cents=self._distribution(drawdowns),
            consistency=self._distribution(consistencies),
            config_hash=_hash(
                {
                    "rules": rules_payload,
                    "paths": self.paths,
                    "seed": self.seed,
                    "average_block_sessions": self.average_block_sessions,
                }
            ),
            source_hash=source_hash,
            input_hash=_hash(inputs_payload),
            sample_hash=_hash(sampled_indices),
            promotion_passed=promotion,
            ppo_eligible=promotion,
        )

    def _stationary_indices(
        self, source_length: int, path_length: int, rng: random.Random
    ) -> list[int]:
        restart_probability = 1.0 / self.average_block_sessions
        current = rng.randrange(source_length)
        indices = [current]
        while len(indices) < path_length:
            if rng.random() < restart_probability:
                current = rng.randrange(source_length)
            else:
                current = (current + 1) % source_length
            indices.append(current)
        return indices

    @staticmethod
    def _distribution(values: list[int] | list[float]) -> dict[str, float]:
        if not values:
            return {"median": 0.0, "maximum": 0.0}
        return {"median": float(statistics.median(values)), "maximum": float(max(values))}

    def _validate_days(self, days: tuple[QualificationDay, ...], *, chronological: bool) -> None:
        if not isinstance(days, tuple) or any(
            not isinstance(day, QualificationDay) for day in days
        ):
            raise TopstepQualificationError(
                "days must be a tuple of complete QualificationDay values"
            )
        dates = [day.session_date for day in days]
        if chronological and dates != sorted(set(dates)):
            raise TopstepQualificationError("days must be unique and chronological")
        for day in days:
            previous_exit: datetime | None = None
            for decision in day.decisions:
                self._validate_decision(day.session_date, decision)
                if previous_exit is not None and decision.entry_ts < previous_exit:
                    raise TopstepQualificationError("decisions overlap within a session")
                previous_exit = decision.exit_ts

    def _validate_decision(self, session_date: date, decision: MNQDecision) -> None:
        if not decision.decision_id or decision.side not in {"long", "short"}:
            raise TopstepQualificationError("decision identity or side is invalid")
        timestamps = (decision.decision_ts, decision.entry_ts, decision.exit_ts)
        if any(value.tzinfo is None for value in timestamps):
            raise TopstepQualificationError("decision timestamps must include a timezone")
        if not decision.decision_ts < decision.entry_ts <= decision.exit_ts:
            raise TopstepQualificationError("decision timestamps are not ordered")
        if decision.entry_bar != decision.decision_bar + 1:
            raise TopstepQualificationError("entry must fill on the next bar")
        if decision.entry_ts - decision.decision_ts != timedelta(
            minutes=self.rules.strategy_bar_minutes
        ):
            raise TopstepQualificationError("decision is stale or not filled on the next bar")
        values = (
            decision.decision_bar,
            decision.entry_bar,
            decision.entry_ticks,
            decision.stop_ticks,
            decision.exit_ticks,
            decision.worst_ticks,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise TopstepQualificationError("decision bars and prices must be nonnegative integers")
        if decision.side == "long" and decision.stop_ticks >= decision.entry_ticks:
            raise TopstepQualificationError("long stop must be below entry")
        if decision.side == "short" and decision.stop_ticks <= decision.entry_ticks:
            raise TopstepQualificationError("short stop must be above entry")
        if decision.side == "long" and decision.worst_ticks > min(
            decision.entry_ticks, decision.exit_ticks
        ):
            raise TopstepQualificationError("long worst price is inconsistent")
        if decision.side == "short" and decision.worst_ticks < max(
            decision.entry_ticks, decision.exit_ticks
        ):
            raise TopstepQualificationError("short worst price is inconsistent")
        zone = ZoneInfo(self.rules.session_timezone)
        local_entry = decision.entry_ts.astimezone(zone)
        local_exit = decision.exit_ts.astimezone(zone)
        local_decision = decision.decision_ts.astimezone(zone)
        decision_clock = local_decision.time().replace(tzinfo=None)
        entry_clock = local_entry.time().replace(tzinfo=None)
        local_exit_clock = local_exit.time().replace(tzinfo=None)
        entry_start = time.fromisoformat(self.rules.strategy_entry_start)
        entry_end = time.fromisoformat(self.rules.strategy_entry_end)
        force_flat = time.fromisoformat(self.rules.strategy_force_flat)
        if (
            self.account_session_date(decision.decision_ts) != session_date
            or self.account_session_date(decision.entry_ts) != session_date
            or self.account_session_date(decision.exit_ts) != session_date
        ):
            raise TopstepQualificationError("decision crosses the Topstep account session")
        if not entry_start <= decision_clock < entry_end:
            raise TopstepQualificationError("decision is outside the strategy RTH window")
        if not entry_start <= entry_clock < entry_end:
            raise TopstepQualificationError("entry is outside the strategy RTH window")
        if local_exit_clock > force_flat:
            raise TopstepQualificationError("decision violates the strategy force-flat time")

    def _replay(self, days: tuple[QualificationDay, ...]) -> AccountReplay:
        rules = self.rules
        balance = rules.starting_balance_cents
        floor = rules.initial_mll_floor_cents
        peak = balance
        maximum_drawdown = 0
        day_profits: list[int] = []
        trades: list[TradeReplay] = []
        replayed_days: list[DayReplay] = []
        status: Literal["active", "passed", "failed"] = "active"
        reason = "evaluation window ended"
        time_to_pass: int | None = None
        fault_seen = False

        for session_number, day in enumerate(days, start=1):
            if status != "active":
                break
            fault_seen = fault_seen or day.operational_fault
            start_balance = balance
            disabled_reason: str | None = None
            consecutive_stops = 0
            if balance - floor <= rules.mll_buffer_cents:
                disabled_reason = "minimum MLL buffer"
            for decision in day.decisions:
                if disabled_reason is not None:
                    break
                stop_ticks = abs(decision.entry_ticks - decision.stop_ticks)
                quantity = self.mnq_quantity(
                    headroom_cents=balance - floor,
                    stop_ticks=stop_ticks,
                    session=session_number,
                    operational_fault=fault_seen,
                )
                if quantity == 0:
                    disabled_reason = "risk budget cannot fund one MNQ"
                    break
                direction = 1 if decision.side == "long" else -1
                costs = (
                    2 * rules.adverse_slippage_ticks_per_side * rules.mnq_tick_value_cents
                    + rules.mnq_round_trip_fee_cents
                ) * quantity
                worst_pnl = (
                    direction
                    * (decision.worst_ticks - decision.entry_ticks)
                    * rules.mnq_tick_value_cents
                    * quantity
                    - costs
                )
                worst_equity = balance + worst_pnl
                if worst_equity <= floor:
                    balance = worst_equity
                    status = "failed"
                    reason = "maximum loss limit breached intraday"
                    disabled_reason = reason
                elif worst_equity <= floor + rules.mll_buffer_cents:
                    balance = worst_equity
                    disabled_reason = "minimum MLL buffer"
                elif worst_equity <= start_balance - rules.daily_stop_cents:
                    balance = start_balance - rules.daily_stop_cents
                    disabled_reason = "daily loss stop"
                else:
                    pnl = (
                        direction
                        * (decision.exit_ticks - decision.entry_ticks)
                        * rules.mnq_tick_value_cents
                        * quantity
                        - costs
                    )
                    balance += pnl
                pnl = balance - (
                    trades[-1].balance_after_cents if trades else rules.starting_balance_cents
                )
                full_stop = (
                    decision.exit_ticks <= decision.stop_ticks
                    if decision.side == "long"
                    else decision.exit_ticks >= decision.stop_ticks
                )
                consecutive_stops = consecutive_stops + 1 if full_stop else 0
                trades.append(
                    TradeReplay(
                        decision_id=decision.decision_id,
                        session_number=session_number,
                        quantity=quantity,
                        pnl_cents=pnl,
                        balance_after_cents=balance,
                        full_risk_stop=full_stop,
                    )
                )
                peak = max(peak, balance)
                maximum_drawdown = max(maximum_drawdown, peak - min(balance, worst_equity))
                if consecutive_stops >= rules.consecutive_full_risk_stop_limit:
                    disabled_reason = "two consecutive full-risk stops"
                if status == "failed":
                    break

            day_profit = balance - start_balance
            day_profits.append(day_profit)
            if status != "failed":
                floor = max(
                    floor, min(balance - rules.mll_distance_cents, rules.mll_lock_balance_cents)
                )
            replayed_days.append(
                DayReplay(
                    session_date=day.session_date,
                    session_number=session_number,
                    net_pnl_cents=day_profit,
                    ending_balance_cents=balance,
                    mll_floor_cents=floor,
                    disabled_reason=disabled_reason,
                )
            )
            profit = balance - rules.starting_balance_cents
            best_day = max((value for value in day_profits if value > 0), default=0)
            consistency = best_day / profit if profit > 0 else float("inf")
            trading_days = sum(value != 0 for value in day_profits)
            if (
                status == "active"
                and profit >= rules.profit_target_cents
                and trading_days >= rules.minimum_trading_days
                and consistency <= rules.consistency_limit
            ):
                status = "passed"
                reason = "profit target, consistency, and minimum trading days satisfied"
                time_to_pass = session_number

        profit = balance - rules.starting_balance_cents
        best_day = max((value for value in day_profits if value > 0), default=0)
        consistency = best_day / profit if profit > 0 else float("inf")
        return AccountReplay(
            status=status,
            reason=reason,
            ending_balance_cents=balance,
            maximum_loss_floor_cents=floor,
            consistency_ratio=consistency,
            time_to_pass=time_to_pass,
            maximum_drawdown_cents=maximum_drawdown,
            trades=tuple(trades),
            days=tuple(replayed_days),
        )
