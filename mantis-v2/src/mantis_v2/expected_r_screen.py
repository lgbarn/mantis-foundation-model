"""Causal expected-net-R raw-window ridge screening."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mantis_v2.rl_provenance import source_digest
from mantis_v2.strategy import trend_magic_state, wilder_atr


class ExpectedRScreenError(ValueError):
    """Raised when the expected-R contract cannot be satisfied."""


@dataclass(frozen=True)
class ExpectedRScreenConfig:
    """Frozen expected-R label, feature, split, and selection contract."""

    window_bars: int = 512
    ridge_alpha: float = 10.0
    stop_r: float = 1.0
    target_r: float = 3.0
    trail_activation_r: float = 2.0
    trail_giveback_r: float = 0.75
    horizon_bars: int = 120
    point_value: float = 2.0
    tick_size: float = 0.25
    round_trip_commission: float = 1.22
    slippage_ticks: float = 2.0
    session_timezone: str = "America/Chicago"
    session_start: str = "17:00"
    entry_start: str = "08:30"
    session_exit: str = "15:00"
    train_start: str = "2023-07-01"
    train_end: str = "2025-07-01"
    validation_start: str = "2025-07-01"
    validation_end: str = "2025-10-01"
    test_start: str = "2025-10-01"
    test_end: str = "2026-01-01"
    entries_per_active_session: float = 2.0
    bootstrap_replicates: int = 1000
    bootstrap_restart_probability: float = 0.2
    seed: int = 84
    timestamp_semantics: Literal["bar_open", "bar_close"] = "bar_open"
    timeframe_minutes: int = 3
    atr_period: int = 20
    stop_atr: float = 0.5
    cci_period: int = 20
    trend_magic_atr_period: int = 5
    trend_magic_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.window_bars < 2 or self.horizon_bars < 1:
            raise ExpectedRScreenError("window_bars must be >= 2 and horizon_bars must be >= 1")
        positive = (
            self.ridge_alpha,
            self.stop_r,
            self.target_r,
            self.trail_activation_r,
            self.trail_giveback_r,
            self.point_value,
            self.tick_size,
            self.entries_per_active_session,
            self.stop_atr,
            self.trend_magic_multiplier,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ExpectedRScreenError(
                "screen scale and execution values must be finite and positive"
            )
        if self.target_r <= self.trail_activation_r:
            raise ExpectedRScreenError("target_r must exceed trail_activation_r")
        if not 0 < self.bootstrap_restart_probability <= 1:
            raise ExpectedRScreenError("bootstrap_restart_probability must be in (0, 1]")
        if self.bootstrap_replicates < 1:
            raise ExpectedRScreenError("bootstrap_replicates must be >= 1")
        if self.timeframe_minutes != 3:
            raise ExpectedRScreenError("timeframe_minutes must be 3")
        periods = (
            self.atr_period,
            self.cci_period,
            self.trend_magic_atr_period,
        )
        if not all(period >= 1 for period in periods):
            raise ExpectedRScreenError("indicator periods must be >= 1")
        costs = (self.round_trip_commission, self.slippage_ticks)
        if not all(np.isfinite(value) and value >= 0 for value in costs):
            raise ExpectedRScreenError("execution costs must be finite and nonnegative")
        split_dates = tuple(
            pd.Timestamp(value)
            for value in (
                self.train_start,
                self.train_end,
                self.validation_start,
                self.validation_end,
                self.test_start,
                self.test_end,
            )
        )
        train_start, train_end, validation_start, validation_end, test_start, test_end = split_dates
        if not (
            train_start < train_end
            and train_end <= validation_start < validation_end
            and validation_end <= test_start < test_end
        ):
            raise ExpectedRScreenError("chronological split boundaries must be strictly ordered")
        entry_start = _clock(self.entry_start)
        session_exit = _clock(self.session_exit)
        _clock(self.session_start)
        if entry_start >= session_exit:
            raise ExpectedRScreenError("entry_start must precede session_exit")


_MARKET_COLUMNS = ("open", "high", "low", "close", "volume")
_CONTEXT_COLUMNS = ("trend_magic_direction", "trend_magic_distance_r", "session_minute")
RidgePredictor = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]
ThresholdSelector = Callable[[pd.DataFrame, np.ndarray, int], float]
IntervalEvaluator = Callable[
    [pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, float],
    dict[str, list[float] | None],
]


def _clock(value: str) -> time:
    try:
        parsed = time.fromisoformat(value)
    except ValueError as error:
        raise ExpectedRScreenError(f"invalid session time: {value}") from error
    if parsed.second or parsed.microsecond or parsed.tzinfo is not None:
        raise ExpectedRScreenError(f"session time must be HH:MM: {value}")
    return parsed


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _average_uniqueness(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    if not len(starts):
        return np.empty(0, dtype=np.float64)
    width = int(ends.max()) + 2
    changes = np.zeros(width, dtype=np.int64)
    np.add.at(changes, starts, 1)
    np.add.at(changes, ends + 1, -1)
    concurrency = np.cumsum(changes)
    inverse = np.divide(1.0, concurrency, out=np.zeros(width), where=concurrency > 0)
    prefix = np.concatenate(([0.0], np.cumsum(inverse)))
    return np.asarray((prefix[ends + 1] - prefix[starts]) / (ends - starts + 1))


class ExpectedRScreen:
    """Generate causal candidates and fit the fixed raw-window ridge control."""

    def __init__(self, config: ExpectedRScreenConfig | None = None) -> None:
        self.config = config or ExpectedRScreenConfig()

    def generate_candidates(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return every warmed Trend Magic state with next-bar execution and net-R label."""
        required = {"timestamp", *_MARKET_COLUMNS}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ExpectedRScreenError(f"market frame missing columns: {', '.join(missing)}")
        data = frame.reset_index(drop=True).copy()
        timestamps = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
        bar_duration = pd.to_timedelta(self.config.timeframe_minutes, unit="min")
        open_timestamps = (
            timestamps
            if self.config.timestamp_semantics == "bar_open"
            else timestamps - bar_duration
        )
        close_timestamps = (
            timestamps + bar_duration
            if self.config.timestamp_semantics == "bar_open"
            else timestamps
        )
        if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise ExpectedRScreenError("timestamps must be sorted and unique")
        deltas = timestamps.diff().dropna().dt.total_seconds().to_numpy()
        expected_seconds = self.config.timeframe_minutes * 60
        if len(deltas) and (
            float(deltas.min()) != expected_seconds
            or not np.all(np.remainder(deltas, expected_seconds) == 0)
        ):
            raise ExpectedRScreenError("market frame must use 3-minute bars")
        sealed_start = pd.Timestamp(self.config.test_end, tz="UTC")
        if bool((close_timestamps >= sealed_start).any()):
            raise ExpectedRScreenError("market frame reaches the sealed holdout")
        market = data.loc[:, _MARKET_COLUMNS]
        if not np.isfinite(market.to_numpy(dtype=np.float64)).all():
            raise ExpectedRScreenError("market values must be finite")
        context_columns = {"trend_magic_direction", "trend_magic_line", "risk_points"}
        supplied_context = context_columns.intersection(data.columns)
        if supplied_context and supplied_context != context_columns:
            missing_context = sorted(context_columns.difference(data.columns))
            raise ExpectedRScreenError(
                f"partial fixture context is missing columns: {', '.join(missing_context)}"
            )
        if not supplied_context:
            line, derived_direction = trend_magic_state(
                data,
                self.config.cci_period,
                self.config.trend_magic_atr_period,
                self.config.trend_magic_multiplier,
            )
            data["trend_magic_line"] = line
            data["trend_magic_direction"] = derived_direction
            data["risk_points"] = wilder_atr(data, self.config.atr_period) * self.config.stop_atr
        direction = data["trend_magic_direction"].to_numpy(dtype=np.int8)
        if not np.isin(direction, [-1, 0, 1]).all():
            raise ExpectedRScreenError("trend_magic_direction must contain only -1, 0, or 1")

        rows: list[dict[str, Any]] = []
        features: list[np.ndarray] = []
        first = self.config.window_bars - 1
        for decision_index in range(first, len(data) - 1):
            side = int(direction[decision_index])
            risk = float(data.at[decision_index, "risk_points"])
            entry_local = open_timestamps.iloc[decision_index + 1].tz_convert(
                self.config.session_timezone
            )
            entry_close_local = close_timestamps.iloc[decision_index + 1].tz_convert(
                self.config.session_timezone
            )
            if (
                side == 0
                or not np.isfinite(risk)
                or risk <= 0
                or not np.isfinite(float(data.at[decision_index, "trend_magic_line"]))
                or not self._in_session(entry_local, entry_close_local)
            ):
                continue
            outcome = self._resolve(
                data,
                open_timestamps,
                close_timestamps,
                decision_index,
                side,
                risk,
            )
            if outcome["exit_reason"] == "truncated":
                continue
            window = data.loc[decision_index - self.config.window_bars + 1 : decision_index]
            raw = window.loc[:, _MARKET_COLUMNS].to_numpy(dtype=np.float32).reshape(-1)
            local = close_timestamps.iloc[decision_index].tz_convert(self.config.session_timezone)
            context = np.array(
                [
                    side,
                    (
                        float(data.at[decision_index, "close"])
                        - float(data.at[decision_index, "trend_magic_line"])
                    )
                    / risk,
                    (local.hour * 60 + local.minute) / 1440.0,
                ],
                dtype=np.float32,
            )
            features.append(np.concatenate((raw, context)))
            rows.append(
                {
                    "row_id": hashlib.sha256(
                        f"{timestamps.iloc[decision_index].isoformat()}:{decision_index}".encode()
                    ).hexdigest(),
                    "decision_index": decision_index,
                    "feature_start_index": decision_index - self.config.window_bars + 1,
                    "entry_index": decision_index + 1,
                    "outcome_index": outcome["outcome_index"],
                    "decision_ts": close_timestamps.iloc[decision_index],
                    "feature_start_ts": close_timestamps.iloc[
                        decision_index - self.config.window_bars + 1
                    ],
                    "entry_ts": open_timestamps.iloc[decision_index + 1],
                    "outcome_ts": close_timestamps.iloc[outcome["outcome_index"]],
                    "direction": side,
                    "entry_price": float(data.at[decision_index + 1, "open"]),
                    "risk_points": risk,
                    **{key: value for key, value in outcome.items() if key != "outcome_index"},
                }
            )
        if not rows:
            raise ExpectedRScreenError("no eligible candidates")
        candidates = pd.DataFrame(rows)
        candidates["average_uniqueness"] = _average_uniqueness(
            candidates["entry_index"].to_numpy(dtype=np.int64),
            candidates["outcome_index"].to_numpy(dtype=np.int64),
        )
        candidates.attrs["raw_features"] = np.asarray(features, dtype=np.float32)
        candidates.attrs["data_sha256"] = _sha256_json(
            {
                "columns": [*sorted(required), *sorted(context_columns)],
                "rows": pd.util.hash_pandas_object(
                    data.loc[:, [*sorted(required), *sorted(context_columns)]], index=False
                )
                .to_numpy(dtype=np.uint64)
                .tolist(),
            }
        )
        return candidates

    def _resolve(
        self,
        frame: pd.DataFrame,
        open_timestamps: pd.Series,
        close_timestamps: pd.Series,
        decision_index: int,
        side: int,
        risk: float,
    ) -> dict[str, Any]:
        entry_index = decision_index + 1
        entry_local = open_timestamps.iloc[entry_index].tz_convert(
            ZoneInfo(self.config.session_timezone)
        )
        entry = float(frame.at[entry_index, "open"])
        stop = entry - side * self.config.stop_r * risk
        target = entry + side * self.config.target_r * risk
        peak_r = 0.0
        trail_r: float | None = None
        end = min(len(frame) - 1, entry_index + self.config.horizon_bars - 1)
        exit_price = float(frame.at[end, "close"])
        reason = "horizon" if end == entry_index + self.config.horizon_bars - 1 else "truncated"
        outcome_index = end
        cutoff_local = self._session_cutoff(entry_local)
        for index in range(entry_index, end + 1):
            local_close = close_timestamps.iloc[index].tz_convert(
                ZoneInfo(self.config.session_timezone)
            )
            if local_close > cutoff_local:
                previous = index - 1
                exit_price = float(frame.at[previous, "close"])
                reason, outcome_index = "session", previous
                break
            high = float(frame.at[index, "high"])
            low = float(frame.at[index, "low"])
            bar_open = float(frame.at[index, "open"])
            trail_price = entry + side * trail_r * risk if trail_r is not None else None
            active_stop = trail_price if trail_price is not None else stop
            stop_hit = low <= active_stop if side > 0 else high >= active_stop
            target_hit = high >= target if side > 0 else low <= target
            target_gap = bar_open >= target if side > 0 else bar_open <= target
            if target_gap:
                exit_price, reason, outcome_index = target, "target", index
                break
            if stop_hit:
                gap = bar_open < active_stop if side > 0 else bar_open > active_stop
                exit_price, reason, outcome_index = (
                    bar_open if gap else active_stop,
                    "trail" if trail_price is not None else "stop",
                    index,
                )
                break
            if target_hit:
                exit_price, reason, outcome_index = target, "target", index
                break
            favorable = (high - entry) / risk if side > 0 else (entry - low) / risk
            peak_r = max(peak_r, favorable)
            if peak_r >= self.config.trail_activation_r:
                trail_r = peak_r - self.config.trail_giveback_r
            if local_close >= cutoff_local:
                exit_price = float(frame.at[index, "close"])
                reason, outcome_index = "session", index
                break
        gross_r = side * (exit_price - entry) / risk
        cost_r = self.config.round_trip_commission / (risk * self.config.point_value)
        cost_r += self.config.slippage_ticks * self.config.tick_size / risk
        return {
            "outcome_index": outcome_index,
            "exit_price": exit_price,
            "exit_reason": reason,
            "gross_r": gross_r,
            "cost_r": cost_r,
            "net_r": gross_r - cost_r,
        }

    def _session_cutoff(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        start = _clock(self.config.session_start)
        end = _clock(self.config.session_exit)
        local_time = timestamp.time().replace(tzinfo=None)
        cutoff = timestamp.replace(
            hour=end.hour,
            minute=end.minute,
            second=0,
            microsecond=0,
        )
        if local_time >= start:
            cutoff += pd.offsets.Day(1)
        return cutoff

    def _in_session(self, timestamp: pd.Timestamp, close_timestamp: pd.Timestamp) -> bool:
        start = _clock(self.config.entry_start)
        end = _clock(self.config.session_exit)
        local_time = timestamp.time().replace(tzinfo=None)
        within_hours = start <= local_time <= end
        return bool(within_hours and close_timestamp <= self._session_cutoff(timestamp))

    def run(self, frame: pd.DataFrame, artifact_path: Path) -> dict[str, Any]:
        """Generate, fit, and atomically publish the pass or stopped artifact."""
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        if artifact_path.exists() or temporary.exists():
            raise ExpectedRScreenError(f"artifact path already exists: {artifact_path}")
        artifact = self.fit(self.generate_candidates(frame))
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("x") as stream:
            stream.write(json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True) + "\n")
        temporary.replace(artifact_path)
        return artifact

    def fit(
        self,
        candidates: pd.DataFrame,
        *,
        ridge_predictor: RidgePredictor | None = None,
        threshold_selector: ThresholdSelector | None = None,
        interval_evaluator: IntervalEvaluator | None = None,
        stage_reporter: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Fit train-only scaling/ridge, freeze validation threshold, and evaluate test."""
        features = candidates.attrs.get("raw_features")
        if not isinstance(features, np.ndarray) or len(features) != len(candidates):
            raise ExpectedRScreenError("candidates are missing bound raw-window features")
        original_attrs = candidates.attrs.copy()
        try:
            del candidates.attrs["raw_features"]
            working_candidates = candidates.copy(deep=False)
        finally:
            candidates.attrs.clear()
            candidates.attrs.update(original_attrs)
        candidates = working_candidates
        decision = pd.to_datetime(candidates["decision_ts"], utc=True)
        masks = {
            "train": self._split_mask(candidates, self.config.train_start, self.config.train_end),
            "validation": self._split_mask(
                candidates,
                self.config.validation_start,
                self.config.validation_end,
            ),
            "test": self._split_mask(candidates, self.config.test_start, self.config.test_end),
        }
        if any(not mask.any() for mask in masks.values()):
            empty = [name for name, mask in masks.items() if not mask.any()]
            raise ExpectedRScreenError(f"empty chronological split: {', '.join(empty)}")
        train = masks["train"]
        weights = candidates["average_uniqueness"].to_numpy(dtype=np.float64, copy=True)
        for mask in masks.values():
            weights[mask] = _average_uniqueness(
                candidates.loc[mask, "entry_index"].to_numpy(dtype=np.int64),
                candidates.loc[mask, "outcome_index"].to_numpy(dtype=np.int64),
            )
        train_weight = weights[train]
        mean, variance = self._weighted_stats(features, train, train_weight)
        scale = np.sqrt(variance)
        scale[scale == 0] = 1.0
        mean32 = mean.astype(np.float32)
        scale32 = scale.astype(np.float32)
        scaled = features.copy()
        for start in range(0, len(scaled), 4096):
            chunk = scaled[start : start + 4096]
            chunk -= mean32
            chunk /= scale32
        targets = candidates["net_r"].to_numpy(dtype=np.float64)
        if ridge_predictor is None:
            model = Ridge(alpha=self.config.ridge_alpha, fit_intercept=True, solver="lsqr")
            model.fit(scaled[train], targets[train], sample_weight=train_weight)
            predictions = model.predict(scaled)
        else:
            predictions = ridge_predictor(scaled, train, targets, weights, self.config.ridge_alpha)
            if predictions.shape != targets.shape or not np.isfinite(predictions).all():
                raise ExpectedRScreenError("ridge predictor returned invalid predictions")
        validation = masks["validation"]
        active_days = np.unique(self._session_keys(decision[validation])).size
        desired = max(1, round(active_days * self.config.entries_per_active_session))
        validation_rows = candidates.loc[validation]
        validation_scores = predictions[validation]
        if threshold_selector is None:
            thresholds = np.unique(validation_scores)
            threshold = float(
                min(
                    thresholds,
                    key=lambda value: (
                        abs(
                            int(
                                self._executed_mask(validation_rows, validation_scores, value).sum()
                            )
                            - desired
                        ),
                        -value,
                    ),
                )
            )
        else:
            threshold = threshold_selector(validation_rows, validation_scores, desired)
        if stage_reporter is not None:
            stage_reporter("threshold_complete")
        test = masks["test"]
        if stage_reporter is not None:
            stage_reporter("test_selection_started")
        selected = self._executed_mask(candidates.loc[test], predictions[test], threshold)
        if stage_reporter is not None:
            stage_reporter("test_selection_complete")
        test_y = targets[test]
        selected_y = test_y[selected]
        train_mean = float(np.average(targets[train], weights=train_weight))
        mse = float(np.mean((predictions[test] - test_y) ** 2))
        constant_mse = float(np.mean((train_mean - test_y) ** 2))
        selected_expectancy = float(selected_y.mean()) if len(selected_y) else None
        take_all = float(test_y.mean())
        if stage_reporter is not None:
            stage_reporter("metrics_complete")
        interval_rows = candidates.loc[test]
        intervals = (
            self._paired_intervals(interval_rows, test_y, predictions[test], selected, train_mean)
            if interval_evaluator is None
            else interval_evaluator(interval_rows, test_y, predictions[test], selected, train_mean)
        )
        if stage_reporter is not None:
            stage_reporter("intervals_complete")
        buckets = self._score_buckets(predictions[test], test_y)
        if stage_reporter is not None:
            stage_reporter("buckets_complete")
        gate_parts = {
            "mse_beats_constant": mse < constant_mse,
            "expectancy_positive": bool(
                selected_expectancy is not None and selected_expectancy > 0
            ),
            "expectancy_beats_take_all": bool(
                selected_expectancy is not None and selected_expectancy > take_all
            ),
        }
        rows_payload = [
            {
                "row_id": row_id,
                "prediction": float(prediction),
                "realized_net_r": float(realized),
                "average_uniqueness": float(weight),
            }
            for row_id, prediction, realized, weight in zip(
                candidates["row_id"], predictions, targets, weights, strict=True
            )
        ]
        if stage_reporter is not None:
            stage_reporter("rows_complete")
        source_sha = source_digest(Path(__file__).resolve().parents[3])
        artifact: dict[str, Any] = {
            "schema_version": 1,
            "status": "passed" if all(gate_parts.values()) else "stopped",
            "config_sha256": _sha256_json(asdict(self.config)),
            "data_sha256": candidates.attrs.get("data_sha256"),
            "source_sha256": source_sha,
            "features": {
                "window_bars": self.config.window_bars,
                "market_columns": list(_MARKET_COLUMNS),
                "context_columns": list(_CONTEXT_COLUMNS),
                "regularization": self.config.ridge_alpha,
                "dtype": str(features.dtype),
                "scaler_fit": "train_only_weighted",
            },
            "rows": {"count": len(candidates), "values": rows_payload},
            "threshold": {
                "value": threshold,
                "selected_on": "validation",
                "target_entries_per_active_session": self.config.entries_per_active_session,
            },
            "splits": {
                name: {
                    "rows": int(mask.sum()),
                    **({"threshold": threshold} if name == "test" else {}),
                }
                for name, mask in masks.items()
            },
            "test": {
                "mse": mse,
                "training_mean_constant_mse": constant_mse,
                "selected_expectancy": selected_expectancy,
                "take_all_expectancy": take_all,
                "selected_trades": int(selected.sum()),
                "active_sessions": int(np.unique(self._session_keys(decision[test])).size),
                "score_buckets": buckets,
                "paired_stationary_day_block_intervals_95": intervals,
            },
            "gate": {"passed": all(gate_parts.values()), **gate_parts},
        }
        artifact["artifact_sha256"] = _sha256_json(artifact)
        return artifact

    @staticmethod
    def _weighted_stats(
        features: np.ndarray,
        train_mask: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = np.flatnonzero(train_mask)
        total_weight = float(weights.sum())
        feature_sum = np.zeros(features.shape[1], dtype=np.float64)
        square_sum = np.zeros(features.shape[1], dtype=np.float64)
        for start in range(0, len(indices), 1024):
            selected = indices[start : start + 1024]
            chunk = features[selected]
            chunk_weights = weights[start : start + len(selected), np.newaxis]
            feature_sum += np.sum(chunk * chunk_weights, axis=0, dtype=np.float64)
            square_sum += np.sum(chunk * chunk * chunk_weights, axis=0, dtype=np.float64)
        mean = feature_sum / total_weight
        variance = np.maximum(square_sum / total_weight - mean * mean, 0.0)
        return mean, variance

    @staticmethod
    def _date_mask(
        decisions: pd.Series,
        outcomes: pd.Series,
        start: str,
        end: str,
    ) -> np.ndarray:
        start_timestamp = pd.Timestamp(start, tz="UTC")
        end_timestamp = pd.Timestamp(end, tz="UTC")
        return np.asarray(
            (
                (decisions >= start_timestamp)
                & (decisions < end_timestamp)
                & (outcomes < end_timestamp)
            ).to_numpy(),
            dtype=bool,
        )

    def _split_mask(self, candidates: pd.DataFrame, start: str, end: str) -> np.ndarray:
        decisions = pd.to_datetime(candidates["decision_ts"], utc=True)
        outcomes = pd.to_datetime(candidates["outcome_ts"], utc=True)
        if "feature_start_ts" in candidates:
            feature_starts = pd.to_datetime(candidates["feature_start_ts"], utc=True)
        else:
            feature_starts = decisions
        start_timestamp = pd.Timestamp(start, tz="UTC")
        return self._date_mask(decisions, outcomes, start, end) & np.asarray(
            (feature_starts >= start_timestamp).to_numpy(), dtype=bool
        )

    @staticmethod
    def _executed_mask(rows: pd.DataFrame, scores: np.ndarray, threshold: float) -> np.ndarray:
        selected = np.zeros(len(rows), dtype=bool)
        busy_until = -1
        for offset, ((_, row), score) in enumerate(zip(rows.iterrows(), scores, strict=True)):
            if int(row["entry_index"]) > busy_until and score >= threshold:
                selected[offset] = True
                busy_until = int(row["outcome_index"])
        return selected

    @staticmethod
    def _score_buckets(scores: np.ndarray, outcomes: np.ndarray) -> list[dict[str, Any]]:
        quantiles = np.quantile(scores, np.linspace(0, 1, 6))
        buckets: list[dict[str, Any]] = []
        for index in range(5):
            mask = (scores >= quantiles[index]) & (
                scores <= quantiles[index + 1] if index == 4 else scores < quantiles[index + 1]
            )
            buckets.append(
                {
                    "lower": float(quantiles[index]),
                    "upper": float(quantiles[index + 1]),
                    "rows": int(mask.sum()),
                    "realized_mean_net_r": float(outcomes[mask].mean()) if mask.any() else None,
                }
            )
        return buckets

    def _paired_intervals(
        self,
        rows: pd.DataFrame,
        outcomes: np.ndarray,
        predictions: np.ndarray,
        selected: np.ndarray,
        training_mean: float,
    ) -> dict[str, list[float] | None]:
        days = self._session_keys(rows["decision_ts"])
        unique_days = np.unique(days)
        if len(unique_days) < 2 or not selected.any():
            return {
                "mse_improvement_over_constant": None,
                "selected_expectancy": None,
                "selected_minus_take_all": None,
            }
        rng = np.random.default_rng(self.config.seed)
        selected_means = np.empty(self.config.bootstrap_replicates)
        differences = np.empty(self.config.bootstrap_replicates)
        mse_improvements = np.empty(self.config.bootstrap_replicates)
        for replicate in range(self.config.bootstrap_replicates):
            sampled: list[np.datetime64] = []
            position = int(rng.integers(len(unique_days)))
            while len(sampled) < len(unique_days):
                sampled.append(unique_days[position])
                if rng.random() < self.config.bootstrap_restart_probability:
                    position = int(rng.integers(len(unique_days)))
                else:
                    position = (position + 1) % len(unique_days)
            indices = np.concatenate([np.flatnonzero(days == day) for day in sampled])
            chosen = indices[selected[indices]]
            selected_means[replicate] = outcomes[chosen].mean() if len(chosen) else np.nan
            differences[replicate] = selected_means[replicate] - outcomes[indices].mean()
            mse_improvements[replicate] = np.mean(
                (training_mean - outcomes[indices]) ** 2
            ) - np.mean((predictions[indices] - outcomes[indices]) ** 2)

        def interval(values: np.ndarray) -> list[float] | None:
            finite = values[np.isfinite(values)]
            if not len(finite):
                return None
            return [float(value) for value in np.quantile(finite, [0.025, 0.975])]

        return {
            "mse_improvement_over_constant": interval(mse_improvements),
            "selected_expectancy": interval(selected_means),
            "selected_minus_take_all": interval(differences),
        }

    def _session_keys(self, timestamps: pd.Series) -> np.ndarray:
        """Map close timestamps to the Chicago trading session they belong to."""
        local = pd.to_datetime(timestamps, utc=True).dt.tz_convert(self.config.session_timezone)
        start_hour, start_minute = (int(part) for part in self.config.session_start.split(":"))
        minute = local.dt.hour * 60 + local.dt.minute
        starts_next_date = minute >= start_hour * 60 + start_minute
        offsets = pd.to_timedelta(starts_next_date.to_numpy(dtype=np.int8), unit="D")
        keys = local.dt.normalize() + offsets
        values: list[str] = keys.dt.strftime("%Y-%m-%d").tolist()
        return np.asarray(values, dtype="datetime64[D]")
