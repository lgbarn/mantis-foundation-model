"""Deterministic detection and reporting of discontinuous market-data boundaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Discontinuity:
    row: int
    timestamp: str
    previous_close: float
    close: float
    relative_jump: float


def detect_discontinuities(
    timestamps: np.ndarray,
    prices: np.ndarray,
    max_relative_jump: float,
) -> tuple[np.ndarray, tuple[Discontinuity, ...]]:
    """Mark bars whose OHLC excursion from the previous close exceeds the limit."""
    if len(timestamps) != len(prices):
        raise ValueError("timestamps and price values must have equal length")
    if prices.ndim == 1:
        high = low = close = prices.astype(np.float64, copy=False)
    elif prices.ndim == 2 and prices.shape[1] == 3:
        high, low, close = prices.astype(np.float64, copy=False).T
    else:
        raise ValueError("prices must be close values or ordered high, low, close rows")
    boundaries = np.zeros(len(close), dtype=bool)
    if len(close) < 2:
        return boundaries, ()
    previous_close = close[:-1]
    previous = np.abs(previous_close)
    denominator = np.maximum(previous, np.finfo(np.float64).eps)
    jumps = (
        np.maximum.reduce(
            (
                np.abs(high[1:] - previous_close),
                np.abs(low[1:] - previous_close),
                np.abs(close[1:] - previous_close),
            )
        )
        / denominator
    )
    rows = np.flatnonzero(jumps > max_relative_jump).astype(np.int64) + 1
    boundaries[rows] = True
    events = tuple(
        Discontinuity(
            row=int(row),
            timestamp=str(timestamps[row]),
            previous_close=float(previous_close[row - 1]),
            close=float(close[row]),
            relative_jump=float(jumps[row - 1]),
        )
        for row in rows
    )
    return boundaries, events


def report_digest(report: dict[str, object]) -> str:
    """Hash one canonical contamination report."""
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def stream_report(name: str, events: tuple[Discontinuity, ...]) -> dict[str, object]:
    """Serialize one stream's ordered findings."""
    return {
        "stream": name,
        "count": len(events),
        "events": [asdict(event) for event in events],
    }
