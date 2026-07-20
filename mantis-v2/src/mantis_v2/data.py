"""Leak-safe stream loading and NextLeg target construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.nn import functional as F
from torch.utils.data import Dataset

from mantis_v2.config import DataConfig, ModelConfig, TargetConfig
from mantis_v2.contamination import (
    Discontinuity,
    detect_discontinuities,
    report_digest,
    stream_report,
)
from mantis_v2.corpus import CorpusRepairError, validate_corpus_binding

Split = Literal["train", "validation", "holdout"]


class DataContractError(ValueError):
    """Raised when source data violates the configured schema or split contract."""


@dataclass(frozen=True)
class Stream:
    name: str
    timestamps: np.ndarray
    values: np.ndarray
    discontinuities: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    discontinuity_events: tuple[Discontinuity, ...] = ()


@dataclass(frozen=True)
class StreamSummary:
    name: str
    rows: int
    first_timestamp: str
    last_timestamp: str


@dataclass(frozen=True)
class Anchor:
    stream_index: int
    confirmation: int
    first_leg: int
    second_leg: int


def discover_paths(config: DataConfig) -> list[Path]:
    """Resolve all configured symbol/interval files and fail on any omission."""
    if config.root == "synthetic":
        return []
    root = Path(config.root)
    if not root.is_dir():
        raise DataContractError(f"data root does not exist: {root}")
    extension = config.file_format
    paths = [
        root / f"{symbol}_{interval}.{extension}"
        for symbol in config.symbols
        for interval in config.intervals
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise DataContractError("missing configured streams: " + ", ".join(missing))
    if config.file_format == "parquet":
        assert config.corpus_manifest_path is not None
        try:
            validate_corpus_binding(
                root,
                config.corpus_manifest_path,
                config.corpus_manifest_sha256,
                paths,
            )
        except CorpusRepairError as exc:
            raise DataContractError(str(exc)) from exc
    return paths


def _read_columns(path: Path, columns: list[str], file_format: str) -> pd.DataFrame:
    if file_format == "parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def inspect_streams(config: DataConfig) -> list[StreamSummary]:
    """Read only timestamp columns to validate the external corpus cheaply."""
    if config.root == "synthetic":
        stream = synthetic_stream()
        return [_summarize(stream)]
    summaries: list[StreamSummary] = []
    for path in discover_paths(config):
        frame = _read_columns(path, [config.timestamp_column], config.file_format)
        if frame.empty:
            raise DataContractError(f"empty stream: {path}")
        timestamps = pd.to_datetime(frame[config.timestamp_column], utc=True, errors="raise")
        if not timestamps.is_monotonic_increasing:
            raise DataContractError(f"timestamps are not sorted: {path}")
        summaries.append(
            StreamSummary(
                name=path.stem,
                rows=len(frame),
                first_timestamp=timestamps.iloc[0].isoformat(),
                last_timestamp=timestamps.iloc[-1].isoformat(),
            )
        )
    return summaries


def load_streams(config: DataConfig) -> list[Stream]:
    """Load configured OHLCV streams independently as float32 arrays."""
    if config.root == "synthetic":
        return [synthetic_stream()]
    required = [config.timestamp_column, *config.feature_columns]
    streams: list[Stream] = []
    for path in discover_paths(config):
        columns = [*required, "quality_flag"] if config.file_format == "parquet" else required
        frame = _read_columns(path, columns, config.file_format)
        if frame.empty:
            raise DataContractError(f"empty stream: {path}")
        timestamps = pd.to_datetime(frame[config.timestamp_column], utc=True, errors="raise")
        if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise DataContractError(f"timestamps must be sorted and unique: {path}")
        values = frame.loc[:, config.feature_columns].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise DataContractError(f"non-finite feature value: {path}")
        boundaries, events = detect_discontinuities(
            timestamps.to_numpy(dtype="datetime64[ns]"),
            values[:, [1, 2, 3]],
            config.max_relative_close_jump,
        )
        if config.file_format == "parquet":
            boundaries |= frame["quality_flag"].to_numpy(dtype=bool)
        streams.append(
            Stream(
                name=path.stem,
                timestamps=timestamps.to_numpy(dtype="datetime64[ns]"),
                values=values,
                discontinuities=boundaries,
                discontinuity_events=events,
            )
        )
    return streams


def synthetic_stream(rows: int = 4096) -> Stream:
    """Create a deterministic OHLCV fixture with alternating local pivots."""
    x = np.arange(rows, dtype=np.float32)
    close = 100.0 + 3.0 * np.sin(x / 9.0) + 0.4 * np.sin(x / 2.7)
    open_ = close + 0.05 * np.sin(x / 3.0)
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    volume = 1000.0 + 50.0 * (1.0 + np.sin(x / 13.0))
    values = np.column_stack((open_, high, low, close, volume)).astype(np.float32)
    start = np.datetime64("2025-01-01T00:00:00", "ns")
    timestamps = start + np.arange(rows).astype("timedelta64[m]")
    return Stream(name="SYNTH_1min", timestamps=timestamps, values=values)


def _summarize(stream: Stream) -> StreamSummary:
    return StreamSummary(
        name=stream.name,
        rows=len(stream.values),
        first_timestamp=str(stream.timestamps[0]),
        last_timestamp=str(stream.timestamps[-1]),
    )


def contamination_report(streams: list[Stream], config: DataConfig) -> dict[str, object]:
    """Return the canonical detector report bound into training provenance."""
    report: dict[str, object] = {
        "schema_version": 1,
        "max_relative_close_jump": config.max_relative_close_jump,
        "streams": [stream_report(stream.name, stream.discontinuity_events) for stream in streams],
    }
    report["digest"] = report_digest(report)
    return report


def split_bounds(stream: Stream, config: DataConfig, split: Split) -> tuple[int, int]:
    """Return half-open bounds with the 2026 holdout isolated from tuning."""
    holdout = np.datetime64(
        config.holdout_start.astimezone(UTC).replace(tzinfo=None),
        "ns",
    )
    holdout_index = int(np.searchsorted(stream.timestamps, holdout, side="left"))
    validation_start = int(holdout_index * (1.0 - config.validation_fraction))
    bounds = {
        "train": (0, validation_start),
        "validation": (validation_start, holdout_index),
        "holdout": (holdout_index, len(stream.values)),
    }
    return bounds[split]


def alternating_pivots(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect unique centered pivots and discard repeated directions, per stream."""
    if len(values) < 2 * k + 1:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty
    highs = sliding_window_view(values[:, 1], 2 * k + 1)
    lows = sliding_window_view(values[:, 2], 2 * k + 1)
    high_center = highs[:, k]
    low_center = lows[:, k]
    is_high = (high_center == highs.max(axis=1)) & (
        (highs == high_center[:, None]).sum(axis=1) == 1
    )
    is_low = (low_center == lows.min(axis=1)) & ((lows == low_center[:, None]).sum(axis=1) == 1)
    is_high &= ~is_low
    candidates = np.flatnonzero(is_high | is_low).astype(np.int64) + k
    directions = np.where(is_low[is_high | is_low], -1, 1).astype(np.int8)
    if not len(candidates):
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty
    keep = np.ones(len(candidates), dtype=bool)
    keep[1:] = directions[1:] != directions[:-1]
    origins = candidates[keep]
    directions = directions[keep]
    confirmations = origins + k
    return origins, confirmations, directions


def build_anchors(
    streams: list[Stream],
    data: DataConfig,
    target: TargetConfig,
    split: Split,
) -> list[Anchor]:
    """Build two-leg targets whose context and complete future remain in one split."""
    max_context = max(data.context_lengths)
    max_horizon = max(target.horizons)
    anchors: list[Anchor] = []
    for stream_index, stream in enumerate(streams):
        start, end = split_bounds(stream, data, split)
        boundaries = (
            np.flatnonzero(stream.discontinuities).astype(np.int64)
            if len(stream.discontinuities)
            else np.array([], dtype=np.int64)
        )
        cuts: list[tuple[int, int]] = []
        segment_start = 0
        for boundary in boundaries:
            if segment_start < boundary:
                cuts.append((segment_start, int(boundary)))
            segment_start = int(boundary) + 1
        if segment_start < len(stream.values):
            cuts.append((segment_start, len(stream.values)))
        for segment_start, segment_end in cuts:
            segment_values = (
                stream.values
                if segment_start == 0 and segment_end == len(stream.values)
                else stream.values[segment_start:segment_end]
            )
            origins, confirmations, _ = alternating_pivots(segment_values, target.leg_k)
            origins += segment_start
            confirmations += segment_start
            legal_start = max(start, int(segment_start))
            legal_end = min(end, int(segment_end))
            for index in range(len(origins) - 2):
                confirmation = int(confirmations[index])
                first_leg = int(origins[index + 1] - confirmation)
                second_leg = int(origins[index + 2] - origins[index + 1])
                complete_future = int(origins[index + 2])
                if first_leg <= 0 or second_leg <= 0:
                    continue
                if first_leg > target.leg_cap or second_leg > target.leg_cap:
                    continue
                if confirmation - max_context + 1 < legal_start:
                    continue
                if max(complete_future, confirmation + max_horizon) >= legal_end:
                    continue
                anchors.append(Anchor(stream_index, confirmation, first_leg, second_leg))
    return anchors


class NextLegDataset(Dataset[dict[str, torch.Tensor]]):
    """Materialize normalized, resized contexts and both NextLeg target heads."""

    def __init__(
        self,
        streams: list[Stream],
        anchors: list[Anchor],
        data: DataConfig,
        model: ModelConfig,
        target: TargetConfig,
    ) -> None:
        self.streams = streams
        self.anchors = anchors
        self.data = data
        self.model = model
        self.target = target

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        anchor = self.anchors[index]
        stream = self.streams[anchor.stream_index]
        context_length = self.data.context_lengths[index % len(self.data.context_lengths)]
        start = anchor.confirmation - context_length + 1
        context = stream.values[start : anchor.confirmation + 1].T.copy()
        mean = context.mean(axis=1, keepdims=True)
        std = context.std(axis=1, keepdims=True)
        std = np.maximum(std, 1e-6)
        normalized = np.clip(
            (context - mean) / std,
            -self.target.normalization_clamp,
            self.target.normalization_clamp,
        )
        context_tensor = torch.from_numpy(normalized).unsqueeze(0)
        resized = F.interpolate(
            context_tensor,
            size=self.model.input_length,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

        horizons = np.asarray(self.target.horizons, dtype=np.int64)
        future = stream.values[anchor.confirmation + horizons].T
        future_normalized = np.clip(
            (future - mean) / std,
            -self.target.normalization_clamp,
            self.target.normalization_clamp,
        )
        candle_target = future_normalized - normalized[:, -1:]
        leg_target = np.log1p(np.asarray([anchor.first_leg, anchor.second_leg], dtype=np.float32))
        return {
            "context": resized.to(torch.float32),
            "candle_target": torch.from_numpy(candle_target.astype(np.float32)),
            "leg_target": torch.from_numpy(leg_target),
            "stream_index": torch.tensor(anchor.stream_index, dtype=torch.int64),
            "confirmation": torch.tensor(anchor.confirmation, dtype=torch.int64),
        }
