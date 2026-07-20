"""Partition-safe Monte Carlo episode schedules and mmap observation lookup."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from mantis_v2.downstream_config import DownstreamConfig, load_downstream_config
from mantis_v2.rl_config import RlConfig
from mantis_v2.rl_provenance import _build_manifest, sha256_file
from mantis_v2.walk_forward import build_folds, fold_masks


class EpisodeContractError(RuntimeError):
    """Raised when an episode schedule cannot prove temporal ownership."""


@dataclass(frozen=True)
class Partition:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class Episode:
    number: int
    ticker: str
    profile: str
    start_session: date
    end_session: date
    trading_days: int
    first_candidate: int
    candidate_count: int
    observation_spans: tuple[ObservationSpan, ...]
    lookback_start: pd.Timestamp
    decision_start: pd.Timestamp
    exit_end: pd.Timestamp
    terminal_end: pd.Timestamp


@dataclass(frozen=True)
class ObservationSpan:
    shard: int
    first_row: int
    row_count: int


_CHICAGO = ZoneInfo("America/Chicago")


def _session_id(timestamp: pd.Timestamp) -> date:
    local = timestamp.tz_convert(_CHICAGO)
    return cast(date, (local.to_pydatetime() - timedelta(hours=17)).date())


def _session_ordinal(session: date) -> int:
    trade_date = session + timedelta(days=1)
    return int(np.busday_count("1970-01-01", trade_date.isoformat()))


def _session_is_complete(values: pd.Series) -> bool:
    ordered = values.sort_values()
    if len(ordered) < 2:
        return False
    gaps = ordered.diff().dropna()
    first_local = ordered.iloc[0].tz_convert(_CHICAGO)
    last_local = ordered.iloc[-1].tz_convert(_CHICAGO)
    return bool(
        (17, 0) <= (first_local.hour, first_local.minute) <= (17, 3)
        and last_local.date() > first_local.date()
        and (last_local.hour, last_local.minute) >= (15, 9)
        and (gaps.empty or gaps.max() <= timedelta(minutes=3))
    )


def _spans(rows: pd.DataFrame) -> tuple[ObservationSpan, ...]:
    spans: list[ObservationSpan] = []
    for shard, group in rows.groupby("shard", sort=False):
        offsets = group["row"].astype(int).to_numpy()
        if len(offsets) > 1 and not np.all(np.diff(offsets) == 1):
            raise EpisodeContractError("episode observation rows are not contiguous")
        spans.append(ObservationSpan(int(shard), int(offsets[0]), len(offsets)))
    return tuple(spans)


def _valid_windows(
    ticker: str, rows: pd.DataFrame, partition: Partition, trading_days: int
) -> list[
    tuple[
        date,
        date,
        int,
        int,
        tuple[ObservationSpan, ...],
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
    ]
]:
    required = {
        "symbol",
        "decision_ts",
        "label_end_ts",
        "lookback_start_ts",
        "terminal_ts",
        "session_complete",
        "session_ordinal",
        "rollover_safe",
        "horizon_complete",
        "shard",
        "row",
    }
    missing = required - set(rows.columns)
    if missing:
        raise EpisodeContractError(
            f"episode metadata missing columns: {', '.join(sorted(missing))}"
        )
    if set(rows["symbol"].unique()) != {ticker}:
        raise EpisodeContractError(f"episode metadata does not match owning stream: {ticker}")
    frame = rows.copy()
    for column in ("decision_ts", "label_end_ts", "lookback_start_ts", "terminal_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if not frame["decision_ts"].is_monotonic_increasing or frame["decision_ts"].duplicated().any():
        raise EpisodeContractError("episode metadata must be strictly chronological")
    owned = frame[
        (frame["lookback_start_ts"] >= partition.start)
        & (frame["decision_ts"] >= partition.start)
        & (frame["label_end_ts"] < partition.end)
        & (frame["terminal_ts"] < partition.end)
    ].copy()
    owned["session"] = owned["decision_ts"].map(_session_id)
    sessions = list(dict.fromkeys(owned["session"].tolist()))
    windows: list[
        tuple[
            date,
            date,
            int,
            int,
            tuple[ObservationSpan, ...],
            pd.Timestamp,
            pd.Timestamp,
            pd.Timestamp,
            pd.Timestamp,
        ]
    ] = []
    for offset in range(len(sessions) - trading_days + 1):
        selected = sessions[offset : offset + trading_days]
        mask = owned["session"].isin(selected)
        indices = np.flatnonzero(mask.to_numpy())
        if len(indices) == 0:
            continue
        selected_rows = owned.iloc[indices]
        ordinals = (
            selected_rows.groupby("session", sort=False)["session_ordinal"].first().to_numpy()
        )
        if len(ordinals) > 1 and not np.all(np.diff(ordinals) == 1):
            continue
        if not selected_rows["session_complete"].astype(bool).all():
            continue
        if not selected_rows["rollover_safe"].astype(bool).all():
            continue
        if not selected_rows["horizon_complete"].astype(bool).all():
            continue
        windows.append(
            (
                selected[0],
                selected[-1],
                int(selected_rows.index[0]),
                len(indices),
                _spans(selected_rows),
                pd.Timestamp(selected_rows["lookback_start_ts"].min()),
                pd.Timestamp(selected_rows["decision_ts"].min()),
                pd.Timestamp(selected_rows["label_end_ts"].max()),
                pd.Timestamp(selected_rows["terminal_ts"].max()),
            )
        )
    return windows


def _balanced_values(
    values: tuple[str, ...], count: int, generator: np.random.Generator
) -> list[str]:
    result = [values[index % len(values)] for index in range(count)]
    generator.shuffle(result)
    return result


def build_episode_schedule(
    sources: Mapping[str, pd.DataFrame],
    partition: Partition,
    *,
    seed: int,
    episode_count: int,
    trading_days: int = 20,
) -> tuple[Episode, ...]:
    """Build a reproducible, balanced schedule from complete ticker-local windows."""
    if episode_count < 1:
        raise EpisodeContractError("episode_count must be positive")
    tickers = tuple(sorted(sources))
    if not tickers:
        raise EpisodeContractError("episode sources are empty")
    generator = np.random.default_rng(seed)
    windows = {
        ticker: _valid_windows(ticker, sources[ticker], partition, trading_days)
        for ticker in tickers
    }
    empty = [ticker for ticker, available in windows.items() if not available]
    if empty:
        raise EpisodeContractError(f"no complete episode windows for: {', '.join(empty)}")
    selected_tickers = _balanced_values(tickers, episode_count, generator)
    profile_queues = {
        ticker: _balanced_values(
            ("one_mini",) if ticker == "ZB" else ("one_mini", "ten_micros"),
            selected_tickers.count(ticker),
            generator,
        )
        for ticker in tickers
    }
    profile_offsets = {ticker: 0 for ticker in tickers}
    episodes: list[Episode] = []
    for number, ticker in enumerate(selected_tickers):
        (
            start,
            end,
            first,
            candidate_count,
            spans,
            lookback_start,
            decision_start,
            exit_end,
            terminal_end,
        ) = windows[ticker][int(generator.integers(0, len(windows[ticker])))]
        profile_index = profile_offsets[ticker]
        profile = profile_queues[ticker][profile_index]
        profile_offsets[ticker] += 1
        episodes.append(
            Episode(
                number,
                ticker,
                profile,
                start,
                end,
                trading_days,
                first,
                candidate_count,
                spans,
                lookback_start,
                decision_start,
                exit_end,
                terminal_end,
            )
        )
    return tuple(episodes)


def read_observation(path: Path, row: int, width: int) -> np.ndarray:
    """Read one immutable feature row without loading its shard into memory."""
    features = np.load(path, mmap_mode="r", allow_pickle=False)
    if features.ndim != 2 or features.shape[1] != width:
        raise EpisodeContractError("embedding shard feature width mismatch")
    if row < 0 or row >= features.shape[0]:
        raise EpisodeContractError("embedding observation row is out of range")
    observation = np.asarray(features[row], dtype=np.float32)
    if not np.isfinite(observation).all():
        raise EpisodeContractError("embedding observation is not finite")
    return observation


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeContractError(f"{label} manifest is invalid") from exc
    if not isinstance(raw, dict):
        raise EpisodeContractError(f"{label} manifest is invalid")
    return raw


def _verified_path(identity: object, root: Path, label: str) -> Path:
    if not isinstance(identity, dict):
        raise EpisodeContractError(f"{label} identity is invalid")
    path_value = identity.get("path")
    expected = identity.get("sha256")
    size = identity.get("size")
    if (
        not isinstance(path_value, str)
        or not isinstance(expected, str)
        or not isinstance(size, int)
    ):
        raise EpisodeContractError(f"{label} identity is invalid")
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != expected:
        raise EpisodeContractError(f"{label} content identity mismatch")
    return path


def _validate_corpus(
    config: RlConfig, downstream: DownstreamConfig, repository_root: Path
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[str, set[date]],
    dict[str, dict[date, int]],
    dict[str, set[date]],
]:
    path = config.upstream.corpus_manifest_path
    if not path.is_absolute():
        path = repository_root / path
    manifest = _load_json(path, "corpus")
    if manifest.get("validated") is not True:
        raise EpisodeContractError("corpus is not validated")
    quality_rows = manifest.get("quality")
    if not isinstance(quality_rows, list):
        raise EpisodeContractError("corpus quality identity is invalid")
    reviewed_roll_sessions: dict[str, set[date]] = {}
    for quality in quality_rows:
        if not isinstance(quality, dict):
            raise EpisodeContractError("corpus quality identity is invalid")
        accepted_rolls = quality.get("accepted_roll_dislocations", [])
        accepted_same = quality.get("accepted_same_contract_dislocations", [])
        if not isinstance(accepted_rolls, list) or not isinstance(accepted_same, list):
            raise EpisodeContractError("corpus quality identity is invalid")
        if quality.get("roll_dislocations") != len(accepted_rolls) or quality.get(
            "remaining_dislocations"
        ) != len(accepted_rolls) + len(accepted_same):
            raise EpisodeContractError("corpus contains an unreviewed rollover discontinuity")
        symbol = str(quality.get("symbol"))
        reviewed_roll_sessions[symbol] = {
            _session_id(pd.Timestamp(str(event["timestamp"])))
            for event in accepted_rolls
            if isinstance(event, dict) and "timestamp" in event
        }
    timestamps: dict[tuple[str, str], np.ndarray] = {}
    rollover_sessions = {
        symbol: set(reviewed_roll_sessions.get(symbol, set())) for symbol in downstream.data.symbols
    }
    session_ordinals: dict[str, dict[date, int]] = {}
    complete_sessions: dict[str, set[date]] = {}
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise EpisodeContractError("corpus manifest has no outputs")
    wanted = {
        (symbol, timeframe)
        for symbol in downstream.data.symbols
        for timeframe in downstream.data.timeframes
    }
    for identity in outputs:
        if not isinstance(identity, dict) or identity.get("kind") != "market":
            continue
        key = (str(identity.get("symbol")), str(identity.get("timeframe")))
        if key not in wanted:
            continue
        stream_path = _verified_path(identity, path.parent, f"corpus stream {key[0]} {key[1]}")
        columns = [downstream.data.timestamp_column]
        if key[1] == "3min":
            columns.append("segment")
        frame = pd.read_parquet(stream_path, columns=columns)
        values = pd.to_datetime(frame[downstream.data.timestamp_column], utc=True, errors="raise")
        if not values.is_monotonic_increasing or values.duplicated().any():
            raise EpisodeContractError(f"corpus stream is not chronological: {key[0]} {key[1]}")
        timestamps[key] = values.to_numpy()
        if key[1] == "3min":
            sessions = values.map(_session_id)
            ordered_sessions = list(dict.fromkeys(sessions.tolist()))
            session_ordinals[key[0]] = {
                session: _session_ordinal(session) for session in ordered_sessions
            }
            complete: set[date] = set()
            for session, group in values.groupby(sessions):
                if _session_is_complete(group):
                    complete.add(session)
            complete_sessions[key[0]] = complete
            segments = frame["segment"].to_numpy()
            changes = np.flatnonzero(segments[1:] != segments[:-1]) + 1
            rollover_sessions[key[0]].update(
                _session_id(pd.Timestamp(values.iloc[i])) for i in changes
            )
    if set(timestamps) != wanted:
        raise EpisodeContractError("corpus manifest is missing an owned stream")
    return timestamps, rollover_sessions, session_ordinals, complete_sessions


def _load_embeddings(
    config: RlConfig,
    timestamps: Mapping[tuple[str, str], np.ndarray],
    downstream: DownstreamConfig,
    repository_root: Path,
    rollover_sessions: Mapping[str, set[date]],
    session_ordinals: Mapping[str, Mapping[date, int]],
    complete_sessions: Mapping[str, set[date]],
) -> tuple[pd.DataFrame, list[dict[str, object]], int]:
    manifest_path = config.upstream.embedding_manifest_path
    if not manifest_path.is_absolute():
        manifest_path = repository_root / manifest_path
    manifest = _load_json(manifest_path, "embedding")
    width = manifest.get("feature_width")
    outputs = manifest.get("outputs")
    if not isinstance(width, int) or not isinstance(outputs, list):
        raise EpisodeContractError("embedding manifest contract is invalid")
    frames: list[pd.DataFrame] = []
    output_records: list[dict[str, object]] = []
    total_rows = 0
    for identity in outputs:
        if not isinstance(identity, dict) or not isinstance(identity.get("number"), int):
            raise EpisodeContractError("embedding shard identity is invalid")
        number = int(identity["number"])
        features_path = _verified_path(
            identity.get("features"), manifest_path.parent, f"features shard {number}"
        )
        metadata_path = _verified_path(
            identity.get("metadata"), manifest_path.parent, f"metadata shard {number}"
        )
        features = np.load(features_path, mmap_mode="r", allow_pickle=False)
        rows = identity.get("rows")
        if features.shape != (rows, width):
            raise EpisodeContractError(f"embedding shard shape mismatch: {number}")
        metadata = pd.read_parquet(metadata_path)
        if len(metadata) != rows:
            raise EpisodeContractError(f"embedding metadata row count mismatch: {number}")
        metadata["shard"] = number
        metadata["row"] = np.arange(len(metadata), dtype=np.int64)
        frames.append(metadata)
        total_rows += len(metadata)
        output_records.append(
            {
                "number": number,
                "rows": rows,
                "features": identity["features"],
                "metadata": identity["metadata"],
            }
        )
    if total_rows != manifest.get("rows"):
        raise EpisodeContractError("embedding manifest total row count mismatch")
    metadata = pd.concat(frames, ignore_index=True)
    context = int(downstream.data.context_bars)
    lookbacks = pd.Series(pd.NaT, index=metadata.index, dtype="datetime64[ns, UTC]")
    for symbol, group in metadata.groupby("symbol", sort=False):
        starts: list[pd.DatetimeIndex] = []
        for timeframe in downstream.data.timeframes:
            indices = group[f"{timeframe}_index"].to_numpy(dtype=np.int64)
            if np.any(indices < context - 1) or np.any(
                indices >= len(timestamps[(str(symbol), timeframe)])
            ):
                raise EpisodeContractError("embedding context index is out of range")
            starts.append(
                pd.DatetimeIndex(timestamps[(str(symbol), timeframe)][indices - context + 1])
            )
        nanos = np.minimum.reduce([values.asi8 for values in starts])
        lookbacks.loc[group.index] = pd.to_datetime(nanos, utc=True)
    metadata["lookback_start_ts"] = lookbacks
    terminal = pd.Series(pd.NaT, index=metadata.index, dtype="datetime64[ns, UTC]")
    horizon_complete = pd.Series(False, index=metadata.index, dtype=bool)
    for symbol, group in metadata.groupby("symbol", sort=False):
        indices = group["3min_index"].to_numpy(dtype=np.int64) + int(config.exit.horizon_bars)
        valid = indices < len(timestamps[(str(symbol), "3min")])
        if np.any(valid):
            terminal.loc[group.index[valid]] = pd.to_datetime(
                timestamps[(str(symbol), "3min")][indices[valid]], utc=True
            )
        horizon_complete.loc[group.index] = valid
    metadata["terminal_ts"] = terminal
    sessions = [_session_id(pd.Timestamp(timestamp)) for timestamp in metadata["decision_ts"]]
    metadata["session_complete"] = [
        session in complete_sessions[str(symbol)]
        for symbol, session in zip(metadata["symbol"], sessions, strict=True)
    ]
    metadata["session_ordinal"] = [
        session_ordinals[str(symbol)].get(session, -1)
        for symbol, session in zip(metadata["symbol"], sessions, strict=True)
    ]
    rollover_safe: list[bool] = []
    for symbol, lookback, terminal, complete in zip(
        metadata["symbol"],
        metadata["lookback_start_ts"],
        metadata["terminal_ts"],
        horizon_complete,
        strict=True,
    ):
        if not complete or pd.isna(terminal):
            rollover_safe.append(False)
            continue
        first_ordinal = _session_ordinal(_session_id(pd.Timestamp(lookback)))
        last_ordinal = _session_ordinal(_session_id(pd.Timestamp(terminal)))
        banned = {_session_ordinal(session) for session in rollover_sessions[str(symbol)]}
        rollover_safe.append(not any(first_ordinal <= value <= last_ordinal for value in banned))
    metadata["rollover_safe"] = rollover_safe
    metadata["horizon_complete"] = horizon_complete
    return metadata, output_records, width


def _atomic_resume(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() == encoded:
            return
        raise EpisodeContractError(f"completed episode manifest already exists: {path}")
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise EpisodeContractError(f"completed episode manifest already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_episode_manifest(
    config: RlConfig,
    *,
    fold_number: int,
    partition_name: str,
    episode_count: int,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Validate immutable inputs, sample episodes, and publish one atomic schedule."""
    if partition_name not in {"training", "validation", "test"}:
        raise EpisodeContractError("partition must be training, validation, or test")
    root = repository_root.resolve() if repository_root else Path(__file__).resolve().parents[3]
    output = (
        config.run.artifact_root
        / config.run.name
        / "episodes"
        / f"fold-{fold_number:02d}-{partition_name}-seed-{config.run.seed}.json"
    )
    identities = _build_manifest(config, root, output)["identities"]
    downstream_path = config.upstream.downstream_config_path
    if not downstream_path.is_absolute():
        downstream_path = root / downstream_path
    downstream = load_downstream_config(downstream_path)
    timestamps, rollover_sessions, session_ordinals, complete_sessions = _validate_corpus(
        config, downstream, root
    )
    metadata, embedding_outputs, width = _load_embeddings(
        config,
        timestamps,
        downstream,
        root,
        rollover_sessions,
        session_ordinals,
        complete_sessions,
    )
    folds = build_folds(metadata, downstream)
    if fold_number < 0 or fold_number >= len(folds):
        raise EpisodeContractError("fold number is out of range")
    fold = folds[fold_number]
    embargo = pd.Timedelta(seconds=180 * int(downstream.walk_forward.embargo_bars))
    boundaries = {
        "training": (fold.train_start, fold.train_end - embargo),
        "validation": (fold.validation_start + embargo, fold.validation_end - embargo),
        "test": (fold.test_start + embargo, fold.test_end),
    }
    start, end = boundaries[partition_name]
    partition = Partition(partition_name, start, end)
    partition_masks = dict(
        zip(("training", "validation", "test"), fold_masks(metadata, fold, downstream), strict=True)
    )
    owned_metadata = metadata.loc[partition_masks[partition_name]].copy()
    sources = {
        symbol: owned_metadata[owned_metadata["symbol"] == symbol].copy()
        for symbol in downstream.data.symbols
    }
    episodes = build_episode_schedule(
        sources,
        partition,
        seed=config.run.seed,
        episode_count=episode_count,
        trading_days=config.episode.timeout_trading_days,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "rl-episode-schedule",
        "fold": fold_number,
        "partition": {"name": partition_name, "start": str(start), "end": str(end)},
        "seed": config.run.seed,
        "episode_count": episode_count,
        "account_start": config.episode.account_start,
        "identities": identities,
        "embedding": {"feature_width": width, "outputs": embedding_outputs},
        "episodes": [asdict(episode) for episode in episodes],
    }
    _atomic_resume(output, payload)
    return payload
