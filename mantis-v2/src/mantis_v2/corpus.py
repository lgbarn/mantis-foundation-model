"""Contract-aware repair and immutable Parquet publication for futures corpora."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyarrow import ArrowInvalid

from mantis_v2.contamination import detect_discontinuities, stream_report
from mantis_v2.corpus_config import CorpusRepairConfig, CorpusSource


class CorpusRepairError(RuntimeError):
    """Raised when a source cannot produce a validated immutable corpus."""


_OHLC = ["open", "high", "low", "close"]
_OHLCV = [*_OHLC, "volume"]
_EXACT_FUTURE = re.compile(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d{1,2})$")
_MONTH_NUMBER = {code: month for month, code in enumerate("FGHJKMNQUVXZ", start=1)}
_CANONICAL_MONTHS = {
    "ES": frozenset("HMUZ"),
    "NQ": frozenset("HMUZ"),
    "RTY": frozenset("HMUZ"),
    "YM": frozenset("HMUZ"),
    "GC": frozenset("GJMQVZ"),
    "SI": frozenset("HKNUZ"),
    "CL": frozenset("FGHJKMNQUVXZ"),
    "ZB": frozenset("HMUZ"),
    "ZN": frozenset("HMUZ"),
}
_CHUNK_ROWS = 1_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class _Contract:
    symbol: str
    bars: pd.DataFrame


@dataclass(frozen=True)
class _Continuous:
    raw: pd.DataFrame
    adjusted: pd.DataFrame
    rolls: pd.DataFrame
    repairs: pd.DataFrame


def _json_digest(payload: object) -> str:
    encoded = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _file_identity(path: Path, rows: int, corpus_root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(corpus_root)),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
    }


def _builder_identity() -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[3]
    source_paths = sorted((repository_root / "mantis-v2" / "src").rglob("*.py"))
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(str(path.relative_to(repository_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "mantis-v2/src",
                "mantis-v2/pyproject.toml",
                "uv.lock",
            ],
            cwd=repository_root,
            capture_output=True,
            check=False,
            text=True,
        ).stdout.strip()
    )
    lock_path = repository_root / "uv.lock"
    if not lock_path.is_file():
        raise CorpusRepairError("uv.lock is required for corpus provenance")
    return {
        "source_revision": revision or "uncommitted",
        "source_dirty": dirty,
        "source_digest": digest.hexdigest(),
        "lock_digest": sha256_file(lock_path),
        "databento_version": importlib.metadata.version("databento"),
    }


def _cache_dir(config: CorpusRepairConfig, source: CorpusSource, source_sha: str) -> Path:
    symbols = "-".join(source.symbols)
    return config.output_root / ".mantis-dbn-cache" / f"{source_sha[:16]}-{symbols}"


def _decoder_digest() -> str:
    return _json_digest(
        {
            "schema_version": 3,
            "source_sha256": sha256_file(Path(__file__)),
            "databento_version": importlib.metadata.version("databento"),
            "canonical_months": {
                symbol: sorted(months) for symbol, months in _CANONICAL_MONTHS.items()
            },
        }
    )


def _cache_is_valid(
    cache: Path,
    manifest: object,
    source_sha: str,
    source: CorpusSource,
) -> bool:
    if not isinstance(manifest, dict):
        return False
    if (
        manifest.get("schema_version") != 3
        or manifest.get("source_sha256") != source_sha
        or manifest.get("decoder_digest") != _decoder_digest()
        or manifest.get("symbols") != list(source.symbols)
    ):
        return False
    identities = manifest.get("parts")
    if not isinstance(identities, list) or not identities:
        return False
    expected_paths: set[Path] = set()
    try:
        for identity in identities:
            if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
                return False
            path = cache / identity["path"]
            expected_paths.add(path)
            if (
                not path.is_file()
                or path.stat().st_size != identity.get("size")
                or pq.ParquetFile(path).metadata.num_rows != identity.get("rows")
                or sha256_file(path) != identity.get("sha256")
            ):
                return False
    except (ArrowInvalid, OSError, ValueError):
        return False
    actual_paths = set(cache.glob("*/*/part-*.parquet"))
    return actual_paths == expected_paths


def _chunk_roots(frame: pd.DataFrame, roots: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    if "symbol" not in frame:
        raise CorpusRepairError("DBN decode did not provide contract symbols")
    symbols = frame["symbol"].astype(str)
    parsed = symbols.str.extract(_EXACT_FUTURE)
    parsed.columns = ["root", "month", "year"]
    accepted = parsed["root"].isin(roots)
    for root in roots:
        root_rows = parsed["root"].eq(root)
        accepted &= ~root_rows | parsed["month"].isin(_CANONICAL_MONTHS[root])
    selected = frame.loc[accepted, [*_OHLCV, "symbol"]].copy()
    if selected.empty:
        return {}
    selected.index = pd.to_datetime(selected.index, utc=True)
    selected.index.name = "datetime"
    selected_parsed = parsed.loc[accepted]
    return {
        root: selected.loc[selected_parsed["root"].eq(root).to_numpy()]
        for root in roots
        if selected_parsed["root"].eq(root).any()
    }


def _decode_source(
    config: CorpusRepairConfig, source: CorpusSource
) -> tuple[Path, dict[str, object]]:
    source_sha = sha256_file(source.path)
    if source_sha != source.sha256:
        raise CorpusRepairError(
            f"source archive digest mismatch: {source.path}; expected {source.sha256}, "
            f"got {source_sha}"
        )
    identity = {
        "path": str(source.path.resolve()),
        "size": source.path.stat().st_size,
        "sha256": source_sha,
        "symbols": list(source.symbols),
    }
    cache = _cache_dir(config, source, source_sha)
    manifest_path = cache / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
        if _cache_is_valid(cache, manifest, source_sha, source):
            return cache, identity
        raise CorpusRepairError(
            f"decoded cache failed content validation: {cache}; preserve it for diagnosis "
            "and move it aside before rebuilding"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f"{cache.name}.{os.getpid()}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    root_counts = {symbol: 0 for symbol in source.symbols}
    contract_counts: dict[tuple[str, str], int] = {}
    identities: list[dict[str, object]] = []
    try:
        iterator = db.DBNStore.from_file(str(source.path)).to_df(count=_CHUNK_ROWS)
        for frame in iterator:
            for root, root_frame in _chunk_roots(frame, source.symbols).items():
                for contract_name, contract_frame in root_frame.groupby("symbol", sort=False):
                    contract = str(contract_name)
                    key = (root, contract)
                    part_number = contract_counts.get(key, 0)
                    contract_dir = temporary / root / contract
                    contract_dir.mkdir(parents=True, exist_ok=True)
                    part = contract_dir / f"part-{part_number:05d}.parquet"
                    contract_frame.to_parquet(part)
                    identities.append(_file_identity(part, len(contract_frame), temporary))
                    contract_counts[key] = part_number + 1
                    root_counts[root] += 1
        missing = [symbol for symbol, count in root_counts.items() if count == 0]
        if missing:
            raise CorpusRepairError(
                f"{source.path}: no canonical outright rows for {', '.join(missing)}"
            )
        _atomic_json(
            temporary / "manifest.json",
            {
                "schema_version": 3,
                "source_sha256": source_sha,
                "decoder_digest": _decoder_digest(),
                "symbols": list(source.symbols),
                "parts": identities,
            },
        )
        temporary.rename(cache)
    except Exception:
        raise
    if not _cache_is_valid(
        cache, json.loads((cache / "manifest.json").read_text()), source_sha, source
    ):
        raise CorpusRepairError(f"new decoded cache failed content validation: {cache}")
    return cache, identity


def _load_contracts(cache: Path, symbol: str, config: CorpusRepairConfig) -> list[_Contract]:
    contract_dirs = [path for path in (cache / symbol).iterdir() if path.is_dir()]
    if not contract_dirs:
        raise CorpusRepairError(f"decoded cache has no {symbol} parts")
    ordered = sorted(contract_dirs, key=lambda path: _expiry_symbol_key(path.name))
    selected: list[_Contract] = []
    roll_dates: list[pd.Timestamp] = []
    for contract_dir in ordered:
        name = contract_dir.name
        bars = pd.read_parquet(contract_dir).sort_index()
        bars = bars.loc[(bars.index >= config.start) & (bars.index < config.end)]
        if bars.empty:
            continue
        cleaned, _ = _classify_isolated_bad_prints(bars.drop(columns="symbol"), name)
        contract = _Contract(name, cleaned)
        if not selected:
            selected.append(contract)
            continue
        date = _roll_date(
            selected[-1].bars,
            contract.bars,
            config,
            roll_dates[-1] if roll_dates else None,
        )
        if date is None:
            continue
        selected.append(contract)
        roll_dates.append(date)
    if not selected:
        raise CorpusRepairError(f"{symbol}: no contract rows in configured interval")
    return selected


def _classify_isolated_bad_prints(
    bars: pd.DataFrame, symbol: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify isolated prints without fabricating or deleting source bars."""
    frame = bars.loc[bars["close"] > 0, _OHLCV].copy().sort_index()
    maximum_volume = frame.groupby(level=0, sort=False)["volume"].transform("max")
    frame = frame.loc[frame["volume"].eq(maximum_volume)]
    frame = frame.loc[~frame.index.duplicated(keep="first")].sort_index()
    repairs: list[dict[str, object]] = []
    for _ in range(3):
        previous = frame["close"].shift(1)
        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous).abs(),
                (frame["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        scale = true_range.shift(1).rolling(20, min_periods=20).median()
        positive = true_range.where(true_range > 0).shift(1).expanding(min_periods=1).median()
        scale = scale.mask(scale.eq(0), positive)
        jump = (frame["close"] - previous).abs()
        recovery = pd.Series(False, index=frame.index)
        for offset in range(1, 4):
            recovery |= (frame["open"].shift(-offset) - previous).abs() <= 0.20 * jump
            recovery |= (frame["close"].shift(-offset) - previous).abs() <= 0.20 * jump
        bad = ((jump > 8.0 * scale) & recovery).fillna(False)
        if not bad.any():
            break
        for timestamp, row in frame.loc[bad].iterrows():
            repairs.append(
                {
                    "timestamp": timestamp,
                    "contract": symbol,
                    "reason": "isolated_bad_print_preserved",
                    **{column: float(row[column]) for column in _OHLCV},
                }
            )
        break
    return frame, pd.DataFrame(repairs)


def _expiry_key(contract: _Contract) -> tuple[int, int, pd.Timestamp]:
    match = _EXACT_FUTURE.fullmatch(contract.symbol.upper())
    if match is None:
        return 9999, 12, contract.bars.index[0]
    year_code = int(match.group(3))
    year = 2020 + year_code if year_code < 10 else 2000 + year_code
    return year, _MONTH_NUMBER[match.group(2)], contract.bars.index[0]


def _expiry_symbol_key(symbol: str) -> tuple[int, int, str]:
    match = _EXACT_FUTURE.fullmatch(symbol.upper())
    if match is None:
        return 9999, 12, symbol
    year_code = int(match.group(3))
    year = 2020 + year_code if year_code < 10 else 2000 + year_code
    return year, _MONTH_NUMBER[match.group(2)], symbol


def _session(index: pd.DatetimeIndex, config: CorpusRepairConfig) -> pd.DatetimeIndex:
    local = pd.to_datetime(index, utc=True).tz_convert(config.session_timezone)
    offset = 24 - config.session_start_hour
    return (local + timedelta(hours=offset)).normalize()


def _roll_date(
    old: pd.DataFrame,
    new: pd.DataFrame,
    config: CorpusRepairConfig,
    after: pd.Timestamp | None,
) -> pd.Timestamp | None:
    if after is not None:
        old = old.loc[old.index >= after]
        new = new.loc[new.index >= after]
    old_daily = (
        pd.Series(old["volume"].to_numpy(), index=_session(old.index, config))
        .groupby(level=0)
        .sum()
    )
    new_daily = (
        pd.Series(new["volume"].to_numpy(), index=_session(new.index, config))
        .groupby(level=0)
        .sum()
    )
    daily = pd.concat({"old": old_daily, "new": new_daily}, axis=1, join="inner").dropna()
    required = config.roll_confirmation_sessions
    if len(daily) <= required:
        return None
    wins = daily["new"] > daily["old"]
    confirmed = wins.rolling(required, min_periods=required).sum().eq(required)
    if not confirmed.any():
        return None
    confirmation = confirmed[confirmed].index[0]
    new_sessions = _session(new.index, config).unique().sort_values()
    later = new_sessions[new_sessions > confirmation]
    if not len(later):
        return None
    activation = later[0]
    sessions = _session(new.index, config)
    candidates = new.index[sessions >= activation]
    return candidates[0] if len(candidates) else None


def _roll_ratio(old: pd.DataFrame, new: pd.DataFrame, date: pd.Timestamp) -> float:
    old_before = old.loc[old.index <= date]
    new_after = new.loc[new.index >= date]
    if old_before.empty or new_after.empty:
        raise CorpusRepairError(f"cannot price roll at {date}")
    if date - old_before.index[-1] > timedelta(days=3):
        raise CorpusRepairError(f"old contract is stale at roll {date}")
    if new_after.index[0] - date > timedelta(days=3):
        raise CorpusRepairError(f"new contract is stale at roll {date}")
    ratio = float(new_after["close"].iloc[0]) / float(old_before["close"].iloc[-1])
    if not np.isfinite(ratio) or ratio <= 0:
        raise CorpusRepairError(f"invalid ratio at roll {date}")
    return ratio


def _build_continuous(
    symbol: str, contracts: list[_Contract], config: CorpusRepairConfig
) -> _Continuous:
    cleaned: list[_Contract] = []
    repairs: list[pd.DataFrame] = []
    anomaly_timestamps: dict[str, set[pd.Timestamp]] = {}
    for contract in contracts:
        bars, dropped = _classify_isolated_bad_prints(contract.bars, contract.symbol)
        if not bars.empty:
            cleaned.append(_Contract(contract.symbol, bars))
        if not dropped.empty:
            repairs.append(dropped)
            flagged = anomaly_timestamps.setdefault(contract.symbol, set())
            for timestamp in pd.to_datetime(dropped["timestamp"], utc=True):
                flagged.add(timestamp)
                position = bars.index.get_indexer([timestamp])[0]
                for offset in range(1, 4):
                    recovery_position = position + offset
                    if 0 <= recovery_position < len(bars):
                        flagged.add(bars.index[recovery_position])
    cleaned.sort(key=_expiry_key)
    if not cleaned:
        raise CorpusRepairError(f"{symbol}: no clean contracts")
    selected = [cleaned[0]]
    roll_dates: list[pd.Timestamp] = []
    ratios: list[float] = []
    next_index = 1
    while next_index < len(cleaned):
        old = selected[-1]
        chosen: _Contract | None = None
        chosen_index = next_index
        chosen_date: pd.Timestamp | None = None
        for candidate_index in range(next_index, len(cleaned)):
            candidate = cleaned[candidate_index]
            date = _roll_date(
                old.bars,
                candidate.bars,
                config,
                roll_dates[-1] if roll_dates else None,
            )
            if date is not None:
                chosen = candidate
                chosen_index = candidate_index
                chosen_date = date
                break
        if chosen is None or chosen_date is None:
            break
        selected.append(chosen)
        roll_dates.append(chosen_date)
        ratios.append(_roll_ratio(old.bars, chosen.bars, chosen_date))
        next_index = chosen_index + 1
    cleaned = selected
    factors = np.ones(len(cleaned), dtype=np.float64)
    for index in range(len(cleaned) - 2, -1, -1):
        factors[index] = factors[index + 1] * ratios[index]
    bounds: list[pd.Timestamp | None] = [None, *roll_dates, None]
    raw_parts: list[pd.DataFrame] = []
    adjusted_parts: list[pd.DataFrame] = []
    for index, contract in enumerate(cleaned):
        lower, upper = bounds[index], bounds[index + 1]
        mask = np.ones(len(contract.bars), dtype=bool)
        if lower is not None:
            mask &= contract.bars.index >= lower
        if upper is not None:
            mask &= contract.bars.index < upper
        raw = contract.bars.loc[mask, _OHLCV].copy()
        if raw.empty:
            continue
        raw["contract"] = contract.symbol
        raw["segment"] = index
        raw["adjustment_factor"] = factors[index]
        raw["quality_flag"] = raw.index.isin(anomaly_timestamps.get(contract.symbol, set()))
        adjusted = raw.copy()
        adjusted[_OHLC] = adjusted[_OHLC].to_numpy() * factors[index]
        raw_parts.append(raw)
        adjusted_parts.append(adjusted)
    raw = pd.concat(raw_parts).sort_index()
    adjusted = pd.concat(adjusted_parts).sort_index()
    if raw.index.duplicated().any() or adjusted.index.duplicated().any():
        raise CorpusRepairError(f"{symbol}: duplicate timestamps after stable roll")
    transitions = raw["contract"].ne(raw["contract"].shift()).to_numpy()
    transition_dates = raw.index[np.flatnonzero(transitions)[1:]].tolist()
    if transition_dates != roll_dates:
        raise CorpusRepairError(f"{symbol}: contract transitions do not match roll ledger")
    rolls = pd.DataFrame(
        {
            "roll_date": roll_dates,
            "old_contract": [cleaned[index].symbol for index in range(len(roll_dates))],
            "new_contract": [cleaned[index + 1].symbol for index in range(len(roll_dates))],
            "ratio": ratios,
            "old_cumulative_factor": factors[:-1],
            "new_cumulative_factor": factors[1:],
        }
    )
    repair_frame = pd.concat(repairs, ignore_index=True) if repairs else pd.DataFrame()
    return _Continuous(raw=raw, adjusted=adjusted, rolls=rolls, repairs=repair_frame)


def _resample(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "1min":
        return frame.copy()
    aggregate = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "contract": "last",
        "segment": "last",
        "adjustment_factor": "last",
        "quality_flag": "max",
    }
    result = frame.resample(timeframe, closed="left", label="left", origin="epoch").agg(aggregate)
    return result.dropna(subset=["open", "high", "low", "close"])


def _validate_frame(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        raise CorpusRepairError(f"{label}: empty output")
    if not frame.index.is_monotonic_increasing or frame.index.duplicated().any():
        raise CorpusRepairError(f"{label}: timestamps must be sorted and unique")
    numeric = frame[_OHLCV].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise CorpusRepairError(f"{label}: non-finite OHLCV")
    if (frame[_OHLC] <= 0).any().any() or (frame["volume"] < 0).any():
        raise CorpusRepairError(f"{label}: invalid price or volume")
    if (
        (frame["high"] < frame[["open", "close"]].max(axis=1)).any()
        or (frame["low"] > frame[["open", "close"]].min(axis=1)).any()
        or (frame["high"] < frame["low"]).any()
    ):
        raise CorpusRepairError(f"{label}: OHLC envelope violation")


def _validate_coverage(frame: pd.DataFrame, config: CorpusRepairConfig, label: str) -> None:
    tolerance = timedelta(days=7)
    if frame.index[0].to_pydatetime() > config.start + tolerance:
        raise CorpusRepairError(f"{label}: starts too late at {frame.index[0]}")
    if frame.index[-1].to_pydatetime() < config.end - tolerance:
        raise CorpusRepairError(f"{label}: ends too early at {frame.index[-1]}")


def _validate_persisted_aggregation(
    source: pd.DataFrame,
    path: Path,
    timeframe: str,
    label: str,
) -> None:
    persisted = pd.read_parquet(path).set_index("datetime")
    persisted.index = pd.to_datetime(persisted.index, utc=True)
    if timeframe == "1min":
        expected = source
    else:
        buckets = source.index.floor(timeframe)
        grouped = source.groupby(buckets, sort=True)
        expected = pd.DataFrame(
            {
                "open": grouped["open"].first(),
                "high": grouped["high"].max(),
                "low": grouped["low"].min(),
                "close": grouped["close"].last(),
                "volume": grouped["volume"].sum(),
                "contract": grouped["contract"].last(),
                "segment": grouped["segment"].last(),
                "adjustment_factor": grouped["adjustment_factor"].last(),
                "quality_flag": grouped["quality_flag"].max(),
            }
        )
    if not expected.index.equals(persisted.index):
        raise CorpusRepairError(f"{label}: persisted aggregation timestamps mismatch")
    numeric_columns = [*_OHLCV, "segment", "adjustment_factor", "quality_flag"]
    if not np.allclose(
        expected[numeric_columns].to_numpy(dtype=np.float64),
        persisted[numeric_columns].to_numpy(dtype=np.float64),
        rtol=1e-12,
        atol=0.0,
    ) or not expected["contract"].equals(persisted["contract"]):
        raise CorpusRepairError(f"{label}: persisted aggregation values mismatch")


def _write_frame(frame: pd.DataFrame, path: Path, corpus_root: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.reset_index(names="datetime")
    output.to_parquet(path, index=False)
    return _file_identity(path, len(output), corpus_root)


def validate_corpus(corpus_root: Path) -> dict[str, object]:
    """Verify a published corpus manifest and every content-addressed output."""
    manifest_path = corpus_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError as exc:
        raise CorpusRepairError(f"corpus manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise CorpusRepairError(f"invalid corpus manifest: {manifest_path}") from exc
    recorded_digest = manifest.pop("manifest_digest", None)
    if recorded_digest != _json_digest(manifest):
        raise CorpusRepairError("corpus manifest digest mismatch")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise CorpusRepairError("corpus manifest contains no outputs")
    root = corpus_root.resolve()
    checked_rows = 0
    for identity in outputs:
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            raise CorpusRepairError("corpus manifest contains an invalid output identity")
        path = (root / identity["path"]).resolve()
        if path.parent != root and root not in path.parents:
            raise CorpusRepairError("corpus manifest output escapes the corpus root")
        if not path.is_file():
            raise CorpusRepairError(f"corpus output missing: {path}")
        if path.stat().st_size != identity.get("size"):
            raise CorpusRepairError(f"corpus output size mismatch: {path}")
        if sha256_file(path) != identity.get("sha256"):
            raise CorpusRepairError(f"corpus output digest mismatch: {path}")
        rows = pq.ParquetFile(path).metadata.num_rows
        if rows != identity.get("rows"):
            raise CorpusRepairError(f"corpus output row-count mismatch: {path}")
        checked_rows += rows
    if manifest.get("validated") is not True:
        raise CorpusRepairError("corpus manifest is not marked validated")
    manifest["manifest_digest"] = recorded_digest
    return {
        "corpus_id": manifest.get("corpus_id"),
        "manifest_digest": recorded_digest,
        "outputs_checked": len(outputs),
        "rows_checked": checked_rows,
        "valid": True,
    }


def validate_corpus_binding(
    market_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    required_files: list[Path],
) -> str:
    """Fail closed unless selected market files match a validated corpus manifest."""
    try:
        actual_manifest_sha = sha256_file(manifest_path)
    except FileNotFoundError as exc:
        raise CorpusRepairError(f"configured corpus manifest not found: {manifest_path}") from exc
    if actual_manifest_sha != manifest_sha256:
        raise CorpusRepairError(f"configured corpus manifest digest mismatch: {manifest_path}")
    _validate_bound_corpus(str(manifest_path.parent.resolve()), manifest_sha256)
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise CorpusRepairError(f"invalid corpus manifest: {manifest_path}") from exc
    recorded_digest = manifest.pop("manifest_digest", None)
    if recorded_digest != _json_digest(manifest) or manifest.get("validated") is not True:
        raise CorpusRepairError("configured corpus manifest is invalid or unvalidated")
    corpus_root = manifest_path.parent.resolve()
    if market_root.resolve() != corpus_root / "market":
        raise CorpusRepairError("Parquet data root must be the bound corpus market directory")
    identities = {
        str(identity.get("path")): identity
        for identity in manifest.get("outputs", [])
        if isinstance(identity, dict) and identity.get("kind") == "market"
    }
    for path in required_files:
        relative = str(path.resolve().relative_to(corpus_root))
        identity = identities.get(relative)
        if identity is None:
            raise CorpusRepairError(f"configured stream is absent from corpus manifest: {path}")
        if not path.is_file() or path.stat().st_size != identity.get("size"):
            raise CorpusRepairError(f"configured stream size mismatch: {path}")
        if sha256_file(path) != identity.get("sha256"):
            raise CorpusRepairError(f"configured stream digest mismatch: {path}")
    return str(recorded_digest)


@lru_cache(maxsize=8)
def _validate_bound_corpus(corpus_root: str, manifest_sha256: str) -> None:
    manifest_path = Path(corpus_root) / "manifest.json"
    if sha256_file(manifest_path) != manifest_sha256:
        raise CorpusRepairError(f"configured corpus manifest digest mismatch: {manifest_path}")
    validate_corpus(Path(corpus_root))


def repair_corpus(config: CorpusRepairConfig) -> dict[str, object]:
    """Build, validate, and atomically publish an immutable repaired corpus."""
    output = config.output_path
    if output.exists():
        if not config.allow_overwrite:
            raise CorpusRepairError(f"corpus already exists: {output}")
        raise CorpusRepairError(
            "immutable corpus overwrite is not supported; choose a new corpus_id"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.{os.getpid()}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    source_for_symbol: dict[str, Path] = {}
    source_identities: list[dict[str, object]] = []
    for source in config.sources:
        cache, identity = _decode_source(config, source)
        source_identities.append(identity)
        for symbol in source.symbols:
            source_for_symbol[symbol] = cache
    outputs: list[dict[str, object]] = []
    quality: list[dict[str, object]] = []
    accepted_lookup = {
        (item.symbol, pd.Timestamp(item.timestamp).tz_convert("UTC")): item.reason
        for item in config.accepted_dislocations
    }
    used_acceptances: set[tuple[str, pd.Timestamp]] = set()
    for symbol in config.symbols:
        continuous = _build_continuous(
            symbol, _load_contracts(source_for_symbol[symbol], symbol, config), config
        )
        _validate_frame(continuous.raw, f"{symbol} raw 1min")
        _validate_frame(continuous.adjusted, f"{symbol} adjusted 1min")
        _validate_coverage(continuous.raw, config, f"{symbol} raw 1min")
        _validate_coverage(continuous.adjusted, config, f"{symbol} adjusted 1min")
        reconciled = (
            continuous.raw[_OHLC].to_numpy()
            * continuous.raw["adjustment_factor"].to_numpy()[:, None]
        )
        if not np.allclose(reconciled, continuous.adjusted[_OHLC].to_numpy(), rtol=1e-12):
            raise CorpusRepairError(f"{symbol}: adjustment reconciliation failed")
        raw_path = partial / "raw" / f"{symbol}_1min.parquet"
        outputs.append(
            {
                "kind": "raw",
                "symbol": symbol,
                "timeframe": "1min",
                **_write_frame(continuous.raw, raw_path, partial),
            }
        )
        for timeframe in config.timeframes:
            market = _resample(continuous.adjusted, timeframe)
            _validate_frame(market, f"{symbol} {timeframe}")
            market_path = partial / "market" / f"{symbol}_{timeframe}.parquet"
            identity = _write_frame(market, market_path, partial)
            _validate_persisted_aggregation(
                continuous.adjusted,
                market_path,
                timeframe,
                f"{symbol} {timeframe}",
            )
            outputs.append(
                {
                    "kind": "market",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    **identity,
                }
            )
        rolls_path = partial / "rolls" / f"{symbol}.parquet"
        repairs_path = partial / "repairs" / f"{symbol}.parquet"
        outputs.append(
            {
                "kind": "rolls",
                "symbol": symbol,
                **_write_frame(
                    continuous.rolls.set_index("roll_date")
                    if not continuous.rolls.empty
                    else pd.DataFrame(index=pd.DatetimeIndex([], name="roll_date")),
                    rolls_path,
                    partial,
                ),
            }
        )
        repairs = continuous.repairs
        if repairs.empty:
            repairs = pd.DataFrame(columns=["timestamp", "contract", "reason", *_OHLCV])
        outputs.append(
            {
                "kind": "repairs",
                "symbol": symbol,
                **_write_frame(
                    repairs.set_index(pd.to_datetime(repairs["timestamp"], utc=True)).drop(
                        columns="timestamp"
                    ),
                    repairs_path,
                    partial,
                ),
            }
        )
        boundaries, events = detect_discontinuities(
            continuous.adjusted.index.to_numpy(dtype="datetime64[ns]"),
            continuous.adjusted[["high", "low", "close"]].to_numpy(),
            config.max_relative_price_dislocation,
        )
        transition_rows = set(
            np.flatnonzero(
                continuous.raw["contract"].ne(continuous.raw["contract"].shift()).to_numpy()
            )[1:].tolist()
        )
        roll_dislocations = [event for event in events if event.row in transition_rows]
        if roll_dislocations:
            raise CorpusRepairError(
                f"{symbol}: {len(roll_dislocations)} adjusted roll boundaries still exceed "
                "the dislocation threshold"
            )
        classified_rows = set(
            np.flatnonzero(continuous.adjusted["quality_flag"].to_numpy(dtype=bool)).tolist()
        )
        unclassified: list[str] = []
        accepted_events: list[dict[str, str]] = []
        for event in events:
            if event.row in transition_rows or event.row in classified_rows:
                continue
            timestamp = pd.Timestamp(event.timestamp)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            key = (symbol, timestamp)
            if key not in accepted_lookup:
                unclassified.append(f"{symbol}@{timestamp.isoformat()}")
                continue
            used_acceptances.add(key)
            accepted_events.append(
                {"timestamp": timestamp.isoformat(), "reason": accepted_lookup[key]}
            )
        if unclassified:
            preview = ", ".join(unclassified[:10])
            raise CorpusRepairError(
                f"{symbol}: unclassified same-contract dislocations: {preview}; "
                "review the source and add explicit accepted_dislocations entries"
            )
        quality.append(
            {
                "symbol": symbol,
                "rolls": len(continuous.rolls),
                "repairs": len(continuous.repairs),
                "remaining_dislocations": int(boundaries.sum()),
                "roll_dislocations": 0,
                "classified_anomaly_rows": len(classified_rows),
                "accepted_same_contract_dislocations": accepted_events,
                "dislocations": stream_report(f"{symbol}_1min", events),
            }
        )
    unused_acceptances = set(accepted_lookup) - used_acceptances
    if unused_acceptances:
        unused = ", ".join(
            f"{symbol}@{timestamp.isoformat()}" for symbol, timestamp in sorted(unused_acceptances)
        )
        raise CorpusRepairError(f"configured accepted dislocations were not observed: {unused}")
    config_payload = asdict(config)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "corpus_id": config.corpus_id,
        "config": config_payload,
        "config_digest": _json_digest(config_payload),
        "builder": _builder_identity(),
        "sources": source_identities,
        "outputs": outputs,
        "quality": quality,
        "validated": True,
    }
    manifest["manifest_digest"] = _json_digest(manifest)
    _atomic_json(partial / "manifest.json", manifest)
    validate_corpus(partial)
    partial.rename(output)
    return manifest
