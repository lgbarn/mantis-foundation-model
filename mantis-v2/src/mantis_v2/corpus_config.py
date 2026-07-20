"""Strict configuration for immutable contract-aware corpus repair."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mantis_v2.config import ConfigError


@dataclass(frozen=True)
class CorpusSource:
    path: Path
    sha256: str
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class AcceptedDislocation:
    symbol: str
    timestamp: datetime
    reason: str


@dataclass(frozen=True)
class CorpusRepairConfig:
    corpus_id: str
    output_root: Path
    sources: tuple[CorpusSource, ...]
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    start: datetime
    end: datetime
    roll_confirmation_sessions: int
    session_timezone: str
    session_start_hour: int
    adjustment: str
    max_relative_price_dislocation: float
    accepted_dislocations: tuple[AcceptedDislocation, ...]
    allow_overwrite: bool

    @property
    def output_path(self) -> Path:
        return self.output_root / self.corpus_id


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid [{name}] section")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ConfigError(f"unknown [{name}] keys: {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigError(f"missing [{name}] keys: {', '.join(sorted(missing))}")


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ConfigError(f"{field} must be a non-empty string array")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ConfigError(f"{field} must not contain duplicates")
    return result


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be an ISO-8601 string")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"{field} is not valid ISO-8601") from exc
    if result.tzinfo is None:
        raise ConfigError(f"{field} must include a timezone")
    return result


def load_corpus_repair_config(path: str | Path) -> CorpusRepairConfig:
    """Load one fully explicit corpus-repair contract."""
    config_path = Path(path)
    try:
        raw = tomllib.loads(config_path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc
    if set(raw) != {"corpus", "sources"}:
        raise ConfigError("corpus repair config requires only [corpus] and [[sources]]")
    corpus = _mapping(raw["corpus"], "corpus")
    expected = {
        "corpus_id",
        "output_root",
        "symbols",
        "timeframes",
        "start",
        "end",
        "roll_confirmation_sessions",
        "session_timezone",
        "session_start_hour",
        "adjustment",
        "max_relative_price_dislocation",
        "accepted_dislocations",
        "allow_overwrite",
    }
    _exact_keys(corpus, expected, "corpus")
    source_rows = raw["sources"]
    if not isinstance(source_rows, list) or not source_rows:
        raise ConfigError("[[sources]] must contain at least one source")
    sources: list[CorpusSource] = []
    for index, value in enumerate(source_rows):
        source = _mapping(value, f"sources[{index}]")
        _exact_keys(source, {"path", "sha256", "symbols"}, f"sources[{index}]")
        source_sha256 = str(source["sha256"])
        if len(source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in source_sha256
        ):
            raise ConfigError(f"sources[{index}].sha256 must be a lowercase SHA-256 digest")
        sources.append(
            CorpusSource(
                path=Path(str(source["path"])),
                sha256=source_sha256,
                symbols=_strings(source["symbols"], f"sources[{index}].symbols"),
            )
        )
    symbols = _strings(corpus["symbols"], "corpus.symbols")
    provided = tuple(symbol for source in sources for symbol in source.symbols)
    if len(set(provided)) != len(provided) or set(provided) != set(symbols):
        raise ConfigError("sources must map every corpus symbol exactly once")
    timeframes = _strings(corpus["timeframes"], "corpus.timeframes")
    if timeframes != ("1min", "3min", "5min", "15min"):
        raise ConfigError("corpus.timeframes must be ordered 1min, 3min, 5min, 15min")
    start = _timestamp(corpus["start"], "corpus.start")
    end = _timestamp(corpus["end"], "corpus.end")
    if start >= end:
        raise ConfigError("corpus.start must be earlier than corpus.end")
    confirmations = corpus["roll_confirmation_sessions"]
    if not isinstance(confirmations, int) or isinstance(confirmations, bool) or confirmations < 1:
        raise ConfigError("corpus.roll_confirmation_sessions must be an integer >= 1")
    start_hour = corpus["session_start_hour"]
    if not isinstance(start_hour, int) or isinstance(start_hour, bool) or not 0 <= start_hour <= 23:
        raise ConfigError("corpus.session_start_hour must be in [0, 23]")
    threshold = corpus["max_relative_price_dislocation"]
    if not isinstance(threshold, int | float) or isinstance(threshold, bool):
        raise ConfigError("corpus.max_relative_price_dislocation must be numeric")
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ConfigError("corpus.max_relative_price_dislocation must be finite and positive")
    if corpus["adjustment"] != "ratio_back_adjusted":
        raise ConfigError("corpus.adjustment must be ratio_back_adjusted")
    accepted_raw = corpus["accepted_dislocations"]
    if not isinstance(accepted_raw, list):
        raise ConfigError("corpus.accepted_dislocations must be an array of tables")
    accepted: list[AcceptedDislocation] = []
    accepted_keys: set[tuple[str, datetime]] = set()
    for index, value in enumerate(accepted_raw):
        item = _mapping(value, f"corpus.accepted_dislocations[{index}]")
        _exact_keys(
            item,
            {"symbol", "timestamp", "reason"},
            f"corpus.accepted_dislocations[{index}]",
        )
        accepted_symbol = str(item["symbol"])
        accepted_timestamp = _timestamp(
            item["timestamp"], f"corpus.accepted_dislocations[{index}].timestamp"
        )
        reason = str(item["reason"]).strip()
        if accepted_symbol not in symbols or not reason:
            raise ConfigError("accepted dislocations require a configured symbol and reason")
        key = (accepted_symbol, accepted_timestamp)
        if key in accepted_keys:
            raise ConfigError("corpus.accepted_dislocations must not contain duplicates")
        accepted_keys.add(key)
        accepted.append(AcceptedDislocation(accepted_symbol, accepted_timestamp, reason))
    if not isinstance(corpus["allow_overwrite"], bool):
        raise ConfigError("corpus.allow_overwrite must be true or false")
    if not all(source.path.is_file() for source in sources):
        missing = [str(source.path) for source in sources if not source.path.is_file()]
        raise ConfigError("missing corpus source archives: " + ", ".join(missing))
    return CorpusRepairConfig(
        corpus_id=str(corpus["corpus_id"]),
        output_root=Path(str(corpus["output_root"])),
        sources=tuple(sources),
        symbols=symbols,
        timeframes=timeframes,
        start=start,
        end=end,
        roll_confirmation_sessions=confirmations,
        session_timezone=str(corpus["session_timezone"]),
        session_start_hour=start_hour,
        adjustment=str(corpus["adjustment"]),
        max_relative_price_dislocation=threshold,
        accepted_dislocations=tuple(accepted),
        allow_overwrite=corpus["allow_overwrite"],
    )
