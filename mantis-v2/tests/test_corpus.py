from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mantis_v2 import corpus as corpus_module
from mantis_v2.config import ConfigError
from mantis_v2.corpus import (
    CorpusRepairError,
    _build_continuous,
    _cache_is_valid,
    _classify_isolated_bad_prints,
    _Contract,
    _decode_source,
    _decoder_digest,
    _expiry_symbol_key,
    _file_identity,
    _json_digest,
    _load_contracts,
    _resample,
    _roll_basis,
    _roll_ratio,
    _validate_coverage,
    _validate_persisted_aggregation,
    _write_frame,
    validate_corpus,
    validate_corpus_binding,
)
from mantis_v2.corpus_config import (
    AcceptedDislocation,
    CorpusRepairConfig,
    CorpusSource,
    load_corpus_repair_config,
)
from mantis_v2.provenance import sha256_file


def _config(tmp_path: Path) -> CorpusRepairConfig:
    source = tmp_path / "source.dbn.zst"
    source.touch()
    return CorpusRepairConfig(
        corpus_id="test-corpus",
        output_root=tmp_path,
        sources=(CorpusSource(source, "0" * 64, ("ES",)),),
        symbols=("ES",),
        timeframes=("1min", "3min", "5min", "15min"),
        start=pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime(),
        end=pd.Timestamp("2025-01-10", tz="UTC").to_pydatetime(),
        roll_confirmation_sessions=2,
        session_timezone="UTC",
        session_start_hour=0,
        adjustment="ratio_back_adjusted",
        max_relative_price_dislocation=0.05,
        accepted_dislocations=(),
        allow_overwrite=False,
    )


def _bars(index: pd.DatetimeIndex, close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_corpus_config_is_strict_and_requires_real_sources(tmp_path: Path) -> None:
    source = tmp_path / "source.dbn.zst"
    source.touch()
    config_path = tmp_path / "corpus.toml"
    config_path.write_text(
        f'''[corpus]
corpus_id = "v1"
output_root = "{tmp_path}"
symbols = ["ES"]
timeframes = ["1min", "3min", "5min", "15min"]
start = "2025-01-01T00:00:00+00:00"
end = "2025-02-01T00:00:00+00:00"
roll_confirmation_sessions = 2
session_timezone = "America/Chicago"
session_start_hour = 17
adjustment = "ratio_back_adjusted"
max_relative_price_dislocation = 0.05
accepted_dislocations = []
allow_overwrite = false

[[sources]]
path = "{source}"
sha256 = "{"0" * 64}"
symbols = ["ES"]
'''
    )

    config = load_corpus_repair_config(config_path)

    assert config.symbols == ("ES",)
    assert config.timeframes == ("1min", "3min", "5min", "15min")
    reviewed_path = tmp_path / "reviewed-corpus.toml"
    reviewed_path.write_text(
        config_path.read_text().replace(
            "accepted_dislocations = []",
            'accepted_dislocations = [{ symbol = "ES", timestamp = '
            '"2025-01-04T00:00:00+00:00", kind = "roll", contracts = '
            '["ESF25", "ESH25"], reason = "Reviewed roll." }]',
        )
    )
    reviewed = load_corpus_repair_config(reviewed_path)
    assert reviewed.accepted_dislocations[0].kind == "roll"
    assert reviewed.accepted_dislocations[0].contracts == ("ESF25", "ESH25")
    source.unlink()
    with pytest.raises(ConfigError, match="missing corpus source archives"):
        load_corpus_repair_config(config_path)


def test_roll_requires_two_liquidity_wins_and_activates_next_session(tmp_path: Path) -> None:
    config = _config(tmp_path)
    index = pd.date_range("2025-01-01", periods=5, freq="1D", tz="UTC")
    old = _bars(index, np.full(5, 100.0), np.asarray([100, 40, 30, 20, 10]))
    new = _bars(index, np.full(5, 120.0), np.asarray([50, 60, 70, 80, 90]))

    continuous = _build_continuous("ES", [_Contract("ESH25", old), _Contract("ESM25", new)], config)

    assert continuous.rolls["roll_date"].tolist() == [index[3]]
    assert continuous.raw.loc[index[2], "contract"] == "ESH25"
    assert continuous.raw.loc[index[3], "contract"] == "ESM25"
    assert continuous.raw.loc[index[2], "adjustment_factor"] == pytest.approx(1.2)
    assert continuous.adjusted.loc[index[2], "close"] == pytest.approx(120.0)
    assert continuous.rolls.loc[0, "roll_date"] == index[3]
    assert continuous.rolls.loc[0, "pricing_timestamp"] == index[3]


def test_roll_ratio_rejects_missing_recent_shared_pricing_timestamp() -> None:
    old_index = pd.DatetimeIndex([pd.Timestamp("2025-01-01", tz="UTC")])
    new_index = pd.DatetimeIndex([pd.Timestamp("2025-01-01", tz="UTC")])
    old = _bars(old_index, np.asarray([100.0]), np.ones(1))
    new = _bars(new_index, np.asarray([110.0]), np.ones(1))

    with pytest.raises(CorpusRepairError, match="shared pricing timestamp within three days"):
        _roll_ratio(old, new, pd.Timestamp("2025-01-10", tz="UTC"))

    assert _roll_ratio(old, new, pd.Timestamp("2025-01-04", tz="UTC")) == pytest.approx(1.1)


def test_roll_basis_uses_latest_shared_timestamp_without_looking_forward() -> None:
    activation = pd.Timestamp("2022-12-13 23:00", tz="UTC")
    shared = pd.Timestamp("2022-12-13 21:59", tz="UTC")
    older = activation - timedelta(days=2)
    old_index = pd.DatetimeIndex([older, shared, activation + timedelta(minutes=7)])
    new_index = pd.DatetimeIndex([older, shared, activation])
    old = _bars(old_index, np.asarray([1800.0, 1833.1, 1835.5]), np.ones(3))
    new = _bars(new_index, np.asarray([1900.0, 1848.3, 1848.7]), np.ones(3))

    ratio, pricing_timestamp = _roll_basis(old, new, activation)

    assert pricing_timestamp == shared
    assert ratio == pytest.approx(1848.3 / 1833.1)


def test_roll_chain_skips_an_illiquid_delivery_month(tmp_path: Path) -> None:
    config = _config(tmp_path)
    index = pd.date_range("2025-01-01", periods=5, freq="1D", tz="UTC")
    old = _bars(index, np.full(5, 100.0), np.asarray([100, 90, 80, 20, 10]))
    illiquid = _bars(index, np.full(5, 110.0), np.asarray([1, 1, 1, 1, 1]))
    next_front = _bars(index, np.full(5, 120.0), np.asarray([10, 20, 90, 100, 110]))

    continuous = _build_continuous(
        "CL",
        [
            _Contract("CLF25", old),
            _Contract("CLG25", illiquid),
            _Contract("CLH25", next_front),
        ],
        config,
    )

    assert continuous.rolls["old_contract"].tolist() == ["CLF25"]
    assert continuous.rolls["new_contract"].tolist() == ["CLH25"]
    assert "CLG25" not in set(continuous.raw["contract"])


def test_cache_loader_reads_each_partition_and_retains_only_selected_fronts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    index = pd.date_range("2025-01-01", periods=5, freq="1D", tz="UTC")
    inputs = {
        "CLZ24": _bars(index[1:2], np.full(1, 90.0), np.asarray([1_000_000])),
        "CLF25": _bars(index, np.full(5, 100.0), np.asarray([100, 90, 80, 20, 10])),
        "CLG25": _bars(index, np.full(5, 110.0), np.asarray([1, 1, 1, 1, 1])),
        "CLH25": _bars(index, np.full(5, 120.0), np.asarray([10, 20, 90, 100, 110])),
    }
    cache = tmp_path / "cache"
    for contract, bars in inputs.items():
        frame = bars.copy()
        frame["symbol"] = contract
        path = cache / "CL" / contract / "part-00000.parquet"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path)

    contracts = _load_contracts(cache, "CL", config)

    assert [contract.symbol for contract in contracts] == ["CLF25", "CLH25"]


def test_single_digit_contract_year_is_inferred_from_first_trade() -> None:
    assert _expiry_symbol_key("SIU1", pd.Timestamp("2021-07-14", tz="UTC"))[0] == 2021
    assert _expiry_symbol_key("SIN0", pd.Timestamp("2026-06-09", tz="UTC"))[0] == 2030


def test_si_loader_rejects_far_dated_recycled_year_as_initial_front(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        symbols=("SI",),
        start=pd.Timestamp("2021-07-14", tz="UTC").to_pydatetime(),
        end=pd.Timestamp("2026-07-14", tz="UTC").to_pydatetime(),
    )
    cache = tmp_path / "cache"
    inputs = {
        "SIU1": _bars(
            pd.date_range("2021-07-14", periods=3, freq="1min", tz="UTC"),
            np.full(3, 26.0),
            np.asarray([100, 100, 100]),
        ),
        "SIN0": _bars(
            pd.DatetimeIndex([pd.Timestamp("2026-06-09 19:02", tz="UTC")]),
            np.asarray([80.0]),
            np.asarray([2]),
        ),
    }
    for contract, bars in inputs.items():
        frame = bars.copy()
        frame.index.name = "datetime"
        frame["symbol"] = contract
        path = cache / "SI" / contract / "part-00000.parquet"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path)

    contracts = _load_contracts(cache, "SI", config)

    assert [contract.symbol for contract in contracts] == ["SIU1"]
    assert contracts[0].bars.index[0] == pd.Timestamp("2021-07-14", tz="UTC")


def test_bad_print_audit_preserves_source_bar_and_classifies_reversion() -> None:
    index = pd.date_range("2025-01-01", periods=32, freq="1min", tz="UTC")
    close = np.full(32, 100.0)
    close[25:27] = 80.0
    reverted, repairs = _classify_isolated_bad_prints(_bars(index, close, np.ones(32)), "CLZ25")
    persistent_close = close.copy()
    persistent_close[25:] = 80.0
    persistent, persistent_repairs = _classify_isolated_bad_prints(
        _bars(index, persistent_close, np.ones(32)), "CLZ25"
    )

    assert index[25] in reverted.index
    assert repairs["timestamp"].tolist() == [index[25]]
    assert index[25] in persistent.index
    assert persistent_repairs.empty


@pytest.mark.parametrize("timeframe", ["3min", "5min", "15min"])
def test_derived_timeframe_is_exact_ohlcv_aggregation(timeframe: str) -> None:
    periods = 30
    index = pd.date_range("2025-01-01", periods=periods, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": np.arange(10, 10 + periods),
            "high": np.arange(11, 11 + periods),
            "low": np.arange(9, 9 + periods),
            "close": np.arange(10.5, 10.5 + periods),
            "volume": np.arange(1, periods + 1),
            "contract": ["ESH25"] * periods,
            "segment": [0] * periods,
            "adjustment_factor": [1.0] * periods,
            "quality_flag": [False] * periods,
        },
        index=index,
    )

    result = _resample(frame, timeframe)
    minutes = int(timeframe.removesuffix("min"))

    assert result.iloc[0]["open"] == 10
    assert result.iloc[0]["high"] == 10 + minutes
    assert result.iloc[0]["low"] == 9
    assert result.iloc[0]["close"] == pytest.approx(9.5 + minutes)
    assert result.iloc[0]["volume"] == sum(range(1, minutes + 1))


def test_persisted_aggregation_is_recomputed_from_one_minute_source(tmp_path: Path) -> None:
    index = pd.date_range("2025-01-01", periods=6, freq="1min", tz="UTC")
    close = np.arange(100.0, 106.0)
    source = _bars(index, close, np.ones(6))
    source["contract"] = "ESH25"
    source["segment"] = 0
    source["adjustment_factor"] = 1.0
    source["quality_flag"] = False
    path = tmp_path / "ES_3min.parquet"
    _resample(source, "3min").reset_index(names="datetime").to_parquet(path, index=False)

    _validate_persisted_aggregation(source, path, "3min", "ES 3min")
    tampered = pd.read_parquet(path)
    tampered.loc[0, "volume"] += 1
    tampered.to_parquet(path, index=False)
    with pytest.raises(CorpusRepairError, match="persisted aggregation values mismatch"):
        _validate_persisted_aggregation(source, path, "3min", "ES 3min")


def test_coverage_gate_rejects_a_truncated_contract_chain(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        end=pd.Timestamp("2025-02-01", tz="UTC").to_pydatetime(),
    )
    frame = _bars(
        pd.date_range("2025-01-01", periods=2, freq="1D", tz="UTC"),
        np.asarray([100.0, 101.0]),
        np.ones(2),
    )

    with pytest.raises(CorpusRepairError, match="ends too early"):
        _validate_coverage(frame, config, "ES")


def test_decode_cache_reuse_is_content_addressed(tmp_path: Path) -> None:
    source = CorpusSource(tmp_path / "source.dbn.zst", "a" * 64, ("ES",))
    cache = tmp_path / "cache"
    frame = _bars(
        pd.date_range("2025-01-01", periods=2, freq="1min", tz="UTC"),
        np.asarray([100.0, 101.0]),
        np.ones(2),
    )
    frame["symbol"] = "ESH25"
    part = cache / "ES" / "ESH25" / "part-00000.parquet"
    part.parent.mkdir(parents=True)
    frame.to_parquet(part)
    manifest = {
        "schema_version": 4,
        "source_sha256": source.sha256,
        "decoder_digest": _decoder_digest(),
        "symbols": ["ES"],
        "parts": [_file_identity(part, 2, cache)],
    }

    assert _cache_is_valid(cache, manifest, source.sha256, source)
    part.write_bytes(part.read_bytes() + b"tamper")
    assert not _cache_is_valid(cache, manifest, source.sha256, source)


def test_decode_rejects_a_source_archive_with_the_wrong_digest(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(CorpusRepairError, match="source archive digest mismatch"):
        _decode_source(config, config.sources[0])


def test_validate_corpus_rejects_tampered_parquet(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    frame = _bars(
        pd.date_range("2025-01-01", periods=2, freq="1min", tz="UTC"),
        np.asarray([100.0, 101.0]),
        np.asarray([1.0, 2.0]),
    )
    identity = {
        "kind": "market",
        **_write_frame(frame, corpus_root / "market" / "ES_1min.parquet", corpus_root),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "corpus_id": "test",
        "outputs": [identity],
        "validated": True,
    }
    manifest["manifest_digest"] = _json_digest(manifest)
    (corpus_root / "manifest.json").write_text(json.dumps(manifest))
    manifest_sha = sha256_file(corpus_root / "manifest.json")

    assert validate_corpus(corpus_root)["rows_checked"] == 2
    assert (
        validate_corpus_binding(
            corpus_root / "market",
            corpus_root / "manifest.json",
            manifest_sha,
            [corpus_root / str(identity["path"])],
        )
        == manifest["manifest_digest"]
    )
    path = corpus_root / str(identity["path"])
    path.write_bytes(path.read_bytes() + b"tamper")
    assert sha256_file(path) != identity["sha256"]
    with pytest.raises(CorpusRepairError, match="size mismatch"):
        validate_corpus(corpus_root)


def test_existing_corpus_id_cannot_be_overwritten(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.output_path.mkdir()

    from mantis_v2.corpus import repair_corpus

    with pytest.raises(CorpusRepairError, match="already exists"):
        repair_corpus(config)
    with pytest.raises(CorpusRepairError, match="overwrite is not supported"):
        repair_corpus(replace(config, allow_overwrite=True))


def _mock_corpus_source(
    monkeypatch: pytest.MonkeyPatch,
    config: CorpusRepairConfig,
    close: np.ndarray | None = None,
) -> None:
    index = pd.date_range("2025-01-01", periods=10, freq="1D", tz="UTC")
    values = np.full(10, 100.0) if close is None else close
    contract = _Contract("ESH25", _bars(index, values, np.ones(10)))
    monkeypatch.setattr(
        corpus_module,
        "_decode_source",
        lambda _config, source: (
            config.output_root / "cache",
            {
                "path": str(source.path),
                "size": source.path.stat().st_size,
                "sha256": source.sha256,
                "symbols": list(source.symbols),
            },
        ),
    )
    monkeypatch.setattr(
        corpus_module,
        "_load_contracts",
        lambda *_args: [contract],
    )


def _mock_roll_gap_source(
    monkeypatch: pytest.MonkeyPatch, config: CorpusRepairConfig
) -> pd.Timestamp:
    index = pd.date_range("2025-01-01", periods=10, freq="1D", tz="UTC")
    old_close = np.asarray([100.0] * 3 + [120.0] * 7)
    new_close = np.asarray([110.0] * 3 + [132.0] * 7)
    old = _Contract(
        "ESF25",
        _bars(index, old_close, np.asarray([100, 40, 30, 20, 10, 10, 10, 10, 10, 10])),
    )
    new = _Contract(
        "ESH25",
        _bars(index, new_close, np.asarray([50, 60, 70, 80, 90, 90, 90, 90, 90, 90])),
    )
    _mock_corpus_source(monkeypatch, config)
    monkeypatch.setattr(corpus_module, "_load_contracts", lambda *_args: [old, new])
    return index[3]


def test_repair_self_audits_before_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _mock_corpus_source(monkeypatch, config)

    manifest = corpus_module.repair_corpus(config)

    assert config.output_path.is_dir()
    assert manifest["validated"] is True
    assert validate_corpus(config.output_path)["valid"] is True


def test_failed_self_audit_never_publishes_final_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _mock_corpus_source(monkeypatch, config)
    monkeypatch.setattr(
        corpus_module,
        "validate_corpus",
        lambda _path: (_ for _ in ()).throw(CorpusRepairError("injected audit failure")),
    )

    with pytest.raises(CorpusRepairError, match="injected audit failure"):
        corpus_module.repair_corpus(config)

    assert not config.output_path.exists()


def test_unclassified_persistent_dislocation_blocks_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    close = np.asarray([100.0] * 5 + [80.0] * 5)
    _mock_corpus_source(monkeypatch, config, close)

    with pytest.raises(CorpusRepairError, match="unclassified same-contract dislocations"):
        corpus_module.repair_corpus(config)

    assert not config.output_path.exists()


def test_unreviewed_market_gap_at_roll_blocks_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _mock_roll_gap_source(monkeypatch, config)

    with pytest.raises(CorpusRepairError, match="unclassified adjusted roll dislocations"):
        corpus_module.repair_corpus(config)

    assert not config.output_path.exists()


def test_exact_reviewed_roll_dislocation_is_recorded_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_time = pd.Timestamp("2025-01-04", tz="UTC")
    reason = "Reviewed market-wide gap coinciding with a valid roll."
    config = replace(
        _config(tmp_path),
        accepted_dislocations=(
            AcceptedDislocation(
                "ES", event_time.to_pydatetime(), "roll", ("ESF25", "ESH25"), reason
            ),
        ),
    )
    assert _mock_roll_gap_source(monkeypatch, config) == event_time

    manifest = corpus_module.repair_corpus(config)

    quality_rows = manifest["quality"]
    assert isinstance(quality_rows, list)
    quality = next(item for item in quality_rows if item["symbol"] == "ES")
    assert quality["roll_dislocations"] == 1
    assert quality["accepted_roll_dislocations"] == [
        {
            "timestamp": event_time.isoformat(),
            "kind": "roll",
            "contracts": ["ESF25", "ESH25"],
            "reason": reason,
        }
    ]
    rolls = pd.read_parquet(config.output_path / "rolls" / "ES.parquet")
    assert pd.to_datetime(rolls.loc[0, "datetime"], utc=True) == event_time
    assert pd.to_datetime(rolls.loc[0, "pricing_timestamp"], utc=True) == event_time
    assert rolls.loc[0, "ratio"] == pytest.approx(1.1)


@pytest.mark.parametrize(
    ("kind", "contracts"),
    [
        ("same_contract", ("ESH25",)),
        ("roll", ("ESG25", "ESH25")),
    ],
)
def test_roll_acceptance_rejects_wrong_event_class_or_contract_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    contracts: tuple[str, ...],
) -> None:
    event_time = pd.Timestamp("2025-01-04", tz="UTC")
    config = replace(
        _config(tmp_path),
        accepted_dislocations=(
            AcceptedDislocation(
                "ES",
                event_time.to_pydatetime(),
                kind,
                contracts,
                "A mismatched review must not be consumed.",
            ),
        ),
    )
    _mock_roll_gap_source(monkeypatch, config)

    with pytest.raises(CorpusRepairError, match="unclassified adjusted roll dislocations"):
        corpus_module.repair_corpus(config)

    assert not config.output_path.exists()


def test_exact_reviewed_dislocation_is_recorded_in_published_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_time = pd.Timestamp("2025-01-06", tz="UTC")
    reason = "Reviewed persistent same-contract market repricing."
    config = replace(
        _config(tmp_path),
        accepted_dislocations=(
            AcceptedDislocation(
                "ES", event_time.to_pydatetime(), "same_contract", ("ESH25",), reason
            ),
        ),
    )
    close = np.asarray([100.0] * 5 + [80.0] * 5)
    _mock_corpus_source(monkeypatch, config, close)

    manifest = corpus_module.repair_corpus(config)

    quality_rows = manifest["quality"]
    assert isinstance(quality_rows, list)
    quality = next(item for item in quality_rows if item["symbol"] == "ES")
    assert quality["accepted_same_contract_dislocations"] == [
        {
            "timestamp": event_time.isoformat(),
            "kind": "same_contract",
            "contracts": ["ESH25"],
            "reason": reason,
        }
    ]


def test_stale_reviewed_dislocation_blocks_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        _config(tmp_path),
        accepted_dislocations=(
            AcceptedDislocation(
                "ES",
                pd.Timestamp("2025-01-07", tz="UTC").to_pydatetime(),
                "same_contract",
                ("ESH25",),
                "Wrong timestamp must not waive the observed event.",
            ),
        ),
    )
    _mock_corpus_source(monkeypatch, config)

    with pytest.raises(
        CorpusRepairError, match="configured accepted dislocations were not observed"
    ):
        corpus_module.repair_corpus(config)

    assert not config.output_path.exists()


def test_classified_round_trip_flags_spike_and_recovery_without_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        _config(tmp_path),
        end=pd.Timestamp("2025-01-02", tz="UTC").to_pydatetime(),
    )
    index = pd.date_range("2025-01-01", periods=32, freq="1min", tz="UTC")
    close = np.full(32, 100.0)
    close[25:27] = 80.0
    bars = _bars(index, close, np.ones(32))
    contract = _Contract("ESH25", bars)
    monkeypatch.setattr(
        corpus_module,
        "_decode_source",
        lambda _config, source: (
            config.output_root / "cache",
            {
                "path": str(source.path),
                "size": source.path.stat().st_size,
                "sha256": source.sha256,
                "symbols": list(source.symbols),
            },
        ),
    )
    monkeypatch.setattr(corpus_module, "_load_contracts", lambda *_args: [contract])

    corpus_module.repair_corpus(config)

    market = pd.read_parquet(config.output_path / "market" / "ES_1min.parquet")
    assert market.loc[25:28, "quality_flag"].tolist() == [True, True, True, True]
