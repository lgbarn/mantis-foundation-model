from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mantis_v2 import cli
from mantis_v2.downstream_config import load_downstream_config
from mantis_v2.rl_config import load_rl_config
from mantis_v2.rl_episodes import (
    EpisodeContractError,
    Partition,
    _atomic_resume,
    _independent_exchange_calendar,
    _load_embedding_metadata_only,
    _session_id,
    _session_is_complete,
    _session_ordinal,
    _verified_path,
    build_chronological_episode_schedule,
    build_episode_schedule,
    build_overlapping_episode_coverage,
    read_observation,
)

ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_schedule_greedily_keeps_chronological_nonoverlapping_attempts() -> None:
    sources = {symbol: _rows(symbol, "2025-01-02", sessions=65) for symbol in ("NQ", "ZB")}
    partition = Partition(
        name="test",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-05-01", tz="UTC"),
    )

    episodes = build_chronological_episode_schedule(sources, partition, trading_days=20)
    overlap_coverage = build_overlapping_episode_coverage(sources, partition, trading_days=20)

    assert episodes == build_chronological_episode_schedule(sources, partition, trading_days=20)
    assert overlap_coverage == build_overlapping_episode_coverage(
        sources, partition, trading_days=20
    )
    assert {
        (episode.ticker, episode.profile, episode.decision_start, episode.terminal_end)
        for episode in episodes
    }.isdisjoint(
        {
            (episode.ticker, episode.profile, episode.decision_start, episode.terminal_end)
            for episode in overlap_coverage
        }
    )
    for episode in overlap_coverage:
        assert any(
            episode.ticker == headline.ticker
            and episode.profile == headline.profile
            and episode.decision_start <= headline.terminal_end
            and headline.decision_start <= episode.terminal_end
            for headline in episodes
        )
    assert {episode.profile for episode in episodes if episode.ticker == "ZB"} == {"one_mini"}
    for ticker in ("NQ", "ZB"):
        for profile in ("one_mini", "ten_micros"):
            selected = [
                episode
                for episode in episodes
                if episode.ticker == ticker and episode.profile == profile
            ]
            if ticker == "ZB" and profile == "ten_micros":
                assert selected == []
                continue
            assert selected
            assert all(
                previous.terminal_end < current.decision_start
                for previous, current in pairwise(selected)
            )


def test_evaluation_scheduler_reads_metadata_columns_without_opening_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    downstream = load_downstream_config(ROOT / "configs" / "downstream-trend-magic-smoke.toml")
    metadata_path = tmp_path / "metadata.parquet"
    feature_path = tmp_path / "features.npy"
    context = downstream.data.context_bars
    first = pd.Timestamp("2025-01-06T18:00:00Z")
    metadata = pd.DataFrame(
        {
            "symbol": ["NQ"],
            "decision_ts": [first],
            **{f"{timeframe}_index": [context] for timeframe in downstream.data.timeframes},
            "label": [1],
            "future_return": [999.0],
        }
    )
    metadata.to_parquet(metadata_path, index=False)
    feature_path.write_bytes(b"must-not-open")
    metadata_bytes = metadata_path.read_bytes()
    manifest_path = tmp_path / "embedding.json"
    manifest_path.write_text(
        json.dumps(
            {
                "feature_width": 2,
                "rows": 1,
                "outputs": [
                    {
                        "number": 0,
                        "rows": 1,
                        "features": {
                            "path": str(feature_path),
                            "sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
                            "size": feature_path.stat().st_size,
                        },
                        "metadata": {
                            "path": str(metadata_path),
                            "sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                            "size": len(metadata_bytes),
                        },
                    }
                ],
            }
        )
    )
    config = replace(
        config,
        upstream=replace(config.upstream, embedding_manifest_path=manifest_path),
    )
    timestamps = {
        ("NQ", timeframe): pd.date_range(
            "2025-01-01", periods=context + 200, freq=timeframe, tz="UTC"
        ).to_numpy()
        for timeframe in downstream.data.timeframes
    }
    columns_seen: list[str] = []
    original_read = pd.read_parquet

    def guarded_read(path: Path, *, columns: list[str]):
        columns_seen.extend(columns)
        return original_read(path, columns=columns)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    monkeypatch.setattr(
        np,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("feature shard opened by metadata scheduler")
        ),
    )
    session = _session_id(first)

    loaded, outputs, width = _load_embedding_metadata_only(
        config,
        timestamps,
        downstream,
        tmp_path,
        {"NQ": set()},
        {"NQ": {session: _session_ordinal(session)}},
        {"NQ": {session}},
    )

    assert width == 2
    assert len(loaded) == 1
    assert outputs[0]["features"]["path"] == str(feature_path)
    assert "label" not in columns_seen
    assert "future_return" not in columns_seen


def _rows(symbol: str, first: str, sessions: int = 22) -> pd.DataFrame:
    days = pd.bdate_range(first, periods=sessions, tz="UTC")
    records: list[dict[str, object]] = []
    for index, day in enumerate(days):
        decision = day + timedelta(hours=18)
        records.append(
            {
                "symbol": symbol,
                "decision_ts": decision,
                "label_end_ts": decision + timedelta(minutes=6),
                "lookback_start_ts": decision - timedelta(hours=1),
                "terminal_ts": decision + timedelta(minutes=9),
                "session_complete": True,
                "session_ordinal": _session_ordinal(_session_id(pd.Timestamp(decision))),
                "rollover_safe": True,
                "horizon_complete": True,
                "shard": index // 8,
                "row": index % 8,
            }
        )
    return pd.DataFrame.from_records(records)


def test_schedule_is_seeded_uniform_and_keeps_zb_mini_only() -> None:
    sources = {symbol: _rows(symbol, "2025-01-02") for symbol in ("ES", "NQ", "ZB")}
    partition = Partition(
        name="training",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-03-01", tz="UTC"),
    )

    first = build_episode_schedule(sources, partition, seed=17, episode_count=600)
    second = build_episode_schedule(sources, partition, seed=17, episode_count=600)

    assert first == second
    ticker_counts = pd.Series([episode.ticker for episode in first]).value_counts()
    assert ticker_counts.to_dict() == {"ES": 200, "NQ": 200, "ZB": 200}
    for ticker in ("ES", "NQ"):
        profiles = [episode.profile for episode in first if episode.ticker == ticker]
        assert profiles.count("one_mini") == profiles.count("ten_micros") == 100
    assert {episode.profile for episode in first if episode.ticker == "ZB"} == {"one_mini"}
    assert {episode.trading_days for episode in first} == {20}
    assert all(isinstance(episode.start_session, date) for episode in first)


def test_schedule_rejects_rows_stitched_from_another_instrument() -> None:
    rows = _rows("ES", "2025-01-02")
    rows.loc[10, "symbol"] = "NQ"
    partition = Partition(
        name="training",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-03-01", tz="UTC"),
    )

    with pytest.raises(EpisodeContractError, match="owning stream"):
        build_episode_schedule({"ES": rows}, partition, seed=5, episode_count=1)


def test_cli_exposes_bounded_episode_manifest_builder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = {"stage": "rl-episode-schedule"}
    sentinel = object()
    monkeypatch.setattr(cli, "load_rl_config", lambda _path: sentinel)
    monkeypatch.setattr(cli, "build_episode_manifest", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mantis-v2",
            "rl-build-episodes",
            "--config",
            "rl.toml",
            "--fold",
            "2",
            "--partition",
            "training",
            "--episodes",
            "24",
        ],
    )

    cli.main()

    assert json.loads(capsys.readouterr().out) == expected


def test_schedule_rejects_missing_session_before_sampling() -> None:
    rows = _rows("ES", "2025-01-02")
    rows.loc[8, "session_complete"] = False
    partition = Partition(
        name="training",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-03-01", tz="UTC"),
    )

    with pytest.raises(EpisodeContractError, match="no complete episode"):
        build_episode_schedule({"ES": rows}, partition, seed=5, episode_count=1)


def test_complete_session_calendar_allows_a_day_without_candidates() -> None:
    complete = _rows("ES", "2025-01-02", sessions=20)
    sessions = tuple(
        dict.fromkeys(_session_id(pd.Timestamp(value)) for value in complete["decision_ts"])
    )
    candidates = complete.drop(index=10).reset_index(drop=True)
    partition = Partition(
        name="training",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-03-01", tz="UTC"),
    )

    (episode,) = build_episode_schedule(
        {"ES": candidates},
        partition,
        seed=5,
        episode_count=1,
        session_calendars={"ES": sessions},
    )

    assert episode.trading_days == 20
    assert episode.candidate_count == 19
    assert sum(span.row_count for span in episode.observation_spans) == 19


def test_schedule_does_not_bridge_absent_or_rollover_sessions() -> None:
    partition = Partition(
        name="training",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-06-01", tz="UTC"),
    )
    absent = _rows("ES", "2025-01-02").drop(index=10).reset_index(drop=True)
    rollover = _rows("ES", "2025-01-02")
    rollover["rollover_safe"] = False

    with pytest.raises(EpisodeContractError, match="no complete episode"):
        build_episode_schedule({"ES": absent}, partition, seed=5, episode_count=1)
    with pytest.raises(EpisodeContractError, match="no complete episode"):
        build_episode_schedule({"ES": rollover}, partition, seed=5, episode_count=1)


def test_session_calendar_detects_whole_and_partial_session_gaps() -> None:
    monday = date(2025, 1, 5)
    wednesday = date(2025, 1, 7)
    assert _session_ordinal(wednesday) - _session_ordinal(monday) == 2

    complete = pd.Series(
        pd.date_range(
            "2025-01-05T17:00:00-06:00",
            "2025-01-06T15:09:00-06:00",
            freq="3min",
        )
    )
    assert _session_is_complete(complete)
    assert _session_is_complete(complete.drop(index=100))
    assert not _session_is_complete(complete.iloc[[0, -1]])
    assert not _session_is_complete(complete.iloc[:2])


def test_exchange_calendar_rejects_ticker_local_missing_or_rollover_session() -> None:
    complete = _rows("ES", "2025-01-02", sessions=20)
    sessions = tuple(
        dict.fromkeys(_session_id(pd.Timestamp(value)) for value in complete["decision_ts"])
    )
    partition = Partition(
        name="training",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-03-01", tz="UTC"),
    )

    with pytest.raises(EpisodeContractError, match="no complete episode"):
        build_episode_schedule(
            {"ES": complete},
            partition,
            seed=5,
            episode_count=1,
            session_calendars={"ES": sessions},
            unsafe_sessions={"ES": frozenset({sessions[10]})},
        )


def test_independent_calendar_keeps_provider_wide_missing_session_visible() -> None:
    sessions = _independent_exchange_calendar(
        pd.Timestamp("2025-01-06", tz="UTC"), pd.Timestamp("2025-01-10", tz="UTC")
    )

    assert len(sessions) == 5
    assert date(2025, 1, 7) in sessions


def test_schedule_excludes_candidate_with_rollover_overlap_from_safe_window() -> None:
    rows = _rows("CL", "2025-01-02", sessions=20)
    rows.loc[10, "rollover_safe"] = False
    sessions = tuple(
        dict.fromkeys(_session_id(pd.Timestamp(value)) for value in rows["decision_ts"])
    )
    partition = Partition(
        name="training",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-03-01", tz="UTC"),
    )

    (episode,) = build_episode_schedule(
        {"CL": rows},
        partition,
        seed=5,
        episode_count=1,
        session_calendars={"CL": sessions},
        unsafe_sessions={"CL": frozenset()},
    )

    assert episode.candidate_count == 19
    assert sum(span.row_count for span in episode.observation_spans) == 19


def test_schedule_contains_complete_lookback_exit_and_terminal_horizons() -> None:
    rows = _rows("ES", "2025-01-02")
    rows.loc[0, "lookback_start_ts"] = pd.Timestamp("2024-12-31T23:00:00Z")
    rows.loc[21, "terminal_ts"] = pd.Timestamp("2025-03-01T00:00:00Z")
    partition = Partition(
        name="training",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-03-01", tz="UTC"),
    )

    (episode,) = build_episode_schedule({"ES": rows}, partition, seed=9, episode_count=1)

    assert episode.lookback_start >= partition.start
    assert episode.decision_start >= partition.start
    assert episode.exit_end < partition.end
    assert episode.terminal_end < partition.end


def test_embedding_identity_and_mmap_lookup_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "features.npy"
    np.save(path, np.arange(12, dtype=np.float16).reshape(3, 4), allow_pickle=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {"path": str(path), "size": path.stat().st_size, "sha256": digest}

    assert read_observation(path, 1, 4).tolist() == [4.0, 5.0, 6.0, 7.0]
    path.write_bytes(path.read_bytes() + b"modified")

    with pytest.raises(EpisodeContractError, match="content identity mismatch"):
        _verified_path(identity, tmp_path, "features shard 0")


def test_atomic_schedule_resume_preserves_completed_output(tmp_path: Path) -> None:
    path = tmp_path / "episodes" / "schedule.json"
    payload: dict[str, object] = {"schema_version": 1, "episodes": [{"ticker": "ES"}]}

    _atomic_resume(path, payload)
    original = path.read_bytes()
    _atomic_resume(path, payload)

    assert path.read_bytes() == original
    with pytest.raises(EpisodeContractError, match="already exists"):
        _atomic_resume(path, {"schema_version": 1, "episodes": [{"ticker": "NQ"}]})
    assert path.read_bytes() == original
