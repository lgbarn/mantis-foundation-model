from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from mantis_v2 import data as data_module
from mantis_v2.config import load_config
from mantis_v2.contamination import detect_discontinuities
from mantis_v2.corpus import _json_digest, _write_frame
from mantis_v2.data import (
    Anchor,
    NextLegDataset,
    Stream,
    alternating_pivots,
    build_anchors,
    load_streams,
    split_bounds,
    synthetic_stream,
)
from mantis_v2.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_pivots_alternate_and_confirm_after_origin() -> None:
    stream = synthetic_stream(512)
    origins, confirmations, directions = alternating_pivots(stream.values, k=2)
    assert len(origins) > 10
    np.testing.assert_array_equal(confirmations, origins + 2)
    assert np.all(directions[1:] != directions[:-1])


def test_discontinuity_detector_marks_the_new_segment_boundary() -> None:
    timestamps = np.datetime64("2025-01-01T00:00:00", "ns") + np.arange(4).astype("timedelta64[m]")
    boundaries, events = detect_discontinuities(
        timestamps,
        np.asarray([100.0, 100.5, 80.0, 100.0]),
        0.05,
    )
    assert np.flatnonzero(boundaries).tolist() == [2, 3]
    assert [event.row for event in events] == [2, 3]


def test_discontinuity_detector_marks_an_intrabar_stitch_that_closes_normally() -> None:
    timestamps = np.datetime64("2025-01-01T00:00:00", "ns") + np.arange(3).astype("timedelta64[m]")
    high_low_close = np.asarray(
        [
            [100.5, 99.5, 100.0],
            [100.5, 80.0, 100.0],
            [100.5, 99.5, 100.0],
        ]
    )

    boundaries, events = detect_discontinuities(timestamps, high_low_close, 0.05)

    assert np.flatnonzero(boundaries).tolist() == [1]
    assert events[0].relative_jump == pytest.approx(0.20)


def test_foundation_loader_detects_intrabar_stitches_from_full_ohlc(tmp_path: Path) -> None:
    base = load_config(ROOT / "configs" / "smoke.toml")
    config = replace(
        base.data,
        root=str(tmp_path),
        max_relative_close_jump=0.05,
    )
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=3, freq="1min", tz="UTC"),
            "open": [100.0, 100.0, 100.0],
            "high": [100.5, 100.5, 100.5],
            "low": [99.5, 80.0, 99.5],
            "close": [100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0],
        }
    ).to_csv(tmp_path / "SYNTH_1min.csv", index=False)

    stream = load_streams(config)[0]

    assert np.flatnonzero(stream.discontinuities).tolist() == [1]


def test_foundation_loader_reads_manifest_bound_parquet(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    base = load_config(ROOT / "configs" / "smoke.toml")
    corpus_root = tmp_path / "corpus"
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [100.5, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
            "close": [100.0, 101.0, 102.0],
            "volume": [1.0, 2.0, 3.0],
            "quality_flag": [False, True, False],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="1min", tz="UTC"),
    )
    identity = {
        "kind": "market",
        "symbol": "SYNTH",
        "timeframe": "1min",
        **_write_frame(frame, corpus_root / "market" / "SYNTH_1min.parquet", corpus_root),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "corpus_id": "test",
        "outputs": [identity],
        "validated": True,
    }
    manifest["manifest_digest"] = _json_digest(manifest)
    manifest_path = corpus_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = replace(
        base.data,
        root=str(corpus_root / "market"),
        file_format="parquet",
        corpus_manifest_path=manifest_path,
        corpus_manifest_sha256=sha256_file(manifest_path),
    )

    stream = load_streams(config)[0]

    assert stream.name == "SYNTH_1min"
    assert stream.values[:, 3].tolist() == [100.0, 101.0, 102.0]
    assert np.flatnonzero(stream.discontinuities).tolist() == [1]


def test_anchors_never_cross_a_discontinuous_boundary() -> None:
    config = load_config(ROOT / "configs" / "smoke.toml")
    original = synthetic_stream()
    boundaries = np.zeros(len(original.values), dtype=bool)
    boundary = len(boundaries) // 2
    boundaries[boundary] = True
    stream = replace(original, discontinuities=boundaries)

    anchors = build_anchors([stream], config.data, config.target, "train")

    assert anchors
    for anchor in anchors:
        window_start = anchor.confirmation - max(config.data.context_lengths) + 1
        window_end = max(
            anchor.confirmation + max(config.target.horizons),
            anchor.confirmation + anchor.first_leg + anchor.second_leg,
        )
        assert not window_start <= boundary <= window_end


def test_anchors_keep_context_and_both_legs_inside_split() -> None:
    config = load_config(ROOT / "configs" / "smoke.toml")
    stream = synthetic_stream()
    for split in ("train", "validation"):
        start, end = split_bounds(stream, config.data, split)
        anchors = build_anchors([stream], config.data, config.target, split)
        assert anchors
        origins, confirmations, _ = alternating_pivots(stream.values, config.target.leg_k)
        confirmation_to_index = {int(value): index for index, value in enumerate(confirmations)}
        for anchor in anchors:
            pivot_index = confirmation_to_index[anchor.confirmation]
            assert anchor.confirmation - max(config.data.context_lengths) + 1 >= start
            assert origins[pivot_index + 2] < end
            assert anchor.confirmation + max(config.target.horizons) < end


def test_context_normalization_is_causal_and_has_fixed_shape() -> None:
    config = load_config(ROOT / "configs" / "smoke.toml")
    stream = synthetic_stream()
    anchors = build_anchors([stream], config.data, config.target, "train")
    dataset = NextLegDataset([stream], anchors, config.data, config.model, config.target)
    item = dataset[0]
    assert item["context"].shape == (5, 512)
    assert item["candle_target"].shape == (5, 1)
    assert item["leg_target"].shape == (2,)
    assert torch.isfinite(item["context"]).all()

    modified = stream.values.copy()
    modified[anchors[0].confirmation + 1 :] += 1_000_000.0
    changed_stream = replace(stream, values=modified)
    changed = NextLegDataset([changed_stream], anchors, config.data, config.model, config.target)[0]
    torch.testing.assert_close(changed["context"], item["context"])
    assert not torch.allclose(changed["candle_target"], item["candle_target"])


def test_future_target_is_clamped_when_context_variance_is_near_zero() -> None:
    config = load_config(ROOT / "configs" / "smoke.toml")
    rows = 80
    values = np.tile(
        np.asarray([107.625, 107.625, 107.625, 107.625, 1_000.0], dtype=np.float32),
        (rows, 1),
    )
    confirmation = 63
    values[confirmation + 5, 2] = np.float32(107.640625)
    stream = Stream(
        name="ZN_1min",
        timestamps=np.datetime64("2024-04-29T00:00:00", "ns")
        + np.arange(rows).astype("timedelta64[m]"),
        values=values,
    )
    anchor = Anchor(stream_index=0, confirmation=confirmation, first_leg=4, second_leg=5)

    item = NextLegDataset([stream], [anchor], config.data, config.model, config.target)[0]

    assert item["candle_target"].abs().max().item() <= 2 * config.target.normalization_clamp
    assert item["candle_target"][2, 0].item() == config.target.normalization_clamp


def test_anchor_detection_runs_independently_per_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "smoke.toml")
    streams = [synthetic_stream(), synthetic_stream()]
    calls: list[int] = []
    original = data_module.alternating_pivots

    def recording_pivots(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calls.append(id(values))
        return original(values, k)

    monkeypatch.setattr(data_module, "alternating_pivots", recording_pivots)
    anchors = build_anchors(streams, config.data, config.target, "train")
    assert calls == [id(stream.values) for stream in streams]
    assert {anchor.stream_index for anchor in anchors} == {0, 1}
