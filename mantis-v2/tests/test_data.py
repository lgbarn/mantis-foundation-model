from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from mantis_v2 import data as data_module
from mantis_v2.config import load_config
from mantis_v2.data import (
    Anchor,
    NextLegDataset,
    Stream,
    alternating_pivots,
    build_anchors,
    split_bounds,
    synthetic_stream,
)

ROOT = Path(__file__).resolve().parents[1]


def test_pivots_alternate_and_confirm_after_origin() -> None:
    stream = synthetic_stream(512)
    origins, confirmations, directions = alternating_pivots(stream.values, k=2)
    assert len(origins) > 10
    np.testing.assert_array_equal(confirmations, origins + 2)
    assert np.all(directions[1:] != directions[:-1])


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
