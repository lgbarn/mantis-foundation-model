from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mantis_v2.frozen_expected_r import (
    FrozenExpectedRConfig,
    FrozenExpectedRError,
    FrozenMantisEmbedder,
    compare_frozen_to_raw,
    write_frozen_input,
    write_paid_preflight,
)


def _candidates(rows: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02T15:00:00Z", periods=rows, freq="7D")
    return pd.DataFrame(
        {
            "row_id": [f"row-{index:04d}" for index in range(rows)],
            "decision_ts": dates,
            "outcome_ts": dates + pd.Timedelta(minutes=30),
            "entry_index": np.arange(rows) * 2,
            "outcome_index": np.arange(rows) * 2 + 1,
            "average_uniqueness": np.ones(rows),
            "net_r": np.tile([-1.0, 1.5, 2.0], rows // 3),
        }
    )


def test_contract_pins_official_layer_pooling_and_rejects_adaptation() -> None:
    config = FrozenExpectedRConfig()
    assert config.source_revision == "0c94f8ceb9f1d1421dd292ed917090df8c31605b"
    assert config.hub_revision == "99fe0f548960e272fbfa4b82fd9b5b5956779dfd"
    assert config.return_transf_layer == 2
    assert config.output_token == "combined"
    assert config.preprocessing == "native_mantis_v2"

    with pytest.raises(FrozenExpectedRError, match="official frozen"):
        FrozenExpectedRConfig(checkpoint_kind="adapted")


def test_input_is_content_bound_and_rejects_modified_rows(tmp_path: Path) -> None:
    candidates = _candidates(6)
    candidates.attrs["raw_features"] = np.ones((6, 2563), dtype=np.float32)
    windows = np.ones((6, 5, 512), dtype=np.float32)
    manifest = write_frozen_input(candidates, windows, np.zeros((6, 3)), tmp_path / "input")

    changed = pd.read_parquet(tmp_path / "input" / "candidates.parquet")
    changed.loc[0, "row_id"] = "changed"
    changed.to_parquet(tmp_path / "input" / "candidates.parquet", index=False)

    with pytest.raises(FrozenExpectedRError, match="candidate digest"):
        FrozenMantisEmbedder(FrozenExpectedRConfig()).validate_input(manifest)


def test_atomic_embedding_resume_skips_complete_shards(tmp_path: Path) -> None:
    candidates = _candidates(6)
    windows = np.arange(6 * 5 * 512, dtype=np.float32).reshape(6, 5, 512)
    manifest = write_frozen_input(candidates, windows, np.zeros((6, 3)), tmp_path / "input")

    class FakeModel:
        calls = 0

        def __call__(self, values: np.ndarray, _precision: str) -> np.ndarray:
            FakeModel.calls += len(values)
            return np.repeat(values.mean(axis=2), 512, axis=1)

    embedder = FrozenMantisEmbedder(
        FrozenExpectedRConfig(shard_rows=2), model_factory=lambda _config: FakeModel()
    )
    embedder.embed(manifest, tmp_path / "embed", maximum_shards=1)
    assert FakeModel.calls == 6  # FP32/BF16 fixture plus one two-row shard.

    result = embedder.embed(manifest, tmp_path / "embed")
    assert FakeModel.calls == 14  # Parity reruns; only the remaining four rows are embedded.
    assert result["rows"] == 6
    assert len(result["shards"]) == 3


def test_input_manifest_remains_valid_after_directory_move(tmp_path: Path) -> None:
    candidates = _candidates(6)
    manifest = write_frozen_input(
        candidates,
        np.ones((6, 5, 512), dtype=np.float32),
        np.zeros((6, 3), dtype=np.float32),
        tmp_path / "source",
    )
    shutil.move(tmp_path / "source", tmp_path / "remote")

    _, moved_candidates, _, _ = FrozenMantisEmbedder(
        FrozenExpectedRConfig()
    ).validate_input(tmp_path / "remote" / manifest.name)

    assert moved_candidates["row_id"].tolist() == candidates["row_id"].tolist()


def test_comparison_rejects_mismatched_rows() -> None:
    candidates = _candidates()
    raw = np.ones((len(candidates), 3), dtype=np.float32)
    mantis = np.ones((len(candidates) - 1, 3), dtype=np.float32)

    with pytest.raises(FrozenExpectedRError, match="identical candidate rows"):
        compare_frozen_to_raw(candidates, raw, mantis, FrozenExpectedRConfig())


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"unknown": True}))
    with pytest.raises(FrozenExpectedRError, match="unknown config"):
        FrozenExpectedRConfig.from_json(path)


def test_paid_preflight_enforces_budget_deadline_health_and_four_checks(tmp_path: Path) -> None:
    candidates = _candidates(6)
    manifest = write_frozen_input(
        candidates,
        np.ones((6, 5, 512), dtype=np.float32),
        np.zeros((6, 3), dtype=np.float32),
        tmp_path / "input",
    )
    checks = {
        "causality_next_fill": True,
        "label_replay_parity": True,
        "topstep_accounting": True,
        "artifact_resume": True,
    }
    receipt = write_paid_preflight(
        manifest,
        tmp_path / "embed",
        tmp_path / "preflight.json",
        exact_command="uv run mantis-v2 frozen-screen-embed --input input/manifest.json",
        hourly_rate_usd=0.99,
        budget_usd=10.0,
        deadline_hours=6.0,
        check_duration_seconds=4.0,
        checks=checks,
    )
    assert receipt["ready"] is True
    assert receipt["health_interval_seconds"] == 30

    with pytest.raises(FrozenExpectedRError, match="rate-derived"):
        write_paid_preflight(
            manifest,
            tmp_path / "other-embed",
            tmp_path / "other.json",
            exact_command="embed",
            hourly_rate_usd=2.0,
            budget_usd=10.0,
            deadline_hours=6.0,
            check_duration_seconds=4.0,
            checks=checks,
        )
