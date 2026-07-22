from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from mantis_v2 import foundation_fixture as fixture_module
from mantis_v2.foundation_fixture import embed_diagnostic_fixture, freeze_diagnostic_fixture

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "trend-magic-topstep-100k.toml"


def _candidates(symbol: str) -> pd.DataFrame:
    fit = pd.date_range("2023-01-01", periods=6, freq="3min", tz="UTC")
    score = pd.date_range("2025-01-01", periods=6, freq="3min", tz="UTC")
    decisions = fit.append(score)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "decision_index": np.arange(12),
            "decision_ts": decisions,
            "label_end_ts": decisions + np.timedelta64(3, "m"),
            "label": np.tile([0, 1], 6),
            "1min_index": np.arange(12) + 200,
            "3min_index": np.arange(12) + 200,
            "5min_index": np.arange(12) + 200,
            "15min_index": np.arange(12) + 200,
            "is_holdout": False,
        }
    )


def test_freezer_builds_one_class_agnostic_content_addressed_four_timeframe_fixture(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[str, str]] = []

    def build(config, symbol, split="pre_holdout"):
        calls.append((symbol, split))
        return _candidates(symbol)

    monkeypatch.setattr(fixture_module, "build_symbol_candidates", build)

    first = freeze_diagnostic_fixture(CONFIG, tmp_path)
    second = freeze_diagnostic_fixture(CONFIG, tmp_path)
    manifest = json.loads(first.read_text())
    rows = pd.read_parquet(first.parent / manifest["rows"]["path"])
    contexts = pd.read_parquet(first.parent / manifest["contexts"]["path"])

    assert first == second
    assert first.parent.name == manifest["fixture_digest"]
    assert manifest["selection"] == {
        "rng": "PCG64",
        "seed": 0,
        "fit_cap": 25000,
        "score_cap": 25000,
    }
    assert manifest["timeframes"] == ["1min", "3min", "5min", "15min"]
    assert rows.columns.tolist() == ["row_id", "decision_at", "label_end_at", "label"]
    assert contexts.columns.tolist() == [
        "row_id",
        "symbol",
        "1min_index",
        "3min_index",
        "5min_index",
        "15min_index",
    ]
    assert rows["row_id"].tolist() == contexts["row_id"].tolist()
    assert not rows["row_id"].duplicated().any()
    assert set(pd.to_datetime(rows["decision_at"], utc=True).dt.year) == {2023, 2025}
    assert len(calls) == 14


def test_embedder_preserves_frozen_row_order_and_publishes_relative_atomic_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        fixture_module,
        "build_symbol_candidates",
        lambda config, symbol, split="pre_holdout": _candidates(symbol),
    )
    fixture = freeze_diagnostic_fixture(CONFIG, tmp_path / "fixture")
    export = tmp_path / "export"
    export.mkdir()
    weights = export / "model.safetensors"
    weights.write_bytes(b"weights")
    foundation = {
        "export_role": "diagnostic_candidate",
        "weights": str(weights),
        "weights_sha256": fixture_module._sha256_file(weights),
        "config": {"run": {"seed": 43}},
    }
    foundation_path = export / "manifest.json"
    foundation_path.write_text(json.dumps(foundation))

    class Loaded:
        embedding_dim = 2
        weights_sha256 = foundation["weights_sha256"]

    monkeypatch.setattr(fixture_module, "load_foundation", lambda config: Loaded())

    def embeddings(candidates, symbol, config, loaded):
        del config, loaded
        values = np.column_stack(
            (
                candidates["1min_index"].to_numpy(dtype=np.float32),
                np.full(len(candidates), float(ord(symbol[0])), dtype=np.float32),
            )
        )
        yield 0, len(candidates), values

    monkeypatch.setattr(fixture_module, "iter_symbol_embeddings", embeddings)
    first = embed_diagnostic_fixture(CONFIG, fixture, foundation_path, tmp_path / "features")
    second = embed_diagnostic_fixture(CONFIG, fixture, foundation_path, tmp_path / "features")
    manifest = json.loads(first.read_text())
    values = np.load(first.parent / manifest["features"]["path"])
    contexts = pd.read_parquet(fixture.parent / "contexts.parquet")

    assert first == second
    assert manifest["seed"] == 43
    assert manifest["features"]["path"] == "features.npy"
    assert values.shape == (len(contexts), 2)
    assert values[:, 0].tolist() == contexts["1min_index"].astype(float).tolist()
