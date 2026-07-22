from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mantis_v2.foundation_diagnostic import DiagnosticError, score_diagnostic_fixture


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> Path:
    fit_count = 80
    score_count = 60
    rows = pd.DataFrame(
        {
            "row_id": [f"row-{index:03d}" for index in range(fit_count + score_count)],
            "decision_at": [
                *(
                    pd.Timestamp("2023-01-01", tz="UTC")
                    + pd.to_timedelta(np.arange(fit_count), unit="h")
                ),
                *(
                    pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.to_timedelta(np.arange(score_count), unit="h")
                ),
            ],
            "label_end_at": [
                *(
                    pd.Timestamp("2023-01-01 00:03", tz="UTC")
                    + pd.to_timedelta(np.arange(fit_count), unit="h")
                ),
                *(
                    pd.Timestamp("2025-01-01 00:03", tz="UTC")
                    + pd.to_timedelta(np.arange(score_count), unit="h")
                ),
            ],
            "label": np.tile([0, 1], (fit_count + score_count) // 2),
        }
    )
    rows_path = tmp_path / "rows.parquet"
    rows.to_parquet(rows_path, index=False)
    row_ids_digest = hashlib.sha256(
        json.dumps(rows["row_id"].tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    core = {
        "schema_version": 1,
        "label_semantics": "fixed_3r_trend_magic",
        "holdout_start": "2026-01-01T00:00:00+00:00",
        "fit_before": "2024-01-01T00:00:00+00:00",
        "score_start": "2025-01-01T00:00:00+00:00",
        "score_end": "2026-01-01T00:00:00+00:00",
        "selection": {"rng": "PCG64", "seed": 0, "fit_cap": 25000, "score_cap": 25000},
        "rows": {"path": str(rows_path), "sha256": _sha(rows_path), "count": len(rows)},
        "row_ids_sha256": row_ids_digest,
    }
    manifest = {
        **core,
        "fixture_digest": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    return path


def _write_features(tmp_path: Path, fixture: Path, name: str, values: np.ndarray) -> Path:
    fixture_data = json.loads(fixture.read_text())
    feature_path = tmp_path / f"{name}.npy"
    np.save(feature_path, values.astype(np.float32), allow_pickle=False)
    core = {
        "schema_version": 1,
        "fixture_digest": fixture_data["fixture_digest"],
        "row_ids_sha256": fixture_data["row_ids_sha256"],
        "seed": 42,
        "candidate_id": name,
        "features": {
            "path": str(feature_path),
            "sha256": _sha(feature_path),
            "rows": values.shape[0],
            "width": values.shape[1],
        },
    }
    manifest = {
        **core,
        "feature_digest": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    return path


def test_scorer_uses_exact_registered_probe_and_reports_full_fixture_diagnostics(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    labels = np.tile([0.0, 1.0], 70)
    candidate = np.column_stack((labels * 2 - 1, np.arange(140) / 140))
    reference = np.column_stack((np.zeros(140), np.arange(140) / 140))
    candidate_manifest = _write_features(tmp_path, fixture, "candidate", candidate)
    reference_manifest = _write_features(tmp_path, fixture, "reference", reference)

    result_path = score_diagnostic_fixture(
        fixture, candidate_manifest, reference_manifest, tmp_path / "score.json"
    )
    result = json.loads(result_path.read_text())

    assert result["probe"] == {
        "scaler": "StandardScaler",
        "classifier": "LogisticRegression",
        "C": 0.0001,
        "solver": "lbfgs",
        "max_iter": 1000,
        "tol": 0.0001,
        "class_weight": "balanced",
        "random_state": 0,
        "threshold": None,
        "calibrator": None,
    }
    assert result["candidate"]["log_loss"] < result["reference"]["log_loss"]
    assert result["candidate"]["brier"] < result["reference"]["brier"]
    assert len(result["candidate"]["calibration_bins"]) == 10
    assert sum(item["count"] for item in result["candidate"]["calibration_bins"]) == 60
    assert result["score_rows"] == 60


def test_scorer_resolves_frozen_artifacts_relative_to_each_manifest(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    values = np.column_stack((np.tile([0.0, 1.0], 70), np.arange(140) / 140))
    candidate = _write_features(tmp_path, fixture, "candidate-relative", values)
    reference = _write_features(tmp_path, fixture, "reference-relative", values)
    fixture_data = json.loads(fixture.read_text())
    fixture_data["rows"]["path"] = Path(fixture_data["rows"]["path"]).name
    core = {key: value for key, value in fixture_data.items() if key != "fixture_digest"}
    fixture_data["fixture_digest"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    fixture.write_text(json.dumps(fixture_data, sort_keys=True))
    for path in (candidate, reference):
        record = json.loads(path.read_text())
        record["fixture_digest"] = fixture_data["fixture_digest"]
        record["features"]["path"] = Path(record["features"]["path"]).name
        core = {key: value for key, value in record.items() if key != "feature_digest"}
        record["feature_digest"] = hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        path.write_text(json.dumps(record, sort_keys=True))

    result = score_diagnostic_fixture(
        fixture, candidate, reference, tmp_path / "relative-score.json"
    )

    assert result.is_file()


def test_scorer_rejects_row_identity_nonfinite_missing_class_and_2024_rows(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    values = np.arange(280, dtype=np.float32).reshape(140, 2)
    candidate = _write_features(tmp_path, fixture, "candidate", values)
    reference = _write_features(tmp_path, fixture, "reference", values)

    candidate_data = json.loads(candidate.read_text())
    candidate_data["row_ids_sha256"] = "0" * 64
    core = {key: value for key, value in candidate_data.items() if key != "feature_digest"}
    candidate_data["feature_digest"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    candidate.write_text(json.dumps(candidate_data))
    with pytest.raises(DiagnosticError, match="row identity"):
        score_diagnostic_fixture(fixture, candidate, reference, tmp_path / "bad-row.json")

    candidate_data["row_ids_sha256"] = json.loads(fixture.read_text())["row_ids_sha256"]
    core = {key: value for key, value in candidate_data.items() if key != "feature_digest"}
    candidate_data["feature_digest"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    candidate.write_text(json.dumps(candidate_data))
    array = np.load(candidate_data["features"]["path"])
    array[0, 0] = np.nan
    np.save(candidate_data["features"]["path"], array, allow_pickle=False)
    candidate_data["features"]["sha256"] = _sha(Path(candidate_data["features"]["path"]))
    core = {key: value for key, value in candidate_data.items() if key != "feature_digest"}
    candidate_data["feature_digest"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    candidate.write_text(json.dumps(candidate_data))
    with pytest.raises(DiagnosticError, match="non-finite"):
        score_diagnostic_fixture(fixture, candidate, reference, tmp_path / "bad-finite.json")

    fixture_data = json.loads(fixture.read_text())
    rows_path = Path(fixture_data["rows"]["path"])
    rows = pd.read_parquet(rows_path)
    rows.loc[0, "decision_at"] = pd.Timestamp("2024-06-01", tz="UTC")
    rows.to_parquet(rows_path, index=False)
    fixture_data["rows"]["sha256"] = _sha(rows_path)
    core = {key: value for key, value in fixture_data.items() if key != "fixture_digest"}
    fixture_data["fixture_digest"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    fixture.write_text(json.dumps(fixture_data))
    with pytest.raises(DiagnosticError, match="2024"):
        score_diagnostic_fixture(fixture, candidate, reference, tmp_path / "bad-2024.json")
