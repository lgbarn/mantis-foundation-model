from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from mantis_v2 import frozen_expected_r
from mantis_v2.cli import _parser
from mantis_v2.frozen_expected_r import (
    FrozenExpectedRConfig,
    FrozenExpectedRError,
    FrozenMantisEmbedder,
    compare_frozen_to_raw,
    cuda_threshold,
    load_frozen_embeddings,
    run_paid_frozen_screen,
    validate_paid_runner_contract,
    write_frozen_input,
    write_paid_planning_inputs,
    write_paid_preflight,
)


def _candidates(rows: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02T15:00:00Z", periods=rows, freq="7D")
    return pd.DataFrame(
        {
            "row_id": [f"row-{index:04d}" for index in range(rows)],
            "feature_start_ts": dates - pd.Timedelta(minutes=511 * 3),
            "decision_ts": dates,
            "outcome_ts": dates + pd.Timedelta(minutes=30),
            "entry_index": np.arange(rows) * 2,
            "feature_start_index": np.arange(rows) * 2 - 512,
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


def test_partial_resume_rejects_changed_selected_precision(tmp_path: Path) -> None:
    candidates = _candidates(6)
    manifest = write_frozen_input(
        candidates,
        np.ones((6, 5, 512), dtype=np.float32),
        np.zeros((6, 3), dtype=np.float32),
        tmp_path / "input",
    )

    class PassingModel:
        def __call__(self, values: np.ndarray, _precision: str) -> np.ndarray:
            return np.ones((len(values), 2560), dtype=np.float32)

    class FailingModel:
        def __call__(self, values: np.ndarray, precision: str) -> np.ndarray:
            factor = 1.0 if precision == "fp32" else 2.0
            return np.full((len(values), 2560), factor, dtype=np.float32)

    config = FrozenExpectedRConfig(shard_rows=2)
    FrozenMantisEmbedder(config, model_factory=lambda _config: PassingModel()).embed(
        manifest, tmp_path / "embed", maximum_shards=1
    )
    with pytest.raises(FrozenExpectedRError, match="stale embedding shard receipt"):
        FrozenMantisEmbedder(config, model_factory=lambda _config: FailingModel()).embed(
            manifest, tmp_path / "embed"
        )


def test_completed_embedding_rejects_changed_config_and_modified_shard(tmp_path: Path) -> None:
    candidates = _candidates(3)
    manifest = write_frozen_input(
        candidates,
        np.ones((3, 5, 512), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        tmp_path / "input",
    )

    class Model:
        def __call__(self, values: np.ndarray, _precision: str) -> np.ndarray:
            return np.ones((len(values), 2560), dtype=np.float32)

    output = tmp_path / "embed"
    FrozenMantisEmbedder(
        FrozenExpectedRConfig(requested_precision="fp32"),
        model_factory=lambda _config: Model(),
    ).embed(manifest, output)
    with pytest.raises(FrozenExpectedRError, match="stale embedding manifest"):
        FrozenMantisEmbedder(
            FrozenExpectedRConfig(requested_precision="fp32", batch_size=1),
            model_factory=lambda _config: pytest.fail("completed output must not load weights"),
        ).embed(manifest, output)
    with (output / "features-00000.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(FrozenExpectedRError, match="embedding features changed"):
        FrozenMantisEmbedder(
            FrozenExpectedRConfig(requested_precision="fp32"),
            model_factory=lambda _config: pytest.fail("completed output must not load weights"),
        ).embed(manifest, output)


def test_embedding_rejects_stale_input_producer_before_loading_model(
    tmp_path: Path,
) -> None:
    manifest_path = write_frozen_input(
        _candidates(3),
        np.ones((3, 5, 512), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        tmp_path / "input",
    )
    payload = json.loads(manifest_path.read_text())
    payload["expected_r_config_sha256"] = "0" * 64
    payload["manifest_sha256"] = frozen_expected_r._json_digest(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(FrozenExpectedRError, match="expected-R config mismatch"):
        FrozenMantisEmbedder(
            FrozenExpectedRConfig(),
            model_factory=lambda _config: pytest.fail("paid model must not be loaded"),
        ).embed(manifest_path, tmp_path / "embed")


def test_embedding_accepts_valid_prior_producer_identity_and_rejects_malformed_one(
    tmp_path: Path,
) -> None:
    manifest_path = write_frozen_input(
        _candidates(3),
        np.ones((3, 5, 512), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        tmp_path / "input",
    )
    payload = json.loads(manifest_path.read_text())
    payload["producer_source_sha256"] = "0" * 64
    payload["manifest_sha256"] = frozen_expected_r._json_digest(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(payload))

    validated, *_ = FrozenMantisEmbedder(FrozenExpectedRConfig()).validate_input(manifest_path)
    assert validated["producer_source_sha256"] == "0" * 64

    payload["producer_source_sha256"] = "invalid"
    payload["manifest_sha256"] = frozen_expected_r._json_digest(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(FrozenExpectedRError, match="producer source identity is invalid"):
        FrozenMantisEmbedder(FrozenExpectedRConfig()).validate_input(manifest_path)


def test_input_manifest_remains_valid_after_directory_move(tmp_path: Path) -> None:
    candidates = _candidates(6)
    manifest = write_frozen_input(
        candidates,
        np.ones((6, 5, 512), dtype=np.float32),
        np.zeros((6, 3), dtype=np.float32),
        tmp_path / "source",
    )
    shutil.move(tmp_path / "source", tmp_path / "remote")

    _, moved_candidates, _, _ = FrozenMantisEmbedder(FrozenExpectedRConfig()).validate_input(
        tmp_path / "remote" / manifest.name
    )

    assert moved_candidates["row_id"].tolist() == candidates["row_id"].tolist()


def test_embedding_artifact_remains_valid_after_replication_move(tmp_path: Path) -> None:
    candidates = _candidates(3)
    manifest = write_frozen_input(
        candidates,
        np.ones((3, 5, 512), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        tmp_path / "input",
    )

    class Model:
        def __call__(self, values: np.ndarray, _precision: str) -> np.ndarray:
            return np.ones((len(values), 2560), dtype=np.float32)

    config = FrozenExpectedRConfig(requested_precision="fp32")
    FrozenMantisEmbedder(config, model_factory=lambda _config: Model()).embed(
        manifest, tmp_path / "pod-embed"
    )
    shutil.move(tmp_path / "pod-embed", tmp_path / "replicated-embed")

    loaded_candidates, _raw, features = load_frozen_embeddings(
        manifest, tmp_path / "replicated-embed" / "manifest.json", config
    )

    assert loaded_candidates["row_id"].tolist() == candidates["row_id"].tolist()
    assert features.shape == (3, 2560)


def test_comparison_rejects_mismatched_rows() -> None:
    candidates = _candidates()
    raw = np.ones((len(candidates), 3), dtype=np.float32)
    mantis = np.ones((len(candidates) - 1, 3), dtype=np.float32)

    with pytest.raises(FrozenExpectedRError, match="identical candidate rows"):
        compare_frozen_to_raw(candidates, raw, mantis, FrozenExpectedRConfig())


def test_parallel_comparison_is_numerically_identical_to_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _candidates(240)
    dates = pd.date_range("2024-01-01T15:00:00Z", periods=len(candidates), freq="D")
    candidates["decision_ts"] = dates
    candidates["outcome_ts"] = dates + pd.Timedelta(minutes=30)
    monkeypatch.setattr(
        frozen_expected_r,
        "_INITIAL_SCREEN",
        (
            "initial_screen",
            "2024-01-01",
            "2024-03-01",
            "2024-03-01",
            "2024-04-01",
            "2024-04-01",
            "2024-05-01",
        ),
    )
    monkeypatch.setattr(
        frozen_expected_r,
        "_ANCHORED_FOLDS",
        (
            (
                "fold_1",
                "2024-01-01",
                "2024-03-01",
                "2024-03-01",
                "2024-04-01",
                "2024-04-01",
                "2024-05-01",
            ),
            (
                "fold_2",
                "2024-01-01",
                "2024-05-01",
                "2024-05-01",
                "2024-06-01",
                "2024-06-01",
                "2024-07-01",
            ),
            (
                "fold_3",
                "2024-01-01",
                "2024-07-01",
                "2024-07-01",
                "2024-08-01",
                "2024-08-01",
                "2024-09-01",
            ),
        ),
    )
    rng = np.random.default_rng(86)
    raw = rng.normal(size=(len(candidates), 3)).astype(np.float32)
    mantis = rng.normal(size=(len(candidates), 12)).astype(np.float32)
    config = FrozenExpectedRConfig(bootstrap_replicates=10)

    serial = compare_frozen_to_raw(candidates, raw, mantis, config, maximum_workers=1)
    parallel = compare_frozen_to_raw(candidates, raw, mantis, config, maximum_workers=2)

    assert parallel == serial
    with pytest.raises(FrozenExpectedRError, match=r"comparison workers must be in \[1, 2\]"):
        compare_frozen_to_raw(candidates, raw, mantis, config, maximum_workers=3)


def test_cuda_comparison_fails_closed_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _candidates()
    features = np.ones((len(candidates), 3), dtype=np.float32)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(FrozenExpectedRError, match="CUDA comparison requested"):
        compare_frozen_to_raw(
            candidates,
            features,
            features,
            FrozenExpectedRConfig(),
            comparison_device="cuda",
        )


def test_production_compare_cli_defaults_to_cuda_and_cpu_requires_exception(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "frozen-screen-compare",
            "--config",
            "config.json",
            "--input",
            "input.json",
            "--embeddings",
            "embeddings.json",
            "--output",
            "output.json",
        ]
    )
    assert args.comparison_device == "cuda"
    with pytest.raises(FrozenExpectedRError, match="CPU comparison requires"):
        frozen_expected_r.compare_frozen_artifacts(
            tmp_path / "input.json",
            tmp_path / "embeddings.json",
            tmp_path / "output.json",
            FrozenExpectedRConfig(),
            comparison_device="cpu",
        )


def test_cuda_comparison_bypasses_executor_on_calling_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller = threading.get_ident()
    observed: list[int] = []

    def fit_fold(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        observed.append(threading.get_ident())
        return ({"name": "fold"}, True, False, [("day", 1.0)], [])

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(frozen_expected_r, "_fit_fold", fit_fold)
    monkeypatch.setattr(
        frozen_expected_r,
        "_comparison_backend",
        lambda _device: {"device": "cuda"},
    )
    candidates = _candidates()
    features = np.ones((len(candidates), 3), dtype=np.float32)

    result = compare_frozen_to_raw(
        candidates,
        features,
        features,
        FrozenExpectedRConfig(bootstrap_replicates=2),
        comparison_device="cuda",
    )

    assert observed == [caller, caller, caller, caller]
    assert result["comparison_backend"] == {"device": "cuda"}


def test_comparison_progress_is_atomic_and_monotonic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter((10.0, 10.0, 12.0))
    monkeypatch.setattr(frozen_expected_r.time, "monotonic", lambda: next(ticks))
    path = tmp_path / "progress.json"
    progress = frozen_expected_r._ComparisonProgress(path)

    progress.write("cuda_threshold", fold="fold_1", arm="raw", thresholds_total=100)
    first = json.loads(path.read_text())
    progress.write(
        "cuda_threshold",
        fold="fold_1",
        arm="raw",
        thresholds_done=100,
        thresholds_total=100,
    )
    second = json.loads(path.read_text())

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert second["elapsed_seconds"] >= first["elapsed_seconds"]
    assert second["thresholds_done"] >= first["thresholds_done"]
    assert second["throughput_per_second"] == 50.0
    assert second["eta_seconds"] == 0.0
    assert not path.with_suffix(".json.tmp").exists()


def test_parallel_progress_writers_share_one_atomic_file(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    progress = frozen_expected_r._ComparisonProgress(path)
    with frozen_expected_r.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda index: progress.write("parallel", fold=str(index)), range(100)))

    assert json.loads(path.read_text())["sequence"] == 100
    assert not path.with_suffix(".json.tmp").exists()


def test_initial_gate_is_distinct_and_not_counted_as_development_win(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fit_fold(fold: tuple[str, ...], *_args: object, **_kwargs: object) -> tuple[object, ...]:
        calls.append(fold[0])
        return ({"name": fold[0]}, True, False, [(fold[0], 1.0)], [])

    monkeypatch.setattr(frozen_expected_r, "_fit_fold", fit_fold)
    monkeypatch.setattr(frozen_expected_r, "_stationary_expectancy_interval", lambda *_: [1.0, 1.0])
    candidates = _candidates()
    features = np.ones((len(candidates), 3), dtype=np.float32)

    result = compare_frozen_to_raw(candidates, features, features, FrozenExpectedRConfig())

    assert calls == ["initial_screen", "fold_1", "fold_2", "fold_3"]
    assert result["initial_screen"]["name"] == "initial_screen"
    assert result["promotion"]["raw_wins"] == 3


def test_failed_initial_gate_stops_before_development_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fit_fold(fold: tuple[str, ...], *_args: object, **_kwargs: object) -> tuple[object, ...]:
        calls.append(fold[0])
        return ({"name": fold[0]}, False, False, [], [])

    monkeypatch.setattr(frozen_expected_r, "_fit_fold", fit_fold)
    candidates = _candidates()
    features = np.ones((len(candidates), 3), dtype=np.float32)

    result = compare_frozen_to_raw(candidates, features, features, FrozenExpectedRConfig())

    assert calls == ["initial_screen"]
    assert result["selected"] == "stop"
    assert result["folds"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity requires a GPU")
def test_cuda_comparison_matches_cpu_predictions_metrics_and_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _candidates(240)
    dates = pd.date_range("2024-01-01T15:00:00Z", periods=len(candidates), freq="D")
    candidates["decision_ts"] = dates
    candidates["outcome_ts"] = dates + pd.Timedelta(minutes=30)
    monkeypatch.setattr(
        frozen_expected_r,
        "_ANCHORED_FOLDS",
        (
            (
                "fold_1",
                "2024-01-01",
                "2024-03-01",
                "2024-03-01",
                "2024-04-01",
                "2024-04-01",
                "2024-05-01",
            ),
            (
                "fold_2",
                "2024-01-01",
                "2024-05-01",
                "2024-05-01",
                "2024-06-01",
                "2024-06-01",
                "2024-07-01",
            ),
            (
                "fold_3",
                "2024-01-01",
                "2024-07-01",
                "2024-07-01",
                "2024-08-01",
                "2024-08-01",
                "2024-09-01",
            ),
        ),
    )
    rng = np.random.default_rng(86)
    raw = rng.normal(size=(len(candidates), 15)).astype(np.float32)
    mantis = rng.normal(size=(len(candidates), 12)).astype(np.float32)
    config = FrozenExpectedRConfig(bootstrap_replicates=10)

    cpu = compare_frozen_to_raw(candidates, raw, mantis, config, maximum_workers=1)
    cuda = compare_frozen_to_raw(
        candidates,
        raw,
        mantis,
        config,
        maximum_workers=1,
        comparison_device="cuda",
    )

    assert cuda["comparison_backend"]["device"] == "cuda"
    assert cuda["selected"] == cpu["selected"]
    assert cuda["status"] == cpu["status"]
    for cpu_fold, cuda_fold in zip(cpu["folds"], cuda["folds"], strict=True):
        for arm in ("raw", "mantis"):
            cpu_arm = cpu_fold[arm]
            cuda_arm = cuda_fold[arm]
            cpu_predictions = np.asarray([row["prediction"] for row in cpu_arm["rows"]["values"]])
            cuda_predictions = np.asarray([row["prediction"] for row in cuda_arm["rows"]["values"]])
            np.testing.assert_allclose(cuda_predictions, cpu_predictions, rtol=1e-3, atol=1e-3)
            assert cuda_arm["test"]["mse"] == pytest.approx(
                cpu_arm["test"]["mse"], rel=1e-4, abs=1e-6
            )
            assert cuda_arm["test"]["selected_trades"] == cpu_arm["test"]["selected_trades"]
            assert cuda_arm["test"]["selected_expectancy"] == pytest.approx(
                cpu_arm["test"]["selected_expectancy"], abs=1e-12
            )
            assert cuda_arm["threshold"]["value"] == pytest.approx(
                cpu_arm["threshold"]["value"], abs=1e-3
            )
            assert cuda_arm["gate"] == cpu_arm["gate"]
            cpu_intervals = cpu_arm["test"]["paired_stationary_day_block_intervals_95"]
            cuda_intervals = cuda_arm["test"]["paired_stationary_day_block_intervals_95"]
            assert cuda_intervals.keys() == cpu_intervals.keys()
            for key, cpu_interval in cpu_intervals.items():
                cuda_interval = cuda_intervals[key]
                if cpu_interval is None:
                    assert cuda_interval is None
                else:
                    tolerance = 1e-3 if key == "mse_improvement_over_constant" else 1e-12
                    np.testing.assert_allclose(cuda_interval, cpu_interval, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA threshold requires a GPU")
def test_cuda_threshold_is_exact_and_bounded() -> None:
    rows = 2048
    rng = np.random.default_rng(860)
    entries = np.arange(rows, dtype=np.int64) * 2
    outcomes = entries + rng.integers(1, 25, size=rows)
    scores = rng.normal(size=rows)
    frame = pd.DataFrame({"entry_index": entries, "outcome_index": outcomes})
    desired = 160

    def selected_count(threshold: float) -> int:
        count = 0
        busy_until = -1
        for entry, outcome, score in zip(entries, outcomes, scores, strict=True):
            if entry > busy_until and score >= threshold:
                count += 1
                busy_until = int(outcome)
        return count

    expected = min(
        np.unique(scores),
        key=lambda threshold: (
            abs(selected_count(float(threshold)) - desired),
            -threshold,
        ),
    )

    actual = cuda_threshold(frame, scores, desired, progress_label="performance")
    assert actual == expected

    performance_rows = 16384
    performance_entries = np.arange(performance_rows, dtype=np.int64) * 2
    performance_frame = pd.DataFrame(
        {
            "entry_index": performance_entries,
            "outcome_index": performance_entries + rng.integers(1, 25, size=performance_rows),
        }
    )
    performance_scores = rng.normal(size=performance_rows)
    torch.cuda.synchronize()
    started = time.monotonic()
    cuda_threshold(
        performance_frame,
        performance_scores,
        desired,
        progress_label="performance",
    )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, f"CUDA threshold took {elapsed:.3f}s; expected <2.0s"


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

    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"files": [{"sha256": receipt["input_manifest"]["sha256"]}]}))
    command = receipt["exact_command"]
    workload = {
        "start_command": ["bash", "-lc", f"{command} || {command}"],
        "artifacts": {"pod": receipt["embedding_output"]},
        "monitor": {"poll_seconds": 30},
        "resume": {"enabled": True, "same_run_only": True, "provenance_required": True},
        "maximum_duration_seconds": 6 * 3600,
        "quoted_rates": {"compute_usd_per_hour": "0.99"},
        "budget_guard": {"next_cell_maximum_usd": "9.99"},
        "input_bundle": {"manifest": {"controller_path": str(bundle)}},
    }
    assert validate_paid_runner_contract(tmp_path / "preflight.json", workload) == receipt

    workload["start_command"] = ["bash", "-lc", command]
    with pytest.raises(FrozenExpectedRError, match="exactly one safe resume"):
        validate_paid_runner_contract(tmp_path / "preflight.json", workload)

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


def test_paid_workload_resumes_embedding_and_reuses_complete_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({}))
    input_path = tmp_path / "input.json"
    input_path.write_text("{}")
    embed = tmp_path / "run" / "embed"
    comparison = tmp_path / "run" / "selection.json"
    progress = tmp_path / "run" / "selection.progress.json"
    calls: list[str] = []

    class Embedder:
        def __init__(self, _config: FrozenExpectedRConfig) -> None:
            pass

        def embed(self, _input: Path, output: Path) -> dict[str, object]:
            calls.append("embed")
            output.mkdir(parents=True, exist_ok=True)
            (output / "manifest.json").write_text("{}")
            return {"status": "complete"}

    def compare(*_args: object, **_kwargs: object) -> dict[str, object]:
        if comparison.exists():
            return {"status": "stopped"}
        calls.append("compare")
        comparison.write_text('{"status":"stopped"}')
        return {"status": "stopped"}

    monkeypatch.setattr(frozen_expected_r, "FrozenMantisEmbedder", Embedder)
    monkeypatch.setattr(frozen_expected_r, "compare_frozen_artifacts", compare)
    monkeypatch.setattr(
        frozen_expected_r.FrozenExpectedRConfig,
        "from_json",
        classmethod(lambda _cls, _path: FrozenExpectedRConfig()),
    )

    first = run_paid_frozen_screen(config_path, input_path, embed, comparison, progress)
    second = run_paid_frozen_screen(config_path, input_path, embed, comparison, progress)

    assert first == {"embedding_status": "complete", "selection_status": "stopped"}
    assert second == first
    assert calls == ["embed", "compare", "embed"]


def test_paid_workload_rejects_partial_or_modified_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({}))
    input_path = tmp_path / "input.json"
    input_path.write_text("{}")
    embed = tmp_path / "run" / "embed"
    embed.mkdir(parents=True)
    (embed / "manifest.json").write_text("{}")
    comparison = tmp_path / "run" / "selection.json"
    progress = tmp_path / "run" / "selection.progress.json"
    comparison.with_suffix(".json.tmp").write_text("partial")

    monkeypatch.setattr(
        frozen_expected_r.FrozenExpectedRConfig,
        "from_json",
        classmethod(lambda _cls, _path: FrozenExpectedRConfig()),
    )
    monkeypatch.setattr(
        frozen_expected_r.FrozenMantisEmbedder,
        "embed",
        lambda *_args, **_kwargs: {"status": "complete"},
    )

    with pytest.raises(FrozenExpectedRError, match="partial comparison"):
        run_paid_frozen_screen(config_path, input_path, embed, comparison, progress)


def test_completed_comparison_resume_is_bound_to_all_frozen_identities(tmp_path: Path) -> None:
    config = FrozenExpectedRConfig()
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"manifest_sha256": "a" * 64}))
    embed_path = tmp_path / "embed.json"
    embed_path.write_text(
        json.dumps(
            {
                "artifact_sha256": "b" * 64,
                "precision": "bf16",
                "bf16_parity": {"maximum_absolute_error": 0.0, "minimum_cosine": 1.0},
            }
        )
    )
    output = tmp_path / "selection.json"
    result = {
        "schema_version": 1,
        "config_sha256": config.digest,
        "comparison_backend": {"device": "cuda"},
        "selected": "stop",
        "status": "stopped",
        "provenance": frozen_expected_r._comparison_provenance(input_path, embed_path, config),
    }
    result["artifact_sha256"] = frozen_expected_r._json_digest(result)
    output.write_text(json.dumps(result))

    assert (
        frozen_expected_r._validated_completed_comparison(
            input_path, embed_path, output, config, "cuda"
        )
        == result
    )

    result["provenance"]["weights_sha256"] = "0" * 64
    result["artifact_sha256"] = frozen_expected_r._json_digest(
        {key: value for key, value in result.items() if key != "artifact_sha256"}
    )
    output.write_text(json.dumps(result))
    with pytest.raises(FrozenExpectedRError, match="provenance mismatch"):
        frozen_expected_r._validated_completed_comparison(
            input_path, embed_path, output, config, "cuda"
        )


def test_paid_planning_inputs_are_config_driven_and_gpu_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.json"
    input_core = {"schema_version": 1, "rows": 1}
    input_path.write_text(
        json.dumps({**input_core, "manifest_sha256": frozen_expected_r._json_digest(input_core)})
    )
    control = {
        "schema_version": 1,
        "run_id": "frozen-paid-test",
        "source_revision": "a" * 40,
        "frozen_config": str(
            Path(__file__).resolve().parents[1] / "configs/frozen-expected-r-v1.json"
        ),
        "input_manifest": str(input_path),
        "input_bundle_manifest": str(tmp_path / "bundle.json"),
        "dependency_lock": str(tmp_path / "uv.lock"),
        "source_archive": str(tmp_path / "source.tar.gz"),
        "official_bootstrap_receipt": str(tmp_path / "bootstrap.json"),
        "spend_ledger": str(tmp_path / "ledger.json"),
        "authorization": str(tmp_path / "authorization.json"),
        "heartbeat_token": str(tmp_path / "heartbeat.token"),
        "pod_paths": {
            "authorization": "/workspace/mantis/control/authorization.json",
            "dependency_lock": "/workspace/mantis/control/uv.lock",
            "frozen_config": "/workspace/mantis/inputs/digest/config.json",
            "heartbeat_token": "/workspace/mantis/control/heartbeat.token",
            "input_bundle_manifest": "/workspace/mantis/control/bundle.json",
            "input_manifest": "/workspace/mantis/inputs/digest/input/manifest.json",
            "official_bootstrap_receipt": "/workspace/mantis/control/bootstrap.json",
            "preflight": "/workspace/mantis/control/preflight.json",
            "source_archive": "/workspace/mantis/control/source.tar.gz",
            "spend_ledger": "/workspace/mantis/control/ledger.json",
            "workload_experiment": "/workspace/mantis/control/experiment.json",
        },
        "artifacts": {
            "controller": str(tmp_path / "controller"),
            "backup": str(tmp_path / "backup"),
        },
        "provider": {
            "gpu_type": "NVIDIA H100 80GB HBM3",
            "hourly_rate_usd": 2.0,
            "budget_usd": 10.0,
            "deadline_hours": 6.0,
            "datacenter_id": "US-MO-1",
            "volume_id": "volume",
            "volume_size_gb": 150,
            "vcpu": 8,
            "ram_gb": 32,
            "storage_usd_per_gb_hour": "0.000137",
        },
        "runpodctl": {
            "version": "2.7.2",
            "source_commit": "309512b4926eb7d218bbc8a8f11d380ce54f59c4",
            "binary_sha256": "b" * 64,
        },
    }
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps(control))

    def focused(path: Path) -> dict[str, object]:
        payload = {
            "checks": {name: True for name in frozen_expected_r._FOCUSED_CHECKS},
            "duration_seconds": 1.0,
        }
        path.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(frozen_expected_r, "write_focused_check_receipt", focused)
    result = write_paid_planning_inputs(control_path, tmp_path / "plan")
    intent = json.loads((tmp_path / "plan" / "intent.json").read_text())

    assert intent["gpu_type"] == "NVIDIA H100 80GB HBM3"
    assert json.loads((tmp_path / "plan" / "preflight.json").read_text())["gpu"] == intent[
        "gpu_type"
    ]
    assert intent["gpu_count"] == 1
    assert intent["maximum_duration_seconds"] == 5 * 3600
    assert (
        "gzip -dc /workspace/mantis/inputs/digest/input/windows.npy.gz" in result["exact_command"]
    )
    assert "mv /workspace/mantis/inputs/digest/input/.windows.npy.tmp" in result["exact_command"]
    assert "frozen-screen-paid-workload" in result["exact_command"]
    assert (
        _parser()
        .parse_args(
            ["frozen-screen-plan-paid", "--control-config", "control.json", "--output", "run"]
        )
        .command
        == "frozen-screen-plan-paid"
    )

    del control["provider"]["gpu_type"]
    control_path.write_text(json.dumps(control))
    write_paid_planning_inputs(control_path, tmp_path / "l40s-plan")
    l40s_intent = json.loads((tmp_path / "l40s-plan" / "intent.json").read_text())
    assert l40s_intent["gpu_type"] == "NVIDIA L40S"

    control["provider"]["gpu_type"] = "NVIDIA H100"
    control_path.write_text(json.dumps(control))
    with pytest.raises(FrozenExpectedRError, match="gpu_type is unsupported"):
        write_paid_planning_inputs(control_path, tmp_path / "invalid-plan")
