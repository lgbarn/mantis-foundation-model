from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mantis_v2 import strategy as strategy_module
from mantis_v2.config import ConfigError
from mantis_v2.downstream_config import load_downstream_config
from mantis_v2.downstream_pipeline import (
    DownstreamPipelineError,
    _manifest_base,
    evaluate_holdout,
    smoke,
)
from mantis_v2.embedding import EmbeddingContractError, _validated_evidence, load_foundation
from mantis_v2.model import sha256_file
from mantis_v2.strategy import (
    _label_chunk,
    build_symbol_candidates,
    causal_context_indices,
    supertrend_state,
)
from mantis_v2.topstep import TopstepContractError, simulate_topstep
from mantis_v2.walk_forward import (
    Fold,
    fit_logistic_head,
    fold_masks,
    predict_head,
    probability_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "supertrend-topstep-100k.toml"


def test_downstream_config_is_strict_and_supports_recorded_overrides() -> None:
    config = load_downstream_config(CONFIG, ("strategy.cooldown_bars=7",))
    assert config.data.timeframes == ("1min", "3min", "15min")
    assert config.topstep.starting_balance == 100_000
    assert config.strategy.cooldown_bars == 7
    unlocked = replace(
        config,
        evaluation=replace(config.evaluation, allow_holdout=True),
    )
    assert unlocked.digest != config.digest
    assert unlocked.workflow_digest == config.workflow_digest
    with pytest.raises(ConfigError, match="unknown override key"):
        load_downstream_config(CONFIG, ("strategy.magic=7",))


def test_causal_alignment_cannot_see_a_forming_higher_timeframe_bar() -> None:
    decisions = np.array(["2026-01-01T09:42", "2026-01-01T09:45"], dtype="datetime64[ns]")
    completed_15m = np.array(
        ["2026-01-01T09:30", "2026-01-01T09:45", "2026-01-01T10:00"],
        dtype="datetime64[ns]",
    )
    indices = causal_context_indices(decisions, completed_15m, context_bars=1)
    np.testing.assert_array_equal(indices, [0, 1])


def test_supertrend_emits_state_on_each_warmed_bar_not_only_flips() -> None:
    close = np.linspace(100.0, 120.0, 30)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.ones_like(close),
        }
    )
    state = supertrend_state(frame, period=3, multiplier=2.0)
    assert np.all(state[2:] == 1)
    assert len(state[2:]) > 1


def test_candidate_builder_emits_each_eligible_state_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_downstream_config(CONFIG)
    config = replace(
        config,
        data=replace(config.data, context_bars=1, timestamp_semantics="bar_close"),
        strategy=replace(
            config.strategy,
            atr_period=3,
            supertrend_period=3,
            horizon_bars=2,
            analysis_targets_r=(3.0,),
        ),
    )
    close = np.linspace(100.0, 120.0, 30)
    timestamps = pd.date_range("2025-01-01T14:00:00Z", periods=30, freq="3min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close_timestamp": timestamps,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.ones_like(close),
        }
    )
    monkeypatch.setattr(strategy_module, "load_market_frame", lambda *_: frame.copy())
    candidates = build_symbol_candidates(config, "NQ")
    np.testing.assert_array_equal(np.diff(candidates["decision_index"]), 1)
    assert len(candidates) == 26
    assert candidates.iloc[0]["entry_ts"] == candidates.iloc[0]["decision_ts"]


def test_pre_holdout_builder_never_evaluates_a_holdout_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_downstream_config(CONFIG)
    config = replace(
        config,
        data=replace(config.data, context_bars=1),
        strategy=replace(
            config.strategy,
            atr_period=3,
            supertrend_period=3,
            horizon_bars=2,
            analysis_targets_r=(3.0,),
        ),
    )
    close = np.linspace(100.0, 120.0, 30)
    timestamps = pd.date_range("2025-12-31T22:30:00Z", periods=30, freq="3min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close_timestamp": timestamps + timedelta(minutes=3),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.ones_like(close),
        }
    )
    original_label = strategy_module._label_chunk

    def guarded_label(
        candidate_frame: pd.DataFrame,
        candidate_indices: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        future_end = candidate_frame["close_timestamp"].iloc[candidate_indices + 2]
        assert (future_end < pd.Timestamp(config.data.holdout_start)).all()
        return original_label(candidate_frame, candidate_indices, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(strategy_module, "load_market_frame", lambda *_: frame.copy())
    monkeypatch.setattr(strategy_module, "_label_chunk", guarded_label)
    candidates = build_symbol_candidates(config, "NQ")
    assert not candidates["is_holdout"].any()


def test_next_open_label_resolves_same_bar_tie_as_stop() -> None:
    frame = pd.DataFrame(
        {
            "open": [99.0, 100.0, 100.0],
            "high": [100.0, 102.0, 101.0],
            "low": [98.0, 98.0, 99.0],
            "close": [99.0, 100.0, 100.0],
        }
    )
    label, reward, exit_price, exit_index, _ = _label_chunk(
        frame,
        np.array([0]),
        np.array([1]),
        np.array([1.0]),
        target_r=2.0,
        horizon=2,
        cost_r=0.03,
    )
    assert label.tolist() == [0]
    assert reward[0] == pytest.approx(-1.03)
    assert exit_price.tolist() == [99.0]
    assert exit_index.tolist() == [1]


def test_label_force_closes_before_a_later_session_target() -> None:
    frame = pd.DataFrame(
        {
            "open": [99.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.5, 100.5, 103.0],
            "low": [98.0, 99.5, 99.5, 99.5],
            "close": [99.0, 100.0, 100.5, 103.0],
        }
    )
    label, reward, exit_price, exit_index, _ = _label_chunk(
        frame,
        np.array([0]),
        np.array([1]),
        np.array([1.0]),
        target_r=2.0,
        horizon=3,
        cost_r=0.03,
        effective_horizons=np.array([2]),
    )
    assert label.tolist() == [0]
    assert reward[0] == pytest.approx(0.47)
    assert exit_price.tolist() == [100.5]
    assert exit_index.tolist() == [2]


def test_fold_masks_purge_event_spans_at_partition_boundary() -> None:
    config = load_downstream_config(CONFIG)
    config = replace(config, walk_forward=replace(config.walk_forward, embargo_bars=0))
    fold = Fold(
        0,
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-02-01", tz="UTC"),
        pd.Timestamp("2022-02-01", tz="UTC"),
        pd.Timestamp("2022-03-01", tz="UTC"),
        pd.Timestamp("2022-03-01", tz="UTC"),
        pd.Timestamp("2022-04-01", tz="UTC"),
    )
    metadata = pd.DataFrame(
        {
            "decision_ts": [
                "2022-01-15T00:00:00Z",
                "2022-01-31T00:00:00Z",
                "2022-02-15T00:00:00Z",
                "2022-03-15T00:00:00Z",
                "2022-03-31T00:00:00Z",
            ],
            "label_end_ts": [
                "2022-01-16T00:00:00Z",
                "2022-02-02T00:00:00Z",
                "2022-02-16T00:00:00Z",
                "2022-03-16T00:00:00Z",
                "2022-04-02T00:00:00Z",
            ],
        }
    )
    train, validation, test = fold_masks(metadata, fold, config)
    assert train.tolist() == [True, False, False, False, False]
    assert validation.tolist() == [False, False, True, False, False]
    assert test.tolist() == [False, False, False, True, False]


def test_logistic_scaler_and_threshold_are_not_fit_on_test() -> None:
    config = load_downstream_config(CONFIG)
    train_x = np.array([[0.0], [1.0], [2.0], [3.0]])
    train_y = np.array([0, 0, 1, 1])
    validation_x = np.array([[1.0], [2.0], [100.0], [101.0]])
    validation_y = np.array([0, 1, 1, 1])
    head, metrics = fit_logistic_head(train_x, train_y, validation_x, validation_y, config)
    assert head.scaler_mean.tolist() == [1.5]
    probabilities = predict_head(head, validation_x)
    assert head.threshold == pytest.approx(np.percentile(probabilities, 50.0))
    assert np.isfinite(metrics["log_loss"])


def test_single_class_diagnostics_serialize_as_null_not_nan() -> None:
    metrics = probability_metrics(np.ones(3, dtype=np.int8), np.full(3, 0.8))
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert "NaN" not in json.dumps(metrics)


def test_legacy_foundation_requires_finite_identity_bound_evaluation(tmp_path: Path) -> None:
    export_root = tmp_path / "run" / "export"
    export_root.mkdir(parents=True)
    manifest_path = export_root / "manifest.json"
    manifest = {
        "provenance": {"config_digest": "config", "dataset_digest": "data"},
    }
    evaluation_path = tmp_path / "run" / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "split": "validation",
                "config_digest": "config",
                "dataset_digest": "data",
                "metrics": {"total": 1.0},
            }
        )
    )
    path, digest = _validated_evidence(manifest_path, manifest)
    assert path == evaluation_path
    assert len(digest) == 64
    evaluation_path.unlink()
    with pytest.raises(EmbeddingContractError, match="neither a validation gate"):
        _validated_evidence(manifest_path, manifest)


def test_foundation_weights_require_a_trusted_config_digest(tmp_path: Path) -> None:
    config = load_downstream_config(CONFIG)
    export_root = tmp_path / "run" / "export"
    export_root.mkdir(parents=True)
    weights_path = export_root / "model.safetensors"
    weights_path.write_bytes(b"replaced")
    manifest_path = export_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "parity": {"verified": True},
                "weights": str(weights_path),
                "provenance": {"config_digest": "config", "dataset_digest": "data"},
            }
        )
    )
    (tmp_path / "run" / "evaluation.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "config_digest": "config",
                "dataset_digest": "data",
                "metrics": {"total": 1.0},
            }
        )
    )
    config = replace(
        config,
        foundation=replace(
            config.foundation,
            manifest_path=manifest_path,
            weights_sha256="0" * 64,
        ),
    )
    with pytest.raises(EmbeddingContractError, match="does not match config"):
        load_foundation(config)


def test_holdout_requires_config_and_command_unlock() -> None:
    config = load_downstream_config(CONFIG)
    with pytest.raises(DownstreamPipelineError, match="holdout is locked"):
        evaluate_holdout(config, "REVIEWED-ONE-TIME-HOLDOUT")


def test_holdout_rejects_a_modified_head_before_scoring(tmp_path: Path) -> None:
    config = load_downstream_config(CONFIG)
    config = replace(
        config,
        run=replace(config.run, artifact_root=tmp_path),
        evaluation=replace(config.evaluation, allow_holdout=True),
    )
    root = tmp_path / config.run.name
    embed_manifest_path = root / "embed" / "manifest.json"
    embed_manifest_path.parent.mkdir(parents=True)
    embed_manifest_path.write_text(json.dumps({**_manifest_base(config, "embed"), "outputs": []}))
    head_path = root / "walk-forward" / "head.npz"
    head_path.parent.mkdir(parents=True)
    head_path.write_bytes(b"modified")
    walk_manifest = {
        **_manifest_base(config, "walk-forward"),
        "embed_manifest_sha256": sha256_file(embed_manifest_path),
        "folds": [{"head": {"path": str(head_path), "sha256": "0" * 64}}],
    }
    (root / "walk-forward" / "manifest.json").write_text(json.dumps(walk_manifest))
    with pytest.raises(DownstreamPipelineError, match="head digest mismatch"):
        evaluate_holdout(config, config.evaluation.holdout_unlock)


def test_downstream_smoke_writes_and_verifies_all_stage_manifests(tmp_path: Path) -> None:
    config = load_downstream_config(ROOT / "configs" / "downstream-smoke.toml")
    config = replace(config, run=replace(config.run, artifact_root=tmp_path))
    result = smoke(config)
    assert result["verified"] is True
    assert result["device"] == "cpu"
    for stage in ("prepare", "embed", "walk-forward", "simulate"):
        assert (tmp_path / config.run.name / stage / "manifest.json").is_file()


def _prediction(reward_r: float, mae_r: float, timestamp: str) -> dict[str, object]:
    return {
        "symbol": "NQ",
        "decision_index": 100,
        "decision_ts": timestamp,
        "entry_ts": timestamp,
        "label_end_ts": pd.Timestamp(timestamp) + timedelta(seconds=180),
        "probability": 0.9,
        "threshold": 0.5,
        "reward_r": reward_r,
        "mae_r": mae_r,
        "atr": 10.0,
    }


def test_topstep_fails_when_intraday_equity_touches_mll() -> None:
    config = load_downstream_config(CONFIG)
    predictions = pd.DataFrame([_prediction(1.0, -30.0, "2025-01-06T15:00:00Z")])
    result, trades = simulate_topstep(predictions, config)
    assert result.status == "failed"
    assert result.reason == "maximum loss limit touched intraday"
    assert result.ending_balance == pytest.approx(97_000)
    assert len(trades) == 1


def test_topstep_mll_ratchets_at_end_of_day_and_never_above_lock() -> None:
    config = load_downstream_config(CONFIG)
    config = replace(config, strategy=replace(config.strategy, cooldown_bars=0))
    predictions = pd.DataFrame(
        [
            _prediction(20.0, -0.1, "2025-01-06T15:00:00Z"),
            {**_prediction(20.0, -0.1, "2025-01-07T15:00:00Z"), "decision_index": 200},
        ]
    )
    result, _ = simulate_topstep(predictions, config)
    assert result.maximum_loss_floor == 100_000
    assert result.ending_balance == 104_000


def test_topstep_consistency_and_minimum_day_boundaries() -> None:
    config = load_downstream_config(CONFIG)
    config = replace(config, strategy=replace(config.strategy, cooldown_bars=0))
    two_days = pd.DataFrame(
        [
            _prediction(30.0, -0.1, "2025-01-06T15:00:00Z"),
            {**_prediction(30.0, -0.1, "2025-01-07T15:00:00Z"), "decision_index": 200},
        ]
    )
    strict, _ = simulate_topstep(two_days, config)
    assert strict.status == "active"
    inclusive_config = replace(
        config,
        topstep=replace(config.topstep, consistency_comparator="inclusive"),
    )
    inclusive, _ = simulate_topstep(two_days, inclusive_config)
    assert inclusive.status == "passed"
    one_day = simulate_topstep(two_days.iloc[:1], inclusive_config)[0]
    assert one_day.status == "active"


def test_topstep_rejects_oversize_and_does_not_count_rejected_days() -> None:
    config = load_downstream_config(CONFIG)
    oversize = replace(config, topstep=replace(config.topstep, contracts=11))
    with pytest.raises(TopstepContractError, match="exceed"):
        simulate_topstep(pd.DataFrame([_prediction(1.0, -0.1, "2025-01-06T15:00:00Z")]), oversize)
    below_threshold = pd.DataFrame([_prediction(1.0, -0.1, "2025-01-06T15:00:00Z")])
    below_threshold["probability"] = 0.1
    result, _ = simulate_topstep(below_threshold, config)
    assert result.trading_days == 0


def test_topstep_books_pnl_at_exit_and_rejects_cross_session_inputs() -> None:
    config = load_downstream_config(CONFIG)
    normal = pd.DataFrame([_prediction(1.0, -0.1, "2025-01-06T15:00:00Z")])
    _, trades = simulate_topstep(normal, config)
    assert trades.iloc[0]["pnl_booked_ts"] == trades.iloc[0]["label_end_ts"]
    crossing = normal.copy()
    crossing["label_end_ts"] = pd.Timestamp("2025-01-07T00:30:00Z")
    with pytest.raises(TopstepContractError, match="crosses"):
        simulate_topstep(crossing, config)


def test_optional_daily_loss_limit_is_a_soft_halt_not_account_failure() -> None:
    config = load_downstream_config(CONFIG)
    config = replace(
        config,
        strategy=replace(config.strategy, cooldown_bars=0),
        topstep=replace(config.topstep, daily_loss_limit_enabled=True),
    )
    predictions = pd.DataFrame(
        [
            _prediction(-25.0, -25.0, "2025-01-06T15:00:00Z"),
            {**_prediction(10.0, -0.1, "2025-01-06T16:00:00Z"), "decision_index": 200},
        ]
    )
    result, trades = simulate_topstep(predictions, config)
    assert result.status == "active"
    assert result.ending_balance == 98_000
    assert len(trades) == 1
