from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mantis_v2 import cli
from mantis_v2 import strategy as strategy_module
from mantis_v2.config import ConfigError
from mantis_v2.corpus import _json_digest, _write_frame
from mantis_v2.downstream_config import DownstreamConfig, load_downstream_config
from mantis_v2.downstream_pipeline import (
    DownstreamPipelineError,
    _embedding_manifest_input,
    _manifest_base,
    _walk_forward_quality_gate,
    evaluate_holdout,
    simulate,
    smoke,
    walk_forward,
)
from mantis_v2.embedding import EmbeddingContractError, _validated_evidence, load_foundation
from mantis_v2.instrumentation import parse_tensorboard_events
from mantis_v2.model import sha256_file
from mantis_v2.strategy import (
    _label_chunk,
    _session_horizons,
    build_symbol_candidates,
    causal_context_indices,
    load_market_frame,
    supertrend_state,
    trend_magic_state,
)
from mantis_v2.topstep import TopstepContractError, simulate_topstep
from mantis_v2.walk_forward import (
    Fold,
    WalkForwardContractError,
    fit_logistic_head,
    fold_masks,
    predict_head,
    probability_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "supertrend-topstep-100k.toml"
TUNED_CONFIG = ROOT / "configs" / "supertrend-topstep-100k-head-c0001-v2.toml"
TREND_MAGIC_CONFIG = ROOT / "configs" / "trend-magic-topstep-100k.toml"
TREND_MAGIC_TUNED_CONFIG = ROOT / "configs" / "trend-magic-topstep-100k-head-c0001-v2.toml"


def test_downstream_config_is_strict_and_supports_recorded_overrides() -> None:
    config = load_downstream_config(CONFIG, ("strategy.cooldown_bars=7",))
    assert config.data.timeframes == ("1min", "3min", "5min", "15min")
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
    with pytest.raises(ConfigError, match="must be set together"):
        load_downstream_config(
            CONFIG,
            ('walk_forward.embed_manifest_path="/tmp/embed.json"',),
        )


def test_trend_magic_config_is_strict_and_has_distinct_embedding_identity(
    tmp_path: Path,
) -> None:
    supertrend = load_downstream_config(CONFIG)
    trend_magic = load_downstream_config(TREND_MAGIC_CONFIG)
    assert trend_magic.strategy.kind == "trend_magic"
    assert trend_magic.strategy.cci_period == 20
    assert trend_magic.strategy.trend_magic_atr_period == 5
    assert trend_magic.strategy_contract == {
        "version": "trend_magic_fixed_3r_v1",
        "input_timeframes": ["1min", "3min", "5min", "15min"],
        "decision_timeframe": "3min",
        "candidate_rule": "every_eligible_closed_3min_state_bar",
        "direction_owner": "trend_magic",
        "entry_rule": "next_eligible_bar_open",
        "risk_rule": "0.5_atr_20",
        "primary_label_rule": "strict_3r_target_before_1r_stop",
        "analysis_targets_r": [2.0, 3.0, 4.0, 6.0],
        "horizon_rule": "120_bars_session_bounded",
        "session": "17:00-15:10 America/Chicago",
        "round_trip_cost_r": 0.03,
        "same_bar_policy": "stop_first",
    }
    assert trend_magic.embedding_contract_digest != supertrend.embedding_contract_digest
    invalid = TREND_MAGIC_CONFIG.read_text().replace(
        "trend_magic_multiplier = 1.0",
        "trend_magic_multiplier = 1.0\nsupertrend_period = 10",
    )
    path = tmp_path / "invalid-trend-magic.toml"
    path.write_text(invalid)
    with pytest.raises(ConfigError, match=r"unknown \[strategy\] keys: supertrend_period"):
        load_downstream_config(path)


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        ("atr_period = 20", "atr_period = 14", "strategy.atr_period"),
        ("cci_period = 20", "cci_period = 14", "strategy.cci_period"),
        (
            "trend_magic_atr_period = 5",
            "trend_magic_atr_period = 7",
            "strategy.trend_magic_atr_period",
        ),
        (
            "trend_magic_multiplier = 1.0",
            "trend_magic_multiplier = 1.5",
            "strategy.trend_magic_multiplier",
        ),
        ("stop_atr = 0.5", "stop_atr = 0.75", "strategy.stop_atr"),
        ("target_r = 3.0", "target_r = 4.0", "strategy.target_r"),
        (
            "analysis_targets_r = [2.0, 3.0, 4.0, 6.0]",
            "analysis_targets_r = [2.0, 3.0, 4.0]",
            "strategy.analysis_targets_r",
        ),
        ("horizon_bars = 120", "horizon_bars = 60", "strategy.horizon_bars"),
        (
            "round_trip_cost_r = 0.03",
            "round_trip_cost_r = 0.04",
            "strategy.round_trip_cost_r",
        ),
        (
            'timestamp_semantics = "bar_open"',
            'timestamp_semantics = "bar_close"',
            "data.timestamp_semantics",
        ),
        (
            'session_timezone = "America/Chicago"',
            'session_timezone = "UTC"',
            "topstep.session_timezone",
        ),
        (
            "session_start_hour = 17",
            "session_start_hour = 18",
            "topstep.session_start_hour",
        ),
        (
            "session_end_minute = 10",
            "session_end_minute = 0",
            "topstep.session_end_minute",
        ),
    ],
)
def test_trend_magic_fixed_3r_contract_rejects_recipe_drift(
    tmp_path: Path, old: str, new: str, field: str
) -> None:
    path = tmp_path / "drifted-trend-magic.toml"
    path.write_text(TREND_MAGIC_CONFIG.read_text().replace(old, new, 1))

    with pytest.raises(ConfigError, match=rf"{field} must be"):
        load_downstream_config(path)


def test_prepare_manifest_base_names_the_trend_magic_contract() -> None:
    config = load_downstream_config(TREND_MAGIC_CONFIG)

    manifest = _manifest_base(config, "prepare")

    assert manifest["strategy_contract"] == config.strategy_contract


def test_cli_verifies_the_trend_magic_contract_without_running_a_stage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["mantis-v2", "downstream-verify", "--config", str(TREND_MAGIC_CONFIG)],
    )

    cli.main()

    result = json.loads(capsys.readouterr().out)
    assert result["strategy_contract"]["version"] == "trend_magic_fixed_3r_v1"
    assert result["workflow_digest"] == load_downstream_config(TREND_MAGIC_CONFIG).workflow_digest


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_downstream_config_rejects_non_finite_discontinuity_threshold(
    tmp_path: Path, value: str
) -> None:
    source = CONFIG.read_text().replace(
        "context_bars = 200", f"context_bars = 200\nmax_relative_close_jump = {value}"
    )
    path = tmp_path / "invalid-threshold.toml"
    path.write_text(source)

    with pytest.raises(ConfigError, match="must be finite"):
        load_downstream_config(path)


def test_tuned_production_config_pins_reusable_embeddings_and_head_settings() -> None:
    config = load_downstream_config(TUNED_CONFIG)
    assert config.run.name == "mantisv2-supertrend-topstep-100k-head-c0001-v2"
    assert config.walk_forward.regularization_c == pytest.approx(0.0001)
    assert config.walk_forward.solver == "lbfgs"
    assert config.walk_forward.convergence_policy == "fail"
    assert config.walk_forward.embed_manifest_path is not None
    assert len(config.walk_forward.embed_manifest_sha256) == 64
    assert config.walk_forward.embed_producer_config_path == CONFIG
    assert config.walk_forward.embed_producer_config_sha256 == sha256_file(CONFIG)


def test_rejected_three_timeframe_trend_magic_config_cannot_reuse_four_timeframe_embeddings() -> (
    None
):
    config = load_downstream_config(TREND_MAGIC_TUNED_CONFIG)
    assert config.run.name == "mantisv2-trend-magic-topstep-100k-head-c0001-v2"
    assert config.strategy.kind == "trend_magic"
    assert config.walk_forward.max_iter == 1000
    assert config.walk_forward.regularization_c == pytest.approx(0.0001)
    assert config.walk_forward.solver == "lbfgs"
    assert config.walk_forward.convergence_policy == "fail"
    assert config.walk_forward.embed_manifest_sha256 == (
        "bffb74988497aec1c1c51821188e9c77b8a028c3016ddab30a843cbd785eea35"
    )
    assert config.walk_forward.embed_producer_config_path == TREND_MAGIC_CONFIG
    assert config.walk_forward.embed_producer_config_sha256 == sha256_file(TREND_MAGIC_CONFIG)
    producer = load_downstream_config(TREND_MAGIC_CONFIG)
    assert config.data.timeframes == ("1min", "3min", "15min")
    assert producer.data.timeframes == ("1min", "3min", "5min", "15min")
    assert config.embedding_contract_digest != producer.embedding_contract_digest
    with pytest.raises(DownstreamPipelineError, match="ordered 1min, 3min, 5min, 15min"):
        _embedding_manifest_input(config)


def test_head_config_digest_changes_without_changing_embed_identity() -> None:
    config = load_downstream_config(CONFIG)
    embed_sha = "a" * 64
    changed = replace(
        config,
        walk_forward=replace(config.walk_forward, regularization_c=0.0001),
    )
    assert config.head_config_digest(embed_sha) != changed.head_config_digest(embed_sha)
    assert changed.head_config_digest(embed_sha) == changed.head_config_digest(embed_sha)
    changed_holdout = replace(
        config,
        data=replace(config.data, holdout_start=config.data.holdout_start + timedelta(days=1)),
    )
    assert config.head_config_digest(embed_sha) != changed_holdout.head_config_digest(embed_sha)


@pytest.mark.parametrize("use_legacy_digest", [False, True])
def test_reusable_embed_manifest_requires_its_exact_digest(
    tmp_path: Path, use_legacy_digest: bool
) -> None:
    config = load_downstream_config(CONFIG)
    manifest_path = tmp_path / "embed-manifest.json"
    workflow_digest = config.legacy_workflow_digest if use_legacy_digest else config.workflow_digest
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "embed",
                "workflow_digest": workflow_digest,
                "foundation_weights_sha256": config.foundation.weights_sha256,
                "embedding_dim_per_channel": 256,
                "feature_width": 5120,
                "rows": 1,
                "outputs": [{"rows": 1}],
            }
        )
    )
    config = replace(
        config,
        walk_forward=replace(
            config.walk_forward,
            embed_manifest_path=manifest_path,
            embed_manifest_sha256=sha256_file(manifest_path),
            embed_producer_config_path=CONFIG,
            embed_producer_config_sha256=sha256_file(CONFIG),
        ),
    )
    manifest, path, digest = _embedding_manifest_input(config)
    assert manifest["rows"] == 1
    assert path == manifest_path.resolve()
    assert digest == config.walk_forward.embed_manifest_sha256
    manifest_path.write_text("{}")
    with pytest.raises(DownstreamPipelineError, match="digest mismatch"):
        _embedding_manifest_input(config)


def test_reusable_embed_manifest_rejects_declared_strategy_contract_drift(
    tmp_path: Path,
) -> None:
    producer = load_downstream_config(TREND_MAGIC_CONFIG)
    manifest_path = tmp_path / "embed-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "embed",
                "workflow_digest": producer.workflow_digest,
                "strategy_contract": {"version": "tampered"},
                "foundation_weights_sha256": producer.foundation.weights_sha256,
                "embedding_dim_per_channel": 256,
                "feature_width": 5120,
                "rows": 1,
                "outputs": [{"rows": 1}],
            }
        )
    )
    consumer = replace(
        producer,
        walk_forward=replace(
            producer.walk_forward,
            embed_manifest_path=manifest_path,
            embed_manifest_sha256=sha256_file(manifest_path),
            embed_producer_config_path=TREND_MAGIC_CONFIG,
            embed_producer_config_sha256=sha256_file(TREND_MAGIC_CONFIG),
        ),
    )

    with pytest.raises(DownstreamPipelineError, match="strategy_contract"):
        _embedding_manifest_input(consumer)


def test_reusable_embed_manifest_rejects_same_width_semantic_changes(
    tmp_path: Path,
) -> None:
    producer = load_downstream_config(CONFIG)
    manifest_path = tmp_path / "embed-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "embed",
                "workflow_digest": producer.legacy_workflow_digest,
                "foundation_weights_sha256": producer.foundation.weights_sha256,
                "embedding_dim_per_channel": 256,
                "feature_width": 5120,
                "rows": 1,
                "outputs": [{"rows": 1}],
            }
        )
    )
    changed = replace(
        producer,
        foundation=replace(producer.foundation, return_transf_layer=0),
        walk_forward=replace(
            producer.walk_forward,
            embed_manifest_path=manifest_path,
            embed_manifest_sha256=sha256_file(manifest_path),
            embed_producer_config_path=CONFIG,
            embed_producer_config_sha256=sha256_file(CONFIG),
        ),
    )
    with pytest.raises(DownstreamPipelineError, match="semantics mismatch"):
        _embedding_manifest_input(changed)


def _write_embedding_fixture(config: DownstreamConfig, rows: int = 122) -> tuple[Path, Path]:
    root = config.run.artifact_root / config.run.name
    shard_root = root / "embed" / "shards"
    shard_root.mkdir(parents=True)
    feature_path = shard_root / "features-00000.npy"
    metadata_path = shard_root / "metadata-00000.parquet"
    generator = np.random.default_rng(42)
    with feature_path.open("wb") as handle:
        np.save(handle, generator.normal(size=(rows, 4)).astype(np.float32))
    decision = pd.date_range("2025-01-01T00:00:00Z", periods=rows, freq="1D")
    metadata = pd.DataFrame(
        {
            "symbol": ["NQ"] * rows,
            "decision_ts": decision,
            "label_end_ts": decision + timedelta(minutes=3),
            "label": np.arange(rows, dtype=np.int8) % 2,
        }
    )
    metadata.to_parquet(metadata_path, index=False)
    outputs = [
        {
            "number": 0,
            "rows": rows,
            "features": {
                "path": str(feature_path),
                "size": feature_path.stat().st_size,
                "sha256": sha256_file(feature_path),
            },
            "metadata": {
                "path": str(metadata_path),
                "size": metadata_path.stat().st_size,
                "sha256": sha256_file(metadata_path),
            },
        }
    ]
    manifest = {**_manifest_base(config, "embed"), "outputs": outputs, "rows": rows}
    (root / "embed" / "manifest.json").write_text(json.dumps(manifest))
    return feature_path, metadata_path


@pytest.mark.parametrize("changed_kind", ["features", "metadata"])
def test_walk_forward_rehashes_every_embedding_shard_before_fit(
    tmp_path: Path, changed_kind: str
) -> None:
    config = load_downstream_config(CONFIG)
    config = replace(
        config,
        run=replace(config.run, name=f"changed-{changed_kind}", artifact_root=tmp_path),
    )
    feature_path, metadata_path = _write_embedding_fixture(config)
    changed = feature_path if changed_kind == "features" else metadata_path
    with changed.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(DownstreamPipelineError, match=f"embedding {changed_kind} changed"):
        walk_forward(config)


def test_walk_forward_persists_failure_diagnostics_on_nonconvergence(
    tmp_path: Path,
) -> None:
    config = load_downstream_config(CONFIG)
    config = replace(
        config,
        run=replace(config.run, name="nonconvergent", artifact_root=tmp_path),
        walk_forward=replace(
            config.walk_forward,
            train_months=1,
            validation_months=1,
            test_months=1,
            stride_months=1,
            embargo_bars=0,
            max_fit_rows=100,
            max_iter=1,
        ),
    )
    _write_embedding_fixture(config)
    embed_manifest_path = tmp_path / config.run.name / "embed" / "manifest.json"
    embed_manifest_sha = sha256_file(embed_manifest_path)
    with pytest.raises(WalkForwardContractError, match="did not converge"):
        walk_forward(config)
    failure = json.loads((tmp_path / config.run.name / "walk-forward" / "failure.json").read_text())
    assert failure["fold"]["number"] == 0
    assert failure["embed_manifest_sha256"] == embed_manifest_sha
    assert failure["head_config_digest"] == config.head_config_digest(embed_manifest_sha)
    assert failure["convergence"]["converged"] is False
    assert failure["convergence"]["n_iter"] == 1


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


def test_trend_magic_matches_pinned_close_cci_sma_tr_fixture() -> None:
    close = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0])
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones_like(close),
        }
    )
    line, state = trend_magic_state(frame, cci_period=3, atr_period=2, multiplier=1.0)
    np.testing.assert_allclose(line[2:], [10.0, 10.0, 10.0, 10.0])
    np.testing.assert_array_equal(state, [0, 0, 1, -1, -1, -1])


def test_trend_magic_is_append_invariant_and_resets_at_discontinuities() -> None:
    close = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0])
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones_like(close),
            "_discontinuity": np.arange(len(close)) == 4,
        }
    )
    prefix_line, prefix_state = trend_magic_state(
        frame.iloc[:7], cci_period=3, atr_period=2, multiplier=1.0
    )
    full_line, full_state = trend_magic_state(frame, cci_period=3, atr_period=2, multiplier=1.0)
    np.testing.assert_allclose(full_line[:7], prefix_line, equal_nan=True)
    np.testing.assert_array_equal(full_state[:7], prefix_state)
    assert np.isnan(full_line[4:7]).all()
    np.testing.assert_array_equal(full_state[4:7], 0)
    assert np.isfinite(full_line[7])


def test_candidate_builder_dispatches_trend_magic_parameters_and_uses_next_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_downstream_config(TREND_MAGIC_CONFIG)
    config = replace(
        config,
        data=replace(config.data, context_bars=1),
        strategy=replace(
            config.strategy,
            atr_period=3,
            cci_period=4,
            trend_magic_atr_period=5,
            trend_magic_multiplier=1.25,
            horizon_bars=2,
            analysis_targets_r=(3.0,),
        ),
    )
    close = np.linspace(100.0, 120.0, 30)
    timestamps = pd.date_range("2025-01-01T14:00:00Z", periods=30, freq="3min")
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
    observed: list[tuple[int, int, float]] = []

    def fake_trend_magic(
        candidate_frame: pd.DataFrame,
        cci_period: int,
        atr_period: int,
        multiplier: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        observed.append((cci_period, atr_period, multiplier))
        return np.zeros(len(candidate_frame)), np.ones(len(candidate_frame), dtype=np.int8)

    monkeypatch.setattr(strategy_module, "load_market_frame", lambda *_: frame.copy())
    monkeypatch.setattr(strategy_module, "trend_magic_state", fake_trend_magic)
    monkeypatch.setattr(
        strategy_module,
        "supertrend_state",
        lambda *_: pytest.fail("Trend Magic config dispatched to Supertrend"),
    )

    candidates = build_symbol_candidates(config, "NQ")

    assert observed == [(4, 5, 1.25)]
    assert (candidates["direction"] == 1).all()
    np.testing.assert_array_equal(np.diff(candidates["decision_index"]), 1)
    assert (candidates["entry_ts"] == candidates["decision_ts"]).all()
    np.testing.assert_array_equal(
        candidates["entry_price"], frame["open"].iloc[candidates["decision_index"] + 1]
    )


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


def test_candidates_never_cross_a_discontinuous_context_or_label_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_downstream_config(CONFIG)
    config = replace(
        config,
        data=replace(config.data, context_bars=2),
        strategy=replace(
            config.strategy,
            atr_period=3,
            supertrend_period=3,
            horizon_bars=2,
            analysis_targets_r=(3.0,),
        ),
    )
    close = np.linspace(100.0, 120.0, 40)
    timestamps = pd.date_range("2025-01-01T14:00:00Z", periods=40, freq="3min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close_timestamp": timestamps + timedelta(minutes=3),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.ones_like(close),
            "_discontinuity": np.arange(40) == 20,
        }
    )
    monkeypatch.setattr(strategy_module, "load_market_frame", lambda *_: frame.copy())

    candidates = build_symbol_candidates(config, "NQ")

    assert len(candidates) > 0
    for decision_index in candidates["decision_index"]:
        assert not decision_index - 1 <= 20 <= decision_index + 2


def test_downstream_loader_detects_intrabar_stitches_from_full_ohlc(tmp_path: Path) -> None:
    config = load_downstream_config(CONFIG)
    config = replace(
        config,
        data=replace(config.data, root=tmp_path, max_relative_close_jump=0.05),
    )
    pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=3, freq="3min", tz="UTC"),
            "open": [100.0, 100.0, 100.0],
            "high": [100.5, 100.5, 100.5],
            "low": [99.5, 80.0, 99.5],
            "close": [100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0],
        }
    ).to_csv(tmp_path / "NQ_3min.csv", index=False)

    frame = load_market_frame(config, "NQ", "3min")

    assert np.flatnonzero(frame["_discontinuity"].to_numpy()).tolist() == [1]


def test_downstream_loader_reads_manifest_bound_parquet(tmp_path: Path) -> None:
    config = load_downstream_config(CONFIG)
    corpus_root = tmp_path / "corpus"
    source = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [100.5, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
            "close": [100.0, 101.0, 102.0],
            "volume": [1.0, 2.0, 3.0],
            "quality_flag": [False, True, False],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="3min", tz="UTC"),
    )
    identity = {
        "kind": "market",
        "symbol": "NQ",
        "timeframe": "3min",
        **_write_frame(source, corpus_root / "market" / "NQ_3min.parquet", corpus_root),
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
        config,
        data=replace(
            config.data,
            root=corpus_root / "market",
            file_format="parquet",
            corpus_manifest_path=manifest_path,
            corpus_manifest_sha256=sha256_file(manifest_path),
        ),
    )

    frame = load_market_frame(config, "NQ", "3min")

    assert frame["close"].tolist() == [100.0, 101.0, 102.0]
    assert np.flatnonzero(frame["_discontinuity"].to_numpy()).tolist() == [1]


def test_next_open_label_resolves_same_bar_tie_as_stop() -> None:
    frame = pd.DataFrame(
        {
            "open": [99.0, 100.0, 100.0],
            "high": [100.0, 102.0, 101.0],
            "low": [98.0, 98.0, 99.0],
            "close": [99.0, 100.0, 100.0],
        }
    )
    label, reward, exit_price, exit_index, mae = _label_chunk(
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
    assert mae[0] == pytest.approx(-1.03)
    assert exit_price.tolist() == [99.0]
    assert exit_index.tolist() == [1]


def test_fixed_3r_label_ignores_a_target_reached_after_an_earlier_stop() -> None:
    frame = pd.DataFrame(
        {
            "open": [99.0, 100.0, 100.0],
            "high": [100.0, 101.0, 103.0],
            "low": [98.0, 99.0, 99.5],
            "close": [99.0, 99.0, 103.0],
        }
    )

    label, reward, exit_price, exit_index, _ = _label_chunk(
        frame,
        np.array([0]),
        np.array([1]),
        np.array([1.0]),
        target_r=3.0,
        horizon=2,
        cost_r=0.03,
    )

    assert label.tolist() == [0]
    assert reward[0] == pytest.approx(-1.03)
    assert exit_price.tolist() == [99.0]
    assert exit_index.tolist() == [1]


def test_session_horizon_enforces_the_configured_1510_chicago_cutoff() -> None:
    config = load_downstream_config(TREND_MAGIC_CONFIG)
    close_timestamps = pd.Series(
        pd.to_datetime(
            [
                "2025-01-06T21:03:00Z",
                "2025-01-06T21:06:00Z",
                "2025-01-06T21:09:00Z",
                "2025-01-06T21:12:00Z",
            ],
            utc=True,
        )
    )
    entry_timestamps = pd.Series(
        pd.to_datetime(
            ["2025-01-06T21:06:00Z", "2025-01-06T21:12:00Z"],
            utc=True,
        )
    )

    horizons = _session_horizons(
        entry_timestamps,
        close_timestamps,
        np.array([1, 3]),
        config,
    )

    np.testing.assert_array_equal(horizons, [2, 0])


@pytest.mark.parametrize(
    ("direction", "high", "low", "target"),
    [
        (1, [100.0, 103.0], [99.0, 99.5], 103.0),
        (-1, [101.0, 100.5], [100.0, 97.0], 97.0),
    ],
)
def test_fixed_3r_label_is_symmetric_for_long_and_short_targets(
    direction: int, high: list[float], low: list[float], target: float
) -> None:
    frame = pd.DataFrame(
        {
            "open": [99.0, 100.0],
            "high": high,
            "low": low,
            "close": [99.5, target],
        }
    )

    label, reward, exit_price, exit_index, _ = _label_chunk(
        frame,
        np.array([0]),
        np.array([direction]),
        np.array([1.0]),
        target_r=3.0,
        horizon=1,
        cost_r=0.03,
    )

    assert label.tolist() == [1]
    assert reward[0] == pytest.approx(2.97)
    assert exit_price.tolist() == [target]
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
    head, metrics, convergence = fit_logistic_head(
        train_x, train_y, validation_x, validation_y, config
    )
    assert head.scaler_mean.tolist() == [1.5]
    probabilities = predict_head(head, validation_x)
    assert head.threshold == pytest.approx(np.percentile(probabilities, 50.0))
    assert np.isfinite(metrics["log_loss"])
    assert convergence.converged is True


def test_logistic_head_fails_closed_when_optimizer_does_not_converge() -> None:
    config = load_downstream_config(CONFIG)
    config = replace(
        config,
        walk_forward=replace(config.walk_forward, max_iter=1),
    )
    train_x = np.column_stack(
        [
            np.linspace(-5.0, 5.0, 200),
            np.sin(np.linspace(-5.0, 5.0, 200)),
        ]
    )
    train_y = (train_x[:, 0] + train_x[:, 1] > 0).astype(np.int8)

    with pytest.raises(WalkForwardContractError, match="did not converge"):
        fit_logistic_head(train_x, train_y, train_x, train_y, config)


def test_walk_forward_quality_gate_requires_both_primary_baselines() -> None:
    passing = [{"test_metrics": {"weighted_log_loss": 0.68, "weighted_brier": 0.24}}]
    failing = [{"test_metrics": {"weighted_log_loss": 0.68, "weighted_brier": 0.26}}]
    assert _walk_forward_quality_gate(passing)["passed"] is True
    assert _walk_forward_quality_gate(failing)["passed"] is False


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
        "convergence_gate_passed": True,
        "quality_gate": {"passed": True},
        "folds": [{"head": {"path": str(head_path), "sha256": "0" * 64}}],
    }
    (root / "walk-forward" / "manifest.json").write_text(json.dumps(walk_manifest))
    with pytest.raises(DownstreamPipelineError, match="head digest mismatch"):
        evaluate_holdout(config, config.evaluation.holdout_unlock)


@pytest.mark.parametrize(
    ("convergence_passed", "quality_passed", "message"),
    [
        (False, True, "convergence gate"),
        (True, False, "quality gate"),
    ],
)
def test_simulation_rejects_an_ungated_walk_forward_run(
    tmp_path: Path,
    convergence_passed: bool,
    quality_passed: bool,
    message: str,
) -> None:
    config = load_downstream_config(CONFIG)
    config = replace(config, run=replace(config.run, artifact_root=tmp_path))
    path = tmp_path / config.run.name / "walk-forward" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                **_manifest_base(config, "walk-forward"),
                "convergence_gate_passed": convergence_passed,
                "quality_gate": {"passed": quality_passed},
                "folds": [],
            }
        )
    )
    with pytest.raises(DownstreamPipelineError, match=message):
        simulate(config)


def test_downstream_smoke_writes_and_verifies_all_stage_manifests(tmp_path: Path) -> None:
    config = load_downstream_config(ROOT / "configs" / "downstream-smoke.toml")
    config = replace(config, run=replace(config.run, artifact_root=tmp_path))
    result = smoke(config)
    assert result["verified"] is True
    assert result["device"] == "cpu"
    for stage in ("prepare", "embed", "walk-forward", "simulate"):
        assert (tmp_path / config.run.name / stage / "manifest.json").is_file()
    root = tmp_path / config.run.name
    for stage, manifest_stage in (("embed", "embed"), ("walk_forward", "walk-forward")):
        telemetry = json.loads((root / "instrumentation" / f"{stage}.json").read_text())
        manifest = json.loads((root / manifest_stage / "manifest.json").read_text())
        assert manifest["instrumentation"] == {
            "path": str(root / "instrumentation" / f"{stage}.json"),
            **telemetry,
        }
    events = parse_tensorboard_events(root / "events")["scalars"]
    assert events["stage/embed/rows"][-1]["value"] == 12
    assert events["stage/embed/shards"][-1]["value"] == 1
    assert events["stage/walk_forward/folds"][-1]["value"] == 1
    assert events["stage/walk_forward/validation_weighted_log_loss"]


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
    assert result.trading_days == 1
    assert len(trades) == 1


def test_topstep_rejects_stale_exact_stop_mae_below_the_stop_outcome() -> None:
    config = load_downstream_config(CONFIG)
    predictions = pd.DataFrame([_prediction(-1.03, -63.63, "2025-01-06T15:00:00Z")])

    with pytest.raises(TopstepContractError, match="exact-stop MAE"):
        simulate_topstep(predictions, config)


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
