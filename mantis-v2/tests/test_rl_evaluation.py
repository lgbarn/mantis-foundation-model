"""Contract tests for the fixed Topstep promotion evaluator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from mantis_v2 import cli, rl_evaluation
from mantis_v2.rl_config import load_rl_config
from mantis_v2.rl_environment import BarData, CandidateData, EnvironmentEpisode
from mantis_v2.rl_evaluation import (
    EvaluationContractError,
    EvaluationRequest,
    paired_calendar_block_lcb,
    promotion_verdict,
    run_topstep_evaluation,
    summarize_attempts,
    wilson_one_sided,
    write_evaluation_plan,
)
from mantis_v2.rl_validation import LoadedEpisodes

ROOT = Path(__file__).resolve().parents[1]


def _attempt(index: int, *, status: str = "PASS", seed: int = 42) -> dict[str, object]:
    start = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=index)
    end = start + timedelta(hours=23)
    return {
        "fold": index // 60,
        "seed": seed,
        "ticker": "NQ",
        "profile": "one_mini",
        "regime_block": f"regime-{index // 30}",
        "calendar_block": f"week-{index // 5:03d}",
        "episode_id": f"episode-{index:04d}",
        "start_ts": start.isoformat(),
        "end_ts": end.isoformat(),
        "status": status,
        "trading_days": 5.0,
        "calendar_days": 7.0,
        "accepted_trades": 10,
        "eligible_entries": 20,
        "net_pnl": 3_100.0 if status == "PASS" else -200.0,
        "costs": 100.0,
        "commission_costs": 20.0,
        "slippage_costs": 75.0,
        "gap_costs": 5.0,
        "expectancy": 310.0 if status == "PASS" else -20.0,
        "maximum_drawdown": 500.0,
        "minimum_mll_cushion": 1_000.0,
        "best_day_profit": 800.0,
        "consistency_ratio": 0.25,
        "action_entropy": 0.69,
        "ambiguity_count": 0,
        "gap_adjusted_fills": 0,
        "latency_cancellations": 0,
        "missed_fill_count": 0,
        "finite": True,
        "accepted_opportunity_ids": [f"opportunity-{index}-{item}" for item in range(10)],
        "stress_identity_sha256": "d" * 64,
        "stress_invariants_valid": True,
    }


def test_wilson_one_sided_matches_independent_reference_values() -> None:
    lower, upper = wilson_one_sided(180, 300)
    assert lower == pytest.approx(0.552782, abs=1e-6)
    assert upper == pytest.approx(0.645430, abs=1e-6)
    _, zero_blow_upper = wilson_one_sided(0, 300)
    assert zero_blow_upper == pytest.approx(0.008938, abs=1e-6)


def test_summary_reports_raw_account_and_robust_seed_metrics() -> None:
    rows = [_attempt(index, seed=42) for index in range(300)]
    summary = summarize_attempts(rows)
    assert summary["attempt_count"] == 300
    assert summary["replay_outcomes"] == 300
    assert summary["outcomes"] == {"PASS": 300, "BLOW": 0, "TIMEOUT": 0}
    assert summary["rates"]["pass"] == 1.0
    assert summary["rates"]["blow_wilson_ucb_95"] < 0.01
    assert summary["trading_days"]["median"] == 5.0
    assert summary["accepted_trades"]["mean"] == 10.0
    assert summary["participation_rate"]["mean"] == 0.5
    assert summary["skip_rate"]["mean"] == 0.5
    assert summary["seed_robustness"]["worst_pass_rate"] == 1.0
    assert summary["seed_robustness"]["interquartile_mean_pass_rate"] == 1.0


def test_summary_rejects_overlap_nonchronology_and_nonfinite_metrics() -> None:
    rows = [_attempt(0), _attempt(1)]
    rows[1]["start_ts"] = rows[0]["start_ts"]
    with pytest.raises(EvaluationContractError, match="strictly chronological"):
        summarize_attempts(rows)
    broken = [_attempt(0)]
    broken[0]["net_pnl"] = float("nan")
    with pytest.raises(EvaluationContractError, match="finite"):
        summarize_attempts(broken)


def test_baseline_result_differences_cover_every_view_without_bootstrap() -> None:
    rows: list[dict[str, object]] = []
    for policy in sorted(rl_evaluation._POLICIES):
        for index in range(2):
            rows.append({**_attempt(index), "policy": policy, "stress": "primary"})

    reports = rl_evaluation._baseline_result_differences(rows, "primary")

    assert {report["baseline"] for report in reports} == rl_evaluation._POLICIES - {"candidate"}
    for report in reports:
        assert set(report["views"]) == {
            "pooled",
            "fold",
            "ticker",
            "profile",
            "regime_block",
            "seed",
            "cell",
        }
        assert report["views"]["pooled"][0]["pass_rate_difference"] == 0.0


def test_headline_wilson_uses_unique_seed_42_attempts_and_preserves_negative_cushion() -> None:
    rows = []
    for index in range(300):
        for seed in range(42, 52):
            row = _attempt(index, seed=seed, status="PASS" if index < 180 else "TIMEOUT")
            if index == 299 and seed == 51:
                row["status"] = "BLOW"
                row["minimum_mll_cushion"] = -25.0
            rows.append(row)

    summary = summarize_attempts(rows)

    assert summary["attempt_count"] == 300
    assert summary["replay_outcomes"] == 3_000
    assert summary["outcomes"]["PASS"] == 180
    assert summary["rates"]["pass_wilson_lcb_95"] == pytest.approx(0.5527824804455851)
    assert summary["minimum_mll_cushion"]["minimum"] == -25.0


def test_synchronized_bootstrap_resamples_whole_calendar_blocks() -> None:
    candidate = []
    baseline = []
    for block in range(20):
        for seed in range(42, 52):
            identity = {
                "fold": 0,
                "calendar_block": f"2025-W{block + 1:02d}",
                "episode_id": f"episode-{block}",
                "ticker": "NQ",
                "profile": "one_mini",
                "seed": seed,
            }
            candidate.append({**identity, "passed": block % 4 != 0})
            baseline.append({**identity, "passed": False})
    result = paired_calendar_block_lcb(
        candidate,
        baseline,
        evaluation_plan_payload_sha256="a" * 64,
        expected_seeds=tuple(range(42, 52)),
        replicates=2_000,
    )
    assert result["point_difference"] == pytest.approx(0.75)
    assert result["lcb_95"] > 0.5
    assert result["resampling_unit"] == "synchronized_calendar_block"
    assert result["seed_uncertainty"]["assignments"] == 1024
    assert len(result["index_matrix_sha256"]) == 64
    with pytest.raises(EvaluationContractError, match="synchronized calendar block"):
        paired_calendar_block_lcb(
            candidate,
            baseline,
            evaluation_plan_payload_sha256="a" * 64,
            expected_seeds=tuple(range(42, 52)),
            replicates=2_000,
            ordered_block_ids=list(reversed(result["ordered_block_ids"])),
        )


def test_calendar_bootstrap_uses_complete_cross_cell_block_intersection() -> None:
    cells = {("ES", "one_mini"), ("CL", "one_mini")}
    rows: list[dict[str, object]] = []
    for ticker, profile in sorted(cells):
        for seed in range(42, 52):
            for block in range(21):
                if ticker == "CL" and block == 20:
                    continue
                rows.append(
                    {
                        "policy": "candidate",
                        "stress": "primary",
                        "ticker": ticker,
                        "profile": profile,
                        "seed": seed,
                        "calendar_block": f"2025-W{block + 1:02d}",
                    }
                )

    blocks = rl_evaluation._complete_calendar_blocks(rows, cells, tuple(range(42, 52)))

    assert blocks == [f"2025-W{block + 1:02d}" for block in range(20)]


def test_paired_point_and_bootstrap_keep_raw_attempt_denominator() -> None:
    candidate = []
    baseline = []
    for seed in range(42, 52):
        for block in range(20):
            repetitions = 100 if block == 0 else 1
            for attempt in range(repetitions):
                identity = {
                    "fold": 0,
                    "calendar_block": f"2025-W{block + 1:02d}",
                    "episode_id": f"episode-{seed}-{block}-{attempt}",
                    "ticker": "NQ",
                    "profile": "one_mini",
                    "seed": seed,
                }
                candidate.append({**identity, "passed": block == 0})
                baseline.append({**identity, "passed": block != 0})
    result = paired_calendar_block_lcb(
        candidate,
        baseline,
        evaluation_plan_payload_sha256="a" * 64,
        expected_seeds=tuple(range(42, 52)),
        replicates=2_000,
    )
    assert result["point_difference"] == pytest.approx((100 - 19) / 119)


def test_distribution_uses_fractionally_weighted_true_iqm() -> None:
    result = rl_evaluation._distribution([0, 0, 1, 2, 3, 4, 5, 100, 100, 100])
    assert result["interquartile_mean"] == pytest.approx(12.9)


def test_promotion_is_fail_closed_for_each_profile_seed_baseline_and_stress() -> None:
    pooled = {
        "attempts": 600,
        "pass_rate": 0.70,
        "pass_lcb": 0.65,
        "blows": 0,
        "blow_ucb": 0.006,
    }
    profiles = {
        "one_mini": dict(pooled, attempts=300),
        "ten_micros": dict(pooled, attempts=300),
    }
    effects = {
        scope: {
            "take_all": {"point_difference": 0.05, "lcb_95": 0.01},
            "matched_random_take": {"point_difference": 0.04, "lcb_95": 0.005},
        }
        for scope in ("pooled", "one_mini", "ten_micros")
    }
    transfer_cells = {
        (ticker, profile)
        for ticker in ("ES", "NQ", "RTY", "YM", "GC", "CL", "ZB")
        for profile in (("one_mini",) if ticker == "ZB" else ("one_mini", "ten_micros"))
    }
    stress = {
        "pooled": pooled,
        "profiles": profiles,
        "seed_pass_rates": {
            scope: {seed: 0.65 for seed in range(42, 52)}
            for scope in ("pooled", "one_mini", "ten_micros")
        },
        "seed_blow_counts": {
            scope: {seed: 0 for seed in range(42, 52)}
            for scope in ("pooled", "one_mini", "ten_micros")
        },
        "baseline_effects": effects,
        "transfer_effects": {
            ("POOLED", "pooled"): {"point_difference": 0.01, "lcb_95": 0.0},
            **{cell: {"point_difference": 0.01, "lcb_95": 0.0} for cell in transfer_cells},
        },
    }
    verdict = promotion_verdict(
        numeric_stresses={"primary": stress, "two_tick": stress},
        sensitivity_stresses={
            name: {"complete": True}
            for name in (
                "fee_2x",
                "gap_adverse_tick",
                "latency_1bar",
                "missed_fill_10pct",
                "overlapping_starts",
                "same_bar_adverse",
            )
        },
        required_seeds=set(range(42, 52)),
        required_transfer_cells=transfer_cells,
        required_artifacts_complete=True,
    )
    assert verdict["promoted"] is True

    bad_profiles = dict(profiles)
    bad_profiles["ten_micros"] = dict(profiles["ten_micros"], pass_rate=0.59)
    bad_stress = dict(stress)
    bad_stress["profiles"] = bad_profiles
    rejected = promotion_verdict(
        numeric_stresses={"primary": bad_stress, "two_tick": stress},
        sensitivity_stresses={
            name: {"complete": True}
            for name in (
                "fee_2x",
                "gap_adverse_tick",
                "latency_1bar",
                "missed_fill_10pct",
                "overlapping_starts",
                "same_bar_adverse",
            )
        },
        required_seeds=set(range(42, 52)),
        required_transfer_cells=transfer_cells,
        required_artifacts_complete=True,
    )
    assert rejected["promoted"] is False
    assert "primary:profile:ten_micros:raw_pass_rate" in rejected["failed_clauses"]


def test_sensitivity_performance_does_not_create_a_post_hoc_gate() -> None:
    transfer_cells = {("NQ", "one_mini"), ("NQ", "ten_micros")}
    core = {"attempts": 600, "pass_rate": 0.7, "pass_lcb": 0.6, "blows": 0, "blow_ucb": 0.005}
    profiles = {profile: dict(core, attempts=300) for profile in ("one_mini", "ten_micros")}
    stress = {
        "pooled": core,
        "profiles": profiles,
        "seed_pass_rates": {
            scope: {seed: 0.6 for seed in range(42, 52)}
            for scope in ("pooled", "one_mini", "ten_micros")
        },
        "seed_blow_counts": {
            scope: {seed: 0 for seed in range(42, 52)}
            for scope in ("pooled", "one_mini", "ten_micros")
        },
        "baseline_effects": {
            scope: {
                baseline: {"point_difference": 0.01, "lcb_95": 0.001}
                for baseline in ("take_all", "matched_random_take")
            }
            for scope in ("pooled", "one_mini", "ten_micros")
        },
        "transfer_effects": {
            cell: {"point_difference": 0.0, "lcb_95": 0.0}
            for cell in transfer_cells | {("POOLED", "pooled")}
        },
    }
    sensitivity = {
        name: {"complete": True, "observed_pass_rate": 0.0}
        for name in (
            "fee_2x",
            "gap_adverse_tick",
            "latency_1bar",
            "missed_fill_10pct",
            "overlapping_starts",
            "same_bar_adverse",
        )
    }
    accepted = promotion_verdict(
        numeric_stresses={"primary": stress, "two_tick": stress},
        sensitivity_stresses=sensitivity,
        required_seeds=set(range(42, 52)),
        required_transfer_cells=transfer_cells,
        required_artifacts_complete=True,
    )
    assert accepted["promoted"] is True

    incomplete = dict(sensitivity)
    incomplete["latency_1bar"] = {"complete": False, "observed_pass_rate": 1.0}
    rejected = promotion_verdict(
        numeric_stresses={"primary": stress, "two_tick": stress},
        sensitivity_stresses=incomplete,
        required_seeds=set(range(42, 52)),
        required_transfer_cells=transfer_cells,
        required_artifacts_complete=True,
    )
    assert rejected["promoted"] is False
    assert "sensitivity:latency_1bar:complete" in rejected["failed_clauses"]


def test_checkpoint_bytes_are_revalidated_before_loading(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"changed-after-freeze")
    with pytest.raises(EvaluationContractError, match="changed after freeze"):
        rl_evaluation._loaded_actor(
            checkpoint,
            rl_evaluation.PolicyVariant.SHARED_TICKER_VALUE,
            "0" * 64,
        )


def test_matched_random_admits_next_sha_rank_when_dynamic_legality_blocks_one() -> None:
    def accepted(selected: frozenset[int]) -> int:
        return len(selected) - int({1, 2} <= selected)

    selected = rl_evaluation._sha_ranked_exact_selection([1, 2, 3, 4], 2, accepted)
    assert selected == frozenset({1, 2, 3})
    assert accepted(selected) == 2

    with pytest.raises(EvaluationContractError, match="dynamic legality"):
        rl_evaluation._sha_ranked_exact_selection([1, 2, 3], 2, lambda _selected: 1)


def test_latency_queues_then_releases_on_next_legal_bar_even_if_policy_rejects() -> None:
    action, pending, submitted = rl_evaluation._latency_one_bar_action(1, "bar-1", None)
    assert (action, pending, submitted) == (0, "bar-1", None)

    action, pending, submitted = rl_evaluation._latency_one_bar_action(0, "bar-2", pending)
    assert (action, pending, submitted) == (1, None, "bar-1")


def test_canonical_pair_key_matches_frozen_field_order() -> None:
    assert (
        rl_evaluation._canonical_pair_key(2, "2025-W03", "episode-7", "NQ", "ten_micros", 42)
        == '[2,"2025-W03","episode-7","NQ","ten_micros",42]'
    )


def test_access_plan_exists_before_metadata_and_recursively_validates_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    embedding = tmp_path / "embedding.json"
    corpus = tmp_path / "corpus.json"
    embedding.write_text('{"outputs":[]}')
    corpus.write_text('{"outputs":[]}')
    config = replace(
        config,
        upstream=replace(
            config.upstream,
            embedding_manifest_path=embedding,
            embedding_manifest_sha256=hashlib.sha256(embedding.read_bytes()).hexdigest(),
            corpus_manifest_path=corpus,
            corpus_manifest_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
        ),
    )
    serving = tmp_path / "serving.json"
    baseline = tmp_path / "baseline.json"
    selector = tmp_path / "selector.json"
    shield = tmp_path / "shield.json"
    for path in (serving, baseline, selector, shield):
        path.write_text("{}")
    parent_calls: list[Path] = []

    def serving_validator(_config: object, path: Path):
        parent_calls.append(path)
        return ({"final_timesteps": 5_000_000}, "a" * 64)

    def baseline_validator(_config: object, path: Path):
        parent_calls.append(path)
        return ({}, "b" * 64)

    monkeypatch.setattr(rl_evaluation, "_validate_serving_freeze", serving_validator)
    monkeypatch.setattr(rl_evaluation, "_validated_baseline_freeze", baseline_validator)
    serving_contract = {
        "deployment_selector": {"path": str(selector.resolve())},
        "risk_shield": {"path": str(shield.resolve())},
    }
    monkeypatch.setattr(
        rl_evaluation,
        "_validated_pre_promotion_serving_contract",
        lambda *_args: serving_contract,
    )
    result = rl_evaluation.write_evaluation_access_plan(
        config,
        serving,
        baseline,
        selector,
        shield,
        tmp_path / "access",
        run_identity="fixed-test-v1",
        created_at="2026-07-22T16:00:00+00:00",
    )
    assert result["test_metadata_accessed"] is False
    assert result["test_features_accessed"] is False
    assert result["sealed_holdout_accessed"] is False
    validated, digest = rl_evaluation._validated_evaluation_access_plan(
        config, Path(result["access_plan_path"])
    )
    assert digest == result["access_plan_sha256"]
    assert validated["status"] == "frozen_before_test_metadata"
    assert result["pre_promotion_serving"] == serving_contract
    assert parent_calls == [serving, baseline, serving, baseline]


def test_plan_is_content_addressed_and_frozen_before_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    serving = tmp_path / f"serving-freeze-{'a' * 64}.json"
    serving.write_text("{}")
    access = tmp_path / f"evaluation-access-plan-{'f' * 64}.json"
    access.write_text("{}")
    baseline_payload = {
        "schema_version": 1,
        "stage": "baseline-freeze-v1",
        "status": "complete",
        "config_sha256": config.digest,
        "test_accessed": False,
        "sealed_holdout_accessed": False,
    }
    baseline_bytes = json.dumps(baseline_payload, sort_keys=True, separators=(",", ":")).encode()
    baseline_digest = hashlib.sha256(baseline_bytes).hexdigest()
    baseline = tmp_path / f"baseline-freeze-{baseline_digest}.json"
    baseline.write_bytes(baseline_bytes)
    monkeypatch.setattr(
        rl_evaluation,
        "_validate_serving_freeze",
        lambda *_args: (
            {
                "serving_seed": config.training.serving_seed,
                "seed_artifacts": [
                    {
                        "fold": fold,
                        "seed": seed,
                        "checkpoint_bundle_path": str(tmp_path / f"bundle-{fold}-{seed}.json"),
                        "checkpoint_bundle_sha256": "b" * 64,
                    }
                    for fold in (0, 1)
                    for seed in range(42, 52)
                ],
            },
            "a" * 64,
        ),
    )
    monkeypatch.setattr(
        rl_evaluation,
        "_validated_baseline_freeze",
        lambda *_args: (baseline_payload, baseline_digest),
    )
    monkeypatch.setattr(
        rl_evaluation,
        "_validated_evaluation_access_plan",
        lambda *_args: (
            {
                "serving_freeze_path": str(serving),
                "serving_freeze_sha256": "a" * 64,
                "baseline_freeze_path": str(baseline),
                "baseline_freeze_sha256": baseline_digest,
                "pre_promotion_serving": {
                    "deployment_selector": {"path": str(tmp_path / "selector.json")},
                    "risk_shield": {"path": str(tmp_path / "shield.json")},
                },
            },
            "f" * 64,
        ),
    )
    schedules = []
    for fold in (0, 1):
        path = tmp_path / f"test-{fold}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": "rl-episode-schedule",
                    "fold": fold,
                    "partition": {
                        "name": "test",
                        "start": f"2025-0{fold + 1}-01T00:00:00+00:00",
                        "end": f"2025-0{fold + 2}-01T00:00:00+00:00",
                    },
                    "schedule_mode": "chronological_greedy_nonoverlap_v1",
                    "overlapping_starts": False,
                    "identities": {"config": {"sha256": config.digest}},
                    "episodes": [{"number": 0}],
                },
                sort_keys=True,
            )
        )
        schedules.append(path)
    cells = [
        (ticker, profile)
        for ticker in ("ES", "NQ", "RTY", "YM", "GC", "CL", "ZB")
        for profile in (("one_mini",) if ticker == "ZB" else ("one_mini", "ten_micros"))
    ]
    synthetic = {
        path: {
            "fold": fold,
            "schedule_proof": {
                "access_plan_sha256": "f" * 64,
                "test_metadata_accessed": True,
                "test_features_accessed": False,
            },
            "episodes": [
                {
                    "number": index,
                    "ticker": cells[index % len(cells)][0],
                    "profile": cells[index % len(cells)][1],
                }
                for index in range(600)
            ],
        }
        for fold, path in enumerate(schedules)
    }
    for path in schedules:
        path.write_text(json.dumps(synthetic[path], sort_keys=True, separators=(",", ":")))
    monkeypatch.setattr(
        rl_evaluation,
        "_validated_test_schedule_reference",
        lambda _config, path: (
            synthetic[path],
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        ),
    )

    result = write_evaluation_plan(
        config,
        access,
        schedules,
        tmp_path / "plan",
        run_identity="fixed-test-v1",
        created_at="2026-07-22T16:00:00+00:00",
    )

    plan_path = Path(result["evaluation_plan_path"])
    assert plan_path.name == f"evaluation-plan-{result['evaluation_plan_sha256']}.json"
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == result["evaluation_plan_sha256"]
    assert result["test_accessed"] is False
    assert result["sealed_holdout_accessed"] is False
    assert result["stress_definitions"]["missed_fill_10pct"]["offset_derivation"].endswith(
        ":missed-fill-v1')) mod 10"
    )
    validated, validated_sha256 = rl_evaluation._validated_evaluation_plan(config, plan_path)
    assert validated_sha256 == result["evaluation_plan_sha256"]
    assert validated["serving_seed"] == config.training.serving_seed

    mutations = {
        "thresholds": lambda payload: payload["thresholds"].update({"minimum_raw_pass_rate": 0.59}),
        "stress_definition": lambda payload: payload["stress_definitions"]["fee_2x"].update(
            {"numeric_promotion_gates": True}
        ),
        "report_only": lambda payload: payload["report_only_stresses"].remove("fee_2x"),
        "wilson": lambda payload: payload["wilson"].update({"z": 1.96}),
        "bootstrap": lambda payload: payload["bootstrap"].update({"replicates": 99_999}),
        "required_views": lambda payload: payload["required_views"]["numeric"].pop(),
    }
    frozen_payload = json.loads(plan_path.read_text())
    for mutate in mutations.values():
        mutated = json.loads(json.dumps(frozen_payload))
        mutate(mutated)
        data = json.dumps(mutated, sort_keys=True, separators=(",", ":")).encode()
        mutated_path = tmp_path / f"evaluation-plan-{hashlib.sha256(data).hexdigest()}.json"
        mutated_path.write_bytes(data)
        with pytest.raises(EvaluationContractError, match="preregistered contract mismatch"):
            rl_evaluation._validated_evaluation_plan(config, mutated_path)


def test_pre_promotion_contract_binds_selector_and_shared_shield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    serving_sha256 = "a" * 64
    bundles = [tmp_path / f"bundle-{fold}.json" for fold in (0, 1)]
    serving = {
        "seed_artifacts": [
            {
                "fold": fold,
                "seed": config.training.serving_seed,
                "checkpoint_bundle_path": str(bundles[fold]),
                "checkpoint_bundle_sha256": str(fold + 1) * 64,
            }
            for fold in (0, 1)
        ]
    }
    selector_payload = {
        "schema_version": 1,
        "stage": "deployment-checkpoint-selection-v1",
        "status": "complete",
        "selector": "deployment-checkpoint-selector-v1",
        "serving_seed": config.training.serving_seed,
        "selected_fold": 1,
        "serving_freeze_sha256": serving_sha256,
        "checkpoint_bundle_path": str(bundles[1].resolve()),
        "checkpoint_bundle_sha256": "2" * 64,
        "config_sha256": config.digest,
        "source_sha256": config.upstream.source_digest,
        "dependency_lock_sha256": config.upstream.lock_digest,
        "test_accessed": False,
        "sealed_holdout_accessed": False,
    }

    def write_addressed(prefix: str, payload: dict[str, object]) -> Path:
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        path = tmp_path / f"{prefix}-{hashlib.sha256(data).hexdigest()}.json"
        path.write_bytes(data)
        return path

    selector = write_addressed("deployment-checkpoint-selection", selector_payload)
    reference_paths: dict[str, Path] = {}
    for name in (
        "mask_source",
        "observation_schema",
        "rule_snapshot",
        "calendar_snapshot",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(name)
        reference_paths[name] = path

    def expected_references(_config: object) -> dict[str, dict[str, str]]:
        result = {
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in reference_paths.items()
        }
        result["fee_snapshot"] = {
            "identity": config.fees.snapshot,
            "sha256": config.fee_digest,
        }
        return result

    monkeypatch.setattr(rl_evaluation, "_expected_risk_shield_identities", expected_references)
    references = expected_references(config)
    shield_payload = {
        "schema_version": 1,
        "stage": "risk-shield-contract-v1",
        "status": "complete",
        "version": "RiskShieldV1",
        "config_sha256": config.digest,
        "source_sha256": config.upstream.source_digest,
        "dependency_lock_sha256": config.upstream.lock_digest,
        "consumers": ["training", "evaluation", "serving", "parity", "benchmark"],
        "test_accessed": False,
        "sealed_holdout_accessed": False,
        **references,
    }
    shield = write_addressed("risk-shield-contract", shield_payload)

    result = rl_evaluation._validated_pre_promotion_serving_contract(
        config, serving, serving_sha256, selector, shield
    )

    assert result["deployment_selector"]["selected_fold"] == 1
    assert result["risk_shield"]["version"] == "RiskShieldV1"

    selector_mutations = {
        "schema": ("schema_version", 2),
        "stage": ("stage", "wrong"),
        "status": ("status", "incomplete"),
        "selector": ("selector", "metric-ranked"),
        "seed": ("serving_seed", 43),
        "fold": ("selected_fold", 0),
        "serving": ("serving_freeze_sha256", "f" * 64),
        "checkpoint_path": ("checkpoint_bundle_path", str(bundles[0].resolve())),
        "checkpoint_sha": ("checkpoint_bundle_sha256", "f" * 64),
        "config": ("config_sha256", "f" * 64),
        "source": ("source_sha256", "f" * 64),
        "lock": ("dependency_lock_sha256", "f" * 64),
        "test": ("test_accessed", True),
        "holdout": ("sealed_holdout_accessed", True),
    }
    for name, (field, value) in selector_mutations.items():
        mutated = dict(selector_payload)
        mutated[field] = value
        path = write_addressed(f"deployment-checkpoint-selection-{name}", mutated)
        renamed = tmp_path / (
            f"deployment-checkpoint-selection-{hashlib.sha256(path.read_bytes()).hexdigest()}.json"
        )
        path.rename(renamed)
        with pytest.raises(EvaluationContractError, match="selection is not authorized"):
            rl_evaluation._validated_pre_promotion_serving_contract(
                config, serving, serving_sha256, renamed, shield
            )

    shield_mutations = {
        "schema": ("schema_version", 2),
        "stage": ("stage", "wrong"),
        "status": ("status", "incomplete"),
        "version": ("version", "RiskShieldV0"),
        "config": ("config_sha256", "f" * 64),
        "source": ("source_sha256", "f" * 64),
        "lock": ("dependency_lock_sha256", "f" * 64),
        "consumers": ("consumers", ["serving"]),
        "test": ("test_accessed", True),
        "holdout": ("sealed_holdout_accessed", True),
    }
    for name, (field, value) in shield_mutations.items():
        mutated = dict(shield_payload)
        mutated[field] = value
        path = write_addressed(f"risk-shield-contract-{name}", mutated)
        renamed = (
            tmp_path / f"risk-shield-contract-{hashlib.sha256(path.read_bytes()).hexdigest()}.json"
        )
        path.rename(renamed)
        with pytest.raises(EvaluationContractError, match="RiskShield contract"):
            rl_evaluation._validated_pre_promotion_serving_contract(
                config, serving, serving_sha256, selector, renamed
            )

    altered_authority = dict(shield_payload)
    altered_authority["mask_source"] = {
        "path": str((tmp_path / "attacker.py").resolve()),
        "sha256": "f" * 64,
    }
    altered_shield = write_addressed("risk-shield-contract", altered_authority)
    with pytest.raises(EvaluationContractError, match="approved identities"):
        rl_evaluation._validated_pre_promotion_serving_contract(
            config, serving, serving_sha256, selector, altered_shield
        )

    incomplete_serving = {
        "seed_artifacts": [
            serving["seed_artifacts"][0],
            {**serving["seed_artifacts"][1], "fold": 2},
        ]
    }
    with pytest.raises(EvaluationContractError, match="incomplete or nonchronological"):
        rl_evaluation._validated_pre_promotion_serving_contract(
            config, incomplete_serving, serving_sha256, selector, shield
        )

    Path(references["calendar_snapshot"]["path"]).write_text("changed")
    with pytest.raises(EvaluationContractError, match="approved identities"):
        rl_evaluation._validated_pre_promotion_serving_contract(
            config, serving, serving_sha256, selector, shield
        )


def test_test_schedule_validator_requires_content_addressed_filename(tmp_path: Path) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    episode = {
        "number": 0,
        "ticker": "NQ",
        "profile": "one_mini",
        "lookback_start": "2025-01-06T12:00:00+00:00",
        "decision_start": "2025-01-06T13:00:00+00:00",
        "exit_end": "2025-01-06T14:00:00+00:00",
        "terminal_end": "2025-01-06T15:00:00+00:00",
    }
    selected = [
        {
            key: episode[key]
            for key in (
                "number",
                "ticker",
                "profile",
                "lookback_start",
                "decision_start",
                "exit_end",
                "terminal_end",
            )
        }
    ]
    builder = ROOT / "src" / "mantis_v2" / "rl_episodes.py"
    schedule = {
        "schema_version": 1,
        "stage": "rl-episode-schedule",
        "fold": 0,
        "partition": {
            "name": "test",
            "start": "2025-01-01T00:00:00+00:00",
            "end": "2025-02-01T00:00:00+00:00",
        },
        "schedule_mode": "chronological_greedy_nonoverlap_v1",
        "overlapping_starts": False,
        "identities": {"config": {"sha256": config.digest}},
        "episodes": [episode],
        "stress_coverage": {
            "overlapping_starts": {
                "stage": "overlapping-start-coverage-v1",
                "promotion_denominator": False,
                "overlapping_starts": True,
                "episode_count": 1,
                "identity_sha256": "",
                "episodes": [
                    {
                        "number": 1,
                        "ticker": "NQ",
                        "profile": "one_mini",
                        "lookback_start": "2025-01-06T13:30:00+00:00",
                        "decision_start": "2025-01-06T14:00:00+00:00",
                        "exit_end": "2025-01-06T15:00:00+00:00",
                        "terminal_end": "2025-01-06T16:00:00+00:00",
                    }
                ],
            }
        },
        "schedule_proof": {
            "algorithm": "chronological_greedy_nonoverlap_v1",
            "selected_identity_sha256": hashlib.sha256(
                json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "builder_code_path": str(builder),
            "builder_code_sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
        },
    }
    overlap = schedule["stress_coverage"]["overlapping_starts"]
    overlap["identity_sha256"] = hashlib.sha256(
        json.dumps(overlap["episodes"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    data = json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(data).hexdigest()
    mutable_name = tmp_path / "test-schedule.json"
    mutable_name.write_bytes(data)
    with pytest.raises(EvaluationContractError, match="provenance mismatch"):
        rl_evaluation._validated_test_schedule_reference(config, mutable_name)
    frozen = tmp_path / f"evaluation-schedule-{digest}.json"
    frozen.write_bytes(data)
    _payload, reference = rl_evaluation._validated_test_schedule_reference(config, frozen)
    assert reference == {"path": str(frozen.resolve()), "sha256": digest}


@pytest.mark.parametrize(
    "mutation",
    ["parameters", "manifest", "lineage", "ledger", "budget", "source"],
)
def test_independent_plan_recomputes_every_issue9_constraint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    candidate_path = tmp_path / "candidate.json"
    serving_path = tmp_path / "serving.json"
    architecture_path = tmp_path / "architecture.json"
    qualification_path = tmp_path / "qualification.json"
    for path in (candidate_path, serving_path, architecture_path):
        path.write_text("{}")
    training = tmp_path / "training.json"
    validation = tmp_path / "validation.json"
    training.write_text("{}")
    validation.write_text("{}")
    pair = {
        "fold": 0,
        "training_manifest_path": str(training.resolve()),
        "training_manifest_sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
        "validation_manifest_path": str(validation.resolve()),
        "validation_manifest_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
    }
    report = tmp_path / "independent-report.json"
    report.write_text(
        json.dumps(
            {
                "variant": "independent_actor",
                "fold": 0,
                "seed": 42,
                "partition": "validation",
                "validation_manifest_sha256": pair["validation_manifest_sha256"],
                "test_accessed": False,
                "sealed_holdout_accessed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    report_digest = hashlib.sha256(report.read_bytes()).hexdigest()
    qualification_path.write_text(
        json.dumps({"reports": [{"path": str(report), "sha256": report_digest}]})
    )
    candidate = {
        "architecture_plan_path": str(architecture_path),
        "qualification_report_path": str(qualification_path),
        "optuna_parameters": {"learning_rate": 0.001},
    }
    serving = {
        "candidate_sha256": "a" * 64,
        "final_timesteps": config.training.confirmation_timesteps_per_seed,
        "continuation_decision_path": str(tmp_path / "decision.json"),
        "continuation_decision_sha256": "d" * 64,
    }
    monkeypatch.setattr(rl_evaluation, "_validated_candidate", lambda *_args: (candidate, "a" * 64))
    monkeypatch.setattr(
        rl_evaluation, "_validate_serving_freeze", lambda *_args: (serving, "b" * 64)
    )
    monkeypatch.setattr(rl_evaluation, "_validated_plan_pairs", lambda *_args: [pair])
    ledger = [
        {
            "ordinal": 1,
            "fold": 0,
            "seed": 42,
            "variant": "independent_actor",
            "timesteps": config.training.confirmation_timesteps_per_seed,
        }
    ]
    plan = {
        "schema_version": 1,
        "stage": "independent-baseline-campaign-plan-v1",
        "status": "frozen_before_training",
        "config_sha256": config.digest,
        "source_sha256": config.upstream.source_digest,
        "dependency_lock_sha256": config.upstream.lock_digest,
        "candidate_path": str(candidate_path),
        "candidate_sha256": "a" * 64,
        "serving_freeze_path": str(serving_path),
        "serving_freeze_sha256": "b" * 64,
        "continuation_decision_path": serving["continuation_decision_path"],
        "continuation_decision_sha256": serving["continuation_decision_sha256"],
        "final_timesteps": config.training.confirmation_timesteps_per_seed,
        "parameters": candidate["optuna_parameters"],
        "manifest_pairs": [pair],
        "issue9_independent_lineage": [
            {"fold": 0, "seed": 42, "path": str(report.resolve()), "sha256": report_digest}
        ],
        "expected_append_only_ledger": ledger,
        "test_accessed": False,
        "sealed_holdout_accessed": False,
    }
    rl_evaluation._validated_independent_campaign_plan_constraints(config, plan)
    mutated = json.loads(json.dumps(plan))
    if mutation == "parameters":
        mutated["parameters"] = {"learning_rate": 0.002}
    elif mutation == "manifest":
        mutated["manifest_pairs"][0]["training_manifest_sha256"] = "0" * 64
    elif mutation == "lineage":
        mutated["issue9_independent_lineage"][0]["seed"] = 43
    elif mutation == "ledger":
        mutated["expected_append_only_ledger"][0]["variant"] = "shared_ticker_value"
    elif mutation == "budget":
        mutated["final_timesteps"] -= 1
    else:
        mutated["source_sha256"] = "0" * 64
    with pytest.raises(EvaluationContractError):
        rl_evaluation._validated_independent_campaign_plan_constraints(config, mutated)


def test_independent_campaign_runs_every_fold_seed_fresh_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    plan_path = tmp_path / "campaign-plan.json"
    plan_path.write_text("{}")
    expected = [
        {
            "ordinal": ordinal,
            "fold": fold,
            "seed": seed,
            "variant": "independent_actor",
            "timesteps": 5_000_000,
        }
        for ordinal, (fold, seed) in enumerate(
            ((fold, seed) for fold in (0, 1) for seed in range(42, 52)), start=1
        )
    ]
    plan = {
        "manifest_pairs": [
            {
                "fold": fold,
                "training_manifest_sha256": f"{fold + 1}" * 64,
                "validation_manifest_sha256": f"{fold + 3}" * 64,
            }
            for fold in (0, 1)
        ],
        "expected_append_only_ledger": expected,
        "final_timesteps": 5_000_000,
        "parameters": {},
        "candidate_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        rl_evaluation,
        "_validated_independent_campaign_plan",
        lambda *_args: (plan, "b" * 64),
    )
    monkeypatch.setattr(rl_evaluation, "_result_passed", lambda _result: True)
    monkeypatch.setattr(rl_evaluation, "_lineage_passed", lambda _request, _result: True)
    requests: list[rl_evaluation.ConfirmationRequest] = []

    def runner(request: rl_evaluation.ConfirmationRequest) -> dict[str, object]:
        requests.append(request)
        artifact_root = request.output / "checkpoint"
        artifact_root.mkdir(parents=True)
        checkpoint = artifact_root / "checkpoint.pt"
        checkpoint.write_bytes(f"{request.fold}:{request.seed}".encode())
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        bundle = artifact_root / "bundle.json"
        bundle.write_text("{}")
        training = request.output / "manifest.json"
        training.write_text("{}")
        validation = request.output / "validation.json"
        validation.write_text("{}")
        return {
            "status": "complete",
            "finite": True,
            "action_collapsed": False,
            "blows": 0,
            "all_gates_passed": True,
            "pass_rate": 0.7,
            "artifact_sha256": checkpoint_sha,
            "completed_timesteps": request.timesteps,
            "lineage_parent_sha256": None,
            "training_manifest_path": str(training),
            "training_manifest_sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
            "checkpoint_bundle_path": str(bundle),
            "checkpoint_bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            "validation_report_path": str(validation),
            "validation_report_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
        }

    result = rl_evaluation.run_independent_baseline_campaign(
        config, plan_path, tmp_path / "campaign", runner=runner
    )
    assert len(requests) == 20
    assert {(request.fold, request.seed) for request in requests} == {
        (fold, seed) for fold in (0, 1) for seed in range(42, 52)
    }
    assert all(request.variant == "independent_actor" for request in requests)
    assert all(request.timesteps == 5_000_000 for request in requests)
    assert all(request.parent_artifact_sha256 is None for request in requests)
    assert len(result["attempt_ledger"]) == 20

    resumed = rl_evaluation.run_independent_baseline_campaign(
        config,
        plan_path,
        tmp_path / "campaign",
        runner=lambda _request: (_ for _ in ()).throw(AssertionError("must not rerun")),
        resume=True,
    )
    assert resumed["campaign_freeze_sha256"] == result["campaign_freeze_sha256"]

    first_ledger = tmp_path / "campaign" / "ledger" / "attempt-0001.json"
    tampered = json.loads(first_ledger.read_text())
    tampered["result"]["completed_timesteps"] = 4_999_999
    first_ledger.write_text(json.dumps(tampered, sort_keys=True, separators=(",", ":")))
    with pytest.raises(EvaluationContractError, match="attempt lineage mismatch"):
        rl_evaluation.run_independent_baseline_campaign(
            config,
            plan_path,
            tmp_path / "campaign",
            runner=lambda _request: (_ for _ in ()).throw(AssertionError("must not rerun")),
            resume=True,
        )


def test_independent_campaign_freeze_recomputes_artifacts_from_full_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    campaign_root = tmp_path / "campaign"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    final_timesteps = config.training.confirmation_timesteps_per_seed
    request_identity = {
        "ordinal": 1,
        "fold": 0,
        "seed": 42,
        "variant": "independent_actor",
        "timesteps": final_timesteps,
    }
    plan = {
        "manifest_pairs": [
            {
                "fold": 0,
                "training_manifest_sha256": "1" * 64,
                "validation_manifest_sha256": "2" * 64,
            }
        ],
        "expected_append_only_ledger": [request_identity],
        "final_timesteps": final_timesteps,
        "parameters": {"learning_rate": 0.001},
        "candidate_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        rl_evaluation,
        "_validated_independent_campaign_plan",
        lambda *_args: (plan, "b" * 64),
    )
    monkeypatch.setattr(rl_evaluation, "_result_passed", lambda _result: True)
    monkeypatch.setattr(rl_evaluation, "_lineage_passed", lambda _request, _result: True)
    output = campaign_root / "runs" / "fold-0" / "seed-42"
    artifact_root = output / "checkpoint"
    artifact_root.mkdir(parents=True)
    checkpoint = artifact_root / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    bundle = artifact_root / "bundle.json"
    bundle.write_text("{}")
    training = output / "manifest.json"
    validation = output / "validation.json"
    training.write_text("{}")
    validation.write_text("{}")
    result = {
        "status": "complete",
        "finite": True,
        "action_collapsed": False,
        "blows": 0,
        "completed_timesteps": final_timesteps,
        "lineage_parent_sha256": None,
        "artifact_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_manifest_path": str(training),
        "training_manifest_sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
        "checkpoint_bundle_path": str(bundle),
        "checkpoint_bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "validation_report_path": str(validation),
        "validation_report_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
    }
    record = {
        "schema_version": 1,
        "stage": "independent-baseline-attempt-v1",
        "campaign_plan_sha256": "b" * 64,
        "request": request_identity,
        "execution_request": {
            "phase": "independent_final",
            "fold": 0,
            "training_manifest_sha256": "1" * 64,
            "validation_manifest_sha256": "2" * 64,
            "seed": 42,
            "timesteps": final_timesteps,
            "variant": "independent_actor",
            "parameters": plan["parameters"],
            "candidate_sha256": "a" * 64,
            "output": str(output),
            "resume": False,
            "parent_artifact_sha256": None,
            "required_milestone_timesteps": config.training.development_timesteps_per_seed,
        },
        "result": result,
        "test_accessed": False,
        "sealed_holdout_accessed": False,
    }
    ledger = campaign_root / "ledger" / "attempt-0001.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))
    artifact = rl_evaluation._independent_artifact_from_result(request_identity, result)
    campaign = {
        "schema_version": 1,
        "stage": "independent-baseline-campaign-v1",
        "status": "complete",
        "campaign_plan_path": str(plan_path),
        "campaign_plan_sha256": "b" * 64,
        "config_sha256": config.digest,
        "source_sha256": config.upstream.source_digest,
        "dependency_lock_sha256": config.upstream.lock_digest,
        "final_timesteps": final_timesteps,
        "artifacts": [artifact],
        "attempt_ledger": [
            {
                "path": str(ledger.resolve()),
                "sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            }
        ],
        "test_accessed": False,
        "sealed_holdout_accessed": False,
    }

    def write_campaign(payload: dict[str, object]) -> Path:
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        path = (
            campaign_root / f"independent-baseline-campaign-{hashlib.sha256(data).hexdigest()}.json"
        )
        path.write_bytes(data)
        return path

    valid_path = write_campaign(campaign)
    rl_evaluation._validated_independent_campaign(config, valid_path)
    mutated = json.loads(json.dumps(campaign))
    mutated["artifacts"][0]["timesteps"] -= 1
    with pytest.raises(EvaluationContractError, match="differ from ledger"):
        rl_evaluation._validated_independent_campaign(config, write_campaign(mutated))


def test_baseline_freeze_rejects_manifest_order_different_from_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    candidate_path = tmp_path / "candidate.json"
    campaign_path = tmp_path / "campaign.json"
    plan_path = tmp_path / "plan.json"
    for path in (candidate_path, campaign_path, plan_path):
        path.write_text("{}")
    training = [tmp_path / f"training-{fold}.json" for fold in (0, 1)]
    validation = [tmp_path / f"validation-{fold}.json" for fold in (0, 1)]
    for path in [*training, *validation]:
        path.write_text("{}")
    pairs = [
        {
            "fold": fold,
            "training_manifest_path": str(training[fold].resolve()),
            "training_manifest_sha256": hashlib.sha256(training[fold].read_bytes()).hexdigest(),
            "validation_manifest_path": str(validation[fold].resolve()),
            "validation_manifest_sha256": hashlib.sha256(validation[fold].read_bytes()).hexdigest(),
        }
        for fold in (0, 1)
    ]
    monkeypatch.setattr(rl_evaluation, "_validated_candidate", lambda *_args: ({}, "a" * 64))
    monkeypatch.setattr(
        rl_evaluation,
        "_validated_independent_campaign",
        lambda *_args: (
            {"campaign_plan_path": str(plan_path), "artifacts": []},
            "b" * 64,
        ),
    )
    monkeypatch.setattr(
        rl_evaluation,
        "_validated_independent_campaign_plan",
        lambda *_args: ({"candidate_sha256": "a" * 64, "manifest_pairs": pairs}, "c" * 64),
    )

    class Manifest:
        def __init__(self, fold: int) -> None:
            self.fold = fold

    fold_by_path = {path: fold for fold, path in enumerate(training)}
    monkeypatch.setattr(
        rl_evaluation,
        "load_episode_manifest",
        lambda _config, path: Manifest(fold_by_path[path]),
    )
    with pytest.raises(EvaluationContractError, match="differ from independent campaign"):
        rl_evaluation.freeze_evaluation_baselines(
            config,
            candidate_path,
            list(reversed(training)),
            list(reversed(validation)),
            campaign_path,
            tmp_path / "baselines",
        )


def test_baseline_validator_recomputes_hgb_and_rejects_substituted_independent_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    candidate_path = tmp_path / "candidate.json"
    campaign_path = tmp_path / "campaign.json"
    plan_path = tmp_path / "plan.json"
    training_path = tmp_path / "training.json"
    validation_path = tmp_path / "validation.json"
    historical_path = tmp_path / "historical.json"
    independent_path = tmp_path / "independent.pt"
    independent_bundle = tmp_path / "independent-bundle.json"
    independent_validation = tmp_path / "independent-validation.json"
    for path in (
        candidate_path,
        campaign_path,
        plan_path,
        training_path,
        validation_path,
        historical_path,
        independent_path,
        independent_bundle,
        independent_validation,
    ):
        path.write_text("{}")
    candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    training_sha256 = hashlib.sha256(training_path.read_bytes()).hexdigest()
    validation_sha256 = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    pair = {
        "fold": 0,
        "training_manifest_path": str(training_path.resolve()),
        "training_manifest_sha256": training_sha256,
        "validation_manifest_path": str(validation_path.resolve()),
        "validation_manifest_sha256": validation_sha256,
    }
    monkeypatch.setattr(
        rl_evaluation,
        "_validated_candidate",
        lambda *_args: ({}, candidate_sha256),
    )

    class Manifest:
        def __init__(self, partition: str) -> None:
            self.partition = partition
            self.fold = 0

    manifests = {
        training_path: Manifest("training"),
        validation_path: Manifest("validation"),
    }
    monkeypatch.setattr(
        rl_evaluation, "load_episode_manifest", lambda _config, path: manifests[path]
    )
    input_lineage = {
        "training_features": {"shape": [4, 2], "dtype": "<f8", "sha256": "1" * 64},
        "training_labels": {"shape": [4], "dtype": "|i1", "sha256": "2" * 64},
        "validation_features": {"shape": [4, 2], "dtype": "<f8", "sha256": "3" * 64},
        "validation_labels": {"shape": [4], "dtype": "|i1", "sha256": "4" * 64},
    }
    hgb_contract = {
        "estimator_schema": {
            "format": "python_pickle",
            "pickle_protocol": 5,
            "module": "sklearn.ensemble._hist_gradient_boosting.gradient_boosting",
            "class": "HistGradientBoostingClassifier",
            "library": "scikit-learn",
            "library_version": rl_evaluation.sklearn.__version__,
        },
        "fit_code_path": str(ROOT / "src" / "mantis_v2" / "rl_baselines.py"),
        "fit_code_sha256": "5" * 64,
        "fit_evidence": {
            "training_rows": 4,
            "validation_rows": 4,
            "threshold_source": "validation",
            "threshold_selection": "maximum_balanced_accuracy_then_highest_threshold_v1",
            "threshold": 0.75,
            "hyperparameters": {"max_iter": 100},
            "random_state": config.run.seed,
        },
        "threshold_selection": {
            "source": "validation",
            "method": "maximum_balanced_accuracy_then_highest_threshold_v1",
            "threshold": 0.75,
            "validation_features_sha256": "3" * 64,
            "validation_labels_sha256": "4" * 64,
        },
        "input_lineage": input_lineage,
        "test_labels_accessed": False,
    }
    estimator_bytes = b"deterministic-estimator-v1"
    monkeypatch.setattr(
        rl_evaluation,
        "_hgb_freeze_material",
        lambda *_args: (estimator_bytes, input_lineage, hgb_contract),
    )
    historical_contract = {
        "path": str(historical_path.resolve()),
        "sha256": hashlib.sha256(historical_path.read_bytes()).hexdigest(),
        "threshold": 0.6,
        "estimator_schema": {
            "format": "portable_rejected_logistic_head",
            "class": "HistoricalLogisticPolicy",
        },
        "source_code_path": str(ROOT / "src" / "mantis_v2" / "rl_validation.py"),
        "source_code_sha256": "6" * 64,
    }
    monkeypatch.setattr(
        rl_evaluation,
        "_historical_logistic_contract",
        lambda *_args: historical_contract,
    )
    estimator_sha256 = hashlib.sha256(estimator_bytes).hexdigest()
    estimator_path = tmp_path / f"hgb-fold-0-{estimator_sha256}.pkl"
    estimator_path.write_bytes(estimator_bytes)
    independent_sha256 = hashlib.sha256(independent_path.read_bytes()).hexdigest()
    independent_bundle_sha256 = hashlib.sha256(independent_bundle.read_bytes()).hexdigest()
    independent_validation_sha256 = hashlib.sha256(independent_validation.read_bytes()).hexdigest()
    campaign_artifact = {
        "fold": 0,
        "seed": 42,
        "variant": "independent_actor",
        "timesteps": config.training.confirmation_timesteps_per_seed,
        "checkpoint_path": str(independent_path),
        "checkpoint_sha256": independent_sha256,
        "training_manifest_path": str(training_path),
        "training_manifest_sha256": training_sha256,
        "checkpoint_bundle_path": str(independent_bundle),
        "checkpoint_bundle_sha256": independent_bundle_sha256,
        "validation_report_path": str(independent_validation),
        "validation_report_sha256": independent_validation_sha256,
        "finite": True,
        "action_collapsed": False,
        "blows": 0,
    }
    campaign_plan = {"manifest_pairs": [pair], "parameters": {"learning_rate": 0.001}}
    campaign = {
        "campaign_plan_path": str(plan_path),
        "config_sha256": config.digest,
        "source_sha256": config.upstream.source_digest,
        "dependency_lock_sha256": config.upstream.lock_digest,
        "final_timesteps": config.training.confirmation_timesteps_per_seed,
        "artifacts": [campaign_artifact],
    }
    monkeypatch.setattr(
        rl_evaluation,
        "_validated_independent_campaign",
        lambda *_args: (campaign, "c" * 64),
    )
    monkeypatch.setattr(
        rl_evaluation,
        "_validated_independent_campaign_plan",
        lambda *_args: (campaign_plan, "p" * 64),
    )
    fold = {
        **pair,
        **input_lineage,
        "historical_logistic": historical_contract,
        "hist_gradient_boosting": {
            "path": str(estimator_path.resolve()),
            "sha256": estimator_sha256,
            **hgb_contract,
        },
        "independent_ppo": [
            rl_evaluation._independent_ppo_baseline_contract(
                campaign_artifact, campaign, "c" * 64, campaign_plan, "p" * 64
            )
        ],
    }
    evaluation_code = ROOT / "src" / "mantis_v2" / "rl_evaluation.py"
    environment_code = ROOT / "src" / "mantis_v2" / "rl_environment.py"
    baseline_code = ROOT / "src" / "mantis_v2" / "rl_baselines.py"
    baseline = {
        "schema_version": 1,
        "stage": "baseline-freeze-v1",
        "status": "complete",
        "config_sha256": config.digest,
        "source_sha256": config.upstream.source_digest,
        "dependency_lock_sha256": config.upstream.lock_digest,
        "architecture_candidate_path": str(candidate_path),
        "architecture_candidate_sha256": candidate_sha256,
        "independent_campaign_path": str(campaign_path),
        "independent_campaign_sha256": "c" * 64,
        "evaluation_code_path": str(evaluation_code),
        "evaluation_code_sha256": hashlib.sha256(evaluation_code.read_bytes()).hexdigest(),
        "environment_code_path": str(environment_code),
        "environment_code_sha256": hashlib.sha256(environment_code.read_bytes()).hexdigest(),
        "baseline_code_path": str(baseline_code),
        "baseline_code_sha256": hashlib.sha256(baseline_code.read_bytes()).hexdigest(),
        "folds": [fold],
        "test_accessed": False,
        "sealed_holdout_accessed": False,
    }

    def write_baseline(payload: dict[str, object]) -> Path:
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        path = tmp_path / f"baseline-freeze-{hashlib.sha256(data).hexdigest()}.json"
        path.write_bytes(data)
        return path

    rl_evaluation._validated_baseline_freeze(config, write_baseline(baseline))
    mutations = {
        "source_identity": lambda payload: payload.update({"baseline_code_sha256": "0" * 64}),
        "fit_evidence": lambda payload: payload["folds"][0]["hist_gradient_boosting"][
            "fit_evidence"
        ].update({"training_rows": 5}),
        "estimator_schema": lambda payload: payload["folds"][0]["hist_gradient_boosting"][
            "estimator_schema"
        ].update({"pickle_protocol": 4}),
        "threshold_selection": lambda payload: payload["folds"][0]["hist_gradient_boosting"][
            "threshold_selection"
        ].update({"threshold": 0.5}),
        "input_lineage": lambda payload: payload["folds"][0]["training_features"].update(
            {"sha256": "9" * 64}
        ),
    }
    for mutate in mutations.values():
        mutated = json.loads(json.dumps(baseline))
        mutate(mutated)
        with pytest.raises(EvaluationContractError):
            rl_evaluation._validated_baseline_freeze(config, write_baseline(mutated))

    altered_bytes = b"altered-readdressed-estimator"
    altered_sha256 = hashlib.sha256(altered_bytes).hexdigest()
    altered_path = tmp_path / f"hgb-fold-0-{altered_sha256}.pkl"
    altered_path.write_bytes(altered_bytes)
    altered = json.loads(json.dumps(baseline))
    altered_hgb = altered["folds"][0]["hist_gradient_boosting"]
    altered_hgb["path"] = str(altered_path)
    altered_hgb["sha256"] = altered_sha256
    with pytest.raises(EvaluationContractError, match="HGB baseline recomputation mismatch"):
        rl_evaluation._validated_baseline_freeze(config, write_baseline(altered))

    alternate_checkpoint = tmp_path / "alternate-independent.pt"
    alternate_report = tmp_path / "alternate-independent-validation.json"
    alternate_checkpoint.write_bytes(b"valid-looking-alternate-checkpoint")
    alternate_report.write_text('{"status":"complete","partition":"validation"}')
    substituted = json.loads(json.dumps(baseline))
    substituted_entry = substituted["folds"][0]["independent_ppo"][0]
    substituted_checkpoint_sha256 = hashlib.sha256(alternate_checkpoint.read_bytes()).hexdigest()
    substituted_report_sha256 = hashlib.sha256(alternate_report.read_bytes()).hexdigest()
    substituted_entry["path"] = str(alternate_checkpoint)
    substituted_entry["sha256"] = substituted_checkpoint_sha256
    substituted_entry["validation_report_path"] = str(alternate_report)
    substituted_entry["validation_report_sha256"] = substituted_report_sha256
    substituted_entry["campaign_artifact"]["checkpoint_path"] = str(alternate_checkpoint)
    substituted_entry["campaign_artifact"]["checkpoint_sha256"] = substituted_checkpoint_sha256
    substituted_entry["campaign_artifact"]["validation_report_path"] = str(alternate_report)
    substituted_entry["campaign_artifact"]["validation_report_sha256"] = substituted_report_sha256
    with pytest.raises(
        EvaluationContractError,
        match="independent PPO baseline differs from validated campaign artifacts",
    ):
        rl_evaluation._validated_baseline_freeze(config, write_baseline(substituted))


def test_evaluation_writes_durable_terminal_failure_before_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    plan_path = tmp_path / "evaluation-plan.json"
    plan_path.write_text("{}")
    (tmp_path / "test.json").write_text('{"fold":0}')
    monkeypatch.setattr(
        rl_evaluation,
        "_validated_evaluation_plan",
        lambda *_args: (
            {
                "run_identity": "fixed-test-v1",
                "serving_freeze_path": str(tmp_path / "serving.json"),
                "serving_freeze_sha256": "b" * 64,
                "test_schedules": [{"path": str(tmp_path / "test.json"), "sha256": "c" * 64}],
                "expected_append_only_ledger": [
                    {
                        "ordinal": 1,
                        "fold": 0,
                        "seed": 42,
                        "policy": "candidate",
                        "stress": "primary",
                    }
                ],
                "stress_definitions": {},
            },
            "a" * 64,
        ),
    )

    def fail(_request: EvaluationRequest) -> list[dict[str, object]]:
        raise RuntimeError("deterministic replay failed")

    with pytest.raises(EvaluationContractError, match="fold 0 seed 42 candidate primary"):
        run_topstep_evaluation(config, plan_path, tmp_path / "evaluation", runner=fail)

    failures = list((tmp_path / "evaluation" / "failures").glob("terminal-failure-*.json"))
    assert len(failures) == 1
    payload = json.loads(failures[0].read_text())
    assert payload["status"] == "not_promoted"
    assert payload["exception_message"] == "deterministic replay failed"
    assert payload["sealed_holdout_accessed"] is False
    assert (tmp_path / "evaluation" / "ledger" / "attempt-0001.json").is_file()
    verdicts = list((tmp_path / "evaluation").glob("promotion-verdict-*.json"))
    assert len(verdicts) == 1

    def must_not_retry(_request: EvaluationRequest) -> list[dict[str, object]]:
        raise AssertionError("terminal replay was retried")

    resumed = run_topstep_evaluation(
        config,
        plan_path,
        tmp_path / "evaluation",
        runner=must_not_retry,
        resume=True,
    )
    assert resumed["status"] == "not_promoted"
    assert resumed["verdict_path"] == str(verdicts[0])


def test_post_replay_estimator_failure_writes_durable_not_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    plan_path = tmp_path / "evaluation-plan.json"
    schedule_path = tmp_path / "test.json"
    plan_path.write_text("{}")
    schedule_path.write_text('{"fold":0}')
    plan = {
        "run_identity": "fixed-test-v1",
        "serving_freeze_path": str(tmp_path / "serving.json"),
        "serving_freeze_sha256": "b" * 64,
        "test_schedules": [{"path": str(schedule_path), "sha256": "c" * 64}],
        "expected_append_only_ledger": [
            {
                "ordinal": 1,
                "fold": 0,
                "seed": 42,
                "policy": "candidate",
                "stress": "primary",
            }
        ],
        "stress_definitions": {},
    }
    monkeypatch.setattr(
        rl_evaluation, "_validated_evaluation_plan", lambda *_args: (plan, "a" * 64)
    )
    monkeypatch.setattr(
        rl_evaluation,
        "evaluate_topstep_promotion",
        lambda *_args: (_ for _ in ()).throw(EvaluationContractError("degenerate bootstrap")),
    )

    with pytest.raises(EvaluationContractError, match="finalization failed"):
        run_topstep_evaluation(
            config,
            plan_path,
            tmp_path / "evaluation",
            runner=lambda _request: [_attempt(0)],
        )

    failure = json.loads(
        next((tmp_path / "evaluation" / "failures").glob("terminal-failure-*.json")).read_text()
    )
    assert failure["failure_phase"] == "post_replay_estimation_or_report"
    verdict_path = next((tmp_path / "evaluation").glob("promotion-verdict-*.json"))
    verdict = json.loads(verdict_path.read_text())
    assert verdict["status"] == "not_promoted"
    assert verdict["failed_clauses"] == ["post_replay_non_estimable_or_report_failure"]

    resumed = run_topstep_evaluation(
        config,
        plan_path,
        tmp_path / "evaluation",
        runner=lambda _request: (_ for _ in ()).throw(AssertionError("must not retry")),
        resume=True,
    )
    assert resumed["verdict_path"] == str(verdict_path)


def test_successful_evaluation_report_serializes_transfer_cell_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    serving = tmp_path / "serving.json"
    replay = tmp_path / "replay.json"
    serving.write_text("{}")
    replay.write_text("{}")
    cells = [
        (ticker, profile)
        for ticker in ("ES", "NQ", "RTY", "YM", "GC", "CL", "ZB")
        for profile in (("one_mini",) if ticker == "ZB" else ("one_mini", "ten_micros"))
    ]
    rows: list[dict[str, object]] = []
    for stress in (
        "primary",
        "two_tick",
        "fee_2x",
        "gap_adverse_tick",
        "latency_1bar",
        "missed_fill_10pct",
        "overlapping_starts",
        "same_bar_adverse",
    ):
        for seed in range(42, 52):
            for block in range(20):
                for ticker, profile in cells:
                    rows.append(
                        {
                            "policy": "candidate",
                            "stress": stress,
                            "seed": seed,
                            "ticker": ticker,
                            "profile": profile,
                            "calendar_block": f"2025-W{block + 1:02d}",
                            "status": "PASS",
                        }
                    )
    monkeypatch.setattr(
        rl_evaluation,
        "_validate_serving_freeze",
        lambda *_args: ({}, "b" * 64),
    )
    monkeypatch.setattr(
        rl_evaluation,
        "_validated_replay",
        lambda *_args: (
            {
                "evaluation_plan_sha256": "a" * 64,
                "test_schedules": ["c" * 64],
                "mechanics_sha256": "d" * 64,
            },
            rows,
        ),
    )
    summary = {
        "attempt_count": 300,
        "outcomes": {"PASS": 300, "BLOW": 0, "TIMEOUT": 0},
        "rates": {
            "pass": 1.0,
            "pass_wilson_lcb_95": 0.9,
            "blow_wilson_ucb_95": 0.005,
        },
    }
    monkeypatch.setattr(
        rl_evaluation,
        "_group_summaries",
        lambda _rows, **_kwargs: {
            "pooled": summary,
            "profile": [
                {"profile": "one_mini", "summary": summary},
                {"profile": "ten_micros", "summary": summary},
            ],
        },
    )
    bootstrap_matrix_ids: set[int] = set()

    def fixed_effect(*args: object) -> dict[str, object]:
        bootstrap_matrix_ids.add(id(args[-1]))
        return {
            "point_difference": 0.01,
            "lcb_95": 0.001,
            "index_matrix_sha256": "e" * 64,
        }

    monkeypatch.setattr(rl_evaluation, "_point_and_lcb", fixed_effect)
    result_differences = [
        {
            "baseline": baseline,
            "views": {
                view: []
                for view in (
                    "pooled",
                    "fold",
                    "ticker",
                    "profile",
                    "regime_block",
                    "seed",
                    "cell",
                )
            },
        }
        for baseline in sorted(rl_evaluation._POLICIES - {"candidate"})
    ]
    monkeypatch.setattr(
        rl_evaluation,
        "_baseline_result_differences",
        lambda _rows, _stress: result_differences,
    )
    monkeypatch.setattr(
        rl_evaluation,
        "promotion_verdict",
        lambda **_kwargs: {"promoted": True, "failed_clauses": []},
    )

    result = rl_evaluation.evaluate_topstep_promotion(config, serving, replay, tmp_path / "report")

    payload = json.loads(Path(result["report_path"]).read_text())
    transfer = payload["numeric_gate_evidence"]["primary"]["transfer_effects"]
    assert len(transfer) == len(cells) + 1
    assert {"ticker", "profile", "point_difference", "lcb_95"} <= set(transfer[0])
    assert payload["stress_metrics"]["primary"]["result_differences"] == result_differences
    assert (
        payload["stress_metrics"]["overlapping_starts"]["result_differences"] == result_differences
    )
    assert len(bootstrap_matrix_ids) == 1


def test_default_production_runner_replays_real_environment_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    first = datetime(2025, 1, 6, 15, 0, tzinfo=UTC)
    candidate = CandidateData(
        embedding=np.array([1.0, -1.0], dtype=np.float32),
        direction=1,
        trend_line=99.0,
        atr=2.0,
        bars_since_direction_change=2,
        label=1,
    )
    episode = EnvironmentEpisode(
        "ES",
        "one_mini",
        (
            BarData(first, 100.0, 101.0, 99.0, 100.0, 10.0, candidate),
            BarData(
                first + timedelta(minutes=3),
                100.0,
                101.0,
                99.0,
                100.0,
                10.0,
                candidate,
                discontinuity=True,
            ),
            BarData(
                first + timedelta(minutes=6),
                100.0,
                101.0,
                99.0,
                100.0,
                10.0,
                candidate,
            ),
            BarData(first + timedelta(minutes=9), 100.0, 101.0, 99.0, 100.0, 10.0),
        ),
    )
    schedule = tmp_path / "schedule.json"
    serving = tmp_path / "serving.json"
    baseline = tmp_path / "baseline.json"
    for path in (schedule, serving, baseline):
        path.write_text("{}")
    objects = {
        serving: {"seed_artifacts": []},
        baseline: {"folds": [{"fold": 0}]},
        schedule: {"episodes": [{"number": 7}]},
    }
    monkeypatch.setattr(
        rl_evaluation,
        "_load_object",
        lambda path, _description: objects[path],
    )
    monkeypatch.setattr(
        rl_evaluation,
        "load_test_episode_manifest",
        lambda *_args, **_kwargs: LoadedEpisodes(
            episodes=(episode,),
            feature_refs=(),
            manifest_sha256="c" * 64,
            partition="test",
            fold=0,
        ),
    )
    runner = rl_evaluation._production_evaluation_runner(
        config,
        {
            "serving_freeze_path": str(serving),
            "baseline_freeze_path": str(baseline),
            "stress_definitions": {
                "primary": {
                    "adverse_slippage_ticks_per_side": 1.0,
                    "fee": "pinned_product_round_turn_snapshot",
                },
                "latency_1bar": {"base": "primary", "entry_defer_eligible_bars": 1},
            },
        },
    )
    rows = runner(
        EvaluationRequest(
            ordinal=1,
            fold=0,
            seed=42,
            policy="reject_all",
            stress="primary",
            evaluation_plan_sha256="a" * 64,
            test_manifest_path=schedule,
            test_manifest_sha256="c" * 64,
            output=tmp_path / "run",
        )
    )
    assert len(rows) == 1
    assert rows[0]["episode_id"] == "7"
    assert rows[0]["accepted_trades"] == 0
    assert rows[0]["calendar_block"] == "2025-W02"
    assert rows[0]["stress_invariants_valid"] is True

    latency_rows = runner(
        EvaluationRequest(
            ordinal=2,
            fold=0,
            seed=42,
            policy="take_all",
            stress="latency_1bar",
            evaluation_plan_sha256="a" * 64,
            test_manifest_path=schedule,
            test_manifest_sha256="c" * 64,
            output=tmp_path / "latency-run",
        )
    )
    assert latency_rows[0]["latency_cancellations"] == 2
    assert first.isoformat() not in latency_rows[0]["accepted_opportunity_ids"]


def test_production_runner_covers_policy_stress_and_cost_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    first = datetime(2025, 1, 6, 15, 0, tzinfo=UTC)
    candidate = CandidateData(
        embedding=np.array([1.0, -1.0], dtype=np.float32),
        direction=1,
        trend_line=99.0,
        atr=2.0,
        bars_since_direction_change=2,
        label=1,
    )
    episode = EnvironmentEpisode(
        "ES",
        "one_mini",
        (
            BarData(first, 100.0, 101.0, 99.0, 100.0, 10.0, candidate),
            BarData(first + timedelta(minutes=3), 100.5, 102.0, 100.0, 101.5, 11.0),
            BarData(first + timedelta(minutes=6), 101.5, 102.0, 101.0, 101.75, 12.0),
        ),
    )
    schedule = tmp_path / "schedule.json"
    serving = tmp_path / "serving.json"
    baseline = tmp_path / "baseline.json"
    hgb_path = tmp_path / "hgb.pkl"
    candidate_attempt = tmp_path / "candidate-attempt.json"
    checkpoint_bundle = tmp_path / "bundle.json"
    independent_checkpoint = tmp_path / "independent.pt"
    for path in (
        schedule,
        serving,
        baseline,
        checkpoint_bundle,
        independent_checkpoint,
    ):
        path.write_text("{}")
    hgb_path.write_bytes(b"frozen-hgb")
    plan_sha256 = "a" * 64
    objects = {
        serving: {
            "seed_artifacts": [
                {
                    "fold": 0,
                    "seed": 42,
                    "checkpoint_bundle_path": str(checkpoint_bundle),
                    "artifact_sha256": "1" * 64,
                }
            ]
        },
        baseline: {
            "folds": [
                {
                    "fold": 0,
                    "independent_ppo": [
                        {
                            "seed": 42,
                            "path": str(independent_checkpoint),
                            "sha256": "2" * 64,
                        }
                    ],
                    "hist_gradient_boosting": {
                        "path": str(hgb_path),
                        "sha256": hashlib.sha256(hgb_path.read_bytes()).hexdigest(),
                        "fit_evidence": {"threshold": 0.5},
                    },
                }
            ]
        },
        schedule: {"episodes": [{"number": 7}]},
        candidate_attempt: {
            "stage": "evaluation-attempt-v1",
            "status": "complete",
            "evaluation_plan_sha256": plan_sha256,
            "rows": [{"episode_id": "7", "accepted_trades": 1}],
        },
    }
    monkeypatch.setattr(rl_evaluation, "_load_object", lambda path, _description: objects[path])
    monkeypatch.setattr(
        rl_evaluation,
        "load_test_episode_manifest",
        lambda *_args, **_kwargs: LoadedEpisodes(
            episodes=(episode,),
            feature_refs=(),
            manifest_sha256="c" * 64,
            partition="test",
            fold=0,
        ),
    )

    class AlwaysTakeActor:
        def __call__(self, observation: torch.Tensor, *_args: object):
            return (
                torch.tensor([[0.0, 1.0]], dtype=observation.dtype),
                torch.tensor([0.0], dtype=observation.dtype),
            )

    class AlwaysTakeHistorical:
        def action(self, _observation: object, _mask: np.ndarray) -> int:
            return 1

    class AlwaysTakeHgb:
        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.0, 1.0]]), (len(features), 1))

    monkeypatch.setattr(rl_evaluation, "_loaded_actor", lambda *_args: AlwaysTakeActor())
    monkeypatch.setattr(
        rl_evaluation,
        "historical_logistic_artifact",
        lambda *_args: (AlwaysTakeHistorical(), tmp_path / "historical.json", "3" * 64),
    )
    monkeypatch.setattr(rl_evaluation.pickle, "load", lambda _handle: AlwaysTakeHgb())
    runner = rl_evaluation._production_evaluation_runner(
        config,
        {
            "serving_freeze_path": str(serving),
            "baseline_freeze_path": str(baseline),
            "stress_definitions": {
                "primary": {
                    "adverse_slippage_ticks_per_side": 1.0,
                    "fee": "pinned_product_round_turn_snapshot",
                },
                "gap_adverse_tick": {"base": "primary", "historical_adverse_gap_ticks": 1},
                "missed_fill_10pct": {
                    "base": "primary",
                    "cancel_without_carry_every": 10,
                },
            },
        },
    )

    for ordinal, policy in enumerate(
        (
            "take_all",
            "candidate",
            "independent_ticker_ppo",
            "historical_rejected_logistic_head",
            "hist_gradient_boosting",
        ),
        start=1,
    ):
        row = runner(
            EvaluationRequest(
                ordinal=ordinal,
                fold=0,
                seed=42,
                policy=policy,
                stress="primary",
                evaluation_plan_sha256=plan_sha256,
                test_manifest_path=schedule,
                test_manifest_sha256="c" * 64,
                output=tmp_path / policy,
            )
        )[0]
        assert row["accepted_trades"] == 1
        assert row["commission_costs"] == pytest.approx(config.fees.es)
        assert row["slippage_costs"] == pytest.approx(25.0)
        assert row["gap_costs"] == 0.0
        assert row["costs"] == pytest.approx(
            cast(float, row["commission_costs"]) + cast(float, row["slippage_costs"])
        )
        assert row["accepted_opportunity_ids"] == [first.isoformat()]

    matched = runner(
        EvaluationRequest(
            ordinal=20,
            fold=0,
            seed=42,
            policy="matched_random_take",
            stress="primary",
            evaluation_plan_sha256=plan_sha256,
            test_manifest_path=schedule,
            test_manifest_sha256="c" * 64,
            output=tmp_path / "matched",
            candidate_attempt_path=candidate_attempt,
        )
    )[0]
    assert matched["accepted_trades"] == 1

    gap = runner(
        EvaluationRequest(
            ordinal=21,
            fold=0,
            seed=42,
            policy="take_all",
            stress="gap_adverse_tick",
            evaluation_plan_sha256=plan_sha256,
            test_manifest_path=schedule,
            test_manifest_sha256="c" * 64,
            output=tmp_path / "gap",
        )
    )[0]
    assert gap["gap_adjusted_fills"] == 1
    assert gap["gap_costs"] == pytest.approx(12.5)

    pair_key = rl_evaluation._canonical_pair_key(0, "2025-W02", "7", "ES", "one_mini", 42)
    missed_plan = next(
        f"{value:064x}"
        for value in range(10_000)
        if int.from_bytes(
            hashlib.sha256(f"{value:064x}{pair_key}:missed-fill-v1".encode()).digest()[:8],
            "big",
        )
        % 10
        == 0
    )
    missed = runner(
        EvaluationRequest(
            ordinal=22,
            fold=0,
            seed=42,
            policy="take_all",
            stress="missed_fill_10pct",
            evaluation_plan_sha256=missed_plan,
            test_manifest_path=schedule,
            test_manifest_sha256="c" * 64,
            output=tmp_path / "missed",
        )
    )[0]
    assert missed["accepted_trades"] == 0
    assert missed["missed_fill_count"] == 1


def test_cli_exposes_frozen_topstep_evaluation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = object()
    expected = {"status": "not_promoted"}
    monkeypatch.setattr(cli, "load_rl_config", lambda _path: sentinel)
    monkeypatch.setattr(cli, "run_topstep_evaluation", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mantis-v2",
            "rl-evaluate-topstep",
            "--config",
            "rl.toml",
            "--plan",
            "plan.json",
            "--output",
            "evaluation",
            "--resume",
        ],
    )

    cli.main()

    assert json.loads(capsys.readouterr().out) == expected
