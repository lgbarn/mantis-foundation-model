"""Contract tests for frozen architecture and seed confirmation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mantis_v2 import cli, rl_confirmation
from mantis_v2.rl_config import load_rl_config
from mantis_v2.rl_confirmation import (
    ConfirmationError,
    decide_continuation,
    freeze_architecture_plan,
    qualify_architecture,
    run_architecture_ablation,
    run_production_seed_campaign,
    run_seed_confirmation,
)
from mantis_v2.rl_policy import PROFILES, TICKERS, EntryActorCritic, PolicyVariant
from mantis_v2.rl_training import PpoHyperparameters, _canonical_digest

ROOT = Path(__file__).resolve().parents[1]
_REAL_ARCHITECTURE_REPLAY = rl_confirmation._replay_architecture_rows


@pytest.fixture(autouse=True)
def _stub_large_architecture_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep statistical fixtures small; dedicated tests exercise the real replay path."""
    monkeypatch.setattr(
        rl_confirmation,
        "_replay_architecture_rows",
        lambda _config, report, _checkpoint: (
            list(report["rows"]),
            dict(report["policy_diagnostics"]),
        ),
    )


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def _calendar_quarter(year: int, week: int) -> str:
    month = datetime.fromisocalendar(year, week, 1).month
    return f"calendar-quarter-{(month - 1) // 3 + 1}"


def _winner(path: Path) -> Path:
    parameters = {
        "learning_rate": 0.0003,
        "rollout_length": 14,
        "batch_size": 256,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "entropy_coefficient": 0.01,
        "value_loss_coefficient": 0.5,
        "max_grad_norm": 0.5,
        "hidden_width": 64,
    }
    return _write(
        path,
        {
            "schema_version": 1,
            "stage": "rl-optuna-winner",
            "study_name": "frozen-v1",
            "study_identity_sha256": "a" * 64,
            "trial_number": 7,
            "parameters": parameters,
            "parameters_sha256": hashlib.sha256(
                json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "validation": {"aggregate_blows": 0},
            "sealed_holdout_accessed": False,
        },
    )


def _evidence(path: Path, winner: Path, *, candidate_passes: int = 9) -> Path:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    training = path.parent / "training-schedule.json"
    validation = path.parent / "validation-schedule.json"
    for manifest, partition, start, end in (
        (
            training,
            "training",
            "2025-01-01T00:00:00+00:00",
            "2025-06-30T23:59:59+00:00",
        ),
        (
            validation,
            "validation",
            "2025-07-01T00:00:00+00:00",
            "2025-12-31T23:59:59+00:00",
        ),
    ):
        _write(
            manifest,
            {
                "schema_version": 1,
                "stage": "rl-episode-schedule",
                "fold": 0,
                "partition": {"name": partition, "start": start, "end": end},
                "identities": {"config": {"sha256": config.digest}},
            },
        )
    plan = freeze_architecture_plan(
        config,
        winner,
        training,
        validation,
        path.parent / "plans",
        created_at="2026-07-22T12:00:00+00:00",
        runtime_identities={
            "source": {
                "revision": "test-fixture",
                "dirty": False,
                "sha256": config.upstream.source_digest,
            },
            "lock": {"sha256": config.upstream.lock_digest},
        },
    )
    rows = []
    for variant in (
        PolicyVariant.INDEPENDENT_ACTOR.value,
        PolicyVariant.SHARED_CRITIC.value,
        PolicyVariant.SHARED_TICKER_VALUE.value,
    ):
        for seed in (42, 43, 44, 45, 46):
            for ticker in TICKERS:
                profiles = ("one_mini",) if ticker == "ZB" else PROFILES
                for profile in profiles:
                    for block_index in range(1, 21):
                        passes = 8
                        if variant == PolicyVariant.SHARED_TICKER_VALUE.value:
                            passes = candidate_passes + int(block_index % 2 == 0)
                        for episode in range(10):
                            rows.append(
                                {
                                    "variant": variant,
                                    "fold": 0,
                                    "seed": seed,
                                    "ticker": ticker,
                                    "profile": profile,
                                    "regime": _calendar_quarter(2025, block_index),
                                    "calendar_block": f"2025-W{block_index:02d}",
                                    "episode_id": f"{ticker}-{profile}-{block_index:02d}-{episode}",
                                    "outcome": "PASS" if episode < passes else "TIMEOUT",
                                    "finite": True,
                                    "action_collapsed": False,
                                }
                            )
    reports = []
    for variant in PolicyVariant:
        for seed in config.training.development_seeds:
            report_rows = [
                row for row in rows if row["variant"] == variant.value and row["seed"] == seed
            ]
            run_root = path.parent / "runs" / variant.value / f"seed-{seed}"
            checkpoint = run_root / "checkpoints" / "update-000001" / "checkpoint.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            parameters = json.loads(winner.read_text())["parameters"]
            hyperparameters = asdict(PpoHyperparameters.from_search(parameters))
            identities = {
                "completion_mode": "production_timesteps",
                "target_timesteps": config.training.development_timesteps_per_seed,
                "target_updates": None,
                "config_sha256": config.digest,
                "schedule_sha256": plan["manifest_pairs"][0]["training_manifest_sha256"],
                "fold": 0,
                "seed": seed,
                "variant": variant.value,
                "ppo_hyperparameters": hyperparameters,
                "partition": "training",
                "source_sha256": config.upstream.source_digest,
                "dependency_lock_sha256": config.upstream.lock_digest,
            }
            checkpoint_payload = {
                "schema_version": 3,
                "identities": identities,
                "update": 1,
                "episode_cursor": 0,
                "model": EntryActorCritic(
                    42, variant, hidden_width=hyperparameters["hidden_width"]
                ).state_dict(),
                "optimizer": {"state": {}},
                "normalizers": {},
                "constraint_controller": {},
                "rng": {"torch": torch.zeros(1, dtype=torch.uint8)},
                "metrics": [{"timesteps": 2_000_000, "pass_rate": 0.8}],
            }
            torch.save(checkpoint_payload, checkpoint)
            bundle = checkpoint.parent / "bundle.json"
            _write(
                bundle,
                {
                    "schema_version": 3,
                    "update": 1,
                    "episode_cursor": 0,
                    "identities": identities,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                },
            )
            training_report = run_root / "manifest.json"
            _write(
                training_report,
                {
                    "schema_version": 2,
                    "stage": "rl-train",
                    "status": "complete",
                    "fold": 0,
                    "seed": seed,
                    "variant": variant.value,
                    "finite_gradients": True,
                    "deterministic_reload_actions": True,
                    "minimum_target_timesteps": config.training.development_timesteps_per_seed,
                    "completed_timesteps": config.training.development_timesteps_per_seed,
                    "ppo_hyperparameters": hyperparameters,
                    "identities": identities,
                    "sealed_holdout_accessed": False,
                },
            )
            report = {
                "schema_version": 1,
                "stage": "rl-architecture-validation-report",
                "plan_sha256": plan["plan_sha256"],
                "variant": variant.value,
                "seed": seed,
                "fold": 0,
                "training_manifest_path": str(training_report),
                "training_manifest_sha256": hashlib.sha256(
                    training_report.read_bytes()
                ).hexdigest(),
                "checkpoint_bundle_path": str(bundle),
                "checkpoint_bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "validation_manifest_path": str(validation),
                "validation_manifest_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
                "evaluator_code_path": str(ROOT / "src" / "mantis_v2" / "rl_confirmation.py"),
                "evaluator_code_sha256": hashlib.sha256(
                    (ROOT / "src" / "mantis_v2" / "rl_confirmation.py").read_bytes()
                ).hexdigest(),
                "partition": "validation",
                "rows": report_rows,
                "learning_curve": [{"timesteps": 2_000_000, "pass_rate": 0.8}],
                "policy_diagnostics": {"action_collapse_detected": False},
                "test_accessed": False,
                "sealed_holdout_accessed": False,
            }
            data = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
            digest = hashlib.sha256(data).hexdigest()
            report_path = path.parent / "reports" / f"report-{digest}.json"
            report_path.parent.mkdir(exist_ok=True)
            report_path.write_bytes(data)
            reports.append({"path": str(report_path), "sha256": digest})
    return _write(
        path,
        {
            "schema_version": 1,
            "stage": "rl-architecture-ablation-evidence",
            "partition": "validation",
            "config_sha256": config.digest,
            "optuna_winner_sha256": hashlib.sha256(winner.read_bytes()).hexdigest(),
            "plan_path": plan["plan_path"],
            "plan_sha256": plan["plan_sha256"],
            "created_at": "2026-07-22T12:00:00+00:00",
            "folds": [0],
            "training_schedule_sha256": plan["training_manifest_sha256"],
            "validation_schedule_sha256": plan["validation_manifest_sha256"],
            "schedule_sha256": plan["validation_manifest_sha256"],
            "reports": reports,
            "sealed_holdout_accessed": False,
        },
    )


def _campaign_result(request) -> dict[str, object]:
    request.output.mkdir(parents=True, exist_ok=True)
    identities = {
        "fold": request.fold,
        "seed": request.seed,
        "variant": request.variant,
        "campaign_phase": request.phase,
    }
    components = {
        "model": {"weight": torch.zeros(1)},
        "optimizer": {"state": {}},
        "rng": {"torch": torch.zeros(1, dtype=torch.uint8)},
        "normalizers": {},
        "constraint_controller": {},
        "metrics": [{"completed_timesteps": request.timesteps}],
    }
    checkpoint_payload = {
        "schema_version": 3,
        "identities": identities,
        "update": 1,
        "episode_cursor": 0,
        **components,
    }
    checkpoint = request.output / "checkpoint.pt"
    torch.save(checkpoint_payload, checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    bundle = _write(
        request.output / "checkpoint_bundle.json",
        {
            "schema_version": 3,
            "identities": identities,
            "update": 1,
            "episode_cursor": 0,
            "checkpoint_sha256": checkpoint_sha256,
        },
    )
    manifest = _write(
        request.output / "training_manifest.json",
        {
            "schema_version": 2,
            "stage": "rl-train",
            "status": "complete",
            "identities": identities,
            "sealed_holdout_accessed": False,
        },
    )
    rows = []
    for ticker in TICKERS:
        profiles = ("one_mini",) if ticker == "ZB" else PROFILES
        for profile in profiles:
            for block_index in range(1, 21):
                rows.append(
                    {
                        "fold": request.fold,
                        "seed": request.seed,
                        "ticker": ticker,
                        "profile": profile,
                        "regime": _calendar_quarter(2025, block_index),
                        "calendar_block": f"2025-W{block_index:02d}",
                        "episode_id": f"{ticker}-{profile}-{block_index:02d}",
                        "outcome": (
                            "PASS"
                            if request.phase != "development" and block_index % 2 == 0
                            else "TIMEOUT"
                        ),
                        "finite": True,
                        "action_collapsed": False,
                    }
                )
    report = _write(
        request.output / "validation_report.json",
        {
            "schema_version": 1,
            "stage": "rl-campaign-validation",
            "partition": "validation",
            "fold": request.fold,
            "seed": request.seed,
            "phase": request.phase,
            "validation_manifest_sha256": request.validation_manifest_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "finite": True,
            "action_collapsed": False,
            "rows": rows,
            "test_accessed": False,
            "sealed_holdout_accessed": False,
        },
    )
    endpoint = {
        "model_sha256": _canonical_digest(components["model"]),
        "optimizer_sha256": _canonical_digest(components["optimizer"]),
        "rng_sha256": _canonical_digest(components["rng"]),
        "normalizer_sha256": _canonical_digest(components["normalizers"]),
        "controller_sha256": _canonical_digest(components["constraint_controller"]),
        "metrics_sha256": _canonical_digest(components["metrics"]),
    }
    result = {
        "status": "complete",
        "finite": True,
        "action_collapsed": False,
        "blows": 0,
        "all_gates_passed": True,
        "pass_rate": sum(row["outcome"] == "PASS" for row in rows) / len(rows),
        "learning_curve_improving": True,
        "artifact_sha256": checkpoint_sha256,
        "training_manifest_path": str(manifest),
        "training_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "checkpoint_bundle_path": str(bundle),
        "checkpoint_bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "validation_report_path": str(report),
        "validation_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "completed_timesteps": request.timesteps,
        "milestone_2m_sha256": request.parent_artifact_sha256 or checkpoint_sha256,
        "lineage_parent_sha256": request.parent_artifact_sha256,
        "endpoint_bundle": endpoint,
    }
    if request.phase == "confirmation":
        milestone_checkpoint = checkpoint
        if request.parent_artifact_sha256 is not None:
            milestone_checkpoint = (
                request.output.parent.parent
                / "development"
                / f"seed-{request.seed}"
                / "checkpoint.pt"
            )
        milestone_checkpoint_sha256 = hashlib.sha256(milestone_checkpoint.read_bytes()).hexdigest()
        milestone_rows = [{**row, "outcome": "TIMEOUT"} for row in rows]
        milestone_report = _write(
            request.output / "milestone_validation_report.json",
            {
                "schema_version": 1,
                "stage": "rl-campaign-validation",
                "partition": "validation",
                "fold": request.fold,
                "seed": request.seed,
                "phase": "milestone_2m",
                "validation_manifest_sha256": request.validation_manifest_sha256,
                "checkpoint_sha256": milestone_checkpoint_sha256,
                "finite": True,
                "action_collapsed": False,
                "rows": milestone_rows,
                "test_accessed": False,
                "sealed_holdout_accessed": False,
            },
        )
        result.update(
            {
                "milestone_checkpoint_path": str(milestone_checkpoint),
                "milestone_checkpoint_sha256": milestone_checkpoint_sha256,
                "milestone_validation_report_path": str(milestone_report),
                "milestone_validation_report_sha256": hashlib.sha256(
                    milestone_report.read_bytes()
                ).hexdigest(),
            }
        )
    return result


def test_qualification_freezes_only_preregistered_candidate_from_validation(tmp_path: Path) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    winner = _winner(tmp_path / "winner.json")
    evidence = _evidence(tmp_path / "evidence.json", winner)

    result = qualify_architecture(config, winner, evidence, tmp_path / "candidates")

    candidate = Path(result["candidate_path"])
    payload = json.loads(candidate.read_text())
    assert candidate.name == (
        f"candidate-freeze-{hashlib.sha256(candidate.read_bytes()).hexdigest()}.json"
    )
    assert payload["selected_variant"] == PolicyVariant.SHARED_TICKER_VALUE.value
    assert payload["development"] == {
        "seeds": [42, 43, 44, 45, 46],
        "timesteps_per_seed": 2_000_000,
    }
    assert payload["confirmation"] == {
        "seeds": list(range(42, 52)),
        "timesteps_per_seed": 5_000_000,
    }
    assert payload["maximum_timesteps_per_seed"] == 10_000_000
    assert payload["serving_seed"] == 42
    assert payload["test_accessed"] is False
    assert payload["sealed_holdout_accessed"] is False
    assert all(gate["point_difference"] >= 0 for gate in payload["gates"])
    assert all(gate["paired_lcb_95"] >= 0 for gate in payload["gates"])
    assert len(payload["gates"]) == 14
    assert all(gate["seed_uncertainty"]["assignments"] == 32 for gate in payload["gates"])
    expected_seed = int.from_bytes(
        hashlib.sha256(
            f"{payload['architecture_plan_sha256']}:market-bootstrap-v1".encode()
        ).digest()[:8],
        "big",
    )
    assert payload["market_bootstrap"]["rng_seed"] == expected_seed
    repeated = qualify_architecture(config, winner, evidence, tmp_path / "candidates")
    assert (
        repeated["market_bootstrap"]["index_matrix_sha256"]
        == payload["market_bootstrap"]["index_matrix_sha256"]
    )


def test_architecture_plan_freezes_all_manifest_pairs(tmp_path: Path) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    winner = _winner(tmp_path / "winner.json")
    training_paths = []
    validation_paths = []
    for fold in (0, 1):
        training = _write(
            tmp_path / f"training-{fold}.json",
            {
                "schema_version": 1,
                "stage": "rl-episode-schedule",
                "fold": fold,
                "partition": {
                    "name": "training",
                    "start": "2024-01-01T00:00:00+00:00",
                    "end": "2024-06-30T23:59:59+00:00",
                },
                "identities": {"config": {"sha256": config.digest}},
            },
        )
        validation = _write(
            tmp_path / f"validation-{fold}.json",
            {
                "schema_version": 1,
                "stage": "rl-episode-schedule",
                "fold": fold,
                "partition": {
                    "name": "validation",
                    "start": "2024-07-01T00:00:00+00:00",
                    "end": "2024-12-31T23:59:59+00:00",
                },
                "identities": {"config": {"sha256": config.digest}},
            },
        )
        training_paths.append(training)
        validation_paths.append(validation)

    plan = freeze_architecture_plan(
        config,
        winner,
        training_paths,
        validation_paths,
        tmp_path / "plans",
        created_at="2026-07-22T12:00:00+00:00",
        runtime_identities={
            "source": {
                "revision": "test-fixture",
                "dirty": False,
                "sha256": config.upstream.source_digest,
            },
            "lock": {"sha256": config.upstream.lock_digest},
        },
    )

    assert plan["folds"] == [0, 1]
    assert [pair["fold"] for pair in plan["manifest_pairs"]] == [0, 1]


def test_real_architecture_replay_ignores_caller_authored_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    validation = _write(tmp_path / "validation.json", {"episodes": [{"number": 7}]})
    episode = SimpleNamespace(
        ticker="ES",
        profile="one_mini",
        bars=[SimpleNamespace(timestamp=datetime(2025, 1, 6, tzinfo=UTC))],
    )

    class TimeoutEnvironment:
        def __init__(self, _config, _episode):
            pass

        def reset(self, seed=None):
            return SimpleNamespace(vector=np.zeros(42, dtype=np.float32)), {}

        def action_mask(self):
            return np.asarray([True, False])

        def step(self, _action):
            return (
                SimpleNamespace(vector=np.zeros(42, dtype=np.float32)),
                0.0,
                True,
                False,
                {"status": "TIMEOUT"},
            )

    monkeypatch.setattr(
        rl_confirmation,
        "load_episode_manifest",
        lambda _config, _path: SimpleNamespace(episodes=(episode,), fold=0),
    )
    monkeypatch.setattr(rl_confirmation, "TopstepEntryEnvironment", TimeoutEnvironment)
    model = EntryActorCritic(42, PolicyVariant.SHARED_TICKER_VALUE, hidden_width=64)
    report = {
        "validation_manifest_path": str(validation),
        "variant": PolicyVariant.SHARED_TICKER_VALUE.value,
        "seed": 42,
        "policy_diagnostics": {"action_collapse_detected": False},
    }
    checkpoint = {
        "model": model.state_dict(),
        "identities": {"ppo_hyperparameters": {"hidden_width": 64}},
    }

    replayed, diagnostics = _REAL_ARCHITECTURE_REPLAY(config, report, checkpoint)

    assert replayed[0]["outcome"] == "TIMEOUT"
    assert replayed[0]["outcome"] != "PASS"
    assert diagnostics["action_collapse_detected"] is True

    forged_report = {
        **report,
        "rows": [{**replayed[0], "action_collapsed": False}],
        "policy_diagnostics": {**diagnostics, "action_collapse_detected": False},
    }
    monkeypatch.setattr(rl_confirmation, "_replay_architecture_rows", _REAL_ARCHITECTURE_REPLAY)
    with pytest.raises(ConfirmationError, match="architecture validation replay mismatch"):
        rl_confirmation._validated_architecture_replay(
            config, forged_report, checkpoint, forged_report["rows"]
        )


def test_architecture_ablation_resumes_completed_attempt_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    winner = _winner(tmp_path / "winner.json")
    evidence = _evidence(tmp_path / "evidence.json", winner)
    evidence_payload = json.loads(evidence.read_text())
    plan_path = Path(evidence_payload["plan_path"])
    calls = []
    episode = SimpleNamespace(
        ticker="ES",
        profile="one_mini",
        bars=[SimpleNamespace(timestamp=datetime(2025, 1, 6, tzinfo=UTC))],
    )

    class FakeEnvironment:
        def __init__(self, _config, owned_episode):
            self.episode = owned_episode

        def reset(self, seed=None):
            return SimpleNamespace(vector=np.zeros(42, dtype=np.float32)), {}

        def action_mask(self):
            return np.asarray([True, False])

        def step(self, _action):
            return (
                SimpleNamespace(vector=np.zeros(42, dtype=np.float32)),
                0.0,
                True,
                False,
                {"status": "PASS"},
            )

    def fake_train(_config, training_path, output, *, seed, variant, hyperparameters, **_kwargs):
        calls.append((variant.value, seed))
        checkpoint_dir = output / "checkpoints" / "update-000001"
        checkpoint_dir.mkdir(parents=True)
        checkpoint = checkpoint_dir / "checkpoint.pt"
        torch.save(
            {
                "schema_version": 3,
                "model": EntryActorCritic(
                    42, variant, hidden_width=hyperparameters.hidden_width
                ).state_dict(),
            },
            checkpoint,
        )
        _write(
            checkpoint_dir / "bundle.json",
            {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        )
        _write(output / "state.json", {"checkpoint": "checkpoints/update-000001"})
        result = {
            "schema_version": 2,
            "stage": "rl-train",
            "status": "complete",
            "finite_gradients": True,
            "policy_diagnostics": {"action_collapse_detected": False},
            "metrics": [],
        }
        _write(output / "manifest.json", result)
        return result

    monkeypatch.setattr(
        rl_confirmation,
        "load_episode_manifest",
        lambda _config, _path: SimpleNamespace(episodes=(episode,), fold=0),
    )
    monkeypatch.setattr(rl_confirmation, "TopstepEntryEnvironment", FakeEnvironment)
    monkeypatch.setattr(rl_confirmation, "train_entry_policy", fake_train)
    monkeypatch.setattr(rl_confirmation, "_validated_rows", lambda *_args: ([], "0" * 64))
    validation_path = Path(
        json.loads(plan_path.read_text())["manifest_pairs"][0]["validation_manifest_path"]
    )
    validation_payload = json.loads(validation_path.read_text())
    validation_payload["episodes"] = [{"number": 1}]
    _write(validation_path, validation_payload)
    plan_payload = json.loads(plan_path.read_text())
    pair = plan_payload["manifest_pairs"][0]
    pair["validation_manifest_sha256"] = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    plan_payload["validation_manifest_sha256"] = hashlib.sha256(
        json.dumps([pair["validation_manifest_sha256"]], separators=(",", ":")).encode()
    ).hexdigest()
    data = json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode()
    replacement = plan_path.with_name(f"architecture-plan-{hashlib.sha256(data).hexdigest()}.json")
    replacement.write_bytes(data)

    output = tmp_path / "ablation"
    run_architecture_ablation(config, replacement, output)
    monkeypatch.setattr(
        rl_confirmation,
        "train_entry_policy",
        lambda *_args, **_kwargs: pytest.fail("completed architecture attempt retrained"),
    )
    resumed = run_architecture_ablation(config, replacement, output, resume=True)

    assert len(calls) == 15
    assert len(resumed["reports"]) == 15

    failed_output = tmp_path / "failed-ablation"
    monkeypatch.setattr(
        rl_confirmation,
        "train_entry_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("trainer exploded")),
    )
    with pytest.raises(RuntimeError, match="trainer exploded"):
        run_architecture_ablation(config, replacement, failed_output)
    failure = json.loads((failed_output / "ledger" / "attempt-0001.json").read_text())
    assert failure["status"] == "failed"
    assert failure["failure"] == {
        "phase": "training",
        "type": "RuntimeError",
        "message": "trainer exploded",
    }
    with pytest.raises(ConfirmationError, match="previously failed: trainer exploded"):
        run_architecture_ablation(config, replacement, failed_output, resume=True)


@pytest.mark.parametrize(
    "mutation",
    (
        "test",
        "missing",
        "negative",
        "nonfinite",
        "collapse",
        "blow",
        "unknown_key",
        "regime_duplicate",
        "report_test_access",
        "checkpoint_bytes",
        "learning_curve",
        "plan_commitment",
    ),
)
def test_qualification_fails_closed_on_invalid_or_failed_evidence(
    tmp_path: Path, mutation: str
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    winner = _winner(tmp_path / "winner.json")
    evidence = _evidence(
        tmp_path / "evidence.json",
        winner,
        candidate_passes=7 if mutation == "negative" else 9,
    )
    payload = json.loads(evidence.read_text())
    if mutation == "test":
        payload["partition"] = "test"
    elif mutation == "plan_commitment":
        payload["plan_sha256"] = "0" * 64
    elif mutation in {
        "missing",
        "nonfinite",
        "collapse",
        "blow",
        "unknown_key",
        "regime_duplicate",
        "report_test_access",
        "learning_curve",
    }:
        reference = payload["reports"][0]
        report_path = Path(reference["path"])
        report = json.loads(report_path.read_text())
        if mutation == "missing":
            report["rows"].pop()
        elif mutation == "nonfinite":
            report["rows"][0]["finite"] = False
        elif mutation == "collapse":
            report["rows"][0]["action_collapsed"] = True
        elif mutation == "unknown_key":
            report["rows"][0]["unexpected"] = True
        elif mutation == "regime_duplicate":
            duplicate = dict(report["rows"][0])
            duplicate["regime"] = "duplicate-weight"
            report["rows"].append(duplicate)
        elif mutation == "report_test_access":
            report["test_accessed"] = True
        elif mutation == "learning_curve":
            report["learning_curve"] = [{"timesteps": 2_000_000, "pass_rate": 1.0}]
        else:
            report["rows"][0]["outcome"] = "BLOW"
        data = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        replacement = report_path.with_name(f"report-{hashlib.sha256(data).hexdigest()}.json")
        replacement.write_bytes(data)
        reference["path"] = str(replacement)
        reference["sha256"] = hashlib.sha256(data).hexdigest()
    elif mutation == "checkpoint_bytes":
        report = json.loads(Path(payload["reports"][0]["path"]).read_text())
        bundle = Path(report["checkpoint_bundle_path"])
        (bundle.parent / "checkpoint.pt").write_bytes(b"not-a-schema-v3-checkpoint")
    _write(evidence, payload)

    with pytest.raises(ConfirmationError):
        qualify_architecture(config, winner, evidence, tmp_path / "candidates")


def test_confirmation_ledgers_every_attempt_and_resumes_without_retraining(tmp_path: Path) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    qualified = qualify_architecture(
        config,
        _winner(tmp_path / "winner.json"),
        _evidence(tmp_path / "evidence.json", tmp_path / "winner.json"),
        tmp_path / "candidates",
    )
    calls = []

    def runner(request):
        calls.append(request)
        return _campaign_result(request)

    output = tmp_path / "confirmation"
    result = run_seed_confirmation(config, Path(qualified["candidate_path"]), output, runner=runner)
    assert result["status"] == "complete"
    assert [(call.phase, call.seed, call.timesteps) for call in calls[:5]] == [
        ("development", seed, 2_000_000) for seed in range(42, 47)
    ]
    assert [(call.phase, call.seed, call.timesteps) for call in calls[5:]] == [
        ("confirmation", seed, 5_000_000) for seed in range(42, 52)
    ] + [("fresh_5m_reference", seed, 5_000_000) for seed in range(42, 47)]
    assert len(list((output / "ledger").glob("attempt-*.json"))) == 20
    assert result["serving_seed"] == 42

    resumed_calls = []
    resumed = run_seed_confirmation(
        config,
        Path(qualified["candidate_path"]),
        output,
        runner=lambda request: resumed_calls.append(request),
        resume=True,
    )
    assert resumed == result
    assert resumed_calls == []

    bundle_path = Path(result["seed_artifacts"][0]["checkpoint_bundle_path"])
    (bundle_path.parent / "checkpoint.pt").write_bytes(b"tampered")
    with pytest.raises(ConfirmationError):
        run_seed_confirmation(
            config,
            Path(qualified["candidate_path"]),
            output,
            runner=lambda request: resumed_calls.append(request),
            resume=True,
        )


def test_candidate_consumption_recomputes_statistical_decision(tmp_path: Path) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    winner = _winner(tmp_path / "winner.json")
    qualified = qualify_architecture(
        config,
        winner,
        _evidence(tmp_path / "evidence.json", winner),
        tmp_path / "candidates",
    )
    candidate = json.loads(Path(qualified["candidate_path"]).read_text())
    candidate["gates"][0]["point_difference"] = 1.0
    data = json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    forged = tmp_path / f"candidate-freeze-{hashlib.sha256(data).hexdigest()}.json"
    forged.write_bytes(data)

    with pytest.raises(ConfirmationError, match="decision recomputation mismatch"):
        run_seed_confirmation(config, forged, tmp_path / "campaign", runner=_campaign_result)


def test_confirmation_stops_on_failed_seed_and_continuation_is_bounded(tmp_path: Path) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    qualified = qualify_architecture(
        config,
        _winner(tmp_path / "winner.json"),
        _evidence(tmp_path / "evidence.json", tmp_path / "winner.json"),
        tmp_path / "candidates",
    )

    def failed(request):
        result = _campaign_result(request)
        result["finite"] = request.seed != 43
        return result

    with pytest.raises(ConfirmationError, match="seed 43"):
        run_seed_confirmation(
            config, Path(qualified["candidate_path"]), tmp_path / "failed", runner=failed
        )
    assert len(list((tmp_path / "failed" / "ledger").glob("attempt-*.json"))) == 2
    first_attempt = tmp_path / "failed" / "ledger" / "attempt-0001.json"
    first_payload = json.loads(first_attempt.read_text())
    original_payload = dict(first_payload)
    first_payload["test_accessed"] = True
    first_attempt.write_text(json.dumps(first_payload, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ConfirmationError, match="attempt provenance mismatch"):
        run_seed_confirmation(
            config,
            Path(qualified["candidate_path"]),
            tmp_path / "failed",
            runner=failed,
            resume=True,
        )
    first_payload = original_payload
    first_payload["result"]["completed_timesteps"] = 999
    first_attempt.write_text(json.dumps(first_payload, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ConfirmationError, match="resume lineage mismatch"):
        run_seed_confirmation(
            config,
            Path(qualified["candidate_path"]),
            tmp_path / "failed",
            runner=failed,
            resume=True,
        )

    with pytest.raises(ConfirmationError, match="continuation"):
        run_seed_confirmation(
            config,
            Path(qualified["candidate_path"]),
            tmp_path / "too-long",
            runner=failed,
            continuation_timesteps=10_000_001,
        )


def test_cli_exposes_architecture_qualification(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = object()
    called = {}
    monkeypatch.setattr(cli, "load_rl_config", lambda _path: sentinel)

    def qualify(config, winner, evidence, output):
        called.update(config=config, winner=winner, evidence=evidence, output=output)
        return {"stage": "candidate-freeze-v1"}

    monkeypatch.setattr(cli, "qualify_architecture", qualify)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mantis-v2",
            "rl-qualify-architecture",
            "--config",
            "rl.toml",
            "--winner",
            "winner.json",
            "--evidence",
            "evidence.json",
            "--output",
            "candidates",
        ],
    )

    cli.main()

    assert json.loads(capsys.readouterr().out)["stage"] == "candidate-freeze-v1"
    assert called == {
        "config": sentinel,
        "winner": Path("winner.json"),
        "evidence": Path("evidence.json"),
        "output": Path("candidates"),
    }


def test_continuation_rejects_incomplete_confirmation_only_progress(tmp_path: Path) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    winner = _winner(tmp_path / "winner.json")
    qualified = qualify_architecture(
        config,
        winner,
        _evidence(tmp_path / "evidence.json", winner),
        tmp_path / "candidates",
    )
    candidate = json.loads(Path(qualified["candidate_path"]).read_text())
    plan = json.loads(Path(candidate["architecture_plan_path"]).read_text())
    pair = plan["manifest_pairs"][0]
    rows = []
    for milestone in ("2m", "5m"):
        for seed in config.training.confirmation_seeds:
            for ticker in TICKERS:
                profiles = ("one_mini",) if ticker == "ZB" else PROFILES
                for profile in profiles:
                    for block_index in range(1, 21):
                        rows.append(
                            {
                                "milestone": milestone,
                                "fold": 0,
                                "seed": seed,
                                "ticker": ticker,
                                "profile": profile,
                                "regime": _calendar_quarter(2025, block_index),
                                "calendar_block": f"2025-W{block_index:02d}",
                                "episode_id": f"{ticker}-{profile}-{block_index:02d}",
                                "outcome": (
                                    "PASS"
                                    if milestone == "5m" and block_index % 2 == 0
                                    else "TIMEOUT"
                                ),
                                "finite": True,
                                "action_collapsed": False,
                            }
                        )
    campaign_directory = tmp_path / "campaign"
    ledger = campaign_directory / "ledger"
    ledger.mkdir(parents=True)
    report_references = []
    ledger_references = []
    for attempt_number, seed in enumerate(config.training.confirmation_seeds, start=1):
        request = type(
            "Request",
            (),
            {
                "output": campaign_directory / "runs" / "fold-0" / "confirmation" / f"seed-{seed}",
                "fold": 0,
                "training_manifest_sha256": pair["training_manifest_sha256"],
                "validation_manifest_sha256": pair["validation_manifest_sha256"],
                "seed": seed,
                "variant": PolicyVariant.SHARED_TICKER_VALUE.value,
                "phase": "confirmation",
                "timesteps": 5_000_000,
                "parent_artifact_sha256": None,
            },
        )()
        result = _campaign_result(request)
        checkpoint_path = request.output / "checkpoint.pt"
        result["milestone_2m_sha256"] = result["artifact_sha256"]
        result["milestone_checkpoint_path"] = str(checkpoint_path)
        result["milestone_checkpoint_sha256"] = result["artifact_sha256"]
        for milestone, report_name in (("2m", "milestone.json"), ("5m", "validation_report.json")):
            report_path = request.output / report_name
            report_rows = [
                {key: value for key, value in row.items() if key != "milestone"}
                for row in rows
                if row["seed"] == seed and row["milestone"] == milestone
            ]
            _write(
                report_path,
                {
                    "schema_version": 1,
                    "stage": "rl-campaign-validation",
                    "partition": "validation",
                    "fold": 0,
                    "seed": seed,
                    "validation_manifest_sha256": pair["validation_manifest_sha256"],
                    "phase": "milestone_2m" if milestone == "2m" else "confirmation",
                    "checkpoint_sha256": result["artifact_sha256"],
                    "finite": True,
                    "action_collapsed": False,
                    "rows": report_rows,
                    "test_accessed": False,
                    "sealed_holdout_accessed": False,
                },
            )
            digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
            if milestone == "2m":
                result["milestone_validation_report_path"] = str(report_path)
                result["milestone_validation_report_sha256"] = digest
            else:
                result["validation_report_path"] = str(report_path)
                result["validation_report_sha256"] = digest
            report_references.append(
                {
                    "attempt_number": attempt_number,
                    "milestone": milestone,
                    "path": str(report_path),
                    "sha256": digest,
                }
            )
        attempt = _write(
            ledger / f"attempt-{attempt_number:04d}.json",
            {
                "schema_version": 1,
                "stage": "rl-seed-confirmation-attempt",
                "attempt_number": attempt_number,
                "candidate_sha256": qualified["candidate_sha256"],
                "request": {
                    "phase": "confirmation",
                    "fold": 0,
                    "training_manifest_sha256": pair["training_manifest_sha256"],
                    "validation_manifest_sha256": pair["validation_manifest_sha256"],
                    "seed": seed,
                    "output": str(request.output),
                },
                "result": result,
                "test_accessed": False,
                "sealed_holdout_accessed": False,
            },
        )
        ledger_references.append(
            {"path": str(attempt), "sha256": hashlib.sha256(attempt.read_bytes()).hexdigest()}
        )
    progress_payload = {
        "schema_version": 1,
        "stage": "rl-seed-campaign-progress-v1",
        "status": "awaiting_validation_budget_decision",
        "candidate_sha256": qualified["candidate_sha256"],
        "attempt_ledger_sha256": ledger_references,
        "test_accessed": False,
        "sealed_holdout_accessed": False,
    }
    progress_bytes = json.dumps(progress_payload, sort_keys=True, separators=(",", ":")).encode()
    progress_path = campaign_directory / (
        f"campaign-progress-{hashlib.sha256(progress_bytes).hexdigest()}.json"
    )
    progress_path.write_bytes(progress_bytes)
    budget_evidence = _write(
        tmp_path / "budget.json",
        {
            "schema_version": 1,
            "stage": "rl-budget-comparison-evidence",
            "partition": "validation",
            "candidate_sha256": qualified["candidate_sha256"],
            "campaign_progress_path": str(progress_path),
            "campaign_progress_sha256": hashlib.sha256(progress_bytes).hexdigest(),
            "reports": report_references,
            "test_accessed": False,
            "sealed_holdout_accessed": False,
        },
    )

    with pytest.raises(ConfirmationError, match="ledger set mismatch"):
        decide_continuation(
            config,
            Path(qualified["candidate_path"]),
            budget_evidence,
            tmp_path / "decisions",
        )


def test_bounded_production_campaign_reaches_serving_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    winner = _winner(tmp_path / "winner.json")
    qualified = qualify_architecture(
        config,
        winner,
        _evidence(tmp_path / "evidence.json", winner),
        tmp_path / "candidates",
    )
    candidate = json.loads(Path(qualified["candidate_path"]).read_text())
    plan = json.loads(Path(candidate["architecture_plan_path"]).read_text())
    pair = plan["manifest_pairs"][0]
    monkeypatch.setattr(
        rl_confirmation,
        "_production_campaign_runner",
        lambda _config, _pairs, _output: _campaign_result,
    )
    campaign = tmp_path / "campaign"

    progress = run_production_seed_campaign(
        config,
        Path(qualified["candidate_path"]),
        Path(pair["training_manifest_path"]),
        Path(pair["validation_manifest_path"]),
        campaign,
    )
    decision = decide_continuation(
        config,
        Path(qualified["candidate_path"]),
        Path(progress["budget_evidence_path"]),
        tmp_path / "decisions",
    )
    serving = run_production_seed_campaign(
        config,
        Path(qualified["candidate_path"]),
        Path(pair["training_manifest_path"]),
        Path(pair["validation_manifest_path"]),
        campaign,
        resume=True,
        continuation_decision_path=Path(decision["decision_path"]),
    )

    assert decision["status"] == "authorized"
    assert serving["status"] == "complete"
    assert serving["final_timesteps"] == 10_000_000
    assert serving["confirmation_statistics"]["phase"] == "continuation"
    repeated = run_production_seed_campaign(
        config,
        Path(qualified["candidate_path"]),
        Path(pair["training_manifest_path"]),
        Path(pair["validation_manifest_path"]),
        campaign,
        resume=True,
        continuation_decision_path=Path(decision["decision_path"]),
    )
    assert repeated == serving
    with pytest.raises(ConfirmationError, match="resume provenance mismatch"):
        run_production_seed_campaign(
            config,
            Path(qualified["candidate_path"]),
            Path(pair["training_manifest_path"]),
            Path(pair["validation_manifest_path"]),
            campaign,
            resume=True,
        )
