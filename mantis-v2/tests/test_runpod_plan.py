from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from mantis_v2 import cli
from mantis_v2.runpod_config import (
    RunpodConfigError,
    load_experiment_config,
    load_local_config,
    load_platform_config,
)

ROOT = Path(__file__).resolve().parents[2]


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    platform = tmp_path / "platform.toml"
    platform.write_text(
        """\
schema_version = 1

[provider]
secure_cloud = true
allowed_gpu_types = ["NVIDIA A40"]
allowed_datacenters = ["US-CA-2"]
minimum_vcpu = 8
minimum_ram_gb = 32
container_disk_gb = 50

[storage]
volume_gb = 150
high_water_bytes = 120000000000
minimum_free_bytes = 30000000000

[lifecycle]
maximum_inventory_age_seconds = 300
maximum_duration_seconds = 7200

[billing]
container_disk_usd_per_gb_month = "0.10"
billing_month_hours = 730

[budget]
account_ceiling_usd = "150.00"
storage_usd = "15.00"
qualification_usd = "10.00"
production_usd = "100.00"
protected_recovery_usd = "25.00"
ordinary_launch_cutoff_usd = "125.00"
"""
    )
    local = tmp_path / "local.toml"
    local.write_text(
        """\
schema_version = 1

[paths]
workspace_root = "/workspace/mantis"
state_root = "/tmp/mantis-runpod"
output_root = "/tmp/mantis-plans"

[secrets]
runpod_api_key_env = "RUNPOD_API_KEY"
s3_access_key_id_env = "RUNPOD_S3_ACCESS_KEY_ID"
s3_secret_access_key_env = "RUNPOD_S3_SECRET_ACCESS_KEY"
"""
    )
    experiment = tmp_path / "experiment.toml"
    experiment.write_text(
        """\
schema_version = 1

[experiment]
name = "mantisv2-cuda-qualification"
model_family = "mantis-v2"
stage = "qualification"
seed = 42
definition_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
sealed_holdout = false
"""
    )
    intent = tmp_path / "intent.json"
    intent.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "intent_id": "qualify-a40-seed42",
                "stage": "qualification",
                "run_name": "mantisv2-cuda-qualification-seed42",
                "gpu_type": "NVIDIA A40",
                "datacenter_id": "US-CA-2",
                "gpu_count": 1,
                "vcpu": 8,
                "ram_gb": 32,
                "container_disk_gb": 50,
                "volume_id": "volume-fixture",
                "volume_size_gb": 150,
                "maximum_duration_seconds": 7200,
            }
        )
        + "\n"
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observed_at": "2026-07-21T12:00:00Z",
                "account_balance_usd": "150.00",
                "volume_free_bytes": 100000000000,
                "offers": [
                    {
                        "gpu_type": "NVIDIA A40",
                        "datacenter_id": "US-CA-2",
                        "price_usd_per_gpu_hour": "0.44",
                        "available": True,
                    }
                ],
                "live_pods": [],
            }
        )
        + "\n"
    )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "actual_spend_usd": "0.00",
                "reserved_spend_usd": "0.00",
                "bucket_actual_spend_usd": {
                    "qualification": "0.00",
                    "production": "0.00",
                    "recovery": "0.00",
                },
                "bucket_reserved_spend_usd": {
                    "qualification": "0.00",
                    "production": "0.00",
                    "recovery": "0.00",
                },
                "active_reservations": [],
                "consumed_authorization_digests": [],
            }
        )
        + "\n"
    )
    return {
        "platform": platform,
        "local": local,
        "experiment": experiment,
        "intent": intent,
        "inventory": inventory,
        "ledger": ledger,
    }


def _run_plan(
    paths: dict[str, Path],
    output: Path,
    monkeypatch,
    capsys,
    authorization: Path | None = None,
) -> dict[str, object]:
    argv = [
        "mantis-v2",
        "runpod-plan",
        "--platform",
        str(paths["platform"]),
        "--local",
        str(paths["local"]),
        "--experiment",
        str(paths["experiment"]),
        "--intent",
        str(paths["intent"]),
        "--inventory",
        str(paths["inventory"]),
        "--ledger",
        str(paths["ledger"]),
    ]
    if authorization is not None:
        argv.extend(("--authorization", str(authorization)))
    argv.extend(
        (
            "--evaluated-at",
            "2026-07-21T12:02:00Z",
            "--output",
            str(output),
        )
    )
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    capsys.readouterr()
    loaded = json.loads(output.read_text())
    assert isinstance(loaded, dict)
    return loaded


def _authorization_for(paths: dict[str, Path], tmp_path: Path, monkeypatch, capsys) -> Path:
    rejection = _run_plan(paths, tmp_path / "authorization-subject.json", monkeypatch, capsys)
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "authorization_id": "approval-001",
                "subject_digest": rejection["authorization_subject_digest"],
                "authorized_at": "2026-07-21T12:01:30Z",
                "expires_at": "2026-07-21T12:10:00Z",
                "maximum_projected_spend_usd": "0.90",
                "approver": "lgbarn",
            }
        )
        + "\n"
    )
    return authorization


def test_plan_command_without_authorization_writes_canonical_rejection(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    provider_calls: list[tuple[object, ...]] = []

    def provider_oracle(*args, **_kwargs):
        provider_calls.append(args)
        raise AssertionError("planning must not invoke a provider subprocess")

    monkeypatch.setattr(subprocess, "run", provider_oracle)
    paths = _write_inputs(tmp_path)
    output = tmp_path / "decisions" / "launch-decision.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "runpod-plan",
            "--platform",
            str(paths["platform"]),
            "--local",
            str(paths["local"]),
            "--experiment",
            str(paths["experiment"]),
            "--intent",
            str(paths["intent"]),
            "--inventory",
            str(paths["inventory"]),
            "--ledger",
            str(paths["ledger"]),
            "--evaluated-at",
            "2026-07-21T12:01:00Z",
            "--output",
            str(output),
        ],
    )

    cli.main()

    result = json.loads(capsys.readouterr().out)
    decision = json.loads(output.read_text())
    assert result == {
        "decision_digest": decision["decision_digest"],
        "decision_path": str(output),
    }
    assert decision["allowed"] is False
    assert decision["reasons"] == ["authorization_required"]
    assert provider_calls == []
    assert output.read_bytes().endswith(b"\n")


def test_plan_command_with_exact_authorization_binds_terms_and_approves(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    paths = _write_inputs(tmp_path)
    rejection_path = tmp_path / "rejection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "runpod-plan",
            "--platform",
            str(paths["platform"]),
            "--local",
            str(paths["local"]),
            "--experiment",
            str(paths["experiment"]),
            "--intent",
            str(paths["intent"]),
            "--inventory",
            str(paths["inventory"]),
            "--ledger",
            str(paths["ledger"]),
            "--evaluated-at",
            "2026-07-21T12:01:00Z",
            "--output",
            str(rejection_path),
        ],
    )
    cli.main()
    capsys.readouterr()
    subject_digest = json.loads(rejection_path.read_text())["authorization_subject_digest"]
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "authorization_id": "approval-001",
                "subject_digest": subject_digest,
                "authorized_at": "2026-07-21T12:01:30Z",
                "expires_at": "2026-07-21T12:10:00Z",
                "maximum_projected_spend_usd": "0.90",
                "approver": "lgbarn",
            }
        )
        + "\n"
    )
    output = tmp_path / "approved.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "runpod-plan",
            "--platform",
            str(paths["platform"]),
            "--local",
            str(paths["local"]),
            "--experiment",
            str(paths["experiment"]),
            "--intent",
            str(paths["intent"]),
            "--inventory",
            str(paths["inventory"]),
            "--ledger",
            str(paths["ledger"]),
            "--authorization",
            str(authorization),
            "--evaluated-at",
            "2026-07-21T12:02:00Z",
            "--output",
            str(output),
        ],
    )

    cli.main()

    result = json.loads(capsys.readouterr().out)
    decision = json.loads(output.read_text())
    assert result["decision_path"] == str(output)
    assert decision["allowed"] is True
    assert decision["reasons"] == []
    assert decision["provider_price_usd_per_gpu_hour"] == "0.44"
    assert decision["inventory_observed_at"] == "2026-07-21T12:00:00Z"
    assert decision["maximum_duration_seconds"] == 7200
    assert decision["projected_spend_usd"] == "0.90"
    assert decision["authorization_digest"]
    assert decision["authorization_expires_at"] == "2026-07-21T12:10:00Z"


@pytest.mark.parametrize(
    ("case", "reason"),
    (
        ("stale_inventory", "inventory_stale"),
        ("changed_price", "authorization_subject_mismatch"),
        ("changed_gpu", "authorization_subject_mismatch"),
        ("changed_datacenter", "authorization_subject_mismatch"),
        ("expired_authorization", "authorization_expired"),
        ("replayed_authorization", "authorization_replayed"),
        ("insufficient_balance", "insufficient_balance"),
        ("reserved_spend_overlap", "reserved_spend_overlap"),
        ("ordinary_launch_cutoff", "ordinary_launch_cutoff_reached"),
        ("ordinary_launch_crosses_cutoff", "ordinary_launch_cutoff_reached"),
        ("insufficient_storage", "insufficient_storage"),
    ),
)
def test_plan_command_rejects_each_policy_violation(
    tmp_path: Path, monkeypatch, capsys, case: str, reason: str
) -> None:
    paths = _write_inputs(tmp_path)
    authorization = _authorization_for(paths, tmp_path, monkeypatch, capsys)
    if case in {
        "stale_inventory",
        "changed_price",
        "insufficient_balance",
        "insufficient_storage",
    }:
        inventory = json.loads(paths["inventory"].read_text())
        if case == "stale_inventory":
            inventory["observed_at"] = "2026-07-21T11:00:00Z"
        elif case == "changed_price":
            inventory["offers"][0]["price_usd_per_gpu_hour"] = "0.45"
        elif case == "insufficient_balance":
            inventory["account_balance_usd"] = "0.10"
        else:
            inventory["volume_free_bytes"] = 29_999_999_999
        paths["inventory"].write_text(json.dumps(inventory) + "\n")
    elif case in {"changed_gpu", "changed_datacenter"}:
        intent = json.loads(paths["intent"].read_text())
        if case == "changed_gpu":
            intent["gpu_type"] = "NVIDIA L40"
        else:
            intent["datacenter_id"] = "EU-RO-1"
        paths["intent"].write_text(json.dumps(intent) + "\n")
    elif case == "expired_authorization":
        approved = json.loads(authorization.read_text())
        approved["expires_at"] = "2026-07-21T12:02:00Z"
        authorization.write_text(json.dumps(approved) + "\n")
    else:
        ledger = json.loads(paths["ledger"].read_text())
        if case == "replayed_authorization":
            approved = _run_plan(
                paths, tmp_path / "first-approved.json", monkeypatch, capsys, authorization
            )
            ledger["consumed_authorization_digests"] = [approved["authorization_digest"]]
        elif case == "reserved_spend_overlap":
            ledger["active_reservations"] = ["reservation-001"]
            ledger["reserved_spend_usd"] = "0.90"
            ledger["bucket_reserved_spend_usd"]["qualification"] = "0.90"
        elif case == "ordinary_launch_cutoff":
            ledger["actual_spend_usd"] = "125.00"
        else:
            ledger["actual_spend_usd"] = "124.50"
        paths["ledger"].write_text(json.dumps(ledger) + "\n")

    decision = _run_plan(paths, tmp_path / f"{case}.json", monkeypatch, capsys, authorization)

    assert decision["allowed"] is False
    assert reason in decision["reasons"]


def test_runpod_configs_have_separate_canonical_identities(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    platform = load_platform_config(paths["platform"])
    local = load_local_config(paths["local"])
    experiment = load_experiment_config(paths["experiment"])

    moved_local = replace(
        local,
        paths=replace(local.paths, state_root=Path("/another/machine/state")),
    )
    changed_experiment = replace(
        experiment,
        experiment=replace(experiment.experiment, seed=43),
    )

    assert len({platform.digest, local.digest, experiment.digest}) == 3
    assert moved_local.digest != local.digest
    assert experiment.digest == load_experiment_config(paths["experiment"]).digest
    assert changed_experiment.digest != experiment.digest


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ("secure_cloud = true", "secure_cloud = false", "secure_cloud must be true"),
        (
            "high_water_bytes = 120000000000",
            "high_water_bytes = 121000000000",
            "high-water policy must leave minimum_free_bytes",
        ),
        (
            'ordinary_launch_cutoff_usd = "125.00"',
            'ordinary_launch_cutoff_usd = "126.00"',
            "ordinary launch cutoff must protect the recovery reserve",
        ),
    ),
)
def test_platform_config_rejects_incompatible_policy(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    paths = _write_inputs(tmp_path)
    paths["platform"].write_text(paths["platform"].read_text().replace(old, new, 1))

    with pytest.raises(RunpodConfigError, match=message):
        load_platform_config(paths["platform"])


def test_plan_command_rejects_unknown_config_key_without_output(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    paths = _write_inputs(tmp_path)
    paths["platform"].write_text(
        paths["platform"]
        .read_text()
        .replace("schema_version = 1", "schema_version = 1\nsurprise = true", 1)
    )
    output = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "runpod-plan",
            "--platform",
            str(paths["platform"]),
            "--local",
            str(paths["local"]),
            "--experiment",
            str(paths["experiment"]),
            "--intent",
            str(paths["intent"]),
            "--inventory",
            str(paths["inventory"]),
            "--ledger",
            str(paths["ledger"]),
            "--evaluated-at",
            "2026-07-21T12:02:00Z",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main()

    captured = capsys.readouterr()
    assert "unknown platform keys: surprise" in captured.err
    assert not output.exists()


def test_local_config_accepts_secret_names_but_rejects_secret_values(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    paths["local"].write_text(
        paths["local"]
        .read_text()
        .replace(
            'runpod_api_key_env = "RUNPOD_API_KEY"',
            'runpod_api_key_env = "actual-secret-value"',
        )
    )

    with pytest.raises(
        RunpodConfigError,
        match="secrets.runpod_api_key_env must name RUNPOD_API_KEY",
    ):
        load_local_config(paths["local"])


def test_decision_is_content_addressed_no_overwrite_and_secret_free(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    paths = _write_inputs(tmp_path)
    output = tmp_path / "decision.json"
    sentinels = {
        "RUNPOD_API_KEY": "runpod-secret-sentinel",
        "RUNPOD_S3_ACCESS_KEY_ID": "s3-id-secret-sentinel",
        "RUNPOD_S3_SECRET_ACCESS_KEY": "s3-key-secret-sentinel",
    }
    for name, value in sentinels.items():
        monkeypatch.setenv(name, value)

    decision = _run_plan(paths, output, monkeypatch, capsys)
    stored_digest = decision.pop("decision_digest")
    expected_digest = hashlib.sha256(
        json.dumps(decision, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert stored_digest == expected_digest
    serialized = output.read_text()
    assert all(value not in serialized for value in sentinels.values())
    assert not list(output.parent.glob("*.tmp"))

    with pytest.raises(SystemExit, match="2"):
        _run_plan(paths, output, monkeypatch, capsys)
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert all(value not in captured.err for value in sentinels.values())


def test_runpod_plan_recipe_requires_explicit_target_paths() -> None:
    listed = subprocess.run(
        ["just", "--list"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    assert "runpod-plan platform local experiment intent inventory ledger evaluated_at output" in (
        listed.stdout
    )

    missing = subprocess.run(
        ["just", "runpod-plan"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert missing.returncode != 0
    assert "Recipe `runpod-plan` got 0 positional arguments but takes 8" in missing.stderr
