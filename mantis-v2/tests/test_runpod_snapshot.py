from __future__ import annotations

import hashlib
import json
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mantis_v2.runpod_snapshot import RunpodSnapshotError, write_provider_snapshot

ROOT = Path(__file__).resolve().parents[2]


def _inputs(tmp_path: Path) -> dict[str, Path]:
    binary = tmp_path / "runpodctl"
    binary.write_bytes(b"pinned runpodctl fixture")
    local = tmp_path / "local.toml"
    state = tmp_path / "state"
    local.write_text(
        f'''\
schema_version = 1

[controller]
hostname = "{socket.gethostname()}"

[paths]
workspace_root = "/workspace/mantis"
state_root = "{state}"
output_root = "{tmp_path / "plans"}"

[secrets]
runpod_api_key_env = "RUNPOD_API_KEY"
s3_access_key_id_env = "RUNPOD_S3_ACCESS_KEY_ID"
s3_secret_access_key_env = "RUNPOD_S3_SECRET_ACCESS_KEY"
'''
    )
    control = json.loads((ROOT / "mantis-v2/configs/frozen-paid-control.example.json").read_text())
    control["source_revision"] = "a" * 40
    control["provider"]["volume_id"] = "volume-1"
    control["runpodctl"]["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps(control))
    intent = {
        "schema_version": 1,
        "intent_id": "frozen-screen",
        "stage": "qualification",
        "run_name": "frozen-screen",
        "gpu_type": "NVIDIA L40S",
        "datacenter_id": "US-MO-1",
        "gpu_count": 1,
        "vcpu": 8,
        "ram_gb": 32,
        "container_disk_gb": 50,
        "image_ref": "runpod/pytorch@sha256:" + "b" * 64,
        "template_id": "runpod-torch-v280",
        "registry_auth_id": "",
        "volume_id": "volume-1",
        "volume_size_gb": 150,
        "volume_mount_path": "/workspace",
        "ports": ["22/tcp"],
        "maximum_duration_seconds": 7200,
    }
    intent_path = tmp_path / "intent.json"
    intent_path.write_text(json.dumps(intent))
    return {
        "platform": ROOT / "infra/runpod/configs/platform-v1.toml",
        "local": local,
        "control": control_path,
        "intent": intent_path,
        "binary": binary,
        "state": state,
    }


def _runner(*, pods: list[dict[str, object]] | None = None, spend: float = 0.0):
    payloads: dict[tuple[str, ...], object] = {
        ("user",): {"clientBalance": 147.25, "currentSpendPerHr": spend},
        ("pod", "list"): pods or [],
        ("gpu", "list"): [
            {
                "gpuId": "NVIDIA L40S",
                "displayName": "L40S",
                "memoryInGb": 48,
                "secureCloud": True,
                "communityCloud": True,
                "stockStatus": "High",
                "available": True,
            }
        ],
        ("datacenter", "list"): [
            {
                "id": "US-MO-1",
                "name": "US Missouri",
                "location": "Missouri",
                "gpuAvailability": [
                    {
                        "gpuId": "NVIDIA L40S",
                        "displayName": "L40S",
                        "stockStatus": "Low",
                    }
                ],
            }
        ],
        ("network-volume", "list"): [
            {"id": "volume-1", "name": "mantis", "size": 150, "dataCenterId": "US-MO-1"}
        ],
    }

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        key = tuple(part for part in args[1:] if not part.startswith("--"))
        return subprocess.CompletedProcess(args, 0, json.dumps(payloads[key]), "")

    return run


def _write_reconciled_receipts(state: Path) -> None:
    pod = {
        "pod_id": "pod-old",
        "authorization_digest": "c" * 64,
        "stage": "qualification",
    }
    termination = {"pod_id": "pod-old"}
    spend = {
        "pod_id": "pod-old",
        "stage": "qualification",
        "spend_state": "reconciled",
        "actual_spend_usd": "2.75",
    }
    for directory, payload in (
        ("pods", pod),
        ("terminations", termination),
        ("spend", spend),
    ):
        target = state / "receipts" / directory
        target.mkdir(parents=True, exist_ok=True)
        (target / "pod-old.json").write_text(json.dumps(payload))


def test_snapshot_emits_fresh_planner_inputs_and_receipt_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)
    _write_reconciled_receipts(paths["state"])
    monkeypatch.setenv("RUNPOD_API_KEY", "fixture-secret")
    output = tmp_path / "snapshot"

    result = write_provider_snapshot(
        platform_path=paths["platform"],
        local_path=paths["local"],
        control_path=paths["control"],
        intent_path=paths["intent"],
        runpodctl_binary=paths["binary"],
        output_root=output,
        runner=_runner(),
        now=lambda: datetime(2026, 7, 25, 12, 0, tzinfo=UTC),
    )

    inventory = json.loads((output / "inventory.json").read_text())
    ledger = json.loads((output / "spend-ledger.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    assert result["observed_at"] == "2026-07-25T12:00:00Z"
    assert inventory["account_balance_usd"] == "147.25"
    assert inventory["offers"] == [
        {
            "available": True,
            "cloud_type": "secure",
            "datacenter_id": "US-MO-1",
            "gpu_type": "NVIDIA L40S",
            "price_usd_per_gpu_hour": "0.39",
        }
    ]
    assert inventory["volumes"][0]["free_bytes"] == 30_000_000_000
    assert inventory["live_pods"] == []
    assert ledger["actual_spend_usd"] == "2.75"
    assert ledger["bucket_actual_spend_usd"]["qualification"] == "2.75"
    assert ledger["active_reservations"] == []
    assert ledger["consumed_authorization_digests"] == ["c" * 64]
    assert provenance["provider_checks"] == {
        "current_spend_per_hour_usd": "0.0",
        "live_pod_count": 0,
    }
    assert provenance["bindings"] == {
        "free_bytes": "platform.storage.minimum_free_bytes",
        "offer_price": "paid_control.provider.hourly_rate_usd",
    }
    assert "fixture-secret" not in (output / "provenance.json").read_text()


@pytest.mark.parametrize(
    ("pods", "spend", "error"),
    [
        ([{"id": "pod-live"}], 0.39, "zero live Pods"),
        ([], 0.39, "current spend must be zero"),
    ],
)
def test_snapshot_rejects_billed_state_without_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pods: list[dict[str, object]],
    spend: float,
    error: str,
) -> None:
    paths = _inputs(tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "fixture-secret")
    output = tmp_path / "snapshot"

    with pytest.raises(RunpodSnapshotError, match=error):
        write_provider_snapshot(
            platform_path=paths["platform"],
            local_path=paths["local"],
            control_path=paths["control"],
            intent_path=paths["intent"],
            runpodctl_binary=paths["binary"],
            output_root=output,
            runner=_runner(pods=pods, spend=spend),
        )

    assert not output.exists()


def test_snapshot_rejects_unreconciled_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)
    pod_dir = paths["state"] / "receipts" / "pods"
    pod_dir.mkdir(parents=True)
    (pod_dir / "pod-old.json").write_text(
        json.dumps({"pod_id": "pod-old", "authorization_digest": "c" * 64})
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "fixture-secret")

    with pytest.raises(RunpodSnapshotError, match="not reconciled"):
        write_provider_snapshot(
            platform_path=paths["platform"],
            local_path=paths["local"],
            control_path=paths["control"],
            intent_path=paths["intent"],
            runpodctl_binary=paths["binary"],
            output_root=tmp_path / "snapshot",
            runner=_runner(),
        )
