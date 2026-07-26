from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from mantis_v2.runpodctl_adapter import (
    RUNPODCTL_COMMIT,
    RUNPODCTL_VERSION,
    RunpodctlCreateAdapter,
    RunpodctlError,
)


class RestStub:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def inventory(self) -> list[dict[str, object]]:
        return []

    def status(self, pod_id: str) -> dict[str, object]:
        self.requested.append(pod_id)
        return {
            "id": pod_id,
            "name": "mantis-run",
            "desiredStatus": "RUNNING",
            "imageName": "registry/mantis@sha256:" + "a" * 64,
            "templateId": "template-1",
            "networkVolumeId": "volume-1",
            "costPerHr": "0.39",
            "vcpuCount": 16,
            "memoryInGb": 62,
            "lastStartedAt": "2026-07-22T12:00:00Z",
        }

    def terminate(self, pod_id: str) -> dict[str, object]:
        return {"deleted": True, "id": pod_id}

    def billing(self, pod_id: str) -> dict[str, object]:
        return {"pod_id": pod_id, "actual_cost_usd": "0.10"}


def _decision() -> dict[str, Any]:
    return {
        "run_name": "mantis-run",
        "gpu_type": "NVIDIA A40",
        "gpu_count": 1,
        "datacenter_id": "US-GA-1",
        "image_ref": "registry/mantis@sha256:" + "a" * 64,
        "template_id": "template-1",
        "registry_auth_id": "registry-1",
        "volume_id": "volume-1",
        "volume_mount_path": "/workspace",
        "container_disk_gb": 50,
        "ports": ["22/tcp", "6006/http"],
        "workload": {
            "docker_args": "uv run mantis-v2 workload-execute --manifest /workspace/launch.json",
            "environment": {
                "HF_HOME": "/workspace/mantis/inputs/bundle/cache/huggingface",
                "HF_HUB_OFFLINE": "1",
                "MANTIS_RUN_ID": "run-identity",
                "MANTIS_WORKLOAD_DIGEST": "d" * 64,
                "MANTIS_WORKLOAD_MANIFEST": "/workspace/launch.json",
            },
        },
    }


def test_pinned_create_uses_exact_deadline_workload_and_returned_pod_id(tmp_path: Path) -> None:
    binary = tmp_path / "runpodctl"
    binary.write_bytes(b"pinned-runpodctl")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    calls: list[list[str]] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps({"id": "pod-123"}), "")

    rest = RestStub()
    adapter = RunpodctlCreateAdapter(
        rest=rest,
        binary=binary,
        binary_sha256=digest,
        version=RUNPODCTL_VERSION,
        source_commit=RUNPODCTL_COMMIT,
        runner=runner,
    )
    deadline = datetime(2026, 7, 22, 14, 12, tzinfo=UTC)

    created = adapter.create(_decision(), deadline)

    assert created["id"] == "pod-123"
    assert rest.requested == ["pod-123"]
    assert calls == [
        [
            str(binary),
            "pod",
            "create",
            "--output=json",
            "--name=mantis-run",
            "--image=registry/mantis@sha256:" + "a" * 64,
            "--template-id=template-1",
            "--gpu-id=NVIDIA A40",
            "--gpu-count=1",
            "--compute-type=GPU",
            "--container-disk-in-gb=50",
            "--volume-mount-path=/workspace",
            "--ports=22/tcp,6006/http",
            "--cloud-type=SECURE",
            "--data-center-ids=US-GA-1",
            "--network-volume-id=volume-1",
            "--registry-auth-id=registry-1",
            "--docker-args=uv run mantis-v2 workload-execute --manifest /workspace/launch.json",
            '--env={"HF_HOME":"/workspace/mantis/inputs/bundle/cache/huggingface","HF_HUB_OFFLINE":"1","MANTIS_RUN_ID":"run-identity","MANTIS_WORKLOAD_DIGEST":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","MANTIS_WORKLOAD_MANIFEST":"/workspace/launch.json"}',
            "--terminate-after=2026-07-22T14:12:00Z",
        ]
    ]


def test_create_rejects_secret_values_in_process_argv(tmp_path: Path) -> None:
    binary = tmp_path / "runpodctl"
    binary.write_bytes(b"pinned-runpodctl")
    decision = _decision()
    decision["workload"]["environment"]["MANTIS_HEARTBEAT_TOKEN"] = "secret"
    adapter = RunpodctlCreateAdapter(
        rest=RestStub(),
        binary=binary,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        version=RUNPODCTL_VERSION,
        source_commit=RUNPODCTL_COMMIT,
    )

    with pytest.raises(RunpodctlError, match="secret"):
        adapter.create(decision, datetime(2026, 7, 22, 14, 12, tzinfo=UTC))


def test_public_official_image_omits_registry_auth_flag(tmp_path: Path) -> None:
    binary = tmp_path / "runpodctl"
    binary.write_bytes(b"pinned-runpodctl")
    calls: list[list[str]] = []
    decision = _decision()
    decision["registry_auth_id"] = ""
    adapter = RunpodctlCreateAdapter(
        rest=RestStub(),
        binary=binary,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        version=RUNPODCTL_VERSION,
        source_commit=RUNPODCTL_COMMIT,
        runner=lambda args: (
            calls.append(args)
            or subprocess.CompletedProcess(args, 0, json.dumps({"id": "pod-123"}), "")
        ),
    )

    adapter.create(decision, datetime(2026, 7, 22, 14, 12, tzinfo=UTC))

    assert not any(arg.startswith("--registry-auth-id=") for arg in calls[0])


def test_create_rejects_binary_tamper_bad_output_and_missing_workload(tmp_path: Path) -> None:
    binary = tmp_path / "runpodctl"
    binary.write_bytes(b"wrong")
    rest = RestStub()

    with pytest.raises(RunpodctlError, match="binary digest"):
        RunpodctlCreateAdapter(
            rest=rest,
            binary=binary,
            binary_sha256="0" * 64,
            version=RUNPODCTL_VERSION,
            source_commit=RUNPODCTL_COMMIT,
            runner=lambda args: subprocess.CompletedProcess(args, 0, "{}", ""),
        )

    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    adapter = RunpodctlCreateAdapter(
        rest=rest,
        binary=binary,
        binary_sha256=digest,
        version=RUNPODCTL_VERSION,
        source_commit=RUNPODCTL_COMMIT,
        runner=lambda args: subprocess.CompletedProcess(args, 0, "{}", ""),
    )
    with pytest.raises(RunpodctlError, match="returned Pod ID"):
        adapter.create(_decision(), datetime(2026, 7, 22, 14, tzinfo=UTC))

    decision = _decision()
    del decision["workload"]
    with pytest.raises(RunpodctlError, match="workload"):
        adapter.create(decision, datetime(2026, 7, 22, 14, tzinfo=UTC))
