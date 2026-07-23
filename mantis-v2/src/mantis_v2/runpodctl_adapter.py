"""Pinned runpodctl create seam with REST-authoritative lifecycle operations."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

RUNPODCTL_VERSION = "2.7.2"
RUNPODCTL_COMMIT = "309512b4926eb7d218bbc8a8f11d380ce54f59c4"
RUNPODCTL_DARWIN_ARM64_SHA256 = "a016e442fdf12e4642ad3425ea6d624a40882d77accdfa043b5e40a4fd08d037"
RUNPODCTL_RELEASE_URL = "https://github.com/runpod/runpodctl/releases/tag/v2.7.2"


class RunpodctlError(RuntimeError):
    """Raised when the pinned create boundary cannot prove the created Pod."""


class RestLifecycle(Protocol):
    def inventory(self) -> Sequence[Mapping[str, object]]: ...

    def status(self, pod_id: str) -> Mapping[str, object]: ...

    def terminate(self, pod_id: str) -> Mapping[str, object]: ...

    def billing(self, pod_id: str) -> Mapping[str, object] | None: ...


Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RunpodctlError("runpodctl binary is unavailable") from exc
    return digest.hexdigest()


def _default_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunpodctlError(f"invalid create field: {field}")
    return value


def _required_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RunpodctlError(f"invalid create field: {field}")
    return value


class RunpodctlCreateAdapter:
    """Use pinned runpodctl only for deadline-backed create; delegate control to REST."""

    def __init__(
        self,
        *,
        rest: RestLifecycle,
        binary: str | Path,
        binary_sha256: str,
        version: str,
        source_commit: str,
        runner: Runner = _default_runner,
    ) -> None:
        self.rest = rest
        self.binary = Path(binary)
        if version != RUNPODCTL_VERSION or source_commit != RUNPODCTL_COMMIT:
            raise RunpodctlError("runpodctl source identity mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", binary_sha256):
            raise RunpodctlError("runpodctl binary digest is invalid")
        if _sha256_file(self.binary) != binary_sha256:
            raise RunpodctlError("runpodctl binary digest mismatch")
        self.binary_sha256 = binary_sha256
        self.runner = runner

    def inventory(self) -> Sequence[Mapping[str, object]]:
        return self.rest.inventory()

    def status(self, pod_id: str) -> Mapping[str, object]:
        return self.rest.status(pod_id)

    def terminate(self, pod_id: str) -> Mapping[str, object]:
        return self.rest.terminate(pod_id)

    def billing(self, pod_id: str) -> Mapping[str, object] | None:
        return self.rest.billing(pod_id)

    def create(self, decision: Mapping[str, object], deadline: datetime) -> Mapping[str, object]:
        workload = decision.get("workload")
        if not isinstance(workload, Mapping):
            raise RunpodctlError("launch decision workload is missing")
        docker_args = _required_text(workload.get("docker_args"), "workload.docker_args")
        environment = workload.get("environment")
        if (
            not isinstance(environment, Mapping)
            or not environment
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in environment.items()
            )
        ):
            raise RunpodctlError("launch decision workload environment is invalid")
        allowed_environment = {
            "MANTIS_RUN_ID",
            "MANTIS_WORKLOAD_MANIFEST",
            "MANTIS_BASE_IMAGE",
            "MANTIS_SOURCE_REVISION",
            "MANTIS_SOURCE_TREE",
            "MANTIS_LOCK_SHA256",
            "MANTIS_LOCK_PATH",
            "MANTIS_IMAGE_CONTRACT_SHA256",
            "MANTIS_UV_VERSION",
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "UV_CACHE_DIR",
            "UV_PROJECT_ENVIRONMENT",
        }
        if not set(environment).issubset(allowed_environment):
            raise RunpodctlError("launch decision environment would expose a secret or unknown key")
        ports = decision.get("ports")
        if (
            not isinstance(ports, list)
            or not ports
            or any(not isinstance(port, str) for port in ports)
        ):
            raise RunpodctlError("invalid create field: ports")
        deadline_utc = deadline.astimezone(UTC).isoformat().replace("+00:00", "Z")
        args = [
            str(self.binary),
            "pod",
            "create",
            "--output=json",
            f"--name={_required_text(decision.get('run_name'), 'run_name')}",
            f"--image={_required_text(decision.get('image_ref'), 'image_ref')}",
            f"--template-id={_required_text(decision.get('template_id'), 'template_id')}",
            f"--gpu-id={_required_text(decision.get('gpu_type'), 'gpu_type')}",
            f"--gpu-count={_required_int(decision.get('gpu_count'), 'gpu_count')}",
            "--compute-type=GPU",
            "--container-disk-in-gb="
            f"{_required_int(decision.get('container_disk_gb'), 'container_disk_gb')}",
            "--volume-mount-path="
            f"{_required_text(decision.get('volume_mount_path'), 'volume_mount_path')}",
            f"--ports={','.join(ports)}",
            "--cloud-type=SECURE",
            f"--data-center-ids={_required_text(decision.get('datacenter_id'), 'datacenter_id')}",
            f"--network-volume-id={_required_text(decision.get('volume_id'), 'volume_id')}",
            f"--docker-args={docker_args}",
            "--env=" + json.dumps(environment, sort_keys=True, separators=(",", ":")),
            f"--terminate-after={deadline_utc}",
        ]
        registry_auth_id = decision.get("registry_auth_id")
        if not isinstance(registry_auth_id, str):
            raise RunpodctlError("invalid create field: registry_auth_id")
        if registry_auth_id:
            args.insert(-3, f"--registry-auth-id={registry_auth_id}")
        completed = self.runner(args)
        if completed.returncode != 0:
            raise RunpodctlError("runpodctl create failed")
        try:
            output: Any = json.loads(completed.stdout)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RunpodctlError("runpodctl create returned invalid JSON") from exc
        pod_id = output.get("id") if isinstance(output, dict) else None
        if not isinstance(pod_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", pod_id
        ):
            raise RunpodctlError("runpodctl create returned Pod ID is invalid")
        status = self.rest.status(pod_id)
        if status.get("id") != pod_id or status.get("name") != decision.get("run_name"):
            raise RunpodctlError("REST status does not match returned Pod ID")
        return status
