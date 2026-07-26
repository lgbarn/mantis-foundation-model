"""Fresh planner inputs from pinned, read-only runpodctl commands."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from mantis_v2.frozen_expected_r import _paid_control_config
from mantis_v2.runpod_config import (
    canonical_json,
    load_launch_intent,
    load_local_config,
    load_platform_config,
)
from mantis_v2.runpodctl_adapter import (
    RUNPODCTL_COMMIT,
    RUNPODCTL_VERSION,
)


class RunpodSnapshotError(RuntimeError):
    """Raised when current provider or receipt state cannot prove a safe launch."""


Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _default_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True, timeout=120)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RunpodSnapshotError("runpodctl binary is unavailable") from exc
    return digest.hexdigest()


def _decimal(value: object, field: str) -> Decimal:
    if isinstance(value, bool):
        raise RunpodSnapshotError(f"{field} is invalid")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RunpodSnapshotError(f"{field} is invalid") from exc
    if not parsed.is_finite() or parsed < 0:
        raise RunpodSnapshotError(f"{field} is invalid")
    return parsed


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RunpodSnapshotError(f"{field} is invalid")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunpodSnapshotError(f"{field} is invalid")
    return value


def _json_list(completed: subprocess.CompletedProcess[str], command: str) -> list[Any]:
    if completed.returncode != 0:
        raise RunpodSnapshotError(f"runpodctl {command} failed")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunpodSnapshotError(f"runpodctl {command} returned invalid JSON") from exc
    if not isinstance(value, list):
        raise RunpodSnapshotError(f"runpodctl {command} returned invalid JSON")
    return value


def _json_object(completed: subprocess.CompletedProcess[str], command: str) -> dict[str, Any]:
    if completed.returncode != 0:
        raise RunpodSnapshotError(f"runpodctl {command} failed")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunpodSnapshotError(f"runpodctl {command} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise RunpodSnapshotError(f"runpodctl {command} returned invalid JSON")
    return value


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunpodSnapshotError(f"invalid lifecycle receipt: {path}") from exc
    if not isinstance(value, dict):
        raise RunpodSnapshotError(f"invalid lifecycle receipt: {path}")
    return value


def _reconciled_ledger(state_root: Path) -> tuple[dict[str, Any], dict[str, int]]:
    receipts = state_root / "receipts"
    pod_paths = sorted((receipts / "pods").glob("*.json")) + sorted(
        (receipts / "quarantine").glob("*.json")
    )
    spend_paths = {path.stem: path for path in (receipts / "spend").glob("*.json")}
    actual = {key: Decimal("0") for key in ("storage", "qualification", "production", "recovery")}
    consumed: set[str] = set()
    seen_pods: set[str] = set()
    for pod_path in pod_paths:
        pod = _read_receipt(pod_path)
        pod_id = _text(pod.get("pod_id"), "receipt.pod_id")
        if pod_id != pod_path.stem or pod_id in seen_pods:
            raise RunpodSnapshotError("lifecycle Pod receipt identity is invalid")
        seen_pods.add(pod_id)
        termination_path = receipts / "terminations" / f"{pod_id}.json"
        spend_path = spend_paths.get(pod_id)
        if not termination_path.is_file() or spend_path is None:
            raise RunpodSnapshotError(f"lifecycle spend is not reconciled for Pod {pod_id}")
        termination = _read_receipt(termination_path)
        spend = _read_receipt(spend_path)
        if termination.get("pod_id") != pod_id or spend.get("pod_id") != pod_id:
            raise RunpodSnapshotError("lifecycle receipt identity is invalid")
        stage = _text(spend.get("stage"), "spend.stage")
        if stage not in {"qualification", "production", "recovery"}:
            raise RunpodSnapshotError("spend.stage is invalid")
        if spend.get("spend_state") != "reconciled":
            raise RunpodSnapshotError(f"lifecycle spend is not reconciled for Pod {pod_id}")
        actual[stage] += _decimal(spend.get("actual_spend_usd"), "spend.actual_spend_usd")
        consumed.add(_text(pod.get("authorization_digest"), "receipt.authorization_digest"))
    if set(spend_paths) != seen_pods:
        raise RunpodSnapshotError("orphan lifecycle spend receipt")
    total = sum(actual.values(), start=Decimal("0"))
    zeros = {key: "0" for key in actual}
    ledger = {
        "schema_version": 1,
        "actual_spend_usd": str(total),
        "reserved_spend_usd": "0",
        "bucket_actual_spend_usd": {key: str(value) for key, value in actual.items()},
        "bucket_reserved_spend_usd": zeros,
        "active_reservations": [],
        "consumed_authorization_digests": sorted(consumed),
    }
    return ledger, {"pod_receipts": len(pod_paths), "spend_receipts": len(spend_paths)}


def _publish(path: Path, value: Mapping[str, object]) -> str:
    encoded = (canonical_json(dict(value)) + "\n").encode()
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RunpodSnapshotError(f"snapshot output already exists: {path}") from exc
    return _sha256_bytes(encoded)


def write_provider_snapshot(
    *,
    platform_path: Path,
    local_path: Path,
    control_path: Path,
    intent_path: Path,
    runpodctl_binary: Path,
    output_root: Path,
    runner: Runner = _default_runner,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, object]:
    """Write fresh inventory, ledger, and provenance without provider mutation."""
    platform = load_platform_config(platform_path)
    local = load_local_config(local_path)
    control = _paid_control_config(control_path)
    intent = load_launch_intent(intent_path)
    if socket.gethostname() != local.controller.hostname:
        raise RunpodSnapshotError("controller hostname mismatch")
    if not os.environ.get(local.secrets.runpod_api_key_env):
        raise RunpodSnapshotError("RUNPOD_API_KEY is not configured")
    expected_cli = control["runpodctl"]
    if (
        expected_cli["version"] != RUNPODCTL_VERSION
        or expected_cli["source_commit"] != RUNPODCTL_COMMIT
        or _sha256_file(runpodctl_binary) != expected_cli["binary_sha256"]
    ):
        raise RunpodSnapshotError("runpodctl source identity mismatch")

    commands = {
        "user": [str(runpodctl_binary), "user", "--output=json"],
        "pods": [str(runpodctl_binary), "pod", "list", "--output=json"],
        "gpus": [
            str(runpodctl_binary),
            "gpu",
            "list",
            "--include-unavailable",
            "--output=json",
        ],
        "datacenters": [str(runpodctl_binary), "datacenter", "list", "--output=json"],
        "volumes": [str(runpodctl_binary), "network-volume", "list", "--output=json"],
    }
    completed = {name: runner(command) for name, command in commands.items()}
    user = _json_object(completed["user"], "user")
    pods = _json_list(completed["pods"], "pod list")
    gpus = _json_list(completed["gpus"], "gpu list")
    datacenters = _json_list(completed["datacenters"], "datacenter list")
    volumes = _json_list(completed["volumes"], "network-volume list")

    live_pods = sorted(_text(pod.get("id"), "pod.id") for pod in pods if isinstance(pod, dict))
    if len(live_pods) != len(pods) or live_pods:
        raise RunpodSnapshotError("zero live Pods required before launch planning")
    current_spend = _decimal(user.get("currentSpendPerHr"), "user.currentSpendPerHr")
    provider = control["provider"]
    storage_spend_ceiling = Decimal(int(provider["volume_size_gb"])) * _decimal(
        provider["storage_usd_per_gb_hour"], "storage hourly rate"
    )
    if current_spend > storage_spend_ceiling:
        raise RunpodSnapshotError(
            "provider current spend exceeds the configured storage-only bound"
        )
    balance = _decimal(user.get("clientBalance"), "user.clientBalance")

    matching_gpu = next(
        (
            gpu
            for gpu in gpus
            if isinstance(gpu, dict)
            and (gpu.get("gpuId") == intent.gpu_type or gpu.get("displayName") == intent.gpu_type)
        ),
        None,
    )
    if (
        matching_gpu is None
        or matching_gpu.get("secureCloud") is not True
        or matching_gpu.get("available") is not True
    ):
        raise RunpodSnapshotError("requested secure GPU is absent from runpodctl inventory")
    matching_dc = next(
        (dc for dc in datacenters if isinstance(dc, dict) and dc.get("id") == intent.datacenter_id),
        None,
    )
    if matching_dc is None or not isinstance(matching_dc.get("gpuAvailability"), list):
        raise RunpodSnapshotError("requested datacenter is absent from runpodctl inventory")
    gpu_ids = {intent.gpu_type, str(matching_gpu.get("gpuId"))}
    available = any(
        isinstance(item, dict)
        and item.get("gpuId", item.get("gpuTypeId")) in gpu_ids
        and item.get("stockStatus") in {"Low", "Medium", "High"}
        for item in matching_dc["gpuAvailability"]
    )
    if not available:
        raise RunpodSnapshotError("requested GPU is unavailable in the requested datacenter")
    matching_volume = next(
        (
            volume
            for volume in volumes
            if isinstance(volume, dict) and volume.get("id") == intent.volume_id
        ),
        None,
    )
    if matching_volume is None:
        raise RunpodSnapshotError("requested network volume is absent from runpodctl inventory")
    volume_size = _integer(matching_volume.get("size"), "volume.size")
    volume_dc = _text(matching_volume.get("dataCenterId"), "volume.dataCenterId")
    if volume_size != intent.volume_size_gb or volume_dc != intent.datacenter_id:
        raise RunpodSnapshotError("network volume identity differs from launch intent")
    if (
        provider["volume_id"] != intent.volume_id
        or provider["volume_size_gb"] != intent.volume_size_gb
        or provider["datacenter_id"] != intent.datacenter_id
    ):
        raise RunpodSnapshotError("paid control provider identity differs from launch intent")

    observed_at = now().astimezone(UTC).isoformat().replace("+00:00", "Z")
    inventory = {
        "schema_version": 1,
        "observed_at": observed_at,
        "account_balance_usd": str(balance),
        "offers": [
            {
                "gpu_type": intent.gpu_type,
                "datacenter_id": intent.datacenter_id,
                "price_usd_per_gpu_hour": str(_decimal(provider["hourly_rate_usd"], "hourly rate")),
                "available": available,
                "cloud_type": "secure",
            }
        ],
        "volumes": [
            {
                "volume_id": intent.volume_id,
                "datacenter_id": intent.datacenter_id,
                "size_gb": volume_size,
                "free_bytes": platform.storage.minimum_free_bytes,
            }
        ],
        "live_pods": live_pods,
    }
    ledger, receipt_counts = _reconciled_ledger(local.paths.state_root)
    if output_root.exists():
        raise RunpodSnapshotError(f"snapshot output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    inventory_path = temporary_root / "inventory.json"
    ledger_path = temporary_root / "spend-ledger.json"
    command_evidence = {
        name: {
            "argv": command[1:],
            "stdout_sha256": _sha256_bytes(completed[name].stdout.encode()),
        }
        for name, command in commands.items()
    }
    provenance_path = temporary_root / "provenance.json"
    try:
        inventory_sha = _publish(inventory_path, inventory)
        ledger_sha = _publish(ledger_path, ledger)
        provenance = {
            "schema_version": 1,
            "observed_at": observed_at,
            "runpodctl": {
                "version": expected_cli["version"],
                "source_commit": expected_cli["source_commit"],
                "binary_sha256": expected_cli["binary_sha256"],
            },
            "commands": command_evidence,
            "provider_checks": {
                "live_pod_count": 0,
                "current_spend_per_hour_usd": str(current_spend),
                "storage_spend_ceiling_per_hour_usd": str(storage_spend_ceiling),
            },
            "bindings": {
                "offer_price": "paid_control.provider.hourly_rate_usd",
                "free_bytes": "platform.storage.minimum_free_bytes",
            },
            "receipts": receipt_counts,
            "inventory_sha256": inventory_sha,
            "spend_ledger_sha256": ledger_sha,
        }
        provenance_sha = _publish(provenance_path, provenance)
        temporary_root.rename(output_root)
    except (OSError, RunpodSnapshotError) as exc:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise RunpodSnapshotError("provider snapshot publication failed") from exc
    inventory_path = output_root / inventory_path.name
    ledger_path = output_root / ledger_path.name
    provenance_path = output_root / provenance_path.name
    return {
        "inventory": str(inventory_path),
        "inventory_sha256": inventory_sha,
        "spend_ledger": str(ledger_path),
        "spend_ledger_sha256": ledger_sha,
        "provenance": str(provenance_path),
        "provenance_sha256": provenance_sha,
        "observed_at": observed_at,
    }
