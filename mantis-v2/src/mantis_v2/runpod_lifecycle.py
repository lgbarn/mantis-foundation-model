"""Fail-closed RunPod lifecycle control with durable local receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

from mantis_v2.runpod_rest_adapter import OPENAPI_IDENTITY, OPENAPI_SHA256, OPENAPI_VERSION

Clock = Callable[[], datetime]


class LifecycleError(RuntimeError):
    """Raised when a lifecycle transition cannot be proven safe."""

    def __init__(self, code: str, details: Mapping[str, object] | None = None) -> None:
        super().__init__(code)
        self.details = dict(details or {})


class LifecycleAdapter(Protocol):
    """Provider boundary kept separate from lifecycle state transitions."""

    def inventory(self) -> Sequence[Mapping[str, object]]: ...

    def create(
        self, decision: Mapping[str, object], deadline: datetime
    ) -> Mapping[str, object]: ...

    def status(self, pod_id: str) -> Mapping[str, object]: ...

    def terminate(self, pod_id: str) -> Mapping[str, object]: ...

    def billing(self, pod_id: str) -> Mapping[str, object] | None: ...


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LifecycleError(f"invalid_launch_decision:{field}")
    return value


def _required_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LifecycleError(f"invalid_launch_decision:{field}")
    return value


def _required_decimal(value: object, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise LifecycleError(f"invalid_launch_decision:{field}") from exc
    if not parsed.is_finite() or parsed < 0:
        raise LifecycleError(f"invalid_launch_decision:{field}")
    return parsed


def _payload_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_decision_digest(decision: Mapping[str, object]) -> str:
    decision_digest = _required_string(decision.get("decision_digest"), "decision_digest")
    if not re.fullmatch(r"[0-9a-f]{64}", decision_digest):
        raise LifecycleError("invalid_launch_decision:decision_digest")
    unsigned = {key: value for key, value in decision.items() if key != "decision_digest"}
    if _payload_digest(unsigned) != decision_digest:
        raise LifecycleError("invalid_launch_decision:decision_digest")
    return decision_digest


def _validate_launch_decision(decision: Mapping[str, object], evaluated_at: datetime) -> str:
    if decision.get("schema_version") != 1 or decision.get("allowed") is not True:
        raise LifecycleError("launch_not_approved")
    decision_digest = _validate_decision_digest(decision)
    local_digest = _required_string(decision.get("local_digest"), "local_digest")
    if not re.fullmatch(r"[0-9a-f]{64}", local_digest):
        raise LifecycleError("invalid_launch_decision:local_digest")
    for field in (
        "run_name",
        "gpu_type",
        "datacenter_id",
        "image_ref",
        "template_id",
        "registry_auth_id",
        "volume_id",
        "volume_mount_path",
    ):
        _required_string(decision.get(field), field)
    for field in (
        "gpu_count",
        "vcpu",
        "ram_gb",
        "container_disk_gb",
        "maximum_duration_seconds",
    ):
        _required_int(decision.get(field), field)
    image_ref = str(decision["image_ref"])
    if not re.search(r"@sha256:[0-9a-f]{64}$", image_ref):
        raise LifecycleError("invalid_launch_decision:image_ref")
    mount_path = str(decision["volume_mount_path"])
    if not mount_path.startswith("/") or ".." in mount_path.split("/"):
        raise LifecycleError("invalid_launch_decision:volume_mount_path")
    ports = decision.get("ports")
    if (
        not isinstance(ports, list)
        or ports != ["22/tcp"]
        or any(not isinstance(port, str) for port in ports)
    ):
        raise LifecycleError("invalid_launch_decision:ports")
    for field, expected in (
        ("openapi_identity", OPENAPI_IDENTITY),
        ("openapi_version", OPENAPI_VERSION),
        ("openapi_sha256", OPENAPI_SHA256),
    ):
        if decision.get(field) != expected:
            raise LifecycleError(f"invalid_launch_decision:{field}")
    _required_decimal(
        decision.get("observed_price_usd_per_gpu_hour"),
        "observed_price_usd_per_gpu_hour",
    )
    _required_decimal(decision.get("projected_spend_usd"), "projected_spend_usd")
    stage = _required_string(decision.get("stage"), "stage")
    if stage not in {"qualification", "production", "recovery"}:
        raise LifecycleError("invalid_launch_decision:stage")
    authorization_digest = _required_string(
        decision.get("authorization_digest"), "authorization_digest"
    )
    if not re.fullmatch(r"[0-9a-f]{64}", authorization_digest):
        raise LifecycleError("invalid_launch_decision:authorization_digest")
    try:
        expires_at = datetime.fromisoformat(
            _required_string(
                decision.get("authorization_expires_at"), "authorization_expires_at"
            ).replace("Z", "+00:00")
        ).astimezone(UTC)
    except ValueError as exc:
        raise LifecycleError("invalid_launch_decision:authorization_expires_at") from exc
    if expires_at <= evaluated_at:
        raise LifecycleError("authorization_expired")
    return decision_digest


def validate_launch_decision(decision: Mapping[str, object], evaluated_at: datetime) -> str:
    """Validate one immutable launch decision without creating a resource."""
    return _validate_launch_decision(decision, evaluated_at.astimezone(UTC))


def _publish_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise LifecycleError("receipt_already_exists") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _acquire_lock(state_root: Path, name: str = "launch.lock") -> tuple[int, Path]:
    state_root.mkdir(parents=True, exist_ok=True)
    lock_path = state_root / name
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise LifecycleError("live_pod_conflict") from exc
    os.write(descriptor, f"pid={os.getpid()}\n".encode())
    os.fsync(descriptor)
    return descriptor, lock_path


def _has_open_receipt(state_root: Path) -> bool:
    termination_dir = state_root / "receipts" / "terminations"
    return any(
        not (termination_dir / receipt.name).exists()
        for receipt in _pod_control_receipt_paths(state_root)
    )


def _pod_control_receipt_paths(state_root: Path) -> tuple[Path, ...]:
    receipts = state_root / "receipts"
    return tuple(
        path for kind in ("pods", "quarantine") for path in (receipts / kind).glob("*.json")
    )


def _has_unresolved_launch_attempt(state_root: Path) -> bool:
    attempt_dir = state_root / "attempts"
    if not attempt_dir.is_dir():
        return False
    resolved = {
        str(_read_receipt(path, path.stem).get("decision_digest"))
        for path in _pod_control_receipt_paths(state_root)
    }
    return any(path.stem not in resolved for path in attempt_dir.glob("*.json"))


def _authorization_consumed(state_root: Path, authorization_digest: str) -> bool:
    return any(
        _read_receipt(path, path.stem).get("authorization_digest") == authorization_digest
        for path in _pod_control_receipt_paths(state_root)
    )


def _read_receipt(path: Path, expected_pod_id: str) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LifecycleError("invalid_receipt") from exc
    if not isinstance(loaded, dict) or loaded.get("pod_id") != expected_pod_id:
        raise LifecycleError("invalid_receipt")
    return loaded


def _read_pod_control_receipt(state_root: Path, pod_id: str) -> dict[str, object]:
    for kind in ("pods", "quarantine"):
        path = state_root / "receipts" / kind / f"{pod_id}.json"
        if path.exists():
            return _read_receipt(path, pod_id)
    raise LifecycleError("invalid_receipt")


def _pod_identity(pod_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", pod_id):
        raise LifecycleError("invalid_pod_identity")
    return pod_id


def _pod_receipt(
    decision: Mapping[str, object],
    provider_pod: Mapping[str, object],
    *,
    started_at: datetime,
    deadline: datetime,
) -> dict[str, object]:
    pod_id = _pod_identity(_required_string(provider_pod.get("id"), "provider.id"))
    expected_fields = {
        "name": decision.get("run_name"),
        "imageName": decision.get("image_ref"),
        "templateId": decision.get("template_id"),
        "networkVolumeId": decision.get("volume_id"),
        "vcpuCount": decision.get("vcpu"),
        "memoryInGb": decision.get("ram_gb"),
    }
    for field, expected in expected_fields.items():
        if provider_pod.get(field) != expected:
            raise LifecycleError(f"incomplete_provider_response:create.{field}")
    if provider_pod.get("desiredStatus") != "RUNNING":
        raise LifecycleError("incomplete_provider_response:create.desiredStatus")
    try:
        observed_price = Decimal(str(provider_pod["costPerHr"]))
    except (KeyError, InvalidOperation, ValueError) as exc:
        raise LifecycleError("incomplete_provider_response:create.costPerHr") from exc
    approved_price = _required_decimal(
        decision.get("observed_price_usd_per_gpu_hour"),
        "observed_price_usd_per_gpu_hour",
    )
    if observed_price != approved_price:
        raise LifecycleError("provider_price_mismatch")

    return {
        "schema_version": 1,
        "pod_id": pod_id,
        "run_name": decision["run_name"],
        "decision_digest": decision["decision_digest"],
        "image_ref": decision["image_ref"],
        "template_id": decision["template_id"],
        "volume_id": decision["volume_id"],
        "vcpu": decision["vcpu"],
        "ram_gb": decision["ram_gb"],
        "stage": decision["stage"],
        "authorization_digest": decision["authorization_digest"],
        "reserved_spend_usd": str(
            _required_decimal(decision.get("projected_spend_usd"), "projected_spend_usd")
        ),
        "spend_state": "reserved",
        "observed_price_usd_per_gpu_hour": str(observed_price),
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "deadline": deadline.isoformat().replace("+00:00", "Z"),
        "status": "RUNNING",
    }


def _publish_quarantine_receipt(
    state_root: Path,
    decision: Mapping[str, object],
    provider_pod: Mapping[str, object],
    *,
    started_at: datetime,
    deadline: datetime,
    failure: LifecycleError,
) -> None:
    if provider_pod.get("name") != decision.get("run_name"):
        return
    try:
        pod_id = _pod_identity(_required_string(provider_pod.get("id"), "provider.id"))
    except LifecycleError:
        return
    receipt: dict[str, object] = {
        "schema_version": 1,
        "pod_id": pod_id,
        "run_name": decision["run_name"],
        "decision_digest": decision["decision_digest"],
        "image_ref": decision["image_ref"],
        "template_id": decision["template_id"],
        "volume_id": decision["volume_id"],
        "vcpu": decision["vcpu"],
        "ram_gb": decision["ram_gb"],
        "stage": decision["stage"],
        "authorization_digest": decision["authorization_digest"],
        "reserved_spend_usd": str(
            _required_decimal(decision.get("projected_spend_usd"), "projected_spend_usd")
        ),
        "spend_state": "reserved",
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "deadline": deadline.isoformat().replace("+00:00", "Z"),
        "status": "QUARANTINED",
        "validation_failure": str(failure),
    }
    _publish_json(state_root / "receipts" / "quarantine" / f"{pod_id}.json", receipt)


def _receipt_for_decision(state_root: Path, decision_digest: str) -> dict[str, object] | None:
    receipt_dir = state_root / "receipts" / "pods"
    if not receipt_dir.is_dir():
        return None
    for receipt_path in receipt_dir.glob("*.json"):
        loaded = _read_receipt(receipt_path, receipt_path.stem)
        if loaded.get("decision_digest") == decision_digest:
            return loaded
    return None


def _quarantine_for_decision(state_root: Path, decision_digest: str) -> dict[str, object] | None:
    receipt_dir = state_root / "receipts" / "quarantine"
    if not receipt_dir.is_dir():
        return None
    for receipt_path in receipt_dir.glob("*.json"):
        loaded = _read_receipt(receipt_path, receipt_path.stem)
        if loaded.get("decision_digest") == decision_digest:
            return loaded
    return None


def _mantis_conflicts(
    inventory: Sequence[Mapping[str, object]], expected_run_name: str
) -> list[dict[str, str]]:
    conflicts: list[dict[str, str]] = []
    for pod in inventory:
        name = str(pod.get("name", ""))
        image = str(pod.get("imageName", ""))
        if not name.lower().startswith("mantis") and "mantis" not in image.lower():
            continue
        pod_id = _pod_identity(_required_string(pod.get("id"), "inventory.id"))
        conflicts.append(
            {
                "pod_id": pod_id,
                "ownership": "managed" if name == expected_run_name else "unmanaged",
            }
        )
    return conflicts


def launch_pod(
    *,
    decision: Mapping[str, object],
    state_root: Path,
    adapter: LifecycleAdapter,
    now: Clock,
) -> dict[str, object]:
    """Create at most one Pod from an approved decision and publish its receipt."""
    started_at = now().astimezone(UTC)
    decision_digest = _validate_launch_decision(decision, started_at)

    descriptor, lock_path = _acquire_lock(state_root)
    try:
        if _has_open_receipt(state_root):
            raise LifecycleError("live_pod_conflict")
        attempt_dir = state_root / "attempts"
        if _has_unresolved_launch_attempt(state_root):
            raise LifecycleError("launch_reconciliation_required")
        authorization_digest = _required_string(
            decision.get("authorization_digest"), "authorization_digest"
        )
        if _authorization_consumed(state_root, authorization_digest):
            raise LifecycleError("authorization_replayed")
        inventory = adapter.inventory()
        conflicts = _mantis_conflicts(
            inventory, _required_string(decision.get("run_name"), "run_name")
        )
        if conflicts:
            raise LifecycleError("live_pod_conflict", {"pods": conflicts})

        duration = _required_int(
            decision.get("maximum_duration_seconds"), "maximum_duration_seconds"
        )
        deadline = started_at + timedelta(seconds=duration + 600 + 120)
        reserved_spend = _required_decimal(
            decision.get("projected_spend_usd"), "projected_spend_usd"
        )
        stage = _required_string(decision.get("stage"), "stage")
        if stage not in {"qualification", "production", "recovery"}:
            raise LifecycleError("invalid_launch_decision:stage")
        if not re.fullmatch(r"[0-9a-f]{64}", authorization_digest):
            raise LifecycleError("invalid_launch_decision:authorization_digest")
        attempt_path = attempt_dir / f"{decision_digest}.json"
        _publish_json(
            attempt_path,
            {
                "schema_version": 1,
                "decision_digest": decision_digest,
                "run_name": decision["run_name"],
                "attempted_at": started_at.isoformat().replace("+00:00", "Z"),
                "deadline": deadline.isoformat().replace("+00:00", "Z"),
                "state": "create_requested",
                "stage": stage,
                "authorization_digest": authorization_digest,
                "reserved_spend_usd": str(reserved_spend),
                "spend_state": "reserved",
            },
        )
        try:
            created = adapter.create(decision, deadline)
        except Exception as exc:
            raise LifecycleError("provider_create_outcome_unknown") from exc
        if not isinstance(created, Mapping):
            raise LifecycleError("incomplete_provider_response:create")
        try:
            receipt = _pod_receipt(decision, created, started_at=started_at, deadline=deadline)
        except LifecycleError as exc:
            _publish_quarantine_receipt(
                state_root,
                decision,
                created,
                started_at=started_at,
                deadline=deadline,
                failure=exc,
            )
            raise
        pod_id = str(receipt["pod_id"])
        _publish_json(state_root / "receipts" / "pods" / f"{pod_id}.json", receipt)
        return receipt
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def reconcile_launch(
    *,
    decision: Mapping[str, object],
    state_root: Path,
    adapter: LifecycleAdapter,
    now: Clock,
) -> dict[str, object]:
    """Bind an uncertain create outcome to fresh inventory without retrying create."""
    decision_digest = _validate_decision_digest(decision)
    quarantine = _quarantine_for_decision(state_root, decision_digest)
    if quarantine is not None:
        raise LifecycleError(
            "provider_create_requires_termination", {"pod_id": quarantine["pod_id"]}
        )
    existing = _receipt_for_decision(state_root, decision_digest)
    if existing is not None:
        return existing

    descriptor, lock_path = _acquire_lock(state_root)
    try:
        existing = _receipt_for_decision(state_root, decision_digest)
        if existing is not None:
            return existing
        quarantine = _quarantine_for_decision(state_root, decision_digest)
        if quarantine is not None:
            raise LifecycleError(
                "provider_create_requires_termination", {"pod_id": quarantine["pod_id"]}
            )
        attempt_path = state_root / "attempts" / f"{decision_digest}.json"
        try:
            attempt = json.loads(attempt_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise LifecycleError("launch_attempt_not_found") from exc
        if not isinstance(attempt, dict) or attempt.get("decision_digest") != decision_digest:
            raise LifecycleError("invalid_launch_attempt")
        try:
            started_at = datetime.fromisoformat(str(attempt["attempted_at"]).replace("Z", "+00:00"))
            deadline = datetime.fromisoformat(str(attempt["deadline"]).replace("Z", "+00:00"))
        except (KeyError, ValueError) as exc:
            raise LifecycleError("invalid_launch_attempt") from exc

        expected_name = _required_string(decision.get("run_name"), "run_name")
        matches = [pod for pod in adapter.inventory() if pod.get("name") == expected_name]
        if not matches:
            raise LifecycleError("provider_create_outcome_unresolved")
        if len(matches) != 1:
            raise LifecycleError("live_pod_conflict")
        try:
            receipt = _pod_receipt(decision, matches[0], started_at=started_at, deadline=deadline)
        except LifecycleError as exc:
            _publish_quarantine_receipt(
                state_root,
                decision,
                matches[0],
                started_at=started_at,
                deadline=deadline,
                failure=exc,
            )
            raise
        receipt["reconciled_at"] = now().astimezone(UTC).isoformat().replace("+00:00", "Z")
        pod_id = str(receipt["pod_id"])
        _publish_json(state_root / "receipts" / "pods" / f"{pod_id}.json", receipt)
        return receipt
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def pod_status(
    *,
    pod_id: str,
    run_name: str,
    state_root: Path,
    adapter: LifecycleAdapter,
    now: Clock,
) -> dict[str, object]:
    """Return an allowlisted status for one receipt-bound Pod and run identity."""
    exact_pod_id = _pod_identity(pod_id)
    pod_receipt = _read_pod_control_receipt(state_root, exact_pod_id)
    if pod_receipt.get("run_name") != run_name:
        raise LifecycleError("run_identity_mismatch")
    try:
        provider_status = adapter.status(exact_pod_id)
    except Exception as exc:
        raise LifecycleError("provider_status_failed") from exc
    if not isinstance(provider_status, Mapping):
        raise LifecycleError("incomplete_provider_response:status")
    if provider_status.get("id") != exact_pod_id or provider_status.get("name") != run_name:
        raise LifecycleError("provider_identity_mismatch")
    desired_status = _required_string(provider_status.get("desiredStatus"), "status.desiredStatus")
    try:
        cost_per_hour = Decimal(str(provider_status["costPerHr"]))
    except (KeyError, InvalidOperation, ValueError) as exc:
        raise LifecycleError("incomplete_provider_response:status.costPerHr") from exc
    observed_at = now().astimezone(UTC)
    try:
        started_at = datetime.fromisoformat(
            str(pod_receipt["started_at"]).replace("Z", "+00:00")
        ).astimezone(UTC)
    except (KeyError, ValueError) as exc:
        raise LifecycleError("invalid_receipt") from exc
    uptime_seconds = max(0, int((observed_at - started_at).total_seconds()))
    return {
        "schema_version": 1,
        "pod_id": exact_pod_id,
        "run_name": run_name,
        "decision_digest": pod_receipt["decision_digest"],
        "desired_status": desired_status,
        "cost_per_hour_usd": str(cost_per_hour),
        "uptime_seconds": uptime_seconds,
        "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
    }


def _terminate_receipted(
    *,
    pod_receipt: Mapping[str, object],
    state_root: Path,
    adapter: LifecycleAdapter,
    observed_at: datetime,
    reason: str,
) -> dict[str, object]:
    exact_pod_id = _pod_identity(str(pod_receipt["pod_id"]))
    termination_path = state_root / "receipts" / "terminations" / f"{exact_pod_id}.json"
    if termination_path.exists():
        return _read_receipt(termination_path, exact_pod_id)

    try:
        descriptor, lock_path = _acquire_lock(state_root, f"terminate-{exact_pod_id}.lock")
    except LifecycleError as exc:
        if termination_path.exists():
            return _read_receipt(termination_path, exact_pod_id)
        raise LifecycleError("termination_in_progress") from exc
    try:
        if termination_path.exists():
            return _read_receipt(termination_path, exact_pod_id)
        attempt_path = state_root / "attempts" / "terminations" / f"{exact_pod_id}.json"
        if attempt_path.exists():
            raise LifecycleError("termination_reconciliation_required")
        _publish_json(
            attempt_path,
            {
                "schema_version": 1,
                "pod_id": exact_pod_id,
                "run_name": pod_receipt["run_name"],
                "decision_digest": pod_receipt["decision_digest"],
                "attempted_at": observed_at.isoformat().replace("+00:00", "Z"),
                "reason": reason,
            },
        )
        try:
            result = adapter.terminate(exact_pod_id)
        except Exception as exc:
            raise LifecycleError("provider_terminate_outcome_unknown") from exc
        if result.get("deleted") is not True or result.get("id") != exact_pod_id:
            raise LifecycleError("incomplete_provider_response:terminate")
        receipt: dict[str, object] = {
            "schema_version": 1,
            "pod_id": exact_pod_id,
            "run_name": pod_receipt["run_name"],
            "decision_digest": pod_receipt["decision_digest"],
            "started_at": pod_receipt["started_at"],
            "deadline": pod_receipt["deadline"],
            "terminated_at": observed_at.isoformat().replace("+00:00", "Z"),
            "reason": reason,
            "provider_deleted": True,
        }
        _publish_json(termination_path, receipt)
        return receipt
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def terminate_pod(
    *,
    pod_id: str,
    run_name: str,
    state_root: Path,
    adapter: LifecycleAdapter,
    now: Clock,
) -> dict[str, object]:
    """Terminate one receipt-bound exact Pod/run pair, idempotently."""
    exact_pod_id = _pod_identity(pod_id)
    pod_receipt = _read_pod_control_receipt(state_root, exact_pod_id)
    if pod_receipt.get("run_name") != run_name:
        raise LifecycleError("run_identity_mismatch")
    return _terminate_receipted(
        pod_receipt=pod_receipt,
        state_root=state_root,
        adapter=adapter,
        observed_at=now().astimezone(UTC),
        reason="operator_requested",
    )


def terminate_pod_for_reason(
    *,
    pod_id: str,
    run_name: str,
    reason: str,
    state_root: Path,
    adapter: LifecycleAdapter,
    now: Clock,
) -> dict[str, object]:
    """Terminate a receipt-bound Pod for one machine-owned fail-closed reason."""
    allowed = {
        "workload_complete",
        "startup_heartbeat_timeout",
        "heartbeat_missed",
        "workload_blocked",
        "workload_failed",
        "provider_not_running",
        "supervisor_error",
    }
    if reason not in allowed:
        raise LifecycleError("invalid_termination_reason")
    exact_pod_id = _pod_identity(pod_id)
    pod_receipt = _read_pod_control_receipt(state_root, exact_pod_id)
    if pod_receipt.get("run_name") != run_name:
        raise LifecycleError("run_identity_mismatch")
    return _terminate_receipted(
        pod_receipt=pod_receipt,
        state_root=state_root,
        adapter=adapter,
        observed_at=now().astimezone(UTC),
        reason=reason,
    )


def reconcile_termination(
    *,
    pod_id: str,
    run_name: str,
    state_root: Path,
    adapter: LifecycleAdapter,
    now: Clock,
) -> dict[str, object]:
    """Resolve an uncertain termination through fresh provider inventory."""
    exact_pod_id = _pod_identity(pod_id)
    pod_receipt = _read_pod_control_receipt(state_root, exact_pod_id)
    if pod_receipt.get("run_name") != run_name:
        raise LifecycleError("run_identity_mismatch")
    termination_path = state_root / "receipts" / "terminations" / f"{exact_pod_id}.json"
    if termination_path.exists():
        return _read_receipt(termination_path, exact_pod_id)

    descriptor, lock_path = _acquire_lock(state_root, f"terminate-{exact_pod_id}.lock")
    try:
        if termination_path.exists():
            return _read_receipt(termination_path, exact_pod_id)
        attempt_path = state_root / "attempts" / "terminations" / f"{exact_pod_id}.json"
        try:
            attempt = json.loads(attempt_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise LifecycleError("termination_attempt_not_found") from exc
        if (
            not isinstance(attempt, dict)
            or attempt.get("pod_id") != exact_pod_id
            or attempt.get("run_name") != run_name
        ):
            raise LifecycleError("invalid_termination_attempt")
        try:
            inventory = adapter.inventory()
        except Exception as exc:
            raise LifecycleError("provider_inventory_failed") from exc
        if any(pod.get("id") == exact_pod_id for pod in inventory):
            raise LifecycleError("provider_terminate_outcome_unresolved")
        observed_at = now().astimezone(UTC)
        receipt: dict[str, object] = {
            "schema_version": 1,
            "pod_id": exact_pod_id,
            "run_name": run_name,
            "decision_digest": pod_receipt["decision_digest"],
            "started_at": pod_receipt["started_at"],
            "deadline": pod_receipt["deadline"],
            "terminated_at": observed_at.isoformat().replace("+00:00", "Z"),
            "reason": attempt["reason"],
            "provider_deleted": True,
            "reconciliation_evidence": "fresh_inventory_absence",
        }
        _publish_json(termination_path, receipt)
        return receipt
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def reconcile_spend(
    *,
    pod_id: str,
    run_name: str,
    state_root: Path,
    adapter: LifecycleAdapter,
    now: Clock,
) -> dict[str, object]:
    """Move reserved spend to actual only from receipt-backed billing evidence."""
    exact_pod_id = _pod_identity(pod_id)
    pod_receipt = _read_pod_control_receipt(state_root, exact_pod_id)
    if pod_receipt.get("run_name") != run_name:
        raise LifecycleError("run_identity_mismatch")
    termination_path = state_root / "receipts" / "terminations" / f"{exact_pod_id}.json"
    if not termination_path.exists():
        raise LifecycleError("termination_receipt_required")
    termination_receipt = _read_receipt(termination_path, exact_pod_id)
    spend_path = state_root / "receipts" / "spend" / f"{exact_pod_id}.json"
    if spend_path.exists():
        return _read_receipt(spend_path, exact_pod_id)

    try:
        descriptor, lock_path = _acquire_lock(state_root, f"spend-{exact_pod_id}.lock")
    except LifecycleError as exc:
        if spend_path.exists():
            return _read_receipt(spend_path, exact_pod_id)
        raise LifecycleError("spend_reconciliation_in_progress") from exc
    try:
        if spend_path.exists():
            return _read_receipt(spend_path, exact_pod_id)
        try:
            billing = adapter.billing(exact_pod_id)
        except Exception as exc:
            raise LifecycleError("provider_billing_failed") from exc
        if billing is None:
            raise LifecycleError("provider_billing_pending")
        if not isinstance(billing, Mapping) or billing.get("pod_id") != exact_pod_id:
            raise LifecycleError("provider_identity_mismatch")
        actual_spend = _required_decimal(billing.get("actual_cost_usd"), "billing.actual_cost_usd")
        reserved_spend = _required_decimal(
            pod_receipt.get("reserved_spend_usd"), "receipt.reserved_spend_usd"
        )
        receipt: dict[str, object] = {
            "schema_version": 1,
            "pod_id": exact_pod_id,
            "run_name": run_name,
            "decision_digest": pod_receipt["decision_digest"],
            "authorization_digest": pod_receipt["authorization_digest"],
            "stage": pod_receipt["stage"],
            "reserved_spend_usd": str(reserved_spend),
            "actual_spend_usd": str(actual_spend),
            "spend_state": "reconciled",
            "pod_receipt_digest": _payload_digest(pod_receipt),
            "termination_receipt_digest": _payload_digest(termination_receipt),
            "reconciled_at": now().astimezone(UTC).isoformat().replace("+00:00", "Z"),
        }
        _publish_json(spend_path, receipt)
        return receipt
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def enforce_deadline(
    *,
    pod_id: str,
    state_root: Path,
    adapter: LifecycleAdapter,
    now: Clock,
) -> dict[str, object]:
    """Terminate one exact Pod after its durable deadline, idempotently."""
    exact_pod_id = _pod_identity(pod_id)
    pod_receipt = _read_pod_control_receipt(state_root, exact_pod_id)
    try:
        deadline = datetime.fromisoformat(str(pod_receipt["deadline"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise LifecycleError("invalid_receipt") from exc
    observed_at = now().astimezone(UTC)
    if observed_at < deadline:
        raise LifecycleError("deadline_not_reached")

    return _terminate_receipted(
        pod_receipt=pod_receipt,
        state_root=state_root,
        adapter=adapter,
        observed_at=observed_at,
        reason="deadline_exceeded",
    )
