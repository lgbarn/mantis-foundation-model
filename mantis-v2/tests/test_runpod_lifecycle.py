from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mantis_v2.runpod_lifecycle import (
    LifecycleError,
    enforce_deadline,
    launch_pod,
    pod_status,
    reconcile_launch,
    reconcile_termination,
    terminate_pod,
)


def _approved_decision() -> dict[str, object]:
    return {
        "schema_version": 1,
        "allowed": True,
        "decision_digest": "d" * 64,
        "run_name": "mantisv2-cuda-qualification-seed42",
        "gpu_type": "NVIDIA A40",
        "gpu_count": 1,
        "vcpu": 8,
        "ram_gb": 32,
        "datacenter_id": "US-CA-2",
        "container_disk_gb": 50,
        "maximum_duration_seconds": 7200,
        "projected_spend_usd": "0.88",
        "observed_price_usd_per_gpu_hour": "0.44",
        "image_ref": "registry.example/mantis@sha256:" + "a" * 64,
        "template_id": "template-fixture",
        "volume_id": "volume-fixture",
        "volume_mount_path": "/workspace",
    }


def test_concurrent_launches_create_once_and_publish_one_receipt(tmp_path: Path) -> None:
    create_entered = threading.Event()
    allow_create = threading.Event()
    calls: list[str] = []
    calls_lock = threading.Lock()

    class FakeAdapter:
        def inventory(self) -> list[dict[str, object]]:
            with calls_lock:
                calls.append("inventory")
            return []

        def create(self, decision: dict[str, object], deadline: datetime) -> dict[str, object]:
            with calls_lock:
                calls.append("create")
            create_entered.set()
            assert allow_create.wait(timeout=2)
            assert deadline == datetime(2026, 7, 21, 14, 0, tzinfo=UTC)
            return {
                "id": "pod-001",
                "name": decision["run_name"],
                "desiredStatus": "RUNNING",
                "imageName": decision["image_ref"],
                "templateId": decision["template_id"],
                "networkVolumeId": decision["volume_id"],
                "costPerHr": 0.44,
                "vcpuCount": 8,
                "memoryInGb": 32,
            }

    state_root = tmp_path / "state"
    adapter = FakeAdapter()

    def now() -> datetime:
        return datetime(2026, 7, 21, 12, 0, tzinfo=UTC)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            launch_pod,
            decision=_approved_decision(),
            state_root=state_root,
            adapter=adapter,
            now=now,
        )
        assert create_entered.wait(timeout=2)
        second = pool.submit(
            launch_pod,
            decision=_approved_decision(),
            state_root=state_root,
            adapter=adapter,
            now=now,
        )
        with pytest.raises(LifecycleError, match="^live_pod_conflict$"):
            second.result(timeout=2)
        allow_create.set()
        receipt = first.result(timeout=2)

    assert receipt["pod_id"] == "pod-001"
    assert calls.count("create") == 1
    receipts = list((state_root / "receipts" / "pods").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text()) == receipt


def test_deadline_watchdog_terminates_once_and_reuses_receipt(tmp_path: Path) -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.terminate_calls: list[str] = []

        def inventory(self) -> list[dict[str, object]]:
            return []

        def create(self, decision: dict[str, object], deadline: datetime) -> dict[str, object]:
            return {
                "id": "pod-deadline",
                "name": decision["run_name"],
                "desiredStatus": "RUNNING",
                "imageName": decision["image_ref"],
                "templateId": decision["template_id"],
                "networkVolumeId": decision["volume_id"],
                "costPerHr": 0.44,
                "vcpuCount": 8,
                "memoryInGb": 32,
            }

        def terminate(self, pod_id: str) -> dict[str, object]:
            self.terminate_calls.append(pod_id)
            return {"deleted": True, "id": pod_id}

    adapter = FakeAdapter()
    state_root = tmp_path / "state"
    current = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
    receipt = launch_pod(
        decision=_approved_decision(),
        state_root=state_root,
        adapter=adapter,
        now=lambda: current,
    )

    current = datetime(2026, 7, 21, 14, 0, 1, tzinfo=UTC)
    first = enforce_deadline(
        pod_id=str(receipt["pod_id"]),
        state_root=state_root,
        adapter=adapter,
        now=lambda: current,
    )
    second = enforce_deadline(
        pod_id=str(receipt["pod_id"]),
        state_root=state_root,
        adapter=adapter,
        now=lambda: current,
    )

    assert adapter.terminate_calls == ["pod-deadline"]
    assert second == first
    assert first["pod_id"] == "pod-deadline"
    assert first["reason"] == "deadline_exceeded"
    termination_path = state_root / "receipts" / "terminations" / "pod-deadline.json"
    assert json.loads(termination_path.read_text()) == first


def test_unknown_create_outcome_never_allows_second_create(tmp_path: Path) -> None:
    class UnknownCreateAdapter:
        def __init__(self) -> None:
            self.create_calls = 0

        def inventory(self) -> list[dict[str, object]]:
            return []

        def create(self, decision: dict[str, object], deadline: datetime) -> dict[str, object]:
            self.create_calls += 1
            raise TimeoutError("RUNPOD_API_KEY=must-not-escape")

    adapter = UnknownCreateAdapter()
    state_root = tmp_path / "state"

    with pytest.raises(LifecycleError, match="^provider_create_outcome_unknown$") as first:
        launch_pod(
            decision=_approved_decision(),
            state_root=state_root,
            adapter=adapter,
            now=lambda: datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        )
    assert "must-not-escape" not in str(first.value)

    with pytest.raises(LifecycleError, match="^launch_reconciliation_required$"):
        launch_pod(
            decision=_approved_decision(),
            state_root=state_root,
            adapter=adapter,
            now=lambda: datetime(2026, 7, 21, 12, 1, tzinfo=UTC),
        )

    assert adapter.create_calls == 1
    attempts = list((state_root / "attempts").glob("*.json"))
    assert len(attempts) == 1
    assert "must-not-escape" not in attempts[0].read_text()


def test_partial_create_reconciles_from_inventory_without_retry(tmp_path: Path) -> None:
    decision = _approved_decision()

    class PartialCreateAdapter:
        def __init__(self) -> None:
            self.create_calls = 0
            self.created = False

        def inventory(self) -> list[dict[str, object]]:
            if not self.created:
                return []
            return [
                {
                    "id": "pod-partial",
                    "name": decision["run_name"],
                    "desiredStatus": "RUNNING",
                    "imageName": decision["image_ref"],
                    "templateId": decision["template_id"],
                    "networkVolumeId": decision["volume_id"],
                    "costPerHr": 0.44,
                    "vcpuCount": 8,
                    "memoryInGb": 32,
                }
            ]

        def create(self, submitted: dict[str, object], deadline: datetime) -> dict[str, object]:
            self.create_calls += 1
            self.created = True
            raise TimeoutError("unknown after provider accepted create")

    adapter = PartialCreateAdapter()
    state_root = tmp_path / "state"
    with pytest.raises(LifecycleError, match="^provider_create_outcome_unknown$"):
        launch_pod(
            decision=decision,
            state_root=state_root,
            adapter=adapter,
            now=lambda: datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        )

    first = reconcile_launch(
        decision=decision,
        state_root=state_root,
        adapter=adapter,
        now=lambda: datetime(2026, 7, 21, 12, 1, tzinfo=UTC),
    )
    second = reconcile_launch(
        decision=decision,
        state_root=state_root,
        adapter=adapter,
        now=lambda: datetime(2026, 7, 21, 12, 2, tzinfo=UTC),
    )

    assert adapter.create_calls == 1
    assert first == second
    assert first["pod_id"] == "pod-partial"
    assert first["decision_digest"] == decision["decision_digest"]


@pytest.mark.parametrize(
    ("pod", "ownership"),
    [
        (
            {
                "id": "pod-managed",
                "name": "mantisv2-cuda-qualification-seed42",
                "imageName": "registry.example/mantis@sha256:" + "a" * 64,
            },
            "managed",
        ),
        (
            {
                "id": "pod-manual",
                "name": "manual-debug-host",
                "imageName": "registry.example/mantis@sha256:" + "a" * 64,
            },
            "unmanaged",
        ),
    ],
)
def test_fresh_inventory_reports_mantis_pod_without_adoption_or_create(
    tmp_path: Path, pod: dict[str, object], ownership: str
) -> None:
    class InventoryAdapter:
        def __init__(self) -> None:
            self.create_calls = 0

        def inventory(self) -> list[dict[str, object]]:
            return [pod]

        def create(self, decision: dict[str, object], deadline: datetime) -> dict[str, object]:
            self.create_calls += 1
            raise AssertionError("create must not be called")

    adapter = InventoryAdapter()
    with pytest.raises(LifecycleError, match="^live_pod_conflict$") as conflict:
        launch_pod(
            decision=_approved_decision(),
            state_root=tmp_path / "state",
            adapter=adapter,
            now=lambda: datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        )

    assert conflict.value.details == {"pods": [{"pod_id": pod["id"], "ownership": ownership}]}
    assert adapter.create_calls == 0
    assert not (tmp_path / "state" / "receipts" / "pods").exists()


def test_status_requires_exact_identities_and_redacts_provider_secrets(tmp_path: Path) -> None:
    decision = _approved_decision()

    class StatusAdapter:
        def __init__(self) -> None:
            self.status_calls: list[str] = []

        def inventory(self) -> list[dict[str, object]]:
            return []

        def create(self, submitted: dict[str, object], deadline: datetime) -> dict[str, object]:
            return {
                "id": "pod-status",
                "name": decision["run_name"],
                "desiredStatus": "RUNNING",
                "imageName": decision["image_ref"],
                "templateId": decision["template_id"],
                "networkVolumeId": decision["volume_id"],
                "costPerHr": 0.44,
                "vcpuCount": 8,
                "memoryInGb": 32,
            }

        def status(self, pod_id: str) -> dict[str, object]:
            self.status_calls.append(pod_id)
            return {
                "id": pod_id,
                "name": decision["run_name"],
                "desiredStatus": "RUNNING",
                "costPerHr": 0.44,
                "uptimeSeconds": 60,
                "env": {"RUNPOD_API_KEY": "must-not-escape"},
                "ssh": {"privateKey": "must-not-escape"},
            }

    adapter = StatusAdapter()
    state_root = tmp_path / "state"
    launch_pod(
        decision=decision,
        state_root=state_root,
        adapter=adapter,
        now=lambda: datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
    )

    status = pod_status(
        pod_id="pod-status",
        run_name=str(decision["run_name"]),
        state_root=state_root,
        adapter=adapter,
        now=lambda: datetime(2026, 7, 21, 12, 1, tzinfo=UTC),
    )
    encoded = json.dumps(status)

    assert adapter.status_calls == ["pod-status"]
    assert status == {
        "schema_version": 1,
        "pod_id": "pod-status",
        "run_name": decision["run_name"],
        "decision_digest": decision["decision_digest"],
        "desired_status": "RUNNING",
        "cost_per_hour_usd": "0.44",
        "uptime_seconds": 60,
        "observed_at": "2026-07-21T12:01:00Z",
    }
    assert "RUNPOD_API_KEY" not in encoded
    assert "must-not-escape" not in encoded

    with pytest.raises(LifecycleError, match="^run_identity_mismatch$"):
        pod_status(
            pod_id="pod-status",
            run_name="different-run",
            state_root=state_root,
            adapter=adapter,
            now=lambda: datetime(2026, 7, 21, 12, 2, tzinfo=UTC),
        )
    assert adapter.status_calls == ["pod-status"]


def test_manual_terminate_requires_exact_run_and_is_idempotent(tmp_path: Path) -> None:
    decision = _approved_decision()

    class TerminateAdapter:
        def __init__(self) -> None:
            self.terminate_calls: list[str] = []

        def inventory(self) -> list[dict[str, object]]:
            return []

        def create(self, submitted: dict[str, object], deadline: datetime) -> dict[str, object]:
            return {
                "id": "pod-manual-terminate",
                "name": decision["run_name"],
                "desiredStatus": "RUNNING",
                "imageName": decision["image_ref"],
                "templateId": decision["template_id"],
                "networkVolumeId": decision["volume_id"],
                "costPerHr": 0.44,
                "vcpuCount": 8,
                "memoryInGb": 32,
            }

        def terminate(self, pod_id: str) -> dict[str, object]:
            self.terminate_calls.append(pod_id)
            return {"deleted": True, "id": pod_id}

    adapter = TerminateAdapter()
    state_root = tmp_path / "state"
    launch_pod(
        decision=decision,
        state_root=state_root,
        adapter=adapter,
        now=lambda: datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
    )

    with pytest.raises(LifecycleError, match="^run_identity_mismatch$"):
        terminate_pod(
            pod_id="pod-manual-terminate",
            run_name="different-run",
            state_root=state_root,
            adapter=adapter,
            now=lambda: datetime(2026, 7, 21, 12, 5, tzinfo=UTC),
        )
    first = terminate_pod(
        pod_id="pod-manual-terminate",
        run_name=str(decision["run_name"]),
        state_root=state_root,
        adapter=adapter,
        now=lambda: datetime(2026, 7, 21, 12, 5, tzinfo=UTC),
    )
    second = terminate_pod(
        pod_id="pod-manual-terminate",
        run_name=str(decision["run_name"]),
        state_root=state_root,
        adapter=adapter,
        now=lambda: datetime(2026, 7, 21, 12, 6, tzinfo=UTC),
    )

    assert adapter.terminate_calls == ["pod-manual-terminate"]
    assert first == second
    assert first["reason"] == "operator_requested"


def test_unknown_terminate_reconciles_absence_without_retry(tmp_path: Path) -> None:
    decision = _approved_decision()

    class UnknownTerminateAdapter:
        def __init__(self) -> None:
            self.live = False
            self.terminate_calls = 0

        def inventory(self) -> list[dict[str, object]]:
            if not self.live:
                return []
            return [{"id": "pod-unknown-delete", "name": decision["run_name"]}]

        def create(self, submitted: dict[str, object], deadline: datetime) -> dict[str, object]:
            self.live = True
            return {
                "id": "pod-unknown-delete",
                "name": decision["run_name"],
                "desiredStatus": "RUNNING",
                "imageName": decision["image_ref"],
                "templateId": decision["template_id"],
                "networkVolumeId": decision["volume_id"],
                "costPerHr": 0.44,
                "vcpuCount": 8,
                "memoryInGb": 32,
            }

        def terminate(self, pod_id: str) -> dict[str, object]:
            self.terminate_calls += 1
            self.live = False
            raise TimeoutError("delete response unknown")

    adapter = UnknownTerminateAdapter()
    state_root = tmp_path / "state"
    launch_pod(
        decision=decision,
        state_root=state_root,
        adapter=adapter,
        now=lambda: datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
    )
    with pytest.raises(LifecycleError, match="^provider_terminate_outcome_unknown$"):
        terminate_pod(
            pod_id="pod-unknown-delete",
            run_name=str(decision["run_name"]),
            state_root=state_root,
            adapter=adapter,
            now=lambda: datetime(2026, 7, 21, 12, 5, tzinfo=UTC),
        )
    with pytest.raises(LifecycleError, match="^termination_reconciliation_required$"):
        terminate_pod(
            pod_id="pod-unknown-delete",
            run_name=str(decision["run_name"]),
            state_root=state_root,
            adapter=adapter,
            now=lambda: datetime(2026, 7, 21, 12, 6, tzinfo=UTC),
        )

    first = reconcile_termination(
        pod_id="pod-unknown-delete",
        run_name=str(decision["run_name"]),
        state_root=state_root,
        adapter=adapter,
        now=lambda: datetime(2026, 7, 21, 12, 7, tzinfo=UTC),
    )
    second = reconcile_termination(
        pod_id="pod-unknown-delete",
        run_name=str(decision["run_name"]),
        state_root=state_root,
        adapter=adapter,
        now=lambda: datetime(2026, 7, 21, 12, 8, tzinfo=UTC),
    )

    assert adapter.terminate_calls == 1
    assert first == second
    assert first["provider_deleted"] is True
    assert first["reconciliation_evidence"] == "fresh_inventory_absence"
