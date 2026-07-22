"""RunPod network-volume staging and off-Pod workload evidence callbacks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from mantis_v2.runpod_s3 import AwsCliS3TransferAdapter, RunpodS3Error
from mantis_v2.runpod_workload import validate_workload_manifest
from mantis_v2.transfer_bundle import load_bundle_manifest


def workspace_key(path: str | Path) -> str:
    absolute = Path(path)
    if not absolute.is_absolute() or not absolute.is_relative_to("/workspace"):
        raise RunpodS3Error("Pod path is outside the mounted /workspace volume")
    relative = absolute.relative_to("/workspace").as_posix()
    if not relative or ".." in PurePosixPath(relative).parts:
        raise RunpodS3Error("Pod path cannot map to a safe object key")
    return relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    path = Path(temporary)
    try:
        shutil.copyfile(source, path)
        if destination.exists():
            if not destination.is_file() or _sha256(destination) != _sha256(path):
                raise RunpodS3Error("immutable backup destination differs")
        else:
            os.link(path, destination)
    finally:
        path.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        candidate = Path(temporary)
        if path.exists():
            if path.read_bytes() != encoded:
                raise RunpodS3Error("immutable replication receipt differs")
        else:
            os.link(candidate, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class RunpodS3WorkloadIO:
    """Concrete pre-create staging and supervisor callbacks over the network volume."""

    def __init__(
        self,
        *,
        adapter: AwsCliS3TransferAdapter,
        manifest_path: str | Path,
        pod_manifest_path: str | Path,
        provider: Any,
        state_root: Path,
    ) -> None:
        self.adapter = adapter
        self.manifest_path = Path(manifest_path)
        self.manifest = validate_workload_manifest(self.manifest_path)
        self.pod_manifest_path = Path(pod_manifest_path)
        self.provider = provider
        self.state_root = state_root

    def stage_control_files(self) -> dict[str, Any]:
        records = [
            self.manifest["dependency_lock"],
            self.manifest["image"]["self_check"],
            self.manifest["experiment_config"],
            self.manifest["matrix_plan"],
            self.manifest["matrix_base_config"],
            self.manifest["input_bundle"]["manifest"],
            self.manifest["dataset_manifest"],
            self.manifest["spend_ledger"],
            self.manifest["authorization"],
            self.manifest["monitor"]["token"],
        ]
        uploaded: list[str] = []
        with tempfile.TemporaryDirectory(prefix="mantis-control-preflight-") as temporary:
            verification_root = Path(temporary)
            for index, record in enumerate(records):
                key = workspace_key(record["pod_path"])
                source = Path(str(record["controller_path"]))
                self.adapter.put_file(key, source)
                downloaded = self.adapter.get_file(key, verification_root / f"control-{index}")
                if (
                    downloaded is None
                    or downloaded.stat().st_size != record["size"]
                    or _sha256(downloaded) != record["sha256"]
                ):
                    raise RunpodS3Error("staged control file failed SHA-256 verification")
                uploaded.append(key)
            manifest_key = workspace_key(self.pod_manifest_path)
            self.adapter.put_file(manifest_key, self.manifest_path)
            staged_manifest = self.adapter.get_file(
                manifest_key, verification_root / "launch-manifest.json"
            )
            if (
                staged_manifest is None
                or staged_manifest.read_bytes() != self.manifest_path.read_bytes()
            ):
                raise RunpodS3Error("staged launch manifest failed byte verification")
            uploaded.append(manifest_key)
        return {
            "schema_version": 1,
            "manifest_digest": self.manifest["manifest_digest"],
            "uploaded": uploaded,
        }

    def verify_input_bundle_staged(self) -> dict[str, Any]:
        bundle_record = self.manifest["input_bundle"]["manifest"]
        bundle = load_bundle_manifest(Path(str(bundle_record["controller_path"])))
        prefix = f"mantis/transfer/incoming/{bundle.bundle_digest}/files"
        invalid: list[str] = []
        with tempfile.TemporaryDirectory(prefix="mantis-runpod-preflight-") as temporary:
            verification_root = Path(temporary)
            for entry in bundle.entries:
                key = f"{prefix}/{entry.path}"
                remote = self.adapter.head_object(key)
                if remote is None or remote.size != entry.size:
                    invalid.append(entry.path)
                    continue
                downloaded = self.adapter.get_file(key, verification_root / entry.path)
                if downloaded is None or _sha256(downloaded) != entry.sha256:
                    invalid.append(entry.path)
            manifest_key = f"mantis/transfer/incoming/{bundle.bundle_digest}/manifest.json"
            staged_manifest = self.adapter.get_file(
                manifest_key, verification_root / "manifest.json"
            )
            if staged_manifest is None or staged_manifest.read_bytes() != bundle.to_bytes():
                invalid.append("manifest.json")
        if invalid:
            raise RunpodS3Error(
                "input bundle is not fully staged and hash-verified: "
                f"{len(invalid)} invalid object(s)"
            )
        return {
            "schema_version": 1,
            "bundle_digest": bundle.bundle_digest,
            "entry_count": len(bundle.entries),
            "total_size": bundle.total_size,
        }

    def heartbeat_source(self, run_id: str, pod_id: str) -> Mapping[str, Any] | None:
        del run_id, pod_id
        payload = self.adapter.get_bytes(workspace_key(self.manifest["monitor"]["heartbeat"]))
        if payload is None:
            return None
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunpodS3Error("heartbeat object is invalid JSON") from exc
        if not isinstance(value, dict):
            raise RunpodS3Error("heartbeat object must be a JSON object")
        return value

    def collect_diagnostics(self, pod_id: str, deadline: datetime) -> Mapping[str, Any]:
        del deadline
        artifact_root = Path(str(self.manifest["artifacts"]["pod"]))
        evidence: dict[str, Any] = {"provider": dict(self.provider.status(pod_id))}
        for name in ("launcher.stdout.log", "runtime-diagnostics.json"):
            key = workspace_key(artifact_root / name)
            remote = self.adapter.head_object(key)
            if remote is None:
                evidence[name] = {"status": "absent"}
            elif remote.size > 2 * 1024 * 1024:
                evidence[name] = {"status": "oversized", "size": remote.size}
            else:
                value = self.adapter.get_bytes(key)
                assert value is not None
                evidence[name] = {
                    "status": "captured",
                    "size": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(),
                    "text": value.decode(errors="replace"),
                }
        return evidence

    def checkpoint(self, pod_id: str, deadline: datetime) -> str | None:
        del pod_id, deadline
        artifact_root = Path(str(self.manifest["artifacts"]["pod"]))
        key = workspace_key(artifact_root / "checkpoints" / "latest.pt")
        destination = (
            self.state_root / "emergency-checkpoints" / str(self.manifest["run_id"]) / "latest.pt"
        )
        downloaded = self.adapter.get_file(key, destination)
        return _sha256(downloaded) if downloaded is not None else None

    def replicate(self, manifest: Mapping[str, Any]) -> None:
        if manifest.get("manifest_digest") != self.manifest["manifest_digest"]:
            raise RunpodS3Error("replication manifest identity mismatch")
        artifact_root = Path(str(self.manifest["artifacts"]["pod"]))
        prefix = workspace_key(artifact_root)
        objects = self.adapter.list_objects(prefix)
        if not objects:
            raise RunpodS3Error("completed workload has no artifact objects")
        controller = Path(str(self.manifest["artifacts"]["controller"]))
        backup = Path(str(self.manifest["artifacts"]["backup"]))
        files: list[dict[str, Any]] = []
        for key, remote in sorted(objects.items()):
            relative = PurePosixPath(key).relative_to(PurePosixPath(prefix)).as_posix()
            if not relative or ".." in PurePosixPath(relative).parts:
                raise RunpodS3Error("artifact object key is unsafe")
            internal_path = controller / relative
            downloaded = self.adapter.get_file(key, internal_path)
            if downloaded is None:
                raise RunpodS3Error("artifact disappeared during replication")
            backup_path = backup / relative
            _atomic_copy(downloaded, backup_path)
            digest = _sha256(downloaded)
            if _sha256(backup_path) != digest or downloaded.stat().st_size != remote.size:
                raise RunpodS3Error("artifact backup verification failed")
            files.append({"path": relative, "size": remote.size, "sha256": digest})
        core = {
            "schema_version": 1,
            "manifest_digest": self.manifest["manifest_digest"],
            "run_id": self.manifest["run_id"],
            "files": files,
        }
        receipt = {
            **core,
            "replication_digest": hashlib.sha256(
                json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        receipt_relative = Path("replication-receipts") / f"{receipt['replication_digest']}.json"
        _atomic_json(controller / receipt_relative, receipt)
        _atomic_json(backup / receipt_relative, receipt)
