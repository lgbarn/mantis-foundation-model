"""Content-addressed transfer bundles for RunPod network volumes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol


class TransferBundleError(ValueError):
    """Raised when transfer content cannot satisfy the bundle contract."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True, order=True)
class BundleEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class BundleManifest:
    entries: tuple[BundleEntry, ...]
    total_size: int
    bundle_digest: str

    def to_bytes(self) -> bytes:
        data = {
            "schema_version": 1,
            "entries": [asdict(entry) for entry in self.entries],
            "total_size": self.total_size,
            "bundle_digest": self.bundle_digest,
        }
        return _canonical_bytes(data) + b"\n"

    @classmethod
    def from_bytes(cls, value: bytes) -> BundleManifest:
        """Load a strict canonical manifest and recompute its content identity."""
        try:
            data = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransferBundleError("manifest.json", "invalid_json") from exc
        if not isinstance(data, dict) or set(data) != {
            "schema_version",
            "entries",
            "total_size",
            "bundle_digest",
        }:
            raise TransferBundleError("manifest.json", "invalid_schema")
        if data["schema_version"] != 1 or not isinstance(data["entries"], list):
            raise TransferBundleError("manifest.json", "invalid_schema")
        entries: list[BundleEntry] = []
        for raw_entry in data["entries"]:
            if not isinstance(raw_entry, dict) or set(raw_entry) != {"path", "size", "sha256"}:
                raise TransferBundleError("manifest.json", "invalid_schema")
            path = raw_entry["path"]
            size = raw_entry["size"]
            sha256 = raw_entry["sha256"]
            if not isinstance(path, str) or _safe_relative_path(path) != path:
                raise TransferBundleError("manifest.json", "invalid_schema")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise TransferBundleError(path, "invalid_size")
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise TransferBundleError(path, "invalid_sha256")
            entries.append(BundleEntry(path=path, size=size, sha256=sha256))
        ordered = tuple(sorted(entries))
        if tuple(entries) != ordered or len({entry.path for entry in entries}) != len(entries):
            raise TransferBundleError("manifest.json", "entries_not_canonical")
        total_size = data["total_size"]
        if (
            not isinstance(total_size, int)
            or isinstance(total_size, bool)
            or total_size != sum(entry.size for entry in entries)
        ):
            raise TransferBundleError("manifest.json", "total_size_mismatch")
        identity = {"schema_version": 1, "entries": [asdict(entry) for entry in entries]}
        expected_digest = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        if data["bundle_digest"] != expected_digest:
            raise TransferBundleError("manifest.json", "bundle_digest_mismatch")
        manifest = cls(
            entries=ordered,
            total_size=total_size,
            bundle_digest=expected_digest,
        )
        if manifest.to_bytes() != value:
            raise TransferBundleError("manifest.json", "noncanonical_serialization")
        return manifest


@dataclass(frozen=True)
class RemoteObject:
    size: int
    etag: str | None = None


class S3TransferAdapter(Protocol):
    """Small S3-compatible seam; credentials remain inside the concrete adapter."""

    def head_object(self, key: str) -> RemoteObject | None: ...

    def put_file(self, key: str, source: Path) -> None: ...

    def put_bytes(self, key: str, value: bytes) -> None: ...


@dataclass(frozen=True)
class StageReceipt:
    bundle_digest: str
    incoming_prefix: str
    uploaded: tuple[str, ...]
    skipped: tuple[str, ...]


@dataclass(frozen=True)
class VerificationReceipt:
    bundle_digest: str
    path: Path


@dataclass(frozen=True)
class PromotionReceipt:
    bundle_digest: str
    path: Path
    promoted: bool


@dataclass(frozen=True)
class VerifiedCopy:
    role: str
    path: Path
    bundle_digest: str
    completed_artifact_digest: str


@dataclass(frozen=True)
class BackupPair:
    internal: VerifiedCopy
    external: VerifiedCopy
    bundle_digest: str
    completed_artifact_digest: str


@dataclass(frozen=True)
class RetentionAuthorization:
    subject_digest: str
    approved_by: str


@dataclass(frozen=True)
class RetentionDecision:
    allowed: bool
    reasons: tuple[str, ...]
    subject_digest: str
    remote_identity: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute():
        raise TransferBundleError(value, "absolute_path")
    if ".." in path.parts:
        raise TransferBundleError(value, "path_traversal")
    return path.as_posix()


def _source_identity(file_stat: object) -> tuple[int, ...]:
    return tuple(
        int(getattr(file_stat, field))
        for field in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    )


def _discover_paths(source_root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(source_root).as_posix()
            for path in source_root.rglob("*")
            if not stat.S_ISDIR(path.lstat().st_mode)
        )
    )


def build_bundle(
    source_root: Path,
    relative_paths: Iterable[str] | None = None,
    *,
    progress: Callable[[str, int], None] | None = None,
) -> BundleManifest:
    """Hash explicitly selected source files into one ordered bundle identity."""
    discovered = relative_paths is None
    if relative_paths is None:
        selected_paths = _discover_paths(source_root)
    else:
        selected_paths = tuple(relative_paths)
    entries: list[BundleEntry] = []
    seen: set[str] = set()
    for provided_path in selected_paths:
        relative_path = _safe_relative_path(provided_path)
        if relative_path in seen:
            raise TransferBundleError(provided_path, "duplicate_normalized_path")
        seen.add(relative_path)
        path = source_root / relative_path
        file_stat = path.lstat()
        if stat.S_ISLNK(file_stat.st_mode):
            raise TransferBundleError(relative_path, "symlink")
        if not stat.S_ISREG(file_stat.st_mode):
            raise TransferBundleError(relative_path, "special_file")
        digest = hashlib.sha256()
        bytes_hashed = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                bytes_hashed += len(chunk)
                if progress is not None:
                    progress(relative_path, bytes_hashed)
        try:
            final_stat = path.lstat()
        except FileNotFoundError as exc:
            raise TransferBundleError(relative_path, "source_changed_during_hash") from exc
        if _source_identity(file_stat) != _source_identity(final_stat):
            raise TransferBundleError(relative_path, "source_changed_during_hash")
        entries.append(
            BundleEntry(path=relative_path, size=file_stat.st_size, sha256=digest.hexdigest())
        )
    if discovered:
        final_paths = _discover_paths(source_root)
        if final_paths != selected_paths:
            changed_path = min(set(final_paths).symmetric_difference(selected_paths))
            raise TransferBundleError(changed_path, "source_tree_changed_during_hash")
    ordered = tuple(sorted(entries))
    identity = {"schema_version": 1, "entries": [asdict(entry) for entry in ordered]}
    return BundleManifest(
        entries=ordered,
        total_size=sum(entry.size for entry in ordered),
        bundle_digest=hashlib.sha256(_canonical_bytes(identity)).hexdigest(),
    )


def _verify_source_entry(source_root: Path, entry: BundleEntry) -> None:
    path = source_root / entry.path
    try:
        initial_stat = path.lstat()
    except FileNotFoundError as exc:
        raise TransferBundleError(entry.path, "source_changed_since_manifest") from exc
    if not stat.S_ISREG(initial_stat.st_mode):
        reason = "symlink" if stat.S_ISLNK(initial_stat.st_mode) else "special_file"
        raise TransferBundleError(entry.path, reason)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    final_stat = path.lstat()
    if (
        _source_identity(initial_stat) != _source_identity(final_stat)
        or final_stat.st_size != entry.size
        or digest.hexdigest() != entry.sha256
    ):
        raise TransferBundleError(entry.path, "source_changed_since_manifest")


def stage_bundle(
    source_root: Path,
    manifest: BundleManifest,
    adapter: S3TransferAdapter,
) -> StageReceipt:
    """Resume a safe upload into the bundle's content-addressed incoming prefix."""
    prefix = f"transfer/incoming/{manifest.bundle_digest}"
    uploaded: list[str] = []
    skipped: list[str] = []
    for entry in manifest.entries:
        _verify_source_entry(source_root, entry)
    for entry in manifest.entries:
        key = f"{prefix}/files/{entry.path}"
        remote = adapter.head_object(key)
        if remote is None or remote.size != entry.size:
            adapter.put_file(key, source_root / entry.path)
            uploaded.append(key)
        else:
            skipped.append(key)
    manifest_key = f"{prefix}/manifest.json"
    manifest_bytes = manifest.to_bytes()
    remote_manifest = adapter.head_object(manifest_key)
    if remote_manifest is None or remote_manifest.size != len(manifest_bytes):
        adapter.put_bytes(manifest_key, manifest_bytes)
        uploaded.append(manifest_key)
    else:
        skipped.append(manifest_key)
    return StageReceipt(
        bundle_digest=manifest.bundle_digest,
        incoming_prefix=prefix,
        uploaded=tuple(uploaded),
        skipped=tuple(skipped),
    )


def verify_bundle(root: Path, manifest: BundleManifest) -> VerificationReceipt:
    """Verify that a mounted directory exactly matches a bundle manifest."""
    expected_files = {entry.path for entry in manifest.entries}
    expected_directories: set[str] = set()
    for entry in manifest.entries:
        parent = PurePosixPath(entry.path).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    for path in sorted(root.rglob("*")):
        relative_path = path.relative_to(root).as_posix()
        file_stat = path.lstat()
        if stat.S_ISLNK(file_stat.st_mode):
            if relative_path in expected_files:
                raise TransferBundleError(relative_path, "symlink")
            raise TransferBundleError(relative_path, "unexpected_path")
        if stat.S_ISDIR(file_stat.st_mode):
            if relative_path not in expected_directories:
                raise TransferBundleError(relative_path, "unexpected_path")
        elif relative_path not in expected_files:
            raise TransferBundleError(relative_path, "unexpected_path")
    for entry in manifest.entries:
        path = root / entry.path
        try:
            initial_stat = path.lstat()
        except FileNotFoundError as exc:
            raise TransferBundleError(entry.path, "missing_file") from exc
        if stat.S_ISLNK(initial_stat.st_mode):
            raise TransferBundleError(entry.path, "symlink")
        if not stat.S_ISREG(initial_stat.st_mode):
            raise TransferBundleError(entry.path, "special_file")
        if initial_stat.st_size != entry.size:
            raise TransferBundleError(entry.path, "size_mismatch")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        final_stat = path.lstat()
        if _source_identity(initial_stat) != _source_identity(final_stat):
            raise TransferBundleError(entry.path, "changed_during_verification")
        if digest.hexdigest() != entry.sha256:
            raise TransferBundleError(entry.path, "sha256_mismatch")
    return VerificationReceipt(bundle_digest=manifest.bundle_digest, path=root)


def verify_and_promote(
    incoming_root: Path,
    final_parent: Path,
    manifest: BundleManifest,
) -> PromotionReceipt:
    """Verify an incoming mounted bundle before one atomic directory rename."""
    verify_bundle(incoming_root, manifest)
    final_path = final_parent / manifest.bundle_digest
    if final_path.exists():
        verify_bundle(final_path, manifest)
        return PromotionReceipt(
            bundle_digest=manifest.bundle_digest,
            path=final_path,
            promoted=False,
        )
    final_parent.mkdir(parents=True, exist_ok=True)
    os.rename(incoming_root, final_path)
    return PromotionReceipt(
        bundle_digest=manifest.bundle_digest,
        path=final_path,
        promoted=True,
    )


def verify_download(
    root: Path,
    *,
    expected_manifest: BundleManifest,
    local_manifest: BundleManifest,
    completed_artifact_digest: str,
    role: str,
) -> VerifiedCopy:
    """Bind one verified downloaded tree to its artifact and storage role."""
    if local_manifest.to_bytes() != expected_manifest.to_bytes():
        raise TransferBundleError("manifest.json", "manifest_mismatch")
    if role not in {"internal_ssd", "external_drive"}:
        raise TransferBundleError(role, "invalid_backup_role")
    verify_bundle(root, expected_manifest)
    return VerifiedCopy(
        role=role,
        path=root.resolve(),
        bundle_digest=expected_manifest.bundle_digest,
        completed_artifact_digest=completed_artifact_digest,
    )


def verify_backup_pair(internal: VerifiedCopy, external: VerifiedCopy) -> BackupPair:
    """Require independent named copies of exactly one completed artifact."""
    if internal.role != "internal_ssd" or external.role != "external_drive":
        raise TransferBundleError("backups", "backup_roles_mismatch")
    paths_overlap = (
        internal.path == external.path
        or internal.path in external.path.parents
        or external.path in internal.path.parents
    )
    if paths_overlap:
        raise TransferBundleError("backups", "backup_paths_not_distinct")
    if internal.bundle_digest != external.bundle_digest:
        raise TransferBundleError("backups", "bundle_digest_mismatch")
    if internal.completed_artifact_digest != external.completed_artifact_digest:
        raise TransferBundleError("backups", "completed_artifact_mismatch")
    return BackupPair(
        internal=internal,
        external=external,
        bundle_digest=internal.bundle_digest,
        completed_artifact_digest=internal.completed_artifact_digest,
    )


def retention_subject_digest(
    *,
    remote_identity: str,
    bundle_digest: str,
    completed_artifact_digest: str,
) -> str:
    """Create the exact content identity a human must authorize for deletion."""
    subject = {
        "schema_version": 1,
        "remote_identity": remote_identity,
        "bundle_digest": bundle_digest,
        "completed_artifact_digest": completed_artifact_digest,
    }
    return hashlib.sha256(_canonical_bytes(subject)).hexdigest()


def decide_retention(
    *,
    remote_identity: str,
    bundle_digest: str,
    completed_artifact_digest: str,
    backups: BackupPair | None,
    authorization: RetentionAuthorization | None,
    run_active: bool,
) -> RetentionDecision:
    """Return a deterministic, fail-closed remote retention decision."""
    subject_digest = retention_subject_digest(
        remote_identity=remote_identity,
        bundle_digest=bundle_digest,
        completed_artifact_digest=completed_artifact_digest,
    )
    reasons: list[str] = []
    if run_active:
        reasons.append("active_run")
    if backups is None:
        reasons.append("verified_backup_pair_required")
    else:
        if backups.bundle_digest != bundle_digest:
            reasons.append("bundle_digest_mismatch")
        if backups.completed_artifact_digest != completed_artifact_digest:
            reasons.append("completed_artifact_mismatch")
    if authorization is None:
        reasons.append("authorization_required")
    elif authorization.subject_digest != subject_digest or not authorization.approved_by:
        reasons.append("authorization_mismatch")
    return RetentionDecision(
        allowed=not reasons,
        reasons=tuple(reasons),
        subject_digest=subject_digest,
        remote_identity=remote_identity,
    )


def execute_retention(decision: RetentionDecision, isolated_root: Path) -> None:
    """Delete only an authorized immediate child of an explicit isolated root."""
    if not decision.allowed:
        raise TransferBundleError(decision.remote_identity, "retention_refused")
    relative_identity = _safe_relative_path(decision.remote_identity)
    if PurePosixPath(relative_identity).parent != PurePosixPath("."):
        raise TransferBundleError(decision.remote_identity, "retention_target_not_isolated")
    root = isolated_root.resolve(strict=True)
    target = root / relative_identity
    file_stat = target.lstat()
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISDIR(file_stat.st_mode):
        raise TransferBundleError(decision.remote_identity, "retention_target_not_directory")
    shutil.rmtree(target)
