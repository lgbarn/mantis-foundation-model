"""Strict machine-local configuration for Transfer Bundle operations."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from mantis_v2.transfer_bundle import RemoteObject, RetentionAuthorization


class TransferConfigError(ValueError):
    """Raised when transfer configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class TransferSourceConfig:
    root: Path
    include: tuple[str, ...]
    manifest: Path


@dataclass(frozen=True)
class TransferMountedConfig:
    incoming_root: Path
    final_parent: Path


@dataclass(frozen=True)
class TransferBackupConfig:
    internal_root: Path
    internal_manifest: Path
    external_root: Path
    external_manifest: Path


@dataclass(frozen=True)
class TransferConfig:
    schema_version: int
    remote_identity: str
    source: TransferSourceConfig
    mounted: TransferMountedConfig
    backups: TransferBackupConfig


def _exact(raw: object, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TransferConfigError(f"{context} must be a table")
    unknown = set(raw) - expected
    if unknown:
        raise TransferConfigError(f"unknown {context} keys: {', '.join(sorted(unknown))}")
    missing = expected - set(raw)
    if missing:
        raise TransferConfigError(f"missing {context} keys: {', '.join(sorted(missing))}")
    return raw


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TransferConfigError(f"{field} must be a non-empty string")
    return value


def _path(value: object, field: str) -> Path:
    return Path(_text(value, field))


def _include(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TransferConfigError("source.include must be a non-empty array")
    paths: list[str] = []
    for item in value:
        path = _text(item, "source.include")
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != path:
            raise TransferConfigError(f"source.include path is not canonical and relative: {path}")
        if path in paths:
            raise TransferConfigError(f"source.include contains duplicate path: {path}")
        paths.append(path)
    return tuple(paths)


def load_transfer_config(path: str | Path) -> TransferConfig:
    """Load a strict v1 transfer TOML without reading credentials or data."""
    source_path = Path(path)
    try:
        raw_value = tomllib.loads(source_path.read_text())
    except FileNotFoundError as exc:
        raise TransferConfigError(f"config not found: {source_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise TransferConfigError(f"invalid TOML in {source_path}: {exc}") from exc
    raw = _exact(
        raw_value,
        {"schema_version", "remote_identity", "source", "mounted", "backups"},
        "transfer config",
    )
    if raw["schema_version"] != 1:
        raise TransferConfigError("schema_version must be 1")
    remote_identity = _text(raw["remote_identity"], "remote_identity")
    if PurePosixPath(remote_identity).name != remote_identity:
        raise TransferConfigError("remote_identity must be one path-safe name")
    source = _exact(raw["source"], {"root", "include", "manifest"}, "[source]")
    mounted = _exact(raw["mounted"], {"incoming_root", "final_parent"}, "[mounted]")
    backups = _exact(
        raw["backups"],
        {"internal_root", "internal_manifest", "external_root", "external_manifest"},
        "[backups]",
    )
    return TransferConfig(
        schema_version=1,
        remote_identity=remote_identity,
        source=TransferSourceConfig(
            root=_path(source["root"], "source.root"),
            include=_include(source["include"]),
            manifest=_path(source["manifest"], "source.manifest"),
        ),
        mounted=TransferMountedConfig(
            incoming_root=_path(mounted["incoming_root"], "mounted.incoming_root"),
            final_parent=_path(mounted["final_parent"], "mounted.final_parent"),
        ),
        backups=TransferBackupConfig(
            internal_root=_path(backups["internal_root"], "backups.internal_root"),
            internal_manifest=_path(backups["internal_manifest"], "backups.internal_manifest"),
            external_root=_path(backups["external_root"], "backups.external_root"),
            external_manifest=_path(backups["external_manifest"], "backups.external_manifest"),
        ),
    )


def load_retention_authorization(path: str | Path) -> RetentionAuthorization:
    """Load an exact human-created retention authorization without credentials."""
    source = Path(path)
    try:
        raw_value = json.loads(source.read_text())
    except FileNotFoundError as exc:
        raise TransferConfigError(f"authorization not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise TransferConfigError(f"invalid authorization JSON in {source}: {exc}") from exc
    raw = _exact(
        raw_value,
        {"schema_version", "subject_digest", "approved_by"},
        "retention authorization",
    )
    if raw["schema_version"] != 1:
        raise TransferConfigError("retention authorization schema_version must be 1")
    subject_digest = _text(raw["subject_digest"], "authorization.subject_digest")
    if len(subject_digest) != 64 or any(
        character not in "0123456789abcdef" for character in subject_digest
    ):
        raise TransferConfigError("authorization.subject_digest must be lowercase SHA-256")
    return RetentionAuthorization(
        subject_digest=subject_digest,
        approved_by=_text(raw["approved_by"], "authorization.approved_by"),
    )


def load_remote_inventory(path: str | Path) -> dict[str, RemoteObject]:
    """Load a synthetic HEAD inventory for a zero-I/O S3 staging dry run."""
    source = Path(path)
    try:
        raw_value = json.loads(source.read_text())
    except FileNotFoundError as exc:
        raise TransferConfigError(f"remote inventory not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise TransferConfigError(f"invalid remote inventory JSON in {source}: {exc}") from exc
    raw = _exact(raw_value, {"schema_version", "objects"}, "remote inventory")
    if raw["schema_version"] != 1 or not isinstance(raw["objects"], list):
        raise TransferConfigError("remote inventory schema_version must be 1 with objects array")
    objects: dict[str, RemoteObject] = {}
    for item in raw["objects"]:
        entry = _exact(item, {"key", "size", "etag"}, "remote inventory object")
        key = _text(entry["key"], "remote inventory object.key")
        size = entry["size"]
        etag = entry["etag"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise TransferConfigError("remote inventory object.size must be an integer >= 0")
        if etag is not None and not isinstance(etag, str):
            raise TransferConfigError("remote inventory object.etag must be a string or null")
        if key in objects:
            raise TransferConfigError(f"duplicate remote inventory object key: {key}")
        objects[key] = RemoteObject(size=size, etag=etag)
    return objects
