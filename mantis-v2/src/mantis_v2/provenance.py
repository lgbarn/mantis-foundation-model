"""Content-addressed provenance for configs, data, source, and artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mantis_v2.config import PipelineConfig
from mantis_v2.data import discover_paths


@dataclass(frozen=True)
class FileIdentity:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Provenance:
    schema_version: int
    config_digest: str
    dataset_digest: str
    dataset_files: tuple[FileIdentity, ...]
    source_revision: str
    source_dirty: bool
    source_digest: str
    lock_digest: str
    upstream_source_revision: str
    upstream_hub_revision: str
    upstream_weights_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_identity(config: PipelineConfig) -> tuple[str, tuple[FileIdentity, ...]]:
    if config.data.root == "synthetic":
        identity = FileIdentity("synthetic:v1:rows=4096", 4096, "deterministic")
        payload = json.dumps(asdict(identity), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest(), (identity,)
    files = tuple(
        FileIdentity(str(path.resolve()), path.stat().st_size, sha256_file(path))
        for path in discover_paths(config.data)
    )
    payload = json.dumps([asdict(item) for item in files], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest(), files


def _git_state(root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
        ).stdout.strip()
    )
    return revision or "uncommitted", dirty


def _source_digest(root: Path) -> str:
    paths = sorted((root / "mantis-v2" / "src").rglob("*.py"))
    if not paths:
        raise RuntimeError("no MantisV2 source files found for provenance")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_provenance(config: PipelineConfig, repository_root: Path) -> Provenance:
    dataset_digest, files = dataset_identity(config)
    source_revision, source_dirty = _git_state(repository_root)
    lock_path = repository_root / "uv.lock"
    if not lock_path.is_file():
        raise RuntimeError("uv.lock is required for reproducible training")
    return Provenance(
        schema_version=1,
        config_digest=config.digest,
        dataset_digest=dataset_digest,
        dataset_files=files,
        source_revision=source_revision,
        source_dirty=source_dirty,
        source_digest=_source_digest(repository_root),
        lock_digest=sha256_file(lock_path),
        upstream_source_revision=config.model.source_revision,
        upstream_hub_revision=config.model.hub_revision,
        upstream_weights_sha256=config.model.weights_sha256,
    )
