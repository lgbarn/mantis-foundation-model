"""Fail-closed identity manifest for an RL entry dry run."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from mantis_v2.rl_config import RlConfig


class RlProvenanceError(RuntimeError):
    """Raised when an RL dry run cannot prove its immutable inputs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_digest(repository_root: Path) -> str:
    """Hash all MantisV2 Python sources with stable repository-relative names."""
    source_root = repository_root / "mantis-v2" / "src"
    paths = sorted(source_root.rglob("*.py"))
    if not paths:
        raise RlProvenanceError("source identity has no MantisV2 Python files")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repository_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolved(path: Path, repository_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _file_identity(
    label: str,
    configured_path: Path,
    expected_digest: str,
    repository_root: Path,
) -> dict[str, Any]:
    path = _resolved(configured_path, repository_root)
    if not path.is_file():
        raise RlProvenanceError(f"{label} identity file does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected_digest:
        raise RlProvenanceError(f"{label} identity digest mismatch")
    return {"path": str(path), "size": path.stat().st_size, "sha256": actual}


def _source_identity(config: RlConfig, repository_root: Path) -> dict[str, Any]:
    actual = source_digest(repository_root)
    if actual != config.upstream.source_digest:
        raise RlProvenanceError("source identity digest mismatch")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            capture_output=True,
            check=False,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision or "uncommitted", "dirty": dirty, "sha256": actual}


def _nested(raw: dict[str, Any], section: str, key: str) -> Any:
    value = raw.get(section)
    if not isinstance(value, dict) or key not in value:
        raise RlProvenanceError(f"downstream config is missing {section}.{key}")
    return value[key]


def _verify_composed_identities(config: RlConfig, repository_root: Path) -> None:
    downstream_path = _resolved(config.upstream.downstream_config_path, repository_root)
    try:
        downstream = tomllib.loads(downstream_path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RlProvenanceError("downstream config is not valid TOML") from exc
    expected = (
        (
            "data.corpus_manifest_path",
            Path(str(_nested(downstream, "data", "corpus_manifest_path"))).resolve(),
            config.upstream.corpus_manifest_path.resolve(),
        ),
        (
            "data.corpus_manifest_sha256",
            _nested(downstream, "data", "corpus_manifest_sha256"),
            config.upstream.corpus_manifest_sha256,
        ),
        (
            "foundation.manifest_path",
            Path(str(_nested(downstream, "foundation", "manifest_path"))).resolve(),
            config.upstream.foundation_manifest_path.resolve(),
        ),
        (
            "foundation.weights_sha256",
            _nested(downstream, "foundation", "weights_sha256"),
            config.upstream.foundation_weights_sha256,
        ),
        (
            "walk_forward.embed_manifest_path",
            Path(str(_nested(downstream, "walk_forward", "embed_manifest_path"))).resolve(),
            config.upstream.embedding_manifest_path.resolve(),
        ),
        (
            "walk_forward.embed_manifest_sha256",
            _nested(downstream, "walk_forward", "embed_manifest_sha256"),
            config.upstream.embedding_manifest_sha256,
        ),
        (
            "data.holdout_start",
            _nested(downstream, "data", "holdout_start"),
            config.evaluation.sealed_holdout_start.isoformat(),
        ),
        (
            "evaluation.allow_holdout",
            _nested(downstream, "evaluation", "allow_holdout"),
            False,
        ),
    )
    for field, actual, configured in expected:
        if actual != configured:
            raise RlProvenanceError(f"downstream config identity mismatch: {field}")


def _verify_manifest_binding(label: str, path: Path, weights_sha256: str) -> None:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RlProvenanceError(f"{label} manifest is not valid JSON") from exc
    if manifest.get("foundation_weights_sha256", manifest.get("weights_sha256")) != weights_sha256:
        raise RlProvenanceError(f"{label} manifest foundation identity mismatch")


def _build_manifest(config: RlConfig, repository_root: Path, output: Path) -> dict[str, Any]:
    lock = _file_identity("lock", Path("uv.lock"), config.upstream.lock_digest, repository_root)
    rule_contract = _file_identity(
        "rule_contract",
        config.upstream.rule_contract_path,
        config.upstream.rule_contract_sha256,
        repository_root,
    )
    _file_identity(
        "downstream",
        config.upstream.downstream_config_path,
        config.upstream.downstream_config_sha256,
        repository_root,
    )
    corpus = _file_identity(
        "corpus",
        config.upstream.corpus_manifest_path,
        config.upstream.corpus_manifest_sha256,
        repository_root,
    )
    embedding = _file_identity(
        "embedding",
        config.upstream.embedding_manifest_path,
        config.upstream.embedding_manifest_sha256,
        repository_root,
    )
    foundation_manifest = _file_identity(
        "foundation",
        config.upstream.foundation_manifest_path,
        config.upstream.foundation_manifest_sha256,
        repository_root,
    )
    foundation_weights = _file_identity(
        "weights",
        config.upstream.foundation_weights_path,
        config.upstream.foundation_weights_sha256,
        repository_root,
    )
    _verify_composed_identities(config, repository_root)
    _verify_manifest_binding(
        "embedding",
        _resolved(config.upstream.embedding_manifest_path, repository_root),
        config.upstream.foundation_weights_sha256,
    )
    _verify_manifest_binding(
        "foundation",
        _resolved(config.upstream.foundation_manifest_path, repository_root),
        config.upstream.foundation_weights_sha256,
    )
    return {
        "schema_version": 1,
        "stage": "rl-entry-dry-run",
        "identities": {
            "source": _source_identity(config, repository_root),
            "lock": lock,
            "corpus": corpus,
            "embedding": embedding,
            "foundation": {
                "manifest": foundation_manifest,
                "weights": foundation_weights,
            },
            "config": {"sha256": config.digest},
            "rule": {"sha256": config.rule_digest, "contract": rule_contract},
            "fee": {
                "snapshot": config.execution.fee_schedule,
                "sha256": config.fee_digest,
            },
            "output": {
                "run_name": config.run.name,
                "artifact_root": str(config.run.artifact_root.resolve()),
                "manifest_path": str(output.resolve()),
            },
        },
        "sealed_holdout": {
            "start": str(config.evaluation.sealed_holdout_start),
            "accessed": False,
        },
    }


def _atomic_no_overwrite(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reservation = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RlProvenanceError(
            f"dry-run manifest reservation already exists: {reservation}"
        ) from exc
    os.close(descriptor)
    temporary: Path | None = None
    try:
        if path.exists():
            raise RlProvenanceError(f"dry-run manifest already exists: {path}")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RlProvenanceError(f"dry-run manifest already exists: {path}") from exc
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        reservation.unlink(missing_ok=True)


def write_rl_dry_run_manifest(
    config: RlConfig,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Verify immutable inputs and atomically publish a bounded dry-run manifest."""
    root = (
        repository_root.resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    output = config.run.artifact_root / config.run.name / "dry-run-manifest.json"
    manifest = _build_manifest(config, root, output)
    _atomic_no_overwrite(output, manifest)
    return manifest
