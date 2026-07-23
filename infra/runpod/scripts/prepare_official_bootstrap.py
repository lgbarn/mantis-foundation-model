#!/usr/bin/env python3
"""Create the immutable source bundle and receipt for the official RunPod template."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

OFFICIAL_TEMPLATE_ID = "runpod-torch-v280"
OFFICIAL_IMAGE_REF = (
    "runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35"
)
UV_VERSION = "0.9.0"


def _git(*args: str) -> str:
    completed = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    if _git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("official bootstrap preparation requires a clean committed worktree")
    revision = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    lock_sha256 = _sha256(Path("uv.lock"))
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mantis-official-bootstrap-") as temporary:
        candidate = Path(temporary) / "source.tar.gz"
        subprocess.run(
            ["git", "archive", "--format=tar.gz", "--output", str(candidate), revision],
            check=True,
        )
        archive_sha256 = _sha256(candidate)
        _replace(args.archive, candidate.read_bytes())

    receipt = {
        "schema_version": 1,
        "passed": True,
        "scope": "official_bootstrap",
        "provider_support": "official_runpod_template",
        "template_id": OFFICIAL_TEMPLATE_ID,
        "image_ref": OFFICIAL_IMAGE_REF,
        "uv_version": UV_VERSION,
        "inventory": {
            "identities": {
                "base_image": OFFICIAL_IMAGE_REF,
                "source_revision": revision,
                "source_tree": source_tree,
                "source_archive_sha256": archive_sha256,
                "lock_sha256": lock_sha256,
            }
        },
    }
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    _replace(args.receipt, encoded)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
