from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "infra"
    / "runpod"
    / "scripts"
    / "prepare_official_bootstrap.py"
)


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_official_bootstrap_archives_exact_clean_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "uv.lock").write_text("lock\n")
    (repo / "source.py").write_text("VALUE = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    archive = tmp_path / "source.tar.gz"
    receipt = tmp_path / "receipt.json"

    completed = _run(
        repo,
        "--archive",
        str(archive),
        "--receipt",
        str(receipt),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(receipt.read_text())
    identities = payload["inventory"]["identities"]
    assert payload["scope"] == "official_bootstrap"
    assert payload["provider_support"] == "official_runpod_template"
    assert payload["template_id"] == "runpod-torch-v280"
    assert payload["image_ref"].startswith("runpod/pytorch@sha256:")
    assert identities["source_revision"] == _git(repo, "rev-parse", "HEAD")
    assert identities["source_tree"] == _git(repo, "rev-parse", "HEAD^{tree}")
    assert identities["lock_sha256"] == hashlib.sha256(b"lock\n").hexdigest()
    assert identities["source_archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_official_bootstrap_rejects_dirty_source(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "uv.lock").write_text("lock\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    (repo / "dirty.txt").write_text("dirty\n")

    completed = _run(
        repo,
        "--archive",
        str(tmp_path / "source.tar.gz"),
        "--receipt",
        str(tmp_path / "receipt.json"),
    )

    assert completed.returncode == 1
    assert "clean committed worktree" in completed.stderr
