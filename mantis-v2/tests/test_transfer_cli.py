from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from mantis_v2 import cli, transfer_bundle
from mantis_v2.transfer_config import TransferConfigError, load_transfer_config

ROOT = Path(__file__).parents[2]


def _write_config(tmp_path: Path, *, extra: str = "") -> tuple[Path, dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "source": tmp_path / "source",
        "manifest": tmp_path / "manifests" / "bundle.json",
        "incoming": tmp_path / "mounted" / "incoming",
        "final": tmp_path / "mounted" / "inputs",
        "internal": tmp_path / "backups" / "internal",
        "internal_manifest": tmp_path / "backups" / "internal-manifest.json",
        "external": tmp_path / "backups" / "external",
        "external_manifest": tmp_path / "backups" / "external-manifest.json",
    }
    config = tmp_path / "transfer.toml"
    config.write_text(
        f'''schema_version = 1
remote_identity = "measured-input-bundle"

[source]
root = "{paths["source"]}"
include = ["alpha.txt"]
manifest = "{paths["manifest"]}"

[mounted]
incoming_root = "{paths["incoming"]}"
final_parent = "{paths["final"]}"

[backups]
internal_root = "{paths["internal"]}"
internal_manifest = "{paths["internal_manifest"]}"
external_root = "{paths["external"]}"
external_manifest = "{paths["external_manifest"]}"
{extra}'''
    )
    return config, paths


def test_transfer_config_is_strict_and_typed(tmp_path: Path) -> None:
    config_path, paths = _write_config(tmp_path)

    config = load_transfer_config(config_path)

    assert config.source.root == paths["source"]
    assert config.source.include == ("alpha.txt",)
    assert config.backups.external_manifest == paths["external_manifest"]
    assert config.remote_identity == "measured-input-bundle"

    invalid_path, _ = _write_config(tmp_path / "invalid", extra="unknown = true\n")
    with pytest.raises(TransferConfigError, match="unknown"):
        load_transfer_config(invalid_path)


def test_transfer_bundle_cli_writes_canonical_manifest_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, paths = _write_config(tmp_path)
    paths["source"].mkdir()
    (paths["source"] / "alpha.txt").write_bytes(b"alpha\n")
    monkeypatch.setattr(sys, "argv", ["mantis-v2", "transfer-bundle", "--config", str(config_path)])

    cli.main()

    result = json.loads(capsys.readouterr().out)
    manifest = transfer_bundle.BundleManifest.from_bytes(paths["manifest"].read_bytes())
    assert result == {
        "bundle_digest": manifest.bundle_digest,
        "manifest": str(paths["manifest"]),
        "total_size": 6,
    }
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    assert "already_exists" in capsys.readouterr().err


def test_transfer_promote_backup_and_retention_cli_are_verification_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, paths = _write_config(tmp_path)
    paths["source"].mkdir()
    (paths["source"] / "alpha.txt").write_bytes(b"alpha\n")
    manifest = transfer_bundle.build_bundle(paths["source"])
    paths["manifest"].parent.mkdir()
    paths["manifest"].write_bytes(manifest.to_bytes())
    paths["incoming"].mkdir(parents=True)
    (paths["incoming"] / "alpha.txt").write_bytes(b"alpha\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["mantis-v2", "transfer-promote", "--config", str(config_path)],
    )

    cli.main()

    promoted = json.loads(capsys.readouterr().out)
    final = paths["final"] / manifest.bundle_digest
    assert promoted["promoted"] is True
    assert Path(promoted["path"]) == final

    for root, local_manifest in (
        (paths["internal"], paths["internal_manifest"]),
        (paths["external"], paths["external_manifest"]),
    ):
        root.mkdir(parents=True)
        (root / "alpha.txt").write_bytes(b"alpha\n")
        local_manifest.parent.mkdir(parents=True, exist_ok=True)
        local_manifest.write_bytes(manifest.to_bytes())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "transfer-backup-verify",
            "--config",
            str(config_path),
            "--completed-artifact-digest",
            "artifact-123",
        ],
    )

    cli.main()

    backups = json.loads(capsys.readouterr().out)
    assert backups["bundle_digest"] == manifest.bundle_digest
    assert backups["completed_artifact_digest"] == "artifact-123"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "transfer-retention-check",
            "--config",
            str(config_path),
            "--completed-artifact-digest",
            "artifact-123",
            "--run-state",
            "inactive",
        ],
    )

    cli.main()

    decision = json.loads(capsys.readouterr().out)
    assert decision["allowed"] is False
    assert decision["reasons"] == ["authorization_required"]


def test_transfer_stage_dry_run_uses_injected_inventory_without_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, paths = _write_config(tmp_path)
    paths["source"].mkdir()
    (paths["source"] / "alpha.txt").write_bytes(b"alpha\n")
    manifest = transfer_bundle.write_bundle_manifest(
        paths["source"], ("alpha.txt",), paths["manifest"]
    )
    prefix = f"mantis/transfer/incoming/{manifest.bundle_digest}"
    inventory = tmp_path / "remote-inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "objects": [
                    {
                        "key": f"{prefix}/files/alpha.txt",
                        "size": 6,
                        "etag": "opaque-multipart-etag",
                    }
                ],
            }
        )
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "transfer-stage-dry-run",
            "--config",
            str(config_path),
            "--remote-inventory",
            str(inventory),
        ],
    )

    cli.main()

    result = json.loads(capsys.readouterr().out)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result["dry_run"] is True
    assert result["skipped"] == [f"{prefix}/files/alpha.txt"]
    assert result["planned_uploads"] == [f"{prefix}/manifest.json"]
    assert before == after


def test_measured_manifest_inspect_is_a_no_upload_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = ROOT / "infra" / "runpod" / "examples" / "measured-input-bundle.fixture.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mantis-v2", "transfer-manifest-inspect", "--manifest", str(fixture)],
    )

    cli.main()

    result = json.loads(capsys.readouterr().out)
    assert result == {
        "bundle_digest": "b8584b5ce25fb96cff66ccb926b0b9b5bed560efc06ea1449f7e56521192e623",
        "entry_count": 3,
        "path_roots": ["cache", "corpus", "dbn"],
        "total_size": 1_395_349_697,
    }


def test_public_just_recipes_expose_transfer_verification_workflow() -> None:
    listed = subprocess.run(
        ["just", "--list"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout

    for recipe in (
        "transfer-bundle config",
        "transfer-stage-dry-run config remote_inventory",
        "transfer-promote config",
        "transfer-backup-verify config completed_artifact_digest",
        "transfer-retention-check config completed_artifact_digest run_state",
    ):
        assert recipe in listed
