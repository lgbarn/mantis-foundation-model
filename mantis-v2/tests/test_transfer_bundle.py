from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
from mantis_v2 import transfer_bundle


class FakeS3Adapter:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.head_calls: list[str] = []
        self.put_file_calls: list[tuple[str, Path]] = []
        self.put_bytes_calls: list[tuple[str, bytes]] = []

    def head_object(self, key: str) -> transfer_bundle.RemoteObject | None:
        self.head_calls.append(key)
        value = self.objects.get(key)
        if value is None:
            return None
        return transfer_bundle.RemoteObject(size=len(value), etag="opaque-not-content-identity")

    def put_file(self, key: str, source: Path) -> None:
        self.put_file_calls.append((key, source))
        self.objects[key] = source.read_bytes()

    def put_bytes(self, key: str, value: bytes) -> None:
        self.put_bytes_calls.append((key, value))
        self.objects[key] = value


class FailOnceS3Adapter(FakeS3Adapter):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def put_file(self, key: str, source: Path) -> None:
        super().put_file(key, source)
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected upload interruption")


def test_build_bundle_orders_entries_and_uses_literal_sha256_oracle(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "alpha.txt").write_bytes(b"alpha\n")
    (source / "nested" / "beta.bin").write_bytes(b"\x00\xff")

    manifest = transfer_bundle.build_bundle(
        source,
        ("nested/beta.bin", "alpha.txt"),
    )

    assert manifest.entries == (
        transfer_bundle.BundleEntry(
            path="alpha.txt",
            size=6,
            sha256="b6a98d9ce9a2d9149288fa3df42d377c3e42737afdcdaf714e33c0a100b51060",
        ),
        transfer_bundle.BundleEntry(
            path="nested/beta.bin",
            size=2,
            sha256="06eb7d6a69ee19e5fbdf749018d3d2abfa04bcbd1365db312eb86dc7169389b8",
        ),
    )
    assert manifest.total_size == 8
    assert manifest.bundle_digest == (
        "01e1041364cc4eb82c1ae0d46cc0b66d241ebdf2cfe6acb3a8f5b780f4e1d9fb"
    )


def test_build_bundle_rejects_absolute_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    selected = source / "alpha.txt"
    selected.write_bytes(b"alpha\n")

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        transfer_bundle.build_bundle(source, (str(selected),))

    assert raised.value.path == str(selected)
    assert raised.value.reason == "absolute_path"


def test_build_bundle_rejects_path_traversal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "escape.txt").write_bytes(b"outside")

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        transfer_bundle.build_bundle(source, ("../escape.txt",))

    assert raised.value.path == "../escape.txt"
    assert raised.value.reason == "path_traversal"


def test_build_bundle_rejects_duplicate_normalized_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "beta.bin").write_bytes(b"beta")

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        transfer_bundle.build_bundle(
            source,
            ("nested/beta.bin", "nested//beta.bin"),
        )

    assert raised.value.path == "nested//beta.bin"
    assert raised.value.reason == "duplicate_normalized_path"


def test_build_bundle_rejects_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "target.bin").write_bytes(b"target")
    (source / "linked.bin").symlink_to(source / "target.bin")

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        transfer_bundle.build_bundle(source, ("linked.bin",))

    assert raised.value.path == "linked.bin"
    assert raised.value.reason == "symlink"


def test_build_bundle_rejects_special_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    os.mkfifo(source / "stream")

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        transfer_bundle.build_bundle(source, ("stream",))

    assert raised.value.path == "stream"
    assert raised.value.reason == "special_file"


def test_build_bundle_rejects_selected_file_changed_during_hash(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    selected = source / "alpha.txt"
    selected.write_bytes(b"alpha\n")

    def mutate_after_read(relative_path: str, bytes_hashed: int) -> None:
        assert relative_path == "alpha.txt"
        assert bytes_hashed == 6
        selected.write_bytes(b"omega\n")

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        transfer_bundle.build_bundle(
            source,
            ("alpha.txt",),
            progress=mutate_after_read,
        )

    assert raised.value.path == "alpha.txt"
    assert raised.value.reason == "source_changed_during_hash"


def test_build_bundle_rejects_discovered_tree_changed_during_hash(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "alpha.txt").write_bytes(b"alpha\n")

    def add_file_after_read(_relative_path: str, _bytes_hashed: int) -> None:
        (source / "late.txt").write_bytes(b"late")

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        transfer_bundle.build_bundle(source, progress=add_file_after_read)

    assert raised.value.path == "late.txt"
    assert raised.value.reason == "source_tree_changed_during_hash"


def test_manifest_serialization_is_canonical_and_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "alpha.txt").write_bytes(b"alpha\n")
    (source / "nested" / "beta.bin").write_bytes(b"\x00\xff")

    first = transfer_bundle.build_bundle(source)
    second = transfer_bundle.build_bundle(source)

    expected = (
        b'{"bundle_digest":"01e1041364cc4eb82c1ae0d46cc0b66d241ebdf2cfe6acb3a8f5b780f4e1d9fb",'
        b'"entries":[{"path":"alpha.txt","sha256":"b6a98d9ce9a2d9149288fa3df42d377c3e42737'
        b'afdcdaf714e33c0a100b51060","size":6},{"path":"nested/beta.bin","sha256":"06eb7d6a'
        b'69ee19e5fbdf749018d3d2abfa04bcbd1365db312eb86dc7169389b8","size":2}],'
        b'"schema_version":1,"total_size":8}\n'
    )
    assert first.to_bytes() == expected
    assert second.to_bytes() == expected


def test_stage_bundle_uploads_only_absent_or_size_mismatched_objects(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "alpha.txt").write_bytes(b"alpha\n")
    (source / "nested" / "beta.bin").write_bytes(b"\x00\xff")
    manifest = transfer_bundle.build_bundle(source)
    prefix = f"transfer/incoming/{manifest.bundle_digest}"
    alpha_key = f"{prefix}/files/alpha.txt"
    beta_key = f"{prefix}/files/nested/beta.bin"
    manifest_key = f"{prefix}/manifest.json"
    adapter = FakeS3Adapter({alpha_key: b"wrong\n", beta_key: b"x"})

    receipt = transfer_bundle.stage_bundle(source, manifest, adapter)

    assert receipt.uploaded == (beta_key, manifest_key)
    assert receipt.skipped == (alpha_key,)
    assert adapter.head_calls == [alpha_key, beta_key, manifest_key]
    assert [call[0] for call in adapter.put_file_calls] == [beta_key]
    assert [call[0] for call in adapter.put_bytes_calls] == [manifest_key]
    assert adapter.objects[alpha_key] == b"wrong\n"
    assert adapter.objects[beta_key] == b"\x00\xff"
    assert adapter.objects[manifest_key] == manifest.to_bytes()


def test_stage_bundle_resumes_after_interrupted_upload(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "alpha.txt").write_bytes(b"alpha\n")
    (source / "beta.txt").write_bytes(b"beta\n")
    manifest = transfer_bundle.build_bundle(source)
    prefix = f"transfer/incoming/{manifest.bundle_digest}"
    alpha_key = f"{prefix}/files/alpha.txt"
    beta_key = f"{prefix}/files/beta.txt"
    manifest_key = f"{prefix}/manifest.json"
    adapter = FailOnceS3Adapter()

    with pytest.raises(RuntimeError, match="injected upload interruption"):
        transfer_bundle.stage_bundle(source, manifest, adapter)

    receipt = transfer_bundle.stage_bundle(source, manifest, adapter)

    assert receipt.skipped == (alpha_key,)
    assert receipt.uploaded == (beta_key, manifest_key)
    assert [key for key, _source in adapter.put_file_calls] == [alpha_key, beta_key]
    assert adapter.objects[alpha_key] == b"alpha\n"
    assert adapter.objects[beta_key] == b"beta\n"
    assert adapter.objects[manifest_key] == manifest.to_bytes()


def test_stage_bundle_rejects_source_changed_since_manifest_before_remote_calls(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "alpha.txt").write_bytes(b"alpha\n")
    selected = source / "zeta.txt"
    selected.write_bytes(b"zeta\n")
    manifest = transfer_bundle.build_bundle(source)
    selected.write_bytes(b"omega\n")
    adapter = FakeS3Adapter()

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        transfer_bundle.stage_bundle(source, manifest, adapter)

    assert raised.value.path == "zeta.txt"
    assert raised.value.reason == "source_changed_since_manifest"
    assert adapter.head_calls == []
    assert adapter.put_file_calls == []
    assert adapter.put_bytes_calls == []


@pytest.mark.parametrize(
    ("mutation", "expected_path", "expected_reason"),
    (
        ("missing", "alpha.txt", "missing_file"),
        ("corrupt", "alpha.txt", "sha256_mismatch"),
        ("wrong_size", "alpha.txt", "size_mismatch"),
        ("unexpected", "extra.txt", "unexpected_path"),
        ("symlink", "alpha.txt", "symlink"),
    ),
)
def test_verify_bundle_rejects_incomplete_or_modified_mounted_content(
    tmp_path: Path,
    mutation: str,
    expected_path: str,
    expected_reason: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    selected = source / "alpha.txt"
    selected.write_bytes(b"alpha\n")
    manifest = transfer_bundle.build_bundle(source)
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    target = mounted / "alpha.txt"
    target.write_bytes(b"alpha\n")
    if mutation == "missing":
        target.unlink()
    elif mutation == "corrupt":
        target.write_bytes(b"omega\n")
    elif mutation == "wrong_size":
        target.write_bytes(b"short")
    elif mutation == "unexpected":
        (mounted / "extra.txt").write_bytes(b"extra")
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to(source / "alpha.txt")

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        transfer_bundle.verify_bundle(mounted, manifest)

    assert raised.value.path == expected_path
    assert raised.value.reason == expected_reason


def test_verify_and_promote_is_atomic_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "alpha.txt").write_bytes(b"alpha\n")
    manifest = transfer_bundle.build_bundle(source)
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "alpha.txt").write_bytes(b"alpha\n")
    final_parent = tmp_path / "bundles"

    first = transfer_bundle.verify_and_promote(incoming, final_parent, manifest)

    final_path = final_parent / manifest.bundle_digest
    assert first.path == final_path
    assert first.promoted is True
    assert not incoming.exists()
    assert (final_path / "alpha.txt").read_bytes() == b"alpha\n"

    incoming.mkdir()
    (incoming / "alpha.txt").write_bytes(b"alpha\n")
    second = transfer_bundle.verify_and_promote(incoming, final_parent, manifest)

    assert second.path == final_path
    assert second.promoted is False
    assert incoming.exists()
    assert (final_path / "alpha.txt").read_bytes() == b"alpha\n"


def test_verify_and_promote_leaves_final_absent_after_failed_verification(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "alpha.txt").write_bytes(b"alpha\n")
    manifest = transfer_bundle.build_bundle(source)
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "alpha.txt").write_bytes(b"omega\n")
    final_parent = tmp_path / "bundles"

    with pytest.raises(transfer_bundle.TransferBundleError):
        transfer_bundle.verify_and_promote(incoming, final_parent, manifest)

    assert not (final_parent / manifest.bundle_digest).exists()
    assert incoming.exists()


def test_verify_download_refuses_different_local_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "alpha.txt").write_bytes(b"alpha\n")
    expected = transfer_bundle.build_bundle(source)
    local = replace(expected, total_size=expected.total_size + 1)

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        transfer_bundle.verify_download(
            source,
            expected_manifest=expected,
            local_manifest=local,
            completed_artifact_digest="artifact-123",
            role="internal_ssd",
        )

    assert raised.value.path == "manifest.json"
    assert raised.value.reason == "manifest_mismatch"


def test_verify_backup_pair_requires_distinct_matching_internal_and_external_copies(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "alpha.txt").write_bytes(b"alpha\n")
    manifest = transfer_bundle.build_bundle(source)
    internal = tmp_path / "internal" / manifest.bundle_digest
    external = tmp_path / "external" / manifest.bundle_digest
    internal.mkdir(parents=True)
    external.mkdir(parents=True)
    (internal / "alpha.txt").write_bytes(b"alpha\n")
    (external / "alpha.txt").write_bytes(b"alpha\n")

    internal_copy = transfer_bundle.verify_download(
        internal,
        expected_manifest=manifest,
        local_manifest=manifest,
        completed_artifact_digest="artifact-123",
        role="internal_ssd",
    )
    external_copy = transfer_bundle.verify_download(
        external,
        expected_manifest=manifest,
        local_manifest=manifest,
        completed_artifact_digest="artifact-123",
        role="external_drive",
    )
    pair = transfer_bundle.verify_backup_pair(internal_copy, external_copy)

    assert pair.bundle_digest == manifest.bundle_digest
    assert pair.completed_artifact_digest == "artifact-123"
    assert pair.internal.path == internal.resolve()
    assert pair.external.path == external.resolve()

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        same_path = replace(external_copy, path=internal.resolve())
        transfer_bundle.verify_backup_pair(internal_copy, same_path)

    assert raised.value.reason == "backup_paths_not_distinct"


def test_verify_backup_pair_rejects_completed_artifact_identity_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "alpha.txt").write_bytes(b"alpha\n")
    manifest = transfer_bundle.build_bundle(source)
    internal = transfer_bundle.VerifiedCopy(
        role="internal_ssd",
        path=(tmp_path / "internal").resolve(),
        bundle_digest=manifest.bundle_digest,
        completed_artifact_digest="artifact-123",
    )
    external = transfer_bundle.VerifiedCopy(
        role="external_drive",
        path=(tmp_path / "external").resolve(),
        bundle_digest=manifest.bundle_digest,
        completed_artifact_digest="artifact-other",
    )

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        transfer_bundle.verify_backup_pair(internal, external)

    assert raised.value.reason == "completed_artifact_mismatch"


def test_retention_refusal_is_deterministic_until_every_gate_agrees(tmp_path: Path) -> None:
    pair = transfer_bundle.BackupPair(
        internal=transfer_bundle.VerifiedCopy(
            role="internal_ssd",
            path=(tmp_path / "internal").resolve(),
            bundle_digest="bundle-123",
            completed_artifact_digest="artifact-123",
        ),
        external=transfer_bundle.VerifiedCopy(
            role="external_drive",
            path=(tmp_path / "external").resolve(),
            bundle_digest="bundle-123",
            completed_artifact_digest="artifact-123",
        ),
        bundle_digest="bundle-123",
        completed_artifact_digest="artifact-123",
    )

    first = transfer_bundle.decide_retention(
        remote_identity="remote-bundle-123",
        bundle_digest="bundle-123",
        completed_artifact_digest="artifact-123",
        backups=None,
        authorization=None,
        run_active=True,
    )
    second = transfer_bundle.decide_retention(
        remote_identity="remote-bundle-123",
        bundle_digest="bundle-123",
        completed_artifact_digest="artifact-123",
        backups=None,
        authorization=None,
        run_active=True,
    )

    assert first == second
    assert first.allowed is False
    assert first.reasons == (
        "active_run",
        "verified_backup_pair_required",
        "authorization_required",
    )

    wrong_authorization = transfer_bundle.RetentionAuthorization(
        subject_digest="wrong",
        approved_by="operator",
    )
    mismatch = transfer_bundle.decide_retention(
        remote_identity="remote-bundle-123",
        bundle_digest="bundle-123",
        completed_artifact_digest="artifact-123",
        backups=pair,
        authorization=wrong_authorization,
        run_active=False,
    )
    assert mismatch.reasons == ("authorization_mismatch",)


def test_exact_retention_authorization_allows_isolated_fixture_deletion(tmp_path: Path) -> None:
    remote_root = tmp_path / "isolated-remote"
    target = remote_root / "remote-bundle-123"
    sibling = remote_root / "keep-me"
    target.mkdir(parents=True)
    sibling.mkdir()
    (target / "artifact.bin").write_bytes(b"delete only this fixture")
    pair = transfer_bundle.BackupPair(
        internal=transfer_bundle.VerifiedCopy(
            role="internal_ssd",
            path=(tmp_path / "internal").resolve(),
            bundle_digest="bundle-123",
            completed_artifact_digest="artifact-123",
        ),
        external=transfer_bundle.VerifiedCopy(
            role="external_drive",
            path=(tmp_path / "external").resolve(),
            bundle_digest="bundle-123",
            completed_artifact_digest="artifact-123",
        ),
        bundle_digest="bundle-123",
        completed_artifact_digest="artifact-123",
    )
    subject_digest = transfer_bundle.retention_subject_digest(
        remote_identity="remote-bundle-123",
        bundle_digest="bundle-123",
        completed_artifact_digest="artifact-123",
    )
    authorization = transfer_bundle.RetentionAuthorization(
        subject_digest=subject_digest,
        approved_by="operator",
    )
    decision = transfer_bundle.decide_retention(
        remote_identity="remote-bundle-123",
        bundle_digest="bundle-123",
        completed_artifact_digest="artifact-123",
        backups=pair,
        authorization=authorization,
        run_active=False,
    )

    assert decision.allowed is True
    assert decision.reasons == ()
    transfer_bundle.execute_retention(decision, remote_root)
    assert not target.exists()
    assert sibling.is_dir()


def test_retention_refusal_cannot_delete(tmp_path: Path) -> None:
    remote_root = tmp_path / "isolated-remote"
    target = remote_root / "remote-bundle-123"
    target.mkdir(parents=True)
    decision = transfer_bundle.decide_retention(
        remote_identity="remote-bundle-123",
        bundle_digest="bundle-123",
        completed_artifact_digest="artifact-123",
        backups=None,
        authorization=None,
        run_active=False,
    )

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        transfer_bundle.execute_retention(decision, remote_root)

    assert raised.value.reason == "retention_refused"
    assert target.is_dir()


def test_measured_input_bundle_fixture_is_canonical_and_excludes_historical_artifacts() -> None:
    fixture = (
        Path(__file__).parent / "fixtures" / "transfer" / "measured-input-bundle.json"
    ).read_bytes()

    manifest = transfer_bundle.BundleManifest.from_bytes(fixture)

    assert manifest.total_size == 1_395_349_697
    assert manifest.bundle_digest == (
        "b8584b5ce25fb96cff66ccb926b0b9b5bed560efc06ea1449f7e56521192e623"
    )
    assert manifest.to_bytes() == fixture
    assert {entry.path.split("/", maxsplit=1)[0] for entry in manifest.entries} == {
        "cache",
        "corpus",
        "dbn",
    }


def test_manifest_loader_rejects_tampered_root_digest() -> None:
    value = b'{"bundle_digest":"wrong","entries":[],"schema_version":1,"total_size":0}\n'

    with pytest.raises(transfer_bundle.TransferBundleError) as raised:
        transfer_bundle.BundleManifest.from_bytes(value)

    assert raised.value.path == "manifest.json"
    assert raised.value.reason == "bundle_digest_mismatch"


def test_s3_credentials_never_enter_bundle_observables(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "alpha.txt").write_bytes(b"alpha\n")
    manifest = transfer_bundle.build_bundle(source)
    adapter = FakeS3Adapter()
    adapter.local_credential = "S3-SECRET-SENTINEL"

    receipt = transfer_bundle.stage_bundle(source, manifest, adapter)

    observable = b"\n".join(
        (
            manifest.to_bytes(),
            repr(receipt).encode(),
            repr(adapter.head_calls).encode(),
            repr(adapter.put_file_calls).encode(),
            repr(adapter.put_bytes_calls).encode(),
            repr(adapter.objects).encode(),
        )
    )
    assert b"S3-SECRET-SENTINEL" not in observable
