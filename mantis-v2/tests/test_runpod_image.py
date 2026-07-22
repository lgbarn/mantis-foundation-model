from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mantis_v2.runpod_image import ImageContractError, _run, runtime_inventory, self_check

ROOT = Path(__file__).resolve().parents[2]


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available
        self.synchronized = False

    def is_available(self) -> bool:
        return self.available

    def synchronize(self) -> None:
        self.synchronized = True


class FakeTorch:
    __version__ = "2.13.0"
    version = SimpleNamespace(cuda="13.0")

    def __init__(self, available: bool) -> None:
        self.cuda = FakeCuda(available)

    def zeros(self, count: int, *, device: str) -> list[int]:
        assert count == 1
        assert device == "cuda"
        return [0]


def _runner(
    driver_available: bool = True,
    uv_output: str = "uv 0.11.30 (x86_64-unknown-linux-musl)",
):
    versions = {
        "git": "git version 2.43.0",
        "ssh": "OpenSSH_9.6p1 Ubuntu-3ubuntu13.18",
        "uv": uv_output,
    }

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                command, 0 if driver_available else 1, "580.65.06\n" if driver_available else "", ""
            )
        return subprocess.CompletedProcess(command, 0, versions[command[0]] + "\n", "")

    return run


def _environment(lock_sha256: str) -> dict[str, str]:
    return {
        "MANTIS_BASE_IMAGE": "nvidia/cuda@sha256:" + "1" * 64,
        "MANTIS_SOURCE_REVISION": "2" * 40,
        "MANTIS_SOURCE_TREE": "3" * 40,
        "MANTIS_LOCK_SHA256": lock_sha256,
        "MANTIS_IMAGE_CONTRACT_SHA256": "4" * 64,
        "MANTIS_UV_VERSION": "0.11.30",
    }


def test_runtime_inventory_is_canonical_and_complete(tmp_path: Path, monkeypatch) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n")
    import hashlib

    environment = _environment(hashlib.sha256(lock.read_bytes()).hexdigest())
    monkeypatch.setattr("mantis_v2.runpod_image.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("mantis_v2.runpod_image.platform.python_version", lambda: "3.12.11")

    result = runtime_inventory(
        environment=environment,
        runner=_runner(),
        torch_module=FakeTorch(True),
        lock_path=lock,
    )

    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    assert json.loads(encoded) == result
    assert result["identities"]["lock_sha256"] == environment["MANTIS_LOCK_SHA256"]
    assert result["driver"] == {"version": "580.65.06", "compatible": True, "error": ""}
    assert result["torch"]["cuda_runtime"] == "13.0"
    assert result["tools"]["uv"] == "uv 0.11.30 (x86_64-unknown-linux-musl)"
    assert result["packages"]


def test_runtime_inventory_rejects_wrong_uv_version_with_platform_suffix(
    tmp_path: Path, monkeypatch
) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n")
    import hashlib

    monkeypatch.setattr("mantis_v2.runpod_image.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("mantis_v2.runpod_image.platform.python_version", lambda: "3.12.11")

    with pytest.raises(ImageContractError, match="installed uv version"):
        runtime_inventory(
            environment=_environment(hashlib.sha256(lock.read_bytes()).hexdigest()),
            runner=_runner(uv_output="uv 0.11.29 (x86_64-unknown-linux-musl)"),
            torch_module=FakeTorch(True),
            lock_path=lock,
        )


def test_self_check_wraps_inventory_in_explicit_pass_receipt(monkeypatch) -> None:
    monkeypatch.setattr("mantis_v2.runpod_image.runtime_inventory", lambda: {"identity": "ok"})

    assert self_check() == {
        "schema_version": 1,
        "passed": True,
        "scope": "runtime_cuda",
        "inventory": {"identity": "ok"},
    }


@pytest.mark.parametrize(
    ("cuda_available", "driver_available"),
    ((False, True), (True, False)),
)
def test_cuda_failure_is_nonzero_before_workload_paths(
    tmp_path: Path, monkeypatch, cuda_available: bool, driver_available: bool
) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n")
    import hashlib

    forbidden = tmp_path / "market-data.parquet"
    run_directory = tmp_path / "runs" / "candidate"
    monkeypatch.setattr("mantis_v2.runpod_image.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("mantis_v2.runpod_image.platform.python_version", lambda: "3.12.11")

    with pytest.raises(ImageContractError) as failure:
        runtime_inventory(
            environment=_environment(hashlib.sha256(lock.read_bytes()).hexdigest()),
            runner=_runner(driver_available),
            torch_module=FakeTorch(cuda_available),
            lock_path=lock,
        )

    assert failure.value.inventory is not None
    assert failure.value.inventory["driver"]["compatible"] is False
    assert not forbidden.exists()
    assert not run_directory.exists()


def test_missing_executable_becomes_a_nonzero_command_result(monkeypatch) -> None:
    def missing(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)

    completed = _run(["nvidia-smi"])

    assert completed.returncode == 127
    assert "FileNotFoundError" in completed.stderr


def _scan_module() -> Any:
    path = ROOT / "infra" / "runpod" / "scripts" / "scan_image_archive.py"
    spec = importlib.util.spec_from_file_location("scan_image_archive", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _docker_archive(path: Path, files: dict[str, bytes]) -> None:
    layer_bytes = io.BytesIO()
    with tarfile.open(fileobj=layer_bytes, mode="w") as layer:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            layer.addfile(info, io.BytesIO(content))
    layer_payload = layer_bytes.getvalue()
    manifest = json.dumps(
        [{"Config": "config.json", "RepoTags": ["test:latest"], "Layers": ["layer.tar"]}]
    ).encode()
    with tarfile.open(path, mode="w") as image:
        for name, content in {"manifest.json": manifest, "layer.tar": layer_payload}.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            image.addfile(info, io.BytesIO(content))


def test_layer_scan_rejects_data_weights_artifacts_and_secrets(tmp_path: Path) -> None:
    archive = tmp_path / "image.tar"
    _docker_archive(
        archive,
        {
            "opt/mantis/data/input.parquet": b"market",
            "root/.ssh/id_ed25519": (
                b"-----BEGIN OPENSSH PRIVATE KEY-----\n"
                b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
                b"-----END OPENSSH PRIVATE KEY-----"
            ),
            "opt/mantis/checkpoints/best.safetensors": b"weights",
            "etc/leak": b"RUNPOD_API_KEY=secret-value",
        },
    )

    layers, violations = _scan_module().scan_archive(archive, b"")

    assert layers == 1
    assert len(violations) >= 4


def test_layer_scan_allows_dependency_metadata_and_embedded_key_labels(tmp_path: Path) -> None:
    archive = tmp_path / "image.tar"
    _docker_archive(
        archive,
        {
            "opt/mantis/.venv/lib/python3.12/site-packages/project.pth": b"metadata",
            "opt/mantis/.venv/lib/python3.12/site-packages/optuna/artifacts/__init__.py": b"",
            "root/.cache/uv/archive-v0/package/tests/fixture.parquet": b"fixture",
            "usr/bin/ssh": b"binary-----BEGIN OPENSSH PRIVATE KEY-----label",
        },
    )

    layers, violations = _scan_module().scan_archive(archive, b"")

    assert layers == 1
    assert violations == []


def test_image_contract_is_digest_pinned_frozen_and_localhost_only() -> None:
    dockerfile = (ROOT / "infra" / "runpod" / "Dockerfile").read_text()
    dockerignore = (ROOT / ".dockerignore").read_text()

    assert dockerfile.count("@sha256:") == 2
    assert "RUN --mount=type=cache,target=/root/.cache/uv" in dockerfile
    assert "uv sync --frozen --no-dev --all-packages" in dockerfile
    assert "python3.12=" in dockerfile
    assert "EXPOSE 22" in dockerfile
    assert "EXPOSE 6006" not in dockerfile
    assert "GatewayPorts no" in dockerfile
    assert "rm -f /etc/ssh/ssh_host_*" in dockerfile
    assert "COPY ." not in dockerfile
    assert dockerignore.startswith("**\n")
    build_script = (ROOT / "infra" / "runpod" / "scripts" / "build-image.sh").read_text()
    assert "git status --porcelain --untracked-files=all" in build_script
    for suffix in ("*.dbn", "*.parquet", "*.pt", "*.safetensors"):
        assert suffix not in dockerfile


def test_entrypoint_installs_only_the_runtime_public_key() -> None:
    entrypoint = (ROOT / "infra" / "runpod" / "scripts" / "container-entrypoint.sh").read_text()

    assert "rm -f /root/.ssh/authorized_keys" in entrypoint
    assert 'if [ -n "${PUBLIC_KEY:-}" ]; then' in entrypoint
    assert "printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys" in entrypoint
    assert "chmod 0600 /root/.ssh/authorized_keys" in entrypoint
    assert "ssh-keygen -l -f /root/.ssh/authorized_keys" in entrypoint


@pytest.mark.parametrize(
    "script",
    ("build-image.sh", "container-entrypoint.sh", "scan-image.sh", "self-check-image.sh"),
)
def test_image_shell_scripts_parse(script: str) -> None:
    completed = subprocess.run(
        ["sh", "-n", str(ROOT / "infra" / "runpod" / "scripts" / script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
