"""Fail-closed runtime inventory for the pinned RunPod CUDA image."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class ImageContractError(RuntimeError):
    """Raised when the image runtime does not satisfy its declared contract."""

    def __init__(self, message: str, inventory: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.inventory = inventory


Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        return subprocess.CompletedProcess(
            command,
            127,
            "",
            f"{type(exc).__name__}: {exc}",
        )


def _version(runner: Runner, command: list[str]) -> str:
    completed = runner(command)
    value = (completed.stdout or completed.stderr).strip().splitlines()
    if completed.returncode != 0 or not value:
        raise ImageContractError(f"required executable failed: {command[0]}")
    return value[0]


def runtime_inventory(
    *,
    environment: Mapping[str, str] = os.environ,
    runner: Runner = _run,
    torch_module: Any | None = None,
    lock_path: Path = Path("/opt/mantis/uv.lock"),
    require_cuda: bool = True,
) -> dict[str, Any]:
    required = (
        "MANTIS_BASE_IMAGE",
        "MANTIS_SOURCE_REVISION",
        "MANTIS_SOURCE_TREE",
        "MANTIS_LOCK_SHA256",
        "MANTIS_IMAGE_CONTRACT_SHA256",
        "MANTIS_UV_VERSION",
    )
    missing = [name for name in required if not environment.get(name)]
    if missing:
        raise ImageContractError(f"missing image identities: {', '.join(missing)}")
    if not lock_path.is_file() or _sha256(lock_path) != environment["MANTIS_LOCK_SHA256"]:
        raise ImageContractError("installed uv.lock does not match the declared digest")

    torch = torch_module
    if torch is None:
        import torch as imported_torch

        torch = imported_torch
    cuda_runtime = str(torch.version.cuda or "unavailable")
    cuda_available = bool(torch.cuda.is_available())
    cuda_error = ""
    if cuda_available:
        try:
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
        except Exception as exc:
            cuda_available = False
            cuda_error = f"{type(exc).__name__}: {exc}"
    driver = runner(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    driver_version = (
        driver.stdout.strip().splitlines()[0] if driver.returncode == 0 else "unavailable"
    )
    driver_error = driver.stderr.strip() if driver.returncode != 0 else ""
    compatible = cuda_available and cuda_runtime.startswith("13.") and driver.returncode == 0
    packages = sorted(
        (
            {
                "name": distribution.metadata["Name"],
                "version": distribution.version,
            }
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        ),
        key=lambda item: item["name"].lower(),
    )
    inventory: dict[str, Any] = {
        "schema_version": 1,
        "identities": {
            "image_contract_sha256": environment["MANTIS_IMAGE_CONTRACT_SHA256"],
            "base_image": environment["MANTIS_BASE_IMAGE"],
            "source_revision": environment["MANTIS_SOURCE_REVISION"],
            "source_tree": environment["MANTIS_SOURCE_TREE"],
            "lock_sha256": environment["MANTIS_LOCK_SHA256"],
        },
        "platform": {"architecture": platform.machine(), "python": platform.python_version()},
        "tools": {
            "git": _version(runner, ["git", "--version"]),
            "ssh": _version(runner, ["ssh", "-V"]),
            "uv": _version(runner, ["uv", "--version"]),
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_runtime": cuda_runtime,
            "cuda_available": cuda_available,
        },
        "driver": {
            "version": driver_version,
            "compatible": compatible,
            "error": cuda_error or driver_error,
        },
        "packages": packages,
    }
    if require_cuda and not compatible:
        raise ImageContractError("CUDA runtime or driver is incompatible", inventory)
    if inventory["platform"]["architecture"] != "x86_64":
        raise ImageContractError("image runtime architecture must be x86_64")
    if inventory["platform"]["python"].split(".")[:2] != ["3", "12"]:
        raise ImageContractError("image runtime requires Python 3.12")
    expected_uv = environment["MANTIS_UV_VERSION"]
    if inventory["tools"]["uv"].split()[:2] != ["uv", expected_uv]:
        raise ImageContractError("installed uv version does not match the image contract")
    return inventory


def self_check(*, require_cuda: bool = True) -> dict[str, Any]:
    """Collect deterministic inventory before any workload path is opened."""
    return {
        "schema_version": 1,
        "passed": True,
        "scope": "runtime_cuda" if require_cuda else "static_image",
        "inventory": runtime_inventory() if require_cuda else runtime_inventory(require_cuda=False),
    }
