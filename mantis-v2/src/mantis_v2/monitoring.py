"""Local-only TensorBoard launch contract."""

from __future__ import annotations

import subprocess
from pathlib import Path


class MonitoringError(ValueError):
    """Raised when monitoring would violate the local-only contract."""


def tensorboard_command(
    run_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 6006,
) -> tuple[str, ...]:
    if host != "127.0.0.1":
        raise MonitoringError("TensorBoard host must be exactly 127.0.0.1")
    if not 1 <= port <= 65535:
        raise MonitoringError("TensorBoard port must be between 1 and 65535")
    return (
        "tensorboard",
        "--logdir",
        str(run_root / "events"),
        "--host",
        host,
        "--port",
        str(port),
    )


def serve_tensorboard(run_root: Path, *, host: str, port: int) -> dict[str, object]:
    command = tensorboard_command(run_root, host=host, port=port)
    completed = subprocess.run(command, check=False)
    return {"command": list(command), "returncode": completed.returncode}
