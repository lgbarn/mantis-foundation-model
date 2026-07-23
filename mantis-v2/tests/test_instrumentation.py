from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from mantis_v2 import cli
from mantis_v2.downstream_config import load_downstream_config
from mantis_v2.downstream_pipeline import _record_stage_instrumentation
from mantis_v2.instrumentation import collect_resource_metrics
from mantis_v2.monitoring import MonitoringError, tensorboard_command
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]


def test_missing_optional_cuda_utilization_does_not_fail_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 123)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 456)

    def missing_nvml(_device: torch.device) -> int:
        raise ModuleNotFoundError("nvidia-ml-py does not seem to be installed")

    monkeypatch.setattr(torch.cuda, "utilization", missing_nvml)

    metrics = collect_resource_metrics(tmp_path, torch.device("cuda"))

    assert metrics["cuda_allocated_bytes"] == 123
    assert metrics["cuda_reserved_bytes"] == 456
    assert metrics["cuda_utilization_percent"] is None
    assert metrics["filesystem_free_bytes"] > 0


def test_downstream_stage_uses_shared_json_and_tensorboard_interface(tmp_path: Path) -> None:
    base = load_downstream_config(ROOT / "configs" / "downstream-smoke.toml")
    config = replace(base, run=replace(base.run, artifact_root=tmp_path, name="observed"))

    telemetry = _record_stage_instrumentation(
        config,
        "embed",
        time.perf_counter(),
        {"step": 2, "rows": 8, "shards": 2},
        torch.device("cpu"),
    )

    assert json.loads(Path(telemetry["path"]).read_text()) == {
        key: value for key, value in telemetry.items() if key != "path"
    }
    events = EventAccumulator(str(tmp_path / "observed" / "events")).Reload()
    assert events.Scalars("stage/embed/rows")[-1].value == 8
    assert events.Scalars("stage/embed/completed")[-1].step == 2


def test_tensorboard_command_is_localhost_only(tmp_path: Path) -> None:
    command = tensorboard_command(tmp_path)
    assert command == (
        "tensorboard",
        "--logdir",
        str(tmp_path / "events"),
        "--host",
        "127.0.0.1",
        "--port",
        "6006",
    )
    for host in ("0.0.0.0", "::", "*", "localhost", "192.168.1.10"):
        with pytest.raises(MonitoringError, match="127.0.0.1"):
            tensorboard_command(tmp_path, host=host)


def test_tensorboard_cli_uses_local_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_serve(run_root: Path, *, host: str, port: int) -> dict[str, object]:
        observed.update(run_root=run_root, host=host, port=port)
        return {"returncode": 0}

    monkeypatch.setattr(cli, "serve_tensorboard", fake_serve)
    monkeypatch.setattr(sys, "argv", ["mantis-v2", "tensorboard", "--run-root", str(tmp_path)])
    cli.main()

    assert observed == {"run_root": tmp_path, "host": "127.0.0.1", "port": 6006}
    assert json.loads(capsys.readouterr().out)["returncode"] == 0


def test_remote_monitoring_docs_use_localhost_and_ssh_tunnel() -> None:
    repository_root = ROOT.parent
    documentation = "\n".join(
        (repository_root / path).read_text()
        for path in ("docs/workflow.md", "infra/runpod/README.md")
    )
    assert "just tensorboard /network/volume/runs/RUN_ID" in documentation
    assert "ssh -N -L 6006:127.0.0.1:6006" in documentation
    assert "--host 0.0.0.0" not in documentation
