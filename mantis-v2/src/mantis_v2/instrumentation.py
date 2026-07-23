"""Non-authoritative TensorBoard instrumentation for durable pipeline state."""

from __future__ import annotations

import json
import os
import resource
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore[import-untyped]
    EventAccumulator,
)
from torch.utils.tensorboard import SummaryWriter


def synchronize_device(device: torch.device) -> None:
    """Synchronize observable accelerator work before taking a measurement."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _host_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def collect_resource_metrics(run_root: Path, device: torch.device) -> dict[str, int | None]:
    """Collect a stable host/CUDA resource snapshot for authoritative JSON."""
    synchronize_device(device)
    allocated: int | None = None
    reserved: int | None = None
    utilization: int | None = None
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        try:
            utilization = int(torch.cuda.utilization(device))
        except (AttributeError, ImportError, OSError, RuntimeError):
            utilization = None
    run_root.mkdir(parents=True, exist_ok=True)
    return {
        "cuda_allocated_bytes": allocated,
        "cuda_reserved_bytes": reserved,
        "cuda_utilization_percent": utilization,
        "host_rss_bytes": _host_rss_bytes(),
        "filesystem_free_bytes": shutil.disk_usage(run_root).free,
    }


def parse_tensorboard_events(event_root: Path) -> dict[str, Any]:
    """Parse scalar and text events through TensorBoard's independent reader."""
    accumulator = EventAccumulator(str(event_root)).Reload()
    scalars = {
        tag: [{"step": event.step, "value": event.value} for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }
    text = {
        tag: [
            {
                "step": event.step,
                "value": event.tensor_proto.string_val[0].decode(),
            }
            for event in accumulator.Tensors(tag)
        ]
        for tag in accumulator.Tags()["tensors"]
        if tag.endswith("/text_summary")
    }
    return {"scalars": scalars, "text": text}


class RunInstrumentation:
    """Write observational events beneath one immutable run identity."""

    def __init__(self, run_root: Path) -> None:
        self._diagnostics = run_root / "instrumentation" / "diagnostics.jsonl"
        self._writer: SummaryWriter | None = None
        try:
            self._writer = SummaryWriter(log_dir=str(run_root / "events"))
        except Exception as exc:
            self._record_failure("writer_initialization", exc)

    def _record_failure(self, operation: str, exc: Exception) -> None:
        payload = {
            "error": str(exc),
            "error_type": type(exc).__name__,
            "operation": operation,
        }
        try:
            self._diagnostics.parent.mkdir(parents=True, exist_ok=True)
            with self._diagnostics.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            pass

    def _disable_writer(self, operation: str, exc: Exception) -> None:
        self._record_failure(operation, exc)
        writer, self._writer = self._writer, None
        if writer is not None:
            try:
                writer.close()
            except Exception as close_exc:
                self._record_failure("writer_close", close_exc)

    def scalars(self, values: Mapping[str, float | int], step: int) -> None:
        if self._writer is None:
            return
        try:
            for tag, value in values.items():
                self._writer.add_scalar(tag, value, step)
            self._writer.flush()
        except Exception as exc:
            self._disable_writer("write_scalars", exc)

    def text(self, tag: str, value: Mapping[str, object], step: int) -> None:
        if self._writer is None:
            return
        try:
            self._writer.add_text(tag, json.dumps(value, sort_keys=True), step)
            self._writer.flush()
        except Exception as exc:
            self._disable_writer("write_text", exc)

    def close(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.close()
        except Exception as exc:
            self._record_failure("writer_close", exc)
        finally:
            self._writer = None
