"""Fail-closed BF16 qualification against the accepted CUDA FP32 reference."""

from __future__ import annotations

import json
import math
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn

from mantis_v2.config import TargetConfig
from mantis_v2.cuda_qualification import (
    NumericEvidence,
    ResumeContract,
    ResumeEvidence,
    Tolerance,
    compare_numeric_evidence,
    compare_resume_evidence,
)
from mantis_v2.instrumentation import synchronize_device
from mantis_v2.model import NextLegOutput, nextleg_loss
from mantis_v2.precision import (
    autocast_context,
    validate_optimizer_state,
    validate_precision_device,
)


class Bf16QualificationError(ValueError):
    """Raised when the BF16 qualification contract or evidence is invalid."""


@dataclass(frozen=True)
class Bf16QualificationConfig:
    schema_version: int
    precision: Literal["bf16"]
    reference_precision: Literal["fp32"]
    parity: Tolerance
    resume: ResumeContract
    export: Tolerance


@dataclass(frozen=True)
class Bf16CandidateEvidence:
    numeric: NumericEvidence
    uninterrupted: ResumeEvidence
    resumed: ResumeEvidence
    export_native: NumericEvidence
    export_reloaded: NumericEvidence
    optimizer_precision: str
    precision_records: Mapping[str, str]


def _flat_values(tensor: torch.Tensor) -> list[float]:
    values = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    if not torch.isfinite(values).all().item():
        raise Bf16QualificationError("BF16 fixture produced non-finite tensor evidence")
    return [float(value) for value in values]


def capture_bf16_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, torch.Tensor],
    target: TargetConfig,
    device: torch.device,
) -> NumericEvidence:
    """Capture one CUDA BF16 fixture step while retaining FP32 parameters and state."""
    validate_precision_device("bf16", device)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise Bf16QualificationError("BF16 qualification requires FP32 model parameters")
    required_batch = {"context", "candle_target", "leg_target"}
    if set(batch) != required_batch:
        raise Bf16QualificationError(f"BF16 parity batch must contain {sorted(required_batch)}")
    moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    synchronize_device(device)
    with autocast_context("bf16", device):
        raw_output = model(moved["context"])
        if not isinstance(raw_output, dict) or set(raw_output) != {"candle", "leg"}:
            raise Bf16QualificationError("fixture model must return candle and leg outputs")
        output = cast(NextLegOutput, raw_output)
        losses = nextleg_loss(output, moved["candle_target"], moved["leg_target"], target)
    if not all(
        torch.isfinite(loss).item() for loss in (losses["candle"], losses["leg"], losses["total"])
    ):
        raise Bf16QualificationError("BF16 fixture produced non-finite loss")
    losses["total"].backward()  # type: ignore[no-untyped-call]
    gradients: dict[str, list[float]] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise Bf16QualificationError(f"trainable fixture parameter has no gradient: {name}")
            gradients[name] = _flat_values(parameter.grad)
    optimizer.step()
    validate_optimizer_state(optimizer)
    synchronize_device(device)
    return {
        "outputs": {
            "candle": _flat_values(output["candle"]),
            "leg": _flat_values(output["leg"]),
        },
        "losses": {
            "candle": float(losses["candle"].detach().cpu()),
            "leg": float(losses["leg"].detach().cpu()),
            "total": float(losses["total"].detach().cpu()),
        },
        "gradients": gradients,
        "updated_parameters": {
            name: _flat_values(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
    }


def _section(raw: dict[str, Any], name: str, expected: set[str]) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict) or set(value) != expected:
        raise Bf16QualificationError(f"invalid [{name}] section")
    return value


def _tolerance(raw: dict[str, Any], name: str) -> Tolerance:
    section = _section(raw, name, {"atol", "rtol"})
    values = (section["atol"], section["rtol"])
    if not all(
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        for value in values
    ):
        raise Bf16QualificationError(f"[{name}] tolerances must be finite and non-negative")
    return Tolerance(float(values[0]), float(values[1]))


def load_bf16_qualification_config(path: str | Path) -> Bf16QualificationConfig:
    """Load the preregistered BF16-vs-FP32 qualification policy."""
    try:
        raw = tomllib.loads(Path(path).read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise Bf16QualificationError(f"cannot load BF16 qualification config: {exc}") from exc
    expected = {"schema_version", "precision", "reference_precision", "parity", "resume", "export"}
    if set(raw) != expected or raw["schema_version"] != 1:
        raise Bf16QualificationError("invalid BF16 qualification schema")
    if raw["precision"] != "bf16" or raw["reference_precision"] != "fp32":
        raise Bf16QualificationError("BF16 qualification must use the FP32 reference")
    resume = _section(raw, "resume", {"total_updates", "interrupt_after_updates", "atol", "rtol"})
    if resume["total_updates"] != 32 or resume["interrupt_after_updates"] != 16:
        raise Bf16QualificationError("BF16 resume oracle must compare 32 with 16+restart")
    resume_values = (resume["atol"], resume["rtol"])
    if not all(
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        for value in resume_values
    ):
        raise Bf16QualificationError("[resume] tolerances must be finite and non-negative")
    return Bf16QualificationConfig(
        schema_version=1,
        precision="bf16",
        reference_precision="fp32",
        parity=_tolerance(raw, "parity"),
        resume=ResumeContract(32, 16, float(resume_values[0]), float(resume_values[1])),
        export=_tolerance(raw, "export"),
    )


def _write_once(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise Bf16QualificationError(f"qualification evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Bf16QualificationError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise Bf16QualificationError(f"{label} must be a JSON object")
    return value


def _load_resume(raw: object, label: str) -> ResumeEvidence:
    expected = {
        "subject_digest",
        "data_digest",
        "source_revision",
        "lock_digest",
        "completed_updates",
        "optimizer_state_hash",
        "rng_state_hash",
        "tensors",
        "metrics",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise Bf16QualificationError(f"invalid {label} resume evidence")
    try:
        return ResumeEvidence(**raw)
    except TypeError as exc:
        raise Bf16QualificationError(f"invalid {label} resume evidence: {exc}") from exc


def load_bf16_candidate_evidence(path: str | Path) -> Bf16CandidateEvidence:
    """Load the strict JSON handoff emitted by a paid CUDA evidence run."""
    raw = _load_json_object(Path(path), "BF16 candidate evidence")
    expected = {
        "numeric",
        "uninterrupted",
        "resumed",
        "export_native",
        "export_reloaded",
        "optimizer_precision",
        "precision_records",
    }
    if set(raw) != expected:
        raise Bf16QualificationError("invalid BF16 candidate evidence schema")
    numeric_fields = ("numeric", "export_native", "export_reloaded")
    for field in numeric_fields:
        value = raw[field]
        if not isinstance(value, dict):
            raise Bf16QualificationError(f"{field} must be a JSON object")
        compare_numeric_evidence(value, value, Tolerance(0.0, 0.0))
    records = raw["precision_records"]
    if not isinstance(records, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in records.items()
    ):
        raise Bf16QualificationError("precision_records must map strings to strings")
    if not isinstance(raw["optimizer_precision"], str):
        raise Bf16QualificationError("optimizer_precision must be a string")
    return Bf16CandidateEvidence(
        numeric=cast(NumericEvidence, raw["numeric"]),
        uninterrupted=_load_resume(raw["uninterrupted"], "uninterrupted"),
        resumed=_load_resume(raw["resumed"], "resumed"),
        export_native=cast(NumericEvidence, raw["export_native"]),
        export_reloaded=cast(NumericEvidence, raw["export_reloaded"]),
        optimizer_precision=raw["optimizer_precision"],
        precision_records=records,
    )


def qualify_bf16_files(
    *,
    config_path: Path,
    reference_path: Path,
    output_path: Path,
    candidate_path: Path | None = None,
    failure: str | None = None,
) -> dict[str, object]:
    """Qualify captured evidence or durably record one failed BF16 attempt."""
    if (candidate_path is None) == (failure is None):
        raise Bf16QualificationError("provide exactly one of candidate_path or failure")
    config = load_bf16_qualification_config(config_path)
    reference = _load_json_object(reference_path, "FP32 reference evidence")
    compare_numeric_evidence(reference, reference, Tolerance(0.0, 0.0))

    def run_candidate() -> Bf16CandidateEvidence:
        if failure is not None:
            raise RuntimeError(failure)
        if candidate_path is None:
            raise Bf16QualificationError("candidate evidence path is missing")
        return load_bf16_candidate_evidence(candidate_path)

    return qualify_bf16(
        config,
        cast(NumericEvidence, reference),
        run_candidate,
        output_path,
    )


def qualify_bf16(
    config: Bf16QualificationConfig,
    fp32_reference: NumericEvidence,
    candidate_runner: Callable[[], Bf16CandidateEvidence],
    output_path: Path,
) -> dict[str, object]:
    """Run one BF16 candidate attempt and durably select BF16 or the FP32 reference."""
    failure: str | None = None
    try:
        candidate = candidate_runner()
        compare_numeric_evidence(fp32_reference, candidate.numeric, config.parity)
        compare_resume_evidence(candidate.uninterrupted, candidate.resumed, config.resume)
        compare_numeric_evidence(candidate.export_native, candidate.export_reloaded, config.export)
        if candidate.optimizer_precision != "fp32":
            raise Bf16QualificationError("BF16 optimizer state must remain FP32")
        required_records = {
            "config",
            "provenance",
            "checkpoint",
            "evaluation",
            "export",
            "tensorboard",
            "qualification",
        }
        if set(candidate.precision_records) != required_records or any(
            value != "bf16" for value in candidate.precision_records.values()
        ):
            raise Bf16QualificationError("BF16 precision records are incomplete or mismatched")
    except (RuntimeError, ValueError) as exc:
        failure = f"{type(exc).__name__}: {exc}"

    passed = failure is None
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "passed" if passed else "rejected",
        "selected_precision": "bf16" if passed else "fp32",
        "candidate_precision": "bf16",
        "reference_precision": "fp32",
        "attempts": 1,
        "automatic_retry": False,
        "tolerances": {
            "parity": {"atol": config.parity.atol, "rtol": config.parity.rtol},
            "resume": {"atol": config.resume.atol, "rtol": config.resume.rtol},
            "export": {"atol": config.export.atol, "rtol": config.export.rtol},
        },
        "failure": failure,
        "holdout_unlocked": False,
    }
    _write_once(output_path, result)
    return result
