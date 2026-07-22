"""Portable, CPU-testable contracts for CUDA FP32 qualification."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import torch
from torch import nn

from mantis_v2.config import PipelineConfig, TargetConfig
from mantis_v2.instrumentation import synchronize_device
from mantis_v2.model import NextLegOutput, nextleg_loss


class QualificationError(ValueError):
    """Raised when qualification inputs or evidence violate the accepted contract."""


@dataclass(frozen=True)
class ProbeContract:
    train_updates: int
    validation_batches: int
    pin_memory: bool
    run_id_pattern: str


@dataclass(frozen=True)
class OfficialBaseContract:
    source_repository: str
    source_revision: str
    hub_model: str
    hub_revision: str
    weights_sha256: str


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


@dataclass(frozen=True)
class BatchEnvelopeContract:
    initial_batch_size: int
    growth_factor: int
    memory_cap_fraction: float


@dataclass(frozen=True)
class ResumeContract:
    total_updates: int
    interrupt_after_updates: int
    atol: float
    rtol: float


@dataclass(frozen=True)
class TrainabilityContract:
    full_finetune_trainable_parameters: int
    full_finetune_frozen_parameters: int
    transformer_finetune_trainable_parameters: int
    transformer_finetune_frozen_parameters: int


@dataclass(frozen=True)
class CudaQualificationConfig:
    schema_version: int
    precision: Literal["fp32"]
    probe: ProbeContract
    official_base: OfficialBaseContract
    parity: Tolerance
    batch_envelope: BatchEnvelopeContract
    resume: ResumeContract
    export: Tolerance
    trainability: TrainabilityContract


TrialOutcome = Literal["clean", "cuda_oom", "host_oom", "error"]


@dataclass(frozen=True)
class BatchTrial:
    batch_size: int
    outcome: TrialOutcome
    throughput_samples_per_second: float | None
    peak_gpu_bytes: int | None
    peak_host_bytes: int | None
    data_wait_seconds: float | None
    synchronized: bool
    fresh_process: bool


@dataclass(frozen=True)
class BatchEnvelopeResult:
    selected_batch_size: int
    first_oom_batch_size: int
    trials: tuple[BatchTrial, ...]


@dataclass(frozen=True)
class ResumeEvidence:
    subject_digest: str
    data_digest: str
    source_revision: str
    lock_digest: str
    completed_updates: int
    optimizer_state_hash: str
    rng_state_hash: str
    tensors: Mapping[str, Sequence[float]]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class TrainabilityEvidence:
    mode: str
    trainable_parameters: int
    frozen_parameters: int
    initialization_mode: str
    source_repository: str
    source_revision: str
    hub_model: str
    hub_revision: str
    weights_sha256: str
    fallback_checkpoint: str | None


@dataclass(frozen=True)
class EnvironmentEvidence:
    image_digest: str
    base_image_digest: str
    source_revision: str
    lock_digest: str
    torch_version: str
    cuda_runtime: str
    cuda_driver: str
    device_name: str
    device_capability: str
    device_total_memory_bytes: int
    device: str
    precision: str
    pin_memory: bool
    read_artifacts: tuple[str, ...]
    emitted_artifacts: tuple[str, ...]
    holdout_unlocked: bool


@dataclass(frozen=True)
class ArtifactParityEvidence:
    format: str
    selected_checkpoint_digest: str
    evaluated_checkpoint_digest: str
    exported_checkpoint_digest: str
    config_digest: str
    data_digest: str
    source_revision: str
    lock_digest: str
    upstream_weights_sha256: str
    native_outputs: Mapping[str, Sequence[float]]
    reloaded_outputs: Mapping[str, Sequence[float]]
    holdout_unlocked: bool


class NumericEvidence(TypedDict):
    outputs: dict[str, list[float]]
    losses: dict[str, float]
    gradients: dict[str, list[float]]
    updated_parameters: dict[str, list[float]]


class FixtureBatch(TypedDict):
    context: torch.Tensor
    candle_target: torch.Tensor
    leg_target: torch.Tensor


def _section(raw: dict[str, Any], name: str, expected: set[str]) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise QualificationError(f"missing or invalid [{name}] section")
    unknown = set(value) - expected
    if unknown:
        raise QualificationError(f"unknown [{name}] keys: {', '.join(sorted(unknown))}")
    missing = expected - set(value)
    if missing:
        raise QualificationError(f"missing [{name}] keys: {', '.join(sorted(missing))}")
    return value


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise QualificationError(f"{field} must be a positive integer")
    return value


def _tolerance(raw: dict[str, Any], section: str) -> Tolerance:
    values = _section(raw, section, {"atol", "rtol"})
    atol = values["atol"]
    rtol = values["rtol"]
    valid = all(
        isinstance(value, int | float) and math.isfinite(value) and value >= 0
        for value in (atol, rtol)
    )
    if not valid:
        raise QualificationError(f"[{section}] tolerances must be finite and non-negative")
    return Tolerance(float(atol), float(rtol))


def load_qualification_config(path: str | Path) -> CudaQualificationConfig:
    """Load the strict, versioned CUDA qualification contract."""
    config_path = Path(path)
    try:
        raw = tomllib.loads(config_path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise QualificationError(f"cannot load qualification config {config_path}: {exc}") from exc

    expected_top = {
        "schema_version",
        "precision",
        "probe",
        "official_base",
        "parity",
        "batch_envelope",
        "resume",
        "export",
        "trainability",
    }
    unknown = set(raw) - expected_top
    missing = expected_top - set(raw)
    if unknown:
        raise QualificationError(f"unknown qualification keys: {', '.join(sorted(unknown))}")
    if missing:
        raise QualificationError(f"missing qualification keys: {', '.join(sorted(missing))}")
    if raw["schema_version"] != 1:
        raise QualificationError("schema_version must be 1")
    if raw["precision"] != "fp32":
        raise QualificationError("precision must be fp32")

    probe_raw = _section(
        raw, "probe", {"train_updates", "validation_batches", "pin_memory", "run_id_pattern"}
    )
    probe = ProbeContract(
        train_updates=_positive_int(probe_raw["train_updates"], "probe.train_updates"),
        validation_batches=_positive_int(
            probe_raw["validation_batches"], "probe.validation_batches"
        ),
        pin_memory=probe_raw["pin_memory"],
        run_id_pattern=probe_raw["run_id_pattern"],
    )
    if probe.train_updates != 32 or probe.validation_batches != 1 or probe.pin_memory is not True:
        raise QualificationError(
            "probe must require 32 train updates, one validation batch, and pinning"
        )
    if not isinstance(probe.run_id_pattern, str):
        raise QualificationError("probe.run_id_pattern must be a string")
    try:
        re.compile(probe.run_id_pattern)
    except re.error as exc:
        raise QualificationError(f"probe.run_id_pattern is invalid: {exc}") from exc

    base_raw = _section(
        raw,
        "official_base",
        {"source_repository", "source_revision", "hub_model", "hub_revision", "weights_sha256"},
    )
    if not all(isinstance(value, str) and value for value in base_raw.values()):
        raise QualificationError("official-base identities must be non-empty strings")
    official_base = OfficialBaseContract(**base_raw)

    envelope_raw = _section(
        raw, "batch_envelope", {"initial_batch_size", "growth_factor", "memory_cap_fraction"}
    )
    batch_envelope = BatchEnvelopeContract(
        initial_batch_size=_positive_int(
            envelope_raw["initial_batch_size"], "batch_envelope.initial_batch_size"
        ),
        growth_factor=_positive_int(envelope_raw["growth_factor"], "batch_envelope.growth_factor"),
        memory_cap_fraction=float(envelope_raw["memory_cap_fraction"]),
    )
    if batch_envelope.initial_batch_size != 32 or batch_envelope.growth_factor != 2:
        raise QualificationError("batch envelope must start at 32 and double")
    if not 0 < batch_envelope.memory_cap_fraction <= 0.8:
        raise QualificationError("batch envelope memory cap must be in (0, 0.8]")

    resume_raw = _section(
        raw, "resume", {"total_updates", "interrupt_after_updates", "atol", "rtol"}
    )
    resume = ResumeContract(
        total_updates=_positive_int(resume_raw["total_updates"], "resume.total_updates"),
        interrupt_after_updates=_positive_int(
            resume_raw["interrupt_after_updates"], "resume.interrupt_after_updates"
        ),
        atol=float(resume_raw["atol"]),
        rtol=float(resume_raw["rtol"]),
    )
    if resume.total_updates != 32 or resume.interrupt_after_updates != 16:
        raise QualificationError("resume oracle must compare 32 updates with a restart after 16")
    if not all(math.isfinite(value) and value >= 0 for value in (resume.atol, resume.rtol)):
        raise QualificationError("[resume] tolerances must be finite and non-negative")

    trainability_raw = _section(
        raw,
        "trainability",
        {
            "full_finetune_trainable_parameters",
            "full_finetune_frozen_parameters",
            "transformer_finetune_trainable_parameters",
            "transformer_finetune_frozen_parameters",
        },
    )
    trainability = TrainabilityContract(
        **{
            key: _positive_int(value, f"trainability.{key}")
            for key, value in trainability_raw.items()
        }
    )

    return CudaQualificationConfig(
        schema_version=1,
        precision="fp32",
        probe=probe,
        official_base=official_base,
        parity=_tolerance(raw, "parity"),
        batch_envelope=batch_envelope,
        resume=resume,
        export=_tolerance(raw, "export"),
        trainability=trainability,
    )


def build_probe_config(
    pipeline: PipelineConfig,
    qualification: CudaQualificationConfig,
    *,
    run_id: str,
    artifact_root: Path,
) -> PipelineConfig:
    """Build a fail-closed, disposable CUDA probe configuration."""
    if re.fullmatch(qualification.probe.run_id_pattern, run_id) is None:
        raise QualificationError("run_id is not a unique disposable run identity")
    if (artifact_root / run_id).exists():
        raise QualificationError(
            f"qualification artifact path already exists: {artifact_root / run_id}"
        )
    if pipeline.evaluation.allow_holdout:
        raise QualificationError("CUDA qualification must not unlock the holdout")
    if pipeline.data.root == "synthetic":
        raise QualificationError("CUDA qualification requires real pre-holdout data")
    if pipeline.model.mode == "scratch":
        raise QualificationError("CUDA qualification requires official-base initialization")
    if pipeline.model.mode not in {"full_finetune", "transformer_finetune"}:
        raise QualificationError("CUDA qualification requires a supported fine-tune mode")

    expected_base = qualification.official_base
    actual_base = pipeline.model
    for field in (
        "source_repository",
        "source_revision",
        "hub_model",
        "hub_revision",
        "weights_sha256",
    ):
        if getattr(actual_base, field) != getattr(expected_base, field):
            raise QualificationError(f"official-base identity mismatch: {field}")

    return replace(
        pipeline,
        run=replace(
            pipeline.run,
            name=run_id,
            artifact_root=artifact_root,
            device="cuda",
            require_accelerator=True,
            allow_overwrite=False,
        ),
        training=replace(
            pipeline.training,
            epochs=1,
            resume=False,
            max_steps_per_epoch=qualification.probe.train_updates,
            validation_max_steps=qualification.probe.validation_batches,
        ),
        export=replace(
            pipeline.export,
            verify_atol=qualification.export.atol,
            verify_rtol=qualification.export.rtol,
        ),
    )


def qualification_subject_digest(config: PipelineConfig) -> str:
    """Identify model/data/training semantics across isolated disposable process roots."""
    payload = asdict(config)
    run = cast(dict[str, object], payload["run"])
    run["name"] = "<disposable-run>"
    run["artifact_root"] = "<disposable-root>"
    canonical = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _compare_value(reference: Any, candidate: Any, tolerance: Tolerance, path: str) -> None:
    if isinstance(reference, Mapping):
        if not isinstance(candidate, Mapping) or set(reference) != set(candidate):
            raise QualificationError(f"numeric evidence shape mismatch at {path}")
        for key in sorted(reference):
            child = f"{path}.{key}" if path else str(key)
            _compare_value(reference[key], candidate[key], tolerance, child)
        return
    if isinstance(reference, Sequence) and not isinstance(reference, str | bytes):
        if not isinstance(candidate, Sequence) or isinstance(candidate, str | bytes):
            raise QualificationError(f"numeric evidence shape mismatch at {path}")
        if len(reference) != len(candidate):
            raise QualificationError(f"numeric evidence shape mismatch at {path}")
        for index, (expected, actual) in enumerate(zip(reference, candidate, strict=True)):
            _compare_value(expected, actual, tolerance, f"{path}[{index}]")
        return
    if not isinstance(reference, int | float) or not isinstance(candidate, int | float):
        if reference != candidate:
            raise QualificationError(f"numeric evidence mismatch at {path}")
        return
    if not math.isfinite(float(reference)) or not math.isfinite(float(candidate)):
        raise QualificationError(f"non-finite numeric evidence at {path}")
    if not math.isclose(
        float(reference), float(candidate), abs_tol=tolerance.atol, rel_tol=tolerance.rtol
    ):
        raise QualificationError(
            f"numeric evidence mismatch at {path}: expected {reference}, observed {candidate}"
        )


def compare_numeric_evidence(
    cpu: Mapping[str, Any], cuda: Mapping[str, Any], tolerance: Tolerance
) -> None:
    """Compare the complete fixed-fixture FP32 evidence tree."""
    required = {"outputs", "losses", "gradients", "updated_parameters"}
    if set(cpu) != required or set(cuda) != required:
        raise QualificationError(
            "numeric evidence must contain outputs, losses, gradients, and updated_parameters"
        )
    _compare_value(cpu, cuda, tolerance, "")


def select_batch_envelope(
    trials: Sequence[BatchTrial],
    contract: BatchEnvelopeContract,
    *,
    gpu_total_bytes: int,
    host_total_bytes: int,
) -> BatchEnvelopeResult:
    """Validate fresh-process doubling trials and select the largest capped clean batch."""
    if gpu_total_bytes <= 0 or host_total_bytes <= 0:
        raise QualificationError("GPU and host total memory must be positive")
    if not trials:
        raise QualificationError("batch envelope requires trials")

    expected_batch_size = contract.initial_batch_size
    clean: list[BatchTrial] = []
    first_oom: BatchTrial | None = None
    for trial in trials:
        if not trial.fresh_process:
            raise QualificationError(f"batch {trial.batch_size} was not run in a fresh process")
        if trial.batch_size != expected_batch_size:
            raise QualificationError(
                f"batch trials must double from {contract.initial_batch_size}; "
                f"expected {expected_batch_size}, observed {trial.batch_size}"
            )
        if trial.outcome == "error":
            raise QualificationError(f"unclassified batch trial failure at {trial.batch_size}")
        if trial.outcome in {"cuda_oom", "host_oom"}:
            first_oom = trial
            break
        if not trial.synchronized:
            raise QualificationError(f"batch {trial.batch_size} telemetry was not synchronized")
        metrics = (
            trial.throughput_samples_per_second,
            trial.peak_gpu_bytes,
            trial.peak_host_bytes,
            trial.data_wait_seconds,
        )
        if any(value is None or not math.isfinite(float(value)) or value < 0 for value in metrics):
            raise QualificationError(f"batch {trial.batch_size} has invalid telemetry")
        clean.append(trial)
        expected_batch_size *= contract.growth_factor

    if first_oom is None:
        raise QualificationError("batch trials must continue through the first classified OOM")
    if trials[-1] is not first_oom:
        raise QualificationError("batch trials must stop at the first classified OOM")

    gpu_cap = gpu_total_bytes * contract.memory_cap_fraction
    host_cap = host_total_bytes * contract.memory_cap_fraction
    eligible = [
        trial
        for trial in clean
        if trial.peak_gpu_bytes is not None
        and trial.peak_host_bytes is not None
        and trial.peak_gpu_bytes <= gpu_cap
        and trial.peak_host_bytes <= host_cap
    ]
    if not eligible:
        raise QualificationError("no clean batch satisfies both 80% memory caps")
    return BatchEnvelopeResult(eligible[-1].batch_size, first_oom.batch_size, tuple(trials))


def compare_resume_evidence(
    uninterrupted: ResumeEvidence,
    resumed: ResumeEvidence,
    contract: ResumeContract,
) -> None:
    """Compare 32 uninterrupted updates with the 16-update restart path."""
    exact_fields = (
        "subject_digest",
        "data_digest",
        "source_revision",
        "lock_digest",
        "completed_updates",
        "optimizer_state_hash",
        "rng_state_hash",
    )
    for field in exact_fields:
        left = getattr(uninterrupted, field)
        right = getattr(resumed, field)
        if left != right:
            raise QualificationError(f"resume evidence mismatch: {field}")
    if uninterrupted.completed_updates != contract.total_updates:
        raise QualificationError(
            f"resume evidence must complete exactly {contract.total_updates} updates"
        )
    tolerance = Tolerance(contract.atol, contract.rtol)
    _compare_value(uninterrupted.tensors, resumed.tensors, tolerance, "tensors")
    _compare_value(uninterrupted.metrics, resumed.metrics, tolerance, "metrics")


def cpu_skip_record(*, cuda_available: bool, reason: str) -> dict[str, object]:
    """Return the canonical CPU-only schema without touching qualification artifacts."""
    if cuda_available:
        raise QualificationError("cannot emit a CPU skip record with available CUDA")
    if not reason.strip():
        raise QualificationError("CPU skip reason must be non-empty")
    return {
        "schema_version": 1,
        "status": "skipped",
        "device": "cuda",
        "precision": "fp32",
        "reason_code": "cuda_unavailable",
        "reason": reason,
        "holdout_unlocked": False,
    }


def validate_trainability(
    evidence: TrainabilityEvidence, qualification: CudaQualificationConfig
) -> None:
    """Validate exact mode counts and official-base initialization evidence."""
    if evidence.mode not in {"full_finetune", "transformer_finetune"}:
        raise QualificationError(f"unsupported qualification mode: {evidence.mode}")
    expected_trainable = getattr(
        qualification.trainability, f"{evidence.mode}_trainable_parameters"
    )
    expected_frozen = getattr(qualification.trainability, f"{evidence.mode}_frozen_parameters")
    if evidence.trainable_parameters != expected_trainable:
        raise QualificationError(
            f"trainable_parameters mismatch for {evidence.mode}: "
            f"expected {expected_trainable}, observed {evidence.trainable_parameters}"
        )
    if evidence.frozen_parameters != expected_frozen:
        raise QualificationError(
            f"frozen_parameters mismatch for {evidence.mode}: "
            f"expected {expected_frozen}, observed {evidence.frozen_parameters}"
        )
    if evidence.initialization_mode != "official":
        raise QualificationError("qualification initialization must be official")
    if evidence.fallback_checkpoint is not None:
        raise QualificationError("qualification forbids fallback checkpoints and scratch state")
    base = qualification.official_base
    for field in (
        "source_repository",
        "source_revision",
        "hub_model",
        "hub_revision",
        "weights_sha256",
    ):
        if getattr(evidence, field) != getattr(base, field):
            raise QualificationError(f"official-base identity mismatch: {field}")


def validate_environment_evidence(evidence: EnvironmentEvidence) -> None:
    """Validate complete CUDA runtime identity and absence of holdout access."""
    identity_fields = (
        "image_digest",
        "base_image_digest",
        "source_revision",
        "lock_digest",
        "torch_version",
        "cuda_runtime",
        "cuda_driver",
        "device_name",
        "device_capability",
    )
    for field in identity_fields:
        if not getattr(evidence, field).strip():
            raise QualificationError(f"environment evidence is missing {field}")
    if evidence.device_total_memory_bytes <= 0:
        raise QualificationError("environment evidence has invalid device memory")
    if evidence.device != "cuda" or evidence.precision != "fp32" or not evidence.pin_memory:
        raise QualificationError("environment evidence must prove CUDA FP32 with pinned memory")
    if evidence.holdout_unlocked:
        raise QualificationError("qualification must not unlock the holdout")
    paths = (*evidence.read_artifacts, *evidence.emitted_artifacts)
    if any("holdout" in path.casefold() for path in paths):
        raise QualificationError("qualification evidence references a holdout artifact")


def validate_artifact_parity(
    evidence: ArtifactParityEvidence, qualification: CudaQualificationConfig
) -> None:
    """Validate selected-checkpoint evaluation and safetensors reload parity."""
    if evidence.format != "safetensors":
        raise QualificationError("qualification export format must be safetensors")
    if evidence.holdout_unlocked:
        raise QualificationError("qualification evaluation must remain pre-holdout")
    selected = evidence.selected_checkpoint_digest
    if not selected:
        raise QualificationError("selected_checkpoint_digest must be non-empty")
    for field in ("evaluated_checkpoint_digest", "exported_checkpoint_digest"):
        if getattr(evidence, field) != selected:
            raise QualificationError(f"{field} does not match selected_checkpoint_digest")
    for field in (
        "config_digest",
        "data_digest",
        "source_revision",
        "lock_digest",
        "upstream_weights_sha256",
    ):
        if not getattr(evidence, field):
            raise QualificationError(f"{field} must be non-empty")
    _compare_value(
        evidence.native_outputs,
        evidence.reloaded_outputs,
        qualification.export,
        "outputs",
    )


def _flat_values(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def load_fp32_fixture(path: str | Path) -> FixtureBatch:
    """Materialize the immutable parity input recipe as CPU float32 tensors."""
    fixture_path = Path(path)
    try:
        payload = json.loads(fixture_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(f"cannot load FP32 fixture {fixture_path}: {exc}") from exc
    expected = {
        "schema_version",
        "generator",
        "context_shape",
        "context_start",
        "context_stop",
        "candle_target_shape",
        "candle_target_value",
        "leg_target",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise QualificationError("FP32 fixture has an invalid schema")
    if payload["schema_version"] != 1 or payload["generator"] != "linspace":
        raise QualificationError("FP32 fixture version or generator is unsupported")
    context_shape = payload["context_shape"]
    candle_shape = payload["candle_target_shape"]
    if context_shape != [1, 5, 512] or candle_shape != [1, 5, 4]:
        raise QualificationError("FP32 fixture shapes must match the production input contract")
    try:
        context = torch.linspace(
            float(payload["context_start"]),
            float(payload["context_stop"]),
            math.prod(context_shape),
            dtype=torch.float32,
        ).reshape(context_shape)
        candle_target = torch.full(
            candle_shape,
            float(payload["candle_target_value"]),
            dtype=torch.float32,
        )
        leg_target = torch.tensor(payload["leg_target"], dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise QualificationError(f"FP32 fixture values are invalid: {exc}") from exc
    if leg_target.shape != (1, 2):
        raise QualificationError("FP32 fixture leg target must have shape [1, 2]")
    tensors = (context, candle_target, leg_target)
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise QualificationError("FP32 fixture values must be finite")
    return {
        "context": context,
        "candle_target": candle_target,
        "leg_target": leg_target,
    }


def capture_fp32_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, torch.Tensor],
    target: TargetConfig,
    device: torch.device,
) -> NumericEvidence:
    """Capture the fixed-fixture forward, loss, gradient, and update evidence."""
    if device.type not in {"cpu", "cuda"}:
        raise QualificationError("FP32 parity capture supports only CPU and CUDA")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise QualificationError("FP32 parity capture requires float32 model parameters")
    required_batch = {"context", "candle_target", "leg_target"}
    if set(batch) != required_batch:
        raise QualificationError(f"FP32 parity batch must contain {sorted(required_batch)}")
    has_non_fp32 = any(
        tensor.is_floating_point() and tensor.dtype != torch.float32 for tensor in batch.values()
    )
    if has_non_fp32:
        raise QualificationError("FP32 parity capture requires float32 batch tensors")

    moved = {
        key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()
    }
    model.train()
    optimizer.zero_grad(set_to_none=True)
    synchronize_device(device)
    raw_output = model(moved["context"])
    if not isinstance(raw_output, dict) or set(raw_output) != {"candle", "leg"}:
        raise QualificationError("fixture model must return candle and leg outputs")
    output = cast(NextLegOutput, raw_output)
    losses = nextleg_loss(output, moved["candle_target"], moved["leg_target"], target)
    if not all(
        torch.isfinite(loss).item() for loss in (losses["candle"], losses["leg"], losses["total"])
    ):
        raise QualificationError("FP32 fixture produced non-finite loss")
    losses["total"].backward()  # type: ignore[no-untyped-call]

    gradients: dict[str, list[float]] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise QualificationError(f"trainable fixture parameter has no gradient: {name}")
            gradients[name] = _flat_values(parameter.grad)
    optimizer.step()
    synchronize_device(device)

    updated = {
        name: _flat_values(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
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
        "updated_parameters": updated,
    }


def run_batch_envelope_trials(
    fresh_process_worker: Callable[[int], BatchTrial],
    contract: BatchEnvelopeContract,
    *,
    gpu_total_bytes: int,
    host_total_bytes: int,
) -> BatchEnvelopeResult:
    """Request fresh-process trials until the first explicitly classified OOM."""
    trials: list[BatchTrial] = []
    batch_size = contract.initial_batch_size
    while True:
        trial = fresh_process_worker(batch_size)
        trials.append(trial)
        if trial.outcome in {"cuda_oom", "host_oom", "error"}:
            break
        batch_size *= contract.growth_factor
    return select_batch_envelope(
        trials,
        contract,
        gpu_total_bytes=gpu_total_bytes,
        host_total_bytes=host_total_bytes,
    )


PipelineRunner = Callable[[PipelineConfig], dict[str, Any]]
SubprocessRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError(f"qualification result is missing {field}")
    return value


def run_probe_and_export(
    pipeline: PipelineConfig,
    qualification: CudaQualificationConfig,
    *,
    run_id: str,
    artifact_root: Path,
    cuda_available: bool,
    probe_runner: PipelineRunner | None = None,
    release_runner: PipelineRunner | None = None,
) -> dict[str, Any]:
    """Execute the real 32-update CUDA probe and its validation/export gate."""
    if not cuda_available:
        return cpu_skip_record(
            cuda_available=False,
            reason="torch.cuda.is_available() is false",
        )
    configured = build_probe_config(
        pipeline,
        qualification,
        run_id=run_id,
        artifact_root=artifact_root,
    )
    if probe_runner is None or release_runner is None:
        from mantis_v2.pipeline import probe, validated_export

        probe_runner = probe if probe_runner is None else probe_runner
        release_runner = validated_export if release_runner is None else release_runner

    trained = probe_runner(configured)
    if trained.get("device") != "cuda":
        raise QualificationError("probe did not execute on explicit CUDA")
    last = _mapping(trained.get("last"), "last training record")
    if last.get("global_step") != qualification.probe.train_updates:
        raise QualificationError("probe did not complete exactly 32 train updates")
    metadata = _mapping(trained.get("metadata"), "training metadata")
    if metadata.get("precision") != "float32":
        raise QualificationError("probe did not execute in FP32")
    validate_trainability(
        TrainabilityEvidence(
            mode=configured.model.mode,
            trainable_parameters=cast(int, metadata.get("trainable_parameters")),
            frozen_parameters=cast(int, metadata.get("frozen_parameters")),
            initialization_mode=(
                "official"
                if metadata.get("initialization_mode") == "pinned_official_checkpoint"
                else str(metadata.get("initialization_mode"))
            ),
            source_repository=configured.model.source_repository,
            source_revision=configured.model.source_revision,
            hub_model=configured.model.hub_model,
            hub_revision=str(metadata.get("upstream_revision")),
            weights_sha256=str(metadata.get("upstream_weights_sha256")),
            fallback_checkpoint=None,
        ),
        qualification,
    )

    released = release_runner(configured)
    evaluation = _mapping(released.get("evaluation"), "evaluation")
    export = _mapping(released.get("export"), "export")
    checkpoint = _mapping(evaluation.get("checkpoint"), "evaluation checkpoint")
    validation_gate = _mapping(export.get("validation_gate"), "export validation gate")
    parity = _mapping(export.get("parity"), "export parity")
    if evaluation.get("split") != "validation" or evaluation.get("batches") != 1:
        raise QualificationError("probe must evaluate exactly one pre-holdout validation batch")
    if checkpoint.get("global_step") != 32 or validation_gate.get("global_step") != 32:
        raise QualificationError("evaluation/export checkpoint must bind global step 32")
    if checkpoint.get("sha256") != validation_gate.get("checkpoint_sha256"):
        raise QualificationError("evaluation/export selected checkpoint mismatch")
    if export.get("format") != "safetensors" or parity != {
        "atol": qualification.export.atol,
        "rtol": qualification.export.rtol,
        "verified": True,
    }:
        raise QualificationError("safetensors reload parity was not verified at pinned tolerances")

    return {
        "schema_version": 1,
        "status": "passed",
        "device": "cuda",
        "precision": "fp32",
        "run_id": run_id,
        "train_updates": 32,
        "validation_batches": 1,
        "pin_memory": qualification.probe.pin_memory,
        "holdout_unlocked": False,
        "subject_digest": qualification_subject_digest(configured),
        "checkpoint_sha256": checkpoint["sha256"],
        "train": trained,
        "evaluation": evaluation,
        "export": export,
    }


def _run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def run_subprocess_batch_trial(
    worker_command: Sequence[str],
    batch_size: int,
    *,
    runner: SubprocessRunner = _run_subprocess,
) -> BatchTrial:
    """Run one batch point in a fresh process and parse its canonical evidence."""
    if batch_size <= 0 or not worker_command:
        raise QualificationError("batch subprocess requires a command and positive batch size")
    command = [*worker_command, "--batch-size", str(batch_size)]
    completed = runner(command)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit {completed.returncode}"
        raise QualificationError(f"unclassified subprocess failure: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise QualificationError("batch subprocess did not emit canonical JSON") from exc
    if not isinstance(payload, dict):
        raise QualificationError("batch subprocess evidence must be an object")
    expected = {
        "schema_version",
        "batch_size",
        "outcome",
        "throughput_samples_per_second",
        "peak_gpu_bytes",
        "peak_host_bytes",
        "data_wait_seconds",
        "synchronized",
    }
    if set(payload) != expected or payload.get("schema_version") != 1:
        raise QualificationError("batch subprocess evidence has an invalid schema")
    if payload.get("batch_size") != batch_size:
        raise QualificationError("batch subprocess evidence has the wrong batch size")
    outcome = payload.get("outcome")
    if outcome not in {"clean", "cuda_oom", "host_oom"}:
        raise QualificationError("batch subprocess failure is unclassified")
    return BatchTrial(
        batch_size=batch_size,
        outcome=cast(TrialOutcome, outcome),
        throughput_samples_per_second=cast(float | None, payload["throughput_samples_per_second"]),
        peak_gpu_bytes=cast(int | None, payload["peak_gpu_bytes"]),
        peak_host_bytes=cast(int | None, payload["peak_host_bytes"]),
        data_wait_seconds=cast(float | None, payload["data_wait_seconds"]),
        synchronized=payload["synchronized"] is True,
        fresh_process=True,
    )
