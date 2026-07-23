"""End-to-end train, evaluate, export, and smoke workflows."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from collections.abc import Mapping, Sized
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset, RandomSampler

from mantis_v2.checkpoint import (
    checkpoint_adaptation_state,
    load_checkpoint,
    save_checkpoint,
)
from mantis_v2.config import PipelineConfig
from mantis_v2.data import (
    Anchor,
    NextLegDataset,
    build_anchors,
    contamination_report,
    load_streams,
)
from mantis_v2.instrumentation import (
    RunInstrumentation,
    collect_resource_metrics,
    parse_tensorboard_events,
    synchronize_device,
)
from mantis_v2.lora import adapter_state, lora_metadata, merged_lora_copy
from mantis_v2.model import (
    MantisV2Adapter,
    NextLegModel,
    download_verified_weights,
    nextleg_loss,
    nextleg_loss_per_sample,
)
from mantis_v2.precision import (
    autocast_context,
    validate_optimizer_state,
    validate_precision_device,
)
from mantis_v2.provenance import Provenance, build_provenance, sha256_file
from mantis_v2.runtime import seed_everything, select_device


class PipelineError(RuntimeError):
    """Raised when a training stage cannot satisfy its contract."""


@dataclass(frozen=True)
class _LoadedTrained:
    model: NextLegModel
    provenance: Provenance
    device: torch.device
    validation_dataset: NextLegDataset
    checkpoint_path: Path
    checkpoint_epoch: int
    global_step: int
    checkpoint_sha256: str
    adaptation: dict[str, object] | None


_PROVENANCE_IDENTITY_KEYS = (
    "precision",
    "config_digest",
    "dataset_digest",
    "source_digest",
    "lock_digest",
    "upstream_source_revision",
    "upstream_hub_revision",
    "upstream_weights_sha256",
    "contamination_digest",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def artifact_root(config: PipelineConfig) -> Path:
    path = config.run.artifact_root
    base = path if path.is_absolute() else repository_root() / path
    return base / config.run.name


def _assert_run_writable(config: PipelineConfig, root: Path) -> None:
    existing = any(
        (root / relative).exists()
        for relative in (
            "checkpoints/latest.pt",
            "checkpoints/best.pt",
            "metrics.json",
            "run-metadata.json",
            "provenance.json",
            "train-result.json",
            "evaluation.json",
            "export/model.safetensors",
            "export/manifest.json",
        )
    )
    checkpoint_exists = any(
        (root / "checkpoints" / name).is_file() for name in ("latest.pt", "pending.pt")
    )
    if config.training.resume and existing and not checkpoint_exists:
        raise PipelineError("run artifacts exist but no resumable checkpoint is present")
    if not config.training.resume and existing and not config.run.allow_overwrite:
        raise PipelineError(
            f"run artifacts already exist for {config.run.name}; choose a new run.name or enable "
            "run.allow_overwrite explicitly"
        )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _load_history(root: Path, start_epoch: int) -> list[dict[str, Any]]:
    if start_epoch == 0:
        return []
    path = root / "metrics.json"
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise PipelineError("resumable checkpoint requires valid metrics.json history") from exc
    if not isinstance(payload, list) or len(payload) < start_epoch:
        raise PipelineError("metrics.json does not cover the checkpointed epochs")
    history = payload[:start_epoch]
    if any(
        not isinstance(record, dict) or record.get("epoch") != epoch
        for epoch, record in enumerate(history)
    ):
        raise PipelineError("metrics.json epoch history is not contiguous")
    return history


def _datasets(
    config: PipelineConfig,
) -> tuple[NextLegDataset, NextLegDataset, dict[str, object]]:
    streams = load_streams(config.data)
    contamination = contamination_report(streams, config.data)
    train_anchors = build_anchors(streams, config.data, config.target, "train")
    validation_anchors = build_anchors(streams, config.data, config.target, "validation")
    if len(train_anchors) < config.target.minimum_train_anchors:
        raise PipelineError(
            f"only {len(train_anchors)} training anchors; require "
            f"{config.target.minimum_train_anchors}"
        )
    if len(validation_anchors) < config.target.minimum_validation_anchors:
        raise PipelineError(
            f"only {len(validation_anchors)} validation anchors; require "
            f"{config.target.minimum_validation_anchors}"
        )
    return (
        NextLegDataset(streams, train_anchors, config.data, config.model, config.target),
        NextLegDataset(streams, validation_anchors, config.data, config.model, config.target),
        contamination,
    )


def _loader(
    dataset: Dataset[dict[str, torch.Tensor]],
    config: PipelineConfig,
    *,
    shuffle: bool,
    epoch: int = 0,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(config.run.seed + epoch)
    sampler: RandomSampler | list[int] | None = None
    if shuffle and config.training.max_steps_per_epoch:
        sampler = RandomSampler(
            cast(Sized, dataset),
            replacement=True,
            num_samples=config.training.batch_size * config.training.max_steps_per_epoch,
            generator=generator,
        )
    elif not shuffle and config.training.validation_max_steps:
        if not isinstance(dataset, NextLegDataset):
            raise PipelineError("bounded validation requires a NextLegDataset")
        sampler = _stratified_validation_indices(
            dataset.anchors,
            config.training.batch_size * config.training.validation_max_steps,
        )
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.run.device == "cuda",
        generator=generator,
        persistent_workers=config.training.num_workers > 0,
    )


def _stratified_validation_indices(anchors: list[Anchor], num_samples: int) -> list[int]:
    """Select deterministic, evenly spaced validation anchors from every stream."""
    if num_samples >= len(anchors):
        return list(range(len(anchors)))
    by_stream: dict[int, list[int]] = {}
    for index, anchor in enumerate(anchors):
        by_stream.setdefault(anchor.stream_index, []).append(index)
    stream_ids = sorted(by_stream)
    if not stream_ids:
        return []
    if num_samples < len(stream_ids):
        raise PipelineError("validation sample cap must include every configured stream")
    base, extra = divmod(num_samples, len(stream_ids))
    selected: dict[int, list[int]] = {}
    for order, stream_index in enumerate(stream_ids):
        candidates = by_stream[stream_index]
        quota = base + (order < extra)
        if quota > len(candidates):
            raise PipelineError("validation sample cap exceeds anchors in a configured stream")
        selected[stream_index] = [
            candidates[min(((2 * index + 1) * len(candidates)) // (2 * quota), len(candidates) - 1)]
            for index in range(quota)
        ]
    result: list[int] = []
    for row in range(max(map(len, selected.values()), default=0)):
        for stream_index in stream_ids:
            if row < len(selected[stream_index]):
                result.append(selected[stream_index][row])
    return result


def _validation_state(history: list[dict[str, Any]]) -> tuple[float, int]:
    """Recover best validation loss and patience count from durable history."""
    losses = [float(record["validation"]["total"]) for record in history]
    if not losses:
        return float("inf"), 0
    best = min(losses)
    first_best = losses.index(best)
    return best, len(losses) - first_best - 1


def _emit_training_event(
    instrumentation: RunInstrumentation,
    record: dict[str, Any],
    telemetry: dict[str, Any],
) -> None:
    instrumentation.scalars(
        {
            "run/epoch": record["epoch"],
            "run/global_step": record["global_step"],
            "loss/train/total": record["train"]["total"],
            "loss/train/candle": record["train"]["candle"],
            "loss/train/leg": record["train"]["leg"],
            "loss/validation/total": record["validation"]["total"],
            "loss/validation/candle": record["validation"]["candle"],
            "loss/validation/leg": record["validation"]["leg"],
            "optimizer/learning_rate": record["learning_rate"],
            "optimization/gradient_norm": telemetry["gradient_norm"],
            "throughput/examples_per_second": telemetry["examples_per_second"],
            "throughput/updates_per_second": telemetry["updates_per_second"],
            "events/checkpoint_saved": telemetry["checkpoint_saved"],
            "events/best_checkpoint": telemetry["best_checkpoint"],
            "events/early_stop": telemetry["early_stop"],
            "io/data_wait_seconds": telemetry["data_wait_seconds"],
            "checkpoint/duration_seconds": telemetry["checkpoint_duration_seconds"],
            "checkpoint/size_bytes": telemetry["checkpoint_size_bytes"],
            "host/rss_bytes": telemetry["host_rss_bytes"],
            "filesystem/free_bytes": telemetry["filesystem_free_bytes"],
        },
        record["global_step"],
    )
    instrumentation.scalars(
        {
            tag: value
            for tag, value in {
                "cuda/allocated_bytes": telemetry["cuda_allocated_bytes"],
                "cuda/reserved_bytes": telemetry["cuda_reserved_bytes"],
                "cuda/utilization_percent": telemetry["cuda_utilization_percent"],
            }.items()
            if value is not None
        },
        record["global_step"],
    )


def _model(config: PipelineConfig, device: torch.device) -> NextLegModel:
    model = NextLegModel(config.model, config.target, len(config.data.feature_columns), device)
    return model.to(device)


def _optimizer(model: NextLegModel, config: PipelineConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise PipelineError("freeze policy left no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )


def _model_state_digest(model: NextLegModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _optimizer_identity(model: NextLegModel, phase: str) -> str:
    names = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    return hashlib.sha256(
        json.dumps({"phase": phase, "parameters": names}, sort_keys=True).encode()
    ).hexdigest()


def _adaptation_state(
    model: NextLegModel,
    phase: str,
    warm_start_updates: int,
    lora_updates: int,
    transition_parent: str | None,
) -> dict[str, object]:
    phase_updates = warm_start_updates if phase == "warm_start" else lora_updates
    return {
        "phase": phase,
        "phase_updates": phase_updates,
        "total_updates": warm_start_updates + lora_updates,
        "warm_start_updates": warm_start_updates,
        "lora_updates": lora_updates,
        "transition_parent": transition_parent,
        "optimizer_identity": _optimizer_identity(model, phase),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
    }


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _run_epoch(
    model: NextLegModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    config: PipelineConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_steps: int,
    *,
    completed_steps: int = 0,
    telemetry: dict[str, float] | None = None,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "candle": 0.0, "leg": 0.0}
    gradient_norm_total = 0.0
    examples = 0
    batches = 0
    data_wait_seconds = 0.0
    epoch_started = time.perf_counter()
    steps_per_epoch = config.training.max_steps_per_epoch or len(loader)
    context = torch.enable_grad() if training else torch.inference_mode()
    synchronize_device(device)
    with context:
        iterator = iter(loader)
        while True:
            wait_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            data_wait_seconds += time.perf_counter() - wait_started
            moved = _move(batch, device)
            if optimizer is not None:
                learning_rate = config.training.learning_rate_for_step(
                    completed_steps + batches + 1,
                    steps_per_epoch,
                )
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(config.training.precision, device):
                output = model(moved["context"])
                losses = nextleg_loss(
                    output,
                    moved["candle_target"],
                    moved["leg_target"],
                    config.target,
                )
            if not torch.isfinite(losses["total"]):
                phase = "training" if training else "validation"
                raise PipelineError(f"non-finite {phase} loss")
            if optimizer is not None:
                losses["total"].backward()  # type: ignore[no-untyped-call]
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                if not torch.isfinite(gradient_norm):
                    raise PipelineError(
                        "non-finite training gradient norm at global step "
                        f"{completed_steps + batches + 1}"
                    )
                gradient_norm_total += float(gradient_norm.detach().cpu())
                optimizer.step()
                validate_optimizer_state(optimizer)
            for key in ("total", "candle", "leg"):
                totals[key] += float(losses[key].detach().cpu())
            batches += 1
            examples += len(moved["context"])
            if max_steps and batches >= max_steps:
                break
    if not batches:
        raise PipelineError("data loader produced no batches")
    synchronize_device(device)
    duration_seconds = max(time.perf_counter() - epoch_started, 1e-12)
    if telemetry is not None:
        telemetry.update(
            {
                "data_wait_seconds": data_wait_seconds,
                "duration_seconds": duration_seconds,
                "examples_per_second": examples / duration_seconds,
                "updates_per_second": batches / duration_seconds,
                "gradient_norm": gradient_norm_total / batches if training else 0.0,
            }
        )
    return {key: value / batches for key, value in totals.items()}, batches


def _mean_metrics(losses: dict[str, torch.Tensor], selected: torch.Tensor) -> dict[str, float]:
    if not bool(selected.any()):
        raise PipelineError("evaluation metric group contains no samples")
    return {key: float(values[selected].mean().item()) for key, values in losses.items()}


def _stream_macro_metrics(
    by_stream: Mapping[str, Mapping[str, float]], members: list[str]
) -> dict[str, float]:
    if not members or any(member not in by_stream for member in members):
        raise PipelineError("evaluation macro group is incomplete")
    keys = ("total", "candle", "leg")
    return {
        key: sum(float(by_stream[member][key]) for member in members) / len(members) for key in keys
    }


def _run_evaluation_views(
    model: NextLegModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    config: PipelineConfig,
    device: torch.device,
    max_steps: int,
) -> tuple[dict[str, Any], int]:
    """Evaluate micro and unbiased stream/symbol/timeframe views on fixed rows."""
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {key: [] for key in ("total", "candle", "leg")}
    stream_indices: list[torch.Tensor] = []
    batches = 0
    with torch.inference_mode():
        for batch in loader:
            moved = _move(batch, device)
            with autocast_context(config.training.precision, device):
                output = model(moved["context"])
                losses = nextleg_loss_per_sample(
                    output,
                    moved["candle_target"],
                    moved["leg_target"],
                    config.target,
                )
            if not (
                torch.isfinite(losses["total"]).all()
                and torch.isfinite(losses["candle"]).all()
                and torch.isfinite(losses["leg"]).all()
            ):
                raise PipelineError("non-finite validation loss")
            collected["total"].append(losses["total"].detach().cpu())
            collected["candle"].append(losses["candle"].detach().cpu())
            collected["leg"].append(losses["leg"].detach().cpu())
            stream_indices.append(moved["stream_index"].detach().cpu())
            batches += 1
            if max_steps and batches >= max_steps:
                break
    if not batches:
        raise PipelineError("data loader produced no batches")
    loss_tensors = {key: torch.cat(values) for key, values in collected.items()}
    indices = torch.cat(stream_indices)
    dataset = loader.dataset
    if not isinstance(dataset, NextLegDataset):
        raise PipelineError("evaluation views require a NextLegDataset")
    names = [stream.name for stream in dataset.streams]
    present = sorted(set(int(value) for value in indices.tolist()))
    if any(index < 0 or index >= len(names) for index in present):
        raise PipelineError("evaluation stream index is out of range")
    by_stream = {names[index]: _mean_metrics(loss_tensors, indices == index) for index in present}
    grouped: dict[str, dict[str, list[int]]] = {"symbol": {}, "timeframe": {}}
    for index in present:
        try:
            symbol, timeframe = names[index].rsplit("_", 1)
        except ValueError as exc:
            raise PipelineError(f"stream name has no timeframe suffix: {names[index]}") from exc
        grouped["symbol"].setdefault(symbol, []).append(index)
        grouped["timeframe"].setdefault(timeframe, []).append(index)

    def group_metrics(members: list[int]) -> dict[str, float]:
        return _stream_macro_metrics(by_stream, [names[member] for member in members])

    macro = {
        key: sum(metrics[key] for metrics in by_stream.values()) / len(by_stream)
        for key in loss_tensors
    }
    return (
        {
            "micro": _mean_metrics(loss_tensors, torch.ones_like(indices, dtype=torch.bool)),
            "macro": macro,
            "by_stream": by_stream,
            "by_symbol": {
                name: group_metrics(members) for name, members in sorted(grouped["symbol"].items())
            },
            "by_timeframe": {
                name: group_metrics(members)
                for name, members in sorted(grouped["timeframe"].items())
            },
        },
        batches,
    )


def train(config: PipelineConfig, *, process_epoch_limit: int | None = None) -> dict[str, Any]:
    if process_epoch_limit is not None and process_epoch_limit <= 0:
        raise PipelineError("process_epoch_limit must be positive")
    seed_everything(config.run.seed)
    device = select_device(config.run)
    validate_precision_device(config.training.precision, device)
    root = artifact_root(config)
    _assert_run_writable(config, root)
    train_dataset, validation_dataset, contamination = _datasets(config)
    provenance = build_provenance(config, repository_root(), contamination)
    _write_json(root / "contamination.json", contamination)
    _write_json(root / "provenance.json", provenance.to_dict())
    model = _model(config, device)
    optimizer = _optimizer(model, config)
    bundled = config.adaptation is not None
    phase = "warm_start" if bundled else "standard"
    warm_start_updates = 0
    lora_updates = 0
    transition_parent: str | None = None
    checkpoint_path = root / "checkpoints" / "latest.pt"
    pending_checkpoint = checkpoint_path.with_name("pending.pt")
    start_epoch = 0
    global_step = 0
    resume_source: str | None = None
    history: list[dict[str, Any]] = []

    def restore(path: Path) -> tuple[int, int, torch.optim.Optimizer]:
        nonlocal phase, warm_start_updates, lora_updates, transition_parent
        state = checkpoint_adaptation_state(path, provenance)
        if bundled:
            if state is None:
                raise PipelineError("bundled checkpoint has no adaptation state")
            phase = cast(str, state["phase"])
            warm_start_updates = cast(int, state["warm_start_updates"])
            lora_updates = cast(int, state["lora_updates"])
            transition_parent = cast(str | None, state["transition_parent"])
            model.set_adaptation_phase(phase)
            if config.adaptation is None:
                raise PipelineError("bundled checkpoint has no adaptation config")
            if (
                warm_start_updates > config.adaptation.warm_start_updates
                or warm_start_updates + lora_updates > config.adaptation.total_updates
            ):
                raise PipelineError("checkpoint adaptation update budget exceeds config")
            reconstructed = _adaptation_state(
                model,
                phase,
                warm_start_updates,
                lora_updates,
                transition_parent,
            )
            if state != reconstructed:
                raise PipelineError("checkpoint adaptation identity mismatch")
        elif state is not None:
            raise PipelineError("standard checkpoint contains adaptation state")
        restored_optimizer = _optimizer(model, config)
        next_epoch, restored_step = load_checkpoint(
            path,
            model,
            restored_optimizer,
            provenance,
            state,
        )
        if bundled and restored_step != warm_start_updates + lora_updates:
            raise PipelineError("checkpoint total update count is inconsistent")
        return next_epoch, restored_step, restored_optimizer

    if config.training.resume and pending_checkpoint.is_file():
        resume_source = str(pending_checkpoint.resolve())
        start_epoch, global_step, optimizer = restore(pending_checkpoint)
        try:
            history = _load_history(root, start_epoch)
        except PipelineError:
            if checkpoint_path.is_file():
                resume_source = str(checkpoint_path.resolve())
                start_epoch, global_step, optimizer = restore(checkpoint_path)
                history = _load_history(root, start_epoch)
            elif not (root / "metrics.json").exists():
                seed_everything(config.run.seed)
                model = _model(config, device)
                optimizer = _optimizer(model, config)
                start_epoch = 0
                global_step = 0
                phase = "warm_start" if bundled else "standard"
                warm_start_updates = 0
                lora_updates = 0
                transition_parent = None
                resume_source = None
            else:
                raise
        else:
            pending_checkpoint.replace(checkpoint_path)
    elif config.training.resume and checkpoint_path.is_file():
        resume_source = str(checkpoint_path.resolve())
        start_epoch, global_step, optimizer = restore(checkpoint_path)
        history = _load_history(root, start_epoch)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_parameters = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    first_parameter = next(model.parameters())
    metadata: dict[str, object] = {
        "initialization_mode": (
            "random" if config.model.mode == "scratch" else "pinned_official_checkpoint"
        ),
        "upstream_revision": config.model.hub_revision,
        "upstream_weights_sha256": config.model.weights_sha256,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
        "seed": config.run.seed,
        "precision": config.training.precision,
        "parameter_precision": str(first_parameter.dtype).removeprefix("torch."),
        "optimizer_precision": "fp32",
        "resume_source": resume_source,
    }
    _write_json(root / "run-metadata.json", metadata)
    try:
        if not (root / "events").is_dir():
            raise FileNotFoundError
        prior_events = parse_tensorboard_events(root / "events")
        prior_steps = prior_events["scalars"].get("run/global_step", [])
        last_event_step = max((event["step"] for event in prior_steps), default=-1)
    except (FileNotFoundError, KeyError, OSError, RuntimeError):
        last_event_step = -1
    instrumentation = RunInstrumentation(root)
    instrumentation.text("run/metadata", metadata, global_step)
    instrumentation.text(
        "run/precision",
        {
            "compute": config.training.precision,
            "parameters": metadata["parameter_precision"],
            "optimizer": metadata["optimizer_precision"],
        },
        global_step,
    )
    telemetry_path = root / "instrumentation" / "telemetry.json"
    try:
        loaded_telemetry = json.loads(telemetry_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        loaded_telemetry = []
    telemetry_history: list[Any] = (
        loaded_telemetry[:start_epoch] if isinstance(loaded_telemetry, list) else []
    )
    telemetry_history.extend([None] * (start_epoch - len(telemetry_history)))
    for historical_record, historical_telemetry in zip(history, telemetry_history, strict=False):
        if (
            isinstance(historical_telemetry, dict)
            and historical_record["global_step"] > last_event_step
        ):
            _emit_training_event(instrumentation, historical_record, historical_telemetry)
    selection_history = history
    if bundled:
        selection_history = [
            record for record in history if record.get("adaptation", {}).get("phase") == phase
        ]
    best_validation, epochs_without_improvement = _validation_state(selection_history)
    stopped_early = bool(
        config.training.early_stopping_patience
        and epochs_without_improvement >= config.training.early_stopping_patience
    )
    end_epoch = config.training.epochs
    if process_epoch_limit is not None:
        end_epoch = min(end_epoch, start_epoch + process_epoch_limit)
    for epoch in range(start_epoch, end_epoch):
        if bundled and config.adaptation is not None:
            if global_step >= config.adaptation.total_updates:
                break
            if phase == "warm_start" and (
                warm_start_updates >= config.adaptation.warm_start_updates or stopped_early
            ):
                transition_parent = _model_state_digest(model)
                phase = "lora"
                model.set_adaptation_phase(phase)
                optimizer = _optimizer(model, config)
                best_validation = math.inf
                epochs_without_improvement = 0
                stopped_early = False
        if stopped_early:
            break
        train_loader = _loader(train_dataset, config, shuffle=True, epoch=epoch)
        validation_loader = _loader(validation_dataset, config, shuffle=False, epoch=epoch)
        max_steps = config.training.max_steps_per_epoch
        completed_steps = global_step
        if bundled and config.adaptation is not None:
            configured_steps = max_steps or len(train_loader)
            total_remaining = config.adaptation.total_updates - global_step
            phase_remaining = (
                config.adaptation.warm_start_updates - warm_start_updates
                if phase == "warm_start"
                else total_remaining
            )
            max_steps = min(configured_steps, total_remaining, phase_remaining)
            completed_steps = warm_start_updates if phase == "warm_start" else lora_updates
        train_telemetry: dict[str, float] = {}
        validation_telemetry: dict[str, float] = {}
        train_metrics, steps = _run_epoch(
            model,
            train_loader,
            config,
            device,
            optimizer,
            max_steps,
            completed_steps=completed_steps,
            telemetry=train_telemetry,
        )
        validation_metrics, _ = _run_epoch(
            model,
            validation_loader,
            config,
            device,
            None,
            config.training.validation_max_steps,
            telemetry=validation_telemetry,
        )
        global_step += steps
        if bundled:
            if phase == "warm_start":
                warm_start_updates += steps
            else:
                lora_updates += steps
        learning_rate = float(optimizer.param_groups[0]["lr"])
        improved = validation_metrics["total"] < best_validation
        if improved:
            best_validation = validation_metrics["total"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        current_adaptation = (
            _adaptation_state(
                model,
                phase,
                warm_start_updates,
                lora_updates,
                transition_parent,
            )
            if bundled
            else None
        )
        if current_adaptation is not None:
            record["adaptation"] = current_adaptation
        stopped_early = bool(
            config.training.early_stopping_patience
            and epochs_without_improvement >= config.training.early_stopping_patience
        )
        process_stop = epoch + 1 == end_epoch and end_epoch < config.training.epochs
        checkpoint_due = (
            (epoch + 1) % config.training.checkpoint_every == 0
            or epoch + 1 == config.training.epochs
            or stopped_early
            or process_stop
            or (
                bundled
                and config.adaptation is not None
                and global_step >= config.adaptation.total_updates
            )
            or (
                bundled
                and config.adaptation is not None
                and phase == "warm_start"
                and warm_start_updates >= config.adaptation.warm_start_updates
            )
        )
        checkpoint_started = time.perf_counter()
        synchronize_device(device)
        if improved:
            save_checkpoint(
                root / "checkpoints" / "best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                provenance,
                current_adaptation,
            )
        if checkpoint_due:
            save_checkpoint(
                pending_checkpoint,
                model,
                optimizer,
                epoch,
                global_step,
                provenance,
                current_adaptation,
            )
        synchronize_device(device)
        checkpoint_duration_seconds = time.perf_counter() - checkpoint_started
        checkpoint_size_path = (
            pending_checkpoint if checkpoint_due else root / "checkpoints" / "best.pt"
        )
        checkpoint_size_bytes = (
            checkpoint_size_path.stat().st_size if checkpoint_size_path.is_file() else 0
        )
        resource_metrics = collect_resource_metrics(root, device)
        telemetry_record = {
            "epoch": epoch,
            "global_step": global_step,
            "gradient_norm": train_telemetry["gradient_norm"],
            "examples_per_second": train_telemetry["examples_per_second"],
            "updates_per_second": train_telemetry["updates_per_second"],
            "data_wait_seconds": train_telemetry["data_wait_seconds"]
            + validation_telemetry["data_wait_seconds"],
            "checkpoint_duration_seconds": checkpoint_duration_seconds,
            "checkpoint_size_bytes": checkpoint_size_bytes,
            "checkpoint_saved": int(checkpoint_due),
            "best_checkpoint": int(improved),
            "early_stop": int(stopped_early),
            **resource_metrics,
        }
        history.append(record)
        telemetry_history.append(telemetry_record)
        _write_json(telemetry_path, telemetry_history)
        _write_json(root / "metrics.json", history)
        if checkpoint_due:
            pending_checkpoint.replace(checkpoint_path)
        _emit_training_event(instrumentation, record, telemetry_record)
    _write_json(root / "provenance.json", provenance.to_dict())
    _write_json(root / "contamination.json", contamination)
    final_adaptation = (
        _adaptation_state(
            model,
            phase,
            warm_start_updates,
            lora_updates,
            transition_parent,
        )
        if bundled
        else None
    )
    if final_adaptation is not None:
        metadata["adaptation"] = final_adaptation
        metadata["trainable_parameters"] = final_adaptation["trainable_parameters"]
        metadata["frozen_parameters"] = final_adaptation["frozen_parameters"]
        _write_json(root / "run-metadata.json", metadata)
    result = {
        "device": str(device),
        "train_anchors": len(train_dataset),
        "validation_anchors": len(validation_dataset),
        "checkpoint": str(checkpoint_path),
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_validation_loss": best_validation,
        "last": history[-1] if history else None,
        "telemetry": telemetry_history[-1] if telemetry_history else None,
        "metadata": metadata,
        "precision": config.training.precision,
        "adaptation": final_adaptation,
        "provenance": provenance.to_dict(),
    }
    _write_json(root / "train-result.json", result)
    instrumentation.close()
    return result


def _load_trained(
    config: PipelineConfig,
) -> _LoadedTrained:
    seed_everything(config.run.seed)
    device = select_device(config.run)
    validate_precision_device(config.training.precision, device)
    train_dataset, validation_dataset, contamination = _datasets(config)
    del train_dataset
    provenance = build_provenance(config, repository_root(), contamination)
    model = _model(config, device)
    optimizer = _optimizer(model, config)
    checkpoint_path = artifact_root(config) / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise PipelineError(f"checkpoint not found: {checkpoint_path}")
    digest = hashlib.sha256()
    snapshot_path: Path | None = None
    try:
        with (
            checkpoint_path.open("rb") as source,
            tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".best-snapshot-",
                suffix=".pt",
                dir=checkpoint_path.parent,
                delete=False,
            ) as snapshot,
        ):
            snapshot_path = Path(snapshot.name)
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
                snapshot.write(block)
        adaptation = checkpoint_adaptation_state(snapshot_path, provenance)
        if config.adaptation is not None:
            if adaptation is None or adaptation.get("phase") != "lora":
                raise PipelineError("validated export requires a completed LoRA-phase checkpoint")
            model.set_adaptation_phase("lora")
            optimizer = _optimizer(model, config)
        elif adaptation is not None:
            raise PipelineError("standard checkpoint contains adaptation state")
        next_epoch, global_step = load_checkpoint(
            snapshot_path,
            model,
            optimizer,
            provenance,
            adaptation,
        )
    finally:
        if snapshot_path is not None:
            snapshot_path.unlink(missing_ok=True)
    return _LoadedTrained(
        model=model,
        provenance=provenance,
        device=device,
        validation_dataset=validation_dataset,
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=next_epoch - 1,
        global_step=global_step,
        checkpoint_sha256=digest.hexdigest(),
        adaptation=adaptation,
    )


def evaluate(config: PipelineConfig) -> dict[str, Any]:
    if config.evaluation.allow_holdout:
        raise PipelineError(
            "holdout evaluation is intentionally not automated; create an explicit release config"
        )
    return _evaluate_loaded(config, _load_trained(config))


def _evaluate_loaded(config: PipelineConfig, loaded: _LoadedTrained) -> dict[str, Any]:
    metrics, batches = _run_evaluation_views(
        loaded.model,
        _loader(loaded.validation_dataset, config, shuffle=False),
        config,
        loaded.device,
        config.training.validation_max_steps,
    )
    if sha256_file(loaded.checkpoint_path) != loaded.checkpoint_sha256:
        raise PipelineError("best checkpoint changed during evaluation; run evaluate again")
    result = {
        "schema_version": 1,
        "precision": config.training.precision,
        "created_at": datetime.now(UTC).isoformat(),
        "passed": True,
        "split": "validation",
        "anchors": len(loaded.validation_dataset),
        "batches": batches,
        "metrics": metrics,
        "checkpoint": {
            "path": str(loaded.checkpoint_path),
            "sha256": loaded.checkpoint_sha256,
            "epoch": loaded.checkpoint_epoch,
            "global_step": loaded.global_step,
        },
        "provenance": loaded.provenance.to_dict(),
        "config_digest": loaded.provenance.config_digest,
        "dataset_digest": loaded.provenance.dataset_digest,
        "adaptation": loaded.adaptation,
    }
    _write_json(artifact_root(config) / "evaluation.json", result)
    return result


def _validated_evaluation(
    config: PipelineConfig,
    loaded: _LoadedTrained,
) -> dict[str, Any]:
    root = artifact_root(config)
    evaluation_path = root / "evaluation.json"
    try:
        evaluation = json.loads(evaluation_path.read_text())
    except FileNotFoundError as exc:
        raise PipelineError("run evaluate before export: evaluation.json is missing") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError("run evaluate before export: evaluation.json is invalid") from exc
    if not isinstance(evaluation, dict) or evaluation.get("schema_version") != 1:
        raise PipelineError("run evaluate before export: unsupported evaluation schema")
    if not isinstance(evaluation.get("created_at"), str) or not evaluation["created_at"]:
        raise PipelineError("run evaluate before export: completion time is missing")
    if evaluation.get("passed") is not True or evaluation.get("split") != "validation":
        raise PipelineError("run evaluate before export: validation did not pass")
    if evaluation.get("anchors") != len(loaded.validation_dataset):
        raise PipelineError("run evaluate before export: validation corpus changed")
    if not isinstance(evaluation.get("batches"), int) or evaluation["batches"] <= 0:
        raise PipelineError("run evaluate before export: evaluation has no completed batches")
    metrics = evaluation.get("metrics")
    micro = metrics.get("micro") if isinstance(metrics, dict) else None
    if not isinstance(micro, dict) or any(
        not isinstance(micro.get(key), int | float) or not math.isfinite(float(micro[key]))
        for key in ("total", "candle", "leg")
    ):
        raise PipelineError("run evaluate before export: evaluation metrics are incomplete")

    checkpoint = evaluation.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise PipelineError("run evaluate before export: checkpoint identity is missing")
    if checkpoint.get("sha256") != loaded.checkpoint_sha256:
        raise PipelineError("best checkpoint changed after evaluation; run evaluate before export")
    if (
        checkpoint.get("epoch") != loaded.checkpoint_epoch
        or checkpoint.get("global_step") != loaded.global_step
    ):
        raise PipelineError(
            "best checkpoint metadata changed after evaluation; run evaluate before export"
        )

    recorded_provenance = evaluation.get("provenance")
    expected_provenance = loaded.provenance.to_dict()
    for key in _PROVENANCE_IDENTITY_KEYS:
        if (
            not isinstance(recorded_provenance, dict)
            or recorded_provenance.get(key) != (expected_provenance[key])
        ):
            raise PipelineError(
                f"evaluation provenance mismatch: {key}; run evaluate before export"
            )

    if evaluation.get("adaptation") != loaded.adaptation:
        raise PipelineError("evaluation adaptation state mismatch; run evaluate again")

    metrics_path = root / "metrics.json"
    try:
        history = json.loads(metrics_path.read_text())
        selected = history[loaded.checkpoint_epoch]
        eligible_epochs = [
            epoch
            for epoch, record in enumerate(history)
            if config.adaptation is None or record.get("adaptation", {}).get("phase") == "lora"
        ]
        best_epoch = min(
            eligible_epochs,
            key=lambda epoch: float(history[epoch]["validation"]["total"]),
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise PipelineError("run evaluate before export: metrics history is invalid") from exc
    if (
        not isinstance(selected, dict)
        or selected.get("epoch") != loaded.checkpoint_epoch
        or selected.get("global_step") != loaded.global_step
        or best_epoch != loaded.checkpoint_epoch
    ):
        raise PipelineError("best checkpoint does not match validation selection history")
    return evaluation


def export(config: PipelineConfig) -> dict[str, Any]:
    return _export_loaded(config, _load_trained(config))


def _export_loaded(config: PipelineConfig, loaded: _LoadedTrained) -> dict[str, Any]:
    evaluation = _validated_evaluation(config, loaded)
    loaded.model.eval()
    export_root = artifact_root(config) / "export"
    export_root.mkdir(parents=True, exist_ok=True)
    bundled_evaluation_path = export_root / "evaluation.json"
    _write_json(bundled_evaluation_path, evaluation)
    lora: dict[str, Any] | None = None
    adapter_reloaded: NextLegModel | None = None
    export_model: NextLegModel = loaded.model
    if config.model.mode.startswith("lora_"):
        adapter_path = export_root / "adapter.safetensors"
        saved_adapter = adapter_state(loaded.model)
        save_file(saved_adapter, adapter_path)
        metadata = dict(lora_metadata(loaded.model))
        adapter_reloaded = _model(config, loaded.device)
        incompatible_adapter = adapter_reloaded.load_state_dict(
            load_file(adapter_path), strict=False
        )
        if incompatible_adapter.unexpected_keys or not set(saved_adapter).isdisjoint(
            incompatible_adapter.missing_keys
        ):
            raise PipelineError("LoRA adapter reload state is incompatible")
        adapter_reloaded.eval()
        export_model = cast(NextLegModel, merged_lora_copy(loaded.model)).to(loaded.device).eval()
        lora = {
            **metadata,
            "base_weights_sha256": loaded.provenance.upstream_weights_sha256,
            "adapter": str(adapter_path),
            "adapter_sha256": sha256_file(adapter_path),
            "adapter_tensors": sorted(saved_adapter),
            "includes_trainable_task_heads": True,
            "merged": True,
        }
    weights_path = export_root / "model.safetensors"
    state = {
        key: value.detach().cpu().contiguous() for key, value in export_model.state_dict().items()
    }
    save_file(state, weights_path)
    loaded_state = load_file(weights_path)
    for key, tensor in state.items():
        if key not in loaded_state or not torch.equal(tensor, loaded_state[key]):
            raise PipelineError(f"export tensor parity failed for {key}")
    reloaded = _model(config, loaded.device)
    if config.model.mode.startswith("lora_"):
        reloaded = cast(NextLegModel, merged_lora_copy(reloaded)).to(loaded.device)
    reloaded.load_state_dict(loaded_state, strict=True)
    reloaded.eval()
    fixture = loaded.validation_dataset[0]["context"].unsqueeze(0).to(loaded.device)
    with torch.inference_mode():
        native = loaded.model(fixture)
        restored = reloaded(fixture)
        adapter_restored = adapter_reloaded(fixture) if adapter_reloaded is not None else None
    for key in ("candle", "leg"):
        if not torch.allclose(
            native[key],
            restored[key],
            atol=config.export.verify_atol,
            rtol=config.export.verify_rtol,
        ):
            raise PipelineError(f"export parity failed for {key}")
        if adapter_restored is not None and not torch.allclose(
            native[key],
            adapter_restored[key],
            atol=config.export.verify_atol,
            rtol=config.export.verify_rtol,
        ):
            raise PipelineError(f"LoRA adapter reload parity failed for {key}")
    manifest: dict[str, Any] = {
        "format": config.export.format,
        "precision": config.training.precision,
        "export_role": "diagnostic_candidate",
        "weights": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "config": asdict(config),
        "provenance": loaded.provenance.to_dict(),
        "adaptation": loaded.adaptation,
        "validation_gate": {
            "verified": True,
            "evaluation": str(bundled_evaluation_path),
            "evaluation_sha256": sha256_file(bundled_evaluation_path),
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "checkpoint_epoch": loaded.checkpoint_epoch,
            "global_step": loaded.global_step,
        },
        "parity": {
            "atol": config.export.verify_atol,
            "rtol": config.export.verify_rtol,
            "verified": True,
        },
    }
    if lora is not None:
        manifest["lora"] = lora
        manifest["parity"]["lora_merge_verified"] = True
        manifest["parity"]["lora_adapter_reload_verified"] = True
    _write_json(export_root / "manifest.json", manifest)
    return manifest


def validated_export(config: PipelineConfig) -> dict[str, Any]:
    """Evaluate the selected checkpoint and export it only after the gate passes."""
    if config.evaluation.allow_holdout:
        raise PipelineError(
            "holdout evaluation is intentionally not automated; create an explicit release config"
        )
    loaded = _load_trained(config)
    evaluation = _evaluate_loaded(config, loaded)
    manifest = _export_loaded(config, loaded)
    return {"evaluation": evaluation, "export": manifest}


def smoke(config: PipelineConfig) -> dict[str, Any]:
    if config.data.root != "synthetic" or config.training.max_steps_per_epoch == 0:
        raise PipelineError("smoke requires synthetic data and a bounded max_steps_per_epoch")
    trained = train(config)
    released = validated_export(config)
    return {
        "train": trained,
        "evaluate": released["evaluation"],
        "export": released["export"]["parity"],
    }


def probe(config: PipelineConfig) -> dict[str, Any]:
    """Run a bounded real-data optimizer stress test and one validation batch."""
    if (
        config.data.root == "synthetic"
        or config.training.epochs != 1
        or config.training.max_steps_per_epoch != 32
        or config.training.validation_max_steps != 1
        or config.training.resume
    ):
        raise PipelineError(
            "probe requires real data, one epoch, 32 train steps, one validation step, "
            "and resume=false"
        )
    return train(config)


def verify_upstream(config: PipelineConfig) -> dict[str, Any]:
    """Verify the real pinned checkpoint and upstream inference contract on CPU."""
    weights = download_verified_weights(config.model)
    adapter = MantisV2Adapter(config.model, torch.device("cpu"))
    fixture = torch.zeros(
        1,
        len(config.data.feature_columns),
        config.model.input_length,
        dtype=torch.float32,
    )
    adapter.eval()
    with torch.inference_mode():
        embedding = adapter(fixture)
    return {
        "hub_model": config.model.hub_model,
        "hub_revision": config.model.hub_revision,
        "weights": str(weights),
        "weights_sha256": config.model.weights_sha256,
        "embedding_shape": list(embedding.shape),
        "verified": True,
    }


def anchor_counts(config: PipelineConfig) -> dict[str, int]:
    return data_audit(config)["anchors"]  # type: ignore[return-value]


def data_audit(config: PipelineConfig) -> dict[str, object]:
    """Validate eligibility and report every configured discontinuity before training."""
    streams = load_streams(config.data)
    result: dict[str, int] = {}
    for split in ("train", "validation", "holdout"):
        result[split] = len(build_anchors(streams, config.data, config.target, split))
    return {"anchors": result, "contamination": contamination_report(streams, config.data)}
