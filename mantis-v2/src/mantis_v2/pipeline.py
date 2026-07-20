"""End-to-end train, evaluate, export, and smoke workflows."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Sized
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset, RandomSampler

from mantis_v2.checkpoint import load_checkpoint, save_checkpoint
from mantis_v2.config import PipelineConfig
from mantis_v2.data import (
    Anchor,
    NextLegDataset,
    build_anchors,
    contamination_report,
    load_streams,
)
from mantis_v2.model import MantisV2Adapter, NextLegModel, download_verified_weights, nextleg_loss
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


_PROVENANCE_IDENTITY_KEYS = (
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
    best = min(
        (float(record["validation"]["total"]) for record in history),
        default=float("inf"),
    )
    without_improvement = 0
    for record in reversed(history):
        if float(record["validation"]["total"]) <= best:
            break
        without_improvement += 1
    return best, without_improvement


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


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _run_epoch(
    model: NextLegModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    config: PipelineConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_steps: int,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "candle": 0.0, "leg": 0.0}
    batches = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            moved = _move(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            for key in ("total", "candle", "leg"):
                totals[key] += float(losses[key].detach().cpu())
            batches += 1
            if max_steps and batches >= max_steps:
                break
    if not batches:
        raise PipelineError("data loader produced no batches")
    return {key: value / batches for key, value in totals.items()}, batches


def train(config: PipelineConfig) -> dict[str, Any]:
    seed_everything(config.run.seed)
    root = artifact_root(config)
    _assert_run_writable(config, root)
    device = select_device(config.run)
    train_dataset, validation_dataset, contamination = _datasets(config)
    provenance = build_provenance(config, repository_root(), contamination)
    _write_json(root / "contamination.json", contamination)
    _write_json(root / "provenance.json", provenance.to_dict())
    model = _model(config, device)
    optimizer = _optimizer(model, config)
    checkpoint_path = root / "checkpoints" / "latest.pt"
    pending_checkpoint = checkpoint_path.with_name("pending.pt")
    start_epoch = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    if config.training.resume and pending_checkpoint.is_file():
        start_epoch, global_step = load_checkpoint(pending_checkpoint, model, optimizer, provenance)
        try:
            history = _load_history(root, start_epoch)
        except PipelineError:
            if checkpoint_path.is_file():
                start_epoch, global_step = load_checkpoint(
                    checkpoint_path, model, optimizer, provenance
                )
                history = _load_history(root, start_epoch)
            elif not (root / "metrics.json").exists():
                seed_everything(config.run.seed)
                model = _model(config, device)
                optimizer = _optimizer(model, config)
                start_epoch = 0
                global_step = 0
            else:
                raise
        else:
            pending_checkpoint.replace(checkpoint_path)
    elif config.training.resume and checkpoint_path.is_file():
        start_epoch, global_step = load_checkpoint(checkpoint_path, model, optimizer, provenance)
        history = _load_history(root, start_epoch)
    best_validation, epochs_without_improvement = _validation_state(history)
    stopped_early = bool(
        config.training.early_stopping_patience
        and epochs_without_improvement >= config.training.early_stopping_patience
    )
    for epoch in range(start_epoch, config.training.epochs):
        if stopped_early:
            break
        learning_rate = config.training.learning_rate_for_epoch(epoch)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        train_loader = _loader(train_dataset, config, shuffle=True, epoch=epoch)
        validation_loader = _loader(validation_dataset, config, shuffle=False, epoch=epoch)
        train_metrics, steps = _run_epoch(
            model,
            train_loader,
            config,
            device,
            optimizer,
            config.training.max_steps_per_epoch,
        )
        validation_metrics, _ = _run_epoch(
            model,
            validation_loader,
            config,
            device,
            None,
            config.training.validation_max_steps,
        )
        global_step += steps
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
        history.append(record)
        stopped_early = bool(
            config.training.early_stopping_patience
            and epochs_without_improvement >= config.training.early_stopping_patience
        )
        checkpoint_due = (epoch + 1) % config.training.checkpoint_every == 0 or (
            epoch + 1 == config.training.epochs or stopped_early
        )
        if improved:
            save_checkpoint(
                root / "checkpoints" / "best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                provenance,
            )
        if checkpoint_due:
            save_checkpoint(pending_checkpoint, model, optimizer, epoch, global_step, provenance)
        _write_json(root / "metrics.json", history)
        if checkpoint_due:
            pending_checkpoint.replace(checkpoint_path)
    _write_json(root / "provenance.json", provenance.to_dict())
    _write_json(root / "contamination.json", contamination)
    result = {
        "device": str(device),
        "train_anchors": len(train_dataset),
        "validation_anchors": len(validation_dataset),
        "checkpoint": str(checkpoint_path),
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_validation_loss": best_validation,
        "last": history[-1] if history else None,
    }
    _write_json(root / "train-result.json", result)
    return result


def _load_trained(
    config: PipelineConfig,
) -> _LoadedTrained:
    seed_everything(config.run.seed)
    device = select_device(config.run)
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
        next_epoch, global_step = load_checkpoint(snapshot_path, model, optimizer, provenance)
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
    )


def evaluate(config: PipelineConfig) -> dict[str, Any]:
    if config.evaluation.allow_holdout:
        raise PipelineError(
            "holdout evaluation is intentionally not automated; create an explicit release config"
        )
    return _evaluate_loaded(config, _load_trained(config))


def _evaluate_loaded(config: PipelineConfig, loaded: _LoadedTrained) -> dict[str, Any]:
    metrics, batches = _run_epoch(
        loaded.model,
        _loader(loaded.validation_dataset, config, shuffle=False),
        config,
        loaded.device,
        None,
        config.training.validation_max_steps,
    )
    if sha256_file(loaded.checkpoint_path) != loaded.checkpoint_sha256:
        raise PipelineError("best checkpoint changed during evaluation; run evaluate again")
    result = {
        "schema_version": 1,
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
    if not isinstance(metrics, dict) or any(
        not isinstance(metrics.get(key), int | float) or not math.isfinite(float(metrics[key]))
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

    metrics_path = root / "metrics.json"
    try:
        history = json.loads(metrics_path.read_text())
        selected = history[loaded.checkpoint_epoch]
        best_epoch = min(
            range(len(history)),
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
    weights_path = export_root / "model.safetensors"
    state = {
        key: value.detach().cpu().contiguous() for key, value in loaded.model.state_dict().items()
    }
    save_file(state, weights_path)
    loaded_state = load_file(weights_path)
    for key, tensor in state.items():
        if key not in loaded_state or not torch.equal(tensor, loaded_state[key]):
            raise PipelineError(f"export tensor parity failed for {key}")
    reloaded = _model(config, loaded.device)
    reloaded.load_state_dict(loaded_state, strict=True)
    reloaded.eval()
    fixture = loaded.validation_dataset[0]["context"].unsqueeze(0).to(loaded.device)
    with torch.inference_mode():
        native = loaded.model(fixture)
        restored = reloaded(fixture)
    for key in ("candle", "leg"):
        if not torch.allclose(
            native[key],
            restored[key],
            atol=config.export.verify_atol,
            rtol=config.export.verify_rtol,
        ):
            raise PipelineError(f"export parity failed for {key}")
    manifest = {
        "format": config.export.format,
        "weights": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "config": asdict(config),
        "provenance": loaded.provenance.to_dict(),
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
    """Run one real-data train and validation batch behind a strict scale guard."""
    if (
        config.data.root == "synthetic"
        or config.training.epochs != 1
        or config.training.max_steps_per_epoch != 1
        or config.training.validation_max_steps != 1
        or config.training.resume
    ):
        raise PipelineError(
            "probe requires real data, one epoch, one train step, one validation step, "
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
