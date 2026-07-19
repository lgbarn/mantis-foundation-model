"""End-to-end train, evaluate, export, and smoke workflows."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

from mantis_v2.checkpoint import load_checkpoint, save_checkpoint
from mantis_v2.config import PipelineConfig
from mantis_v2.data import NextLegDataset, build_anchors, load_streams
from mantis_v2.model import MantisV2Adapter, NextLegModel, download_verified_weights, nextleg_loss
from mantis_v2.provenance import Provenance, build_provenance
from mantis_v2.runtime import seed_everything, select_device


class PipelineError(RuntimeError):
    """Raised when a training stage cannot satisfy its contract."""


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


def _datasets(config: PipelineConfig) -> tuple[NextLegDataset, NextLegDataset]:
    streams = load_streams(config.data)
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
    )


def _loader(
    dataset: Dataset[dict[str, torch.Tensor]],
    config: PipelineConfig,
    *,
    shuffle: bool,
    epoch: int = 0,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(config.run.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        pin_memory=config.run.device == "cuda",
        generator=generator,
        persistent_workers=config.training.num_workers > 0,
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


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _run_epoch(
    model: NextLegModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    config: PipelineConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
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
            if optimizer is not None:
                losses["total"].backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            for key in ("total", "candle", "leg"):
                totals[key] += float(losses[key].detach().cpu())
            batches += 1
            if (
                config.training.max_steps_per_epoch
                and batches >= config.training.max_steps_per_epoch
            ):
                break
    if not batches:
        raise PipelineError("data loader produced no batches")
    return {key: value / batches for key, value in totals.items()}, batches


def train(config: PipelineConfig) -> dict[str, Any]:
    seed_everything(config.run.seed)
    root = artifact_root(config)
    _assert_run_writable(config, root)
    device = select_device(config.run)
    train_dataset, validation_dataset = _datasets(config)
    provenance = build_provenance(config, repository_root())
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
    for epoch in range(start_epoch, config.training.epochs):
        train_loader = _loader(train_dataset, config, shuffle=True, epoch=epoch)
        validation_loader = _loader(validation_dataset, config, shuffle=False, epoch=epoch)
        train_metrics, steps = _run_epoch(model, train_loader, config, device, optimizer)
        validation_metrics, _ = _run_epoch(model, validation_loader, config, device, None)
        global_step += steps
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        checkpoint_due = (epoch + 1) % config.training.checkpoint_every == 0 or (
            epoch + 1 == config.training.epochs
        )
        if checkpoint_due:
            save_checkpoint(pending_checkpoint, model, optimizer, epoch, global_step, provenance)
        _write_json(root / "metrics.json", history)
        if checkpoint_due:
            pending_checkpoint.replace(checkpoint_path)
    _write_json(root / "provenance.json", provenance.to_dict())
    result = {
        "device": str(device),
        "train_anchors": len(train_dataset),
        "validation_anchors": len(validation_dataset),
        "checkpoint": str(checkpoint_path),
        "epochs_completed": len(history),
        "last": history[-1] if history else None,
    }
    _write_json(root / "train-result.json", result)
    return result


def _load_trained(
    config: PipelineConfig,
) -> tuple[NextLegModel, Provenance, torch.device, NextLegDataset]:
    seed_everything(config.run.seed)
    device = select_device(config.run)
    train_dataset, validation_dataset = _datasets(config)
    del train_dataset
    provenance = build_provenance(config, repository_root())
    model = _model(config, device)
    optimizer = _optimizer(model, config)
    checkpoint_path = artifact_root(config) / "checkpoints" / "latest.pt"
    if not checkpoint_path.is_file():
        raise PipelineError(f"checkpoint not found: {checkpoint_path}")
    load_checkpoint(checkpoint_path, model, optimizer, provenance)
    return model, provenance, device, validation_dataset


def evaluate(config: PipelineConfig) -> dict[str, Any]:
    if config.evaluation.allow_holdout:
        raise PipelineError(
            "holdout evaluation is intentionally not automated; create an explicit release config"
        )
    model, provenance, device, validation_dataset = _load_trained(config)
    metrics, batches = _run_epoch(
        model,
        _loader(validation_dataset, config, shuffle=False),
        config,
        device,
        None,
    )
    result = {
        "split": "validation",
        "anchors": len(validation_dataset),
        "batches": batches,
        "metrics": metrics,
        "config_digest": provenance.config_digest,
        "dataset_digest": provenance.dataset_digest,
    }
    _write_json(artifact_root(config) / "evaluation.json", result)
    return result


def export(config: PipelineConfig) -> dict[str, Any]:
    model, provenance, device, validation_dataset = _load_trained(config)
    model.eval()
    export_root = artifact_root(config) / "export"
    export_root.mkdir(parents=True, exist_ok=True)
    weights_path = export_root / "model.safetensors"
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, weights_path)
    loaded_state = load_file(weights_path)
    reloaded = _model(config, device)
    reloaded.load_state_dict(loaded_state, strict=True)
    reloaded.eval()
    fixture = validation_dataset[0]["context"].unsqueeze(0).to(device)
    with torch.inference_mode():
        native = model(fixture)
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
        "config": asdict(config),
        "provenance": provenance.to_dict(),
        "parity": {
            "atol": config.export.verify_atol,
            "rtol": config.export.verify_rtol,
            "verified": True,
        },
    }
    _write_json(export_root / "manifest.json", manifest)
    return manifest


def smoke(config: PipelineConfig) -> dict[str, Any]:
    if config.data.root != "synthetic" or config.training.max_steps_per_epoch == 0:
        raise PipelineError("smoke requires synthetic data and a bounded max_steps_per_epoch")
    trained = train(config)
    evaluated = evaluate(config)
    exported = export(config)
    return {"train": trained, "evaluate": evaluated, "export": exported["parity"]}


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
    streams = load_streams(config.data)
    result: dict[str, int] = {}
    for split in ("train", "validation", "holdout"):
        result[split] = len(build_anchors(streams, config.data, config.target, split))
    return result
