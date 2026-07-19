from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from mantis_v2 import pipeline as pipeline_module
from mantis_v2.config import load_config
from mantis_v2.data import Anchor
from mantis_v2.pipeline import (
    PipelineError,
    _assert_run_writable,
    _loader,
    _stratified_validation_indices,
    _validation_state,
    probe,
    train,
)
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]


class IndexDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 32

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"index": torch.tensor(index)}


def test_loader_order_is_deterministic_per_epoch() -> None:
    base = load_config(ROOT / "configs" / "smoke.toml")
    config = replace(base, training=replace(base.training, batch_size=8))
    first = next(iter(_loader(IndexDataset(), config, shuffle=True, epoch=4)))["index"]
    resumed = next(iter(_loader(IndexDataset(), config, shuffle=True, epoch=4)))["index"]
    next_epoch = next(iter(_loader(IndexDataset(), config, shuffle=True, epoch=5)))["index"]
    torch.testing.assert_close(first, resumed)
    assert not torch.equal(first, next_epoch)


def test_bounded_training_samples_with_replacement() -> None:
    base = load_config(ROOT / "configs" / "smoke.toml")
    config = replace(
        base,
        training=replace(base.training, batch_size=8, max_steps_per_epoch=2),
    )
    loader = _loader(IndexDataset(), config, shuffle=True, epoch=4)
    sampled = torch.cat([batch["index"] for batch in loader])
    assert len(sampled) == 16
    assert loader.sampler.replacement is True


def test_validation_sampling_is_deterministic_and_stream_stratified() -> None:
    anchors = [
        Anchor(stream_index=stream, confirmation=index, first_leg=1, second_leg=1)
        for stream in range(4)
        for index in range(100)
    ]
    first = _stratified_validation_indices(anchors, 40)
    second = _stratified_validation_indices(anchors, 40)
    assert first == second
    assert len(first) == len(set(first)) == 40
    assert {anchors[index].stream_index for index in first} == {0, 1, 2, 3}
    assert [anchors[index].stream_index for index in first[:4]] == [0, 1, 2, 3]


def test_validation_state_recovers_resume_patience() -> None:
    history = [{"validation": {"total": loss}} for loss in (5.0, 4.0, 4.2, 4.3, 4.4)]
    assert _validation_state(history) == (4.0, 3)


def test_non_finite_validation_loss_reports_validation_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "smoke.toml")
    loader = DataLoader(
        [
            {
                "context": torch.zeros(5, config.model.input_length),
                "candle_target": torch.zeros(5, len(config.target.horizons)),
                "leg_target": torch.zeros(2),
            }
        ],
        batch_size=1,
    )

    def non_finite_loss(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        del args, kwargs
        value = torch.tensor(float("nan"))
        return {"total": value, "candle": value, "leg": value}

    monkeypatch.setattr(pipeline_module, "nextleg_loss", non_finite_loss)
    with pytest.raises(PipelineError, match="non-finite validation loss"):
        pipeline_module._run_epoch(
            torch.nn.Identity(), loader, config, torch.device("cpu"), None, 1
        )


def test_terminal_epoch_is_checkpointed_and_collision_is_rejected(tmp_path: Path) -> None:
    base = load_config(ROOT / "configs" / "smoke.toml")
    config = replace(
        base,
        run=replace(base.run, artifact_root=tmp_path, allow_overwrite=True),
        training=replace(base.training, max_steps_per_epoch=1, checkpoint_every=5),
    )
    result = train(config)
    checkpoint = Path(result["checkpoint"])
    assert checkpoint.is_file()
    assert result["epochs_completed"] == 1
    assert (checkpoint.parent / "best.pt").is_file()

    protected = replace(
        config,
        run=replace(config.run, allow_overwrite=False),
        training=replace(config.training, resume=False),
    )
    with pytest.raises(PipelineError, match="run artifacts already exist"):
        _assert_run_writable(protected, checkpoint.parents[1])


@pytest.mark.parametrize(
    "training_change",
    [
        {"epochs": 2},
        {"max_steps_per_epoch": 2},
        {"validation_max_steps": 2},
        {"resume": True},
    ],
)
def test_probe_rejects_each_unbounded_setting(training_change: dict[str, Any]) -> None:
    base = load_config(ROOT / "configs" / "nextleg-mps-probe.toml")
    config = replace(base, training=replace(base.training, **training_change))
    with pytest.raises(PipelineError, match="probe requires real data"):
        probe(config)


def test_probe_rejects_synthetic_data_independently() -> None:
    base = load_config(ROOT / "configs" / "smoke.toml")
    config = replace(
        base,
        training=replace(
            base.training,
            max_steps_per_epoch=1,
            validation_max_steps=1,
        ),
    )
    with pytest.raises(PipelineError, match="probe requires real data"):
        probe(config)


def test_best_checkpoint_and_early_stopping_follow_validation_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load_config(ROOT / "configs" / "smoke.toml")
    config = replace(
        base,
        run=replace(base.run, name="early-stop", artifact_root=tmp_path),
        training=replace(
            base.training,
            epochs=6,
            checkpoint_every=1,
            early_stopping_patience=2,
        ),
    )
    validation_losses = iter((5.0, 4.0, 4.5, 4.6))

    def controlled_epoch(
        model: Any,
        loader: Any,
        pipeline_config: Any,
        device: Any,
        optimizer: torch.optim.Optimizer | None,
        max_steps: int,
    ) -> tuple[dict[str, float], int]:
        del model, loader, pipeline_config, device, max_steps
        total = 1.0 if optimizer is not None else next(validation_losses)
        return {"total": total, "candle": total / 2, "leg": total / 2}, 1

    monkeypatch.setattr(pipeline_module, "_run_epoch", controlled_epoch)
    result = train(config)
    assert result["stopped_early"] is True
    assert result["epochs_completed"] == 4
    assert result["best_validation_loss"] == 4.0
    best = torch.load(
        tmp_path / "early-stop" / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    latest = torch.load(
        tmp_path / "early-stop" / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert best["epoch"] == 1
    assert latest["epoch"] == 3


def test_resumed_training_restores_early_stopping_patience(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load_config(ROOT / "configs" / "smoke.toml")
    config = replace(
        base,
        run=replace(
            base.run,
            name="resume-patience",
            artifact_root=tmp_path,
            allow_overwrite=False,
        ),
        training=replace(
            base.training,
            epochs=6,
            checkpoint_every=1,
            resume=True,
            early_stopping_patience=2,
        ),
    )
    validation_losses = iter((5.0, 4.0, 4.5, 4.6))

    def controlled_epoch(
        model: Any,
        loader: Any,
        pipeline_config: Any,
        device: Any,
        optimizer: torch.optim.Optimizer | None,
        max_steps: int,
    ) -> tuple[dict[str, float], int]:
        del model, loader, pipeline_config, device, max_steps
        total = 1.0 if optimizer is not None else next(validation_losses)
        return {"total": total, "candle": total / 2, "leg": total / 2}, 1

    original_write_json = pipeline_module._write_json
    interrupted = False

    def interrupt_with_patience(path: Path, payload: Any) -> None:
        nonlocal interrupted
        original_write_json(path, payload)
        if path.name == "metrics.json" and isinstance(payload, list) and len(payload) == 3:
            interrupted = True
            raise RuntimeError("simulated patience interruption")

    monkeypatch.setattr(pipeline_module, "_run_epoch", controlled_epoch)
    monkeypatch.setattr(pipeline_module, "_write_json", interrupt_with_patience)
    with pytest.raises(RuntimeError, match="simulated patience interruption"):
        train(config)
    assert interrupted

    monkeypatch.setattr(pipeline_module, "_write_json", original_write_json)
    result = train(config)
    assert result["stopped_early"] is True
    assert result["epochs_completed"] == 4
    assert result["best_validation_loss"] == 4.0


@pytest.mark.parametrize("interrupt_length", [1, 2])
def test_interrupted_resume_matches_uninterrupted_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_length: int,
) -> None:
    base = load_config(ROOT / "configs" / "smoke.toml")
    common_training = replace(
        base.training,
        epochs=3,
        max_steps_per_epoch=1,
        checkpoint_every=1,
        resume=True,
    )
    interrupted = replace(
        base,
        run=replace(
            base.run,
            name=f"interrupted-{interrupt_length}",
            artifact_root=tmp_path,
            allow_overwrite=False,
        ),
        training=common_training,
    )
    original_write_json = pipeline_module._write_json
    did_interrupt = False

    def interrupt_after_first_metrics(path: Path, payload: Any) -> None:
        nonlocal did_interrupt
        original_write_json(path, payload)
        if (
            path.name == "metrics.json"
            and isinstance(payload, list)
            and len(payload) == interrupt_length
        ):
            did_interrupt = True
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(pipeline_module, "_write_json", interrupt_after_first_metrics)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(interrupted)
    assert did_interrupt
    monkeypatch.setattr(pipeline_module, "_write_json", original_write_json)
    resumed_result = train(interrupted)

    uninterrupted = replace(
        interrupted,
        run=replace(interrupted.run, name=f"uninterrupted-{interrupt_length}"),
    )
    uninterrupted_result = train(uninterrupted)
    assert resumed_result["epochs_completed"] == uninterrupted_result["epochs_completed"] == 3

    resumed_checkpoint = torch.load(
        resumed_result["checkpoint"], map_location="cpu", weights_only=True
    )
    uninterrupted_checkpoint = torch.load(
        uninterrupted_result["checkpoint"], map_location="cpu", weights_only=True
    )
    assert resumed_checkpoint["global_step"] == uninterrupted_checkpoint["global_step"] == 3
    for key, value in resumed_checkpoint["model"].items():
        torch.testing.assert_close(value, uninterrupted_checkpoint["model"][key])

    resumed_metrics = json.loads(
        (tmp_path / f"interrupted-{interrupt_length}" / "metrics.json").read_text()
    )
    uninterrupted_metrics = json.loads(
        (tmp_path / f"uninterrupted-{interrupt_length}" / "metrics.json").read_text()
    )
    assert resumed_metrics == uninterrupted_metrics
