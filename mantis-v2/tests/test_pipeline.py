from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from mantis_v2 import pipeline as pipeline_module
from mantis_v2.config import load_config
from mantis_v2.pipeline import PipelineError, _assert_run_writable, _loader, train
from torch.utils.data import Dataset

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

    protected = replace(
        config,
        run=replace(config.run, allow_overwrite=False),
        training=replace(config.training, resume=False),
    )
    with pytest.raises(PipelineError, match="run artifacts already exist"):
        _assert_run_writable(protected, checkpoint.parents[1])


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
