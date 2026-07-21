from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from mantis_v2 import instrumentation as instrumentation_module
from mantis_v2 import pipeline as pipeline_module
from mantis_v2.config import load_config
from mantis_v2.data import Anchor
from mantis_v2.pipeline import (
    PipelineError,
    _assert_run_writable,
    _loader,
    _stratified_validation_indices,
    _validation_state,
    evaluate,
    export,
    probe,
    train,
    validated_export,
)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]


class IndexDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 32

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"index": torch.tensor(index)}


class NonFiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        del ctx
        return value.clone()

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> torch.Tensor:
        del ctx
        return torch.full_like(gradient, float("nan"))


class NonFiniteGradientModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        del context
        value = NonFiniteBackward.apply(self.weight)
        return {"candle": value, "leg": value}


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


def test_non_finite_training_gradient_fails_before_optimizer_step(
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
    model = NonFiniteGradientModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    stepped = False
    original_step = optimizer.step

    def track_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal stepped
        stepped = True
        return original_step(*args, **kwargs)

    def finite_loss(
        output: dict[str, torch.Tensor], *args: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        del args, kwargs
        return {
            "total": output["candle"],
            "candle": output["candle"],
            "leg": output["leg"],
        }

    monkeypatch.setattr(optimizer, "step", track_step)
    monkeypatch.setattr(pipeline_module, "nextleg_loss", finite_loss)
    with pytest.raises(PipelineError, match="non-finite training gradient norm"):
        pipeline_module._run_epoch(
            model,
            loader,
            config,
            torch.device("cpu"),
            optimizer,
            1,
        )
    assert stepped is False
    assert torch.isfinite(model.weight)


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


def test_bounded_training_emits_tensorboard_loss_and_progress_events(tmp_path: Path) -> None:
    base = load_config(ROOT / "configs" / "smoke.toml")
    config = replace(
        base,
        run=replace(base.run, name="instrumented", artifact_root=tmp_path),
        training=replace(
            base.training,
            max_steps_per_epoch=1,
            validation_max_steps=1,
            checkpoint_every=1,
        ),
    )

    result = train(config)
    history = json.loads((tmp_path / "instrumented" / "metrics.json").read_text())
    events = EventAccumulator(str(tmp_path / "instrumented" / "events")).Reload()
    scalar_tags = set(events.Tags()["scalars"])

    assert {
        "run/epoch",
        "run/global_step",
        "loss/train/total",
        "loss/train/candle",
        "loss/train/leg",
        "loss/validation/total",
        "loss/validation/candle",
        "loss/validation/leg",
        "optimizer/learning_rate",
        "optimization/gradient_norm",
        "throughput/examples_per_second",
        "throughput/updates_per_second",
        "events/checkpoint_saved",
        "events/best_checkpoint",
        "events/early_stop",
    } <= scalar_tags
    assert events.Scalars("run/global_step")[-1].value == result["last"]["global_step"]
    assert events.Scalars("loss/train/total")[-1].value == pytest.approx(
        history[-1]["train"]["total"]
    )
    assert events.Scalars("loss/validation/total")[-1].value == pytest.approx(
        history[-1]["validation"]["total"]
    )
    telemetry = json.loads(
        (tmp_path / "instrumented" / "instrumentation" / "telemetry.json").read_text()
    )[-1]
    assert telemetry["gradient_norm"] >= 0.0
    assert telemetry["examples_per_second"] > 0.0
    assert telemetry["updates_per_second"] > 0.0
    assert telemetry["data_wait_seconds"] >= 0.0
    assert telemetry["checkpoint_duration_seconds"] >= 0.0
    assert telemetry["checkpoint_size_bytes"] > 0
    assert telemetry["host_rss_bytes"] > 0
    assert telemetry["filesystem_free_bytes"] > 0
    assert telemetry["cuda_allocated_bytes"] is None
    assert telemetry["cuda_reserved_bytes"] is None
    assert telemetry["cuda_utilization_percent"] is None
    metadata = result["metadata"]
    assert metadata == {
        "initialization_mode": "random",
        "upstream_revision": config.model.hub_revision,
        "upstream_weights_sha256": config.model.weights_sha256,
        "trainable_parameters": 4605639,
        "frozen_parameters": 820800,
        "seed": 7,
        "precision": "float32",
        "resume_source": None,
    }
    assert "run/metadata/text_summary" in events.Tags()["tensors"]


def test_tensorboard_failure_preserves_authoritative_training_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingWriter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("event sink failed")

    base = load_config(ROOT / "configs" / "smoke.toml")
    config = replace(
        base,
        run=replace(base.run, name="failed-events", artifact_root=tmp_path),
        training=replace(
            base.training,
            max_steps_per_epoch=1,
            validation_max_steps=1,
            checkpoint_every=1,
        ),
    )
    monkeypatch.setattr(instrumentation_module, "SummaryWriter", FailingWriter)

    result = train(config)

    run_root = tmp_path / "failed-events"
    assert Path(result["checkpoint"]).is_file()
    assert json.loads((run_root / "metrics.json").read_text())[-1]["global_step"] == 1
    diagnostics = [
        json.loads(line)
        for line in (run_root / "instrumentation" / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert diagnostics == [
        {
            "error": "event sink failed",
            "error_type": "RuntimeError",
            "operation": "writer_initialization",
        }
    ]


def _trained_smoke_config(tmp_path: Path, name: str) -> Any:
    base = load_config(ROOT / "configs" / "smoke.toml")
    config = replace(
        base,
        run=replace(base.run, name=name, artifact_root=tmp_path, allow_overwrite=True),
        training=replace(base.training, max_steps_per_epoch=1, validation_max_steps=1),
    )
    train(config)
    return config


def test_export_requires_completed_evaluation(tmp_path: Path) -> None:
    config = _trained_smoke_config(tmp_path, "missing-evaluation")

    with pytest.raises(PipelineError, match="run evaluate before export"):
        export(config)

    assert not (tmp_path / "missing-evaluation" / "export").exists()


def test_evaluation_authorizes_export_of_exact_best_checkpoint(tmp_path: Path) -> None:
    config = _trained_smoke_config(tmp_path, "validated-export")

    evaluation = evaluate(config)
    manifest = export(config)

    assert evaluation["schema_version"] == 1
    assert evaluation["passed"] is True
    assert evaluation["checkpoint"]["sha256"]
    assert evaluation["checkpoint"]["epoch"] == 0
    assert evaluation["checkpoint"]["global_step"] == 1
    assert manifest["validation_gate"]["verified"] is True
    assert manifest["validation_gate"]["checkpoint_sha256"] == evaluation["checkpoint"]["sha256"]
    bundled_evaluation = Path(manifest["validation_gate"]["evaluation"])
    assert bundled_evaluation == tmp_path / "validated-export" / "export" / "evaluation.json"
    assert json.loads(bundled_evaluation.read_text()) == json.loads(
        json.dumps(evaluation, default=str)
    )
    assert manifest["validation_gate"]["evaluation_sha256"] == pipeline_module.sha256_file(
        bundled_evaluation
    )
    assert manifest["weights_sha256"]


def test_export_rejects_evaluation_for_replaced_best_checkpoint(tmp_path: Path) -> None:
    config = _trained_smoke_config(tmp_path, "stale-evaluation")
    evaluate(config)
    checkpoint_path = tmp_path / "stale-evaluation" / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    first_tensor = next(iter(checkpoint["model"].values()))
    first_tensor.view(-1)[0] += 1
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(PipelineError, match="best checkpoint changed after evaluation"):
        export(config)

    assert not (tmp_path / "stale-evaluation" / "export").exists()


def test_export_rejects_each_stale_evaluation_provenance_identity(tmp_path: Path) -> None:
    config = _trained_smoke_config(tmp_path, "stale-provenance")
    original = evaluate(config)
    evaluation_path = tmp_path / "stale-provenance" / "evaluation.json"
    for key in pipeline_module._PROVENANCE_IDENTITY_KEYS:
        stale = copy.deepcopy(original)
        stale["provenance"][key] = "tampered"
        evaluation_path.write_text(json.dumps(stale))

        with pytest.raises(PipelineError, match=f"evaluation provenance mismatch: {key}"):
            export(config)

    assert not (tmp_path / "stale-provenance" / "export").exists()


def test_export_rejects_checkpoint_that_is_not_metric_history_best(tmp_path: Path) -> None:
    config = _trained_smoke_config(tmp_path, "stale-selection")
    evaluate(config)
    metrics_path = tmp_path / "stale-selection" / "metrics.json"
    history = json.loads(metrics_path.read_text())
    history.append(
        {
            "epoch": 1,
            "global_step": 2,
            "validation": {"total": history[0]["validation"]["total"] - 1},
        }
    )
    metrics_path.write_text(json.dumps(history))

    with pytest.raises(PipelineError, match="does not match validation selection history"):
        export(config)

    assert not (tmp_path / "stale-selection" / "export").exists()


def test_evaluation_rejects_checkpoint_replaced_during_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _trained_smoke_config(tmp_path, "evaluation-race")
    checkpoint_path = tmp_path / "evaluation-race" / "checkpoints" / "best.pt"
    original_run_epoch = pipeline_module._run_epoch

    def replace_after_evaluation(*args: Any, **kwargs: Any) -> tuple[dict[str, float], int]:
        result = original_run_epoch(*args, **kwargs)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        first_tensor = next(iter(checkpoint["model"].values()))
        first_tensor.view(-1)[0] += 1
        torch.save(checkpoint, checkpoint_path)
        return result

    monkeypatch.setattr(pipeline_module, "_run_epoch", replace_after_evaluation)
    with pytest.raises(PipelineError, match="best checkpoint changed during evaluation"):
        evaluate(config)

    assert not (tmp_path / "evaluation-race" / "evaluation.json").exists()


def test_validated_export_reuses_one_loaded_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "smoke.toml")
    loaded = object()
    calls: list[str] = []

    def load_once(_: Any) -> Any:
        calls.append("load")
        return loaded

    def evaluate_loaded(_: Any, candidate: Any) -> dict[str, Any]:
        assert candidate is loaded
        calls.append("evaluate")
        return {"passed": True}

    def export_loaded(_: Any, candidate: Any) -> dict[str, Any]:
        assert candidate is loaded
        calls.append("export")
        return {"parity": {"verified": True}}

    monkeypatch.setattr(pipeline_module, "_load_trained", load_once)
    monkeypatch.setattr(pipeline_module, "_evaluate_loaded", evaluate_loaded)
    monkeypatch.setattr(pipeline_module, "_export_loaded", export_loaded)

    result = validated_export(config)

    assert calls == ["load", "evaluate", "export"]
    assert result == {
        "evaluation": {"passed": True},
        "export": {"parity": {"verified": True}},
    }


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
    base = load_config(ROOT / "configs" / "nextleg-parquet-v2-probe.toml")
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
        *,
        completed_steps: int = 0,
        telemetry: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], int]:
        del model, loader, pipeline_config, device, max_steps, completed_steps
        if telemetry is not None:
            telemetry.update(
                gradient_norm=1.0,
                examples_per_second=1.0,
                updates_per_second=1.0,
                data_wait_seconds=0.0,
                duration_seconds=1.0,
            )
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
        *,
        completed_steps: int = 0,
        telemetry: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], int]:
        del model, loader, pipeline_config, device, max_steps, completed_steps
        if telemetry is not None:
            telemetry.update(
                gradient_norm=1.0,
                examples_per_second=1.0,
                updates_per_second=1.0,
                data_wait_seconds=0.0,
                duration_seconds=1.0,
            )
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
        warmup_epochs=1,
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
    run_root = tmp_path / f"interrupted-{interrupt_length}"
    event_root = run_root / "events"
    original_event_files = set(event_root.glob("events.out.tfevents.*"))
    assert original_event_files
    if interrupt_length == 1:
        (run_root / "instrumentation" / "telemetry.json").unlink()
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
    assert original_event_files < set(event_root.glob("events.out.tfevents.*"))
    parsed_events = instrumentation_module.parse_tensorboard_events(event_root)
    assert [event["step"] for event in parsed_events["scalars"]["run/global_step"]] == (
        [2, 3] if interrupt_length == 1 else [1, 2, 3]
    )
    expected_resume_source = str((run_root / "checkpoints" / "pending.pt").resolve())
    assert resumed_result["metadata"]["resume_source"] == expected_resume_source
    assert json.loads((run_root / "run-metadata.json").read_text()) == resumed_result["metadata"]
    assert (
        json.loads(parsed_events["text"]["run/metadata/text_summary"][-1]["value"])["resume_source"]
        == expected_resume_source
    )
