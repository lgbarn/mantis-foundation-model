from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from mantis.architecture import MantisV2
from mantis_v2 import model as model_module
from mantis_v2.cli import _parser
from mantis_v2.config import load_config
from mantis_v2.cuda_qualification import (
    ArtifactParityEvidence,
    BatchTrial,
    EnvironmentEvidence,
    QualificationError,
    ResumeEvidence,
    TrainabilityEvidence,
    build_probe_config,
    capture_fp32_step,
    compare_numeric_evidence,
    compare_resume_evidence,
    cpu_skip_record,
    load_fp32_fixture,
    load_qualification_config,
    qualification_subject_digest,
    run_batch_envelope_trials,
    run_probe_and_export,
    run_subprocess_batch_trial,
    select_batch_envelope,
    validate_artifact_parity,
    validate_environment_evidence,
    validate_trainability,
)
from mantis_v2.model import NextLegModel
from mantis_v2.pipeline import PipelineError, train
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def test_cuda_qualification_profile_encodes_the_accepted_probe_contract(tmp_path: Path) -> None:
    qualification = load_qualification_config(ROOT / "configs" / "cuda-fp32-qualification.toml")
    pipeline = load_config(ROOT / "configs" / "nextleg-parquet-v2-probe.toml")

    configured = build_probe_config(
        pipeline,
        qualification,
        run_id="cuda-fp32-probe-20260721T120000Z-a1b2c3d4",
        artifact_root=tmp_path,
    )

    assert configured.run.device == "cuda"
    assert configured.run.require_accelerator is True
    assert configured.run.allow_overwrite is False
    assert configured.training.epochs == 1
    assert configured.training.max_steps_per_epoch == 32
    assert configured.training.validation_max_steps == 1
    assert configured.training.resume is False
    assert configured.evaluation.allow_holdout is False
    assert qualification.probe.pin_memory is True
    assert qualification.precision == "fp32"
    assert qualification.export.atol == 1e-5
    assert qualification.export.rtol == 1e-4


def test_probe_contract_rejects_non_unique_or_existing_run_identity(tmp_path: Path) -> None:
    qualification = load_qualification_config(ROOT / "configs" / "cuda-fp32-qualification.toml")
    pipeline = load_config(ROOT / "configs" / "nextleg-parquet-v2-probe.toml")

    with pytest.raises(QualificationError, match="unique disposable run identity"):
        build_probe_config(pipeline, qualification, run_id="cuda-probe", artifact_root=tmp_path)

    run_id = "cuda-fp32-probe-20260721T120000Z-a1b2c3d4"
    (tmp_path / run_id).mkdir()
    with pytest.raises(QualificationError, match="already exists"):
        build_probe_config(pipeline, qualification, run_id=run_id, artifact_root=tmp_path)


def test_probe_contract_rejects_holdout_or_unofficial_initialization(tmp_path: Path) -> None:
    qualification = load_qualification_config(ROOT / "configs" / "cuda-fp32-qualification.toml")
    pipeline = load_config(ROOT / "configs" / "nextleg-parquet-v2-probe.toml")
    run_id = "cuda-fp32-probe-20260721T120000Z-a1b2c3d4"

    with pytest.raises(QualificationError, match="holdout"):
        build_probe_config(
            replace(pipeline, evaluation=replace(pipeline.evaluation, allow_holdout=True)),
            qualification,
            run_id=run_id,
            artifact_root=tmp_path,
        )

    with pytest.raises(QualificationError, match="official-base"):
        build_probe_config(
            replace(pipeline, model=replace(pipeline.model, mode="scratch")),
            qualification,
            run_id=run_id,
            artifact_root=tmp_path,
        )

    with pytest.raises(QualificationError, match="real pre-holdout data"):
        build_probe_config(
            replace(pipeline, data=replace(pipeline.data, root="synthetic")),
            qualification,
            run_id=run_id,
            artifact_root=tmp_path,
        )

    with pytest.raises(QualificationError, match="fine-tune mode"):
        build_probe_config(
            replace(pipeline, model=replace(pipeline.model, mode="head_only")),
            qualification,
            run_id=run_id,
            artifact_root=tmp_path,
        )


def test_fixed_cpu_cuda_fixture_covers_outputs_losses_gradients_and_update() -> None:
    qualification = load_qualification_config(ROOT / "configs" / "cuda-fp32-qualification.toml")
    fixture_path = ROOT / "tests" / "fixtures" / "cuda_qualification" / "fp32_parity.json"
    fixture = load_fp32_fixture(fixture_path)
    assert fixture["context"].shape == (1, 5, 512)
    assert fixture["candle_target"].shape == (1, 5, 4)
    assert fixture["leg_target"].tolist() == [[-0.25, 0.5]]

    cpu = {
        "outputs": {"candle": [0.125, -0.25], "leg": [0.5, 0.75]},
        "losses": {"candle": 0.25, "leg": 0.5, "total": 0.75},
        "gradients": {"head.bias": [0.0625, -0.125]},
        "updated_parameters": {"head.bias": [0.999, -0.998]},
    }
    cuda = {
        **cpu,
        "outputs": {"candle": [0.125001, -0.25], "leg": [0.5, 0.750001]},
    }
    compare_numeric_evidence(cpu, cuda, qualification.parity)

    broken = dict(cuda)
    broken["losses"] = {**broken["losses"], "total": 2.0}
    with pytest.raises(QualificationError, match=r"losses.total"):
        compare_numeric_evidence(cpu, broken, qualification.parity)


def test_batch_envelope_doubles_until_classified_oom_and_applies_both_caps() -> None:
    qualification = load_qualification_config(ROOT / "configs" / "cuda-fp32-qualification.toml")
    trials = (
        BatchTrial(32, "clean", 100.0, 30, 40, 0.10, True, True),
        BatchTrial(64, "clean", 180.0, 70, 65, 0.08, True, True),
        BatchTrial(128, "clean", 240.0, 85, 70, 0.07, True, True),
        BatchTrial(256, "cuda_oom", None, None, None, None, True, True),
    )

    result = select_batch_envelope(
        trials,
        qualification.batch_envelope,
        gpu_total_bytes=100,
        host_total_bytes=100,
    )

    assert result.selected_batch_size == 64
    assert result.first_oom_batch_size == 256


def test_batch_envelope_rejects_in_process_or_unclassified_trials() -> None:
    qualification = load_qualification_config(ROOT / "configs" / "cuda-fp32-qualification.toml")
    valid = BatchTrial(32, "clean", 100.0, 30, 40, 0.1, True, True)

    with pytest.raises(QualificationError, match="fresh process"):
        select_batch_envelope(
            (replace(valid, fresh_process=False),),
            qualification.batch_envelope,
            gpu_total_bytes=100,
            host_total_bytes=100,
        )
    with pytest.raises(QualificationError, match="unclassified"):
        select_batch_envelope(
            (valid, BatchTrial(64, "error", None, None, None, None, True, True)),
            qualification.batch_envelope,
            gpu_total_bytes=100,
            host_total_bytes=100,
        )


def test_resume_oracle_binds_provenance_state_and_final_values() -> None:
    qualification = load_qualification_config(ROOT / "configs" / "cuda-fp32-qualification.toml")
    uninterrupted = ResumeEvidence(
        subject_digest="a" * 64,
        data_digest="b" * 64,
        source_revision="c" * 40,
        lock_digest="d" * 64,
        completed_updates=32,
        optimizer_state_hash="e" * 64,
        rng_state_hash="f" * 64,
        tensors={"head.weight": [1.0, 2.0]},
        metrics={"validation_loss": 0.5},
    )
    resumed = replace(uninterrupted, tensors={"head.weight": [1.000001, 2.0]})

    compare_resume_evidence(uninterrupted, resumed, qualification.resume)

    with pytest.raises(QualificationError, match="optimizer_state_hash"):
        compare_resume_evidence(
            uninterrupted,
            replace(resumed, optimizer_state_hash="0" * 64),
            qualification.resume,
        )


def test_cpu_skip_record_is_explicit_and_contains_no_holdout_unlock() -> None:
    record = cpu_skip_record(cuda_available=False, reason="torch.cuda.is_available() is false")
    assert record == {
        "schema_version": 1,
        "status": "skipped",
        "device": "cuda",
        "precision": "fp32",
        "reason_code": "cuda_unavailable",
        "reason": "torch.cuda.is_available() is false",
        "holdout_unlocked": False,
    }

    with pytest.raises(QualificationError, match="available CUDA"):
        cpu_skip_record(cuda_available=True, reason="operator opted out")


@pytest.mark.parametrize(
    ("mode", "trainable", "frozen"),
    [
        ("full_finetune", 4624854, 820800),
        ("transformer_finetune", 4385494, 1060160),
    ],
)
def test_trainability_is_exact_and_bound_to_official_base(
    mode: str, trainable: int, frozen: int
) -> None:
    qualification = load_qualification_config(ROOT / "configs" / "cuda-fp32-qualification.toml")
    evidence = TrainabilityEvidence(
        mode=mode,
        trainable_parameters=trainable,
        frozen_parameters=frozen,
        initialization_mode="official",
        source_repository=qualification.official_base.source_repository,
        source_revision=qualification.official_base.source_revision,
        hub_model=qualification.official_base.hub_model,
        hub_revision=qualification.official_base.hub_revision,
        weights_sha256=qualification.official_base.weights_sha256,
        fallback_checkpoint=None,
    )

    validate_trainability(evidence, qualification)

    with pytest.raises(QualificationError, match="trainable_parameters"):
        validate_trainability(replace(evidence, trainable_parameters=trainable + 1), qualification)
    with pytest.raises(QualificationError, match="fallback"):
        validate_trainability(replace(evidence, fallback_checkpoint="scratch.pt"), qualification)


def test_environment_evidence_binds_image_runtime_device_and_no_holdout_paths() -> None:
    evidence = EnvironmentEvidence(
        image_digest="sha256:" + "a" * 64,
        base_image_digest="sha256:" + "b" * 64,
        source_revision="c" * 40,
        lock_digest="d" * 64,
        torch_version="2.8.0",
        cuda_runtime="12.8",
        cuda_driver="570.00",
        device_name="NVIDIA RTX 4090",
        device_capability="8.9",
        device_total_memory_bytes=24_000_000_000,
        device="cuda",
        precision="fp32",
        pin_memory=True,
        read_artifacts=("corpus/manifest.json", "weights/model.safetensors"),
        emitted_artifacts=("qualification/report.json",),
        holdout_unlocked=False,
    )

    validate_environment_evidence(evidence)

    with pytest.raises(QualificationError, match="holdout artifact"):
        validate_environment_evidence(replace(evidence, read_artifacts=("corpus/holdout.json",)))


def test_evaluation_and_safetensors_reload_bind_the_selected_checkpoint() -> None:
    qualification = load_qualification_config(ROOT / "configs" / "cuda-fp32-qualification.toml")
    outputs = {"candle": [0.1, 0.2], "leg": [0.3, 0.4]}
    evidence = ArtifactParityEvidence(
        format="safetensors",
        selected_checkpoint_digest="a" * 64,
        evaluated_checkpoint_digest="a" * 64,
        exported_checkpoint_digest="a" * 64,
        config_digest="b" * 64,
        data_digest="c" * 64,
        source_revision="d" * 40,
        lock_digest="e" * 64,
        upstream_weights_sha256="f" * 64,
        native_outputs=outputs,
        reloaded_outputs={"candle": [0.100001, 0.2], "leg": [0.3, 0.4]},
        holdout_unlocked=False,
    )

    validate_artifact_parity(evidence, qualification)

    with pytest.raises(QualificationError, match="evaluated_checkpoint_digest"):
        validate_artifact_parity(
            replace(evidence, evaluated_checkpoint_digest="0" * 64), qualification
        )
    with pytest.raises(QualificationError, match="config_digest"):
        validate_artifact_parity(replace(evidence, config_digest=""), qualification)


def test_fp32_step_capture_records_all_parity_components() -> None:
    class FixtureModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(1, 22)

        def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
            values = self.projection(context[:, :1, :1].reshape(len(context), 1))
            return {"candle": values[:, :20].reshape(-1, 5, 4), "leg": values[:, 20:]}

    config = load_config(ROOT / "configs" / "nextleg-parquet-v2-probe.toml")
    torch.manual_seed(7)
    model = FixtureModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = {
        "context": torch.ones(2, 5, 512),
        "candle_target": torch.zeros(2, 5, 4),
        "leg_target": torch.zeros(2, 2),
    }

    evidence = capture_fp32_step(model, optimizer, batch, config.target, torch.device("cpu"))

    assert set(evidence) == {"outputs", "losses", "gradients", "updated_parameters"}
    assert set(evidence["outputs"]) == {"candle", "leg"}
    assert set(evidence["losses"]) == {"candle", "leg", "total"}
    assert set(evidence["gradients"]) == {"projection.weight", "projection.bias"}
    assert set(evidence["updated_parameters"]) == {"projection.weight", "projection.bias"}


def test_batch_trials_are_requested_in_doubling_order_through_oom() -> None:
    qualification = load_qualification_config(ROOT / "configs" / "cuda-fp32-qualification.toml")
    calls: list[int] = []

    def worker(batch_size: int) -> BatchTrial:
        calls.append(batch_size)
        if batch_size == 128:
            return BatchTrial(batch_size, "cuda_oom", None, None, None, None, True, True)
        return BatchTrial(batch_size, "clean", 100.0, batch_size, 40, 0.1, True, True)

    result = run_batch_envelope_trials(
        worker,
        qualification.batch_envelope,
        gpu_total_bytes=100,
        host_total_bytes=100,
    )

    assert calls == [32, 64, 128]
    assert result.selected_batch_size == 64


def test_process_epoch_limit_creates_a_normal_resumable_checkpoint(tmp_path: Path) -> None:
    base = load_config(ROOT / "configs" / "smoke.toml")
    config = replace(
        base,
        run=replace(
            base.run,
            name="qualification-resume",
            artifact_root=tmp_path,
            allow_overwrite=False,
        ),
        training=replace(
            base.training,
            epochs=2,
            max_steps_per_epoch=1,
            validation_max_steps=1,
            checkpoint_every=2,
            resume=True,
        ),
    )

    partial = train(config, process_epoch_limit=1)
    resumed = train(config)

    assert partial["epochs_completed"] == 1
    assert resumed["epochs_completed"] == 2
    assert resumed["last"]["global_step"] == 2
    assert resumed["metadata"]["resume_source"].endswith("checkpoints/latest.pt")

    with pytest.raises(PipelineError, match="process_epoch_limit"):
        train(replace(config, run=replace(config.run, name="invalid-limit")), process_epoch_limit=0)


def test_qualification_subject_digest_excludes_only_disposable_execution_identity(
    tmp_path: Path,
) -> None:
    base = load_config(ROOT / "configs" / "nextleg-parquet-v2-probe.toml")
    isolated = replace(
        base,
        run=replace(base.run, name="other-process", artifact_root=tmp_path),
    )

    assert qualification_subject_digest(base) == qualification_subject_digest(isolated)
    changed_data = replace(base, data=replace(base.data, corpus_manifest_sha256="0" * 64))
    assert qualification_subject_digest(base) != qualification_subject_digest(changed_data)


def test_probe_and_export_executes_the_public_cuda_contract(tmp_path: Path) -> None:
    qualification = load_qualification_config(ROOT / "configs" / "cuda-fp32-qualification.toml")
    pipeline = load_config(ROOT / "configs" / "nextleg-parquet-v2-probe.toml")
    observed = []

    def probe_runner(config: object) -> dict[str, object]:
        observed.append(config)
        return {
            "device": "cuda",
            "last": {"global_step": 32},
            "metadata": {
                "initialization_mode": "pinned_official_checkpoint",
                "upstream_revision": qualification.official_base.hub_revision,
                "upstream_weights_sha256": qualification.official_base.weights_sha256,
                "trainable_parameters": 4385494,
                "frozen_parameters": 1060160,
                "precision": "fp32",
            },
        }

    def release_runner(config: object) -> dict[str, object]:
        assert config is observed[0]
        return {
            "evaluation": {
                "split": "validation",
                "batches": 1,
                "checkpoint": {"sha256": "a" * 64, "global_step": 32},
            },
            "export": {
                "format": "safetensors",
                "validation_gate": {"checkpoint_sha256": "a" * 64, "global_step": 32},
                "parity": {"atol": 1e-5, "rtol": 1e-4, "verified": True},
            },
        }

    result = run_probe_and_export(
        pipeline,
        qualification,
        run_id="cuda-fp32-probe-20260721T120000Z-a1b2c3d4",
        artifact_root=tmp_path,
        cuda_available=True,
        probe_runner=probe_runner,
        release_runner=release_runner,
    )

    assert result["status"] == "passed"
    assert result["train_updates"] == 32
    assert result["validation_batches"] == 1
    assert result["holdout_unlocked"] is False
    configured = observed[0]
    assert configured.run.device == "cuda"
    assert configured.export.verify_atol == 1e-5
    assert configured.export.verify_rtol == 1e-4


def test_probe_and_export_cpu_path_skips_without_touching_runners(tmp_path: Path) -> None:
    qualification = load_qualification_config(ROOT / "configs" / "cuda-fp32-qualification.toml")
    pipeline = load_config(ROOT / "configs" / "nextleg-parquet-v2-probe.toml")

    def forbidden(_: object) -> dict[str, object]:
        raise AssertionError("CPU skip must not touch pipeline or artifact paths")

    result = run_probe_and_export(
        pipeline,
        qualification,
        run_id="cuda-fp32-probe-20260721T120000Z-a1b2c3d4",
        artifact_root=tmp_path,
        cuda_available=False,
        probe_runner=forbidden,
        release_runner=forbidden,
    )

    assert result["status"] == "skipped"
    assert list(tmp_path.iterdir()) == []


def test_subprocess_batch_trial_accepts_only_canonical_isolated_evidence() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "schema_version": 1,
                    "batch_size": 32,
                    "outcome": "clean",
                    "throughput_samples_per_second": 120.0,
                    "peak_gpu_bytes": 70,
                    "peak_host_bytes": 60,
                    "data_wait_seconds": 0.1,
                    "synchronized": True,
                }
            ),
            "",
        )

    trial = run_subprocess_batch_trial(
        ["uv", "run", "mantis-v2", "cuda-batch-worker"], 32, runner=runner
    )

    assert calls == [["uv", "run", "mantis-v2", "cuda-batch-worker", "--batch-size", "32"]]
    assert trial == BatchTrial(32, "clean", 120.0, 70, 60, 0.1, True, True)

    def crashed(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 137, "", "Killed")

    with pytest.raises(QualificationError, match="unclassified subprocess failure"):
        run_subprocess_batch_trial(["worker"], 32, runner=crashed)


def test_preregistered_counts_match_the_pinned_upstream_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = load_qualification_config(ROOT / "configs" / "cuda-fp32-qualification.toml")
    config = load_config(ROOT / "configs" / "nextleg-parquet-v2-probe.toml")
    monkeypatch.setattr(model_module, "download_verified_weights", lambda _: Path("verified"))
    monkeypatch.setattr(MantisV2, "from_pretrained", lambda self, *_args, **_kwargs: self)

    for mode in ("full_finetune", "transformer_finetune"):
        model = NextLegModel(
            replace(config.model, mode=mode),
            config.target,
            len(config.data.feature_columns),
            torch.device("cpu"),
        )
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        frozen = sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        )
        assert trainable == getattr(qualification.trainability, f"{mode}_trainable_parameters")
        assert frozen == getattr(qualification.trainability, f"{mode}_frozen_parameters")


def test_cuda_probe_cli_requires_explicit_identity_paths() -> None:
    args = _parser().parse_args(
        [
            "cuda-fp32-probe",
            "--config",
            "pipeline.toml",
            "--qualification-config",
            "qualification.toml",
            "--run-id",
            "cuda-fp32-probe-20260721T120000Z-a1b2c3d4",
            "--artifact-root",
            "/artifacts",
        ]
    )

    assert args.command == "cuda-fp32-probe"
    assert args.qualification_config == Path("qualification.toml")
    assert args.artifact_root == Path("/artifacts")
