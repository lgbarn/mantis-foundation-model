from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from mantis_v2.bf16_qualification import (
    Bf16CandidateEvidence,
    Bf16QualificationError,
    load_bf16_qualification_config,
    qualify_bf16,
    qualify_bf16_files,
)
from mantis_v2.cli import _parser
from mantis_v2.cuda_qualification import ResumeEvidence

ROOT = Path(__file__).resolve().parents[1]


def _numeric(offset: float = 0.0):
    return {
        "outputs": {"candle": [1.0 + offset], "leg": [2.0 + offset]},
        "losses": {"candle": 0.25 + offset, "leg": 0.5 + offset, "total": 0.75 + offset},
        "gradients": {"head.weight": [0.125 + offset]},
        "updated_parameters": {"head.weight": [0.875 + offset]},
    }


def _resume() -> ResumeEvidence:
    return ResumeEvidence(
        subject_digest="a" * 64,
        data_digest="b" * 64,
        source_revision="c" * 40,
        lock_digest="d" * 64,
        completed_updates=32,
        optimizer_state_hash="e" * 64,
        rng_state_hash="f" * 64,
        tensors={"head.weight": [0.875]},
        metrics={"validation_loss": 0.75},
    )


def _candidate() -> Bf16CandidateEvidence:
    return Bf16CandidateEvidence(
        numeric=_numeric(0.001),
        uninterrupted=_resume(),
        resumed=replace(_resume(), tensors={"head.weight": [0.876]}),
        export_native=_numeric(),
        export_reloaded=_numeric(0.001),
        optimizer_precision="fp32",
        precision_records={
            name: "bf16"
            for name in (
                "config",
                "provenance",
                "checkpoint",
                "evaluation",
                "export",
                "tensorboard",
                "qualification",
            )
        },
    )


def test_bf16_candidate_passes_registered_parity_resume_and_manifest_contract(
    tmp_path: Path,
) -> None:
    config = load_bf16_qualification_config(ROOT / "configs" / "cuda-bf16-qualification.toml")
    output = tmp_path / "qualification.json"

    result = qualify_bf16(config, _numeric(), _candidate, output)

    assert result["status"] == "passed"
    assert result["selected_precision"] == "bf16"
    assert result["attempts"] == 1
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("unsupported autocast operation"),
        RuntimeError("CUDA out of memory"),
        RuntimeError("non-finite optimizer state"),
    ],
)
def test_bf16_failure_durably_selects_fp32_without_retry(
    tmp_path: Path, failure: RuntimeError
) -> None:
    config = load_bf16_qualification_config(ROOT / "configs" / "cuda-bf16-qualification.toml")
    calls = 0

    def fail() -> Bf16CandidateEvidence:
        nonlocal calls
        calls += 1
        raise failure

    output = tmp_path / "failure.json"
    result = qualify_bf16(config, _numeric(), fail, output)

    assert calls == 1
    assert result["status"] == "rejected"
    assert result["selected_precision"] == "fp32"
    assert result["automatic_retry"] is False
    assert result["tolerances"] == {
        "parity": {"atol": 0.01, "rtol": 0.01},
        "resume": {"atol": 0.01, "rtol": 0.01},
        "export": {"atol": 0.01, "rtol": 0.01},
    }
    assert json.loads(output.read_text()) == result

    with pytest.raises(Bf16QualificationError, match="already exists"):
        qualify_bf16(config, _numeric(), _candidate, output)


def test_bf16_nonfinite_parity_rejects_candidate(tmp_path: Path) -> None:
    config = load_bf16_qualification_config(ROOT / "configs" / "cuda-bf16-qualification.toml")
    candidate = replace(_candidate(), numeric=_numeric(float("nan")))

    result = qualify_bf16(config, _numeric(), lambda: candidate, tmp_path / "nan.json")

    assert result["status"] == "rejected"
    assert result["selected_precision"] == "fp32"
    assert "non-finite" in str(result["failure"])


def test_public_file_path_promotes_complete_bf16_evidence(tmp_path: Path) -> None:
    reference = tmp_path / "fp32.json"
    candidate = tmp_path / "bf16.json"
    output = tmp_path / "qualification.json"
    reference.write_text(json.dumps(_numeric()))
    candidate.write_text(json.dumps(asdict(_candidate())))

    result = qualify_bf16_files(
        config_path=ROOT / "configs" / "cuda-bf16-qualification.toml",
        reference_path=reference,
        candidate_path=candidate,
        output_path=output,
    )

    assert result["status"] == "passed"
    assert result["selected_precision"] == "bf16"


def test_public_file_path_records_paid_run_failure_without_retry(tmp_path: Path) -> None:
    reference = tmp_path / "fp32.json"
    reference.write_text(json.dumps(_numeric()))

    result = qualify_bf16_files(
        config_path=ROOT / "configs" / "cuda-bf16-qualification.toml",
        reference_path=reference,
        failure="CUDA out of memory",
        output_path=tmp_path / "qualification.json",
    )

    assert result["status"] == "rejected"
    assert result["attempts"] == 1
    assert result["selected_precision"] == "fp32"


def test_bf16_cli_requires_candidate_or_failure() -> None:
    args = _parser().parse_args(
        [
            "cuda-bf16-qualify",
            "--qualification-config",
            "qualification.toml",
            "--reference",
            "fp32.json",
            "--candidate",
            "bf16.json",
            "--output",
            "decision.json",
        ]
    )

    assert args.command == "cuda-bf16-qualify"
    assert args.candidate == Path("bf16.json")
    assert args.failure is None
