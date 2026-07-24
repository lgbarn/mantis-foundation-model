from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from mantis_v2.cli import _parser
from mantis_v2.foundation_adaptation_screen import (
    FoundationAdaptationScreenError,
    decide_adaptation_screen,
)


def _write_candidate(
    root: Path,
    *,
    mode: str,
    total: float,
    candle: float,
    leg: float,
    seed: int = 42,
    dataset_digest: str = "d" * 64,
) -> None:
    export = root / "export"
    export.mkdir(parents=True)
    provenance = {
        "precision": "fp32",
        "dataset_digest": dataset_digest,
        "source_digest": "s" * 64,
        "lock_digest": "l" * 64,
        "upstream_source_revision": "0c94f8ceb9f1d1421dd292ed917090df8c31605b",
        "upstream_hub_revision": "99fe0f548960e272fbfa4b82fd9b5b5956779dfd",
        "upstream_weights_sha256": "w" * 64,
        "contamination_digest": "c" * 64,
    }
    evaluation = {
        "passed": True,
        "split": "validation",
        "provenance": provenance,
        "metrics": {"micro": {"total": total, "candle": candle, "leg": leg}},
    }
    evaluation_path = export / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation))
    manifest = {
        "config": {
            "run": {"name": root.name, "seed": seed},
            "data": {"intervals": ["1min", "3min", "15min"]},
            "model": {"mode": mode, "hub_revision": provenance["upstream_hub_revision"]},
            "training": {"precision": "fp32", "max_steps_per_epoch": 200},
            "target": {"kind": "nextleg"},
            "evaluation": {"allow_holdout": False},
            "export": {"format": "safetensors"},
        },
        "provenance": provenance,
        "validation_gate": {
            "verified": True,
            "evaluation_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        },
        "parity": {"verified": True},
    }
    (export / "manifest.json").write_text(json.dumps(manifest))


def test_adaptation_screen_advances_only_for_strict_clean_warm_win(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    warm = tmp_path / "warm"
    _write_candidate(direct, mode="lora_r8_alpha16", total=1.0, candle=0.5, leg=0.5)
    _write_candidate(
        warm,
        mode="lora_r8_alpha16_head_warmstart",
        total=0.9,
        candle=0.5,
        leg=0.4,
    )

    output = tmp_path / "screen" / "screen-decision.json"
    assert decide_adaptation_screen(direct, warm, output) == output
    decision = json.loads(output.read_text())

    assert decision["decision"] == "advance_paired_seeds"
    assert decision["non_promoting"] is True
    assert decision["criteria"] == {
        "warm_total_strictly_lower": True,
        "warm_candle_not_higher": True,
        "warm_leg_not_higher": True,
    }
    assert decide_adaptation_screen(direct, warm, output) == output


def test_adaptation_screen_stops_for_a_mixed_result(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    warm = tmp_path / "warm"
    _write_candidate(direct, mode="lora_r8_alpha16", total=1.0, candle=0.5, leg=0.5)
    _write_candidate(
        warm,
        mode="lora_r8_alpha16_head_warmstart",
        total=0.9,
        candle=0.6,
        leg=0.4,
    )

    output = tmp_path / "screen-decision.json"
    decide_adaptation_screen(direct, warm, output)

    assert json.loads(output.read_text())["decision"] == "stop_lp_lora"


def test_adaptation_screen_rejects_mismatched_dataset_provenance(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    warm = tmp_path / "warm"
    _write_candidate(direct, mode="lora_r8_alpha16", total=1.0, candle=0.5, leg=0.5)
    _write_candidate(
        warm,
        mode="lora_r8_alpha16_head_warmstart",
        total=0.9,
        candle=0.4,
        leg=0.4,
        dataset_digest="x" * 64,
    )

    with pytest.raises(FoundationAdaptationScreenError, match="different provenance"):
        decide_adaptation_screen(direct, warm, tmp_path / "screen-decision.json")


def test_adaptation_screen_command_is_discoverable() -> None:
    args = _parser().parse_args(
        [
            "foundation-adaptation-screen",
            "--direct-run-root",
            "direct",
            "--warm-run-root",
            "warm",
            "--output",
            "screen.json",
        ]
    )

    assert args.command == "foundation-adaptation-screen"
