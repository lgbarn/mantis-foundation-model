"""Fail-closed, non-promoting decision receipt for the LP-LoRA A/B screen."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mantis_v2.provenance import sha256_file


class FoundationAdaptationScreenError(RuntimeError):
    """Raised when a candidate run cannot support a paired screen decision."""


_PROVENANCE_KEYS = (
    "precision",
    "dataset_digest",
    "source_digest",
    "lock_digest",
    "upstream_source_revision",
    "upstream_hub_revision",
    "upstream_weights_sha256",
    "contamination_digest",
)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FoundationAdaptationScreenError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise FoundationAdaptationScreenError(f"{label} must be a JSON object: {path}")
    return value


def _finite_metrics(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or any(
        not isinstance(value.get(key), int | float) or not math.isfinite(float(value[key]))
        for key in ("total", "candle", "leg")
    ):
        raise FoundationAdaptationScreenError(f"{label} has incomplete validation metrics")
    return {key: float(value[key]) for key in ("total", "candle", "leg")}


def _candidate(run_root: Path, expected_mode: str) -> dict[str, Any]:
    manifest_path = run_root / "export" / "manifest.json"
    manifest = _read_object(manifest_path, "export manifest")
    evaluation_path = run_root / "export" / "evaluation.json"
    evaluation = _read_object(evaluation_path, "export evaluation")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise FoundationAdaptationScreenError("export manifest has no config")
    model = config.get("model")
    run = config.get("run")
    if not isinstance(model, dict) or model.get("mode") != expected_mode:
        raise FoundationAdaptationScreenError(f"candidate mode must be {expected_mode}")
    if not isinstance(run, dict) or not isinstance(run.get("seed"), int):
        raise FoundationAdaptationScreenError("export manifest has no run seed")
    validation_gate = manifest.get("validation_gate")
    parity = manifest.get("parity")
    if (
        not isinstance(validation_gate, dict)
        or validation_gate.get("verified") is not True
        or validation_gate.get("evaluation_sha256") != sha256_file(evaluation_path)
        or not isinstance(parity, dict)
        or parity.get("verified") is not True
    ):
        raise FoundationAdaptationScreenError("candidate export validation or parity is unverified")
    if evaluation.get("passed") is not True or evaluation.get("split") != "validation":
        raise FoundationAdaptationScreenError("candidate evaluation did not pass validation")
    provenance = evaluation.get("provenance")
    if not isinstance(provenance, dict) or manifest.get("provenance") != provenance:
        raise FoundationAdaptationScreenError("candidate export and evaluation provenance differ")
    for key in _PROVENANCE_KEYS:
        if not isinstance(provenance.get(key), str) or not provenance[key]:
            raise FoundationAdaptationScreenError(f"candidate provenance lacks {key}")
    return {
        "root": str(run_root),
        "mode": expected_mode,
        "seed": run["seed"],
        "config": config,
        "provenance": {key: provenance[key] for key in _PROVENANCE_KEYS},
        "metrics": _finite_metrics(
            evaluation.get("metrics", {}).get("micro")
            if isinstance(evaluation.get("metrics"), dict)
            else None,
            "candidate",
        ),
        "export_manifest_sha256": sha256_file(manifest_path),
        "evaluation_sha256": sha256_file(evaluation_path),
    }


def _assert_paired(direct: dict[str, Any], warm: dict[str, Any]) -> None:
    if direct["seed"] != warm["seed"]:
        raise FoundationAdaptationScreenError("screen candidates use different seeds")
    if direct["provenance"] != warm["provenance"]:
        raise FoundationAdaptationScreenError("screen candidates have different provenance")
    for section in ("data", "target", "training", "evaluation", "export"):
        if direct["config"].get(section) != warm["config"].get(section):
            raise FoundationAdaptationScreenError(f"screen candidates differ in config.{section}")
    direct_model = direct["config"].get("model")
    warm_model = warm["config"].get("model")
    if not isinstance(direct_model, dict) or not isinstance(warm_model, dict):
        raise FoundationAdaptationScreenError("screen candidates have no model config")
    if {key: value for key, value in direct_model.items() if key != "mode"} != {
        key: value for key, value in warm_model.items() if key != "mode"
    }:
        raise FoundationAdaptationScreenError("screen candidates differ in model identity")


def _write_immutable(path: Path, payload: dict[str, Any]) -> Path:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != encoded:
            raise FoundationAdaptationScreenError(f"immutable screen decision differs: {path}")
        return path
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != encoded:
                raise FoundationAdaptationScreenError(
                    f"immutable screen decision differs: {path}"
                ) from None
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def decide_adaptation_screen(
    direct_run_root: str | Path,
    warm_run_root: str | Path,
    output_path: str | Path,
) -> Path:
    """Write the seed-42 LP-LoRA screen result without selecting or promoting a model."""
    direct = _candidate(Path(direct_run_root), "lora_r8_alpha16")
    warm = _candidate(Path(warm_run_root), "lora_r8_alpha16_head_warmstart")
    _assert_paired(direct, warm)
    warm_wins = (
        warm["metrics"]["total"] < direct["metrics"]["total"]
        and warm["metrics"]["candle"] <= direct["metrics"]["candle"]
        and warm["metrics"]["leg"] <= direct["metrics"]["leg"]
    )
    decision = {
        "schema_version": 1,
        "non_promoting": True,
        "seed": direct["seed"],
        "decision": "advance_paired_seeds" if warm_wins else "stop_lp_lora",
        "criteria": {
            "warm_total_strictly_lower": warm["metrics"]["total"] < direct["metrics"]["total"],
            "warm_candle_not_higher": warm["metrics"]["candle"] <= direct["metrics"]["candle"],
            "warm_leg_not_higher": warm["metrics"]["leg"] <= direct["metrics"]["leg"],
        },
        "direct": {
            key: direct[key]
            for key in ("root", "mode", "metrics", "export_manifest_sha256", "evaluation_sha256")
        },
        "warm": {
            key: warm[key]
            for key in ("root", "mode", "metrics", "export_manifest_sha256", "evaluation_sha256")
        },
        "shared_provenance": direct["provenance"],
    }
    return _write_immutable(Path(output_path), decision)
