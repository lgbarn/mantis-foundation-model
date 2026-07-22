"""Deterministic, fail-closed orchestration for the foundation accuracy matrix."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median
from typing import Any

from mantis_v2.config import PipelineConfig, load_config


class FoundationMatrixError(RuntimeError):
    """Raised when a matrix artifact or gate violates the accepted contract."""


_MODES = (
    "full_finetune",
    "transformer_finetune",
    "lora_r8_alpha16",
    "lora_r16_alpha32",
)
_EXPECTED_KEYS = {
    "schema_version",
    "name",
    "base_config",
    "artifact_root",
    "data_root",
    "corpus_manifest_path",
    "device",
    "screen_seeds",
    "confirmation_seeds",
    "three_timeframes",
    "four_timeframes",
    "screen_train_steps",
    "screen_validation_steps",
    "preserved_train_steps",
    "preserved_validation_steps",
    "batch_size",
    "families",
    "diagnostic",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json_idempotent(path: Path, value: Any) -> None:
    encoded = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FoundationMatrixError(f"immutable matrix artifact differs: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise FoundationMatrixError(f"immutable matrix artifact differs: {path}") from None
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_bytes_idempotent(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value:
            raise FoundationMatrixError(f"immutable matrix artifact differs: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != value:
                raise FoundationMatrixError(f"immutable matrix artifact differs: {path}") from None
    finally:
        Path(temporary).unlink(missing_ok=True)


def _positive_ints(value: Any, field: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value)
    ):
        raise FoundationMatrixError(f"{field} must be a non-empty integer array")
    if len(set(value)) != len(value):
        raise FoundationMatrixError(f"{field} contains duplicates")
    return tuple(value)


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise FoundationMatrixError(f"{field} must be a non-empty string array")
    if len(set(value)) != len(value):
        raise FoundationMatrixError(f"{field} contains duplicates")
    return tuple(value)


def _load(path: str | Path) -> tuple[dict[str, Any], PipelineConfig]:
    matrix_path = Path(path)
    try:
        raw = tomllib.loads(matrix_path.read_text())
    except (FileNotFoundError, tomllib.TOMLDecodeError) as exc:
        raise FoundationMatrixError(f"invalid matrix config: {matrix_path}") from exc
    if set(raw) != _EXPECTED_KEYS:
        missing = sorted(_EXPECTED_KEYS - set(raw))
        unknown = sorted(set(raw) - _EXPECTED_KEYS)
        raise FoundationMatrixError(
            f"matrix config keys mismatch: missing={missing}, unknown={unknown}"
        )
    if raw["schema_version"] != 1:
        raise FoundationMatrixError("unsupported matrix schema")
    runtime_paths = {
        field: Path(str(raw[field]))
        for field in ("artifact_root", "data_root", "corpus_manifest_path")
    }
    if any(
        not path.is_absolute() or not path.is_relative_to("/workspace/mantis")
        for path in runtime_paths.values()
    ):
        raise FoundationMatrixError("matrix runtime paths must be absolute under /workspace/mantis")
    if raw["device"] != "cuda":
        raise FoundationMatrixError("foundation accuracy matrix requires CUDA")
    screen_seeds = _positive_ints(raw["screen_seeds"], "screen_seeds")
    confirmation_seeds = _positive_ints(raw["confirmation_seeds"], "confirmation_seeds")
    if screen_seeds != (42, 43, 44) or confirmation_seeds != (42, 43, 44, 45, 46):
        raise FoundationMatrixError("matrix seeds differ from the accepted sequence")
    three = _strings(raw["three_timeframes"], "three_timeframes")
    four = _strings(raw["four_timeframes"], "four_timeframes")
    if three != ("1min", "3min", "15min") or four != (
        "1min",
        "3min",
        "5min",
        "15min",
    ):
        raise FoundationMatrixError("matrix timeframes differ from the accepted recipes")
    for field, expected in (
        ("screen_train_steps", 200),
        ("screen_validation_steps", 20),
        ("preserved_train_steps", 267),
        ("preserved_validation_steps", 27),
        ("batch_size", 128),
    ):
        if raw[field] != expected:
            raise FoundationMatrixError(f"{field} must equal {expected}")
    if not isinstance(raw["families"], dict) or not raw["families"]:
        raise FoundationMatrixError("families must be declared")
    family_symbols = [
        symbol for values in raw["families"].values() for symbol in _strings(values, "families")
    ]
    if len(family_symbols) != len(set(family_symbols)):
        raise FoundationMatrixError("instrument families overlap")
    diagnostic = raw["diagnostic"]
    if not isinstance(diagnostic, dict) or set(diagnostic) != {
        "fixture_manifest",
        "fixture_manifest_sha256",
        "fixture_digest",
        "fit_before",
        "score_start",
        "score_end",
        "seed",
        "fit_cap",
        "score_cap",
    }:
        raise FoundationMatrixError("diagnostic contract is incomplete")
    if (diagnostic["seed"], diagnostic["fit_cap"], diagnostic["score_cap"]) != (0, 25000, 25000):
        raise FoundationMatrixError("diagnostic sampling contract differs from the accepted values")
    fixture_path = Path(str(diagnostic["fixture_manifest"]))
    if (
        not fixture_path.is_absolute()
        or not fixture_path.is_relative_to("/workspace/mantis/inputs")
        or fixture_path.parent.name != diagnostic["fixture_digest"]
        or not re.fullmatch(r"[0-9a-f]{64}", str(diagnostic["fixture_digest"]))
        or not re.fullmatch(r"[0-9a-f]{64}", str(diagnostic["fixture_manifest_sha256"]))
    ):
        raise FoundationMatrixError("diagnostic fixture identity is invalid")
    base_path = Path(str(raw["base_config"]))
    if not base_path.is_absolute():
        base_path = matrix_path.parent / base_path
    base = load_config(base_path)
    if (
        base.evaluation.allow_holdout
        or base.data.holdout_start.isoformat() != "2026-01-01T00:00:00+00:00"
    ):
        raise FoundationMatrixError("base config does not seal the accepted holdout")
    if set(family_symbols) != set(base.data.symbols):
        raise FoundationMatrixError("instrument families must partition every configured symbol")
    raw["base_config"] = str(base_path.resolve())
    return raw, base


def _matrix_contract_digest(matrix: Mapping[str, Any]) -> str:
    return _digest(
        {
            "runtime": _runtime_contract(matrix),
            "families": matrix["families"],
            "diagnostic": matrix["diagnostic"],
        }
    )


def _cell(
    matrix: Mapping[str, Any],
    base: PipelineConfig,
    *,
    phase: str,
    recipe: str,
    intervals: tuple[str, ...],
    mode: str,
    seed: int,
    train_steps: int,
    validation_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix_contract_digest = _matrix_contract_digest(matrix)
    identity = {
        "matrix": matrix["name"],
        "phase": phase,
        "recipe": recipe,
        "mode": mode,
        "seed": seed,
        "intervals": intervals,
        "train_steps": train_steps,
        "validation_steps": validation_steps,
        "batch_size": matrix["batch_size"],
        "base_config_digest": base.digest,
        "matrix_contract_digest": matrix_contract_digest,
    }
    identity_digest = _digest(identity)
    run_name = f"{matrix['name']}-{phase}-{recipe}-{mode}-s{seed}-{identity_digest[:10]}"
    config = replace(
        base,
        run=replace(
            base.run,
            name=run_name,
            seed=seed,
            artifact_root=Path(str(matrix["artifact_root"])),
            device="cuda",
            require_accelerator=True,
            allow_overwrite=False,
        ),
        data=replace(
            base.data,
            root=str(matrix["data_root"]),
            corpus_manifest_path=Path(str(matrix["corpus_manifest_path"])),
            intervals=intervals,
        ),
        model=replace(base.model, mode=mode),  # type: ignore[arg-type]
        training=replace(
            base.training,
            batch_size=int(matrix["batch_size"]),
            max_steps_per_epoch=train_steps,
            validation_max_steps=validation_steps,
            resume=True,
        ),
    )
    payload = _jsonable(asdict(config))
    cell = {
        **identity,
        "cell_id": identity_digest,
        "run_name": run_name,
        "config_digest": config.digest,
        "config_path": f"configs/{run_name}.json",
        "training": payload["training"],
    }
    return cell, payload


def _runtime_contract(matrix: Mapping[str, Any]) -> dict[str, str]:
    return {
        "artifact_root": str(matrix["artifact_root"]),
        "data_root": str(matrix["data_root"]),
        "corpus_manifest_path": str(matrix["corpus_manifest_path"]),
        "device": str(matrix["device"]),
    }


def _publish_plan(
    output: str | Path,
    core: dict[str, Any],
    configs: list[tuple[str, dict[str, Any]]],
    base_config_path: Path,
) -> Path:
    digest = _digest(core)
    plan = {**core, "plan_digest": digest}
    root = Path(output) / digest
    _atomic_bytes_idempotent(root / "base-config.toml", base_config_path.read_bytes())
    for relative, payload in configs:
        _atomic_json_idempotent(root / relative, payload)
    plan_path = root / "matrix-plan.json"
    _atomic_json_idempotent(plan_path, plan)
    return plan_path


def render_initial_plan(config_path: str | Path, output: str | Path) -> Path:
    matrix, base = _load(config_path)
    cells: list[dict[str, Any]] = []
    configs: list[tuple[str, dict[str, Any]]] = []
    for recipe, intervals in (
        ("3tf", tuple(matrix["three_timeframes"])),
        ("4tf", tuple(matrix["four_timeframes"])),
    ):
        for seed in matrix["screen_seeds"]:
            cell, payload = _cell(
                matrix,
                base,
                phase="initial",
                recipe=recipe,
                intervals=intervals,
                mode="full_finetune",
                seed=seed,
                train_steps=matrix["screen_train_steps"],
                validation_steps=matrix["screen_validation_steps"],
            )
            cells.append(cell)
            configs.append((cell["config_path"], payload))
    core = {
        "schema_version": 1,
        "matrix_name": matrix["name"],
        "phase": "initial",
        "base_config": "base-config.toml",
        "base_config_sha256": hashlib.sha256(
            Path(str(matrix["base_config"])).read_bytes()
        ).hexdigest(),
        "base_config_digest": base.digest,
        "runtime": _runtime_contract(matrix),
        "families": matrix["families"],
        "diagnostic": matrix["diagnostic"],
        "cells": cells,
    }
    return _publish_plan(output, core, configs, Path(str(matrix["base_config"])))


def render_mode_plan(
    config_path: str | Path, promotion_decision: Mapping[str, Any], output: str | Path
) -> Path:
    decision_core = {
        key: value for key, value in promotion_decision.items() if key != "decision_digest"
    }
    if (
        promotion_decision.get("schema_version") != 1
        or promotion_decision.get("gate") != "five_minute"
        or promotion_decision.get("decision") != "promote"
        or promotion_decision.get("decision_digest") != _digest(decision_core)
        or not isinstance(promotion_decision.get("plan_digest"), str)
        or not isinstance(promotion_decision.get("result_digests"), dict)
    ):
        raise FoundationMatrixError("mode plan requires a passed promotion gate")
    matrix, base = _load(config_path)
    if (
        promotion_decision.get("matrix_name") != matrix["name"]
        or promotion_decision.get("base_config_digest") != base.digest
        or promotion_decision.get("matrix_contract_digest") != _matrix_contract_digest(matrix)
    ):
        raise FoundationMatrixError("five-minute decision belongs to another matrix")
    cells: list[dict[str, Any]] = []
    configs: list[tuple[str, dict[str, Any]]] = []
    for mode in _MODES:
        for seed in matrix["screen_seeds"]:
            cell, payload = _cell(
                matrix,
                base,
                phase="mode",
                recipe="4tf",
                intervals=tuple(matrix["four_timeframes"]),
                mode=mode,
                seed=seed,
                train_steps=matrix["preserved_train_steps"],
                validation_steps=matrix["preserved_validation_steps"],
            )
            cells.append(cell)
            configs.append((cell["config_path"], payload))
    core = {
        "schema_version": 1,
        "matrix_name": matrix["name"],
        "phase": "mode",
        "promotion_decision_digest": promotion_decision["decision_digest"],
        "base_config": "base-config.toml",
        "base_config_sha256": hashlib.sha256(
            Path(str(matrix["base_config"])).read_bytes()
        ).hexdigest(),
        "base_config_digest": base.digest,
        "runtime": _runtime_contract(matrix),
        "families": matrix["families"],
        "diagnostic": matrix["diagnostic"],
        "cells": cells,
    }
    return _publish_plan(output, core, configs, Path(str(matrix["base_config"])))


def _finite_number(value: Any, field: str) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise FoundationMatrixError(f"{field} must be finite")
    return float(value)


def _validate_complete_result(result: Mapping[str, Any], gate: str) -> None:
    core = {key: value for key, value in result.items() if key != "result_digest"}
    if result.get("status") != "complete":
        raise FoundationMatrixError(f"every {gate} matrix cell must be complete")
    if (
        result.get("schema_version") != 1
        or result.get("result_digest") != _digest(core)
        or not isinstance(result.get("plan_digest"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(result.get("plan_digest")))
        or not isinstance(result.get("cell_id"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(result.get("cell_id")))
        or not isinstance(result.get("matrix_name"), str)
        or not isinstance(result.get("base_config_digest"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(result.get("base_config_digest")))
        or not isinstance(result.get("matrix_contract_digest"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(result.get("matrix_contract_digest")))
        or result.get("plan_phase") not in {"initial", "mode", "confirmation"}
        or (
            result.get("upstream_decision_digest") is not None
            and not re.fullmatch(r"[0-9a-f]{64}", str(result.get("upstream_decision_digest")))
        )
    ):
        raise FoundationMatrixError(f"{gate} matrix result provenance is invalid")


def _validate_diagnostic_pair(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    reference_export = reference.get("export_manifest_sha256")
    if (
        not isinstance(reference_export, str)
        or candidate.get("diagnostic_reference_id") != reference_export
        or reference.get("diagnostic_candidate_id") != reference_export
        or reference.get("diagnostic_reference_id") != reference_export
        or candidate.get("fixture_digest") != reference.get("fixture_digest")
    ):
        raise FoundationMatrixError(
            "candidate diagnostic is not paired to the exact seed-matched 3-TF export"
        )


def decide_five_minute_gate(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for result in results:
        _validate_complete_result(result, "five-minute")
        key = (int(result.get("seed", -1)), str(result.get("recipe", "")))
        if key in by_key:
            raise FoundationMatrixError(f"duplicate matrix cell: {key}")
        by_key[key] = result
    expected = {(seed, recipe) for seed in (42, 43, 44) for recipe in ("3tf", "4tf")}
    if set(by_key) != expected:
        raise FoundationMatrixError(
            "every expected matrix cell must be complete before aggregation"
        )
    plan_digests = {str(result["plan_digest"]) for result in by_key.values()}
    matrix_names = {str(result["matrix_name"]) for result in by_key.values()}
    base_digests = {str(result["base_config_digest"]) for result in by_key.values()}
    contract_digests = {str(result["matrix_contract_digest"]) for result in by_key.values()}
    if (
        len(plan_digests) != 1
        or len(matrix_names) != 1
        or len(base_digests) != 1
        or len(contract_digests) != 1
    ):
        raise FoundationMatrixError("five-minute results do not share one initial plan")
    if any(
        result.get("plan_phase") != "initial" or result.get("upstream_decision_digest") is not None
        for result in by_key.values()
    ):
        raise FoundationMatrixError("five-minute results do not originate from the initial plan")

    primary_deltas: dict[int, float] = {}
    log_deltas: dict[int, float] = {}
    brier_deltas: dict[int, float] = {}
    family_regressions: list[dict[str, Any]] = []
    for seed in (42, 43, 44):
        reference = by_key[(seed, "3tf")]
        candidate = by_key[(seed, "4tf")]
        _validate_diagnostic_pair(reference, candidate)
        try:
            reference_total = _finite_number(
                reference["metrics"]["by_timeframe"]["3min"]["total"],
                "reference total",
            )
            candidate_total = _finite_number(
                candidate["metrics"]["by_timeframe"]["3min"]["total"],
                "candidate total",
            )
            reference_log = _finite_number(
                reference["diagnostic"]["log_loss"], "reference log loss"
            )
            candidate_log = _finite_number(
                candidate["diagnostic"]["log_loss"], "candidate log loss"
            )
            reference_brier = _finite_number(reference["diagnostic"]["brier"], "reference Brier")
            candidate_brier = _finite_number(candidate["diagnostic"]["brier"], "candidate Brier")
            reference_families = reference["metrics"]["families"]
            candidate_families = candidate["metrics"]["families"]
        except (KeyError, TypeError) as exc:
            raise FoundationMatrixError("matrix result metrics are incomplete") from exc
        primary_deltas[seed] = candidate_total - reference_total
        log_deltas[seed] = candidate_log - reference_log
        brier_deltas[seed] = candidate_brier - reference_brier
        if set(reference_families) != set(candidate_families):
            raise FoundationMatrixError("matrix result family views do not match")
        for family in reference_families:
            reference_value = _finite_number(
                reference_families[family]["3min"]["total"], "reference family total"
            )
            candidate_value = _finite_number(
                candidate_families[family]["3min"]["total"], "candidate family total"
            )
            if reference_value <= 0:
                raise FoundationMatrixError("reference family total must be positive")
            regression = candidate_value / reference_value - 1.0
            if regression > 0.01 and not math.isclose(regression, 0.01, abs_tol=1e-12):
                family_regressions.append(
                    {"seed": seed, "family": family, "regression": regression}
                )

    improving = [seed for seed, delta in primary_deltas.items() if delta < 0]
    proper = [seed for seed in (42, 43, 44) if log_deltas[seed] < 0 and brier_deltas[seed] < 0]
    candidate_logs = [
        _finite_number(by_key[(seed, "4tf")]["diagnostic"]["log_loss"], "candidate log loss")
        for seed in (42, 43, 44)
    ]
    candidate_briers = [
        _finite_number(by_key[(seed, "4tf")]["diagnostic"]["brier"], "candidate Brier")
        for seed in (42, 43, 44)
    ]
    candidate_aucs = [
        _finite_number(by_key[(seed, "4tf")]["diagnostic"]["roc_auc"], "candidate AUC")
        for seed in (42, 43, 44)
    ]
    candidate_eces = [
        _finite_number(by_key[(seed, "4tf")]["diagnostic"]["ece"], "candidate ECE")
        for seed in (42, 43, 44)
    ]
    passed = (
        len(improving) >= 2
        and median(primary_deltas.values()) < 0
        and len(proper) >= 2
        and median(log_deltas.values()) < 0
        and median(brier_deltas.values()) < 0
        and median(candidate_logs) < math.log(2.0)
        and median(candidate_briers) < 0.25
        and not family_regressions
    )
    core: dict[str, Any] = {
        "schema_version": 1,
        "gate": "five_minute",
        "decision": "promote" if passed else "reject",
        "improving_seeds": improving,
        "proper_score_improving_seeds": proper,
        "primary_deltas": primary_deltas,
        "log_loss_deltas": log_deltas,
        "brier_deltas": brier_deltas,
        "family_regressions": family_regressions,
        "median_diagnostics": {
            "log_loss": median(candidate_logs),
            "brier": median(candidate_briers),
            "roc_auc": median(candidate_aucs),
            "ece": median(candidate_eces),
        },
        "plan_digest": next(iter(plan_digests)),
        "matrix_name": next(iter(matrix_names)),
        "base_config_digest": next(iter(base_digests)),
        "matrix_contract_digest": next(iter(contract_digests)),
        "result_digests": {
            f"{seed}:{recipe}": by_key[(seed, recipe)]["result_digest"]
            for seed in (42, 43, 44)
            for recipe in ("3tf", "4tf")
        },
    }
    return {**core, "decision_digest": _digest(core)}


def _paired_summary(
    references: Mapping[int, Mapping[str, Any]],
    candidates: Mapping[int, Mapping[str, Any]],
    seeds: tuple[int, ...],
    required_wins: int,
) -> dict[str, Any]:
    if set(references) != set(seeds) or set(candidates) != set(seeds):
        raise FoundationMatrixError("every seed-matched reference and candidate must be complete")
    deltas: dict[str, dict[int, float]] = {
        key: {} for key in ("total", "candle", "leg", "log_loss", "brier", "roc_auc", "ece")
    }
    family_regressions: list[dict[str, Any]] = []
    candidate_metrics: dict[str, list[float]] = {
        key: [] for key in ("total", "candle", "leg", "log_loss", "brier", "roc_auc", "ece")
    }
    for seed in seeds:
        reference = references[seed]
        candidate = candidates[seed]
        _validate_diagnostic_pair(reference, candidate)
        try:
            reference_timeframe = reference["metrics"]["by_timeframe"]["3min"]
            candidate_timeframe = candidate["metrics"]["by_timeframe"]["3min"]
            reference_families = reference["metrics"]["families"]
            candidate_families = candidate["metrics"]["families"]
        except (KeyError, TypeError) as exc:
            raise FoundationMatrixError("matrix result metrics are incomplete") from exc
        for key in ("total", "candle", "leg"):
            reference_value = _finite_number(reference_timeframe[key], f"reference {key}")
            candidate_value = _finite_number(candidate_timeframe[key], f"candidate {key}")
            deltas[key][seed] = candidate_value - reference_value
            candidate_metrics[key].append(candidate_value)
        for key in ("log_loss", "brier", "roc_auc", "ece"):
            reference_value = _finite_number(reference["diagnostic"][key], f"reference {key}")
            candidate_value = _finite_number(candidate["diagnostic"][key], f"candidate {key}")
            deltas[key][seed] = candidate_value - reference_value
            candidate_metrics[key].append(candidate_value)
        if set(reference_families) != set(candidate_families):
            raise FoundationMatrixError("matrix result family views do not match")
        for family in reference_families:
            reference_value = _finite_number(
                reference_families[family]["3min"]["total"], "reference family total"
            )
            candidate_value = _finite_number(
                candidate_families[family]["3min"]["total"], "candidate family total"
            )
            if reference_value <= 0:
                raise FoundationMatrixError("reference family total must be positive")
            regression = candidate_value / reference_value - 1.0
            if regression > 0.01 and not math.isclose(regression, 0.01, abs_tol=1e-12):
                family_regressions.append(
                    {"seed": seed, "family": family, "regression": regression}
                )
    improving = [seed for seed in seeds if deltas["total"][seed] < 0]
    proper = [seed for seed in seeds if deltas["log_loss"][seed] < 0 and deltas["brier"][seed] < 0]
    medians = {key: median(values) for key, values in candidate_metrics.items()}
    passed = (
        len(improving) >= required_wins
        and median(deltas["total"].values()) < 0
        and len(proper) >= required_wins
        and median(deltas["log_loss"].values()) < 0
        and median(deltas["brier"].values()) < 0
        and medians["log_loss"] < math.log(2.0)
        and medians["brier"] < 0.25
        and not family_regressions
    )
    return {
        "passed": passed,
        "improving_seeds": improving,
        "proper_score_improving_seeds": proper,
        "median_metrics": medians,
        "median_deltas": {key: median(values.values()) for key, values in deltas.items()},
        "family_regressions": family_regressions,
    }


def _lora_noninferior(
    candidate_metrics: Mapping[str, float], full_metrics: Mapping[str, float]
) -> bool:
    return (
        candidate_metrics["total"] <= full_metrics["total"] * 1.005
        and candidate_metrics["candle"] <= full_metrics["candle"] * 1.01
        and candidate_metrics["leg"] <= full_metrics["leg"] * 1.01
        and candidate_metrics["log_loss"] <= full_metrics["log_loss"] + 0.001
        and candidate_metrics["brier"] <= full_metrics["brier"] + 0.0005
    )


def decide_mode_gate(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the most accurate eligible preserved-exposure arm."""
    references: dict[int, Mapping[str, Any]] = {}
    candidates: dict[str, dict[int, Mapping[str, Any]]] = {mode: {} for mode in _MODES}
    seen: set[tuple[int, str, str]] = set()
    for result in results:
        _validate_complete_result(result, "mode")
        seed = int(result.get("seed", -1))
        recipe = str(result.get("recipe", ""))
        mode = str(result.get("mode", ""))
        key = (seed, recipe, mode)
        if key in seen:
            raise FoundationMatrixError(f"duplicate matrix cell: {key}")
        seen.add(key)
        if recipe == "3tf" and mode == "full_finetune":
            references[seed] = result
        elif recipe == "4tf" and mode in candidates:
            candidates[mode][seed] = result
        else:
            raise FoundationMatrixError(f"unexpected mode matrix cell: {key}")
    seeds = (42, 43, 44)
    if set(references) != set(seeds) or any(
        set(values) != set(seeds) for values in candidates.values()
    ):
        raise FoundationMatrixError("every expected mode matrix cell must be complete")
    reference_plan_digests = {str(value["plan_digest"]) for value in references.values()}
    candidate_plan_digests = {
        str(value["plan_digest"]) for mode in candidates.values() for value in mode.values()
    }
    if len(reference_plan_digests) != 1 or len(candidate_plan_digests) != 1:
        raise FoundationMatrixError("mode results do not have stable plan lineage")
    all_results = [
        *references.values(),
        *(item for group in candidates.values() for item in group.values()),
    ]
    matrix_names = {str(value["matrix_name"]) for value in all_results}
    base_digests = {str(value["base_config_digest"]) for value in all_results}
    contract_digests = {str(value["matrix_contract_digest"]) for value in all_results}
    if len(matrix_names) != 1 or len(base_digests) != 1 or len(contract_digests) != 1:
        raise FoundationMatrixError("mode results belong to different matrices")
    upstream_digests = {
        str(value["upstream_decision_digest"])
        for group in candidates.values()
        for value in group.values()
    }
    if (
        any(value.get("plan_phase") != "initial" for value in references.values())
        or any(
            value.get("plan_phase") != "mode"
            for group in candidates.values()
            for value in group.values()
        )
        or len(upstream_digests) != 1
        or not re.fullmatch(r"[0-9a-f]{64}", next(iter(upstream_digests)))
    ):
        raise FoundationMatrixError("mode result lineage is invalid")
    summaries = {
        mode: _paired_summary(references, values, seeds, 2) for mode, values in candidates.items()
    }
    full = summaries["full_finetune"]["median_metrics"]
    for mode, summary in summaries.items():
        if mode.startswith("lora_"):
            metrics = summary["median_metrics"]
            summary["lora_noninferior"] = _lora_noninferior(metrics, full)
        else:
            summary["lora_noninferior"] = None
    eligible = [
        mode
        for mode, summary in summaries.items()
        if summary["passed"] and (not mode.startswith("lora_") or summary["lora_noninferior"])
    ]
    selected = min(
        eligible,
        key=lambda mode: (summaries[mode]["median_metrics"]["total"], _MODES.index(mode)),
        default=None,
    )
    core: dict[str, Any] = {
        "schema_version": 1,
        "gate": "mode_screen",
        "decision": "select" if selected is not None else "reject",
        "selected_mode": selected,
        "modes": summaries,
        "reused_results": {
            str(seed): str(references[seed].get("result_digest", "")) for seed in seeds
        },
        "reference_plan_digest": next(iter(reference_plan_digests)),
        "candidate_plan_digest": next(iter(candidate_plan_digests)),
        "candidate_result_digests": {
            f"{mode}:{seed}": candidates[mode][seed]["result_digest"]
            for mode in _MODES
            for seed in seeds
        },
        "matrix_name": next(iter(matrix_names)),
        "base_config_digest": next(iter(base_digests)),
        "matrix_contract_digest": next(iter(contract_digests)),
        "promotion_decision_digest": next(iter(upstream_digests)),
    }
    return {**core, "decision_digest": _digest(core)}


def render_confirmation_plan(
    config_path: str | Path, selection: Mapping[str, Any], output: str | Path
) -> Path:
    selected_mode = selection.get("selected_mode")
    reused = selection.get("reused_results")
    decision_core = {key: value for key, value in selection.items() if key != "decision_digest"}
    if (
        selection.get("schema_version") != 1
        or selection.get("gate") != "mode_screen"
        or selection.get("decision") != "select"
        or selected_mode not in _MODES
        or selection.get("decision_digest") != _digest(decision_core)
        or not isinstance(reused, dict)
        or set(reused) != {"42", "43", "44"}
        or any(not isinstance(value, str) or len(value) != 64 for value in reused.values())
        or not isinstance(selection.get("reference_plan_digest"), str)
        or not isinstance(selection.get("candidate_plan_digest"), str)
        or not isinstance(selection.get("candidate_result_digests"), dict)
    ):
        raise FoundationMatrixError("confirmation plan requires a complete mode selection")
    matrix, base = _load(config_path)
    if (
        selection.get("matrix_name") != matrix["name"]
        or selection.get("base_config_digest") != base.digest
        or selection.get("matrix_contract_digest") != _matrix_contract_digest(matrix)
    ):
        raise FoundationMatrixError("mode decision belongs to another matrix")
    cells: list[dict[str, Any]] = []
    configs: list[tuple[str, dict[str, Any]]] = []
    candidate_result_digests = selection["candidate_result_digests"]
    reused_full_results = (
        {str(seed): candidate_result_digests[f"full_finetune:{seed}"] for seed in (42, 43, 44)}
        if str(selected_mode).startswith("lora_")
        else {}
    )
    for seed in (45, 46):
        recipes = [
            (
                "3tf",
                tuple(matrix["three_timeframes"]),
                "full_finetune",
                matrix["screen_train_steps"],
                matrix["screen_validation_steps"],
            ),
            (
                "4tf",
                tuple(matrix["four_timeframes"]),
                selected_mode,
                matrix["preserved_train_steps"],
                matrix["preserved_validation_steps"],
            ),
        ]
        if str(selected_mode).startswith("lora_"):
            recipes.append(
                (
                    "4tf",
                    tuple(matrix["four_timeframes"]),
                    "full_finetune",
                    matrix["preserved_train_steps"],
                    matrix["preserved_validation_steps"],
                )
            )
        for recipe, intervals, mode, train_steps, validation_steps in recipes:
            cell, payload = _cell(
                matrix,
                base,
                phase="confirmation",
                recipe=recipe,
                intervals=intervals,
                mode=mode,
                seed=seed,
                train_steps=train_steps,
                validation_steps=validation_steps,
            )
            cells.append(cell)
            configs.append((cell["config_path"], payload))
    core = {
        "schema_version": 1,
        "matrix_name": matrix["name"],
        "phase": "confirmation",
        "selection_decision_digest": selection["decision_digest"],
        "selected_mode": selected_mode,
        "reused_results": reused,
        "reused_full_results": reused_full_results,
        "base_config": "base-config.toml",
        "base_config_sha256": hashlib.sha256(
            Path(str(matrix["base_config"])).read_bytes()
        ).hexdigest(),
        "base_config_digest": base.digest,
        "runtime": _runtime_contract(matrix),
        "families": matrix["families"],
        "diagnostic": matrix["diagnostic"],
        "cells": cells,
    }
    return _publish_plan(output, core, configs, Path(str(matrix["base_config"])))


def decide_confirmation_gate(
    results: Iterable[Mapping[str, Any]], selection: Mapping[str, Any]
) -> dict[str, Any]:
    selection_core = {key: value for key, value in selection.items() if key != "decision_digest"}
    selected_mode = selection.get("selected_mode")
    if (
        selection.get("schema_version") != 1
        or selection.get("gate") != "mode_screen"
        or selection.get("decision") != "select"
        or selection.get("decision_digest") != _digest(selection_core)
    ):
        raise FoundationMatrixError("confirmation requires the exact mode selection")
    if selected_mode not in _MODES:
        raise FoundationMatrixError("confirmation selected mode is invalid")
    references: dict[int, Mapping[str, Any]] = {}
    candidates: dict[int, Mapping[str, Any]] = {}
    full_candidates: dict[int, Mapping[str, Any]] = {}
    seen: set[tuple[int, str, str]] = set()
    for result in results:
        _validate_complete_result(result, "confirmation")
        seed = int(result.get("seed", -1))
        recipe = str(result.get("recipe", ""))
        mode = str(result.get("mode", ""))
        key = (seed, recipe, mode)
        if key in seen:
            raise FoundationMatrixError(f"duplicate matrix cell: {key}")
        seen.add(key)
        if recipe == "3tf" and mode == "full_finetune":
            references[seed] = result
        elif recipe == "4tf" and mode == selected_mode:
            candidates[seed] = result
        elif selected_mode.startswith("lora_") and recipe == "4tf" and mode == "full_finetune":
            full_candidates[seed] = result
        else:
            raise FoundationMatrixError(f"unexpected confirmation matrix cell: {key}")
    seeds = (42, 43, 44, 45, 46)
    all_results = [*references.values(), *candidates.values(), *full_candidates.values()]
    contract_digests = {str(result["matrix_contract_digest"]) for result in all_results}
    if len(contract_digests) != 1 or next(iter(contract_digests)) != selection.get(
        "matrix_contract_digest"
    ):
        raise FoundationMatrixError("confirmation results belong to different matrix contracts")
    for seed in (45, 46):
        new_results = [references[seed], candidates[seed]]
        if selected_mode.startswith("lora_"):
            new_results.append(full_candidates[seed])
        if any(
            result.get("plan_phase") != "confirmation"
            or result.get("upstream_decision_digest") != selection["decision_digest"]
            for result in new_results
        ):
            raise FoundationMatrixError("confirmation result lineage differs from mode selection")
    reused = selection.get("reused_results")
    selected_results = selection.get("candidate_result_digests")
    if not isinstance(reused, dict) or not isinstance(selected_results, dict):
        raise FoundationMatrixError("mode selection reused-result lineage is incomplete")
    for seed in (42, 43, 44):
        if (
            reused.get(str(seed)) != references[seed].get("result_digest")
            or selected_results.get(f"{selected_mode}:{seed}")
            != candidates[seed].get("result_digest")
            or (
                selected_mode.startswith("lora_")
                and selected_results.get(f"full_finetune:{seed}")
                != full_candidates[seed].get("result_digest")
            )
        ):
            raise FoundationMatrixError("confirmation reused result differs from mode selection")
    summary = _paired_summary(references, candidates, seeds, 4)
    if selected_mode.startswith("lora_"):
        full_summary = _paired_summary(references, full_candidates, seeds, 4)
        lora_noninferior = _lora_noninferior(
            summary["median_metrics"], full_summary["median_metrics"]
        )
        summary["lora_noninferior"] = lora_noninferior
        summary["full_finetune_median_metrics"] = full_summary["median_metrics"]
        summary["passed"] = bool(summary["passed"] and lora_noninferior)
    else:
        summary["lora_noninferior"] = None
    selected_seed = min(
        seeds,
        key=lambda seed: (
            _finite_number(
                candidates[seed]["metrics"]["by_timeframe"]["3min"]["total"],
                "candidate total",
            ),
            _finite_number(candidates[seed]["diagnostic"]["log_loss"], "candidate log loss"),
            _finite_number(candidates[seed]["diagnostic"]["brier"], "candidate Brier"),
            seed,
        ),
    )
    selected_cell_id = candidates[selected_seed].get("cell_id")
    if not isinstance(selected_cell_id, str) or not re.fullmatch(r"[0-9a-f]{64}", selected_cell_id):
        raise FoundationMatrixError("confirmation candidate cell identity is invalid")
    core: dict[str, Any] = {
        "schema_version": 1,
        "gate": "confirmation",
        "selected_mode": selected_mode,
        "selection_decision_digest": selection["decision_digest"],
        "result_digests": {
            f"{seed}:{result['recipe']}:{result['mode']}": result["result_digest"]
            for seed in seeds
            for result in (
                references[seed],
                candidates[seed],
                *([full_candidates[seed]] if selected_mode.startswith("lora_") else []),
            )
        },
        "plan_digests": sorted(
            {
                str(result["plan_digest"])
                for result in (
                    *references.values(),
                    *candidates.values(),
                    *full_candidates.values(),
                )
            }
        ),
        "decision": "promote" if summary["passed"] else "reject",
        "selected_cell_id": selected_cell_id if summary["passed"] else None,
        "selected_seed": selected_seed if summary["passed"] else None,
        **summary,
    }
    return {**core, "decision_digest": _digest(core)}


def _load_plan(path: str | Path) -> tuple[Path, dict[str, Any]]:
    plan_path = Path(path)
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FoundationMatrixError("matrix plan is unreadable") from exc
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise FoundationMatrixError("matrix plan schema is invalid")
    core = {key: value for key, value in plan.items() if key != "plan_digest"}
    if plan.get("plan_digest") != _digest(core) or plan_path.parent.name != plan.get("plan_digest"):
        raise FoundationMatrixError("matrix plan digest mismatch")
    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells:
        raise FoundationMatrixError("matrix plan has no cells")
    base_path = Path(str(plan.get("base_config", "")))
    if base_path.is_absolute() or ".." in base_path.parts:
        raise FoundationMatrixError("matrix plan base config path is not portable")
    base_path = plan_path.parent / base_path
    if not base_path.is_file() or hashlib.sha256(base_path.read_bytes()).hexdigest() != plan.get(
        "base_config_sha256"
    ):
        raise FoundationMatrixError("matrix plan base config hash mismatch")
    return plan_path, plan


def _planned_cell(
    plan_path: Path, plan: Mapping[str, Any], cell_id: str
) -> tuple[dict[str, Any], PipelineConfig]:
    matches = [cell for cell in plan["cells"] if cell.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise FoundationMatrixError("matrix cell identity is missing or ambiguous")
    cell = matches[0]
    base = load_config(plan_path.parent / str(plan["base_config"]))
    runtime = plan.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "artifact_root",
        "data_root",
        "corpus_manifest_path",
        "device",
    }:
        raise FoundationMatrixError("matrix runtime contract is missing")
    runtime_paths = [
        Path(str(runtime[field]))
        for field in ("artifact_root", "data_root", "corpus_manifest_path")
    ]
    if runtime.get("device") != "cuda" or any(
        not path.is_absolute() or not path.is_relative_to("/workspace/mantis")
        for path in runtime_paths
    ):
        raise FoundationMatrixError("matrix runtime contract is invalid")
    config = replace(
        base,
        run=replace(
            base.run,
            name=str(cell["run_name"]),
            seed=int(cell["seed"]),
            artifact_root=runtime_paths[0],
            device="cuda",
            require_accelerator=True,
            allow_overwrite=False,
        ),
        data=replace(
            base.data,
            root=str(runtime_paths[1]),
            corpus_manifest_path=runtime_paths[2],
            intervals=tuple(cell["intervals"]),
        ),
        model=replace(base.model, mode=str(cell["mode"])),  # type: ignore[arg-type]
        training=replace(
            base.training,
            batch_size=int(cell["batch_size"]),
            max_steps_per_epoch=int(cell["train_steps"]),
            validation_max_steps=int(cell["validation_steps"]),
            resume=True,
        ),
    )
    config_path = plan_path.parent / str(cell["config_path"])
    try:
        recorded = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FoundationMatrixError("matrix cell config is unreadable") from exc
    if recorded != _jsonable(asdict(config)) or config.digest != cell.get("config_digest"):
        raise FoundationMatrixError("matrix cell config digest mismatch")
    return cell, config


def validate_planned_cell(
    plan_path: str | Path, cell_id: str, experiment_config_path: str | Path
) -> dict[str, Any]:
    """Validate one portable plan/config pair without starting training."""
    path, plan = _load_plan(plan_path)
    cell, config = _planned_cell(path, plan, cell_id)
    expected_config = (path.parent / str(cell["config_path"])).resolve()
    if expected_config != Path(experiment_config_path).resolve():
        raise FoundationMatrixError("workload experiment config is not the planned cell config")
    return {
        "plan_digest": plan["plan_digest"],
        "cell_id": cell["cell_id"],
        "run_name": cell["run_name"],
        "config_digest": config.digest,
        "matrix_name": plan["matrix_name"],
    }


def run_matrix_cell(
    plan_path: str | Path,
    cell_id: str,
    *,
    trainer: Callable[[PipelineConfig], Mapping[str, Any]] | None = None,
    releaser: Callable[[PipelineConfig], Mapping[str, Any]] | None = None,
) -> Path:
    """Run or same-cell resume exactly one planned cell and publish a terminal receipt."""
    path, plan = _load_plan(plan_path)
    cell, config = _planned_cell(path, plan, cell_id)
    receipt_path = path.parent / "cells" / cell_id / "foundation-result.json"
    if receipt_path.exists():
        try:
            existing = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise FoundationMatrixError("matrix cell receipt is invalid") from exc
        if existing.get("cell_id") != cell_id or existing.get("config_digest") != config.digest:
            raise FoundationMatrixError("matrix cell receipt identity mismatch")
        if existing.get("status") == "foundation_complete":
            return receipt_path
        if existing.get("status") == "failed":
            raise FoundationMatrixError("failed matrix cell is terminal and cannot be retried")
        raise FoundationMatrixError("matrix cell receipt status is invalid")
    if trainer is None or releaser is None:
        from mantis_v2.pipeline import train, validated_export

        trainer = train
        releaser = validated_export
    try:
        training = dict(trainer(config))
        released = dict(releaser(config))
        evaluation = released.get("evaluation")
        export = released.get("export")
        if not isinstance(evaluation, dict) or not isinstance(export, dict):
            raise FoundationMatrixError("matrix cell release artifacts are incomplete")
        if (
            export.get("export_role") != "diagnostic_candidate"
            or export.get("parity", {}).get("verified") is not True
        ):
            raise FoundationMatrixError(
                "matrix cell export is not a parity-verified diagnostic candidate"
            )
        from mantis_v2.pipeline import artifact_root

        manifest_path = artifact_root(config) / "export" / "manifest.json"
        if not manifest_path.is_file():
            raise FoundationMatrixError("matrix cell export manifest is missing")
        receipt_core: dict[str, Any] = {
            "schema_version": 1,
            "status": "foundation_complete",
            "plan_digest": plan["plan_digest"],
            "cell_id": cell_id,
            "run_name": config.run.name,
            "seed": cell["seed"],
            "recipe": cell["recipe"],
            "mode": cell["mode"],
            "config_digest": config.digest,
            "training": training,
            "evaluation": evaluation,
            "export": export,
            "export_manifest": str(manifest_path),
            "export_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
        _atomic_json_idempotent(
            receipt_path,
            {**receipt_core, "receipt_digest": _digest(receipt_core)},
        )
        return receipt_path
    except Exception as exc:
        failure_core = {
            "schema_version": 1,
            "status": "failed",
            "cell_id": cell_id,
            "run_name": config.run.name,
            "config_digest": config.digest,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }
        _atomic_json_idempotent(
            receipt_path,
            {**failure_core, "receipt_digest": _digest(failure_core)},
        )
        raise


def finalize_cell_result(
    plan_path: str | Path,
    cell_id: str,
    foundation_receipt: str | Path,
    diagnostic_score: str | Path,
) -> Path:
    """Bind one foundation export to its frozen proper-score result."""
    path, plan = _load_plan(plan_path)
    cell, _ = _planned_cell(path, plan, cell_id)
    try:
        foundation = json.loads(Path(foundation_receipt).read_text())
        diagnostic = json.loads(Path(diagnostic_score).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FoundationMatrixError("cell finalization input is unreadable") from exc
    if (
        not isinstance(foundation, dict)
        or foundation.get("status") != "foundation_complete"
        or foundation.get("cell_id") != cell_id
        or foundation.get("plan_digest") != plan["plan_digest"]
    ):
        raise FoundationMatrixError("foundation receipt identity mismatch")
    foundation_core = {key: value for key, value in foundation.items() if key != "receipt_digest"}
    if foundation.get("receipt_digest") != _digest(foundation_core):
        raise FoundationMatrixError("foundation receipt digest mismatch")
    diagnostic_core = {key: value for key, value in diagnostic.items() if key != "result_digest"}
    if (
        diagnostic.get("schema_version") != 1
        or diagnostic.get("seed") != cell["seed"]
        or diagnostic.get("fixture_digest") != plan["diagnostic"]["fixture_digest"]
        or diagnostic.get("result_digest") != _digest(diagnostic_core)
    ):
        raise FoundationMatrixError("diagnostic result identity mismatch")
    candidate = diagnostic.get("candidate")
    reference = diagnostic.get("reference")
    if not isinstance(candidate, dict) or not isinstance(reference, dict):
        raise FoundationMatrixError("diagnostic candidate or reference metrics are missing")
    if diagnostic.get("candidate_id") != foundation.get("export_manifest_sha256"):
        raise FoundationMatrixError("diagnostic candidate is not this cell export")
    reference_id = diagnostic.get("reference_id")
    if not isinstance(reference_id, str) or not re.fullmatch(r"[0-9a-f]{64}", reference_id):
        raise FoundationMatrixError("diagnostic reference export identity is invalid")

    def diagnostic_metrics(record: Mapping[str, Any], role: str) -> dict[str, Any]:
        bins = record.get("calibration_bins")
        if not isinstance(bins, list) or len(bins) != 10:
            raise FoundationMatrixError(f"diagnostic {role} calibration bins are missing")
        return {
            key: _finite_number(record.get(key), f"diagnostic {role} {key}")
            for key in ("log_loss", "brier", "roc_auc", "ece")
        } | {"calibration_bins": bins}

    diagnostic_candidate = diagnostic_metrics(candidate, "candidate")
    diagnostic_reference = diagnostic_metrics(reference, "reference")
    evaluation = foundation.get("evaluation")
    metrics = evaluation.get("metrics") if isinstance(evaluation, dict) else None
    by_stream = metrics.get("by_stream") if isinstance(metrics, dict) else None
    if not isinstance(metrics, dict) or not isinstance(by_stream, dict):
        raise FoundationMatrixError("foundation evaluation stream views are missing")
    family_views: dict[str, Any] = {}
    for family, symbols in plan["families"].items():
        members: list[dict[str, Any]] = []
        for symbol in symbols:
            member = by_stream.get(f"{symbol}_3min")
            if not isinstance(member, dict):
                raise FoundationMatrixError("foundation evaluation family streams are incomplete")
            members.append(member)
        family_views[family] = {
            "3min": {
                key: sum(_finite_number(member[key], f"family {key}") for member in members)
                / len(members)
                for key in ("total", "candle", "leg")
            }
        }
    completed_metrics = {**metrics, "families": family_views}
    result_core: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "plan_digest": plan["plan_digest"],
        "matrix_name": plan["matrix_name"],
        "base_config_digest": plan["base_config_digest"],
        "matrix_contract_digest": cell["matrix_contract_digest"],
        "plan_phase": plan["phase"],
        "upstream_decision_digest": plan.get("selection_decision_digest")
        or plan.get("promotion_decision_digest"),
        "cell_id": cell_id,
        "run_name": cell["run_name"],
        "seed": cell["seed"],
        "recipe": cell["recipe"],
        "mode": cell["mode"],
        "config_digest": cell["config_digest"],
        "metrics": completed_metrics,
        "diagnostic": diagnostic_candidate,
        "diagnostic_reference": diagnostic_reference,
        "diagnostic_candidate_id": diagnostic["candidate_id"],
        "diagnostic_reference_id": reference_id,
        "diagnostic_result_digest": diagnostic["result_digest"],
        "fixture_digest": diagnostic.get("fixture_digest"),
        "checkpoint": evaluation.get("checkpoint") if isinstance(evaluation, dict) else None,
        "export_manifest": foundation.get("export_manifest"),
        "export_manifest_sha256": foundation.get("export_manifest_sha256"),
    }
    result = {**result_core, "result_digest": _digest(result_core)}
    result_path = path.parent / "cells" / cell_id / "cell-result.json"
    _atomic_json_idempotent(result_path, result)
    return result_path


def _read_verified_json(path: str | Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FoundationMatrixError(f"{description} is unreadable") from exc
    if not isinstance(value, dict):
        raise FoundationMatrixError(f"{description} must be an object")
    return value


def write_matrix_decision(decision: Mapping[str, Any], output: str | Path) -> Path:
    """Persist one digest-verified matrix gate decision idempotently."""
    payload = dict(decision)
    core = {key: value for key, value in payload.items() if key != "decision_digest"}
    if payload.get("decision_digest") != _digest(core):
        raise FoundationMatrixError("matrix decision digest mismatch")
    path = Path(output)
    _atomic_json_idempotent(path, payload)
    return path


def promote_selected_export(
    decision_path: str | Path,
    cell_result_path: str | Path,
    output_root: str | Path,
) -> Path:
    """Copy a selected diagnostic export into a new immutable promoted bundle."""
    decision = _read_verified_json(decision_path, "promotion decision")
    result = _read_verified_json(cell_result_path, "selected cell result")
    decision_core = {key: value for key, value in decision.items() if key != "decision_digest"}
    result_core = {key: value for key, value in result.items() if key != "result_digest"}
    if (
        decision.get("schema_version") != 1
        or decision.get("decision") != "promote"
        or decision.get("decision_digest") != _digest(decision_core)
    ):
        raise FoundationMatrixError("promotion decision is invalid")
    if (
        result.get("status") != "complete"
        or result.get("result_digest") != _digest(result_core)
        or result.get("cell_id") != decision.get("selected_cell_id")
        or result.get("mode") != decision.get("selected_mode")
    ):
        raise FoundationMatrixError("selected cell result does not match promotion decision")
    result_key = f"{result.get('seed')}:{result.get('recipe')}:{result.get('mode')}"
    result_digests = decision.get("result_digests")
    if not isinstance(result_digests, dict) or result_digests.get(result_key) != result.get(
        "result_digest"
    ):
        raise FoundationMatrixError("selected result digest is absent from promotion decision")
    source_manifest_path = Path(str(result.get("export_manifest", "")))
    if not source_manifest_path.is_file() or hashlib.sha256(
        source_manifest_path.read_bytes()
    ).hexdigest() != result.get("export_manifest_sha256"):
        raise FoundationMatrixError("selected export manifest hash mismatch")
    source = _read_verified_json(source_manifest_path, "selected export manifest")
    if (
        source.get("export_role") != "diagnostic_candidate"
        or source.get("parity", {}).get("verified") is not True
        or source.get("validation_gate", {}).get("verified") is not True
    ):
        raise FoundationMatrixError("selected export has not passed validation and parity")
    weights = Path(str(source.get("weights", "")))
    evaluation = Path(str(source["validation_gate"].get("evaluation", "")))
    if (
        not weights.is_file()
        or hashlib.sha256(weights.read_bytes()).hexdigest() != source.get("weights_sha256")
        or not evaluation.is_file()
        or hashlib.sha256(evaluation.read_bytes()).hexdigest()
        != source["validation_gate"].get("evaluation_sha256")
    ):
        raise FoundationMatrixError("selected export artifacts changed after validation")
    source_lora = source.get("lora")
    adapter: Path | None = None
    adapter_sha256: str | None = None
    if source_lora is not None:
        if (
            not isinstance(source_lora, dict)
            or source.get("parity", {}).get("lora_merge_verified") is not True
            or source.get("parity", {}).get("lora_adapter_reload_verified") is not True
        ):
            raise FoundationMatrixError("selected LoRA export parity evidence is incomplete")
        adapter = Path(str(source_lora.get("adapter", "")))
        adapter_sha256 = source_lora.get("adapter_sha256")
        if (
            not isinstance(adapter_sha256, str)
            or not adapter.is_file()
            or hashlib.sha256(adapter.read_bytes()).hexdigest() != adapter_sha256
        ):
            raise FoundationMatrixError("selected LoRA adapter changed after validation")
    identity = {
        "schema_version": 1,
        "promotion_decision_digest": decision["decision_digest"],
        "selected_result_digest": result["result_digest"],
        "source_manifest_sha256": result["export_manifest_sha256"],
        "weights_sha256": source["weights_sha256"],
        "evaluation_sha256": source["validation_gate"]["evaluation_sha256"],
        "adapter_sha256": adapter_sha256,
    }
    bundle_digest = _digest(identity)
    destination = Path(output_root) / bundle_digest
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        existing = _read_verified_json(manifest_path, "promoted manifest")
        existing_weights = destination / "model.safetensors"
        existing_evaluation = destination / "evaluation.json"
        if (
            existing.get("bundle_digest") != bundle_digest
            or not existing_weights.is_file()
            or hashlib.sha256(existing_weights.read_bytes()).hexdigest()
            != identity["weights_sha256"]
            or not existing_evaluation.is_file()
            or hashlib.sha256(existing_evaluation.read_bytes()).hexdigest()
            != identity["evaluation_sha256"]
        ):
            raise FoundationMatrixError("existing promoted bundle identity mismatch")
        if adapter_sha256 is not None:
            existing_adapter = destination / "adapter.safetensors"
            if (
                not existing_adapter.is_file()
                or hashlib.sha256(existing_adapter.read_bytes()).hexdigest() != adapter_sha256
            ):
                raise FoundationMatrixError("existing promoted LoRA adapter hash mismatch")
        return manifest_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle_digest}.", dir=destination.parent))
    try:
        copied_weights = temporary / "model.safetensors"
        copied_evaluation = temporary / "evaluation.json"
        shutil.copyfile(weights, copied_weights)
        shutil.copyfile(evaluation, copied_evaluation)
        copied_adapter: Path | None = None
        if adapter is not None:
            copied_adapter = temporary / "adapter.safetensors"
            shutil.copyfile(adapter, copied_adapter)
        if (
            hashlib.sha256(copied_weights.read_bytes()).hexdigest() != source["weights_sha256"]
            or hashlib.sha256(copied_evaluation.read_bytes()).hexdigest()
            != source["validation_gate"]["evaluation_sha256"]
        ):
            raise FoundationMatrixError("promoted export copy verification failed")
        if (
            copied_adapter is not None
            and hashlib.sha256(copied_adapter.read_bytes()).hexdigest() != adapter_sha256
        ):
            raise FoundationMatrixError("promoted LoRA adapter copy verification failed")
        promoted = {
            **source,
            "export_role": "promoted",
            "weights": str(destination / "model.safetensors"),
            "validation_gate": {
                **source["validation_gate"],
                "evaluation": str(destination / "evaluation.json"),
            },
            **identity,
            "bundle_digest": bundle_digest,
        }
        if source_lora is not None:
            promoted["lora"] = {
                **source_lora,
                "adapter": str(destination / "adapter.safetensors"),
            }
        _atomic_json_idempotent(temporary / "manifest.json", promoted)
        try:
            temporary.rename(destination)
        except FileExistsError:
            if not manifest_path.is_file():
                raise FoundationMatrixError(
                    "promoted bundle publication raced incompletely"
                ) from None
        return manifest_path
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
