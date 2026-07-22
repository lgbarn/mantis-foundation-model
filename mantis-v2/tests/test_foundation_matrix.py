from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from mantis_v2 import pipeline as pipeline_module
from mantis_v2.cli import _parser
from mantis_v2.config import load_config
from mantis_v2.foundation_matrix import (
    FoundationMatrixError,
    _load,
    _matrix_contract_digest,
    decide_confirmation_gate,
    decide_five_minute_gate,
    decide_mode_gate,
    finalize_cell_result,
    promote_selected_export,
    render_confirmation_plan,
    render_initial_plan,
    render_mode_plan,
    run_matrix_cell,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "foundation-matrix-v1.toml"
BASE_DIGEST = load_config(ROOT / "configs" / "nextleg-parquet-v1.toml").digest
MATRIX_CONTRACT_DIGEST = _matrix_contract_digest(_load(CONFIG)[0])
PROMOTION_DECISION_DIGEST = "a" * 64


def test_foundation_matrix_and_diagnostic_commands_are_discoverable() -> None:
    commands = (
        ["foundation-fixture-freeze", "--config", "config.toml", "--output-root", "out"],
        [
            "foundation-fixture-embed",
            "--config",
            "config.toml",
            "--fixture",
            "fixture.json",
            "--foundation-manifest",
            "export.json",
            "--output-root",
            "out",
        ],
        [
            "foundation-diagnostic-score",
            "--fixture",
            "fixture.json",
            "--candidate",
            "candidate.json",
            "--reference",
            "reference.json",
            "--output",
            "score.json",
        ],
        [
            "foundation-matrix-cell",
            "--plan",
            "plan.json",
            "--cell-id",
            "a" * 64,
        ],
    )

    assert [_parser().parse_args(list(argv)).command for argv in commands] == [
        argv[0] for argv in commands
    ]


def _result(
    seed: int,
    recipe: str,
    total: float,
    *,
    log_loss: float,
    brier: float,
    mode: str = "full_finetune",
    plan_digest: str = "9" * 64,
    plan_phase: str = "initial",
    upstream_decision_digest: str | None = None,
) -> dict:
    cell_id = hashlib.sha256(f"{seed}:{recipe}:{mode}".encode()).hexdigest()
    reference_id = hashlib.sha256(f"{seed}:3tf:full_finetune".encode()).hexdigest()
    diagnostic = {
        "log_loss": log_loss,
        "brier": brier,
        "roc_auc": 0.55,
        "ece": 0.05,
        "calibration_bins": [{} for _ in range(10)],
    }
    core = {
        "schema_version": 1,
        "status": "complete",
        "plan_digest": plan_digest,
        "matrix_name": "mantisv2-foundation-accuracy-v1",
        "base_config_digest": BASE_DIGEST,
        "matrix_contract_digest": MATRIX_CONTRACT_DIGEST,
        "plan_phase": plan_phase,
        "upstream_decision_digest": upstream_decision_digest,
        "seed": seed,
        "recipe": recipe,
        "mode": mode,
        "cell_id": cell_id,
        "metrics": {
            "by_timeframe": {"3min": {"total": total, "candle": total / 2, "leg": total / 2}},
            "families": {
                "equity_index": {"3min": {"total": total}},
                "metals": {"3min": {"total": total}},
                "energy": {"3min": {"total": total}},
                "rates": {"3min": {"total": total}},
            },
        },
        "diagnostic": diagnostic,
        "diagnostic_reference": diagnostic,
        "diagnostic_candidate_id": cell_id,
        "diagnostic_reference_id": reference_id,
        "fixture_digest": "8" * 64,
        "export_manifest_sha256": cell_id,
    }
    return {
        **core,
        "result_digest": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _decision(**values) -> dict:
    return {
        **values,
        "decision_digest": hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _confirmation_selection(mode: str, results: list[dict]) -> dict:
    references = {
        result["seed"]: result
        for result in results
        if result["seed"] in (42, 43, 44) and result["recipe"] == "3tf"
    }
    candidates = {
        f"{result['mode']}:{result['seed']}": result["result_digest"]
        for result in results
        if result["seed"] in (42, 43, 44) and result["recipe"] == "4tf"
    }
    return _decision(
        schema_version=1,
        gate="mode_screen",
        decision="select",
        selected_mode=mode,
        matrix_contract_digest=MATRIX_CONTRACT_DIGEST,
        reused_results={str(seed): references[seed]["result_digest"] for seed in references},
        candidate_result_digests=candidates,
    )


def test_initial_plan_is_content_addressed_and_renders_six_unique_cells(tmp_path: Path) -> None:
    plan_path = render_initial_plan(CONFIG, tmp_path)
    plan = json.loads(plan_path.read_text())

    assert plan_path.parent.name == plan["plan_digest"]
    assert plan["base_config"] == "base-config.toml"
    assert (plan_path.parent / plan["base_config"]).is_file()
    assert len(plan["cells"]) == 6
    assert len({cell["run_name"] for cell in plan["cells"]}) == 6
    assert len({cell["config_digest"] for cell in plan["cells"]}) == 6
    assert {(cell["recipe"], cell["seed"]) for cell in plan["cells"]} == {
        (recipe, seed) for recipe in ("3tf", "4tf") for seed in (42, 43, 44)
    }
    for cell in plan["cells"]:
        config_path = plan_path.parent / cell["config_path"]
        assert config_path.is_file()
        config = json.loads(config_path.read_text())
        assert config["training"]["batch_size"] == 128
        assert config["training"]["max_steps_per_epoch"] == 200
        assert config["training"]["validation_max_steps"] == 20
        assert config["model"]["mode"] == "full_finetune"
        assert config["run"]["artifact_root"] == "/workspace/mantis/runs"
        assert config["run"]["device"] == "cuda"
        input_root = "/workspace/mantis/inputs/"
        assert config["data"]["root"].startswith(input_root)
        assert config["data"]["root"].endswith("/corpus/market")
        assert config["data"]["corpus_manifest_path"].startswith(input_root)
        assert config["data"]["corpus_manifest_path"].endswith("/corpus/manifest.json")
        assert "/Volumes/" not in json.dumps(config)
        assert config["data"]["intervals"] == (
            ["1min", "3min", "15min"]
            if cell["recipe"] == "3tf"
            else ["1min", "3min", "5min", "15min"]
        )


def test_mode_plan_is_conditional_and_preserves_four_timeframe_exposure(tmp_path: Path) -> None:
    rejected = _decision(schema_version=1, decision="reject")
    with pytest.raises(FoundationMatrixError, match="promotion gate"):
        render_mode_plan(CONFIG, rejected, tmp_path)

    promoted = decide_five_minute_gate(
        [
            _result(
                seed,
                recipe,
                1.0 if recipe == "3tf" else 0.9,
                log_loss=0.65 if recipe == "3tf" else 0.60,
                brier=0.23 if recipe == "3tf" else 0.20,
            )
            for seed in (42, 43, 44)
            for recipe in ("3tf", "4tf")
        ]
    )
    plan_path = render_mode_plan(CONFIG, promoted, tmp_path)
    plan = json.loads(plan_path.read_text())

    assert len(plan["cells"]) == 12
    assert {cell["mode"] for cell in plan["cells"]} == {
        "full_finetune",
        "transformer_finetune",
        "lora_r8_alpha16",
        "lora_r16_alpha32",
    }
    assert all(cell["training"]["max_steps_per_epoch"] == 267 for cell in plan["cells"])
    assert all(cell["training"]["validation_max_steps"] == 27 for cell in plan["cells"])


def test_five_minute_gate_requires_seed_paired_loss_and_proper_score_wins() -> None:
    results = []
    for seed, reference, candidate in ((42, 1.0, 0.98), (43, 1.0, 0.99), (44, 1.0, 1.01)):
        results.extend(
            (
                _result(seed, "3tf", reference, log_loss=0.65, brier=0.23),
                _result(
                    seed,
                    "4tf",
                    candidate,
                    log_loss=0.64 if seed != 44 else 0.66,
                    brier=0.22 if seed != 44 else 0.24,
                ),
            )
        )

    decision = decide_five_minute_gate(results)

    assert decision["decision"] == "promote"
    assert decision["improving_seeds"] == [42, 43]
    assert decision["proper_score_improving_seeds"] == [42, 43]


def test_gate_blocks_incomplete_failed_or_duplicate_cells() -> None:
    results = [
        _result(seed, recipe, 1.0, log_loss=0.65, brier=0.23)
        for seed in (42, 43, 44)
        for recipe in ("3tf", "4tf")
    ]
    results[-1]["status"] = "failed"
    with pytest.raises(FoundationMatrixError, match="complete"):
        decide_five_minute_gate(results)

    results[-1]["status"] = "complete"
    with pytest.raises(FoundationMatrixError, match="duplicate"):
        decide_five_minute_gate([*results, results[0]])

    tampered = [dict(result) for result in results]
    tampered[0]["diagnostic"] = {"log_loss": 0.1, "brier": 0.1}
    with pytest.raises(FoundationMatrixError, match="provenance"):
        decide_five_minute_gate(tampered)


def test_mode_gate_keeps_accuracy_first_and_enforces_lora_noninferiority() -> None:
    results = []
    for seed in (42, 43, 44):
        results.append(_result(seed, "3tf", 1.0, log_loss=0.66, brier=0.24))
        for mode, total in (
            ("full_finetune", 0.950),
            ("transformer_finetune", 0.960),
            ("lora_r8_alpha16", 0.956),
            ("lora_r16_alpha32", 0.940),
        ):
            results.append(
                _result(
                    seed,
                    "4tf",
                    total,
                    log_loss=0.63,
                    brier=0.21,
                    mode=mode,
                    plan_phase="mode",
                    upstream_decision_digest=PROMOTION_DECISION_DIGEST,
                )
            )

    decision = decide_mode_gate(results)

    assert decision["decision"] == "select"
    assert decision["selected_mode"] == "lora_r16_alpha32"
    assert decision["modes"]["lora_r8_alpha16"]["lora_noninferior"] is False
    assert decision["modes"]["lora_r16_alpha32"]["lora_noninferior"] is True


def test_confirmation_plan_renders_only_new_seed_pairs_and_binds_reused_results(
    tmp_path: Path,
) -> None:
    results = []
    for seed in (42, 43, 44):
        results.append(
            _result(
                seed,
                "3tf",
                1.0,
                log_loss=0.66,
                brier=0.24,
                plan_digest="1" * 64,
            )
        )
        for mode, total in (
            ("full_finetune", 0.96),
            ("transformer_finetune", 0.94),
            ("lora_r8_alpha16", 0.97),
            ("lora_r16_alpha32", 0.98),
        ):
            results.append(
                _result(
                    seed,
                    "4tf",
                    total,
                    log_loss=0.63,
                    brier=0.21,
                    mode=mode,
                    plan_digest="2" * 64,
                    plan_phase="mode",
                    upstream_decision_digest=PROMOTION_DECISION_DIGEST,
                )
            )
    selection = decide_mode_gate(results)

    plan_path = render_confirmation_plan(CONFIG, selection, tmp_path)
    plan = json.loads(plan_path.read_text())

    assert len(plan["cells"]) == 4
    assert {(cell["recipe"], cell["seed"]) for cell in plan["cells"]} == {
        (recipe, seed) for recipe in ("3tf", "4tf") for seed in (45, 46)
    }
    assert plan["reused_results"] == selection["reused_results"]


def test_confirmation_rejects_lora_outside_full_finetune_noninferiority_margin() -> None:
    results = []
    for seed in (42, 43, 44, 45, 46):
        results.extend(
            (
                _result(seed, "3tf", 1.0, log_loss=0.66, brier=0.24),
                _result(
                    seed,
                    "4tf",
                    0.95,
                    log_loss=0.63,
                    brier=0.21,
                    mode="full_finetune",
                ),
                _result(
                    seed,
                    "4tf",
                    0.96,
                    log_loss=0.632,
                    brier=0.211,
                    mode="lora_r8_alpha16",
                ),
            )
        )
    selection = _confirmation_selection("lora_r8_alpha16", results)
    for result in results:
        if result["seed"] in (45, 46):
            result["plan_phase"] = "confirmation"
            result["upstream_decision_digest"] = selection["decision_digest"]
            core = {key: value for key, value in result.items() if key != "result_digest"}
            result["result_digest"] = hashlib.sha256(
                json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()

    decision = decide_confirmation_gate(results, selection)

    assert decision["decision"] == "reject"
    assert decision["lora_noninferior"] is False


def test_lora_confirmation_plan_adds_full_finetune_comparators(tmp_path: Path) -> None:
    results = []
    for seed in (42, 43, 44):
        results.append(_result(seed, "3tf", 1.0, log_loss=0.66, brier=0.24))
        for mode, total in (
            ("full_finetune", 0.95),
            ("transformer_finetune", 0.97),
            ("lora_r8_alpha16", 0.96),
            ("lora_r16_alpha32", 0.949),
        ):
            results.append(
                _result(
                    seed,
                    "4tf",
                    total,
                    log_loss=0.63,
                    brier=0.21,
                    mode=mode,
                    plan_phase="mode",
                    upstream_decision_digest=PROMOTION_DECISION_DIGEST,
                )
            )
    selection = decide_mode_gate(results)
    assert selection["selected_mode"] == "lora_r16_alpha32"

    plan = json.loads(render_confirmation_plan(CONFIG, selection, tmp_path).read_text())

    assert len(plan["cells"]) == 6
    assert {(cell["recipe"], cell["mode"], cell["seed"]) for cell in plan["cells"]} == {
        ("3tf", "full_finetune", seed) for seed in (45, 46)
    } | {("4tf", mode, seed) for mode in ("full_finetune", "lora_r16_alpha32") for seed in (45, 46)}
    assert set(plan["reused_full_results"]) == {"42", "43", "44"}


def test_confirmation_gate_requires_four_of_five_seed_paired_wins() -> None:
    results = []
    for seed in (42, 43, 44, 45, 46):
        results.append(_result(seed, "3tf", 1.0, log_loss=0.66, brier=0.24))
        improves = seed != 46
        results.append(
            _result(
                seed,
                "4tf",
                0.98 if improves else 1.01,
                log_loss=0.64 if improves else 0.67,
                brier=0.22 if improves else 0.25,
                mode="transformer_finetune",
            )
        )

    selection = _confirmation_selection("transformer_finetune", results)
    for result in results:
        if result["seed"] in (45, 46):
            result["plan_phase"] = "confirmation"
            result["upstream_decision_digest"] = selection["decision_digest"]
            core = {key: value for key, value in result.items() if key != "result_digest"}
            result["result_digest"] = hashlib.sha256(
                json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()

    decision = decide_confirmation_gate(results, selection)

    assert decision["decision"] == "promote"
    assert decision["improving_seeds"] == [42, 43, 44, 45]
    assert (
        decision["selected_cell_id"] == hashlib.sha256(b"42:4tf:transformer_finetune").hexdigest()
    )


def test_cell_runner_is_same_cell_resumable_and_failure_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline_module, "artifact_root", lambda config: tmp_path / config.run.name)
    plan_path = render_initial_plan(CONFIG, tmp_path / "plans")
    plan = json.loads(plan_path.read_text())
    cell = plan["cells"][0]
    calls: list[str] = []

    def trainer(config) -> dict:
        calls.append(config.run.name)
        return {"completed": True}

    def releaser(config) -> dict:
        evaluation = {
            "metrics": {
                "micro": {"total": 1.0, "candle": 0.5, "leg": 0.5},
                "macro": {"total": 1.0, "candle": 0.5, "leg": 0.5},
                "by_stream": {
                    f"{symbol}_{timeframe}": {"total": 1.0, "candle": 0.5, "leg": 0.5}
                    for symbol in ("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN")
                    for timeframe in config.data.intervals
                },
                "by_symbol": {},
                "by_timeframe": {
                    timeframe: {"total": 1.0, "candle": 0.5, "leg": 0.5}
                    for timeframe in config.data.intervals
                },
            },
            "checkpoint": {"sha256": "a" * 64, "epoch": 2, "global_step": 600},
        }
        released = {
            "evaluation": evaluation,
            "export": {
                "export_role": "diagnostic_candidate",
                "weights": str(tmp_path / "weights.safetensors"),
                "weights_sha256": "b" * 64,
                "parity": {"verified": True},
            },
        }
        export_root = tmp_path / config.run.name / "export"
        export_root.mkdir(parents=True, exist_ok=True)
        (export_root / "manifest.json").write_text(json.dumps(released["export"]))
        return released

    receipt_path = run_matrix_cell(
        plan_path,
        cell["cell_id"],
        trainer=trainer,
        releaser=releaser,
    )
    assert json.loads(receipt_path.read_text())["status"] == "foundation_complete"
    assert (
        run_matrix_cell(plan_path, cell["cell_id"], trainer=trainer, releaser=releaser)
        == receipt_path
    )
    assert calls == [cell["run_name"]]

    failed_cell = plan["cells"][1]
    with pytest.raises(RuntimeError, match="boom"):
        run_matrix_cell(
            plan_path,
            failed_cell["cell_id"],
            trainer=lambda config: (_ for _ in ()).throw(RuntimeError("boom")),
            releaser=releaser,
        )
    with pytest.raises(FoundationMatrixError, match="terminal"):
        run_matrix_cell(plan_path, failed_cell["cell_id"], trainer=trainer, releaser=releaser)

    malformed_cell = plan["cells"][2]
    with pytest.raises(FoundationMatrixError, match="incomplete"):
        run_matrix_cell(
            plan_path,
            malformed_cell["cell_id"],
            trainer=trainer,
            releaser=lambda config: {},
        )
    with pytest.raises(FoundationMatrixError, match="terminal"):
        run_matrix_cell(plan_path, malformed_cell["cell_id"], trainer=trainer, releaser=releaser)


def test_finalize_binds_diagnostic_and_computes_declared_family_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline_module, "artifact_root", lambda config: tmp_path / config.run.name)
    plan_path = render_initial_plan(CONFIG, tmp_path / "plans")
    plan = json.loads(plan_path.read_text())
    cell = plan["cells"][0]

    def releaser(config) -> dict:
        streams = {
            f"{symbol}_{timeframe}": {"total": float(index + 1), "candle": 0.5, "leg": 0.5}
            for index, symbol in enumerate(("ES", "NQ", "RTY", "YM", "GC", "SI", "CL", "ZB", "ZN"))
            for timeframe in config.data.intervals
        }
        released = {
            "evaluation": {
                "metrics": {
                    "micro": {"total": 1.0, "candle": 0.5, "leg": 0.5},
                    "macro": {"total": 1.0, "candle": 0.5, "leg": 0.5},
                    "by_stream": streams,
                    "by_symbol": {},
                    "by_timeframe": {
                        timeframe: {"total": 1.0, "candle": 0.5, "leg": 0.5}
                        for timeframe in config.data.intervals
                    },
                },
                "checkpoint": {"sha256": "a" * 64, "epoch": 2, "global_step": 600},
            },
            "export": {
                "export_role": "diagnostic_candidate",
                "weights": "/tmp/weights.safetensors",
                "weights_sha256": "b" * 64,
                "parity": {"verified": True},
            },
        }
        export_root = tmp_path / config.run.name / "export"
        export_root.mkdir(parents=True, exist_ok=True)
        (export_root / "manifest.json").write_text(json.dumps(released["export"]))
        return released

    foundation = run_matrix_cell(
        plan_path, cell["cell_id"], trainer=lambda config: {}, releaser=releaser
    )
    foundation_record = json.loads(foundation.read_text())
    export_identity = foundation_record["export_manifest_sha256"]
    diagnostic = tmp_path / "diagnostic.json"
    metrics = {
        "log_loss": 0.64,
        "brier": 0.22,
        "roc_auc": 0.58,
        "ece": 0.04,
        "calibration_bins": [{} for _ in range(10)],
    }
    diagnostic_core = {
        "schema_version": 1,
        "seed": cell["seed"],
        "fixture_digest": plan["diagnostic"]["fixture_digest"],
        "candidate_id": export_identity,
        "reference_id": export_identity,
        "candidate": metrics,
        "reference": metrics,
    }
    diagnostic.write_text(
        json.dumps(
            {
                **diagnostic_core,
                "result_digest": hashlib.sha256(
                    json.dumps(diagnostic_core, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    )

    result_path = finalize_cell_result(plan_path, cell["cell_id"], foundation, diagnostic)
    result = json.loads(result_path.read_text())

    assert result["status"] == "complete"
    assert result["diagnostic"] == metrics
    assert result["metrics"]["families"]["equity_index"]["3min"]["total"] == 2.5
    assert result["checkpoint"]["epoch"] == 2


def test_promotion_copies_and_revalidates_hash_verified_lora_export(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    weights = source / "model.safetensors"
    weights.write_bytes(b"weights")
    evaluation = source / "evaluation.json"
    evaluation.write_text('{"passed":true}')
    adapter = source / "adapter.safetensors"
    adapter.write_bytes(b"adapter-and-heads")
    export_manifest = source / "manifest.json"
    export_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "export_role": "diagnostic_candidate",
                "weights": str(weights),
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "validation_gate": {
                    "verified": True,
                    "evaluation": str(evaluation),
                    "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
                },
                "parity": {
                    "verified": True,
                    "lora_merge_verified": True,
                    "lora_adapter_reload_verified": True,
                },
                "lora": {
                    "adapter": str(adapter),
                    "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                    "includes_trainable_task_heads": True,
                },
            }
        )
    )
    result = tmp_path / "cell-result.json"
    result_core = {
        "schema_version": 1,
        "status": "complete",
        "cell_id": "cell-1",
        "seed": 42,
        "recipe": "4tf",
        "mode": "lora_r8_alpha16",
        "export_manifest": str(export_manifest),
        "export_manifest_sha256": hashlib.sha256(export_manifest.read_bytes()).hexdigest(),
    }
    result.write_text(
        json.dumps(
            {
                **result_core,
                "result_digest": hashlib.sha256(
                    json.dumps(result_core, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    )
    decision = tmp_path / "decision.json"
    decision_core = {
        "schema_version": 1,
        "decision": "promote",
        "selected_mode": "lora_r8_alpha16",
        "selected_cell_id": "cell-1",
        "result_digests": {
            "42:4tf:lora_r8_alpha16": hashlib.sha256(
                json.dumps(result_core, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        },
    }
    decision_digest = hashlib.sha256(
        json.dumps(decision_core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    decision.write_text(json.dumps({**decision_core, "decision_digest": decision_digest}))

    promoted = promote_selected_export(decision, result, tmp_path / "promoted")
    manifest = json.loads(promoted.read_text())

    assert manifest["export_role"] == "promoted"
    assert Path(manifest["weights"]).read_bytes() == b"weights"
    assert Path(manifest["lora"]["adapter"]).read_bytes() == b"adapter-and-heads"
    assert manifest["promotion_decision_digest"] == decision_digest

    Path(manifest["weights"]).write_bytes(b"corrupt")
    with pytest.raises(FoundationMatrixError, match="identity"):
        promote_selected_export(decision, result, tmp_path / "promoted")
