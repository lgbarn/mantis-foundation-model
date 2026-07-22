from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mantis_v2 import cli
from mantis_v2 import downstream_portable as portable_module
from mantis_v2.downstream_config import load_downstream_config
from mantis_v2.downstream_pipeline import _manifest_base
from mantis_v2.downstream_portable import PortableDownstreamError, run_stage
from mantis_v2.model import sha256_file

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "trend-magic-topstep-100k-portable-template.toml"


def _write_promoted_foundation(tmp_path: Path) -> Path:
    weights_sha256 = hashlib.sha256(b"promoted-foundation").hexdigest()
    evaluation_sha256 = hashlib.sha256(b"{}").hexdigest()
    identity = {
        "schema_version": 1,
        "promotion_decision_digest": "b" * 64,
        "selected_result_digest": "c" * 64,
        "source_manifest_sha256": "d" * 64,
        "weights_sha256": weights_sha256,
        "evaluation_sha256": evaluation_sha256,
        "adapter_sha256": None,
    }
    bundle_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    root = tmp_path / "promoted" / bundle_digest
    root.mkdir(parents=True)
    weights = root / "model.safetensors"
    weights.write_bytes(b"promoted-foundation")
    evaluation = root / "evaluation.json"
    evaluation.write_text("{}")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "export_role": "promoted",
                "weights": str(weights),
                "weights_sha256": weights_sha256,
                "parity": {"verified": True},
                "validation_gate": {
                    "verified": True,
                    "evaluation": str(evaluation),
                    "evaluation_sha256": evaluation_sha256,
                },
                "promotion_decision_digest": "b" * 64,
                "selected_result_digest": "c" * 64,
                "source_manifest_sha256": "d" * 64,
                "bundle_digest": bundle_digest,
            }
        )
    )
    return manifest


def test_producer_binding_rejects_corrupt_promoted_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    promotion = _write_promoted_foundation(tmp_path)
    promoted = json.loads(promotion.read_text())
    Path(promoted["validation_gate"]["evaluation"]).write_text("corrupt")
    data_root, corpus = _write_corpus_manifest(tmp_path)
    output = tmp_path / "producer.toml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "downstream-bind-producer",
            "--template",
            str(TEMPLATE),
            "--promotion-manifest",
            str(promotion),
            "--corpus-manifest",
            str(corpus),
            "--data-root",
            str(data_root),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--run-name",
            "producer",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main()
    assert not output.exists()


def _write_corpus_manifest(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "corpus" / "market"
    root.mkdir(parents=True)
    outputs = []
    for symbol in ("ES", "NQ", "RTY", "YM", "GC", "CL", "ZB"):
        for timeframe in ("1min", "3min", "5min", "15min"):
            path = root / f"{symbol}_{timeframe}.parquet"
            pd.DataFrame({"value": [1.0]}).to_parquet(path, index=False)
            outputs.append(
                {
                    "kind": "market",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "path": str(path.relative_to(root.parent)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "rows": 1,
                }
            )
    manifest = root.parent / "manifest.json"
    payload = {"schema_version": 1, "validated": True, "outputs": outputs}
    payload["manifest_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest.write_text(json.dumps(payload))
    return root, manifest


def test_producer_binding_rejects_a_corrupt_corpus_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    promotion = _write_promoted_foundation(tmp_path)
    data_root, corpus = _write_corpus_manifest(tmp_path)
    (data_root / "NQ_3min.parquet").write_bytes(b"corrupt")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "downstream-bind-producer",
            "--template",
            str(TEMPLATE),
            "--promotion-manifest",
            str(promotion),
            "--corpus-manifest",
            str(corpus),
            "--data-root",
            str(data_root),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--run-name",
            "producer",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "producer.toml"),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main()


def test_cli_binds_exact_portable_producer_and_consumer_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    promotion = _write_promoted_foundation(tmp_path)
    relocated_root = tmp_path / "relocated" / promotion.parent.name
    relocated_root.parent.mkdir()
    promotion.parent.rename(relocated_root)
    promotion = relocated_root / "manifest.json"
    data_root, corpus = _write_corpus_manifest(tmp_path)
    artifact_root = tmp_path / "artifacts"
    producer_path = tmp_path / "configs" / "producer.toml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "downstream-bind-producer",
            "--template",
            str(TEMPLATE),
            "--promotion-manifest",
            str(promotion),
            "--corpus-manifest",
            str(corpus),
            "--data-root",
            str(data_root),
            "--artifact-root",
            str(artifact_root),
            "--run-name",
            "trend-magic-4tf-producer-seed42",
            "--device",
            "cuda",
            "--output",
            str(producer_path),
        ],
    )

    cli.main()

    producer_result = json.loads(capsys.readouterr().out)
    producer = load_downstream_config(producer_path)
    assert producer_result["stage"] == "bind-producer"
    assert producer.data.timeframes == ("1min", "3min", "5min", "15min")
    assert producer.foundation.export_role == "promoted"
    assert producer.foundation.manifest_path == promotion.resolve()
    assert producer.foundation.weights_sha256 == json.loads(promotion.read_text())["weights_sha256"]
    assert producer.evaluation.allow_holdout is False
    binding = json.loads(Path(producer_result["binding_manifest"]).read_text())
    assert binding["producer_config_sha256"] == sha256_file(producer_path)
    assert len(binding["source_digest"]) == 64
    assert len(binding["lock_digest"]) == 64

    feature_path = tmp_path / "features.npy"
    with feature_path.open("wb") as handle:
        np.save(handle, np.zeros((1, 80), dtype=np.float32), allow_pickle=False)
    metadata_path = tmp_path / "metadata.parquet"
    metadata_path.write_bytes(b"metadata")
    embed_manifest = tmp_path / "embed-manifest.json"
    embed_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "embed",
                "workflow_digest": producer.workflow_digest,
                "strategy_contract": producer.strategy_contract,
                "foundation_weights_sha256": producer.foundation.weights_sha256,
                "embedding_dim_per_channel": 4,
                "feature_width": 80,
                "rows": 1,
                "outputs": [
                    {
                        "number": 0,
                        "rows": 1,
                        "features": {
                            "path": str(feature_path),
                            "sha256": sha256_file(feature_path),
                        },
                        "metadata": {
                            "path": str(metadata_path),
                            "sha256": sha256_file(metadata_path),
                        },
                    }
                ],
            }
        )
    )
    consumer_path = tmp_path / "configs" / "consumer.toml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "downstream-bind-consumer",
            "--template",
            str(TEMPLATE),
            "--producer-config",
            str(producer_path),
            "--embed-manifest",
            str(embed_manifest),
            "--run-name",
            "trend-magic-4tf-head-seed42",
            "--output",
            str(consumer_path),
        ],
    )

    cli.main()

    consumer_result = json.loads(capsys.readouterr().out)
    consumer = load_downstream_config(consumer_path)
    assert consumer_result["stage"] == "bind-consumer"
    assert consumer.run.name != producer.run.name
    assert consumer.embedding_contract_digest == producer.embedding_contract_digest
    assert consumer.walk_forward.embed_manifest_sha256 == sha256_file(embed_manifest)
    assert consumer.walk_forward.embed_producer_config_sha256 == sha256_file(producer_path)


def test_consumer_binding_rejects_rejected_three_timeframe_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    promotion = _write_promoted_foundation(tmp_path)
    data_root, corpus = _write_corpus_manifest(tmp_path)
    producer_path = tmp_path / "producer.toml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "downstream-bind-producer",
            "--template",
            str(TEMPLATE),
            "--promotion-manifest",
            str(promotion),
            "--corpus-manifest",
            str(corpus),
            "--data-root",
            str(data_root),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--run-name",
            "producer",
            "--device",
            "cpu",
            "--output",
            str(producer_path),
        ],
    )
    cli.main()
    rejected = tmp_path / "rejected.json"
    rejected.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "embed",
                "workflow_digest": "0" * 64,
                "foundation_weights_sha256": "0" * 64,
                "embedding_dim_per_channel": 4,
                "feature_width": 60,
                "rows": 1,
                "outputs": [{"rows": 1}],
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "downstream-bind-consumer",
            "--template",
            str(TEMPLATE),
            "--producer-config",
            str(producer_path),
            "--embed-manifest",
            str(rejected),
            "--run-name",
            "consumer",
            "--output",
            str(tmp_path / "consumer.toml"),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main()


def test_portable_stage_resumes_only_an_exact_completed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "portable.toml"
    config_path.write_text(
        TEMPLATE.read_text().replace("/workspace/artifacts", str(tmp_path / "artifacts"))
    )
    config = load_downstream_config(config_path)
    stage_root = config.run.artifact_root / config.run.name / "prepare"
    stage_root.mkdir(parents=True, exist_ok=True)
    manifest_path = stage_root / "manifest.json"
    manifest_path.write_text(json.dumps({**_manifest_base(config, "prepare"), "rows": 17}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "downstream-portable-stage",
            "--config",
            str(config_path),
            "--stage",
            "prepare",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main()

    manifest_path.unlink()
    inputs = []
    for number in range(len(config.data.symbols) * len(config.data.timeframes)):
        path = stage_root / f"input-{number}.parquet"
        path.write_bytes(str(number).encode())
        inputs.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    outputs = []
    for symbol in config.data.symbols:
        path = stage_root / f"{symbol}.parquet"
        path.write_bytes(symbol.encode())
        outputs.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "symbol": symbol,
                "rows": 1,
            }
        )
    contamination = stage_root / "contamination.json"
    contamination.write_text("{}")
    manifest_path.write_text(
        json.dumps(
            {
                **_manifest_base(config, "prepare"),
                "inputs": inputs,
                "outputs": outputs,
                "rows": len(outputs),
                "holdout_locked": True,
                "contamination": {
                    "path": str(contamination),
                    "size": contamination.stat().st_size,
                    "sha256": sha256_file(contamination),
                },
            }
        )
    )

    cli.main()

    result = json.loads(capsys.readouterr().out)
    assert result["resumed"] is True
    assert result["manifest"]["rows"] == len(outputs)

    changed = load_downstream_config(config_path, ('run.name="changed"',))
    changed_root = changed.run.artifact_root / changed.run.name / "prepare"
    changed_root.mkdir(parents=True, exist_ok=True)
    (changed_root / "manifest.json").write_text(manifest_path.read_text())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mantis-v2",
            "downstream-portable-stage",
            "--config",
            str(config_path),
            "--set",
            'run.name="changed"',
            "--stage",
            "prepare",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        cli.main()


def test_portable_head_stage_raises_after_durable_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "portable.toml"
    config_path.write_text(
        TEMPLATE.read_text().replace("/workspace/artifacts", str(tmp_path / "artifacts"))
    )
    config = load_downstream_config(config_path)
    root = config.run.artifact_root / config.run.name / "walk-forward"

    def failed_head(_config: object) -> dict[str, object]:
        root.mkdir(parents=True)
        head = root / "head.npz"
        predictions = root / "predictions.parquet"
        head.write_bytes(b"head")
        predictions.write_bytes(b"predictions")
        manifest = {
            **_manifest_base(config, "walk-forward"),
            "folds": [
                {
                    "head": {
                        "path": str(head),
                        "size": head.stat().st_size,
                        "sha256": sha256_file(head),
                    },
                    "predictions": {
                        "path": str(predictions),
                        "size": predictions.stat().st_size,
                        "sha256": sha256_file(predictions),
                    },
                }
            ],
            "convergence_gate_passed": False,
            "quality_gate": {"passed": False},
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "failure.json").write_text(json.dumps({"reason": "proper_score_gate_failed"}))
        return manifest

    monkeypatch.setattr(portable_module, "walk_forward", failed_head)

    with pytest.raises(PortableDownstreamError, match="head promotion gates failed"):
        run_stage(config, "head")
    assert (root / "failure.json").is_file()
