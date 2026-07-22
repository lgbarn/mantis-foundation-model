"""Immutable producer/consumer binding for the portable Trend Magic DAG."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from mantis_v2.downstream_config import DownstreamConfig, load_downstream_config
from mantis_v2.downstream_pipeline import (
    _manifest,
    _source_digest,
    artifact_root,
    embed,
    prepare,
    simulate,
    walk_forward,
)
from mantis_v2.embedding_artifacts import FOUR_TIMEFRAME_CONTRACT
from mantis_v2.model import sha256_file


class PortableDownstreamError(RuntimeError):
    """Raised when a portable downstream identity cannot be bound exactly."""


def _digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PortableDownstreamError(f"{description} is unreadable") from exc
    if not isinstance(value, dict):
        raise PortableDownstreamError(f"{description} must be an object")
    return value


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, datetime):
        return json.dumps(value.isoformat())
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise PortableDownstreamError(f"cannot render TOML value of type {type(value).__name__}")


def _render_config(config: DownstreamConfig) -> str:
    sections = asdict(config)
    lines: list[str] = []
    for section, values in sections.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if value is None or value == "" or value == ():
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def _publish_text(path: Path, content: str, description: str) -> None:
    encoded = content.encode()
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise PortableDownstreamError(f"existing {description} identity mismatch")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def _binding_base() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    return {
        "schema_version": 1,
        "source_digest": _source_digest(),
        "lock_digest": sha256_file(root / "uv.lock"),
        "sealed_holdout_accessed": False,
    }


def _publish_binding(config_path: Path, payload: dict[str, Any]) -> Path:
    binding_path = config_path.with_suffix(".binding.json")
    complete = {**payload, "binding_digest": _digest(payload)}
    _publish_text(
        binding_path,
        json.dumps(complete, indent=2, sort_keys=True) + "\n",
        "binding manifest",
    )
    return binding_path


def _validate_promoted_bundle(path: Path, manifest: dict[str, Any]) -> None:
    validation = manifest.get("validation_gate")
    parity = manifest.get("parity")
    lora = manifest.get("lora")
    if not isinstance(validation, dict) or not isinstance(parity, dict):
        raise PortableDownstreamError("promoted foundation evidence is incomplete")
    weights = Path(str(manifest.get("weights", "")))
    if not weights.is_file():
        weights = path.parent / "model.safetensors"
    evaluation = Path(str(validation.get("evaluation", "")))
    if not evaluation.is_file():
        evaluation = path.parent / "evaluation.json"
    weights_sha256 = manifest.get("weights_sha256")
    evaluation_sha256 = validation.get("evaluation_sha256")
    adapter_sha256: str | None = None
    if lora is not None:
        if not isinstance(lora, dict):
            raise PortableDownstreamError("promoted LoRA evidence is incomplete")
        adapter = Path(str(lora.get("adapter", "")))
        if not adapter.is_file():
            adapter = path.parent / "adapter.safetensors"
        adapter_sha256 = lora.get("adapter_sha256")
        if (
            not isinstance(adapter_sha256, str)
            or parity.get("lora_merge_verified") is not True
            or parity.get("lora_adapter_reload_verified") is not True
            or not adapter.is_file()
            or sha256_file(adapter) != adapter_sha256
        ):
            raise PortableDownstreamError("promoted LoRA adapter identity mismatch")
    identity = {
        "schema_version": 1,
        "promotion_decision_digest": manifest.get("promotion_decision_digest"),
        "selected_result_digest": manifest.get("selected_result_digest"),
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
        "weights_sha256": weights_sha256,
        "evaluation_sha256": evaluation_sha256,
        "adapter_sha256": adapter_sha256,
    }
    required_digests = tuple(
        identity[key]
        for key in (
            "promotion_decision_digest",
            "selected_result_digest",
            "source_manifest_sha256",
            "weights_sha256",
            "evaluation_sha256",
        )
    )
    bundle_digest = manifest.get("bundle_digest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("export_role") != "promoted"
        or parity.get("verified") is not True
        or validation.get("verified") is not True
        or any(not isinstance(value, str) or len(value) != 64 for value in required_digests)
        or not isinstance(bundle_digest, str)
        or _digest(identity) != bundle_digest
        or path.parent.name != bundle_digest
        or not weights.is_file()
        or sha256_file(weights) != weights_sha256
        or not evaluation.is_file()
        or sha256_file(evaluation) != evaluation_sha256
    ):
        raise PortableDownstreamError("foundation export is not an intact promoted bundle")


def _verify_file_identity(
    raw: Any, description: str, *, within: Path | None = None
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PortableDownstreamError(f"{description} identity is incomplete")
    path = Path(str(raw.get("path", "")))
    if (
        not path.is_file()
        or raw.get("size") != path.stat().st_size
        or raw.get("sha256") != sha256_file(path)
        or (within is not None and not path.resolve().is_relative_to(within.resolve()))
    ):
        raise PortableDownstreamError(f"{description} identity mismatch")
    return raw


def _validate_completed_manifest(
    manifest: dict[str, Any], config: DownstreamConfig, stage: str, root: Path
) -> None:
    if stage == "prepare":
        inputs = manifest.get("inputs")
        outputs = manifest.get("outputs")
        expected_inputs = len(config.data.symbols) * len(config.data.timeframes)
        if not isinstance(inputs, list) or len(inputs) != expected_inputs:
            raise PortableDownstreamError("completed prepare manifest has incomplete inputs")
        for number, identity in enumerate(inputs):
            _verify_file_identity(identity, f"prepare input {number}")
        if not isinstance(outputs, list) or {
            output.get("symbol") for output in outputs if isinstance(output, dict)
        } != set(config.data.symbols):
            raise PortableDownstreamError("completed prepare manifest has incomplete outputs")
        rows = 0
        for output in outputs:
            identity = _verify_file_identity(output, "prepare output", within=root)
            if not isinstance(identity.get("rows"), int) or identity["rows"] <= 0:
                raise PortableDownstreamError("prepare output row count is invalid")
            rows += int(identity["rows"])
        _verify_file_identity(manifest.get("contamination"), "contamination", within=root)
        if manifest.get("rows") != rows or manifest.get("holdout_locked") is not True:
            raise PortableDownstreamError("completed prepare manifest summary mismatch")
        return
    if stage == "embed":
        outputs = manifest.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise PortableDownstreamError("completed embed manifest has no shards")
        rows = 0
        for output in outputs:
            if not isinstance(output, dict) or not isinstance(output.get("rows"), int):
                raise PortableDownstreamError("embedding shard identity is incomplete")
            _verify_file_identity(output.get("features"), "embedding features", within=root)
            _verify_file_identity(output.get("metadata"), "embedding metadata", within=root)
            rows += int(output["rows"])
        if manifest.get("rows") != rows or not isinstance(manifest.get("feature_width"), int):
            raise PortableDownstreamError("completed embed manifest summary mismatch")
        return
    if stage == "head":
        folds = manifest.get("folds")
        quality = manifest.get("quality_gate")
        if not isinstance(folds, list) or not folds or not isinstance(quality, dict):
            raise PortableDownstreamError("completed head manifest is incomplete")
        for fold in folds:
            if not isinstance(fold, dict):
                raise PortableDownstreamError("completed head fold is incomplete")
            _verify_file_identity(fold.get("head"), "walk-forward head", within=root)
            _verify_file_identity(fold.get("predictions"), "walk-forward predictions", within=root)
        if not isinstance(manifest.get("convergence_gate_passed"), bool):
            raise PortableDownstreamError("completed head gate evidence is incomplete")
        return
    if stage == "simulate":
        _verify_file_identity(manifest.get("trades"), "simulation trades", within=root)
        if not isinstance(manifest.get("result"), dict):
            raise PortableDownstreamError("completed simulation result is incomplete")
        return
    raise PortableDownstreamError(f"unknown completed manifest stage: {stage}")


def bind_producer(
    *,
    template_path: Path,
    promotion_manifest_path: Path,
    corpus_manifest_path: Path,
    data_root: Path,
    artifact_root: Path,
    run_name: str,
    device: str,
    output_path: Path,
) -> dict[str, Any]:
    """Bind one promoted foundation and corpus to a new producer config."""
    template = load_downstream_config(template_path)
    if (
        template.strategy_contract is None
        or template.data.timeframes != FOUR_TIMEFRAME_CONTRACT
        or template.evaluation.allow_holdout
    ):
        raise PortableDownstreamError("producer template is not the locked four-timeframe recipe")
    if not run_name or run_name == template.run.name:
        raise PortableDownstreamError("producer run name must be unique and explicit")
    promotion_path = promotion_manifest_path.resolve()
    promotion = _read_json(promotion_path, "promoted foundation manifest")
    _validate_promoted_bundle(promotion_path, promotion)
    corpus_path = corpus_manifest_path.resolve()
    corpus = _read_json(corpus_path, "corpus manifest")
    if corpus.get("validated") is not True or not isinstance(corpus.get("outputs"), list):
        raise PortableDownstreamError("corpus manifest is not validated")
    available = {
        (output.get("symbol"), output.get("timeframe"))
        for output in corpus["outputs"]
        if isinstance(output, dict) and output.get("kind") == "market"
    }
    required = {
        (symbol, timeframe)
        for symbol in template.data.symbols
        for timeframe in FOUR_TIMEFRAME_CONTRACT
    }
    if not required.issubset(available):
        raise PortableDownstreamError("corpus manifest lacks the configured symbol/timeframe set")
    runtime_identity = _binding_base()
    producer = replace(
        template,
        run=replace(
            template.run,
            name=run_name,
            artifact_root=artifact_root.resolve(),
            device=device,  # type: ignore[arg-type]
            allow_overwrite=False,
            source_digest=str(runtime_identity["source_digest"]),
            lock_digest=str(runtime_identity["lock_digest"]),
        ),
        data=replace(
            template.data,
            root=data_root.resolve(),
            corpus_manifest_path=corpus_path,
            corpus_manifest_sha256=sha256_file(corpus_path),
            timeframes=FOUR_TIMEFRAME_CONTRACT,
        ),
        foundation=replace(
            template.foundation,
            manifest_path=promotion_path,
            weights_sha256=str(promotion["weights_sha256"]),
            export_role="promoted",
        ),
    )
    content = _render_config(producer)
    _publish_text(output_path, content, "producer config")
    rebound = load_downstream_config(output_path)
    if rebound.digest != producer.digest:
        raise PortableDownstreamError("rendered producer config identity mismatch")
    payload = {
        **runtime_identity,
        "stage": "bind-producer",
        "run_name": run_name,
        "producer_config": str(output_path.resolve()),
        "producer_config_sha256": sha256_file(output_path),
        "producer_workflow_digest": rebound.workflow_digest,
        "embedding_contract_digest": rebound.embedding_contract_digest,
        "promotion_manifest": str(promotion_path),
        "promotion_manifest_sha256": sha256_file(promotion_path),
        "foundation_weights_sha256": rebound.foundation.weights_sha256,
        "corpus_manifest": str(corpus_path),
        "corpus_manifest_sha256": rebound.data.corpus_manifest_sha256,
        "timeframes": list(FOUR_TIMEFRAME_CONTRACT),
        "symbols": list(rebound.data.symbols),
    }
    binding_path = _publish_binding(output_path, payload)
    return {**payload, "binding_manifest": str(binding_path)}


def bind_consumer(
    *,
    template_path: Path,
    producer_config_path: Path,
    embed_manifest_path: Path,
    run_name: str,
    output_path: Path,
) -> dict[str, Any]:
    """Bind one exact completed producer embedding to a distinct head config."""
    template = load_downstream_config(template_path)
    producer_path = producer_config_path.resolve()
    producer = load_downstream_config(producer_path)
    if producer.data.timeframes != FOUR_TIMEFRAME_CONTRACT or producer.strategy_contract is None:
        raise PortableDownstreamError("producer is not the portable four-timeframe recipe")
    if not run_name or run_name in {template.run.name, producer.run.name}:
        raise PortableDownstreamError("consumer run name must be unique and explicit")
    embed_path = embed_manifest_path.resolve()
    embed = _read_json(embed_path, "embedding manifest")
    expected_width = len(producer.data.timeframes) * len(producer.data.feature_columns)
    embedding_dim = embed.get("embedding_dim_per_channel")
    outputs = embed.get("outputs")
    if (
        embed.get("schema_version") != 1
        or embed.get("stage") != "embed"
        or embed.get("workflow_digest") != producer.workflow_digest
        or embed.get("strategy_contract") != producer.strategy_contract
        or embed.get("foundation_weights_sha256") != producer.foundation.weights_sha256
        or not isinstance(embedding_dim, int)
        or embed.get("feature_width") != expected_width * embedding_dim
        or not isinstance(outputs, list)
        or not outputs
        or embed.get("rows") != sum(int(output.get("rows", -1)) for output in outputs)
    ):
        raise PortableDownstreamError("embedding manifest does not match the exact producer")
    for output in outputs:
        for role in ("features", "metadata"):
            identity = output.get(role)
            path = Path(str(identity.get("path", ""))) if isinstance(identity, dict) else Path("")
            if not path.is_file() or sha256_file(path) != identity.get("sha256"):
                raise PortableDownstreamError(f"embedding {role} shard identity mismatch")
    consumer = replace(
        template,
        run=replace(
            producer.run,
            name=run_name,
            allow_overwrite=False,
        ),
        data=producer.data,
        foundation=producer.foundation,
        strategy=producer.strategy,
        topstep=producer.topstep,
        evaluation=producer.evaluation,
        walk_forward=replace(
            template.walk_forward,
            embed_manifest_path=embed_path,
            embed_manifest_sha256=sha256_file(embed_path),
            embed_producer_config_path=producer_path,
            embed_producer_config_sha256=sha256_file(producer_path),
        ),
    )
    content = _render_config(consumer)
    _publish_text(output_path, content, "consumer config")
    rebound = load_downstream_config(output_path)
    if rebound.embedding_contract_digest != producer.embedding_contract_digest:
        raise PortableDownstreamError("rendered consumer changed embedding semantics")
    payload = {
        **_binding_base(),
        "stage": "bind-consumer",
        "run_name": run_name,
        "consumer_config": str(output_path.resolve()),
        "consumer_config_sha256": sha256_file(output_path),
        "producer_config": str(producer_path),
        "producer_config_sha256": sha256_file(producer_path),
        "embed_manifest": str(embed_path),
        "embed_manifest_sha256": sha256_file(embed_path),
        "embedding_contract_digest": rebound.embedding_contract_digest,
        "head_config_digest": rebound.head_config_digest(sha256_file(embed_path)),
    }
    binding_path = _publish_binding(output_path, payload)
    return {**payload, "binding_manifest": str(binding_path)}


def run_stage(config: DownstreamConfig, stage: str) -> dict[str, Any]:
    """Run or exactly resume one independently recoverable DAG stage."""
    stages = {
        "prepare": ("prepare", prepare),
        "embed": ("embed", embed),
        "head": ("walk-forward", walk_forward),
        "simulate": ("simulate", simulate),
    }
    if stage not in stages:
        raise PortableDownstreamError(f"unknown portable downstream stage: {stage}")
    manifest_stage, operation = stages[stage]
    root = artifact_root(config) / manifest_stage
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = _manifest(manifest_path, config, manifest_stage)
        _validate_completed_manifest(manifest, config, stage, root)
        return {"stage": stage, "resumed": True, "manifest": manifest}
    if root.exists() and stage != "embed":
        raise PortableDownstreamError(
            f"incomplete {stage} stage cannot be resumed without a completed exact manifest"
        )
    manifest = operation(config)
    return {"stage": stage, "resumed": False, "manifest": manifest}
