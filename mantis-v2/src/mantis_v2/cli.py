"""Command-line interface for MantisV2 workflows."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from mantis_v2.bf16_qualification import Bf16QualificationError, qualify_bf16_files
from mantis_v2.checkpoint import CheckpointError
from mantis_v2.config import ConfigError, PipelineConfig, load_config
from mantis_v2.corpus import CorpusRepairError, repair_corpus, validate_corpus
from mantis_v2.corpus_config import load_corpus_repair_config
from mantis_v2.cuda_qualification import (
    QualificationError,
    load_qualification_config,
    run_probe_and_export,
)
from mantis_v2.data import DataContractError, inspect_streams
from mantis_v2.downstream_config import DownstreamConfig, load_downstream_config
from mantis_v2.downstream_pipeline import (
    DownstreamPipelineError,
)
from mantis_v2.downstream_pipeline import (
    embed as downstream_embed,
)
from mantis_v2.downstream_pipeline import (
    evaluate_holdout as downstream_holdout,
)
from mantis_v2.downstream_pipeline import (
    prepare as downstream_prepare,
)
from mantis_v2.downstream_pipeline import (
    run as downstream_run,
)
from mantis_v2.downstream_pipeline import (
    simulate as downstream_simulate,
)
from mantis_v2.downstream_pipeline import (
    smoke as downstream_smoke,
)
from mantis_v2.downstream_pipeline import verify_contract as downstream_verify
from mantis_v2.downstream_pipeline import (
    walk_forward as downstream_walk_forward,
)
from mantis_v2.embedding import EmbeddingContractError
from mantis_v2.embedding_artifacts import EmbeddingArtifactError
from mantis_v2.embedding_qualification import qualify_embedding_files
from mantis_v2.model import ModelContractError
from mantis_v2.monitoring import MonitoringError, serve_tensorboard
from mantis_v2.pipeline import (
    PipelineError,
    data_audit,
    evaluate,
    export,
    probe,
    smoke,
    train,
    validated_export,
    verify_upstream,
)
from mantis_v2.rl_account import RlAccountError, write_account_replay_manifest
from mantis_v2.rl_config import load_rl_config
from mantis_v2.rl_confirmation import (
    ConfirmationError,
    decide_continuation,
    freeze_architecture_plan,
    qualify_architecture,
    run_architecture_ablation,
    run_production_seed_campaign,
)
from mantis_v2.rl_episodes import EpisodeContractError, build_episode_manifest
from mantis_v2.rl_optuna import (
    OptunaSearchError,
    run_production_optuna_study,
)
from mantis_v2.rl_provenance import RlProvenanceError, write_rl_dry_run_manifest
from mantis_v2.rl_smoke import RlSmokeError, run_maskable_ppo_smoke
from mantis_v2.rl_training import (
    PolicyVariant,
    ProductionTrainingError,
    train_policy_seeds,
)
from mantis_v2.rl_validation import (
    EnvironmentValidationError,
    write_environment_validation,
)
from mantis_v2.runpod_config import RunpodConfigError, load_local_config
from mantis_v2.runpod_image import ImageContractError
from mantis_v2.runpod_image import self_check as runpod_image_self_check
from mantis_v2.runpod_lifecycle import (
    LifecycleError,
    enforce_deadline,
    launch_pod,
    pod_status,
    reconcile_launch,
    reconcile_spend,
    reconcile_termination,
    terminate_pod,
)
from mantis_v2.runpod_plan import LaunchPlanError, write_launch_decision
from mantis_v2.runpod_rest_adapter import RunpodAdapterError, RunpodRestV1Adapter
from mantis_v2.runtime import RuntimeContractError
from mantis_v2.strategy import StrategyContractError
from mantis_v2.topstep import TopstepContractError
from mantis_v2.transfer_bundle import (
    DryRunS3Adapter,
    TransferBundleError,
    decide_retention,
    load_bundle_manifest,
    stage_bundle,
    verify_and_promote,
    verify_backup_pair,
    verify_download,
    write_bundle_manifest,
)
from mantis_v2.transfer_config import (
    TransferConfigError,
    load_remote_inventory,
    load_retention_authorization,
    load_transfer_config,
)
from mantis_v2.walk_forward import WalkForwardContractError

_DOWNSTREAM_COMMANDS = {
    "downstream-prepare",
    "downstream-embed",
    "downstream-walk-forward",
    "downstream-simulate",
    "downstream-run",
    "downstream-holdout",
    "downstream-smoke",
    "downstream-verify",
}
_CORPUS_COMMANDS = {"repair-corpus", "validate-corpus"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mantis-v2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "inspect-data",
        "train",
        "evaluate",
        "export",
        "validated-export",
        "smoke",
        "probe",
        "verify-upstream",
        "rl-dry-run",
        "rl-build-episodes",
        "rl-account-replay",
        "rl-validate-environment",
        "rl-smoke",
        "rl-train",
        "rl-optuna-search",
        "rl-qualify-architecture",
        "rl-freeze-architecture-plan",
        "rl-decide-continuation",
        "rl-run-architecture-ablation",
        "rl-run-seed-campaign",
        "runpod-image-self-check",
        *_CORPUS_COMMANDS,
        *_DOWNSTREAM_COMMANDS,
        "tensorboard",
    ):
        child = subparsers.add_parser(command)
        if command not in ("runpod-image-self-check", "tensorboard"):
            child.add_argument("--config", required=True, type=Path)
        if command == "tensorboard":
            child.add_argument("--run-root", required=True, type=Path)
            child.add_argument("--host", default="127.0.0.1")
            child.add_argument("--port", default=6006, type=int)
        if command == "rl-build-episodes":
            child.add_argument("--fold", required=True, type=int)
            child.add_argument(
                "--partition", required=True, choices=("training", "validation", "test")
            )
            child.add_argument("--episodes", required=True, type=int)
        if command == "rl-account-replay":
            child.add_argument("--input", required=True, type=Path)
            child.add_argument("--output", required=True, type=Path)
        if command == "rl-validate-environment":
            child.add_argument("--training-manifest", required=True, type=Path)
            child.add_argument("--validation-manifest", required=True, type=Path)
            child.add_argument("--output", required=True, type=Path)
        if command == "rl-smoke":
            child.add_argument("--output", required=True, type=Path)
            child.add_argument("--resume", action="store_true")
        if command == "rl-train":
            child.add_argument("--training-manifest", required=True, type=Path)
            child.add_argument("--output", required=True, type=Path)
            child.add_argument(
                "--variant",
                choices=tuple(variant.value for variant in PolicyVariant),
                default=PolicyVariant.SHARED_TICKER_VALUE.value,
            )
            child.add_argument("--resume", action="store_true")
            child.add_argument("--target-updates", type=int)
            child.add_argument("--maximum-updates-this-run", type=int)
        if command == "rl-optuna-search":
            child.add_argument("--training-manifest", required=True, type=Path)
            child.add_argument("--validation-manifest", required=True, type=Path)
            child.add_argument("--output", required=True, type=Path)
            child.add_argument("--study-name", required=True)
            child.add_argument(
                "--variant",
                choices=tuple(variant.value for variant in PolicyVariant),
                default=PolicyVariant.SHARED_TICKER_VALUE.value,
            )
        if command == "rl-qualify-architecture":
            child.add_argument("--winner", required=True, type=Path)
            child.add_argument("--evidence", required=True, type=Path)
            child.add_argument("--output", required=True, type=Path)
        if command == "rl-freeze-architecture-plan":
            child.add_argument("--winner", required=True, type=Path)
            child.add_argument("--training-manifest", required=True, action="append", type=Path)
            child.add_argument("--validation-manifest", required=True, action="append", type=Path)
            child.add_argument("--created-at", required=True)
            child.add_argument("--output", required=True, type=Path)
        if command == "rl-decide-continuation":
            child.add_argument("--candidate", required=True, type=Path)
            child.add_argument("--evidence", required=True, type=Path)
            child.add_argument("--output", required=True, type=Path)
        if command == "rl-run-architecture-ablation":
            child.add_argument("--plan", required=True, type=Path)
            child.add_argument("--resume", action="store_true")
            child.add_argument("--output", required=True, type=Path)
        if command == "rl-run-seed-campaign":
            child.add_argument("--candidate", required=True, type=Path)
            child.add_argument("--training-manifest", required=True, action="append", type=Path)
            child.add_argument("--validation-manifest", required=True, action="append", type=Path)
            child.add_argument("--continuation-decision", type=Path)
            child.add_argument("--resume", action="store_true")
            child.add_argument("--output", required=True, type=Path)
        if command in _DOWNSTREAM_COMMANDS:
            child.add_argument(
                "--set",
                action="append",
                default=[],
                metavar="SECTION.KEY=VALUE",
                help="recorded TOML scalar override; repeat as needed",
            )
        if command == "downstream-holdout":
            child.add_argument("--unlock", required=True)
    runpod_plan = subparsers.add_parser("runpod-plan")
    runpod_plan.add_argument("--platform", required=True, type=Path)
    runpod_plan.add_argument("--local", required=True, type=Path)
    runpod_plan.add_argument("--experiment", required=True, type=Path)
    runpod_plan.add_argument("--intent", required=True, type=Path)
    runpod_plan.add_argument("--inventory", required=True, type=Path)
    runpod_plan.add_argument("--ledger", required=True, type=Path)
    runpod_plan.add_argument("--authorization", type=Path)
    runpod_plan.add_argument("--evaluated-at", required=True)
    runpod_plan.add_argument("--output", required=True, type=Path)
    for command in ("runpod-launch", "runpod-reconcile-launch"):
        lifecycle = subparsers.add_parser(command)
        lifecycle.add_argument("--decision", required=True, type=Path)
        lifecycle.add_argument("--local", required=True, type=Path)
    for command in (
        "runpod-status",
        "runpod-terminate",
        "runpod-reconcile-termination",
        "runpod-reconcile-spend",
    ):
        lifecycle = subparsers.add_parser(command)
        lifecycle.add_argument("--pod-id", required=True)
        lifecycle.add_argument("--run-name", required=True)
        lifecycle.add_argument("--local", required=True, type=Path)
    deadline = subparsers.add_parser("runpod-enforce-deadline")
    deadline.add_argument("--pod-id", required=True)
    deadline.add_argument("--local", required=True, type=Path)
    cuda_probe = subparsers.add_parser("cuda-fp32-probe")
    cuda_probe.add_argument("--config", required=True, type=Path)
    cuda_probe.add_argument("--qualification-config", required=True, type=Path)
    cuda_probe.add_argument("--run-id", required=True)
    cuda_probe.add_argument("--artifact-root", required=True, type=Path)
    bf16_qualification = subparsers.add_parser("cuda-bf16-qualify")
    bf16_qualification.add_argument("--qualification-config", required=True, type=Path)
    bf16_qualification.add_argument("--reference", required=True, type=Path)
    candidate_or_failure = bf16_qualification.add_mutually_exclusive_group(required=True)
    candidate_or_failure.add_argument("--candidate", type=Path)
    candidate_or_failure.add_argument("--failure")
    bf16_qualification.add_argument("--output", required=True, type=Path)
    embedding_qualification = subparsers.add_parser("cuda-embedding-qualify")
    embedding_qualification.add_argument("--qualification-config", required=True, type=Path)
    embedding_qualification.add_argument("--identity", required=True, type=Path)
    embedding_qualification.add_argument("--foundation-manifest", required=True, type=Path)
    embedding_qualification.add_argument("--cpu-features", required=True, type=Path)
    embedding_qualification.add_argument("--cuda-features", required=True, type=Path)
    embedding_qualification.add_argument("--cpu-metadata", required=True, type=Path)
    embedding_qualification.add_argument("--cuda-metadata", required=True, type=Path)
    embedding_qualification.add_argument("--shard-directory", required=True, type=Path)
    embedding_qualification.add_argument("--performance", required=True, type=Path)
    embedding_qualification.add_argument("--output", required=True, type=Path)
    for command in ("transfer-bundle", "transfer-promote", "transfer-backup-verify"):
        transfer = subparsers.add_parser(command)
        transfer.add_argument("--config", required=True, type=Path)
        if command == "transfer-backup-verify":
            transfer.add_argument("--completed-artifact-digest", required=True)
    retention = subparsers.add_parser("transfer-retention-check")
    retention.add_argument("--config", required=True, type=Path)
    retention.add_argument("--completed-artifact-digest", required=True)
    retention.add_argument("--run-state", required=True, choices=("active", "inactive"))
    retention.add_argument("--authorization", type=Path)
    stage_dry_run = subparsers.add_parser("transfer-stage-dry-run")
    stage_dry_run.add_argument("--config", required=True, type=Path)
    stage_dry_run.add_argument("--remote-inventory", required=True, type=Path)
    manifest_inspect = subparsers.add_parser("transfer-manifest-inspect")
    manifest_inspect.add_argument("--manifest", required=True, type=Path)
    return parser


def _inspect(config: PipelineConfig) -> dict[str, Any]:
    summaries = [asdict(summary) for summary in inspect_streams(config.data)]
    audit = data_audit(config)
    return {
        "root": config.data.root,
        "configured_streams": len(config.data.symbols) * len(config.data.intervals),
        "streams": summaries,
        **audit,
    }


def _load_json_object(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        loaded = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    except FileNotFoundError as exc:
        raise LifecycleError("launch_decision_not_found") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise LifecycleError("invalid_launch_decision:json") from exc
    if not isinstance(loaded, dict):
        raise LifecycleError("invalid_launch_decision:json")
    return loaded


def _live_adapter(
    local_path: Path, *, expected_local_digest: str | None = None
) -> tuple[Path, RunpodRestV1Adapter]:
    local = load_local_config(local_path)
    if socket.gethostname() != local.controller.hostname:
        raise LifecycleError("controller_host_mismatch")
    if expected_local_digest is not None and local.digest != expected_local_digest:
        raise LifecycleError("local_config_mismatch")
    key_name = local.secrets.runpod_api_key_env
    api_key = os.environ.get(key_name, "")
    if not api_key:
        raise LifecycleError("runpod_api_key_required")
    return local.paths.state_root, RunpodRestV1Adapter(api_key=api_key)


def main() -> None:
    args = _parser().parse_args()
    commands: dict[str, Callable[[PipelineConfig], dict[str, Any]]] = {
        "inspect-data": _inspect,
        "train": train,
        "evaluate": evaluate,
        "export": export,
        "validated-export": validated_export,
        "smoke": smoke,
        "probe": probe,
        "verify-upstream": verify_upstream,
    }
    result: dict[str, Any]
    try:
        if args.command == "runpod-image-self-check":
            result = runpod_image_self_check()
        elif args.command == "cuda-fp32-probe":
            result = run_probe_and_export(
                load_config(args.config),
                load_qualification_config(args.qualification_config),
                run_id=args.run_id,
                artifact_root=args.artifact_root,
                cuda_available=torch.cuda.is_available(),
            )
        elif args.command == "cuda-bf16-qualify":
            result = qualify_bf16_files(
                config_path=args.qualification_config,
                reference_path=args.reference,
                candidate_path=args.candidate,
                failure=args.failure,
                output_path=args.output,
            )
        elif args.command == "cuda-embedding-qualify":
            result = qualify_embedding_files(
                config_path=args.qualification_config,
                identity_path=args.identity,
                foundation_manifest_path=args.foundation_manifest,
                cpu_features_path=args.cpu_features,
                cuda_features_path=args.cuda_features,
                cpu_metadata_path=args.cpu_metadata,
                cuda_metadata_path=args.cuda_metadata,
                shard_directory=args.shard_directory,
                performance_path=args.performance,
                output_path=args.output,
            )
        elif args.command == "transfer-bundle":
            transfer_config = load_transfer_config(args.config)
            transfer_manifest = write_bundle_manifest(
                transfer_config.source.root,
                transfer_config.source.include,
                transfer_config.source.manifest,
            )
            result = {
                "bundle_digest": transfer_manifest.bundle_digest,
                "manifest": str(transfer_config.source.manifest),
                "total_size": transfer_manifest.total_size,
            }
        elif args.command == "transfer-promote":
            transfer_config = load_transfer_config(args.config)
            promotion = verify_and_promote(
                transfer_config.mounted.incoming_root,
                transfer_config.mounted.final_parent,
                load_bundle_manifest(transfer_config.source.manifest),
            )
            result = asdict(promotion)
        elif args.command == "transfer-stage-dry-run":
            transfer_config = load_transfer_config(args.config)
            transfer_manifest = load_bundle_manifest(transfer_config.source.manifest)
            receipt = stage_bundle(
                transfer_config.source.root,
                transfer_manifest,
                DryRunS3Adapter(load_remote_inventory(args.remote_inventory)),
            )
            result = {
                "bundle_digest": receipt.bundle_digest,
                "dry_run": True,
                "incoming_prefix": receipt.incoming_prefix,
                "planned_uploads": receipt.uploaded,
                "skipped": receipt.skipped,
            }
        elif args.command == "transfer-manifest-inspect":
            transfer_manifest = load_bundle_manifest(args.manifest)
            result = {
                "bundle_digest": transfer_manifest.bundle_digest,
                "entry_count": len(transfer_manifest.entries),
                "path_roots": sorted(
                    {entry.path.split("/", maxsplit=1)[0] for entry in transfer_manifest.entries}
                ),
                "total_size": transfer_manifest.total_size,
            }
        elif args.command in {"transfer-backup-verify", "transfer-retention-check"}:
            transfer_config = load_transfer_config(args.config)
            transfer_manifest = load_bundle_manifest(transfer_config.source.manifest)
            internal = verify_download(
                transfer_config.backups.internal_root,
                expected_manifest=transfer_manifest,
                local_manifest=load_bundle_manifest(transfer_config.backups.internal_manifest),
                completed_artifact_digest=args.completed_artifact_digest,
                role="internal_ssd",
            )
            external = verify_download(
                transfer_config.backups.external_root,
                expected_manifest=transfer_manifest,
                local_manifest=load_bundle_manifest(transfer_config.backups.external_manifest),
                completed_artifact_digest=args.completed_artifact_digest,
                role="external_drive",
            )
            backups = verify_backup_pair(internal, external)
            if args.command == "transfer-backup-verify":
                result = asdict(backups)
            else:
                authorization = (
                    load_retention_authorization(args.authorization) if args.authorization else None
                )
                result = asdict(
                    decide_retention(
                        remote_identity=transfer_config.remote_identity,
                        bundle_digest=transfer_manifest.bundle_digest,
                        completed_artifact_digest=args.completed_artifact_digest,
                        backups=backups,
                        authorization=authorization,
                        run_active=args.run_state == "active",
                    )
                )
        elif args.command == "runpod-plan":
            result = write_launch_decision(
                platform_path=args.platform,
                local_path=args.local,
                experiment_path=args.experiment,
                intent_path=args.intent,
                inventory_path=args.inventory,
                ledger_path=args.ledger,
                authorization_path=args.authorization,
                evaluated_at=args.evaluated_at,
                output_path=args.output,
            )
        elif args.command in {"runpod-launch", "runpod-reconcile-launch"}:
            decision = _load_json_object(args.decision)
            local_digest = decision.get("local_digest")
            if not isinstance(local_digest, str):
                raise LifecycleError("invalid_launch_decision:local_digest")
            state_root, adapter = _live_adapter(args.local, expected_local_digest=local_digest)
            operation = launch_pod if args.command == "runpod-launch" else reconcile_launch
            result = operation(
                decision=decision,
                state_root=state_root,
                adapter=adapter,
                now=lambda: datetime.now(UTC),
            )
        elif args.command in {
            "runpod-status",
            "runpod-terminate",
            "runpod-reconcile-termination",
            "runpod-reconcile-spend",
        }:
            state_root, adapter = _live_adapter(args.local)
            operations = {
                "runpod-status": pod_status,
                "runpod-terminate": terminate_pod,
                "runpod-reconcile-termination": reconcile_termination,
                "runpod-reconcile-spend": reconcile_spend,
            }
            result = operations[args.command](
                pod_id=args.pod_id,
                run_name=args.run_name,
                state_root=state_root,
                adapter=adapter,
                now=lambda: datetime.now(UTC),
            )
        elif args.command == "runpod-enforce-deadline":
            state_root, adapter = _live_adapter(args.local)
            result = enforce_deadline(
                pod_id=args.pod_id,
                state_root=state_root,
                adapter=adapter,
                now=lambda: datetime.now(UTC),
            )
        elif args.command == "tensorboard":
            result = serve_tensorboard(args.run_root, host=args.host, port=args.port)
        elif args.command == "rl-account-replay":
            result = write_account_replay_manifest(
                load_rl_config(args.config), args.input, args.output
            )
        elif args.command == "rl-train":
            rl_config = load_rl_config(args.config)
            result = train_policy_seeds(
                rl_config,
                args.training_manifest,
                args.output,
                variant=PolicyVariant(args.variant),
                target_updates=args.target_updates,
                maximum_updates_this_run=args.maximum_updates_this_run,
                resume=args.resume,
            )
        elif args.command == "rl-optuna-search":
            result = run_production_optuna_study(
                load_rl_config(args.config),
                args.training_manifest,
                args.validation_manifest,
                args.output,
                study_name=args.study_name,
                variant=PolicyVariant(args.variant),
            )
        elif args.command == "rl-qualify-architecture":
            result = qualify_architecture(
                load_rl_config(args.config), args.winner, args.evidence, args.output
            )
        elif args.command == "rl-freeze-architecture-plan":
            result = freeze_architecture_plan(
                load_rl_config(args.config),
                args.winner,
                args.training_manifest,
                args.validation_manifest,
                args.output,
                created_at=args.created_at,
            )
        elif args.command == "rl-decide-continuation":
            result = decide_continuation(
                load_rl_config(args.config), args.candidate, args.evidence, args.output
            )
        elif args.command == "rl-run-architecture-ablation":
            result = run_architecture_ablation(
                load_rl_config(args.config), args.plan, args.output, resume=args.resume
            )
        elif args.command == "rl-run-seed-campaign":
            result = run_production_seed_campaign(
                load_rl_config(args.config),
                args.candidate,
                args.training_manifest,
                args.validation_manifest,
                args.output,
                resume=args.resume,
                continuation_decision_path=args.continuation_decision,
            )
        elif args.command == "rl-smoke":
            result = run_maskable_ppo_smoke(
                load_rl_config(args.config), args.output, resume=args.resume
            )
        elif args.command == "rl-validate-environment":
            result = write_environment_validation(
                load_rl_config(args.config),
                args.training_manifest,
                args.validation_manifest,
                args.output,
            )
        elif args.command == "rl-dry-run":
            result = write_rl_dry_run_manifest(load_rl_config(args.config))
        elif args.command == "rl-build-episodes":
            result = build_episode_manifest(
                load_rl_config(args.config),
                fold_number=args.fold,
                partition_name=args.partition,
                episode_count=args.episodes,
            )
        elif args.command in _CORPUS_COMMANDS:
            corpus_config = load_corpus_repair_config(args.config)
            if args.command == "repair-corpus":
                manifest = repair_corpus(corpus_config)
                result = {
                    "output": corpus_config.output_path,
                    "quality": manifest["quality"],
                    **validate_corpus(corpus_config.output_path),
                }
            else:
                result = validate_corpus(corpus_config.output_path)
        elif args.command in _DOWNSTREAM_COMMANDS:
            downstream_config: DownstreamConfig = load_downstream_config(
                args.config, tuple(args.set)
            )
            downstream_commands: dict[str, Callable[[DownstreamConfig], dict[str, Any]]] = {
                "downstream-prepare": downstream_prepare,
                "downstream-embed": downstream_embed,
                "downstream-walk-forward": downstream_walk_forward,
                "downstream-simulate": downstream_simulate,
                "downstream-run": downstream_run,
                "downstream-smoke": downstream_smoke,
                "downstream-verify": downstream_verify,
            }
            if args.command == "downstream-holdout":
                result = downstream_holdout(downstream_config, args.unlock)
            else:
                result = downstream_commands[args.command](downstream_config)
        else:
            config = load_config(args.config)
            result = commands[args.command](config)
    except ImageContractError as exc:
        if exc.inventory is not None:
            print(json.dumps(exc.inventory, sort_keys=True, separators=(",", ":")))
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except (
        CheckpointError,
        ConfigError,
        CorpusRepairError,
        DataContractError,
        ModelContractError,
        MonitoringError,
        PipelineError,
        RuntimeContractError,
        RlAccountError,
        RlProvenanceError,
        RlSmokeError,
        ProductionTrainingError,
        OptunaSearchError,
        ConfirmationError,
        EpisodeContractError,
        EnvironmentValidationError,
        DownstreamPipelineError,
        EmbeddingContractError,
        EmbeddingArtifactError,
        StrategyContractError,
        TopstepContractError,
        WalkForwardContractError,
        RunpodConfigError,
        LaunchPlanError,
        LifecycleError,
        RunpodAdapterError,
        QualificationError,
        Bf16QualificationError,
        TransferBundleError,
        TransferConfigError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if args.command == "runpod-image-self-check":
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), default=str))
    else:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
