"""Command-line interface for MantisV2 workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mantis_v2.checkpoint import CheckpointError
from mantis_v2.config import ConfigError, PipelineConfig, load_config
from mantis_v2.corpus import CorpusRepairError, repair_corpus, validate_corpus
from mantis_v2.corpus_config import load_corpus_repair_config
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
from mantis_v2.rl_episodes import EpisodeContractError, build_episode_manifest
from mantis_v2.rl_provenance import RlProvenanceError, write_rl_dry_run_manifest
from mantis_v2.rl_smoke import RlSmokeError, run_maskable_ppo_smoke
from mantis_v2.rl_validation import (
    EnvironmentValidationError,
    write_environment_validation,
)
from mantis_v2.runpod_config import RunpodConfigError
from mantis_v2.runpod_image import ImageContractError
from mantis_v2.runpod_image import self_check as runpod_image_self_check
from mantis_v2.runpod_plan import LaunchPlanError, write_launch_decision
from mantis_v2.runtime import RuntimeContractError
from mantis_v2.strategy import StrategyContractError
from mantis_v2.topstep import TopstepContractError
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
        elif args.command == "tensorboard":
            result = serve_tensorboard(args.run_root, host=args.host, port=args.port)
        elif args.command == "rl-account-replay":
            result = write_account_replay_manifest(
                load_rl_config(args.config), args.input, args.output
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
        EpisodeContractError,
        EnvironmentValidationError,
        DownstreamPipelineError,
        EmbeddingContractError,
        StrategyContractError,
        TopstepContractError,
        WalkForwardContractError,
        RunpodConfigError,
        LaunchPlanError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if args.command == "runpod-image-self-check":
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), default=str))
    else:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
