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
from mantis_v2.downstream_pipeline import (
    walk_forward as downstream_walk_forward,
)
from mantis_v2.embedding import EmbeddingContractError
from mantis_v2.model import ModelContractError
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
from mantis_v2.rl_config import load_rl_config
from mantis_v2.rl_episodes import EpisodeContractError, build_episode_manifest
from mantis_v2.rl_provenance import RlProvenanceError, write_rl_dry_run_manifest
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
        *_CORPUS_COMMANDS,
        *_DOWNSTREAM_COMMANDS,
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
        if command == "rl-build-episodes":
            child.add_argument("--fold", required=True, type=int)
            child.add_argument(
                "--partition", required=True, choices=("training", "validation", "test")
            )
            child.add_argument("--episodes", required=True, type=int)
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
    try:
        if args.command == "rl-dry-run":
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
            }
            if args.command == "downstream-holdout":
                result = downstream_holdout(downstream_config, args.unlock)
            else:
                result = downstream_commands[args.command](downstream_config)
        else:
            config = load_config(args.config)
            result = commands[args.command](config)
    except (
        CheckpointError,
        ConfigError,
        CorpusRepairError,
        DataContractError,
        ModelContractError,
        PipelineError,
        RuntimeContractError,
        RlProvenanceError,
        EpisodeContractError,
        DownstreamPipelineError,
        EmbeddingContractError,
        StrategyContractError,
        TopstepContractError,
        WalkForwardContractError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
