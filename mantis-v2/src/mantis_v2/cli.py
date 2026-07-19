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
from mantis_v2.data import DataContractError, inspect_streams
from mantis_v2.model import ModelContractError
from mantis_v2.pipeline import (
    PipelineError,
    anchor_counts,
    evaluate,
    export,
    probe,
    smoke,
    train,
    verify_upstream,
)
from mantis_v2.runtime import RuntimeContractError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mantis-v2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "inspect-data",
        "train",
        "evaluate",
        "export",
        "smoke",
        "probe",
        "verify-upstream",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
    return parser


def _inspect(config: PipelineConfig) -> dict[str, Any]:
    summaries = [asdict(summary) for summary in inspect_streams(config.data)]
    return {
        "root": config.data.root,
        "configured_streams": len(config.data.symbols) * len(config.data.intervals),
        "streams": summaries,
        "anchors": anchor_counts(config),
    }


def main() -> None:
    args = _parser().parse_args()
    commands: dict[str, Callable[[PipelineConfig], dict[str, Any]]] = {
        "inspect-data": _inspect,
        "train": train,
        "evaluate": evaluate,
        "export": export,
        "smoke": smoke,
        "probe": probe,
        "verify-upstream": verify_upstream,
    }
    try:
        config = load_config(args.config)
        result = commands[args.command](config)
    except (
        CheckpointError,
        ConfigError,
        DataContractError,
        ModelContractError,
        PipelineError,
        RuntimeContractError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
