"""Atomic, resumable checkpoints richer than upstream weights-only saves."""

from __future__ import annotations

import pickle
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from mantis_v2.provenance import Provenance


class CheckpointError(RuntimeError):
    """Raised for stale, malformed, or mismatched checkpoints."""


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    provenance: Provenance,
    adaptation_state: Mapping[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_state = cast(tuple[str, np.ndarray[Any, Any], int, int, float], np.random.get_state())
    payload = {
        "schema_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": numpy_state[0],
                "keys": torch.from_numpy(numpy_state[1].copy()),
                "position": numpy_state[2],
                "has_gauss": numpy_state[3],
                "cached_gaussian": numpy_state[4],
            },
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "mps": (
                torch.mps.get_rng_state()
                if torch.backends.mps.is_available()
                else torch.empty(0, dtype=torch.uint8)
            ),
        },
        "provenance": provenance.to_dict(),
        "adaptation": dict(adaptation_state) if adaptation_state is not None else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    provenance: Provenance,
    adaptation_state: Mapping[str, object] | None = None,
) -> tuple[int, int]:
    try:
        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"cannot load checkpoint {path}: {exc}") from exc
    try:
        if not isinstance(payload, dict):
            raise CheckpointError("checkpoint payload must be a mapping")
        if payload.get("schema_version") != 1:
            raise CheckpointError("unsupported checkpoint schema")
        recorded = payload.get("provenance")
        expected = provenance.to_dict()
        if not isinstance(recorded, dict) or recorded.get("precision") != expected["precision"]:
            observed = recorded.get("precision") if isinstance(recorded, dict) else None
            raise CheckpointError(
                "checkpoint precision-provenance mismatch: "
                f"expected {expected['precision']}, observed {observed}"
            )
        for key in (
            "precision",
            "config_digest",
            "dataset_digest",
            "source_digest",
            "lock_digest",
            "upstream_source_revision",
            "upstream_hub_revision",
            "upstream_weights_sha256",
            "contamination_digest",
        ):
            if not isinstance(recorded, dict) or recorded.get(key) != expected[key]:
                raise CheckpointError(f"checkpoint provenance mismatch: {key}")
        if payload.get("adaptation") != (
            dict(adaptation_state) if adaptation_state is not None else None
        ):
            raise CheckpointError("checkpoint adaptation state mismatch")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        rng = payload["rng"]
        random.setstate(rng["python"])
        numpy_state = rng["numpy"]
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                numpy_state["keys"].numpy(),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        if torch.backends.mps.is_available() and rng["mps"].numel():
            torch.mps.set_rng_state(rng["mps"])
        return int(payload["epoch"]) + 1, int(payload["global_step"])
    except CheckpointError:
        raise
    except (AttributeError, KeyError, TypeError, RuntimeError, ValueError) as exc:
        raise CheckpointError(f"malformed checkpoint {path}: {exc}") from exc


def checkpoint_adaptation_state(path: Path, provenance: Provenance) -> dict[str, object] | None:
    """Read phase identity before optimizer construction without restoring mutable state."""
    try:
        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise CheckpointError("unsupported checkpoint schema")
    recorded = payload.get("provenance")
    expected = provenance.to_dict()
    if not isinstance(recorded, dict) or any(
        recorded.get(key) != expected[key]
        for key in (
            "precision",
            "config_digest",
            "dataset_digest",
            "source_digest",
            "lock_digest",
            "upstream_source_revision",
            "upstream_hub_revision",
            "upstream_weights_sha256",
            "contamination_digest",
        )
    ):
        raise CheckpointError("checkpoint provenance mismatch")
    adaptation = payload.get("adaptation")
    if adaptation is None:
        return None
    if not isinstance(adaptation, dict):
        raise CheckpointError("checkpoint adaptation state must be a mapping")
    required = {
        "phase",
        "phase_updates",
        "total_updates",
        "warm_start_updates",
        "lora_updates",
        "transition_parent",
        "optimizer_identity",
        "trainable_parameters",
        "frozen_parameters",
    }
    if set(adaptation) != required:
        raise CheckpointError("checkpoint adaptation state schema mismatch")
    if adaptation["phase"] not in {"warm_start", "lora"}:
        raise CheckpointError("checkpoint adaptation phase is invalid")
    for key in ("phase_updates", "total_updates", "warm_start_updates", "lora_updates"):
        value = adaptation[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CheckpointError(f"checkpoint adaptation {key} is invalid")
    for key in ("trainable_parameters", "frozen_parameters"):
        value = adaptation[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CheckpointError(f"checkpoint adaptation {key} is invalid")
    warm_updates = cast(int, adaptation["warm_start_updates"])
    lora_updates = cast(int, adaptation["lora_updates"])
    phase_updates = cast(int, adaptation["phase_updates"])
    if adaptation["total_updates"] != warm_updates + lora_updates:
        raise CheckpointError("checkpoint adaptation total_updates is inconsistent")
    if phase_updates != (warm_updates if adaptation["phase"] == "warm_start" else lora_updates):
        raise CheckpointError("checkpoint adaptation phase_updates is inconsistent")
    transition_parent = adaptation["transition_parent"]
    if adaptation["phase"] == "warm_start":
        if lora_updates != 0 or transition_parent is not None:
            raise CheckpointError("warm-start checkpoint has invalid transition identity")
    elif not isinstance(transition_parent, str) or len(transition_parent) != 64:
        raise CheckpointError("LoRA checkpoint transition parent is invalid")
    optimizer_identity = adaptation["optimizer_identity"]
    if not isinstance(optimizer_identity, str) or len(optimizer_identity) != 64:
        raise CheckpointError("checkpoint optimizer identity is invalid")
    return cast(dict[str, object], adaptation)
