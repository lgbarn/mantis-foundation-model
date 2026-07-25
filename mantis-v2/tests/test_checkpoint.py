from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from mantis_v2.checkpoint import (
    CheckpointError,
    checkpoint_adaptation_state,
    load_checkpoint,
    save_checkpoint,
)
from mantis_v2.provenance import FileIdentity, Provenance
from torch import nn


def provenance() -> Provenance:
    return Provenance(
        schema_version=1,
        precision="fp32",
        config_digest="config",
        dataset_digest="dataset",
        dataset_files=(FileIdentity("synthetic", 1, "hash"),),
        source_revision="source",
        source_dirty=False,
        source_digest="source-content",
        lock_digest="lock",
        upstream_source_revision="upstream",
        upstream_hub_revision="hub",
        upstream_weights_sha256="weights",
        contamination_digest="contamination",
    )


def test_checkpoint_restores_training_state_and_rejects_stale_data(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    original = model.weight.detach().clone()
    original_moment = optimizer.state[model.weight]["exp_avg"].detach().clone()
    save_checkpoint(path, model, optimizer, epoch=4, global_step=99, provenance=provenance())
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    expected_mps = torch.rand(3, device="mps").cpu() if torch.backends.mps.is_available() else None
    with torch.no_grad():
        model.weight.zero_()
    optimizer.state.clear()
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(999)
    epoch, step = load_checkpoint(path, model, optimizer, provenance())
    torch.testing.assert_close(model.weight, original)
    torch.testing.assert_close(optimizer.state[model.weight]["exp_avg"], original_moment)
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch)
    if expected_mps is not None:
        torch.testing.assert_close(torch.rand(3, device="mps").cpu(), expected_mps)
    assert (epoch, step) == (5, 99)

    stale = replace(provenance(), dataset_digest="changed")
    try:
        load_checkpoint(path, model, optimizer, stale)
    except CheckpointError as exc:
        assert "dataset_digest" in str(exc)
    else:
        raise AssertionError("stale dataset provenance was accepted")

    stale_contamination = replace(provenance(), contamination_digest="changed")
    with pytest.raises(CheckpointError, match="contamination_digest"):
        load_checkpoint(path, model, optimizer, stale_contamination)

    with pytest.raises(
        CheckpointError,
        match="checkpoint precision-provenance mismatch: expected bf16, observed fp32",
    ):
        load_checkpoint(path, model, optimizer, replace(provenance(), precision="bf16"))


class UnsafeCheckpoint:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[Any, tuple[Path]]:
        return Path.touch, (self.marker,)


def test_checkpoint_rejects_executable_pickle(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.pt"
    marker = tmp_path / "executed"
    torch.save(UnsafeCheckpoint(marker), path)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    try:
        load_checkpoint(path, model, optimizer, provenance())
    except CheckpointError:
        pass
    else:
        raise AssertionError("unsafe checkpoint was accepted")
    assert not marker.exists()


def test_checkpoint_round_trips_and_binds_adaptation_state(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    state: dict[str, object] = {
        "phase": "lora",
        "stage_two_phase": "lora",
        "phase_updates": 6,
        "total_updates": 10,
        "warm_start_updates": 4,
        "stage_two_updates": 6,
        "transition_parent": "a" * 64,
        "optimizer_identity": "b" * 64,
        "trainable_parameters": 8,
        "frozen_parameters": 12,
    }
    path = tmp_path / "adaptation.pt"
    save_checkpoint(path, model, optimizer, 3, 10, provenance(), state)

    assert checkpoint_adaptation_state(path, provenance()) == state
    assert load_checkpoint(path, model, optimizer, provenance(), state) == (4, 10)
    with pytest.raises(CheckpointError, match="adaptation state mismatch"):
        load_checkpoint(path, model, optimizer, provenance(), {**state, "total_updates": 11})


def test_checkpoint_preserves_legacy_lora_adaptation_for_export_repair(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    legacy: dict[str, object] = {
        "phase": "lora",
        "phase_updates": 6,
        "total_updates": 10,
        "warm_start_updates": 4,
        "lora_updates": 6,
        "transition_parent": "a" * 64,
        "optimizer_identity": "b" * 64,
        "trainable_parameters": 8,
        "frozen_parameters": 12,
    }
    path = tmp_path / "legacy-adaptation.pt"
    save_checkpoint(path, model, optimizer, 3, 10, provenance(), legacy)

    assert checkpoint_adaptation_state(path, provenance()) == legacy
    assert load_checkpoint(path, model, optimizer, provenance(), legacy) == (4, 10)


def test_checkpoint_wraps_structurally_malformed_safe_payload(tmp_path: Path) -> None:
    path = tmp_path / "malformed.pt"
    torch.save(
        {
            "schema_version": 1,
            "provenance": provenance().to_dict(),
            "model": [],
            "optimizer": {},
        },
        path,
    )
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    try:
        load_checkpoint(path, model, optimizer, provenance())
    except CheckpointError as exc:
        assert "malformed checkpoint" in str(exc)
    else:
        raise AssertionError("malformed checkpoint was accepted")
