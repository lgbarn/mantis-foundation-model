from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env
from mantis_v2 import cli
from mantis_v2 import rl_smoke as rl_smoke_module
from mantis_v2.rl_config import load_rl_config
from mantis_v2.rl_environment import EnvironmentContractError
from mantis_v2.rl_smoke import (
    GymnasiumEntryAdapter,
    NumericAudit,
    RlSmokeError,
    build_synthetic_episodes,
    run_maskable_ppo_smoke,
)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_checker import check_env as check_sb3_env

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")


@pytest.fixture(scope="module")
def completed_smoke(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("rl-smoke") / "complete"
    result = run_maskable_ppo_smoke(_config(), output)
    return output, result


def _checkpoint(output: Path) -> Path:
    state = json.loads((output / "state.json").read_text())
    return output / state["checkpoint"] / "model.zip"


def _assert_torch_tree_close(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.allclose(left, right, rtol=1e-10, atol=1e-12)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_torch_tree_close(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_torch_tree_close(first, second)
    else:
        assert left == right


def test_official_environment_checks_and_action_masks_use_training_adapter() -> None:
    episodes = build_synthetic_episodes(seed=42, count=16)
    environment = GymnasiumEntryAdapter(_config(), episodes, seed=42)

    check_env(environment, skip_render_check=True)
    check_sb3_env(environment)
    observation, _ = environment.reset(seed=42)

    assert np.isfinite(observation).all()
    assert get_action_masks(environment).tolist() == [True, True]
    environment.step(1)
    assert environment.illegal_action_count == 0
    with pytest.raises(EnvironmentContractError, match="invalid action"):
        environment.step(1)


def test_50k_cpu_smoke_beats_reject_all_and_keeps_every_atomic_boundary(
    completed_smoke,
) -> None:
    output, result = completed_smoke

    assert result["timesteps"] == 50_000
    assert result["policy_mean_reward"] > result["reject_all_mean_reward"]
    assert result["finite_policy"] is True
    assert result["numeric_audit"]["finite"] is True
    assert result["illegal_action_count"] == 0
    assert result["loaded_actions_match"] is True
    assert result["schedule_reproduced"] is True
    assert sorted(path.name for path in (output / "checkpoints").iterdir()) == [
        "step-00010000",
        "step-00020000",
        "step-00030000",
        "step-00040000",
        "step-00050000",
    ]
    assert (output / "metrics.json").is_file()
    assert (output / "manifest.json").is_file()


def test_interrupted_resume_matches_uninterrupted_policy_optimizer_and_schedule(
    tmp_path: Path, completed_smoke
) -> None:
    baseline_output, baseline_result = completed_smoke
    config = _config()
    output = tmp_path / "resumable"
    partial = run_maskable_ppo_smoke(config, output, maximum_steps_this_run=20_000)
    assert partial["timesteps"] == 20_000

    original_state = (output / "state.json").read_text()
    tampered = json.loads(original_state)
    tampered["completed_timesteps"] = 30_000
    (output / "state.json").write_text(json.dumps(tampered))
    with pytest.raises(RlSmokeError, match="checkpoint identity mismatch"):
        run_maskable_ppo_smoke(config, output, resume=True)
    (output / "state.json").write_text(original_state)

    checkpoint = _checkpoint(output)
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(checkpoint_bytes + b"tampered")
    with pytest.raises(RlSmokeError, match="checkpoint identity mismatch"):
        run_maskable_ppo_smoke(config, output, resume=True)
    checkpoint.write_bytes(checkpoint_bytes)
    with pytest.raises(RlSmokeError, match="resume provenance mismatch"):
        run_maskable_ppo_smoke(
            replace(config, run=replace(config.run, seed=43)), output, resume=True
        )

    trained = run_maskable_ppo_smoke(config, output, resume=True, stop_after_training=True)
    assert trained == {"status": "trained", "timesteps": 50_000, "resumable": True}
    trained_checkpoint = _checkpoint(output)
    trained_hash = trained_checkpoint.read_bytes()
    completed = run_maskable_ppo_smoke(config, output, resume=True)
    assert completed["status"] == "complete"
    assert trained_checkpoint.read_bytes() == trained_hash

    baseline = MaskablePPO.load(_checkpoint(baseline_output), device="cpu")
    resumed = MaskablePPO.load(_checkpoint(output), device="cpu")
    _assert_torch_tree_close(baseline.policy.state_dict(), resumed.policy.state_dict())
    _assert_torch_tree_close(
        baseline.policy.optimizer.state_dict(), resumed.policy.optimizer.state_dict()
    )
    assert (
        baseline_result["schedule_digests"]["training"] == completed["schedule_digests"]["training"]
    )

    with pytest.raises(RlSmokeError, match="run identity already exists"):
        run_maskable_ppo_smoke(config, output)
    with pytest.raises(RlSmokeError, match="cannot be resumed or overwritten"):
        run_maskable_ppo_smoke(config, output, resume=True)


@pytest.mark.parametrize(
    "surface",
    ("observations", "rewards", "logits", "values", "gradients", "parameters", "optimizer_state"),
)
def test_numeric_audit_rejects_injected_nonfinite_values(surface: str) -> None:
    with pytest.raises(RlSmokeError, match=surface):
        NumericAudit()._finite(surface, np.array([np.nan], dtype=np.float32))


def test_smoke_independently_never_opens_production_or_holdout_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = Path.open
    forbidden = Path("/Volumes/Storage")
    attempted: list[Path] = []

    def guarded(path: Path, *args, **kwargs):
        if path == forbidden or forbidden in path.parents:
            attempted.append(path)
            raise AssertionError(f"forbidden data access: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    result = run_maskable_ppo_smoke(_config(), tmp_path / "guarded", maximum_steps_this_run=10_000)

    assert result["status"] == "incomplete"
    assert attempted == []


@pytest.mark.parametrize("failure", ("save", "bundle_replace", "pointer_replace"))
def test_checkpoint_failures_preserve_last_valid_pointer_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    config = _config()
    output = tmp_path / failure
    run_maskable_ppo_smoke(config, output, maximum_steps_this_run=10_000)
    prior_state = (output / "state.json").read_bytes()
    prior_checkpoint = _checkpoint(output)
    prior_checkpoint_hash = prior_checkpoint.read_bytes()

    with monkeypatch.context() as context:
        if failure == "save":
            context.setattr(
                MaskablePPO,
                "save",
                lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected save failure")),
            )
        else:
            original_replace = rl_smoke_module.os.replace

            def failing_replace(source, target):
                destination = Path(target)
                if failure == "bundle_replace" and destination.name == "step-00020000":
                    raise OSError("injected bundle replace failure")
                if failure == "pointer_replace" and destination == output / "state.json":
                    raise OSError("injected pointer replace failure")
                return original_replace(source, target)

            context.setattr(rl_smoke_module.os, "replace", failing_replace)
        with pytest.raises(OSError, match="injected"):
            run_maskable_ppo_smoke(config, output, resume=True, maximum_steps_this_run=10_000)

    assert (output / "state.json").read_bytes() == prior_state
    assert prior_checkpoint.read_bytes() == prior_checkpoint_hash
    resumed = run_maskable_ppo_smoke(config, output, resume=True, maximum_steps_this_run=10_000)
    assert resumed["timesteps"] == 20_000


def test_cli_exposes_cpu_rl_smoke_and_resume(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config()
    expected = {"stage": "rl-maskable-ppo-smoke"}
    monkeypatch.setattr(cli, "load_rl_config", lambda _path: config)
    monkeypatch.setattr(cli, "run_maskable_ppo_smoke", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        "sys.argv",
        ["mantis-v2", "rl-smoke", "--config", "rl.toml", "--output", "smoke-output", "--resume"],
    )

    cli.main()

    assert json.loads(capsys.readouterr().out) == expected
