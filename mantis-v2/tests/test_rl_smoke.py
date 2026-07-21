from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from mantis_v2 import cli
from mantis_v2.rl_config import load_rl_config
from mantis_v2.rl_smoke import (
    GymnasiumEntryAdapter,
    RlSmokeError,
    build_synthetic_episodes,
    run_maskable_ppo_smoke,
)
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_checker import check_env as check_sb3_env

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")


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


def test_50k_cpu_smoke_beats_reject_all_and_reloads_deterministically(tmp_path: Path) -> None:
    result = run_maskable_ppo_smoke(_config(), str(tmp_path / "run"))

    assert result["timesteps"] == 50_000
    assert result["policy_mean_reward"] > result["reject_all_mean_reward"]
    assert result["finite_policy"] is True
    assert result["illegal_action_count"] == 0
    assert result["loaded_actions_match"] is True
    assert result["schedule_reproduced"] is True
    assert (tmp_path / "run" / "checkpoint.zip").is_file()
    assert (tmp_path / "run" / "metrics.json").is_file()
    assert (tmp_path / "run" / "manifest.json").is_file()


def test_smoke_resume_is_provenance_checked_and_run_identity_never_overwrites(
    tmp_path: Path,
) -> None:
    config = _config()
    output = tmp_path / "resumable"

    partial = run_maskable_ppo_smoke(config, output, maximum_steps_this_run=250)

    assert partial == {"status": "incomplete", "timesteps": 250, "resumable": True}
    checkpoint = output / "checkpoint.zip"
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(checkpoint_bytes + b"tampered")
    with pytest.raises(RlSmokeError, match="checkpoint identity mismatch"):
        run_maskable_ppo_smoke(config, output, resume=True)
    checkpoint.write_bytes(checkpoint_bytes)
    with pytest.raises(RlSmokeError, match="resume provenance mismatch"):
        run_maskable_ppo_smoke(
            replace(config, run=replace(config.run, seed=43)), output, resume=True
        )

    completed = run_maskable_ppo_smoke(config, output, resume=True)

    assert completed["status"] == "complete"
    with pytest.raises(RlSmokeError, match="run identity already exists"):
        run_maskable_ppo_smoke(config, output)
    with pytest.raises(RlSmokeError, match="cannot be resumed or overwritten"):
        run_maskable_ppo_smoke(config, output, resume=True)


def test_cli_exposes_cpu_rl_smoke_and_resume(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config()
    expected = {"stage": "rl-maskable-ppo-smoke"}
    monkeypatch.setattr(cli, "load_rl_config", lambda _path: config)
    monkeypatch.setattr(cli, "run_maskable_ppo_smoke", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mantis-v2",
            "rl-smoke",
            "--config",
            "rl.toml",
            "--output",
            "smoke-output",
            "--resume",
        ],
    )

    cli.main()

    assert json.loads(capsys.readouterr().out) == expected
