from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from mantis_v2.config import ConfigError
from mantis_v2.rl_config import load_rl_config

ROOT = Path(__file__).resolve().parents[1]


def test_production_rl_config_loads_locked_entry_contract() -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")

    assert config.policy.actions == ("skip", "enter")
    assert config.episode.timeout_trading_days == 20
    assert config.exit.activation_r == 2.0
    assert config.exit.giveback_r == 0.75
    assert config.sizing.episode_profiles == ("one_mini", "ten_micros")
    assert config.sizing.mini_only_instruments == ("ZB",)
    assert config.fees.cl == 4.02
    assert config.fees.mcl == 1.52
    assert config.run.device == "cpu"
    assert len(config.digest) == 64


def test_rl_config_rejects_unknown_missing_invalid_and_incompatible_values(
    tmp_path: Path,
) -> None:
    source = (ROOT / "configs" / "rl-entry-topstep-100k.toml").read_text()
    cases = (
        ("seed = 42", "seed = 42\nsurprise = true", r"unknown \[rl.run\] keys: surprise"),
        ('device = "cpu"\n', "", r"missing \[rl.run\] keys: device"),
        ("seed = 42", "seed = -1", r"rl.run.seed must be an integer >= 0"),
        (
            "timeout_trading_days = 20",
            "timeout_trading_days = 19",
            r"rl.episode.timeout_trading_days must be 20",
        ),
    )
    for index, (old, new, message) in enumerate(cases):
        path = tmp_path / f"invalid-{index}.toml"
        path.write_text(source.replace(old, new, 1))
        with pytest.raises(ConfigError, match=message):
            load_rl_config(path)


@pytest.mark.parametrize("name", ("../escaped", "/tmp/escaped", "nested/run", ".", ".."))
def test_rl_run_name_cannot_escape_the_artifact_root(tmp_path: Path, name: str) -> None:
    source = (ROOT / "configs" / "rl-entry-smoke.toml").read_text()
    path = tmp_path / "escaped.toml"
    path.write_text(source.replace('name = "rl-entry-smoke-v1"', f'name = "{name}"'))

    with pytest.raises(ConfigError, match="rl.run.name must be a portable identifier"):
        load_rl_config(path)


@pytest.mark.parametrize(
    ("field", "qualified"),
    (
        ("downstream_config_path", "upstream.downstream_config_path"),
        ("corpus_manifest_path", "upstream.corpus_manifest_path"),
        ("embedding_manifest_path", "upstream.embedding_manifest_path"),
        ("foundation_manifest_path", "upstream.foundation_manifest_path"),
        ("foundation_weights_path", "upstream.foundation_weights_path"),
        ("artifact_root", "rl.run.artifact_root"),
    ),
)
def test_rl_paths_reject_non_string_values(tmp_path: Path, field: str, qualified: str) -> None:
    source = (ROOT / "configs" / "rl-entry-smoke.toml").read_text()
    mutated = "\n".join(
        f"{field} = true" if line.startswith(f"{field} = ") else line
        for line in source.splitlines()
    )
    path = tmp_path / f"invalid-{field}.toml"
    path.write_text(mutated)

    with pytest.raises(ConfigError, match=rf"{qualified} must be a non-empty path string"):
        load_rl_config(path)


def test_every_locked_rl_identity_contributes_to_the_canonical_digest() -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    variants = (
        replace(config, sizing=replace(config.sizing, episode_profiles=("one_mini",))),
        replace(config, sizing=replace(config.sizing, mini_only_instruments=("ZN",))),
        replace(config, execution=replace(config.execution, fee_schedule="changed")),
        replace(config, fees=replace(config.fees, cl=4.03)),
        replace(
            config,
            execution=replace(config.execution, adverse_slippage_ticks_per_side=1.5),
        ),
        replace(config, exit=replace(config.exit, activation_r=2.5)),
        replace(config, exit=replace(config.exit, giveback_r=1.0)),
        replace(config, episode=replace(config.episode, timeout_trading_days=21)),
        replace(config, training=replace(config.training, smoke_timesteps=129)),
        replace(config, training=replace(config.training, development_seeds=(41,))),
        replace(
            config,
            evaluation=replace(config.evaluation, minimum_raw_pass_rate=0.61),
        ),
    )

    assert all(variant.digest != config.digest for variant in variants)
