from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from mantis_v2.config import ConfigError
from mantis_v2.rl_config import load_rl_config

ROOT = Path(__file__).resolve().parents[1]


def test_topstep_rule_contract_pins_account_and_instrument_economics() -> None:
    rules = tomllib.loads((ROOT / "configs" / "topstep-100k-2026-07-20.toml").read_text())

    assert rules["account"] == {
        "starting_balance": 100000.0,
        "initial_mll_floor": 97000.0,
        "mll_distance": 3000.0,
        "profit_target": 6000.0,
        "consistency_limit": 0.50,
        "minimum_trading_days": 2,
        "maximum_position_equivalence": 10,
        "mll_lock_balance": 100000.0,
        "mll_ratchet": "end_of_day_high_water",
        "mll_enforcement": "continuous_realized_and_unrealized",
        "overnight_holding": False,
    }
    assert rules["position_equivalence"] == {
        "micros_per_mini": 10,
        "maximum_minis": 10,
        "maximum_micros": 100,
    }
    assert rules["session"] == {
        "timezone": "America/Chicago",
        "start": "17:00",
        "force_flat": "15:10",
    }
    expected_contracts = {
        "ES": ("mini", "ES", 0.25, 12.50, 10, False),
        "MES": ("micro", "ES", 0.25, 1.25, 1, False),
        "NQ": ("mini", "NQ", 0.25, 5.00, 10, False),
        "MNQ": ("micro", "NQ", 0.25, 0.50, 1, False),
        "RTY": ("mini", "RTY", 0.10, 5.00, 10, False),
        "M2K": ("micro", "RTY", 0.10, 0.50, 1, False),
        "YM": ("mini", "YM", 1.00, 5.00, 10, False),
        "MYM": ("micro", "YM", 1.00, 0.50, 1, False),
        "GC": ("mini", "GC", 0.10, 10.00, 10, False),
        "MGC": ("micro", "GC", 0.10, 1.00, 1, False),
        "CL": ("mini", "CL", 0.01, 10.00, 10, False),
        "MCL": ("micro", "CL", 0.01, 1.00, 1, False),
        "ZB": ("mini", "ZB", 0.03125, 31.25, 10, True),
    }
    assert {
        symbol: (
            contract["contract_class"],
            contract["underlying"],
            contract["tick_size"],
            contract["tick_value"],
            contract["position_units"],
            contract.get("mini_only", False),
        )
        for symbol, contract in rules["contracts"].items()
    } == expected_contracts


def test_production_rl_config_loads_locked_entry_contract() -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")

    assert config.run.profile == "production"
    assert config.policy.actions == ("skip", "enter")
    assert config.episode.timeout_trading_days == 20
    assert config.exit.activation_r == 2.0
    assert config.exit.giveback_r == 0.75
    assert config.sizing.episode_profiles == ("one_mini", "ten_micros")
    assert config.sizing.mini_only_instruments == ("ZB",)
    assert config.fees.cl == 4.02
    assert config.fees.mcl == 1.52
    assert config.run.device == "cpu"
    assert config.constraint.kind == "episodic_blow_lagrangian"
    assert config.constraint.cost_limit == 0.01
    assert config.constraint.cost_gamma == 1.0
    assert config.constraint.lambda_init == 1.0
    assert config.constraint.lambda_lr == 0.01
    assert config.constraint.lambda_max == 100.0
    assert config.constraint.minimum_cushion_role == "observation_metric_only"
    assert len(config.digest) == 64


def test_direct_lora_3tf_rl_config_binds_completed_embedding_identity() -> None:
    config = load_rl_config(
        ROOT / "configs" / "rl-entry-topstep-100k-direct-lora-3tf-v1.toml"
    )

    assert config.run.name == "rl-entry-topstep-100k-direct-lora-3tf-v1"
    assert config.run.device == "cpu"
    assert config.policy.role == "entry"
    assert config.policy.actions == ("skip", "enter")
    assert config.upstream.embedding_manifest_sha256 == (
        "bdcc8819c2d68efff7bd48efc3ffdf4ba02bb73cfe01793d7a23208075b0625a"
    )
    assert config.upstream.foundation_weights_sha256 == (
        "536ae864a1fa292b13d6dc98c61c45f3ab15646dbb354dc7f25d5ed4bf0926f0"
    )


def test_smoke_profile_reduces_scale_without_weakening_promotion_gates() -> None:
    smoke = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    production = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")

    assert smoke.run.profile == "smoke"
    assert smoke.training.smoke_timesteps == production.training.smoke_timesteps == 50_000
    assert (
        smoke.training.development_timesteps_per_seed
        < production.training.development_timesteps_per_seed
    )
    assert smoke.evaluation == production.evaluation


@pytest.mark.parametrize("invalid_version", ("true", "1.0"))
def test_rl_config_rejects_non_integer_schema_versions(
    tmp_path: Path, invalid_version: str
) -> None:
    source = (ROOT / "configs" / "rl-entry-smoke.toml").read_text()
    path = tmp_path / "invalid-schema-version.toml"
    path.write_text(source.replace("schema_version = 1", f"schema_version = {invalid_version}"))

    with pytest.raises(ConfigError, match="schema_version must be integer 1"):
        load_rl_config(path)


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
        (
            "development_seeds = [42, 43, 44, 45, 46]",
            "development_seeds = [42]",
            r"production training settings must match the accepted contract",
        ),
        (
            "minimum_chronological_attempts = 300",
            "minimum_chronological_attempts = 299",
            r"promotion gates must match the accepted contract",
        ),
        (
            "market_block_length_weeks = 2",
            "market_block_length_weeks = 0",
            r"rl.evaluation.market_block_length_weeks must be an integer >= 1",
        ),
        (
            "cost_limit = 0.01",
            "cost_limit = 0.02",
            r"rl.constraint.cost_limit must be 0.01",
        ),
    )
    for index, (old, new, message) in enumerate(cases):
        path = tmp_path / f"invalid-{index}.toml"
        path.write_text(source.replace(old, new, 1))
        with pytest.raises(ConfigError, match=message):
            load_rl_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ("seed = 42", "seed = 99", "rl.run.seed must appear in rl.training.development_seeds"),
        (
            "development_seeds = [42]",
            "development_seeds = [42, 42]",
            "rl.training.development_seeds must be unique",
        ),
        (
            "confirmation_seeds = [42]",
            "confirmation_seeds = [42, 42]",
            "rl.training.confirmation_seeds must be unique",
        ),
        (
            "maximum_search_trials = 1",
            "maximum_search_trials = 31",
            "rl.training.maximum_search_trials must be <= 30",
        ),
        (
            "development_timesteps_per_seed = 50000",
            "development_timesteps_per_seed = 25000",
            "rl.training timestep budgets must be nondecreasing",
        ),
        (
            "minimum_raw_pass_rate = 0.60",
            "minimum_raw_pass_rate = 1.01",
            "rl.evaluation.minimum_raw_pass_rate must be <= 1",
        ),
    ),
)
def test_rl_config_rejects_cross_field_incompatibilities(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    source = (ROOT / "configs" / "rl-entry-smoke.toml").read_text()
    path = tmp_path / "incompatible.toml"
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
        replace(config, constraint=replace(config.constraint, lambda_lr=0.02)),
        replace(
            config,
            evaluation=replace(config.evaluation, minimum_raw_pass_rate=0.61),
        ),
    )

    assert all(variant.digest != config.digest for variant in variants)
