from __future__ import annotations

from pathlib import Path

import pytest
from mantis_v2.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_is_valid_and_content_addressed() -> None:
    config = load_config(ROOT / "configs" / "smoke.toml")
    assert config.model.input_length == 512
    assert config.data.target_reserve == 192
    assert len(config.digest) == 64
    assert config.digest == load_config(ROOT / "configs" / "smoke.toml").digest


def test_precision_is_strict_and_part_of_experiment_identity(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "smoke.toml").read_text()
    fp32_path = tmp_path / "fp32.toml"
    fp32_path.write_text(source)
    bf16_path = tmp_path / "bf16.toml"
    bf16_path.write_text(source.replace('precision = "fp32"', 'precision = "bf16"'))
    invalid_path = tmp_path / "invalid.toml"
    invalid_path.write_text(source.replace('precision = "fp32"', 'precision = "fp16"'))

    fp32 = load_config(fp32_path)
    bf16 = load_config(bf16_path)
    assert fp32.training.precision == "fp32"
    assert bf16.training.precision == "bf16"
    assert fp32.digest != bf16.digest
    with pytest.raises(ConfigError, match="training.precision must be one of: bf16, fp32"):
        load_config(invalid_path)


def test_production_config_is_valid() -> None:
    config = load_config(ROOT / "configs" / "nextleg.toml")
    assert config.model.mode == "full_finetune"
    assert config.data.root == "/Volumes/Storage/trading-research/data/FFM_NEXTLEG"
    assert config.data.intervals == ("1min", "3min", "5min", "15min")
    assert config.run.device == "mps"
    assert config.training.batch_size == 128
    assert config.training.max_steps_per_epoch == 200
    assert config.training.validation_max_steps == 20
    assert config.training.warmup_epochs == 10
    assert config.training.early_stopping_patience == 8
    assert config.run.allow_overwrite is False


def test_transformer_finetune_mode_is_configurable(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "nextleg.toml").read_text()
    path = tmp_path / "transformer-finetune.toml"
    path.write_text(source.replace('mode = "full_finetune"', 'mode = "transformer_finetune"'))
    assert load_config(path).model.mode == "transformer_finetune"


@pytest.mark.parametrize("mode", ["lora_r8_alpha16", "lora_r16_alpha32"])
def test_lora_modes_are_strictly_configurable(tmp_path: Path, mode: str) -> None:
    source = (ROOT / "configs" / "nextleg.toml").read_text()
    path = tmp_path / f"{mode}.toml"
    path.write_text(source.replace('mode = "full_finetune"', f'mode = "{mode}"'))

    assert load_config(path).model.mode == mode


def test_bundled_adaptation_config_is_strict(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "nextleg.toml").read_text()
    path = tmp_path / "bundled.toml"
    path.write_text(
        source.replace(
            'mode = "full_finetune"',
            'mode = "lora_r8_alpha16_head_warmstart"',
        )
        + "\n[adaptation]\n"
        + "warm_start_updates = 2000\n"
        + "total_updates = 10000\n"
        + "lora_rank = 8\n"
        + "lora_alpha = 16\n"
    )

    config = load_config(path)
    assert config.adaptation is not None
    assert config.adaptation.warm_start_updates == 2000
    assert config.adaptation.total_updates == 10000
    assert config.adaptation.lora_rank == 8
    assert config.adaptation.lora_alpha == 16


def test_bundled_production_config_has_fixed_budget_and_four_timeframes() -> None:
    config = load_config(ROOT / "configs" / "nextleg-runpod-cuda-bundled-v1.toml")
    assert config.model.mode == "lora_r8_alpha16_head_warmstart"
    assert config.data.intervals == ("1min", "3min", "5min", "15min")
    assert config.adaptation is not None
    assert config.adaptation.warm_start_updates == 2000
    assert config.adaptation.total_updates == 10000
    assert (config.adaptation.lora_rank, config.adaptation.lora_alpha) == (8, 16)


def test_training_first_three_timeframe_ab_configs_are_paired() -> None:
    direct = load_config(ROOT / "configs" / "nextleg-runpod-cuda-3tf-direct-lora-s42-v1.toml")
    warm = load_config(ROOT / "configs" / "nextleg-runpod-cuda-3tf-lp-lora-s42-v1.toml")

    assert direct.run.seed == warm.run.seed == 42
    assert direct.data.intervals == warm.data.intervals == ("1min", "3min", "15min")
    assert direct.data.root == warm.data.root
    assert direct.data.corpus_manifest_sha256 == warm.data.corpus_manifest_sha256
    assert direct.model.mode == "lora_r8_alpha16"
    assert warm.model.mode == "lora_r8_alpha16_head_warmstart"
    assert direct.training.epochs * direct.training.max_steps_per_epoch == 10000
    assert direct.training.epochs == warm.training.epochs
    assert direct.training.max_steps_per_epoch == warm.training.max_steps_per_epoch
    assert warm.adaptation is not None
    assert warm.adaptation.warm_start_updates == 2000
    assert warm.adaptation.total_updates == 10000
    assert "finetune_learning_rate" not in warm.canonical_json()


def test_lp_ft_config_is_two_stage_without_lora() -> None:
    config = load_config(ROOT / "configs" / "nextleg-runpod-cuda-3tf-lp-ft-s42-v1.toml")

    assert config.model.mode == "lp_ft"
    assert config.data.intervals == ("1min", "3min", "15min")
    assert config.adaptation is not None
    assert config.adaptation.warm_start_updates == 2000
    assert config.adaptation.total_updates == 10000
    assert config.adaptation.finetune_learning_rate == pytest.approx(1e-5)
    assert config.adaptation.finetune_warmup_updates == 0
    assert not hasattr(config.adaptation, "lora_rank")
    assert not hasattr(config.adaptation, "lora_alpha")


def test_lp_ft_rejects_nonpositive_finetune_learning_rate(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "nextleg-runpod-cuda-3tf-lp-ft-s42-v1.toml").read_text()
    path = tmp_path / "invalid-lp-ft.toml"
    path.write_text(
        source.replace("finetune_learning_rate = 0.00001", "finetune_learning_rate = 0")
    )

    with pytest.raises(ConfigError, match="adaptation.finetune_learning_rate must be > 0"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("warm_start_updates", "0", "must be >= 1"),
        ("total_updates", "0", "must be >= 1"),
        ("warm_start_updates", "10000", "must be less than"),
        ("lora_rank", "16", "fixes lora_rank=8"),
        ("lora_alpha", "32", "fixes lora_rank=8"),
    ],
)
def test_bundled_adaptation_rejects_invalid_contract(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    source = (ROOT / "configs" / "nextleg.toml").read_text()
    valid_values = {
        "warm_start_updates": "2000",
        "total_updates": "10000",
        "lora_rank": "8",
        "lora_alpha": "16",
    }
    adaptation = (
        "\n[adaptation]\n"
        "warm_start_updates = 2000\n"
        "total_updates = 10000\n"
        "lora_rank = 8\n"
        "lora_alpha = 16\n"
    ).replace(f"{field} = {valid_values[field]}", f"{field} = {value}")
    path = tmp_path / "invalid-bundled.toml"
    path.write_text(
        source.replace('mode = "full_finetune"', 'mode = "lora_r8_alpha16_head_warmstart"')
        + adaptation
    )
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_legacy_production_config_defaults_to_unbound_csv() -> None:
    config = load_config(ROOT / "configs" / "nextleg.toml")
    assert config.data.file_format == "csv"
    assert config.data.corpus_manifest_path is None
    assert config.data.corpus_manifest_sha256 == ""


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_foundation_config_rejects_non_finite_discontinuity_threshold(
    tmp_path: Path, value: str
) -> None:
    source = (ROOT / "configs" / "nextleg.toml").read_text()
    path = tmp_path / "invalid-threshold.toml"
    path.write_text(
        source.replace(
            "target_reserve = 712",
            f"target_reserve = 712\nmax_relative_close_jump = {value}",
        )
    )

    with pytest.raises(ConfigError, match="must be finite"):
        load_config(path)


def test_parquet_config_requires_manifest_binding(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "smoke.toml").read_text()
    path = tmp_path / "unbound-parquet.toml"
    path.write_text(source.replace('root = "synthetic"', 'root = "/data"\nfile_format = "parquet"'))

    with pytest.raises(ConfigError, match="Parquet data requires"):
        load_config(path)


def test_learning_rate_uses_per_step_warmup_then_cosine_decay() -> None:
    training = load_config(ROOT / "configs" / "nextleg.toml").training
    steps_per_epoch = 200
    assert training.learning_rate_for_step(1, steps_per_epoch) == pytest.approx(
        training.learning_rate / 2000
    )
    assert training.learning_rate_for_step(1999, steps_per_epoch) == pytest.approx(
        training.learning_rate * 1999 / 2000
    )
    assert training.learning_rate_for_step(2000, steps_per_epoch) == pytest.approx(
        training.learning_rate
    )
    assert training.learning_rate_for_step(24000, steps_per_epoch) == pytest.approx(0.0)


def test_probe_config_is_strictly_bounded() -> None:
    config = load_config(ROOT / "configs" / "nextleg-parquet-v2-probe.toml")
    production = load_config(ROOT / "configs" / "nextleg-parquet-v2.toml")
    assert config.run.device == "mps"
    assert config.training.epochs == 1
    assert config.training.batch_size == 36
    assert config.training.max_steps_per_epoch == 32
    assert config.training.validation_max_steps == 1
    assert config.training.resume is False
    assert config.data == production.data
    assert config.model == production.model
    assert config.target == production.target
    assert config.evaluation == production.evaluation
    assert config.export == production.export


def test_bounded_validation_must_cover_every_stream(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "nextleg-parquet-v2-probe.toml").read_text()
    path = tmp_path / "under-sampled.toml"
    path.write_text(source.replace("batch_size = 36", "batch_size = 8"))
    with pytest.raises(ConfigError, match="at least one sample per configured stream"):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "smoke.toml").read_text()
    path = tmp_path / "bad.toml"
    path.write_text(source.replace("seed = 7", "seed = 7\nsurprise = true"))
    with pytest.raises(ConfigError, match=r"unknown \[run\] keys: surprise"):
        load_config(path)


def test_two_leg_reserve_is_enforced(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "smoke.toml").read_text()
    path = tmp_path / "leaky.toml"
    path.write_text(source.replace("target_reserve = 192", "target_reserve = 191"))
    with pytest.raises(ConfigError, match=r"max_context \+ 2 \* target.leg_cap \(192\)"):
        load_config(path)


def test_feature_order_is_strict(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "smoke.toml").read_text()
    path = tmp_path / "reordered.toml"
    path.write_text(
        source.replace(
            '["open", "high", "low", "close", "volume"]',
            '["open", "low", "high", "close", "volume"]',
        )
    )
    with pytest.raises(ConfigError, match="must be ordered exactly"):
        load_config(path)
