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


def test_learning_rate_uses_warmup_then_cosine_decay() -> None:
    training = load_config(ROOT / "configs" / "nextleg.toml").training
    assert training.learning_rate_for_epoch(0) == pytest.approx(training.learning_rate / 10)
    assert training.learning_rate_for_epoch(9) == pytest.approx(training.learning_rate)
    assert training.learning_rate_for_epoch(10) == pytest.approx(training.learning_rate)
    assert training.learning_rate_for_epoch(119) == pytest.approx(0.0)


def test_probe_config_is_strictly_bounded() -> None:
    config = load_config(ROOT / "configs" / "nextleg-mps-probe.toml")
    assert config.run.device == "mps"
    assert config.training.epochs == 1
    assert config.training.batch_size == 36
    assert config.training.max_steps_per_epoch == 1
    assert config.training.validation_max_steps == 1
    assert config.training.resume is False


def test_bounded_validation_must_cover_every_stream(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "nextleg-mps-probe.toml").read_text()
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
