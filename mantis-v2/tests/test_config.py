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
    assert config.run.allow_overwrite is False


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
