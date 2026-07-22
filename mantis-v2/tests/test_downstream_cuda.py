from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mantis_v2 import downstream_pipeline
from mantis_v2.config import ConfigError
from mantis_v2.downstream_config import load_downstream_config
from mantis_v2.embedding import EmbeddingContractError, resolve_embedding_device
from mantis_v2.embedding_artifacts import (
    EmbeddingArtifactError,
    EmbeddingIdentity,
    EmbeddingPerformance,
    compare_embedding_parity,
    publish_embedding_pair,
    scan_embedding_pairs,
    validate_embedding_identity,
)

ROOT = Path(__file__).resolve().parents[1]
FOUR_TIMEFRAMES = ("1min", "3min", "5min", "15min")


def _config_with_device(tmp_path: Path, device: str):
    source = (ROOT / "configs" / "downstream-smoke.toml").read_text()
    path = tmp_path / f"downstream-{device}.toml"
    path.write_text(source.replace('device = "cpu"', f'device = "{device}"'))
    return load_downstream_config(path)


def test_downstream_device_is_explicit_and_cuda_is_a_valid_selection(tmp_path: Path) -> None:
    assert _config_with_device(tmp_path, "cuda").run.device == "cuda"
    assert _config_with_device(tmp_path, "cpu").run.device == "cpu"

    with pytest.raises(ConfigError, match="run.device must be one of: cpu, cuda, mps"):
        _config_with_device(tmp_path, "auto")


def test_cuda_selection_fails_closed_with_injected_capabilities() -> None:
    assert (
        str(
            resolve_embedding_device(
                "cuda", cuda_available=lambda: True, mps_available=lambda: False
            )
        )
        == "cuda"
    )
    assert (
        str(
            resolve_embedding_device("cpu", cuda_available=lambda: True, mps_available=lambda: True)
        )
        == "cpu"
    )

    with pytest.raises(EmbeddingContractError, match="CUDA was requested but is unavailable"):
        resolve_embedding_device("cuda", cuda_available=lambda: False, mps_available=lambda: False)


def test_unavailable_cuda_fails_before_prepare_read_or_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_with_device(tmp_path, "cuda")
    config = config.__class__(
        **{
            **config.__dict__,
            "run": config.run.__class__(
                **{
                    **config.run.__dict__,
                    "artifact_root": tmp_path / "artifacts",
                }
            ),
        }
    )
    monkeypatch.setattr("mantis_v2.embedding.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        downstream_pipeline,
        "_manifest",
        lambda *_args, **_kwargs: pytest.fail("prepare manifest was read before device preflight"),
    )

    with pytest.raises(EmbeddingContractError, match="CUDA was requested but is unavailable"):
        downstream_pipeline.embed(config)

    assert not (tmp_path / "artifacts" / config.run.name / "embed").exists()


def _identity(*, role: str = "promoted") -> EmbeddingIdentity:
    return EmbeddingIdentity(
        export_role=role,
        foundation_export_sha256="a" * 64,
        producer_config_sha256="b" * 64,
        corpus_sha256="c" * 64,
        source_digest="d" * 64,
        lock_digest="e" * 64,
        timeframes=FOUR_TIMEFRAMES,
        feature_width=8,
    )


def test_embedding_identity_separates_diagnostic_candidates_from_promoted_exports() -> None:
    validate_embedding_identity(_identity(), purpose="production")
    validate_embedding_identity(_identity(role="diagnostic_candidate"), purpose="matrix_scoring")

    with pytest.raises(EmbeddingArtifactError, match="promoted export"):
        validate_embedding_identity(_identity(role="diagnostic_candidate"), purpose="production")
    with pytest.raises(EmbeddingArtifactError, match="diagnostic_candidate"):
        validate_embedding_identity(_identity(), purpose="matrix_scoring")
    with pytest.raises(EmbeddingArtifactError, match="ordered four-timeframe"):
        validate_embedding_identity(
            EmbeddingIdentity(**{**_identity().__dict__, "timeframes": FOUR_TIMEFRAMES[:3]}),
            purpose="production",
        )


def test_pair_publication_resumes_partial_pair_and_rehashes_every_identity(
    tmp_path: Path,
) -> None:
    features = np.arange(24, dtype=np.float32).reshape(3, 8)
    metadata = pd.DataFrame({"symbol": ["NQ"] * 3, "row": [0, 1, 2]})

    def interrupt(step: str) -> None:
        if step == "features":
            raise RuntimeError("injected interruption")

    with pytest.raises(RuntimeError, match="injected interruption"):
        publish_embedding_pair(
            tmp_path, 0, 0, features, metadata, _identity(), after_write=interrupt
        )
    partial = tmp_path / "features-00000.npy"
    assert partial.is_file()
    assert not (tmp_path / "pair-00000.json").exists()
    partial_mtime = partial.stat().st_mtime_ns

    record = publish_embedding_pair(tmp_path, 0, 0, features, metadata, _identity())
    assert partial.stat().st_mtime_ns == partial_mtime
    assert record["row_start"] == 0
    assert record["row_stop"] == 3
    assert scan_embedding_pairs(tmp_path, _identity()) == (record,)

    changed_role = _identity(role="diagnostic_candidate")
    with pytest.raises(EmbeddingArtifactError, match="identity mismatch"):
        scan_embedding_pairs(tmp_path, changed_role)

    partial.write_bytes(partial.read_bytes() + b"changed")
    with pytest.raises(EmbeddingArtifactError, match=r"features (size|digest) mismatch"):
        scan_embedding_pairs(tmp_path, _identity())


def test_injected_cpu_cuda_parity_requires_finite_values_metadata_and_tolerances() -> None:
    cpu = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    cuda = cpu + np.asarray([[0.001, 0.0], [0.0, -0.001]], dtype=np.float32)
    metadata = pd.DataFrame({"symbol": ["NQ", "ES"], "row": [3, 4]})

    evidence = compare_embedding_parity(cpu, cuda, metadata, metadata.copy())
    assert evidence["max_abs_difference"] <= 0.01
    assert evidence["minimum_row_cosine"] >= 0.999
    assert evidence["metadata_exact"] is True

    with pytest.raises(EmbeddingArtifactError, match="metadata order"):
        compare_embedding_parity(cpu, cuda, metadata, metadata.iloc[::-1].reset_index(drop=True))
    with pytest.raises(EmbeddingArtifactError, match="finite"):
        compare_embedding_parity(cpu, np.asarray([[np.nan, 0.0], [0.0, 2.0]]), metadata, metadata)


def test_performance_manifest_records_measured_and_projected_four_timeframe_costs() -> None:
    performance = EmbeddingPerformance.build(
        rows=400,
        duration_seconds=2.0,
        data_wait_seconds=0.25,
        peak_vram_bytes=1024,
        peak_rss_bytes=2048,
        disk_bytes=3200,
        checkpoint_free_restart=True,
        measured_timeframes=2,
        projected_timeframes=4,
    )

    assert performance.rows_per_second == 200.0
    assert performance.disk_bytes_per_row == 8.0
    assert performance.projected_four_timeframe_disk_bytes == 6400
    assert performance.checkpoint_free_restart is True
