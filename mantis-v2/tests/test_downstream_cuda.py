from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from mantis_v2 import downstream_pipeline
from mantis_v2.cli import _parser
from mantis_v2.config import ConfigError
from mantis_v2.downstream_config import load_downstream_config
from mantis_v2.embedding import (
    EmbeddingContractError,
    LoadedFoundation,
    iter_symbol_embeddings,
    resolve_embedding_device,
)
from mantis_v2.embedding_artifacts import (
    EmbeddingArtifactError,
    EmbeddingIdentity,
    EmbeddingPerformance,
    compare_embedding_parity,
    publish_embedding_pair,
    scan_embedding_pairs,
    validate_embedding_identity,
)
from mantis_v2.embedding_qualification import qualify_embedding_files
from mantis_v2.model import sha256_file

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


def test_embedding_resume_starts_at_first_uncommitted_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_with_device(tmp_path, "cpu")
    market = pd.DataFrame(
        np.arange(400, dtype=np.float32).reshape(80, 5),
        columns=config.data.feature_columns,
    )
    monkeypatch.setattr("mantis_v2.embedding.load_market_frame", lambda *_args: market)
    candidates = pd.DataFrame(
        {f"{timeframe}_index": np.arange(40, 45) for timeframe in config.data.timeframes}
    )

    class FixtureModel(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.ones((len(values), 2), dtype=torch.float32)

    foundation = LoadedFoundation(
        model=FixtureModel(),
        device=torch.device("cpu"),
        weights_path=tmp_path / "weights",
        weights_sha256="a" * 64,
        validation_evidence_path=tmp_path / "evaluation",
        validation_evidence_sha256="b" * 64,
        embedding_dim=2,
        manifest={},
    )

    batches = list(iter_symbol_embeddings(candidates, "NQ", config, foundation, start_row=3))

    assert [(start, stop) for start, stop, _features in batches] == [(3, 5)]
    assert batches[0][2].shape == (2, 40)


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


def test_cpu_fixture_qualifies_diagnostic_candidate_with_pinned_source_and_receipts(
    tmp_path: Path,
) -> None:
    foundation_manifest = tmp_path / "foundation.json"
    foundation_manifest.write_text(
        json.dumps(
            {
                "export_role": "diagnostic_candidate",
                "config": {
                    "model": {
                        "source_repository": "vfeofanov/mantis",
                        "source_revision": "0c94f8ceb9f1d1421dd292ed917090df8c31605b",
                        "hub_model": "paris-noah/MantisV2",
                        "hub_revision": "99fe0f548960e272fbfa4b82fd9b5b5956779dfd",
                        "weights_sha256": (
                            "49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1"
                        ),
                    }
                },
            }
        )
    )
    identity = EmbeddingIdentity(
        **{
            **_identity(role="diagnostic_candidate").__dict__,
            "foundation_export_sha256": sha256_file(foundation_manifest),
        }
    )
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(asdict(identity)))
    features = np.arange(24, dtype=np.float32).reshape(3, 8)
    cuda = features + 0.001
    metadata = pd.DataFrame({"symbol": ["NQ"] * 3, "row": [0, 1, 2]})
    cpu_features = tmp_path / "cpu.npy"
    cuda_features = tmp_path / "cuda.npy"
    cpu_metadata = tmp_path / "cpu.parquet"
    cuda_metadata = tmp_path / "cuda.parquet"
    np.save(cpu_features, features)
    np.save(cuda_features, cuda)
    metadata.to_parquet(cpu_metadata, index=False)
    metadata.to_parquet(cuda_metadata, index=False)
    shard_directory = tmp_path / "shards"
    publish_embedding_pair(shard_directory, 0, 0, features, metadata, identity)
    performance = EmbeddingPerformance.build(
        rows=3,
        duration_seconds=1.0,
        data_wait_seconds=0.1,
        peak_vram_bytes=100,
        peak_rss_bytes=200,
        disk_bytes=300,
        checkpoint_free_restart=True,
        measured_timeframes=4,
        projected_timeframes=4,
    )
    performance_path = tmp_path / "performance.json"
    performance_path.write_text(json.dumps(asdict(performance)))

    result = qualify_embedding_files(
        config_path=ROOT / "configs" / "cuda-embedding-qualification.toml",
        identity_path=identity_path,
        foundation_manifest_path=foundation_manifest,
        cpu_features_path=cpu_features,
        cuda_features_path=cuda_features,
        cpu_metadata_path=cpu_metadata,
        cuda_metadata_path=cuda_metadata,
        shard_directory=shard_directory,
        performance_path=performance_path,
        output_path=tmp_path / "qualified.json",
    )

    assert result["qualified"] is True
    assert result["purpose"] == "matrix_scoring"
    assert result["committed_pairs"] == 1


def test_embedding_qualification_cli_is_discoverable() -> None:
    args = _parser().parse_args(
        [
            "cuda-embedding-qualify",
            "--qualification-config",
            "qualification.toml",
            "--identity",
            "identity.json",
            "--foundation-manifest",
            "foundation.json",
            "--cpu-features",
            "cpu.npy",
            "--cuda-features",
            "cuda.npy",
            "--cpu-metadata",
            "cpu.parquet",
            "--cuda-metadata",
            "cuda.parquet",
            "--shard-directory",
            "shards",
            "--performance",
            "performance.json",
            "--output",
            "qualified.json",
        ]
    )
    assert args.command == "cuda-embedding-qualify"
    assert args.output == Path("qualified.json")
