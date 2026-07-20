from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from mantis_v2 import cli
from mantis_v2.rl_config import load_rl_config
from mantis_v2.rl_provenance import (
    RlProvenanceError,
    source_digest,
    write_rl_dry_run_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _configured_fixture(tmp_path: Path):
    repository = tmp_path / "repo"
    source = repository / "mantis-v2" / "src" / "mantis_v2" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    lock = repository / "uv.lock"
    lock.write_text("locked\n")
    inputs = {}
    for name in ("downstream", "corpus", "embedding", "foundation", "weights"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"identity": name}) + "\n")
        inputs[name] = path
    inputs["rule_contract"] = tmp_path / "topstep-rules.toml"
    inputs["rule_contract"].write_text(
        (ROOT / "configs" / "topstep-100k-2026-07-20.toml").read_text()
    )
    weights_digest = hashlib.sha256(inputs["weights"].read_bytes()).hexdigest()
    corpus_digest = hashlib.sha256(inputs["corpus"].read_bytes()).hexdigest()
    inputs["embedding"].write_text(json.dumps({"foundation_weights_sha256": weights_digest}) + "\n")
    inputs["foundation"].write_text(json.dumps({"weights_sha256": weights_digest}) + "\n")
    embedding_digest = hashlib.sha256(inputs["embedding"].read_bytes()).hexdigest()
    sealed = tmp_path / "sealed-holdout"
    sealed.mkdir()
    inputs["downstream"].write_text(
        "\n".join(
            (
                "[data]",
                f'root = "{sealed}"',
                f'corpus_manifest_path = "{inputs["corpus"]}"',
                f'corpus_manifest_sha256 = "{corpus_digest}"',
                'holdout_start = "2026-01-01T00:00:00+00:00"',
                "[foundation]",
                f'manifest_path = "{inputs["foundation"]}"',
                f'weights_sha256 = "{weights_digest}"',
                "[walk_forward]",
                f'embed_manifest_path = "{inputs["embedding"]}"',
                f'embed_manifest_sha256 = "{embedding_digest}"',
                "[evaluation]",
                "allow_holdout = false",
                "",
            )
        )
    )
    output = tmp_path / "outputs"
    base = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    upstream = replace(
        base.upstream,
        source_digest=source_digest(repository),
        lock_digest=digest(lock),
        rule_contract_path=inputs["rule_contract"],
        rule_contract_sha256=digest(inputs["rule_contract"]),
        downstream_config_path=inputs["downstream"],
        downstream_config_sha256=digest(inputs["downstream"]),
        corpus_manifest_path=inputs["corpus"],
        corpus_manifest_sha256=digest(inputs["corpus"]),
        embedding_manifest_path=inputs["embedding"],
        embedding_manifest_sha256=digest(inputs["embedding"]),
        foundation_manifest_path=inputs["foundation"],
        foundation_manifest_sha256=digest(inputs["foundation"]),
        foundation_weights_path=inputs["weights"],
        foundation_weights_sha256=digest(inputs["weights"]),
    )
    config = replace(
        base,
        upstream=upstream,
        run=replace(base.run, name="fixture-run", artifact_root=output),
    )
    return config, repository, inputs, source, lock, sealed


def test_dry_run_writes_atomic_no_overwrite_identity_manifest(tmp_path: Path) -> None:
    config, repository, _inputs, _source, _lock, _sealed = _configured_fixture(tmp_path)

    manifest = write_rl_dry_run_manifest(config, repository)
    path = config.run.artifact_root / config.run.name / "dry-run-manifest.json"

    assert json.loads(path.read_text()) == manifest
    assert set(manifest["identities"]) == {
        "source",
        "lock",
        "corpus",
        "embedding",
        "foundation",
        "config",
        "rule",
        "fee",
        "output",
    }
    assert not list(path.parent.glob("*.tmp"))
    with pytest.raises(RlProvenanceError, match="already exists"):
        write_rl_dry_run_manifest(config, repository)


@pytest.mark.parametrize(
    "identity",
    (
        "source",
        "lock",
        "rule_contract",
        "downstream",
        "corpus",
        "embedding",
        "foundation",
        "weights",
    ),
)
def test_dry_run_rejects_each_modified_upstream_identity(tmp_path: Path, identity: str) -> None:
    config, repository, inputs, source, lock, _sealed = _configured_fixture(tmp_path)
    target = source if identity == "source" else lock if identity == "lock" else inputs[identity]
    target.write_text(target.read_text() + "modified\n")

    with pytest.raises(RlProvenanceError, match=f"{identity}.*digest mismatch"):
        write_rl_dry_run_manifest(config, repository)


def test_dry_run_rejects_incompatible_rule_contract_with_refreshed_digest(
    tmp_path: Path,
) -> None:
    config, repository, inputs, _source, _lock, _sealed = _configured_fixture(tmp_path)
    rules = inputs["rule_contract"]
    rules.write_text(rules.read_text().replace('force_flat = "15:10"', 'force_flat = "15:11"'))
    digest = hashlib.sha256(rules.read_bytes()).hexdigest()
    config = replace(
        config,
        upstream=replace(config.upstream, rule_contract_sha256=digest),
    )

    with pytest.raises(
        RlProvenanceError, match=r"rule contract value mismatch: session.force_flat"
    ):
        write_rl_dry_run_manifest(config, repository)


@pytest.mark.parametrize(
    ("old", "new", "field"),
    (
        ("corpus.json", "other-corpus.json", "data.corpus_manifest_path"),
        (
            'corpus_manifest_sha256 = "',
            'corpus_manifest_sha256 = "0',
            "data.corpus_manifest_sha256",
        ),
        ("foundation.json", "other-foundation.json", "foundation.manifest_path"),
        (
            'weights_sha256 = "',
            'weights_sha256 = "0',
            "foundation.weights_sha256",
        ),
        ("embedding.json", "other-embedding.json", "walk_forward.embed_manifest_path"),
        (
            'embed_manifest_sha256 = "',
            'embed_manifest_sha256 = "0',
            "walk_forward.embed_manifest_sha256",
        ),
        (
            "2026-01-01T00:00:00+00:00",
            "2026-02-01T00:00:00+00:00",
            "data.holdout_start",
        ),
        ("allow_holdout = false", "allow_holdout = true", "evaluation.allow_holdout"),
    ),
)
def test_dry_run_rejects_each_nested_downstream_identity_mismatch(
    tmp_path: Path, old: str, new: str, field: str
) -> None:
    config, repository, inputs, _source, _lock, _sealed = _configured_fixture(tmp_path)
    downstream = inputs["downstream"]
    downstream.write_text(downstream.read_text().replace(old, new, 1))
    digest = hashlib.sha256(downstream.read_bytes()).hexdigest()
    config = replace(
        config,
        upstream=replace(config.upstream, downstream_config_sha256=digest),
    )

    with pytest.raises(RlProvenanceError, match=rf"identity mismatch: {field}"):
        write_rl_dry_run_manifest(config, repository)


@pytest.mark.parametrize("identity", ("embedding", "foundation"))
def test_dry_run_rejects_nested_manifest_foundation_mismatch(tmp_path: Path, identity: str) -> None:
    config, repository, inputs, _source, _lock, _sealed = _configured_fixture(tmp_path)
    manifest = inputs[identity]
    key = "foundation_weights_sha256" if identity == "embedding" else "weights_sha256"
    manifest.write_text(json.dumps({key: "0" * 64}) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    digest_field = f"{identity}_manifest_sha256"
    replacements = {digest_field: digest}
    if identity == "embedding":
        downstream = inputs["downstream"]
        downstream.write_text(
            downstream.read_text().replace(config.upstream.embedding_manifest_sha256, digest)
        )
        replacements["downstream_config_sha256"] = hashlib.sha256(
            downstream.read_bytes()
        ).hexdigest()
    config = replace(
        config,
        upstream=replace(config.upstream, **replacements),
    )

    with pytest.raises(
        RlProvenanceError, match=rf"{identity} manifest foundation identity mismatch"
    ):
        write_rl_dry_run_manifest(config, repository)


def test_committed_rl_configs_pin_current_source_and_rule_contract() -> None:
    repository = ROOT.parent
    expected_source = source_digest(repository)
    for name in ("rl-entry-smoke.toml", "rl-entry-topstep-100k.toml"):
        config = load_rl_config(ROOT / "configs" / name)
        rule_path = repository / config.upstream.rule_contract_path

        assert config.upstream.source_digest == expected_source
        assert hashlib.sha256(rule_path.read_bytes()).hexdigest() == (
            config.upstream.rule_contract_sha256
        )


def test_dry_run_never_reads_the_sealed_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, repository, _inputs, _source, _lock, sealed = _configured_fixture(tmp_path)
    original_open = Path.open
    attempted_holdout_reads: list[Path] = []

    def guarded_open(path: Path, *args, **kwargs):
        if path == sealed or sealed in path.parents:
            attempted_holdout_reads.append(path)
            raise AssertionError(f"sealed holdout read attempted: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    manifest = write_rl_dry_run_manifest(config, repository)

    assert manifest["sealed_holdout"]["accessed"] is False
    assert manifest["sealed_holdout"]["start"] == "2026-01-01 00:00:00+00:00"
    assert attempted_holdout_reads == []


def test_atomic_publication_does_not_overwrite_a_concurrent_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, repository, _inputs, _source, _lock, _sealed = _configured_fixture(tmp_path)
    destination = config.run.artifact_root / config.run.name / "dry-run-manifest.json"
    original_link = __import__("os").link

    def racing_link(source, target):
        Path(target).write_text("concurrent-writer\n")
        return original_link(source, target)

    monkeypatch.setattr("os.link", racing_link)

    with pytest.raises(RlProvenanceError, match="already exists"):
        write_rl_dry_run_manifest(config, repository)
    assert destination.read_text() == "concurrent-writer\n"


def test_cli_exposes_bounded_rl_dry_run(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    expected = {"stage": "rl-entry-dry-run"}
    config = load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")
    monkeypatch.setattr(cli, "load_rl_config", lambda _path: config, raising=False)
    monkeypatch.setattr(
        cli,
        "write_rl_dry_run_manifest",
        lambda _config: expected,
        raising=False,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["mantis-v2", "rl-dry-run", "--config", "rl.toml"],
    )

    cli.main()

    assert json.loads(capsys.readouterr().out) == expected
