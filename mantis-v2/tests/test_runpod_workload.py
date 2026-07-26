from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from mantis_v2.cli import _parser
from mantis_v2.foundation_matrix import render_initial_plan
from mantis_v2.frozen_expected_r import _json_digest, write_paid_preflight
from mantis_v2.runpod_config import load_launch_authorization, load_spend_ledger
from mantis_v2.runpod_rest_adapter import OPENAPI_IDENTITY, OPENAPI_SHA256, OPENAPI_VERSION
from mantis_v2.runpod_s3 import RemoteObject
from mantis_v2.runpod_s3_workload import RunpodS3WorkloadIO
from mantis_v2.runpod_workload import (
    WorkloadError,
    _bounded_call,
    _load_authorization_for_role,
    _load_manifest_document,
    bind_workload_decision,
    execute_workload_manifest,
    seal_workload_manifest,
    sign_heartbeat,
    supervise_workload,
    validate_workload_manifest,
)
from mantis_v2.transfer_bundle import load_bundle_manifest, write_bundle_manifest


def _file(path: Path, value: bytes) -> dict[str, object]:
    path.write_bytes(value)
    return {
        "controller_path": str(path.resolve()),
        "pod_path": str(path.resolve()),
        "sha256": hashlib.sha256(value).hexdigest(),
        "size": len(value),
    }


def _existing_file(path: Path) -> dict[str, object]:
    value = path.read_bytes()
    return {
        "controller_path": str(path.resolve()),
        "pod_path": str(path.resolve()),
        "sha256": hashlib.sha256(value).hexdigest(),
        "size": len(value),
    }


def _spec(tmp_path: Path) -> dict[str, object]:
    lock = _file(tmp_path / "uv.lock", b"lock")
    image_check = _file(
        tmp_path / "image-check.json",
        json.dumps(
            {
                "schema_version": 1,
                "passed": True,
                "scope": "static_image",
                "inventory": {
                    "identities": {
                        "source_revision": "a" * 40,
                        "lock_sha256": lock["sha256"],
                    }
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    )
    dataset = _file(tmp_path / "dataset.json", b"dataset")
    bundle_source = tmp_path / "bundle-source"
    (bundle_source / "corpus").mkdir(parents=True, exist_ok=True)
    (bundle_source / "corpus" / "manifest.json").write_bytes(b"dataset")
    bundle_manifest_path = tmp_path / "input-bundle.json"
    bundle = (
        load_bundle_manifest(bundle_manifest_path)
        if bundle_manifest_path.exists()
        else write_bundle_manifest(bundle_source, ("corpus/manifest.json",), bundle_manifest_path)
    )
    bundle_manifest = _existing_file(bundle_manifest_path)
    dataset["pod_path"] = f"/workspace/mantis/inputs/{bundle.bundle_digest}/corpus/manifest.json"
    config_root = Path(__file__).resolve().parents[1] / "configs"
    base_config = tmp_path / "nextleg-parquet-v1.toml"
    base_config.write_text(
        (config_root / "nextleg-parquet-v1.toml")
        .read_text()
        .replace(
            "2d8cf8b708c7c743c849059410b3654bc44f9fe4f7d795928de82526e120d703",
            str(dataset["sha256"]),
        )
    )
    matrix_config = tmp_path / "foundation-matrix-v1.toml"
    matrix_config.write_text(
        (config_root / "foundation-matrix-v1.toml")
        .read_text()
        .replace('base_config = "nextleg-parquet-v1.toml"', f'base_config = "{base_config}"')
        .replace(
            "4d12c3f334dba95fb045905d89654f16d63e75b68f8423ae8726eac06d70fba6",
            bundle.bundle_digest,
        )
    )
    plan_path = render_initial_plan(matrix_config, tmp_path / "plans")
    plan = json.loads(plan_path.read_text())
    cell = plan["cells"][0]
    config_path = plan_path.parent / cell["config_path"]
    config = _existing_file(config_path)
    matrix_plan = _existing_file(plan_path)
    matrix_base_config = _existing_file(plan_path.parent / "base-config.toml")
    authorization_expiry = datetime.now(UTC) + timedelta(hours=1)
    authorization_payload = {
        "schema_version": 1,
        "authorization_id": "qualification-seed42",
        "subject_digest": "f" * 64,
        "authorized_at": (authorization_expiry - timedelta(minutes=5)).isoformat(),
        "expires_at": authorization_expiry.isoformat(),
        "maximum_projected_spend_usd": "1.00",
        "approver": "lgbarn",
        "autopay_disabled": True,
        "ordinary_launch_cutoff_usd": "125.00",
        "campaign_ceiling_usd": "150.00",
        "recovery_authorized": False,
    }
    authorization = _file(
        tmp_path / "authorization.json",
        (json.dumps(authorization_payload, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    ledger_payload = {
        "schema_version": 1,
        "actual_spend_usd": "0.00",
        "reserved_spend_usd": "0.00",
        "bucket_actual_spend_usd": {
            "storage": "0.00",
            "qualification": "0.00",
            "production": "0.00",
            "recovery": "0.00",
        },
        "bucket_reserved_spend_usd": {
            "storage": "0.00",
            "qualification": "0.00",
            "production": "0.00",
            "recovery": "0.00",
        },
        "active_reservations": [],
        "consumed_authorization_digests": [],
    }
    ledger = _file(
        tmp_path / "ledger.json",
        (json.dumps(ledger_payload, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    token = _file(tmp_path / "heartbeat.token", b"token")
    return {
        "schema_version": 1,
        "run_id": cell["run_name"],
        "source_revision": "a" * 40,
        "dependency_lock": lock,
        "image": {
            "ref": "ghcr.io/lgbarn/mantis@sha256:" + "b" * 64,
            "self_check": image_check,
        },
        "experiment_config": config,
        "matrix_plan": matrix_plan,
        "matrix_base_config": matrix_base_config,
        "input_bundle": {
            "manifest": bundle_manifest,
            "bundle_digest": bundle.bundle_digest,
            "incoming_root": (f"/workspace/mantis/transfer/incoming/{bundle.bundle_digest}/files"),
            "final_parent": "/workspace/mantis/inputs",
        },
        "dataset_manifest": dataset,
        "spend_ledger": ledger,
        "foundation_checkpoint": {
            "repository": "paris-noah/MantisV2",
            "revision": "99fe0f548960e272fbfa4b82fd9b5b5956779dfd",
            "sha256": "49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1",
        },
        "start_command": [
            "uv",
            "run",
            "mantis-v2",
            "foundation-matrix-cell",
            "--plan",
            str(plan_path),
            "--cell-id",
            cell["cell_id"],
        ],
        "artifacts": {
            "pod": f"/workspace/mantis/runs/{cell['run_name']}",
            "controller": str(tmp_path / "artifacts"),
            "backup": "/Volumes/Storage/trading-research/artifacts/mantis-foundation-model",
        },
        "resume": {
            "enabled": True,
            "same_run_only": True,
            "provenance_required": True,
        },
        "monitor": {
            "tensorboard": "ssh://pod/tensorboard",
            "heartbeat": f"/workspace/mantis/runs/{cell['run_name']}/heartbeat.json",
            "poll_seconds": 30,
            "first_heartbeat_seconds": 600,
            "miss_limit": 4,
            "token": token,
        },
        "maximum_duration_seconds": 7200,
        "quoted_rates": {
            "compute_usd_per_hour": "0.3900",
            "storage_usd_per_gb_hour": "0.000137",
            "storage_gb": 150,
        },
        "budget_guard": {
            "stage": "qualification",
            "reconciled_spend_usd": "0.00",
            "unbilled_live_accrual_usd": "0.00",
            "stage_reconciled_spend_usd": "0.00",
            "next_cell_maximum_usd": "0.903210",
            "shutdown_reserve_usd": "0.06500570833333333333333333333",
        },
        "authorization": {
            **authorization,
            "expires_at": authorization_expiry.isoformat().replace("+00:00", "Z"),
            "autopay_disabled": True,
            "ordinary_launch_cutoff_usd": "125.00",
            "campaign_ceiling_usd": "150.00",
            "recovery_authorized": False,
        },
        "runpodctl": {
            "version": "2.7.2",
            "source_commit": "309512b4926eb7d218bbc8a8f11d380ce54f59c4",
            "binary_sha256": "a016e442fdf12e4642ad3425ea6d624a40882d77accdfa043b5e40a4fd08d037",
        },
    }


def _official_spec(tmp_path: Path) -> dict[str, object]:
    spec = _spec(tmp_path)
    archive = _file(tmp_path / "source.tar.gz", b"source archive")
    archive["pod_path"] = "/workspace/mantis/control/source/source.tar.gz"
    lock = spec["dependency_lock"]
    assert isinstance(lock, dict)
    image_ref = (
        "runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35"
    )
    receipt = _file(
        tmp_path / "official-bootstrap.json",
        json.dumps(
            {
                "schema_version": 1,
                "passed": True,
                "scope": "official_bootstrap",
                "provider_support": "official_runpod_template",
                "template_id": "runpod-torch-v280",
                "image_ref": image_ref,
                "uv_version": "0.9.0",
                "inventory": {
                    "identities": {
                        "source_revision": "a" * 40,
                        "source_tree": "d" * 40,
                        "source_archive_sha256": archive["sha256"],
                        "lock_sha256": lock["sha256"],
                    }
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    )
    receipt["pod_path"] = "/workspace/mantis/control/source/official-bootstrap.json"
    spec["image"] = {"ref": image_ref, "self_check": receipt}
    spec["bootstrap"] = {
        "source_archive": archive,
        "project_root": "/workspace/mantis/runtime/" + "a" * 40,
        "venv_path": "/opt/mantis/venv",
        "uv_version": "0.9.0",
    }
    return spec


def _official_frozen_spec(tmp_path: Path) -> dict[str, object]:
    spec = _official_spec(tmp_path)
    run_id = str(spec["run_id"])
    pod_root = f"/workspace/mantis/runs/{run_id}"
    input_payload = {"schema_version": 1, "rows": 1}
    input_payload["manifest_sha256"] = _json_digest(input_payload)
    input_path = tmp_path / "frozen-input.json"
    input_path.write_text(json.dumps(input_payload))
    bundle_source = tmp_path / "frozen-bundle"
    (bundle_source / "input").mkdir(parents=True)
    (bundle_source / "input" / "manifest.json").write_bytes(input_path.read_bytes())
    config_path = Path(__file__).resolve().parents[1] / "configs/frozen-expected-r-v1.json"
    (bundle_source / "config").mkdir()
    (bundle_source / "config" / "frozen-expected-r-v1.json").write_bytes(config_path.read_bytes())
    bundle_path = tmp_path / "frozen-bundle.json"
    bundle = write_bundle_manifest(
        bundle_source,
        ("config/frozen-expected-r-v1.json", "input/manifest.json"),
        bundle_path,
    )
    input_record = _existing_file(input_path)
    input_record["pod_path"] = (
        f"/workspace/mantis/inputs/{bundle.bundle_digest}/input/manifest.json"
    )
    bundle_record = _existing_file(bundle_path)
    bundle_record["pod_path"] = "/workspace/mantis/control/frozen/input-bundle.json"
    frozen_config = _existing_file(config_path)
    frozen_config["pod_path"] = (
        f"/workspace/mantis/inputs/{bundle.bundle_digest}/config/frozen-expected-r-v1.json"
    )
    exact_command = (
        "uv run mantis-v2 frozen-screen-paid-workload "
        f"--config {frozen_config['pod_path']} --input {input_record['pod_path']} "
        f"--embedding-output {pod_root}/embed --comparison-output {pod_root}/selection.json "
        f"--progress-output {pod_root}/selection.progress.json"
    )
    preflight_path = tmp_path / "frozen-preflight.json"
    write_paid_preflight(
        input_path,
        Path(pod_root),
        preflight_path,
        exact_command=exact_command,
        hourly_rate_usd=0.39,
        budget_usd=10.0,
        deadline_hours=2.0,
        check_duration_seconds=1.0,
        checks={
            "causality_next_fill": True,
            "label_replay_parity": True,
            "topstep_accounting": True,
            "artifact_resume": True,
        },
    )
    preflight = _existing_file(preflight_path)
    preflight["pod_path"] = "/workspace/mantis/control/frozen/preflight.json"
    experiment_path = tmp_path / "frozen-experiment.json"
    experiment_path.write_text(
        json.dumps(
            {
                "evaluation": {"allow_holdout": False},
                "data": {
                    "holdout_start": "2026-01-01T00:00:00+00:00",
                    "corpus_manifest_sha256": input_record["sha256"],
                    "root": str(Path(str(input_record["pod_path"])).parent),
                    "corpus_manifest_path": input_record["pod_path"],
                },
                "model": {
                    "hub_model": "paris-noah/MantisV2",
                    "hub_revision": "99fe0f548960e272fbfa4b82fd9b5b5956779dfd",
                    "weights_sha256": (
                        "49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1"
                    ),
                },
                "run": {"artifact_root": "/workspace/mantis/runs"},
            }
        )
    )
    experiment = _existing_file(experiment_path)
    experiment["pod_path"] = "/workspace/mantis/control/frozen/experiment.json"
    spec.update(
        {
            "workload_kind": "frozen_expected_r",
            "experiment_config": experiment,
            "matrix_plan": preflight,
            "matrix_base_config": frozen_config,
            "input_bundle": {
                "manifest": bundle_record,
                "bundle_digest": bundle.bundle_digest,
                "incoming_root": (
                    f"/workspace/mantis/transfer/incoming/{bundle.bundle_digest}/files"
                ),
                "final_parent": "/workspace/mantis/inputs",
            },
            "dataset_manifest": input_record,
            "start_command": ["bash", "-lc", f"{exact_command} || {exact_command}"],
            "artifacts": {
                "pod": pod_root,
                "controller": str(tmp_path / "frozen-controller"),
                "backup": str(tmp_path / "frozen-backup"),
            },
            "monitor": {
                **spec["monitor"],
                "tensorboard": None,
                "heartbeat": f"{pod_root}/heartbeat.json",
            },
            "maximum_duration_seconds": 7200,
            "quoted_rates": {
                "compute_usd_per_hour": "0.39",
                "storage_usd_per_gb_hour": "0.000137",
                "storage_gb": 150,
            },
            "budget_guard": {
                **spec["budget_guard"],
                "next_cell_maximum_usd": "0.903210",
            },
        }
    )
    return spec


def test_launch_manifest_is_complete_content_addressed_and_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "sealed"
    spec = _spec(tmp_path)
    first = seal_workload_manifest(spec, output)
    second = seal_workload_manifest(spec, output)

    assert first == second
    manifest = validate_workload_manifest(first)
    assert first.parent.name == manifest["manifest_digest"]
    assert manifest["start_command"][0:3] == ["uv", "run", "mantis-v2"]
    assert manifest["monitor"]["poll_seconds"] == 30
    assert manifest["monitor"]["first_heartbeat_seconds"] == 600
    assert manifest["monitor"]["miss_limit"] == 4


def test_bound_pod_manifest_copy_does_not_require_controller_digest_directory(
    tmp_path: Path,
) -> None:
    controller_path = seal_workload_manifest(_spec(tmp_path), tmp_path / "sealed")
    pod_path = tmp_path / "workspace" / "mantis" / "control" / "frozen" / "launch-manifest.json"
    pod_path.parent.mkdir(parents=True)
    pod_path.write_bytes(controller_path.read_bytes())

    with pytest.raises(WorkloadError, match="path does not match"):
        _load_manifest_document(pod_path)

    assert _load_manifest_document(pod_path, path_role="pod")["manifest_digest"] == (
        controller_path.parent.name
    )


def test_pod_validation_loads_pod_authorization_path(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    authorization = spec["authorization"]
    assert isinstance(authorization, dict)
    authorization["pod_path"] = authorization["controller_path"]
    authorization["controller_path"] = str(tmp_path / "controller-path-does-not-exist.json")

    assert _load_authorization_for_role(authorization, "pod").approver == "lgbarn"


def test_pod_executor_rejects_manifest_overwritten_by_another_bound_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_path = seal_workload_manifest(_spec(tmp_path), tmp_path / "first-sealed")
    first = json.loads(first_path.read_text())
    second_root = tmp_path / "second"
    second_root.mkdir()
    second_path = seal_workload_manifest(_spec(second_root), second_root / "sealed")
    pod_path = tmp_path / "workspace" / "mantis" / "control" / "frozen" / "launch-manifest.json"
    pod_path.parent.mkdir(parents=True)
    pod_path.write_bytes(second_path.read_bytes())
    promoted: list[dict] = []
    monkeypatch.setattr("mantis_v2.runpod_workload._promote_pod_input_bundle", promoted.append)

    with pytest.raises(WorkloadError, match="decision-bound identity mismatch"):
        execute_workload_manifest(
            pod_path,
            environ={
                "MANTIS_RUN_ID": str(first["run_id"]),
                "MANTIS_WORKLOAD_DIGEST": str(first["manifest_digest"]),
                "RUNPOD_POD_ID": "pod-first",
            },
        )

    assert promoted == []


def test_official_frozen_manifest_runs_embed_and_compare_with_one_retry(tmp_path: Path) -> None:
    spec = _official_frozen_spec(tmp_path)
    manifest_path = seal_workload_manifest(spec, tmp_path / "sealed-frozen")
    manifest = validate_workload_manifest(manifest_path)

    assert manifest["workload_kind"] == "frozen_expected_r"
    assert manifest["start_command"][2].count("frozen-screen-paid-workload") == 2
    assert manifest["monitor"]["poll_seconds"] == 30
    assert manifest["monitor"]["tensorboard"] is None


def test_control_staging_leaves_promoted_bundle_members_in_incoming_tree(tmp_path: Path) -> None:
    def record(name: str, pod_path: str) -> dict[str, object]:
        return _file(tmp_path / name, name.encode()) | {"pod_path": pod_path}

    manifest = {
        "dependency_lock": record("lock", "/workspace/mantis/control/lock"),
        "image": {"self_check": record("image", "/workspace/mantis/control/image")},
        "experiment_config": record("experiment", "/workspace/mantis/control/experiment"),
        "matrix_plan": record("plan", "/workspace/mantis/control/plan"),
        "matrix_base_config": record(
            "config", "/workspace/mantis/inputs/digest/config/config.json"
        ),
        "input_bundle": {"manifest": record("bundle", "/workspace/mantis/control/bundle")},
        "dataset_manifest": record(
            "dataset", "/workspace/mantis/inputs/digest/input/manifest.json"
        ),
        "spend_ledger": record("ledger", "/workspace/mantis/control/ledger"),
        "authorization": record("authorization", "/workspace/mantis/control/authorization"),
        "monitor": {"token": record("token", "/workspace/mantis/control/token")},
        "workload_kind": "frozen_expected_r",
        "manifest_digest": "d" * 64,
    }

    class MemoryAdapter:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        def put_file(self, key: str, source: Path) -> None:
            self.objects[key] = source.read_bytes()

        def get_file(self, key: str, destination: Path) -> Path | None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.objects[key])
            return destination

    io = object.__new__(RunpodS3WorkloadIO)
    io.adapter = MemoryAdapter()
    io.manifest = manifest
    io.manifest_path = tmp_path / "launch-manifest.json"
    io.manifest_path.write_text("manifest")
    io.pod_manifest_path = Path("/workspace/mantis/control/run/launch-manifest.json")

    staged = io.stage_control_files()

    assert not any(key.startswith("mantis/inputs/") for key in staged["uploaded"])


def test_completed_artifacts_replicate_idempotently_to_both_backups(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    pod_root = str(spec["artifacts"]["pod"])
    run_id = str(spec["run_id"])
    controller = tmp_path / "controller-artifacts"
    backup = tmp_path / "external-artifacts"
    spec["artifacts"] = {
        "pod": pod_root,
        "controller": str(controller),
        "backup": str(backup),
    }
    manifest_path = seal_workload_manifest(spec, tmp_path / "sealed")
    manifest = validate_workload_manifest(manifest_path)
    key = f"mantis/runs/{run_id}/train-result.json"
    payload = b'{"status":"complete"}\n'

    class MemoryAdapter:
        def list_objects(self, prefix: str) -> dict[str, RemoteObject]:
            assert prefix == f"mantis/runs/{run_id}"
            return {key: RemoteObject(size=len(payload), etag=None)}

        def get_file(self, object_key: str, destination: Path) -> Path | None:
            assert object_key == key
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            return destination

    io = RunpodS3WorkloadIO(
        adapter=MemoryAdapter(),  # type: ignore[arg-type]
        manifest_path=manifest_path,
        pod_manifest_path="/workspace/mantis/control/run/launch-manifest.json",
        provider=object(),
        state_root=tmp_path / "state",
    )

    io.replicate(manifest)
    receipt_paths = list((controller / "replication-receipts").glob("*.json"))
    assert len(receipt_paths) == 1
    first_receipt = receipt_paths[0].read_bytes()
    io.replicate(manifest)

    assert (controller / "train-result.json").read_bytes() == payload
    assert (backup / "train-result.json").read_bytes() == payload
    assert receipt_paths[0].read_bytes() == first_receipt
    assert (backup / "replication-receipts" / receipt_paths[0].name).read_bytes() == first_receipt


def test_launch_decision_is_immutably_bound_to_exact_immediate_workload(tmp_path: Path) -> None:
    manifest_path = seal_workload_manifest(_spec(tmp_path), tmp_path / "sealed")
    manifest = validate_workload_manifest(manifest_path)
    base_decision = _decision(manifest)
    base_decision.pop("workload")
    unsigned = {key: value for key, value in base_decision.items() if key != "decision_digest"}
    base_decision["decision_digest"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    output = bind_workload_decision(
        manifest_path=manifest_path,
        decision=base_decision,
        pod_manifest_path="/workspace/mantis/control/run/launch-manifest.json",
        output_path=tmp_path / "bound-decision.json",
        evaluated_at=datetime(2026, 7, 22, 12, tzinfo=UTC),
    )
    bound = json.loads(output.read_text())

    assert bound["workload"]["manifest_digest"] == manifest["manifest_digest"]
    assert bound["workload"]["docker_args"].endswith(
        "/workspace/mantis/control/run/launch-manifest.json"
    )
    assert bound["workload"]["environment"] == {
        "MANTIS_RUN_ID": manifest["run_id"],
        "MANTIS_WORKLOAD_DIGEST": manifest["manifest_digest"],
        "MANTIS_WORKLOAD_MANIFEST": "/workspace/mantis/control/run/launch-manifest.json",
        "HF_HOME": (
            "/workspace/mantis/inputs/"
            f"{manifest['input_bundle']['bundle_digest']}/cache/huggingface"
        ),
        "HF_HUB_OFFLINE": "1",
    }


def test_official_template_bootstrap_is_hash_bound_and_uses_no_registry_auth(
    tmp_path: Path,
) -> None:
    manifest_path = seal_workload_manifest(_official_spec(tmp_path), tmp_path / "sealed")
    manifest = validate_workload_manifest(manifest_path)
    decision = _decision(manifest)
    decision.pop("workload")
    decision["image_ref"] = manifest["image"]["ref"]
    decision["template_id"] = "runpod-torch-v280"
    decision["registry_auth_id"] = ""
    unsigned = {key: value for key, value in decision.items() if key != "decision_digest"}
    decision["decision_digest"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    output = bind_workload_decision(
        manifest_path=manifest_path,
        decision=decision,
        pod_manifest_path="/workspace/mantis/control/run/launch-manifest.json",
        output_path=tmp_path / "official-decision.json",
        evaluated_at=datetime(2026, 7, 22, 12, tzinfo=UTC),
    )
    workload = json.loads(output.read_text())["workload"]

    assert "/start.sh" in workload["docker_args"]
    assert "sha256sum -c -" in workload["docker_args"]
    assert "uv sync --frozen --no-dev" in workload["docker_args"]
    assert "uv run mantis-v2" in workload["docker_args"]
    assert "bootstrap.log" in workload["docker_args"]
    assert "bootstrap_exit_status" in workload["docker_args"]
    assert "/uv " not in workload["docker_args"]
    assert workload["environment"]["MANTIS_BASE_IMAGE"] == manifest["image"]["ref"]
    assert workload["environment"]["MANTIS_LOCK_PATH"].endswith("/uv.lock")
    assert workload["environment"]["UV_CACHE_DIR"] == "/opt/mantis/cache"
    assert workload["environment"]["UV_PROJECT_ENVIRONMENT"] == "/opt/mantis/venv"
    assert (
        Path(manifest["matrix_plan"]["pod_path"]).parent.name
        == json.loads(Path(manifest["matrix_plan"]["controller_path"]).read_text())["plan_digest"]
    )


def test_manifest_rejects_flattened_pod_matrix_paths(tmp_path: Path) -> None:
    spec = _official_spec(tmp_path)
    spec["matrix_plan"]["pod_path"] = "/workspace/mantis/control/matrix-plan.json"  # type: ignore[index]
    spec["matrix_base_config"]["pod_path"] = (  # type: ignore[index]
        "/workspace/mantis/control/base-config.toml"
    )

    with pytest.raises(WorkloadError, match="preserve its digest"):
        seal_workload_manifest(spec, tmp_path / "flattened-plan")


def test_manifest_fails_closed_on_tamper_missing_field_holdout_or_bad_attestation(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    del spec["start_command"]
    with pytest.raises(WorkloadError, match="keys"):
        seal_workload_manifest(spec, tmp_path / "missing")

    spec = _spec(tmp_path)
    spec["authorization"]["autopay_disabled"] = False  # type: ignore[index]
    with pytest.raises(WorkloadError, match="auto-pay"):
        seal_workload_manifest(spec, tmp_path / "autopay")

    spec = _spec(tmp_path)
    spec["dataset_manifest"]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(WorkloadError, match="hash"):
        seal_workload_manifest(spec, tmp_path / "tamper")

    spec = _spec(tmp_path)
    spec["dataset_manifest"] = _file(tmp_path / "different-dataset.json", b"different")
    with pytest.raises(WorkloadError, match="dataset manifest differs"):
        seal_workload_manifest(spec, tmp_path / "dataset-mismatch")

    spec = _spec(tmp_path)
    spec["source_revision"] = "f" * 40
    with pytest.raises(WorkloadError, match="image self-check source"):
        seal_workload_manifest(spec, tmp_path / "source-mismatch")

    spec = _spec(tmp_path)
    spec["foundation_checkpoint"]["revision"] = "e" * 40  # type: ignore[index]
    with pytest.raises(WorkloadError, match="foundation checkpoint"):
        seal_workload_manifest(spec, tmp_path / "checkpoint-mismatch")

    spec = _spec(tmp_path)
    spec["budget_guard"]["reconciled_spend_usd"] = "124.50"  # type: ignore[index]
    with pytest.raises(WorkloadError, match="spend ledger"):
        seal_workload_manifest(spec, tmp_path / "budget")

    spec = _spec(tmp_path)
    spec["quoted_rates"]["storage_usd_per_gb_hour"] = "0.000140"  # type: ignore[index]
    with pytest.raises(WorkloadError, match="storage allocation"):
        seal_workload_manifest(spec, tmp_path / "storage-budget")

    spec = _spec(tmp_path)
    spec["budget_guard"]["stage"] = "recovery"  # type: ignore[index]
    with pytest.raises(WorkloadError, match="separate exact authorization"):
        seal_workload_manifest(spec, tmp_path / "recovery-authorization")

    manifest_path = seal_workload_manifest(_spec(tmp_path), tmp_path / "valid")
    manifest = json.loads(manifest_path.read_text())
    manifest["maximum_duration_seconds"] = 9999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(WorkloadError, match="digest"):
        validate_workload_manifest(manifest_path)


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 22, 12, tzinfo=UTC)

    def now(self) -> datetime:
        return self.value

    def sleep(self, seconds: float) -> None:
        assert seconds == 30
        self.value += timedelta(seconds=seconds)


class FakeAdapter:
    def __init__(self) -> None:
        self.live = False
        self.terminations = 0
        self.name = "unset"

    def _pod(self) -> dict[str, object]:
        return {
            "id": "pod-123",
            "name": self.name,
            "desiredStatus": "RUNNING",
            "imageName": "ghcr.io/lgbarn/mantis@sha256:" + "b" * 64,
            "templateId": "template-1",
            "networkVolumeId": "volume-1",
            "costPerHr": "0.3900",
            "vcpuCount": 16,
            "memoryInGb": 62,
            "lastStartedAt": "2026-07-22T12:00:00Z",
        }

    def inventory(self) -> list[dict[str, object]]:
        return [self._pod()] if self.live else []

    def create(self, decision: dict[str, object], deadline: datetime) -> dict[str, object]:
        del deadline
        self.name = str(decision["run_name"])
        self.live = True
        return self._pod()

    def status(self, pod_id: str) -> dict[str, object]:
        assert pod_id == "pod-123" and self.live
        return self._pod()

    def terminate(self, pod_id: str) -> dict[str, object]:
        assert pod_id == "pod-123"
        self.live = False
        self.terminations += 1
        return {"deleted": True, "id": pod_id}

    def billing(self, pod_id: str) -> dict[str, object]:
        return {"pod_id": pod_id, "actual_cost_usd": "0.10"}


def _decision(manifest: dict[str, object]) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "allowed": True,
        "run_name": manifest["run_id"],
        "gpu_type": "NVIDIA A40",
        "gpu_count": 1,
        "vcpu": 16,
        "ram_gb": 62,
        "datacenter_id": "US-GA-1",
        "container_disk_gb": 50,
        "image_ref": "ghcr.io/lgbarn/mantis@sha256:" + "b" * 64,
        "template_id": "template-1",
        "registry_auth_id": "registry-1",
        "volume_id": "volume-1",
        "volume_mount_path": "/workspace",
        "ports": ["22/tcp"],
        "openapi_identity": OPENAPI_IDENTITY,
        "openapi_version": OPENAPI_VERSION,
        "openapi_sha256": OPENAPI_SHA256,
        "local_digest": "c" * 64,
        "observed_price_usd_per_gpu_hour": "0.3900",
        "projected_spend_usd": manifest["budget_guard"]["next_cell_maximum_usd"],
        "stage": "qualification",
        "authorization_digest": load_launch_authorization(
            manifest["authorization"]["controller_path"]
        ).digest,
        "ledger_digest": load_spend_ledger(manifest["spend_ledger"]["controller_path"]).digest,
        "authorization_expires_at": manifest["authorization"]["expires_at"],
        "maximum_duration_seconds": 7200,
        "startup_allowance_seconds": manifest["monitor"]["first_heartbeat_seconds"],
        "workload": {
            "manifest_digest": manifest["manifest_digest"],
            "docker_args": "uv run mantis-v2 workload-execute --manifest /workspace/launch.json",
            "environment": {
                "MANTIS_RUN_ID": manifest["run_id"],
                "MANTIS_WORKLOAD_DIGEST": manifest["manifest_digest"],
                "MANTIS_WORKLOAD_MANIFEST": "/workspace/launch.json",
                "HF_HOME": (
                    "/workspace/mantis/inputs/"
                    f"{manifest['input_bundle']['bundle_digest']}/cache/huggingface"
                ),
                "HF_HUB_OFFLINE": "1",
            },
        },
    }
    unsigned["decision_digest"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return unsigned


def _heartbeat(manifest: dict[str, object], stage: str, progress: int, token: str) -> dict:
    return sign_heartbeat(
        {
            "schema_version": 1,
            "run_id": manifest["run_id"],
            "pod_id": "pod-123",
            "process_id": "pid-77",
            "stage": stage,
            "progress": progress,
            "checkpoint_sha256": "e" * 64,
            "gpu_allocation": "NVIDIA A40:0",
            "timestamp": "2026-07-22T12:00:00Z",
        },
        token,
    )


def test_supervisor_starts_before_create_and_deletes_after_verified_completion(
    tmp_path: Path,
) -> None:
    manifest_path = seal_workload_manifest(_spec(tmp_path), tmp_path / "sealed")
    manifest = validate_workload_manifest(manifest_path)
    adapter = FakeAdapter()
    clock = FakeClock()
    replicated: list[str] = []
    heartbeats = [_heartbeat(manifest, "complete", 200, "token")]

    result = supervise_workload(
        manifest_path=manifest_path,
        decision=_decision(manifest),
        state_root=tmp_path / "state",
        adapter=adapter,
        heartbeat_token="token",
        heartbeat_source=lambda run_id, pod_id: heartbeats.pop(0),
        collect_diagnostics=lambda pod_id, deadline: {},
        checkpoint=lambda pod_id, deadline: None,
        replicate=lambda _: replicated.append("verified"),
        now=clock.now,
        sleep=clock.sleep,
    )

    assert result["status"] == "complete"
    assert adapter.terminations == 1
    assert replicated == ["verified"]
    events = [
        json.loads(path.read_text())
        for path in sorted((tmp_path / "state" / "ledger").glob("*.json"))
    ]
    assert [event["event"] for event in events] == [
        "watchdog_started",
        "pod_created",
        "heartbeat",
        "artifacts_verified",
        "pod_deleted",
        "provider_absence_verified",
        "billing_reconciled",
    ]
    assert events[0]["observed_at"] <= events[1]["observed_at"]
    assert all(event["quoted_rates"] == manifest["quoted_rates"] for event in events)
    assert all(event["storage_size_gb"] == 150 for event in events)
    assert events[-1]["billing"]["pod_id"] == "pod-123"


def test_supervisor_honors_extended_cold_start_allowance(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec["monitor"]["first_heartbeat_seconds"] = 1800  # type: ignore[index]
    spec["budget_guard"]["next_cell_maximum_usd"] = "1.05"  # type: ignore[index]
    manifest_path = seal_workload_manifest(spec, tmp_path / "sealed")
    manifest = validate_workload_manifest(manifest_path)
    adapter = FakeAdapter()
    clock = FakeClock()

    def heartbeat_source(run_id: str, pod_id: str) -> dict | None:
        del run_id, pod_id
        if clock.value < datetime(2026, 7, 22, 12, 29, 30, tzinfo=UTC):
            return None
        payload = _heartbeat(manifest, "complete", 200, "token")
        payload["timestamp"] = clock.value.isoformat().replace("+00:00", "Z")
        return sign_heartbeat(
            {key: value for key, value in payload.items() if key != "signature"}, "token"
        )

    result = supervise_workload(
        manifest_path=manifest_path,
        decision=_decision(manifest),
        state_root=tmp_path / "state",
        adapter=adapter,
        heartbeat_token="token",
        heartbeat_source=heartbeat_source,
        collect_diagnostics=lambda pod_id, deadline: {},
        checkpoint=lambda pod_id, deadline: None,
        replicate=lambda _: None,
        now=clock.now,
        sleep=clock.sleep,
    )

    assert result["status"] == "complete"
    assert clock.value == datetime(2026, 7, 22, 12, 29, 30, tzinfo=UTC)


def test_uncertain_create_polls_and_deletes_late_visible_pod(tmp_path: Path) -> None:
    manifest_path = seal_workload_manifest(_spec(tmp_path), tmp_path / "sealed")
    manifest = validate_workload_manifest(manifest_path)
    clock = FakeClock()

    class UncertainCreateAdapter:
        def __init__(self) -> None:
            self.inventory_calls = 0
            self.terminations = 0

        def inventory(self) -> list[dict[str, object]]:
            self.inventory_calls += 1
            if self.inventory_calls <= 5 or self.inventory_calls >= 8:
                return []
            return [{"id": "pod-late", "name": manifest["run_id"]}]

        def create(self, decision: dict[str, object], deadline: datetime) -> dict[str, object]:
            raise RuntimeError("provider response lost after create")

        def terminate(self, pod_id: str) -> dict[str, object]:
            assert pod_id == "pod-late"
            self.terminations += 1
            return {"id": pod_id, "deleted": True}

    adapter = UncertainCreateAdapter()
    with pytest.raises(WorkloadError, match="verified absence"):
        supervise_workload(
            manifest_path=manifest_path,
            decision=_decision(manifest),
            state_root=tmp_path / "state",
            adapter=adapter,  # type: ignore[arg-type]
            heartbeat_token="token",
            heartbeat_source=lambda run_id, pod_id: None,
            collect_diagnostics=lambda pod_id, deadline: {},
            checkpoint=lambda pod_id, deadline: None,
            replicate=lambda _: None,
            now=clock.now,
            sleep=clock.sleep,
        )

    assert adapter.terminations == 2
    assert adapter.inventory() == []


def test_supervisor_deletes_after_four_missing_heartbeats_and_preserves_diagnostics(
    tmp_path: Path,
) -> None:
    manifest_path = seal_workload_manifest(_spec(tmp_path), tmp_path / "sealed")
    manifest = validate_workload_manifest(manifest_path)
    adapter = FakeAdapter()
    clock = FakeClock()
    first = _heartbeat(manifest, "training", 1, "token")
    responses = [first, first, first, first, first]
    diagnostic_deadlines: list[datetime] = []
    checkpoint_deadlines: list[datetime] = []

    result = supervise_workload(
        manifest_path=manifest_path,
        decision=_decision(manifest),
        state_root=tmp_path / "state",
        adapter=adapter,
        heartbeat_token="token",
        heartbeat_source=lambda run_id, pod_id: responses.pop(0),
        collect_diagnostics=lambda pod_id, deadline: diagnostic_deadlines.append(deadline)
        or {"stdout": "captured", "gpu": "captured"},
        checkpoint=lambda pod_id, deadline: checkpoint_deadlines.append(deadline) or ("f" * 64),
        replicate=lambda _: None,
        now=clock.now,
        sleep=clock.sleep,
    )

    assert result["status"] == "terminated"
    assert result["reason"] == "heartbeat_missed"
    assert adapter.terminations == 1
    assert len(diagnostic_deadlines) == len(checkpoint_deadlines) == 1
    assert diagnostic_deadlines[0] == checkpoint_deadlines[0]
    assert diagnostic_deadlines[0] - clock.value <= timedelta(seconds=120)
    events = [
        json.loads(path.read_text())["event"]
        for path in sorted((tmp_path / "state" / "ledger").glob("*.json"))
    ]
    assert events.index("provider_absence_verified") < events.index("artifacts_verified")


@pytest.mark.parametrize("failure_mode", ["missing_started_at", "heartbeat", "replicate"])
def test_every_post_create_exception_still_deletes_and_verifies_absence(
    tmp_path: Path, failure_mode: str
) -> None:
    manifest_path = seal_workload_manifest(_spec(tmp_path), tmp_path / "sealed")
    manifest = validate_workload_manifest(manifest_path)
    adapter = FakeAdapter()
    clock = FakeClock()
    heartbeat = _heartbeat(manifest, "complete", 200, "token")
    if failure_mode == "missing_started_at":
        original_status = adapter.status

        def missing_started_at(pod_id: str) -> dict[str, object]:
            status = original_status(pod_id)
            del status["lastStartedAt"]
            return status

        adapter.status = missing_started_at  # type: ignore[method-assign]

    def heartbeat_source(run_id: str, pod_id: str) -> dict:
        del run_id, pod_id
        if failure_mode == "heartbeat":
            raise RuntimeError("heartbeat transport failed")
        return heartbeat

    def replicate(_: object) -> None:
        if failure_mode == "replicate":
            raise RuntimeError("replication failed")

    with pytest.raises(WorkloadError, match="supervision failed"):
        supervise_workload(
            manifest_path=manifest_path,
            decision=_decision(manifest),
            state_root=tmp_path / "state",
            adapter=adapter,
            heartbeat_token="token",
            heartbeat_source=heartbeat_source,
            collect_diagnostics=lambda pod_id, deadline: {},
            checkpoint=lambda pod_id, deadline: None,
            replicate=replicate,
            now=clock.now,
            sleep=clock.sleep,
        )

    assert adapter.terminations == 1
    assert adapter.live is False


def test_stale_signed_heartbeat_cannot_satisfy_current_pod_readiness(tmp_path: Path) -> None:
    manifest_path = seal_workload_manifest(_spec(tmp_path), tmp_path / "sealed")
    manifest = validate_workload_manifest(manifest_path)
    adapter = FakeAdapter()
    clock = FakeClock()
    stale = _heartbeat(manifest, "training", 1, "token")
    stale["timestamp"] = "2026-07-22T11:59:59Z"
    stale = sign_heartbeat(
        {key: value for key, value in stale.items() if key != "signature"}, "token"
    )

    with pytest.raises(WorkloadError, match="supervision failed"):
        supervise_workload(
            manifest_path=manifest_path,
            decision=_decision(manifest),
            state_root=tmp_path / "state",
            adapter=adapter,
            heartbeat_token="token",
            heartbeat_source=lambda run_id, pod_id: stale,
            collect_diagnostics=lambda pod_id, deadline: {},
            checkpoint=lambda pod_id, deadline: None,
            replicate=lambda _: None,
            now=clock.now,
            sleep=clock.sleep,
        )

    assert adapter.terminations == 1


def test_nonreturning_diagnostic_callback_cannot_extend_grace() -> None:
    never = threading.Event()

    with pytest.raises(WorkloadError, match="exceeded"):
        _bounded_call(
            lambda deadline: never.wait(),
            deadline=datetime.now(UTC) + timedelta(milliseconds=10),
            now=lambda: datetime.now(UTC),
        )


def test_pod_executor_revalidates_then_immediately_runs_and_signs_heartbeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = seal_workload_manifest(_spec(tmp_path), tmp_path / "sealed")
    manifest = json.loads(manifest_path.read_text())
    local_run = tmp_path / "pod-run"
    manifest["artifacts"]["pod"] = str(local_run)
    manifest["monitor"]["heartbeat"] = str(local_run / "heartbeat.json")
    launched: list[tuple[list[str], dict]] = []

    class Process:
        pid = 77

        def __init__(self) -> None:
            self.polls = iter((None, 0))

        def poll(self) -> int | None:
            return next(self.polls)

    class Service:
        pid = 76

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: int) -> int:
            assert timeout == 10
            return 0

    def process_factory(command: list[str], **kwargs) -> Process:
        launched.append((command, kwargs))
        return Process()

    validation_order: list[str] = []

    def load_manifest(_path: Path, *, path_role: str = "controller") -> dict:
        validation_order.append(f"load:{path_role}")
        return manifest

    monkeypatch.setattr(
        "mantis_v2.runpod_workload._load_manifest_document",
        load_manifest,
    )
    monkeypatch.setattr(
        "mantis_v2.runpod_workload._promote_pod_input_bundle",
        lambda _manifest: validation_order.append("promote"),
    )
    monkeypatch.setattr(
        "mantis_v2.runpod_workload.validate_workload_manifest",
        lambda path, path_role="controller": validation_order.append(f"validate:{path_role}")
        or manifest,
    )

    result = execute_workload_manifest(
        manifest_path,
        environ={
            "MANTIS_RUN_ID": str(manifest["run_id"]),
            "MANTIS_WORKLOAD_DIGEST": str(manifest["manifest_digest"]),
            "RUNPOD_POD_ID": "pod-123",
            "NVIDIA_VISIBLE_DEVICES": "GPU-0",
        },
        process_factory=process_factory,
        service_factory=lambda _command, **_kwargs: Service(),
        runtime_self_check=lambda: {
            "schema_version": 1,
            "passed": True,
            "scope": "runtime_cuda",
            "inventory": {},
        },
        diagnostics_runner=lambda command: subprocess.CompletedProcess(
            command, 0, "0, GPU-0, NVIDIA A40, 46068, 1024, 99\n", ""
        ),
        now=lambda: datetime(2026, 7, 22, 12, tzinfo=UTC),
        sleep=lambda seconds: None,
    )

    heartbeat = json.loads(Path(result["heartbeat"]).read_text())
    diagnostics = json.loads(Path(result["diagnostics"]).read_text())
    assert result["status"] == "complete"
    assert launched[0][0] == manifest["start_command"]
    assert heartbeat["stage"] == "complete"
    assert heartbeat["pod_id"] == "pod-123"
    assert heartbeat["signature"]
    assert diagnostics["nvidia_smi_return_code"] == 0
    assert "NVIDIA A40" in diagnostics["nvidia_smi_stdout"]
    assert diagnostics["tensorboard_process_id"] == "76"
    assert "token" not in json.dumps(launched, default=str)
    assert validation_order == ["load:pod", "promote", "validate:pod"]
    parsed = _parser().parse_args(["workload-execute", "--manifest", str(manifest_path)])
    assert parsed.manifest == manifest_path
    replicated = _parser().parse_args(
        [
            "runpod-replicate-workload",
            "--manifest",
            str(manifest_path),
            "--decision",
            "decision.json",
            "--local",
            "local.toml",
        ]
    )
    assert replicated.command == "runpod-replicate-workload"
    assert replicated.manifest == manifest_path
