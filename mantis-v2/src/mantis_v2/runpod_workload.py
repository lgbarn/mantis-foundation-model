"""Immutable no-idle RunPod workload manifest and local preflight."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import queue
import re
import shlex
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from mantis_v2.foundation_matrix import FoundationMatrixError, validate_planned_cell
from mantis_v2.monitoring import tensorboard_command
from mantis_v2.runpod_config import (
    RunpodConfigError,
    load_launch_authorization,
    load_spend_ledger,
)
from mantis_v2.runpod_image import self_check as image_self_check
from mantis_v2.runpod_lifecycle import (
    LifecycleAdapter,
    LifecycleError,
    launch_pod,
    reconcile_launch,
    reconcile_termination,
    terminate_pod_for_reason,
    validate_launch_decision,
)
from mantis_v2.runpodctl_adapter import RUNPODCTL_COMMIT, RUNPODCTL_VERSION
from mantis_v2.transfer_bundle import (
    TransferBundleError,
    load_bundle_manifest,
    verify_and_promote,
)


class WorkloadError(RuntimeError):
    """Raised when a workload cannot be proven ready before paid create."""


_KEYS = {
    "schema_version",
    "run_id",
    "source_revision",
    "dependency_lock",
    "image",
    "experiment_config",
    "matrix_plan",
    "matrix_base_config",
    "input_bundle",
    "dataset_manifest",
    "spend_ledger",
    "foundation_checkpoint",
    "start_command",
    "artifacts",
    "resume",
    "monitor",
    "maximum_duration_seconds",
    "quoted_rates",
    "budget_guard",
    "authorization",
    "runpodctl",
}
_BOOTSTRAP_KEYS = _KEYS | {"bootstrap"}
_FROZEN_KEYS = _KEYS | {"workload_kind"}
_FROZEN_BOOTSTRAP_KEYS = _FROZEN_KEYS | {"bootstrap"}
_OFFICIAL_RUNPOD_TEMPLATE_ID = "runpod-torch-v280"
_OFFICIAL_RUNPOD_IMAGE_REF = (
    "runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35"
)
_BOOTSTRAP_UV_VERSION = "0.9.0"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise WorkloadError(f"launch input is unavailable: {path}") from exc
    return digest.hexdigest()


def _exact(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise WorkloadError(f"{field} keys mismatch")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkloadError(f"{field} must be a non-empty string")
    return value


def _sha(value: object, field: str) -> str:
    result = _text(value, field)
    if not re.fullmatch(r"[0-9a-f]{64}", result):
        raise WorkloadError(f"{field} must be a SHA-256 digest")
    return result


def _file_record(value: object, field: str, *, path_role: str) -> dict[str, Any]:
    record = _exact(value, {"controller_path", "pod_path", "sha256", "size"}, field)
    if path_role not in {"controller", "pod"}:
        raise WorkloadError("file validation role is invalid")
    controller_path = Path(_text(record["controller_path"], f"{field}.controller_path"))
    pod_path = Path(_text(record["pod_path"], f"{field}.pod_path"))
    if not controller_path.is_absolute() or not pod_path.is_absolute():
        raise WorkloadError(f"{field} paths must be absolute")
    path = controller_path if path_role == "controller" else pod_path
    expected = _sha(record["sha256"], f"{field}.sha256")
    if not isinstance(record["size"], int) or isinstance(record["size"], bool):
        raise WorkloadError(f"{field}.size must be an integer")
    if not path.is_file() or path.stat().st_size != record["size"]:
        raise WorkloadError(f"{field} size mismatch")
    if _sha256_file(path) != expected:
        raise WorkloadError(f"{field} hash mismatch")
    return record


def _decimal(value: object, field: str, *, positive: bool = True) -> Decimal:
    if not isinstance(value, str):
        raise WorkloadError(f"{field} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise WorkloadError(f"{field} must be a decimal string") from exc
    if not parsed.is_finite() or (parsed <= 0 if positive else parsed < 0):
        raise WorkloadError(f"{field} is outside its accepted range")
    return parsed


def _validate_core(
    core: dict[str, Any], *, now: datetime, path_role: Literal["controller", "pod"]
) -> None:
    is_frozen = core.get("workload_kind") == "frozen_expected_r"
    accepted_keys = (
        (_FROZEN_KEYS, _FROZEN_BOOTSTRAP_KEYS) if is_frozen else (_KEYS, _BOOTSTRAP_KEYS)
    )
    if all(set(core) != keys for keys in accepted_keys):
        raise WorkloadError("workload manifest keys mismatch")
    if core["schema_version"] != 1:
        raise WorkloadError("unsupported workload manifest schema")
    _text(core["run_id"], "run_id")
    source_revision = _text(core["source_revision"], "source_revision")
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise WorkloadError("source_revision must be an exact commit")
    lock_record = _file_record(core["dependency_lock"], "dependency_lock", path_role=path_role)
    image = _exact(core["image"], {"ref", "self_check"}, "image")
    if not re.search(r"@sha256:[0-9a-f]{64}$", _text(image["ref"], "image.ref")):
        raise WorkloadError("image.ref must be digest pinned")
    self_check = _file_record(image["self_check"], "image.self_check", path_role=path_role)
    try:
        check_payload = json.loads(Path(self_check[f"{path_role}_path"]).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkloadError("image self-check receipt is invalid") from exc
    if (
        not isinstance(check_payload, dict)
        or check_payload.get("passed") is not True
        or check_payload.get("scope") not in {"static_image", "official_bootstrap"}
    ):
        raise WorkloadError("image self-check did not pass")
    identities = check_payload.get("inventory", {}).get("identities", {})
    if not isinstance(identities, dict) or identities.get("source_revision") != source_revision:
        raise WorkloadError("image self-check source revision mismatch")
    if identities.get("lock_sha256") != lock_record["sha256"]:
        raise WorkloadError("image self-check dependency lock mismatch")
    scope = check_payload["scope"]
    if is_frozen and scope != "official_bootstrap":
        raise WorkloadError("frozen screening requires the official RunPod template")
    if scope == "official_bootstrap":
        bootstrap = _exact(
            core.get("bootstrap"),
            {"source_archive", "project_root", "venv_path", "uv_version"},
            "bootstrap",
        )
        source_archive = _file_record(
            bootstrap["source_archive"], "bootstrap.source_archive", path_role=path_role
        )
        expected_project = Path(f"/workspace/mantis/runtime/{source_revision}")
        expected_venv = Path("/opt/mantis/venv")
        if Path(_text(bootstrap["project_root"], "bootstrap.project_root")) != expected_project:
            raise WorkloadError("bootstrap project root does not match the source revision")
        if Path(_text(bootstrap["venv_path"], "bootstrap.venv_path")) != expected_venv:
            raise WorkloadError("bootstrap environment does not match the dependency lock")
        if _text(bootstrap["uv_version"], "bootstrap.uv_version") != _BOOTSTRAP_UV_VERSION:
            raise WorkloadError("bootstrap uv version is not qualified")
        if check_payload.get("uv_version") != _BOOTSTRAP_UV_VERSION:
            raise WorkloadError("bootstrap receipt uv version is not qualified")
        if identities.get("source_archive_sha256") != source_archive["sha256"]:
            raise WorkloadError("bootstrap source archive differs from its receipt")
        if (
            check_payload.get("provider_support") != "official_runpod_template"
            or check_payload.get("template_id") != _OFFICIAL_RUNPOD_TEMPLATE_ID
            or image["ref"] != _OFFICIAL_RUNPOD_IMAGE_REF
        ):
            raise WorkloadError("bootstrap image is not an official RunPod template")
        if check_payload.get("image_ref") != image["ref"]:
            raise WorkloadError("bootstrap image differs from its receipt")
    elif "bootstrap" in core:
        raise WorkloadError("custom image manifests cannot contain a bootstrap")
    config_record = _file_record(
        core["experiment_config"], "experiment_config", path_role=path_role
    )
    try:
        experiment = json.loads(Path(config_record[f"{path_role}_path"]).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkloadError("experiment config is not canonical JSON") from exc
    if (
        not isinstance(experiment, dict)
        or experiment.get("evaluation", {}).get("allow_holdout") is not False
        or experiment.get("data", {}).get("holdout_start") != "2026-01-01T00:00:00+00:00"
    ):
        raise WorkloadError("experiment config does not seal the 2026 holdout")
    plan_record = _file_record(core["matrix_plan"], "matrix_plan", path_role=path_role)
    base_record = _file_record(
        core["matrix_base_config"], "matrix_base_config", path_role=path_role
    )
    if is_frozen:
        from mantis_v2.frozen_expected_r import FrozenExpectedRConfig

        FrozenExpectedRConfig.from_json(Path(str(base_record[f"{path_role}_path"])))
    if not is_frozen:
        try:
            plan_payload = json.loads(Path(plan_record[f"{path_role}_path"]).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkloadError("matrix plan is invalid") from exc
        plan_digest = plan_payload.get("plan_digest") if isinstance(plan_payload, dict) else None
        if not isinstance(plan_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", plan_digest):
            raise WorkloadError("matrix plan digest is invalid")
        for role in ("controller", "pod"):
            plan_path = Path(str(plan_record[f"{role}_path"]))
            base_path = Path(str(base_record[f"{role}_path"]))
            if plan_path.parent.name != plan_digest:
                raise WorkloadError("matrix plan path does not preserve its digest")
            if base_path != plan_path.parent / "base-config.toml":
                raise WorkloadError("matrix base config is not colocated with the matrix plan")
    dataset_record = _file_record(core["dataset_manifest"], "dataset_manifest", path_role=path_role)
    if experiment.get("data", {}).get("corpus_manifest_sha256") != dataset_record["sha256"]:
        raise WorkloadError("dataset manifest differs from experiment config")
    input_bundle = _exact(
        core["input_bundle"],
        {"manifest", "bundle_digest", "incoming_root", "final_parent"},
        "input_bundle",
    )
    bundle_record = _file_record(
        input_bundle["manifest"], "input_bundle.manifest", path_role=path_role
    )
    try:
        bundle = load_bundle_manifest(Path(str(bundle_record[f"{path_role}_path"])))
    except TransferBundleError as exc:
        raise WorkloadError("input bundle manifest is invalid") from exc
    bundle_digest = _sha(input_bundle["bundle_digest"], "input_bundle.bundle_digest")
    incoming_root = Path(_text(input_bundle["incoming_root"], "input_bundle.incoming_root"))
    final_parent = Path(_text(input_bundle["final_parent"], "input_bundle.final_parent"))
    if (
        bundle.bundle_digest != bundle_digest
        or incoming_root != Path(f"/workspace/mantis/transfer/incoming/{bundle_digest}/files")
        or final_parent != Path("/workspace/mantis/inputs")
    ):
        raise WorkloadError("input bundle paths or identity are invalid")
    expected_input_root = final_parent / bundle_digest
    experiment_data = experiment.get("data")
    if not isinstance(experiment_data, dict):
        raise WorkloadError("experiment data contract is missing")
    data_root = Path(str(experiment_data.get("root", "")))
    corpus_manifest_path = Path(str(experiment_data.get("corpus_manifest_path", "")))
    if (
        not data_root.is_absolute()
        or not data_root.is_relative_to(expected_input_root)
        or corpus_manifest_path != Path(str(dataset_record["pod_path"]))
        or not corpus_manifest_path.is_relative_to(expected_input_root)
    ):
        raise WorkloadError("experiment data paths are not bound to the input bundle")
    dataset_relative = corpus_manifest_path.relative_to(expected_input_root).as_posix()
    dataset_entries = {entry.path: entry for entry in bundle.entries}
    bundled_dataset = dataset_entries.get(dataset_relative)
    if (
        bundled_dataset is None
        or bundled_dataset.sha256 != dataset_record["sha256"]
        or bundled_dataset.size != dataset_record["size"]
    ):
        raise WorkloadError("dataset manifest is not an exact input bundle member")
    if is_frozen:
        config_pod_path = Path(str(base_record["pod_path"]))
        if not config_pod_path.is_relative_to(expected_input_root):
            raise WorkloadError("frozen config is not bound to the input bundle")
        bundled_config = dataset_entries.get(
            config_pod_path.relative_to(expected_input_root).as_posix()
        )
        if (
            bundled_config is None
            or bundled_config.sha256 != base_record["sha256"]
            or bundled_config.size != base_record["size"]
        ):
            raise WorkloadError("frozen config is not an exact input bundle member")
    checkpoint = _exact(
        core["foundation_checkpoint"],
        {"repository", "revision", "sha256"},
        "foundation_checkpoint",
    )
    _text(checkpoint["repository"], "foundation_checkpoint.repository")
    if not re.fullmatch(
        r"[0-9a-f]{40}", _text(checkpoint["revision"], "foundation_checkpoint.revision")
    ):
        raise WorkloadError("foundation checkpoint revision must be exact")
    _sha(checkpoint["sha256"], "foundation_checkpoint.sha256")
    experiment_model = experiment.get("model", {})
    if (
        not isinstance(experiment_model, dict)
        or checkpoint["repository"] != experiment_model.get("hub_model")
        or checkpoint["revision"] != experiment_model.get("hub_revision")
        or checkpoint["sha256"] != experiment_model.get("weights_sha256")
    ):
        raise WorkloadError("foundation checkpoint differs from experiment config")
    command = core["start_command"]
    if is_frozen:
        if (
            not isinstance(command, list)
            or len(command) != 3
            or command[:2] != ["bash", "-lc"]
            or not isinstance(command[2], str)
        ):
            raise WorkloadError("frozen start_command must be one shell command")
        from mantis_v2.frozen_expected_r import validate_paid_runner_contract

        validate_paid_runner_contract(
            Path(str(plan_record[f"{path_role}_path"])), core, path_role=path_role
        )
    else:
        if (
            not isinstance(command, list)
            or len(command) != 8
            or any(not isinstance(item, str) or not item for item in command)
            or command[:5] != ["uv", "run", "mantis-v2", "foundation-matrix-cell", "--plan"]
            or command[6] != "--cell-id"
        ):
            raise WorkloadError("start_command must be the exact argv matrix-cell entrypoint")
        try:
            if Path(command[5]).resolve() != Path(str(plan_record["pod_path"])).resolve():
                raise WorkloadError("start_command plan path does not match matrix_plan")
            planned = validate_planned_cell(
                plan_record[f"{path_role}_path"],
                command[7],
                config_record[f"{path_role}_path"],
            )
        except FoundationMatrixError as exc:
            raise WorkloadError("start_command is not bound to a valid matrix cell") from exc
        if planned["run_name"] != core["run_id"]:
            raise WorkloadError("run_id does not match the planned matrix cell")
    artifacts = _exact(core["artifacts"], {"pod", "controller", "backup"}, "artifacts")
    artifact_paths = [Path(_text(artifacts[key], f"artifacts.{key}")) for key in artifacts]
    if any(not path.is_absolute() for path in artifact_paths) or len(set(artifact_paths)) != 3:
        raise WorkloadError("artifact destinations must be distinct absolute paths")
    expected_pod_artifacts = Path("/workspace/mantis/runs") / str(core["run_id"])
    experiment_run = experiment.get("run")
    if (
        not isinstance(experiment_run, dict)
        or Path(str(experiment_run.get("artifact_root", ""))) != expected_pod_artifacts.parent
    ):
        raise WorkloadError("experiment artifact_root differs from the mounted run root")
    if Path(str(artifacts["pod"])) != expected_pod_artifacts:
        raise WorkloadError("artifacts.pod must be the mounted run directory")
    if core["resume"] != {
        "enabled": True,
        "same_run_only": True,
        "provenance_required": True,
    }:
        raise WorkloadError("resume contract must fail closed to the same run")
    monitor = _exact(
        core["monitor"],
        {
            "tensorboard",
            "heartbeat",
            "poll_seconds",
            "first_heartbeat_seconds",
            "miss_limit",
            "token",
        },
        "monitor",
    )
    if is_frozen:
        if monitor["tensorboard"] is not None:
            raise WorkloadError("TensorBoard must be disabled for frozen screening")
    else:
        _text(monitor["tensorboard"], "monitor.tensorboard")
    heartbeat_path = Path(_text(monitor["heartbeat"], "monitor.heartbeat"))
    if heartbeat_path != expected_pod_artifacts / "heartbeat.json":
        raise WorkloadError("monitor.heartbeat must be inside artifacts.pod")
    _file_record(monitor["token"], "monitor.token", path_role=path_role)
    startup_allowance = monitor["first_heartbeat_seconds"]
    if (
        monitor["poll_seconds"] != 30
        or not isinstance(startup_allowance, int)
        or isinstance(startup_allowance, bool)
        or not 600 <= startup_allowance <= 1800
        or monitor["miss_limit"] != 4
    ):
        raise WorkloadError("monitor timing differs from the accepted watchdog contract")
    maximum_duration = core["maximum_duration_seconds"]
    if (
        not isinstance(maximum_duration, int)
        or isinstance(maximum_duration, bool)
        or maximum_duration <= 0
    ):
        raise WorkloadError("maximum_duration_seconds must be positive")
    rates = _exact(
        core["quoted_rates"],
        {"compute_usd_per_hour", "storage_usd_per_gb_hour", "storage_gb"},
        "quoted_rates",
    )
    _decimal(rates["compute_usd_per_hour"], "quoted_rates.compute_usd_per_hour")
    _decimal(rates["storage_usd_per_gb_hour"], "quoted_rates.storage_usd_per_gb_hour")
    if not isinstance(rates["storage_gb"], int) or rates["storage_gb"] <= 0:
        raise WorkloadError("quoted_rates.storage_gb must be positive")
    compute_rate = _decimal(rates["compute_usd_per_hour"], "quoted_rates.compute_usd_per_hour")
    storage_rate = _decimal(
        rates["storage_usd_per_gb_hour"], "quoted_rates.storage_usd_per_gb_hour"
    )
    projected_storage_pool = storage_rate * Decimal(rates["storage_gb"]) * Decimal("720")
    if projected_storage_pool > Decimal("15.00"):
        raise WorkloadError("storage allocation exceeds the $15 campaign sub-pool")
    ledger_record = _file_record(core["spend_ledger"], "spend_ledger", path_role=path_role)
    try:
        ledger = load_spend_ledger(ledger_record[f"{path_role}_path"])
    except RunpodConfigError as exc:
        raise WorkloadError("spend ledger is invalid") from exc
    provider_billed_allowance_seconds = maximum_duration + startup_allowance + 120
    maximum_cell = (
        (compute_rate + storage_rate * Decimal(rates["storage_gb"]))
        * Decimal(provider_billed_allowance_seconds)
        / Decimal("3600")
    )
    minimum_shutdown_reserve = compute_rate / Decimal("6") + (
        storage_rate * Decimal(rates["storage_gb"]) / Decimal("3600")
    )
    budget = _exact(
        core["budget_guard"],
        {
            "stage",
            "reconciled_spend_usd",
            "unbilled_live_accrual_usd",
            "stage_reconciled_spend_usd",
            "next_cell_maximum_usd",
            "shutdown_reserve_usd",
        },
        "budget_guard",
    )
    if budget["stage"] not in {"qualification", "production", "recovery"}:
        raise WorkloadError("budget_guard.stage is invalid")
    reconciled = _decimal(
        budget["reconciled_spend_usd"], "budget_guard.reconciled_spend_usd", positive=False
    )
    live = _decimal(
        budget["unbilled_live_accrual_usd"],
        "budget_guard.unbilled_live_accrual_usd",
        positive=False,
    )
    stage_spend = _decimal(
        budget["stage_reconciled_spend_usd"],
        "budget_guard.stage_reconciled_spend_usd",
        positive=False,
    )
    next_cell = _decimal(budget["next_cell_maximum_usd"], "budget_guard.next_cell_maximum_usd")
    reserve = _decimal(budget["shutdown_reserve_usd"], "budget_guard.shutdown_reserve_usd")
    if next_cell < maximum_cell:
        raise WorkloadError("budget guard understates the next cell maximum")
    if reserve < minimum_shutdown_reserve or reserve > Decimal("25.00"):
        raise WorkloadError("shutdown reserve is outside the protected recovery pool")
    stage_limit = {
        "qualification": Decimal("10.00"),
        "production": Decimal("100.00"),
        "recovery": Decimal("25.00"),
    }[budget["stage"]]
    if stage_spend + next_cell > stage_limit:
        raise WorkloadError("stage budget is exceeded")
    if (
        reconciled != ledger.actual_spend_usd
        or live != ledger.reserved_spend_usd
        or stage_spend != ledger.bucket_actual_spend_usd[budget["stage"]]
    ):
        raise WorkloadError("budget guard differs from the immutable spend ledger")
    if budget["stage"] != "recovery" and reconciled + live + next_cell + reserve > Decimal(
        "125.00"
    ):
        raise WorkloadError("ordinary launch cutoff is exceeded")
    if reconciled + live + next_cell + reserve > Decimal("150.00"):
        raise WorkloadError("campaign ceiling is exceeded")
    authorization = _exact(
        core["authorization"],
        {
            "controller_path",
            "pod_path",
            "sha256",
            "size",
            "expires_at",
            "autopay_disabled",
            "ordinary_launch_cutoff_usd",
            "campaign_ceiling_usd",
            "recovery_authorized",
        },
        "authorization",
    )
    _file_record(
        {key: authorization[key] for key in ("controller_path", "pod_path", "sha256", "size")},
        "authorization",
        path_role=path_role,
    )
    if authorization["autopay_disabled"] is not True:
        raise WorkloadError("Billing-console auto-pay must be attested disabled")
    if authorization["recovery_authorized"] is not (budget["stage"] == "recovery"):
        raise WorkloadError("recovery spend requires a separate exact authorization")
    try:
        expires = datetime.fromisoformat(str(authorization["expires_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkloadError("authorization expiry is invalid") from exc
    if expires.tzinfo is None or expires.astimezone(UTC) <= now.astimezone(UTC):
        raise WorkloadError("authorization is expired")
    if _decimal(
        authorization["ordinary_launch_cutoff_usd"],
        "authorization.ordinary_launch_cutoff_usd",
    ) != Decimal("125.00"):
        raise WorkloadError("ordinary launch cutoff must be $125")
    if _decimal(
        authorization["campaign_ceiling_usd"], "authorization.campaign_ceiling_usd"
    ) != Decimal("150.00"):
        raise WorkloadError("campaign ceiling must be $150")
    try:
        approved = load_launch_authorization(authorization["controller_path"])
    except RunpodConfigError as exc:
        raise WorkloadError("launch authorization file is invalid") from exc
    if (
        approved.autopay_disabled is not authorization["autopay_disabled"]
        or approved.ordinary_launch_cutoff_usd
        != _decimal(
            authorization["ordinary_launch_cutoff_usd"],
            "authorization.ordinary_launch_cutoff_usd",
        )
        or approved.campaign_ceiling_usd
        != _decimal(
            authorization["campaign_ceiling_usd"],
            "authorization.campaign_ceiling_usd",
        )
        or approved.recovery_authorized is not authorization["recovery_authorized"]
    ):
        raise WorkloadError("manifest budget attestation differs from authorization")
    runpodctl = _exact(
        core["runpodctl"],
        {"version", "source_commit", "binary_sha256"},
        "runpodctl",
    )
    if runpodctl["version"] != RUNPODCTL_VERSION or runpodctl["source_commit"] != RUNPODCTL_COMMIT:
        raise WorkloadError("runpodctl source identity changed")
    _sha(runpodctl["binary_sha256"], "runpodctl.binary_sha256")


def _atomic_idempotent(path: Path, value: dict[str, Any]) -> None:
    encoded = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise WorkloadError(f"immutable launch manifest differs: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise WorkloadError(f"immutable launch manifest differs: {path}") from None
    finally:
        Path(temporary).unlink(missing_ok=True)


def seal_workload_manifest(spec: Mapping[str, Any], output_root: str | Path) -> Path:
    core = dict(spec)
    _validate_core(core, now=datetime.now(UTC), path_role="controller")
    digest = _digest(core)
    manifest = {**core, "manifest_digest": digest}
    path = Path(output_root) / digest / "launch-manifest.json"
    _atomic_idempotent(path, manifest)
    return path


def _load_manifest_document(path: str | Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkloadError("launch manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise WorkloadError("launch manifest must be an object")
    core = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if manifest.get("manifest_digest") != _digest(core):
        raise WorkloadError("launch manifest digest mismatch")
    if Path(path).parent.name != manifest["manifest_digest"]:
        raise WorkloadError("launch manifest path does not match its digest")
    return manifest


def validate_workload_manifest(
    path: str | Path, *, path_role: Literal["controller", "pod"] = "controller"
) -> dict[str, Any]:
    manifest = _load_manifest_document(path)
    core = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    _validate_core(core, now=datetime.now(UTC), path_role=path_role)
    return manifest


def _promote_pod_input_bundle(manifest: Mapping[str, Any]) -> None:
    """Verify the staged bundle envelope, then promote before validating its members."""
    input_bundle = _exact(
        manifest.get("input_bundle"),
        {"manifest", "bundle_digest", "incoming_root", "final_parent"},
        "input_bundle",
    )
    bundle_record = _file_record(input_bundle["manifest"], "input_bundle.manifest", path_role="pod")
    try:
        bundle = load_bundle_manifest(Path(str(bundle_record["pod_path"])))
    except TransferBundleError as exc:
        raise WorkloadError("input bundle manifest is invalid") from exc
    bundle_digest = _sha(input_bundle["bundle_digest"], "input_bundle.bundle_digest")
    incoming_root = Path(_text(input_bundle["incoming_root"], "input_bundle.incoming_root"))
    final_parent = Path(_text(input_bundle["final_parent"], "input_bundle.final_parent"))
    if (
        bundle.bundle_digest != bundle_digest
        or incoming_root != Path(f"/workspace/mantis/transfer/incoming/{bundle_digest}/files")
        or final_parent != Path("/workspace/mantis/inputs")
    ):
        raise WorkloadError("input bundle paths or identity are invalid")
    try:
        verify_and_promote(incoming_root, final_parent, bundle)
    except TransferBundleError as exc:
        raise WorkloadError("staged input bundle failed verification") from exc


def _deployment_receipt(manifest: Mapping[str, Any], *, path_role: str) -> dict[str, Any]:
    record = manifest["image"]["self_check"]
    try:
        payload = json.loads(Path(str(record[f"{path_role}_path"])).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkloadError("image self-check receipt is invalid") from exc
    if not isinstance(payload, dict):
        raise WorkloadError("image self-check receipt is invalid")
    return payload


def _bound_workload(
    manifest: Mapping[str, Any], pod_path: Path, *, path_role: str = "controller"
) -> dict[str, Any]:
    input_root = Path(str(manifest["input_bundle"]["final_parent"])) / str(
        manifest["input_bundle"]["bundle_digest"]
    )
    environment = {
        "MANTIS_RUN_ID": str(manifest["run_id"]),
        "MANTIS_WORKLOAD_MANIFEST": str(pod_path),
        "HF_HOME": str(input_root / "cache" / "huggingface"),
        "HF_HUB_OFFLINE": "1",
    }
    receipt = _deployment_receipt(manifest, path_role=path_role)
    if receipt.get("scope") == "static_image":
        docker_args = f"uv run mantis-v2 workload-execute --manifest {pod_path}"
    elif receipt.get("scope") == "official_bootstrap":
        bootstrap = manifest["bootstrap"]
        archive = bootstrap["source_archive"]
        project_root = str(bootstrap["project_root"])
        venv_path = str(bootstrap["venv_path"])
        source_sha256 = str(archive["sha256"])
        source_path = str(archive["pod_path"])
        uv_version = str(bootstrap["uv_version"])
        environment.update(
            {
                "MANTIS_BASE_IMAGE": str(manifest["image"]["ref"]),
                "MANTIS_SOURCE_REVISION": str(manifest["source_revision"]),
                "MANTIS_SOURCE_TREE": str(receipt["inventory"]["identities"]["source_tree"]),
                "MANTIS_LOCK_SHA256": str(manifest["dependency_lock"]["sha256"]),
                "MANTIS_LOCK_PATH": f"{project_root}/uv.lock",
                "MANTIS_IMAGE_CONTRACT_SHA256": str(manifest["image"]["self_check"]["sha256"]),
                "MANTIS_UV_VERSION": uv_version,
                "UV_CACHE_DIR": "/opt/mantis/cache",
                "UV_PROJECT_ENVIRONMENT": venv_path,
            }
        )
        temporary = f"{project_root}.partial"
        shell = (
            "set -euo pipefail; /start.sh & "
            f"archive={shlex.quote(source_path)}; project={shlex.quote(project_root)}; "
            f"expected={shlex.quote(source_sha256)}; temporary={shlex.quote(temporary)}; "
            'printf "%s  %s\\n" "$expected" "$archive" | sha256sum -c -; '
            'rm -rf "$temporary"; mkdir -p "$temporary"; '
            'tar -xzf "$archive" -C "$temporary"; '
            'printf "%s\\n" "$expected" > "$temporary/.mantis-source.sha256"; '
            'rm -rf "$project"; mv "$temporary" "$project"; '
            'cd "$project"; '
            "uv sync --frozen --no-dev; "
            f"exec uv run mantis-v2 workload-execute --manifest {shlex.quote(str(pod_path))}"
        )
        docker_args = f"bash -lc {shlex.quote(shell)}"
    else:
        raise WorkloadError("unsupported deployment receipt scope")
    return {
        "manifest_digest": manifest["manifest_digest"],
        "docker_args": docker_args,
        "environment": environment,
    }


def bind_workload_decision(
    *,
    manifest_path: str | Path,
    decision: Mapping[str, object],
    pod_manifest_path: str | Path,
    output_path: str | Path,
    evaluated_at: datetime,
) -> Path:
    """Bind an approved provider decision to exactly one immediate workload."""
    manifest = validate_workload_manifest(manifest_path)
    validate_launch_decision(decision, evaluated_at)
    if "workload" in decision:
        raise WorkloadError("launch decision is already workload-bound")
    pod_path = Path(pod_manifest_path)
    try:
        authorization = load_launch_authorization(manifest["authorization"]["controller_path"])
    except RunpodConfigError as exc:
        raise WorkloadError("launch authorization file is invalid") from exc
    decision_projected = _decimal(
        decision.get("projected_spend_usd"), "decision.projected_spend_usd"
    )
    manifest_maximum = _decimal(
        manifest["budget_guard"]["next_cell_maximum_usd"],
        "budget_guard.next_cell_maximum_usd",
    )
    deployment_receipt = _deployment_receipt(manifest, path_role="controller")
    official_bootstrap_mismatch = deployment_receipt.get("scope") == "official_bootstrap" and (
        decision.get("template_id") != deployment_receipt.get("template_id")
        or decision.get("registry_auth_id") != ""
    )
    if (
        not pod_path.is_absolute()
        or not pod_path.is_relative_to("/workspace/mantis")
        or pod_path.name != "launch-manifest.json"
    ):
        raise WorkloadError("Pod manifest path must be absolute under /workspace/mantis")
    if (
        decision.get("run_name") != manifest["run_id"]
        or decision.get("image_ref") != manifest["image"]["ref"]
        or decision.get("stage") != manifest["budget_guard"]["stage"]
        or decision.get("maximum_duration_seconds") != manifest["maximum_duration_seconds"]
        or decision.get("startup_allowance_seconds")
        != manifest["monitor"]["first_heartbeat_seconds"]
        or str(decision.get("observed_price_usd_per_gpu_hour"))
        != manifest["quoted_rates"]["compute_usd_per_hour"]
        or decision.get("authorization_digest") != authorization.digest
        or decision.get("ledger_digest")
        != load_spend_ledger(manifest["spend_ledger"]["controller_path"]).digest
        or str(decision.get("authorization_expires_at"))
        != authorization.expires_at.isoformat().replace("+00:00", "Z")
        or manifest["authorization"]["expires_at"]
        != authorization.expires_at.isoformat().replace("+00:00", "Z")
        or decision_projected > manifest_maximum
        or official_bootstrap_mismatch
    ):
        raise WorkloadError("launch decision differs from the workload manifest")
    core = {
        **{key: value for key, value in decision.items() if key != "decision_digest"},
        "workload": _bound_workload(manifest, pod_path),
    }
    bound = {**core, "decision_digest": _digest(core)}
    output = Path(output_path)
    _atomic_idempotent(output, bound)
    return output


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _checkpoint_sha256(manifest: Mapping[str, Any]) -> str:
    checkpoint = Path(str(manifest["artifacts"]["pod"])) / "checkpoints" / "latest.pt"
    return _sha256_file(checkpoint) if checkpoint.is_file() else "0" * 64


def _run_diagnostics(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(command, 127, "", type(exc).__name__)


def execute_workload_manifest(
    manifest_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    process_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    service_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    runtime_self_check: Callable[[], Mapping[str, Any]] = image_self_check,
    diagnostics_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run_diagnostics,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Revalidate one staged manifest inside the Pod and run training immediately."""
    envelope = _load_manifest_document(manifest_path)
    _promote_pod_input_bundle(envelope)
    manifest = validate_workload_manifest(manifest_path, path_role="pod")
    environment = dict(os.environ if environ is None else environ)
    pod_id = _text(environment.get("RUNPOD_POD_ID"), "RUNPOD_POD_ID")
    token_record = manifest["monitor"]["token"]
    token_path = Path(str(token_record["pod_path"]))
    try:
        heartbeat_token = token_path.read_text().strip()
    except OSError as exc:
        raise WorkloadError("heartbeat token cannot be read inside the Pod") from exc
    if not heartbeat_token:
        raise WorkloadError("heartbeat token is empty")
    heartbeat_path = Path(str(manifest["monitor"]["heartbeat"]))
    if not heartbeat_path.is_absolute():
        raise WorkloadError("heartbeat path must be absolute inside the Pod")
    artifact_root = Path(str(manifest["artifacts"]["pod"]))
    artifact_root.mkdir(parents=True, exist_ok=True)
    runtime_check = dict(runtime_self_check())
    if runtime_check.get("passed") is not True or runtime_check.get("scope") != "runtime_cuda":
        raise WorkloadError("in-Pod CUDA self-check did not pass")
    _replace_json(artifact_root / "runtime-image-self-check.json", runtime_check)
    stdout_path = artifact_root / "launcher.stdout.log"
    diagnostics_path = artifact_root / "runtime-diagnostics.json"
    command = [str(item) for item in manifest["start_command"]]
    with stdout_path.open("ab", buffering=0) as output:
        tensorboard = (
            None
            if manifest.get("workload_kind") == "frozen_expected_r"
            else service_factory(
                list(tensorboard_command(artifact_root)),
                stdout=output,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        )
        if tensorboard is not None and tensorboard.poll() is not None:
            raise WorkloadError("TensorBoard failed before training started")
        try:
            process = process_factory(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            progress = 0
            gpu = environment.get(
                "NVIDIA_VISIBLE_DEVICES",
                environment.get("CUDA_VISIBLE_DEVICES", "unreported"),
            )
            while True:
                return_code = process.poll()
                tensorboard_return_code = tensorboard.poll() if tensorboard is not None else None
                if return_code is None and tensorboard_return_code is not None:
                    process.terminate()
                    process.wait(timeout=10)
                    raise WorkloadError("TensorBoard exited while training was active")
                gpu_diagnostics = diagnostics_runner(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ]
                )
                _replace_json(
                    diagnostics_path,
                    {
                        "schema_version": 1,
                        "run_id": manifest["run_id"],
                        "pod_id": pod_id,
                        "process_id": str(process.pid),
                        "tensorboard_process_id": (
                            str(tensorboard.pid) if tensorboard is not None else None
                        ),
                        "tensorboard_return_code": tensorboard_return_code,
                        "process_return_code": return_code,
                        "gpu_allocation": gpu,
                        "nvidia_smi_return_code": gpu_diagnostics.returncode,
                        "nvidia_smi_stdout": gpu_diagnostics.stdout[-65536:],
                        "nvidia_smi_stderr": gpu_diagnostics.stderr[-65536:],
                        "timestamp": now().astimezone(UTC).isoformat().replace("+00:00", "Z"),
                    },
                )
                stage = (
                    "training"
                    if return_code is None
                    else ("complete" if return_code == 0 else "failed")
                )
                heartbeat = sign_heartbeat(
                    {
                        "schema_version": 1,
                        "run_id": manifest["run_id"],
                        "pod_id": pod_id,
                        "process_id": str(process.pid),
                        "stage": stage,
                        "progress": progress,
                        "checkpoint_sha256": _checkpoint_sha256(manifest),
                        "gpu_allocation": gpu,
                        "timestamp": now().astimezone(UTC).isoformat().replace("+00:00", "Z"),
                    },
                    heartbeat_token,
                )
                _replace_json(heartbeat_path, heartbeat)
                if return_code is not None:
                    return {
                        "schema_version": 1,
                        "status": stage,
                        "return_code": return_code,
                        "heartbeat": str(heartbeat_path),
                        "stdout": str(stdout_path),
                        "diagnostics": str(diagnostics_path),
                    }
                progress += 1
                sleep(float(manifest["monitor"]["poll_seconds"]))
        finally:
            if tensorboard is not None:
                tensorboard.terminate()
                tensorboard.wait(timeout=10)


def sign_heartbeat(payload: Mapping[str, Any], token: str) -> dict[str, Any]:
    """Attach an HMAC signature without persisting the heartbeat token."""
    if not token:
        raise WorkloadError("heartbeat token is required")
    core = dict(payload)
    if set(core) != {
        "schema_version",
        "run_id",
        "pod_id",
        "process_id",
        "stage",
        "progress",
        "checkpoint_sha256",
        "gpu_allocation",
        "timestamp",
    }:
        raise WorkloadError("heartbeat keys mismatch")
    signature = hmac.new(token.encode(), _canonical(core), hashlib.sha256).hexdigest()
    return {**core, "signature": signature}


def _validated_heartbeat(
    value: Mapping[str, Any],
    *,
    token: str,
    run_id: str,
    pod_id: str,
    observed_at: datetime,
    provider_started_at: datetime,
    previous_timestamp: datetime | None,
    previous_progress: int | None,
) -> tuple[dict[str, Any], datetime]:
    signature = value.get("signature")
    core = {key: item for key, item in value.items() if key != "signature"}
    if not isinstance(signature, str) or not hmac.compare_digest(
        signature,
        hmac.new(token.encode(), _canonical(core), hashlib.sha256).hexdigest(),
    ):
        raise WorkloadError("heartbeat authentication failed")
    if (
        core.get("schema_version") != 1
        or core.get("run_id") != run_id
        or core.get("pod_id") != pod_id
    ):
        raise WorkloadError("heartbeat run identity mismatch")
    if core.get("stage") not in {"training", "complete", "blocked", "failed"}:
        raise WorkloadError("heartbeat stage is invalid")
    if (
        not isinstance(core.get("process_id"), str)
        or not isinstance(core.get("gpu_allocation"), str)
        or not isinstance(core.get("progress"), int)
        or isinstance(core.get("progress"), bool)
        or int(core["progress"]) < 0
        or not isinstance(core.get("checkpoint_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(core["checkpoint_sha256"]))
    ):
        raise WorkloadError("heartbeat process or checkpoint identity is invalid")
    try:
        timestamp = datetime.fromisoformat(
            str(core["timestamp"]).replace("Z", "+00:00")
        ).astimezone(UTC)
    except (KeyError, ValueError) as exc:
        raise WorkloadError("heartbeat timestamp is invalid") from exc
    if timestamp > observed_at.astimezone(UTC) + timedelta(seconds=5):
        raise WorkloadError("heartbeat timestamp is in the future")
    if timestamp < provider_started_at.astimezone(UTC):
        raise WorkloadError("heartbeat predates the current Pod start")
    if previous_timestamp is not None and timestamp < previous_timestamp:
        raise WorkloadError("heartbeat timestamp regressed")
    if previous_progress is not None and int(core["progress"]) < previous_progress:
        raise WorkloadError("heartbeat progress regressed")
    return core, timestamp


def _append_event(
    state_root: Path,
    event: str,
    observed_at: datetime,
    run_id: str,
    pod_id: str | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ledger = state_root / "ledger"
    ledger.mkdir(parents=True, exist_ok=True)
    existing = sorted(ledger.glob("*.json"))
    previous_digest: str | None = None
    if existing:
        try:
            previous = json.loads(existing[-1].read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkloadError("cost ledger is invalid") from exc
        previous_digest = previous.get("event_digest")
        if not isinstance(previous_digest, str):
            raise WorkloadError("cost ledger digest chain is invalid")
    core = {
        "schema_version": 1,
        "sequence": len(existing) + 1,
        "event": event,
        "observed_at": observed_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "run_id": run_id,
        "pod_id": pod_id,
        "previous_event_digest": previous_digest,
        **dict(extra or {}),
    }
    payload = {**core, "event_digest": _digest(core)}
    path = ledger / f"{len(existing) + 1:06d}-{event}.json"
    _atomic_idempotent(path, payload)
    return payload


def _live_cost(
    manifest: Mapping[str, Any], started_at: datetime, observed_at: datetime
) -> dict[str, str]:
    elapsed_hours = Decimal(str(max(0.0, (observed_at - started_at).total_seconds()))) / Decimal(
        "3600"
    )
    rates = manifest["quoted_rates"]
    compute = Decimal(rates["compute_usd_per_hour"]) * elapsed_hours
    storage = (
        Decimal(rates["storage_usd_per_gb_hour"]) * Decimal(rates["storage_gb"]) * elapsed_hours
    )
    return {
        "elapsed_seconds": str(int(elapsed_hours * Decimal("3600"))),
        "estimated_compute_usd": str(compute),
        "estimated_storage_usd": str(storage),
        "estimated_live_accrual_usd": str(compute + storage),
    }


def _cost_context(
    manifest: Mapping[str, Any],
    observed_at: datetime,
    started_at: datetime | None = None,
) -> dict[str, Any]:
    live = (
        _live_cost(manifest, started_at, observed_at)
        if started_at is not None
        else {
            "elapsed_seconds": "0",
            "estimated_compute_usd": "0",
            "estimated_storage_usd": "0",
            "estimated_live_accrual_usd": "0",
        }
    )
    return {
        "quoted_rates": dict(manifest["quoted_rates"]),
        "storage_size_gb": manifest["quoted_rates"]["storage_gb"],
        **live,
    }


def _verified_absence(
    *,
    adapter: LifecycleAdapter,
    pod_id: str,
    run_id: str,
    state_root: Path,
    manifest: Mapping[str, Any],
    provider_started: datetime,
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
) -> None:
    for _ in range(20):
        inventory = adapter.inventory()
        if not any(item.get("id") == pod_id for item in inventory):
            _append_event(
                state_root,
                "provider_absence_verified",
                now(),
                run_id,
                pod_id,
                _cost_context(manifest, now(), provider_started),
            )
            return
        sleep(30)
    raise WorkloadError("provider absence was not verified after delete")


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, dict):
        return {key: _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    return value


def _bounded_call(
    callback: Callable[..., Any],
    *args: Any,
    deadline: datetime,
    now: Callable[[], datetime],
) -> Any:
    """Run a best-effort callback without allowing it to extend the delete deadline."""
    remaining = (deadline - now().astimezone(UTC)).total_seconds()
    if remaining <= 0:
        raise WorkloadError("diagnostic grace period expired")
    outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            outcome.put((True, callback(*args, deadline)), block=False)
        except BaseException as exc:
            outcome.put((False, exc), block=False)

    thread = threading.Thread(target=invoke, daemon=True, name="mantis-grace-callback")
    thread.start()
    try:
        succeeded, value = outcome.get(timeout=remaining)
    except queue.Empty as exc:
        raise WorkloadError("diagnostic callback exceeded the 2-minute grace period") from exc
    if not succeeded:
        raise value
    return value


def _monitor_created_pod(
    *,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    pod_id: str,
    run_id: str,
    state_root: Path,
    adapter: LifecycleAdapter,
    heartbeat_token: str,
    heartbeat_source: Callable[[str, str], Mapping[str, Any] | None],
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
) -> tuple[str, datetime]:
    status = adapter.status(pod_id)
    try:
        provider_started = datetime.fromisoformat(
            str(status["lastStartedAt"]).replace("Z", "+00:00")
        ).astimezone(UTC)
    except (KeyError, ValueError) as exc:
        raise WorkloadError("provider lastStartedAt is required for startup supervision") from exc
    _append_event(
        state_root,
        "pod_created",
        now(),
        run_id,
        pod_id,
        {
            "decision_digest": receipt["decision_digest"],
            "provider_last_started_at": provider_started.isoformat().replace("+00:00", "Z"),
            **_cost_context(manifest, now(), provider_started),
        },
    )
    last_heartbeat_digest: str | None = None
    last_timestamp: datetime | None = None
    last_progress: int | None = None
    first_seen = False
    missed = 0
    while True:
        observed_at = now().astimezone(UTC)
        status = adapter.status(pod_id)
        if status.get("desiredStatus") != "RUNNING":
            return "provider_not_running", provider_started
        heartbeat = heartbeat_source(run_id, pod_id)
        if heartbeat is not None:
            validated, heartbeat_timestamp = _validated_heartbeat(
                heartbeat,
                token=heartbeat_token,
                run_id=run_id,
                pod_id=pod_id,
                observed_at=observed_at,
                provider_started_at=provider_started,
                previous_timestamp=last_timestamp,
                previous_progress=last_progress,
            )
            heartbeat_digest = _digest(validated)
            if heartbeat_digest != last_heartbeat_digest:
                first_seen = True
                missed = 0
                last_heartbeat_digest = heartbeat_digest
                last_timestamp = heartbeat_timestamp
                last_progress = int(validated["progress"])
                _append_event(
                    state_root,
                    "heartbeat",
                    observed_at,
                    run_id,
                    pod_id,
                    {
                        "process_id": validated["process_id"],
                        "stage": validated["stage"],
                        "progress": validated["progress"],
                        "checkpoint_sha256": validated["checkpoint_sha256"],
                        "gpu_allocation": validated["gpu_allocation"],
                        **_cost_context(manifest, observed_at, provider_started),
                    },
                )
                if validated["stage"] == "complete":
                    return "workload_complete", provider_started
                if validated["stage"] == "blocked":
                    return "workload_blocked", provider_started
                if validated["stage"] == "failed":
                    return "workload_failed", provider_started
            elif first_seen:
                missed += 1
        elif first_seen:
            missed += 1
        startup_allowance = int(manifest["monitor"]["first_heartbeat_seconds"])
        if not first_seen and observed_at >= provider_started + timedelta(
            seconds=startup_allowance
        ):
            return "startup_heartbeat_timeout", provider_started
        if first_seen and missed >= 4:
            return "heartbeat_missed", provider_started
        sleep(30)


def supervise_workload(
    *,
    manifest_path: str | Path,
    decision: Mapping[str, object],
    state_root: Path,
    adapter: LifecycleAdapter,
    heartbeat_token: str,
    heartbeat_source: Callable[[str, str], Mapping[str, Any] | None],
    collect_diagnostics: Callable[[str, datetime], Mapping[str, Any]],
    checkpoint: Callable[[str, datetime], str | None],
    replicate: Callable[[Mapping[str, Any]], None],
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    """Start before create, supervise one immediate workload, and always delete on exit."""
    manifest = validate_workload_manifest(manifest_path)
    run_id = str(manifest["run_id"])
    workload = decision.get("workload")
    workload_environment = workload.get("environment") if isinstance(workload, Mapping) else None
    pod_manifest = (
        Path(str(workload_environment.get("MANTIS_WORKLOAD_MANIFEST", "")))
        if isinstance(workload_environment, Mapping)
        else Path("")
    )
    expected_workload = _bound_workload(manifest, pod_manifest)
    if not isinstance(workload, Mapping) or dict(workload) != expected_workload:
        raise WorkloadError("launch decision does not bind the immediate workload manifest")
    _append_event(
        state_root,
        "watchdog_started",
        now(),
        run_id,
        None,
        _cost_context(manifest, now()),
    )
    try:
        receipt = launch_pod(decision=decision, state_root=state_root, adapter=adapter, now=now)
    except LifecycleError as launch_error:
        receipt = None
        for _ in range(4):
            try:
                receipt = reconcile_launch(
                    decision=decision,
                    state_root=state_root,
                    adapter=adapter,
                    now=now,
                )
                break
            except LifecycleError as reconciliation_error:
                if str(reconciliation_error) == "launch_attempt_not_found":
                    raise launch_error from reconciliation_error
                if str(reconciliation_error) not in {
                    "provider_create_outcome_unresolved",
                    "provider_create_requires_termination",
                }:
                    raise
                sleep(30)
        if receipt is None:
            remaining: list[Mapping[str, object]] = []
            for attempt in range(5):
                remaining = [
                    pod
                    for pod in adapter.inventory()
                    if pod.get("name") == decision.get("run_name")
                ]
                if not remaining:
                    break
                for pod in remaining:
                    pod_id = pod.get("id")
                    if isinstance(pod_id, str):
                        adapter.terminate(pod_id)
                if attempt < 4:
                    sleep(30)
            remaining = [
                pod for pod in adapter.inventory() if pod.get("name") == decision.get("run_name")
            ]
            if remaining:
                raise WorkloadError(
                    "uncertain Pod create could not be terminated"
                ) from launch_error
            raise WorkloadError(
                "uncertain Pod create reconciled to verified absence"
            ) from launch_error
    if receipt is None:
        raise WorkloadError("Pod create did not produce a receipt")
    pod_id = str(receipt["pod_id"])
    failure: Exception | None = None
    provider_started = now().astimezone(UTC)
    try:
        reason, provider_started = _monitor_created_pod(
            manifest=manifest,
            receipt=receipt,
            pod_id=pod_id,
            run_id=run_id,
            state_root=state_root,
            adapter=adapter,
            heartbeat_token=heartbeat_token,
            heartbeat_source=heartbeat_source,
            now=now,
            sleep=sleep,
        )
    except Exception as exc:
        failure = exc
        reason = "supervisor_error"
        _append_event(
            state_root,
            "supervisor_error",
            now(),
            run_id,
            pod_id,
            {
                "error_type": type(exc).__name__,
                **_cost_context(manifest, now(), provider_started),
            },
        )

    def replicate_and_record() -> None:
        nonlocal failure
        try:
            replicate(manifest)
            _append_event(
                state_root,
                "artifacts_verified",
                now(),
                run_id,
                pod_id,
                _cost_context(manifest, now(), provider_started),
            )
        except Exception as exc:
            if failure is None:
                failure = exc
            _append_event(
                state_root,
                "artifacts_failed",
                now(),
                run_id,
                pod_id,
                {
                    "error_type": type(exc).__name__,
                    **_cost_context(manifest, now(), provider_started),
                },
            )

    if reason != "workload_complete":
        grace_deadline = now().astimezone(UTC) + timedelta(seconds=120)
        try:
            diagnostics = _redact(
                dict(
                    _bounded_call(
                        collect_diagnostics,
                        pod_id,
                        deadline=grace_deadline,
                        now=now,
                    )
                ),
                heartbeat_token,
            )
            diagnostics_core = {
                "schema_version": 1,
                "pod_id": pod_id,
                "run_id": run_id,
                "reason": reason,
                "captured_at": now().astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "evidence": diagnostics,
            }
            diagnostics_payload = {
                **diagnostics_core,
                "diagnostics_digest": _digest(diagnostics_core),
            }
            _atomic_idempotent(
                state_root / "failures" / f"{pod_id}-diagnostics.json", diagnostics_payload
            )
            _append_event(
                state_root,
                "diagnostics_captured",
                now(),
                run_id,
                pod_id,
                {
                    "diagnostics_digest": diagnostics_payload["diagnostics_digest"],
                    **_cost_context(manifest, now(), provider_started),
                },
            )
        except Exception as exc:
            _append_event(
                state_root,
                "diagnostics_failed",
                now(),
                run_id,
                pod_id,
                {
                    "error_type": type(exc).__name__,
                    **_cost_context(manifest, now(), provider_started),
                },
            )
        if now().astimezone(UTC) < grace_deadline:
            try:
                checkpoint_sha = _bounded_call(
                    checkpoint,
                    pod_id,
                    deadline=grace_deadline,
                    now=now,
                )
                _append_event(
                    state_root,
                    "checkpoint_requested",
                    now(),
                    run_id,
                    pod_id,
                    {
                        "checkpoint_sha256": checkpoint_sha,
                        **_cost_context(manifest, now(), provider_started),
                    },
                )
            except Exception as exc:
                _append_event(
                    state_root,
                    "checkpoint_failed",
                    now(),
                    run_id,
                    pod_id,
                    {
                        "error_type": type(exc).__name__,
                        **_cost_context(manifest, now(), provider_started),
                    },
                )
    if reason == "workload_complete":
        replicate_and_record()
    try:
        termination = terminate_pod_for_reason(
            pod_id=pod_id,
            run_name=str(decision["run_name"]),
            reason=str(reason),
            state_root=state_root,
            adapter=adapter,
            now=now,
        )
    except LifecycleError as termination_error:
        termination = None
        for attempt in range(5):
            try:
                termination = reconcile_termination(
                    pod_id=pod_id,
                    run_name=str(decision["run_name"]),
                    state_root=state_root,
                    adapter=adapter,
                    now=now,
                )
                break
            except LifecycleError as reconciliation_error:
                if str(reconciliation_error) == "termination_attempt_not_found":
                    raise termination_error from reconciliation_error
                if str(reconciliation_error) != "provider_terminate_outcome_unresolved":
                    raise
                with suppress(Exception):
                    adapter.terminate(pod_id)
                if attempt < 4:
                    sleep(30)
        if termination is None:
            raise WorkloadError("Pod termination could not be reconciled") from termination_error
    _append_event(
        state_root,
        "pod_deleted",
        now(),
        run_id,
        pod_id,
        {
            "reason": reason,
            "termination_receipt_digest": _digest(termination),
            **_cost_context(manifest, now(), provider_started),
        },
    )
    _verified_absence(
        adapter=adapter,
        pod_id=pod_id,
        run_id=run_id,
        state_root=state_root,
        manifest=manifest,
        provider_started=provider_started,
        now=now,
        sleep=sleep,
    )
    if reason != "workload_complete":
        replicate_and_record()
    billing = adapter.billing(pod_id)
    _append_event(
        state_root,
        "billing_reconciled" if billing is not None else "billing_pending",
        now(),
        run_id,
        pod_id,
        {
            "billing": dict(billing) if billing is not None else None,
            **_cost_context(manifest, now(), provider_started),
        },
    )
    if failure is not None:
        raise WorkloadError("workload supervision failed after paid create") from failure
    if reason == "workload_complete":
        return {"schema_version": 1, "status": "complete", "pod_id": pod_id}
    return {
        "schema_version": 1,
        "status": "terminated",
        "reason": reason,
        "pod_id": pod_id,
    }
