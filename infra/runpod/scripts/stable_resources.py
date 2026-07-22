#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPECTED_RESOURCES = {
    "restapi_object.network_volume": "/networkvolumes",
    "restapi_object.pod_template": "/templates",
}
ALLOWED_ACTIONS = {("create",), ("no-op",), ("update",)}
SECRET_ENV_NAMES = ("RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY")
DESIRED_KEYS = {"schema_version", "lifecycle_owner", "volume", "template"}
VOLUME_KEYS = {"dataCenterId", "name", "size"}
TEMPLATE_KEYS = {
    "category",
    "containerDiskInGb",
    "dockerEntrypoint",
    "dockerStartCmd",
    "env",
    "imageName",
    "isPublic",
    "isServerless",
    "name",
    "ports",
    "readme",
    "volumeInGb",
    "volumeMountPath",
}
AUTHORIZATION_KEYS = {
    "authorization_type",
    "authorized_at",
    "authorized_by",
    "desired_digest",
    "expires_at",
    "plan_sha256",
    "resource_addresses",
    "schema_version",
}


class StableResourceError(ValueError):
    pass


def _load_object(path: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StableResourceError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StableResourceError(f"{path} must contain a JSON object")
    return value


def _write_report(path: str, report: dict[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        raise StableResourceError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _desired_violations(desired: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if set(desired) != DESIRED_KEYS:
        violations.append("desired_contract_has_unknown_fields")
    if desired.get("schema_version") != 1:
        violations.append("desired_schema_version_must_equal_1")
    if desired.get("lifecycle_owner") != "terraform-stable-v1":
        violations.append("lifecycle_owner_must_equal_terraform_stable_v1")
    volume = desired.get("volume")
    if not isinstance(volume, dict) or set(volume) != VOLUME_KEYS:
        violations.append("volume_contract_fields_mismatch")
    if not isinstance(volume, dict) or volume.get("size") != 150:
        violations.append("volume_size_must_equal_150_gb")
    template = desired.get("template")
    if not isinstance(template, dict) or set(template) != TEMPLATE_KEYS:
        violations.append("template_contract_fields_mismatch")
        return violations
    image_name = template.get("imageName")
    if (
        not isinstance(image_name, str)
        or re.fullmatch(r".+@sha256:[0-9a-f]{64}", image_name) is None
    ):
        violations.append("template_image_must_use_sha256_digest")
    if template.get("isPublic") is not False:
        violations.append("template_must_be_private")
    if template.get("isServerless") is not False:
        violations.append("template_must_not_be_serverless")
    if template.get("ports") != ["22/tcp"]:
        violations.append("template_ports_must_equal_ssh_only")
    if template.get("env") != {}:
        violations.append("template_env_must_be_empty")
    if template.get("volumeInGb") != 0:
        violations.append("template_local_volume_must_be_zero")
    if template.get("volumeMountPath") != "/workspace/mantis":
        violations.append("template_mount_must_equal_workspace")
    if template.get("containerDiskInGb") != 50:
        violations.append("template_container_disk_must_equal_50_gb")
    if template.get("readme") != "lifecycle-owner: terraform-stable-v1":
        violations.append("template_owner_marker_mismatch")
    return violations


def _policy(
    desired: dict[str, Any], plan: dict[str, Any], adoption: dict[str, Any]
) -> dict[str, Any]:
    actions: dict[str, list[str]] = {}
    violations = _desired_violations(desired)
    adoption_resources = adoption.get("resources")
    if (
        adoption.get("schema_version") != 1
        or adoption.get("plan_allowed") is not True
        or not isinstance(adoption_resources, dict)
        or set(adoption_resources) != set(EXPECTED_RESOURCES)
    ) or any(
        not isinstance(resource, dict) or resource.get("status") not in {"absent", "adopted"}
        for resource in adoption_resources.values()
    ):
        violations.append("adoption_not_plan_allowed")
    provider = (
        plan.get("configuration", {}).get("provider_config", {}).get("restapi", {})
        if isinstance(plan.get("configuration"), dict)
        else {}
    )
    if plan.get("format_version") != "1.2":
        violations.append("unsupported_plan_format")
    if not isinstance(provider, dict) or str(provider.get("full_name", "")).lower() != (
        "registry.opentofu.org/mastercard/restapi"
    ):
        violations.append("provider_source_mismatch")
    if not isinstance(provider, dict) or provider.get("version_constraint") not in {
        "3.0.0",
        "= 3.0.0",
    }:
        violations.append("provider_not_exactly_pinned")
    expected_payloads = {
        "restapi_object.network_volume": desired.get("volume"),
        "restapi_object.pod_template": desired.get("template"),
    }
    changes = plan.get("resource_changes")
    if not isinstance(changes, list):
        raise StableResourceError("plan.resource_changes must be a list")
    serialized_plan = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    for secret_name in SECRET_ENV_NAMES:
        secret_value = os.environ.get(secret_name, "")
        if len(secret_value) >= 8 and secret_value in serialized_plan:
            violations.append(f"secret_value_present_in_plan:{secret_name}")
    seen: set[str] = set()
    for change in changes:
        if not isinstance(change, dict) or not isinstance(change.get("address"), str):
            violations.append("invalid_resource_change")
            continue
        address = change["address"]
        details = change.get("change")
        if address in seen:
            violations.append(f"duplicate_resource:{address}")
        seen.add(address)
        if address not in EXPECTED_RESOURCES or not isinstance(details, dict):
            violations.append(f"unexpected_resource:{address}")
            continue
        resource_actions = details.get("actions")
        after = details.get("after")
        if not isinstance(resource_actions, list) or not all(
            isinstance(action, str) for action in resource_actions
        ):
            violations.append(f"invalid_actions:{address}")
            continue
        actions[address] = resource_actions
        if tuple(resource_actions) not in ALLOWED_ACTIONS:
            violations.append(f"forbidden_actions:{address}")
        if change.get("mode") != "managed" or change.get("type") != "restapi_object":
            violations.append(f"invalid_resource_ownership:{address}")
        if not isinstance(after, dict) or after.get("path") != EXPECTED_RESOURCES[address]:
            violations.append(f"invalid_path:{address}")
            continue
        after_unknown = details.get("after_unknown", {})
        if not isinstance(after_unknown, dict):
            violations.append(f"invalid_after_unknown:{address}")
        else:
            for field in ("path", "data"):
                if after_unknown.get(field):
                    violations.append(f"policy_field_unknown:{address}:{field}")
        try:
            payload = json.loads(after.get("data", ""))
        except (TypeError, json.JSONDecodeError):
            violations.append(f"invalid_data:{address}")
            continue
        if payload != expected_payloads[address]:
            violations.append(f"contract_mismatch:{address}")
    if set(actions) != set(EXPECTED_RESOURCES):
        violations.append("stable_resource_set_mismatch")
    return {
        "accepted": not violations,
        "actions": actions,
        "lifecycle_owner": desired.get("lifecycle_owner"),
        "schema_version": 1,
        "violations": sorted(set(violations)),
    }


def _adoption(
    desired: dict[str, Any], inventory: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    desired_violations = _desired_violations(desired)
    if desired_violations:
        return {
            "plan_allowed": False,
            "resources": {},
            "schema_version": 1,
            "violations": sorted(set(desired_violations)),
        }
    if inventory.get("schema_version") != 1 or state.get("schema_version") != 1:
        raise StableResourceError("inventory and state schema_version must equal 1")
    state_resources = state.get("resources")
    if not isinstance(state_resources, dict):
        raise StableResourceError("state.resources must be an object")
    definitions = (
        (
            "restapi_object.network_volume",
            "volumes",
            "volume",
            "/networkvolumes",
        ),
        ("restapi_object.pod_template", "templates", "template", "/templates"),
    )
    resources: dict[str, dict[str, Any]] = {}
    for address, inventory_key, desired_key, import_path in definitions:
        observed = inventory.get(inventory_key)
        expected = desired.get(desired_key)
        if not isinstance(observed, list) or not isinstance(expected, dict):
            raise StableResourceError(f"invalid adoption input for {address}")
        matches = [
            item
            for item in observed
            if isinstance(item, dict) and item.get("name") == expected.get("name")
        ]
        if not matches:
            resources[address] = {"status": "absent"}
            continue
        if len(matches) > 1:
            resources[address] = {"status": "ambiguous_duplicates"}
            continue
        match = matches[0]
        resource_id = match.get("id")
        if not isinstance(resource_id, str) or not resource_id:
            resources[address] = {"status": "invalid_inventory_id"}
            continue
        observed_contract = {key: value for key, value in match.items() if key != "id"}
        if observed_contract != expected:
            resources[address] = {"id": resource_id, "status": "identity_conflict"}
            continue
        state_id = state_resources.get(address)
        if state_id is None:
            resources[address] = {
                "id": resource_id,
                "import_id": f"{import_path}/{resource_id}",
                "status": "import_required",
            }
        elif state_id == resource_id:
            resources[address] = {"id": resource_id, "status": "adopted"}
        else:
            resources[address] = {"id": resource_id, "status": "state_identity_conflict"}
    return {
        "plan_allowed": all(
            resource["status"] in {"absent", "adopted"} for resource in resources.values()
        ),
        "resources": resources,
        "schema_version": 1,
    }


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise StableResourceError(f"{field} must be an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StableResourceError(f"{field} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StableResourceError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _apply_authorization(
    policy: dict[str, Any],
    authorization: dict[str, Any],
    evaluated_at: str,
    saved_plan_sha256: str,
) -> dict[str, Any]:
    violations: list[str] = []
    if set(authorization) != AUTHORIZATION_KEYS or authorization.get("schema_version") != 1:
        violations.append("authorization_schema_mismatch")
    if authorization.get("authorization_type") != "human":
        violations.append("authorization_must_be_human")
    authorized_by = authorization.get("authorized_by")
    if not isinstance(authorized_by, str) or not authorized_by.strip():
        violations.append("authorized_by_required")
    if policy.get("accepted") is not True or policy.get("violations") != []:
        violations.append("policy_not_accepted")
    for field in ("plan_sha256", "desired_digest"):
        if authorization.get(field) != policy.get(field):
            violations.append(f"authorization_{field}_mismatch")
    if saved_plan_sha256 != policy.get("plan_sha256"):
        violations.append("saved_plan_sha256_mismatch")
    if authorization.get("resource_addresses") != sorted(EXPECTED_RESOURCES):
        violations.append("authorization_resource_addresses_mismatch")
    authorized_at = _parse_timestamp(authorization.get("authorized_at"), "authorized_at")
    expires_at = _parse_timestamp(authorization.get("expires_at"), "expires_at")
    evaluated = _parse_timestamp(evaluated_at, "evaluated_at")
    if evaluated < authorized_at:
        violations.append("authorization_not_yet_valid")
    if evaluated > expires_at:
        violations.append("authorization_expired")
    return {
        "authorization_digest": _canonical_digest(authorization),
        "authorized": not violations,
        "authorized_by": authorized_by,
        "desired_digest": policy.get("desired_digest"),
        "plan_sha256": policy.get("plan_sha256"),
        "schema_version": 1,
        "violations": sorted(set(violations)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    policy = subparsers.add_parser("policy")
    policy.add_argument("--desired", required=True)
    policy.add_argument("--plan", required=True)
    policy.add_argument("--plan-binary", required=True)
    policy.add_argument("--adoption", required=True)
    policy.add_argument("--output", required=True)
    adoption = subparsers.add_parser("adoption")
    adoption.add_argument("--desired", required=True)
    adoption.add_argument("--inventory", required=True)
    adoption.add_argument("--state", required=True)
    adoption.add_argument("--output", required=True)
    apply_authorization = subparsers.add_parser("apply-authorization")
    apply_authorization.add_argument("--policy", required=True)
    apply_authorization.add_argument("--plan-binary", required=True)
    apply_authorization.add_argument("--authorization", required=True)
    apply_authorization.add_argument("--evaluated-at", required=True)
    apply_authorization.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "apply-authorization":
            report = _apply_authorization(
                _load_object(args.policy),
                _load_object(args.authorization),
                args.evaluated_at,
                hashlib.sha256(Path(args.plan_binary).read_bytes()).hexdigest(),
            )
            _write_report(args.output, report)
            return 0 if report["authorized"] else 2
        if args.command == "adoption":
            report = _adoption(
                _load_object(args.desired),
                _load_object(args.inventory),
                _load_object(args.state),
            )
            _write_report(args.output, report)
            return 0 if report["plan_allowed"] else 2
        desired = _load_object(args.desired)
        plan = _load_object(args.plan)
        report = _policy(desired, plan, _load_object(args.adoption))
        report["desired_digest"] = _canonical_digest(desired)
        report["plan_sha256"] = hashlib.sha256(Path(args.plan_binary).read_bytes()).hexdigest()
        _write_report(args.output, report)
    except StableResourceError as exc:
        parser.error(str(exc))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
