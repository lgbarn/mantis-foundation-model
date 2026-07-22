from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[2]
CHECKER = ROOT / "infra" / "runpod" / "scripts" / "stable_resources.py"
TERRAFORM = ROOT / "infra" / "runpod" / "terraform"


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _allowed_adoption(path: Path) -> Path:
    return _write_json(
        path,
        {
            "plan_allowed": True,
            "resources": {
                "restapi_object.network_volume": {"status": "absent"},
                "restapi_object.pod_template": {"status": "absent"},
            },
            "schema_version": 1,
        },
    )


def _desired_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "lifecycle_owner": "terraform-stable-v1",
        "volume": {
            "dataCenterId": "US-MO-1",
            "name": "mantis-v2-standard-volume-v1",
            "size": 150,
        },
        "template": {
            "category": "NVIDIA",
            "containerDiskInGb": 50,
            "dockerEntrypoint": [],
            "dockerStartCmd": [],
            "env": {},
            "imageName": (
                "ghcr.io/lgbarn/mantis-v2-cuda@sha256:"
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            ),
            "isPublic": False,
            "isServerless": False,
            "name": "mantis-v2-private-template-v1",
            "ports": ["22/tcp"],
            "readme": "lifecycle-owner: terraform-stable-v1",
            "volumeInGb": 0,
            "volumeMountPath": "/workspace/mantis",
        },
    }


def _plan(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "configuration": {
            "provider_config": {
                "restapi": {
                    "full_name": "registry.opentofu.org/Mastercard/restapi",
                    "name": "restapi",
                    "version_constraint": "3.0.0",
                }
            }
        },
        "format_version": "1.2",
        "resource_changes": [
            {
                "address": "restapi_object.network_volume",
                "mode": "managed",
                "type": "restapi_object",
                "name": "network_volume",
                "change": {
                    "actions": ["create"],
                    "after": {
                        "path": "/networkvolumes",
                        "data": json.dumps(contract["volume"], sort_keys=True),
                    },
                    "after_unknown": {},
                    "after_sensitive": {},
                },
            },
            {
                "address": "restapi_object.pod_template",
                "mode": "managed",
                "type": "restapi_object",
                "name": "pod_template",
                "change": {
                    "actions": ["create"],
                    "after": {
                        "path": "/templates",
                        "data": json.dumps(contract["template"], sort_keys=True),
                    },
                    "after_unknown": {},
                    "after_sensitive": {},
                },
            },
        ],
    }


def test_policy_accepts_exact_stable_volume_and_private_template(tmp_path: Path) -> None:
    contract = _desired_contract()
    desired = _write_json(tmp_path / "desired.json", contract)
    plan = _write_json(tmp_path / "plan.json", _plan(contract))
    adoption = _allowed_adoption(tmp_path / "adoption.json")
    output = tmp_path / "policy.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "policy",
            "--desired",
            str(desired),
            "--plan",
            str(plan),
            "--plan-binary",
            str(plan),
            "--adoption",
            str(adoption),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "accepted": True,
        "actions": {
            "restapi_object.network_volume": ["create"],
            "restapi_object.pod_template": ["create"],
        },
        "desired_digest": hashlib.sha256(
            (json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "lifecycle_owner": "terraform-stable-v1",
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "schema_version": 1,
        "violations": [],
    }


def test_policy_accepts_exact_imported_no_drift_plan(tmp_path: Path) -> None:
    contract = _desired_contract()
    plan_value = _plan(contract)
    for change in plan_value["resource_changes"]:
        change["change"]["actions"] = ["no-op"]
    desired = _write_json(tmp_path / "desired.json", contract)
    plan = _write_json(tmp_path / "plan.json", plan_value)
    adoption = _write_json(
        tmp_path / "adoption.json",
        {
            "plan_allowed": True,
            "resources": {
                "restapi_object.network_volume": {"id": "volume-123", "status": "adopted"},
                "restapi_object.pod_template": {"id": "template-456", "status": "adopted"},
            },
            "schema_version": 1,
        },
    )
    output = tmp_path / "policy.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "policy",
            "--desired",
            str(desired),
            "--plan",
            str(plan),
            "--plan-binary",
            str(plan),
            "--adoption",
            str(adoption),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["actions"] == {
        "restapi_object.network_volume": ["no-op"],
        "restapi_object.pod_template": ["no-op"],
    }


def test_policy_rejects_a_pod_resource(tmp_path: Path) -> None:
    contract = _desired_contract()
    plan_value = _plan(contract)
    plan_value["resource_changes"].append(
        {
            "address": "restapi_object.pod",
            "mode": "managed",
            "type": "restapi_object",
            "name": "pod",
            "change": {
                "actions": ["create"],
                "after": {"path": "/pods", "data": "{}"},
                "after_unknown": {},
                "after_sensitive": {},
            },
        }
    )
    desired = _write_json(tmp_path / "desired.json", contract)
    plan = _write_json(tmp_path / "plan.json", plan_value)
    adoption = _allowed_adoption(tmp_path / "adoption.json")
    output = tmp_path / "policy.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "policy",
            "--desired",
            str(desired),
            "--plan",
            str(plan),
            "--plan-binary",
            str(plan),
            "--adoption",
            str(adoption),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(output.read_text(encoding="utf-8"))["violations"] == [
        "unexpected_resource:restapi_object.pod"
    ]


def test_policy_rejects_plan_when_adoption_requires_import(tmp_path: Path) -> None:
    contract = _desired_contract()
    desired = _write_json(tmp_path / "desired.json", contract)
    plan = _write_json(tmp_path / "plan.json", _plan(contract))
    adoption = _write_json(
        tmp_path / "adoption.json",
        {
            "plan_allowed": False,
            "resources": {
                "restapi_object.network_volume": {
                    "id": "volume-123",
                    "import_id": "/networkvolumes/volume-123",
                    "status": "import_required",
                },
                "restapi_object.pod_template": {"status": "absent"},
            },
            "schema_version": 1,
        },
    )
    output = tmp_path / "policy.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "policy",
            "--desired",
            str(desired),
            "--plan",
            str(plan),
            "--plan-binary",
            str(plan),
            "--adoption",
            str(adoption),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert (
        "adoption_not_plan_allowed" in json.loads(output.read_text(encoding="utf-8"))["violations"]
    )


def test_policy_rejects_an_unpinned_provider(tmp_path: Path) -> None:
    contract = _desired_contract()
    plan_value = _plan(contract)
    plan_value["configuration"]["provider_config"]["restapi"]["version_constraint"] = ">= 3.0.0"
    desired = _write_json(tmp_path / "desired.json", contract)
    plan = _write_json(tmp_path / "plan.json", plan_value)
    adoption = _allowed_adoption(tmp_path / "adoption.json")
    output = tmp_path / "policy.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "policy",
            "--desired",
            str(desired),
            "--plan",
            str(plan),
            "--plan-binary",
            str(plan),
            "--adoption",
            str(adoption),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(output.read_text(encoding="utf-8"))["violations"] == [
        "provider_not_exactly_pinned"
    ]


def test_policy_rejects_volume_below_exact_policy(tmp_path: Path) -> None:
    contract = _desired_contract()
    contract["volume"]["size"] = 149
    desired = _write_json(tmp_path / "desired.json", contract)
    plan = _write_json(tmp_path / "plan.json", _plan(contract))
    adoption = _allowed_adoption(tmp_path / "adoption.json")
    output = tmp_path / "policy.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "policy",
            "--desired",
            str(desired),
            "--plan",
            str(plan),
            "--plan-binary",
            str(plan),
            "--adoption",
            str(adoption),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert (
        "volume_size_must_equal_150_gb"
        in json.loads(output.read_text(encoding="utf-8"))["violations"]
    )


@pytest.mark.parametrize(
    ("case", "violation"),
    [
        ("tagged_image", "template_image_must_use_sha256_digest"),
        ("public", "template_must_be_private"),
        ("serverless", "template_must_not_be_serverless"),
        ("tensorboard_port", "template_ports_must_equal_ssh_only"),
        ("credential_env", "template_env_must_be_empty"),
        ("local_volume", "template_local_volume_must_be_zero"),
        ("wrong_mount", "template_mount_must_equal_workspace"),
        ("conflicting_owner", "lifecycle_owner_must_equal_terraform_stable_v1"),
        ("unknown_field", "desired_contract_has_unknown_fields"),
    ],
)
def test_policy_rejects_unsafe_desired_contract(tmp_path: Path, case: str, violation: str) -> None:
    contract = _desired_contract()
    if case == "tagged_image":
        contract["template"]["imageName"] = "ghcr.io/lgbarn/mantis-v2-cuda:latest"
    elif case == "public":
        contract["template"]["isPublic"] = True
    elif case == "serverless":
        contract["template"]["isServerless"] = True
    elif case == "tensorboard_port":
        contract["template"]["ports"] = ["22/tcp", "6006/http"]
    elif case == "credential_env":
        contract["template"]["env"] = {"RUNPOD_API_KEY": "sentinel-secret"}
    elif case == "local_volume":
        contract["template"]["volumeInGb"] = 150
    elif case == "wrong_mount":
        contract["template"]["volumeMountPath"] = "/workspace"
    elif case == "conflicting_owner":
        contract["lifecycle_owner"] = "pod-lifecycle-adapter"
        contract["template"]["readme"] = "lifecycle-owner: pod-lifecycle-adapter"
    elif case == "unknown_field":
        contract["pod"] = {"gpuType": "NVIDIA A40"}
    desired = _write_json(tmp_path / "desired.json", contract)
    plan = _write_json(tmp_path / "plan.json", _plan(contract))
    adoption = _allowed_adoption(tmp_path / "adoption.json")
    output = tmp_path / "policy.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "policy",
            "--desired",
            str(desired),
            "--plan",
            str(plan),
            "--plan-binary",
            str(plan),
            "--adoption",
            str(adoption),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert violation in json.loads(output.read_text(encoding="utf-8"))["violations"]


@pytest.mark.parametrize(
    ("case", "violation"),
    [
        ("delete", "forbidden_actions:restapi_object.network_volume"),
        ("replace", "forbidden_actions:restapi_object.network_volume"),
        ("unknown_data", "policy_field_unknown:restapi_object.pod_template:data"),
        ("duplicate", "duplicate_resource:restapi_object.network_volume"),
        ("plaintext_secret", "secret_value_present_in_plan:RUNPOD_API_KEY"),
        ("wrong_provider", "provider_source_mismatch"),
        ("unknown_format", "unsupported_plan_format"),
    ],
)
def test_policy_rejects_unsafe_plan(tmp_path: Path, case: str, violation: str) -> None:
    contract = _desired_contract()
    plan_value = _plan(contract)
    if case == "delete":
        plan_value["resource_changes"][0]["change"]["actions"] = ["delete"]
    elif case == "replace":
        plan_value["resource_changes"][0]["change"]["actions"] = ["delete", "create"]
    elif case == "unknown_data":
        plan_value["resource_changes"][1]["change"]["after_unknown"] = {"data": True}
    elif case == "duplicate":
        plan_value["resource_changes"].append(copy.deepcopy(plan_value["resource_changes"][0]))
    elif case == "plaintext_secret":
        plan_value["configuration"]["debug_note"] = "sentinel-secret-value"
    elif case == "wrong_provider":
        plan_value["configuration"]["provider_config"]["restapi"]["full_name"] = (
            "registry.opentofu.org/example/restapi"
        )
    elif case == "unknown_format":
        plan_value["format_version"] = "2.0"
    desired = _write_json(tmp_path / "desired.json", contract)
    plan = _write_json(tmp_path / "plan.json", plan_value)
    adoption = _allowed_adoption(tmp_path / "adoption.json")
    output = tmp_path / "policy.json"
    environment = os.environ.copy()
    environment["RUNPOD_API_KEY"] = "sentinel-secret-value"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "policy",
            "--desired",
            str(desired),
            "--plan",
            str(plan),
            "--plan-binary",
            str(plan),
            "--adoption",
            str(adoption),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert violation in json.loads(output.read_text(encoding="utf-8"))["violations"]


def test_adoption_requires_import_for_exact_unmanaged_matches(tmp_path: Path) -> None:
    contract = _desired_contract()
    inventory = {
        "schema_version": 1,
        "volumes": [{"id": "volume-123", **contract["volume"]}],
        "templates": [{"id": "template-456", **contract["template"]}],
    }
    desired = _write_json(tmp_path / "desired.json", contract)
    inventory_path = _write_json(tmp_path / "inventory.json", inventory)
    state = _write_json(tmp_path / "state.json", {"schema_version": 1, "resources": {}})
    output = tmp_path / "adoption.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "adoption",
            "--desired",
            str(desired),
            "--inventory",
            str(inventory_path),
            "--state",
            str(state),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "plan_allowed": False,
        "resources": {
            "restapi_object.network_volume": {
                "id": "volume-123",
                "import_id": "/networkvolumes/volume-123",
                "status": "import_required",
            },
            "restapi_object.pod_template": {
                "id": "template-456",
                "import_id": "/templates/template-456",
                "status": "import_required",
            },
        },
        "schema_version": 1,
    }


def test_apply_authorization_binds_exact_human_plan_identity(tmp_path: Path) -> None:
    plan_binary = tmp_path / "saved.tfplan"
    plan_binary.write_bytes(b"opaque saved OpenTofu plan")
    plan_sha256 = hashlib.sha256(plan_binary.read_bytes()).hexdigest()
    desired_digest = "b" * 64
    policy = _write_json(
        tmp_path / "policy.json",
        {
            "accepted": True,
            "actions": {
                "restapi_object.network_volume": ["create"],
                "restapi_object.pod_template": ["create"],
            },
            "desired_digest": desired_digest,
            "lifecycle_owner": "terraform-stable-v1",
            "plan_sha256": plan_sha256,
            "schema_version": 1,
            "violations": [],
        },
    )
    authorization_value = {
        "authorization_type": "human",
        "authorized_at": "2026-07-21T20:00:00Z",
        "authorized_by": "operator@example.invalid",
        "desired_digest": desired_digest,
        "expires_at": "2026-07-21T20:30:00Z",
        "plan_sha256": plan_sha256,
        "resource_addresses": [
            "restapi_object.network_volume",
            "restapi_object.pod_template",
        ],
        "schema_version": 1,
    }
    authorization = _write_json(tmp_path / "authorization.json", authorization_value)
    output = tmp_path / "apply-decision.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "apply-authorization",
            "--policy",
            str(policy),
            "--plan-binary",
            str(plan_binary),
            "--authorization",
            str(authorization),
            "--evaluated-at",
            "2026-07-21T20:05:00Z",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "authorization_digest": hashlib.sha256(
            (json.dumps(authorization_value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "authorized": True,
        "authorized_by": "operator@example.invalid",
        "desired_digest": desired_digest,
        "plan_sha256": plan_sha256,
        "schema_version": 1,
        "violations": [],
    }


def test_apply_authorization_rejects_tampered_saved_plan(tmp_path: Path) -> None:
    original_plan_sha256 = hashlib.sha256(b"original plan").hexdigest()
    plan_binary = tmp_path / "saved.tfplan"
    plan_binary.write_bytes(b"tampered plan")
    desired_digest = "b" * 64
    policy = _write_json(
        tmp_path / "policy.json",
        {
            "accepted": True,
            "actions": {
                "restapi_object.network_volume": ["create"],
                "restapi_object.pod_template": ["create"],
            },
            "desired_digest": desired_digest,
            "lifecycle_owner": "terraform-stable-v1",
            "plan_sha256": original_plan_sha256,
            "schema_version": 1,
            "violations": [],
        },
    )
    authorization = _write_json(
        tmp_path / "authorization.json",
        {
            "authorization_type": "human",
            "authorized_at": "2026-07-21T20:00:00Z",
            "authorized_by": "operator@example.invalid",
            "desired_digest": desired_digest,
            "expires_at": "2026-07-21T20:30:00Z",
            "plan_sha256": original_plan_sha256,
            "resource_addresses": sorted(
                ["restapi_object.network_volume", "restapi_object.pod_template"]
            ),
            "schema_version": 1,
        },
    )
    output = tmp_path / "apply-decision.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "apply-authorization",
            "--policy",
            str(policy),
            "--plan-binary",
            str(plan_binary),
            "--authorization",
            str(authorization),
            "--evaluated-at",
            "2026-07-21T20:05:00Z",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert (
        "saved_plan_sha256_mismatch" in json.loads(output.read_text(encoding="utf-8"))["violations"]
    )


def test_terraform_module_is_pinned_and_owns_only_stable_resources() -> None:
    versions = (TERRAFORM / "versions.tf").read_text(encoding="utf-8")
    providers = (TERRAFORM / "providers.tf").read_text(encoding="utf-8")
    main = (TERRAFORM / "main.tf").read_text(encoding="utf-8")
    lock = (TERRAFORM / ".terraform.lock.hcl").read_text(encoding="utf-8")
    desired = json.loads((TERRAFORM / "desired-resources.example.json").read_text(encoding="utf-8"))

    assert 'required_version = "= 1.11.4"' in versions
    assert 'source  = "Mastercard/restapi"' in versions
    assert 'version = "= 3.0.0"' in versions
    assert 'provider "registry.opentofu.org/mastercard/restapi"' in lock
    assert 'version     = "3.0.0"' in lock
    assert '"h1:' in lock
    assert lock.count('"zh:') >= 15
    assert "ephemeral   = true" in providers
    assert "sensitive   = true" in providers
    assert re.search(r"bearer_token\s*=\s*var\.runpod_api_key", providers)
    assert main.count('resource "restapi_object"') == 2
    assert re.search(r'path\s*=\s*"/networkvolumes"', main)
    assert re.search(r'path\s*=\s*"/templates"', main)
    assert main.count("prevent_destroy = true") == 2
    assert "/pods" not in (versions + providers + main)
    assert "runpod_pod" not in (versions + providers + main)
    assert desired == _desired_contract()


def test_terraform_recipes_are_read_only_until_exact_human_apply() -> None:
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    module_readme = (TERRAFORM / "README.md").read_text(encoding="utf-8")
    plan_recipe = justfile.split("runpod-terraform-plan", maxsplit=1)[1].split(
        "runpod-terraform-import-human", maxsplit=1
    )[0]
    apply_recipe = justfile.split("runpod-terraform-apply-human", maxsplit=1)[1].split(
        "\ntrain config:", maxsplit=1
    )[0]

    assert plan_recipe.index("stable_resources.py adoption") < plan_recipe.index("tofu -chdir")
    assert plan_recipe.index("tofu -chdir") < plan_recipe.index("stable_resources.py policy")
    assert "tofu apply" not in plan_recipe
    assert "apply-authorization" in apply_recipe
    assert "--plan-binary" in apply_recipe
    assert "tofu -chdir=infra/runpod/terraform apply" in apply_recipe
    assert "auto-approve" not in apply_recipe
    assert "signature validation" in module_readme
    assert "not claimed to be signature-verified" in module_readme


@pytest.mark.parametrize(
    ("case", "expected_status", "plan_allowed"),
    [
        ("absent", "absent", True),
        ("adopted", "adopted", True),
        ("drift", "identity_conflict", False),
        ("duplicate", "ambiguous_duplicates", False),
        ("wrong_state_id", "state_identity_conflict", False),
    ],
)
def test_adoption_classifies_each_existing_identity(
    tmp_path: Path, case: str, expected_status: str, plan_allowed: bool
) -> None:
    contract = _desired_contract()
    volumes: list[dict[str, Any]] = []
    state_resources: dict[str, str] = {}
    if case != "absent":
        volumes = [{"id": "volume-123", **contract["volume"]}]
    if case == "adopted":
        state_resources["restapi_object.network_volume"] = "volume-123"
    elif case == "drift":
        volumes[0]["size"] = 151
    elif case == "duplicate":
        volumes.append({"id": "volume-789", **contract["volume"]})
    elif case == "wrong_state_id":
        state_resources["restapi_object.network_volume"] = "volume-other"
    inventory = {"schema_version": 1, "volumes": volumes, "templates": []}
    desired = _write_json(tmp_path / "desired.json", contract)
    inventory_path = _write_json(tmp_path / "inventory.json", inventory)
    state = _write_json(
        tmp_path / "state.json", {"schema_version": 1, "resources": state_resources}
    )
    output = tmp_path / "adoption.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "adoption",
            "--desired",
            str(desired),
            "--inventory",
            str(inventory_path),
            "--state",
            str(state),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == (0 if plan_allowed else 2)
    assert report["plan_allowed"] is plan_allowed
    assert report["resources"]["restapi_object.network_volume"]["status"] == expected_status


def test_adoption_rejects_an_unsafe_desired_contract(tmp_path: Path) -> None:
    contract = _desired_contract()
    contract["template"]["isPublic"] = True
    desired = _write_json(tmp_path / "desired.json", contract)
    inventory = _write_json(
        tmp_path / "inventory.json", {"schema_version": 1, "volumes": [], "templates": []}
    )
    state = _write_json(tmp_path / "state.json", {"schema_version": 1, "resources": {}})
    output = tmp_path / "adoption.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "adoption",
            "--desired",
            str(desired),
            "--inventory",
            str(inventory),
            "--state",
            str(state),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(output.read_text(encoding="utf-8"))["violations"] == [
        "template_must_be_private"
    ]


def test_terraform_runtime_control_plane_files_are_ignored() -> None:
    candidates = [
        "infra/runpod/terraform/.terraform/provider-cache",
        "infra/runpod/terraform/terraform.tfstate",
        "infra/runpod/terraform/terraform.tfstate.backup",
        "infra/runpod/terraform/operator.tfplan",
        "infra/runpod/terraform/operator.tfplan.json",
        "infra/runpod/terraform/operator.auto.tfvars",
        "infra/runpod/terraform/operator.auto.tfvars.json",
        "infra/runpod/terraform/terraform.tfvars",
        "infra/runpod/terraform/operator.backend.hcl",
        "infra/runpod/terraform/desired-resources.json",
        "infra/runpod/terraform/inventory.json",
        "infra/runpod/terraform/state-addresses.json",
        "infra/runpod/terraform/adoption.json",
        "infra/runpod/terraform/policy.json",
        "infra/runpod/terraform/apply-authorization.json",
        "infra/runpod/terraform/apply-decision.json",
    ]
    completed = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        input="\n".join(candidates) + "\n",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == candidates
