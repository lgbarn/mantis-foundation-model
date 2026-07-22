from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from mantis_v2.runpod_rest_adapter import (
    OPENAPI_IDENTITY,
    OPENAPI_SHA256,
    OPENAPI_VERSION,
    HttpResponse,
    RunpodAdapterError,
    RunpodRestV1Adapter,
    _NoRedirectHandler,
)


def _decision() -> dict[str, object]:
    return {
        "run_name": "mantisv2-cuda-qualification-seed42",
        "gpu_type": "NVIDIA A40",
        "gpu_count": 1,
        "vcpu": 8,
        "ram_gb": 32,
        "datacenter_id": "US-CA-2",
        "container_disk_gb": 50,
        "image_ref": "ghcr.io/lgbarn/mantis@sha256:" + "a" * 64,
        "template_id": "template-fixture",
        "registry_auth_id": "registry-auth-fixture",
        "volume_id": "volume-fixture",
        "volume_mount_path": "/workspace",
        "ports": ["22/tcp"],
        "openapi_identity": OPENAPI_IDENTITY,
        "openapi_version": OPENAPI_VERSION,
        "openapi_sha256": OPENAPI_SHA256,
    }


class RecordingTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, str], bytes | None]] = []

    def request(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> HttpResponse:
        self.calls.append((method, url, headers, body))
        return self.responses.pop(0)


def _pod_response() -> dict[str, object]:
    return {
        "id": "pod-001",
        "name": "mantisv2-cuda-qualification-seed42",
        "desiredStatus": "RUNNING",
        "image": "ghcr.io/lgbarn/mantis@sha256:" + "a" * 64,
        "templateId": "template-fixture",
        "networkVolume": {"id": "volume-fixture"},
        "costPerHr": "0.44",
        "vcpuCount": 8,
        "memoryInGb": 32,
        "env": {"RUNPOD_API_KEY": "provider-secret"},
    }


def test_create_uses_exact_pinned_rest_v1_exchange_and_normalizes_response() -> None:
    transport = RecordingTransport(
        [HttpResponse(status=201, body=json.dumps(_pod_response()).encode())]
    )
    adapter = RunpodRestV1Adapter(api_key="secret-sentinel", transport=transport)

    created = adapter.create(_decision(), datetime(2026, 7, 21, 14, tzinfo=UTC))

    method, url, headers, body = transport.calls[0]
    assert method == "POST"
    assert url == "https://rest.runpod.io/v1/pods"
    assert headers == {
        "Authorization": "Bearer secret-sentinel",
        "Content-Type": "application/json",
    }
    assert json.loads(body or b"") == {
        "cloudType": "SECURE",
        "computeType": "GPU",
        "containerDiskInGb": 50,
        "containerRegistryAuthId": "registry-auth-fixture",
        "dataCenterIds": ["US-CA-2"],
        "gpuCount": 1,
        "gpuTypeIds": ["NVIDIA A40"],
        "imageName": "ghcr.io/lgbarn/mantis@sha256:" + "a" * 64,
        "minRAMPerGPU": 32,
        "minVCPUPerGPU": 8,
        "name": "mantisv2-cuda-qualification-seed42",
        "networkVolumeId": "volume-fixture",
        "ports": ["22/tcp"],
        "templateId": "template-fixture",
        "volumeMountPath": "/workspace",
    }
    assert created == {
        "id": "pod-001",
        "name": "mantisv2-cuda-qualification-seed42",
        "desiredStatus": "RUNNING",
        "imageName": "ghcr.io/lgbarn/mantis@sha256:" + "a" * 64,
        "templateId": "template-fixture",
        "networkVolumeId": "volume-fixture",
        "costPerHr": "0.44",
        "vcpuCount": 8,
        "memoryInGb": 32,
    }
    assert "provider-secret" not in json.dumps(created)


def test_inventory_status_delete_and_billing_use_exact_resource_identities() -> None:
    pod = _pod_response()
    transport = RecordingTransport(
        [
            HttpResponse(status=200, body=json.dumps([pod]).encode()),
            HttpResponse(status=200, body=json.dumps(pod).encode()),
            HttpResponse(status=204, body=b""),
            HttpResponse(
                status=200,
                body=json.dumps(
                    [{"podId": "pod-001", "amount": 0.2}, {"podId": "other", "amount": 9}]
                ).encode(),
            ),
        ]
    )
    adapter = RunpodRestV1Adapter(api_key="secret-sentinel", transport=transport)

    assert adapter.inventory()[0]["networkVolumeId"] == "volume-fixture"
    assert adapter.status("pod-001")["id"] == "pod-001"
    assert adapter.terminate("pod-001") == {"deleted": True, "id": "pod-001"}
    assert adapter.billing("pod-001") == {"pod_id": "pod-001", "actual_cost_usd": "0.2"}

    assert [(method, url) for method, url, _, _ in transport.calls] == [
        ("GET", "https://rest.runpod.io/v1/pods?includeNetworkVolume=true"),
        ("GET", "https://rest.runpod.io/v1/pods/pod-001"),
        ("DELETE", "https://rest.runpod.io/v1/pods/pod-001"),
        ("GET", "https://rest.runpod.io/v1/billing/pods?grouping=podId&podId=pod-001"),
    ]


@pytest.mark.parametrize(
    "response",
    (
        HttpResponse(status=200, body=b"not-json"),
        HttpResponse(status=200, body=b"{}"),
        HttpResponse(status=503, body=b'{"error":"RUNPOD_API_KEY=secret-sentinel"}'),
    ),
)
def test_adapter_rejects_malformed_incomplete_or_failed_responses_without_secret_leak(
    response: HttpResponse,
) -> None:
    adapter = RunpodRestV1Adapter(
        api_key="secret-sentinel", transport=RecordingTransport([response])
    )

    with pytest.raises(RunpodAdapterError) as failure:
        adapter.status("pod-001")

    assert "secret-sentinel" not in str(failure.value)


def test_billing_lag_is_pending_not_zero_spend() -> None:
    transport = RecordingTransport([HttpResponse(status=200, body=b"[]")])
    adapter = RunpodRestV1Adapter(api_key="secret", transport=transport)

    assert adapter.billing("pod-001") is None


def test_duplicate_json_keys_are_rejected() -> None:
    response = HttpResponse(status=200, body=b'{"id":"pod-001","id":"pod-002"}')
    adapter = RunpodRestV1Adapter(api_key="secret", transport=RecordingTransport([response]))

    with pytest.raises(RunpodAdapterError, match="^provider_invalid_json$"):
        adapter.status("pod-001")


def test_http_transport_refuses_all_redirects_to_protect_bearer_token() -> None:
    handler = _NoRedirectHandler()

    assert handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example") is None
