"""Pinned official RunPod REST v1 boundary for Pod lifecycle control."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from http.client import HTTPMessage
from typing import IO, Protocol
from urllib.parse import quote

OPENAPI_IDENTITY = "https://rest.runpod.io/v1/openapi.json"
OPENAPI_VERSION = "v1"
OPENAPI_SHA256 = "f4be55173a5392150d805d103b1ee3aeff23defec40052dd3188d606ddedddfc"
REST_V1_BASE_URL = "https://rest.runpod.io/v1"


class RunpodAdapterError(RuntimeError):
    """Raised for a provider exchange that cannot be safely normalized."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes


class HttpTransport(Protocol):
    def request(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> HttpResponse: ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects so the provisioning bearer token never changes origin."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> None:
        return None


class UrllibTransport:
    """Small stdlib transport; adapter tests inject a no-network implementation."""

    def request(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> HttpResponse:
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            opener = urllib.request.build_opener(_NoRedirectHandler())
            with opener.open(request, timeout=30) as response:
                return HttpResponse(status=response.status, body=response.read())
        except urllib.error.HTTPError as exc:
            return HttpResponse(status=exc.code, body=exc.read())
        except (OSError, TimeoutError) as exc:
            raise RunpodAdapterError("provider_transport_failed") from exc


def _pod_id(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise RunpodAdapterError("invalid_pod_identity")
    return value


def _json(response: HttpResponse, expected_status: int) -> object:
    if response.status != expected_status:
        raise RunpodAdapterError(f"provider_http_status:{response.status}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        return json.loads(
            response.body,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RunpodAdapterError("provider_invalid_json") from exc


def _required(raw: Mapping[str, object], field: str, expected: type[object]) -> object:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise RunpodAdapterError(f"provider_invalid_field:{field}")
    return value


def _normalize_pod(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RunpodAdapterError("provider_invalid_pod")
    raw = value
    network_volume = raw.get("networkVolume")
    if not isinstance(network_volume, Mapping):
        raise RunpodAdapterError("provider_invalid_field:networkVolume")
    cost = raw.get("costPerHr")
    try:
        parsed_cost = Decimal(str(cost))
    except (InvalidOperation, ValueError) as exc:
        raise RunpodAdapterError("provider_invalid_field:costPerHr") from exc
    if not parsed_cost.is_finite() or parsed_cost < 0:
        raise RunpodAdapterError("provider_invalid_field:costPerHr")
    normalized: dict[str, object] = {
        "id": _pod_id(raw.get("id")),
        "name": _required(raw, "name", str),
        "desiredStatus": _required(raw, "desiredStatus", str),
        "imageName": _required(raw, "image", str),
        "templateId": _required(raw, "templateId", str),
        "networkVolumeId": _required(network_volume, "id", str),
        "costPerHr": str(parsed_cost),
        "vcpuCount": _required(raw, "vcpuCount", int),
        "memoryInGb": _required(raw, "memoryInGb", int),
    }
    if "uptimeSeconds" in raw:
        normalized["uptimeSeconds"] = _required(raw, "uptimeSeconds", int)
    for field in ("lastStartedAt", "lastStatusChange"):
        if field in raw and raw[field] is not None:
            normalized[field] = _required(raw, field, str)
    return normalized


def _normalize_inventory_pod(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RunpodAdapterError("provider_invalid_pod")
    normalized: dict[str, object] = {
        "id": _pod_id(value.get("id")),
        "name": _required(value, "name", str),
        "imageName": _required(value, "image", str),
    }
    for field in ("desiredStatus", "templateId", "costPerHr", "vcpuCount", "memoryInGb"):
        if field in value:
            normalized[field] = value[field]
    network_volume = value.get("networkVolume")
    if isinstance(network_volume, Mapping) and isinstance(network_volume.get("id"), str):
        normalized["networkVolumeId"] = network_volume["id"]
    return normalized


class RunpodRestV1Adapter:
    """Exact REST v1 adapter pinned to one reviewed OpenAPI document."""

    def __init__(self, *, api_key: str, transport: HttpTransport | None = None) -> None:
        if not api_key:
            raise RunpodAdapterError("runpod_api_key_required")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._transport = transport or UrllibTransport()

    def _request(
        self, method: str, path: str, *, body: Mapping[str, object] | None = None
    ) -> HttpResponse:
        headers = dict(self._headers)
        encoded: bytes | None = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        return self._transport.request(method, REST_V1_BASE_URL + path, headers, encoded)

    def inventory(self) -> Sequence[Mapping[str, object]]:
        raw = _json(self._request("GET", "/pods?includeNetworkVolume=true"), 200)
        if not isinstance(raw, list):
            raise RunpodAdapterError("provider_invalid_inventory")
        return tuple(_normalize_inventory_pod(item) for item in raw)

    def create(self, decision: Mapping[str, object], deadline: datetime) -> Mapping[str, object]:
        del deadline  # The durable local watchdog owns the approved hard deadline.
        for key, expected in (
            ("openapi_identity", OPENAPI_IDENTITY),
            ("openapi_version", OPENAPI_VERSION),
            ("openapi_sha256", OPENAPI_SHA256),
        ):
            if decision.get(key) != expected:
                raise RunpodAdapterError(f"adapter_provenance_mismatch:{key}")
        ports = decision.get("ports")
        if (
            not isinstance(ports, list)
            or not ports
            or any(not isinstance(item, str) for item in ports)
        ):
            raise RunpodAdapterError("invalid_launch_decision:ports")
        request = {
            "cloudType": "SECURE",
            "computeType": "GPU",
            "containerDiskInGb": decision.get("container_disk_gb"),
            "containerRegistryAuthId": decision.get("registry_auth_id"),
            "dataCenterIds": [decision.get("datacenter_id")],
            "gpuCount": decision.get("gpu_count"),
            "gpuTypeIds": [decision.get("gpu_type")],
            "imageName": decision.get("image_ref"),
            "minRAMPerGPU": decision.get("ram_gb"),
            "minVCPUPerGPU": decision.get("vcpu"),
            "name": decision.get("run_name"),
            "networkVolumeId": decision.get("volume_id"),
            "ports": ports,
            "templateId": decision.get("template_id"),
            "volumeMountPath": decision.get("volume_mount_path"),
        }
        return _normalize_pod(_json(self._request("POST", "/pods", body=request), 201))

    def status(self, pod_id: str) -> Mapping[str, object]:
        exact_id = _pod_id(pod_id)
        return _normalize_pod(_json(self._request("GET", f"/pods/{quote(exact_id)}"), 200))

    def terminate(self, pod_id: str) -> Mapping[str, object]:
        exact_id = _pod_id(pod_id)
        response = self._request("DELETE", f"/pods/{quote(exact_id)}")
        if response.status != 204 or response.body not in {b"", b"\n"}:
            raise RunpodAdapterError("provider_invalid_delete_response")
        return {"deleted": True, "id": exact_id}

    def billing(self, pod_id: str) -> Mapping[str, object] | None:
        exact_id = _pod_id(pod_id)
        path = f"/billing/pods?grouping=podId&podId={quote(exact_id)}"
        raw = _json(self._request("GET", path), 200)
        if not isinstance(raw, list):
            raise RunpodAdapterError("provider_invalid_billing")
        matches = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise RunpodAdapterError("provider_invalid_billing")
            if item.get("podId") == exact_id:
                try:
                    matches.append(Decimal(str(item.get("amount"))))
                except (InvalidOperation, ValueError) as exc:
                    raise RunpodAdapterError("provider_invalid_field:amount") from exc
        if not matches:
            return None
        total = sum(matches, start=Decimal("0"))
        if not total.is_finite() or total < 0:
            raise RunpodAdapterError("provider_invalid_field:amount")
        return {"pod_id": exact_id, "actual_cost_usd": str(total)}
