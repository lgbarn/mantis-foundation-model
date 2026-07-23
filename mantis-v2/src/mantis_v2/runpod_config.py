"""Strict, immutable inputs for RunPod launch planning."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, cast

from mantis_v2.runpod_rest_adapter import (
    OPENAPI_IDENTITY,
    OPENAPI_SHA256,
    OPENAPI_VERSION,
    REST_V1_BASE_URL,
)


class RunpodConfigError(ValueError):
    """Raised when a launch-planning input is invalid."""


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for a dataclass or JSON-compatible value."""
    data = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    return json.dumps(
        _canonical_value(data), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _exact(raw: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RunpodConfigError(f"{context} must be an object")
    unknown = set(raw) - expected
    if unknown:
        raise RunpodConfigError(f"unknown {context} keys: {', '.join(sorted(unknown))}")
    missing = expected - set(raw)
    if missing:
        raise RunpodConfigError(f"missing {context} keys: {', '.join(sorted(missing))}")
    return raw


def _integer(value: Any, field: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RunpodConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunpodConfigError(f"{field} must be a non-empty string")
    return value


def _optional_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise RunpodConfigError(f"{field} must be a string")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise RunpodConfigError(f"{field} must be true or false")
    return value


def _choice(value: Any, field: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise RunpodConfigError(f"{field} must be one of: {', '.join(sorted(choices))}")
    return value


def _decimal(value: Any, field: str, *, minimum: Decimal = Decimal("0")) -> Decimal:
    if not isinstance(value, str):
        raise RunpodConfigError(f"{field} must be a decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise RunpodConfigError(f"{field} must be a decimal string") from exc
    if not result.is_finite() or result < minimum:
        raise RunpodConfigError(f"{field} must be finite and >= {minimum}")
    return result


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise RunpodConfigError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunpodConfigError(f"{field} is not valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise RunpodConfigError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RunpodConfigError(f"{field} must be a non-empty array")
    return tuple(_text(item, field) for item in value)


def _read_toml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        return tomllib.loads(source.read_text())
    except FileNotFoundError as exc:
        raise RunpodConfigError(f"input not found: {source}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RunpodConfigError(f"invalid TOML in {source}: {exc}") from exc


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text())
    except FileNotFoundError as exc:
        raise RunpodConfigError(f"input not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise RunpodConfigError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RunpodConfigError(f"JSON input must be an object: {source}")
    return raw


@dataclass(frozen=True)
class ProviderConfig:
    secure_cloud: bool
    allowed_gpu_types: tuple[str, ...]
    allowed_datacenters: tuple[str, ...]
    minimum_vcpu: int
    minimum_ram_gb: int
    container_disk_gb: int


@dataclass(frozen=True)
class AdapterConfig:
    base_url: str
    openapi_identity: str
    openapi_version: str
    openapi_sha256: str


@dataclass(frozen=True)
class StorageConfig:
    volume_gb: int
    high_water_bytes: int
    minimum_free_bytes: int


@dataclass(frozen=True)
class LifecycleConfig:
    maximum_inventory_age_seconds: int
    maximum_duration_seconds: int
    startup_allowance_seconds: int


@dataclass(frozen=True)
class BillingConfig:
    container_disk_usd_per_gb_month: Decimal
    billing_month_hours: int


@dataclass(frozen=True)
class BudgetConfig:
    account_ceiling_usd: Decimal
    storage_usd: Decimal
    qualification_usd: Decimal
    production_usd: Decimal
    protected_recovery_usd: Decimal
    ordinary_launch_cutoff_usd: Decimal


@dataclass(frozen=True)
class PlatformConfig:
    schema_version: int
    provider: ProviderConfig
    adapter: AdapterConfig
    storage: StorageConfig
    lifecycle: LifecycleConfig
    billing: BillingConfig
    budget: BudgetConfig

    @property
    def digest(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True)
class LocalPaths:
    workspace_root: Path
    state_root: Path
    output_root: Path


@dataclass(frozen=True)
class LocalController:
    hostname: str


@dataclass(frozen=True)
class LocalSecrets:
    runpod_api_key_env: str
    s3_access_key_id_env: str
    s3_secret_access_key_env: str


@dataclass(frozen=True)
class LocalConfig:
    schema_version: int
    controller: LocalController
    paths: LocalPaths
    secrets: LocalSecrets

    @property
    def digest(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    model_family: str
    stage: Literal["qualification", "production", "recovery"]
    seed: int
    definition_sha256: str
    sealed_holdout: bool


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment: ExperimentDefinition

    @property
    def digest(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True)
class LaunchIntent:
    schema_version: int
    intent_id: str
    stage: Literal["qualification", "production", "recovery"]
    run_name: str
    gpu_type: str
    datacenter_id: str
    gpu_count: int
    vcpu: int
    ram_gb: int
    container_disk_gb: int
    image_ref: str
    template_id: str
    registry_auth_id: str
    volume_id: str
    volume_size_gb: int
    volume_mount_path: str
    ports: tuple[str, ...]
    maximum_duration_seconds: int

    @property
    def digest(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True)
class InventoryOffer:
    gpu_type: str
    datacenter_id: str
    price_usd_per_gpu_hour: Decimal
    available: bool
    cloud_type: Literal["secure", "community"]


@dataclass(frozen=True)
class InventoryVolume:
    volume_id: str
    datacenter_id: str
    size_gb: int
    free_bytes: int


@dataclass(frozen=True)
class InventorySnapshot:
    schema_version: int
    observed_at: datetime
    account_balance_usd: Decimal
    offers: tuple[InventoryOffer, ...]
    volumes: tuple[InventoryVolume, ...]
    live_pods: tuple[str, ...]

    @property
    def digest(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True)
class SpendLedger:
    schema_version: int
    actual_spend_usd: Decimal
    reserved_spend_usd: Decimal
    bucket_actual_spend_usd: dict[str, Decimal]
    bucket_reserved_spend_usd: dict[str, Decimal]
    active_reservations: tuple[str, ...]
    consumed_authorization_digests: tuple[str, ...]

    @property
    def digest(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True)
class LaunchAuthorization:
    schema_version: int
    authorization_id: str
    subject_digest: str
    authorized_at: datetime
    expires_at: datetime
    maximum_projected_spend_usd: Decimal
    approver: str
    autopay_disabled: bool
    ordinary_launch_cutoff_usd: Decimal
    campaign_ceiling_usd: Decimal
    recovery_authorized: bool

    @property
    def digest(self) -> str:
        return canonical_digest(self)


def _versioned(raw: dict[str, Any], expected: set[str], context: str) -> dict[str, Any]:
    checked = _exact(raw, expected | {"schema_version"}, context)
    if checked["schema_version"] != 1:
        raise RunpodConfigError(f"{context}.schema_version must be 1")
    return checked


def load_platform_config(path: str | Path) -> PlatformConfig:
    raw = _versioned(
        _read_toml(path),
        {"provider", "adapter", "storage", "lifecycle", "billing", "budget"},
        "platform",
    )
    provider = _exact(
        raw["provider"],
        {
            "secure_cloud",
            "allowed_gpu_types",
            "allowed_datacenters",
            "minimum_vcpu",
            "minimum_ram_gb",
            "container_disk_gb",
        },
        "[provider]",
    )
    adapter = _exact(
        raw["adapter"],
        {"base_url", "openapi_identity", "openapi_version", "openapi_sha256"},
        "[adapter]",
    )
    storage = _exact(
        raw["storage"], {"volume_gb", "high_water_bytes", "minimum_free_bytes"}, "[storage]"
    )
    lifecycle = _exact(
        raw["lifecycle"],
        {
            "maximum_inventory_age_seconds",
            "maximum_duration_seconds",
            "startup_allowance_seconds",
        },
        "[lifecycle]",
    )
    billing = _exact(
        raw["billing"],
        {"container_disk_usd_per_gb_month", "billing_month_hours"},
        "[billing]",
    )
    budget = _exact(
        raw["budget"],
        {
            "account_ceiling_usd",
            "storage_usd",
            "qualification_usd",
            "production_usd",
            "protected_recovery_usd",
            "ordinary_launch_cutoff_usd",
        },
        "[budget]",
    )
    config = PlatformConfig(
        schema_version=1,
        provider=ProviderConfig(
            secure_cloud=_boolean(provider["secure_cloud"], "provider.secure_cloud"),
            allowed_gpu_types=_strings(provider["allowed_gpu_types"], "provider.allowed_gpu_types"),
            allowed_datacenters=_strings(
                provider["allowed_datacenters"], "provider.allowed_datacenters"
            ),
            minimum_vcpu=_integer(provider["minimum_vcpu"], "provider.minimum_vcpu"),
            minimum_ram_gb=_integer(provider["minimum_ram_gb"], "provider.minimum_ram_gb"),
            container_disk_gb=_integer(provider["container_disk_gb"], "provider.container_disk_gb"),
        ),
        adapter=AdapterConfig(
            base_url=_text(adapter["base_url"], "adapter.base_url"),
            openapi_identity=_text(adapter["openapi_identity"], "adapter.openapi_identity"),
            openapi_version=_text(adapter["openapi_version"], "adapter.openapi_version"),
            openapi_sha256=_text(adapter["openapi_sha256"], "adapter.openapi_sha256"),
        ),
        storage=StorageConfig(
            volume_gb=_integer(storage["volume_gb"], "storage.volume_gb"),
            high_water_bytes=_integer(storage["high_water_bytes"], "storage.high_water_bytes"),
            minimum_free_bytes=_integer(
                storage["minimum_free_bytes"], "storage.minimum_free_bytes"
            ),
        ),
        lifecycle=LifecycleConfig(
            maximum_inventory_age_seconds=_integer(
                lifecycle["maximum_inventory_age_seconds"],
                "lifecycle.maximum_inventory_age_seconds",
            ),
            maximum_duration_seconds=_integer(
                lifecycle["maximum_duration_seconds"], "lifecycle.maximum_duration_seconds"
            ),
            startup_allowance_seconds=_integer(
                lifecycle["startup_allowance_seconds"], "lifecycle.startup_allowance_seconds"
            ),
        ),
        billing=BillingConfig(
            container_disk_usd_per_gb_month=_decimal(
                billing["container_disk_usd_per_gb_month"],
                "billing.container_disk_usd_per_gb_month",
            ),
            billing_month_hours=_integer(
                billing["billing_month_hours"], "billing.billing_month_hours"
            ),
        ),
        budget=BudgetConfig(
            **{
                key: _decimal(budget[key], f"budget.{key}")
                for key in (
                    "account_ceiling_usd",
                    "storage_usd",
                    "qualification_usd",
                    "production_usd",
                    "protected_recovery_usd",
                    "ordinary_launch_cutoff_usd",
                )
            }
        ),
    )
    if not config.provider.secure_cloud:
        raise RunpodConfigError("provider.secure_cloud must be true")
    if not 600 <= config.lifecycle.startup_allowance_seconds <= 1800:
        raise RunpodConfigError("lifecycle.startup_allowance_seconds must be between 600 and 1800")
    expected_adapter = {
        "base_url": REST_V1_BASE_URL,
        "openapi_identity": OPENAPI_IDENTITY,
        "openapi_version": OPENAPI_VERSION,
        "openapi_sha256": OPENAPI_SHA256,
    }
    for field, expected in expected_adapter.items():
        if getattr(config.adapter, field) != expected:
            raise RunpodConfigError(f"adapter.{field} must be {expected}")
    volume_bytes = config.storage.volume_gb * 1_000_000_000
    if config.storage.high_water_bytes + config.storage.minimum_free_bytes > volume_bytes:
        raise RunpodConfigError(
            "storage high-water policy must leave minimum_free_bytes within volume_gb"
        )
    expected_ceiling = (
        config.budget.storage_usd
        + config.budget.qualification_usd
        + config.budget.production_usd
        + config.budget.protected_recovery_usd
    )
    if expected_ceiling != config.budget.account_ceiling_usd:
        raise RunpodConfigError("budget buckets must sum to account_ceiling_usd")
    if (
        config.budget.ordinary_launch_cutoff_usd
        != config.budget.account_ceiling_usd - config.budget.protected_recovery_usd
    ):
        raise RunpodConfigError("ordinary launch cutoff must protect the recovery reserve")
    return config


def load_local_config(path: str | Path) -> LocalConfig:
    raw = _versioned(_read_toml(path), {"controller", "paths", "secrets"}, "local")
    controller = _exact(raw["controller"], {"hostname"}, "[controller]")
    paths = _exact(raw["paths"], {"workspace_root", "state_root", "output_root"}, "[paths]")
    secrets = _exact(
        raw["secrets"],
        {"runpod_api_key_env", "s3_access_key_id_env", "s3_secret_access_key_env"},
        "[secrets]",
    )
    config = LocalConfig(
        schema_version=1,
        controller=LocalController(hostname=_text(controller["hostname"], "controller.hostname")),
        paths=LocalPaths(**{key: Path(_text(paths[key], f"paths.{key}")) for key in paths}),
        secrets=LocalSecrets(**{key: _text(secrets[key], f"secrets.{key}") for key in secrets}),
    )
    expected_secret_names = {
        "runpod_api_key_env": "RUNPOD_API_KEY",
        "s3_access_key_id_env": "RUNPOD_S3_ACCESS_KEY_ID",
        "s3_secret_access_key_env": "RUNPOD_S3_SECRET_ACCESS_KEY",
    }
    for field, expected in expected_secret_names.items():
        if getattr(config.secrets, field) != expected:
            raise RunpodConfigError(f"secrets.{field} must name {expected}")
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = _versioned(_read_toml(path), {"experiment"}, "experiment config")
    experiment = _exact(
        raw["experiment"],
        {"name", "model_family", "stage", "seed", "definition_sha256", "sealed_holdout"},
        "[experiment]",
    )
    stage = _text(experiment["stage"], "experiment.stage")
    if stage not in {"qualification", "production", "recovery"}:
        raise RunpodConfigError("experiment.stage must be qualification, production, or recovery")
    digest = _text(experiment["definition_sha256"], "experiment.definition_sha256")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RunpodConfigError("experiment.definition_sha256 must be lowercase SHA-256")
    return ExperimentConfig(
        schema_version=1,
        experiment=ExperimentDefinition(
            name=_text(experiment["name"], "experiment.name"),
            model_family=_text(experiment["model_family"], "experiment.model_family"),
            stage=stage,  # type: ignore[arg-type]
            seed=_integer(experiment["seed"], "experiment.seed", minimum=0),
            definition_sha256=digest,
            sealed_holdout=_boolean(experiment["sealed_holdout"], "experiment.sealed_holdout"),
        ),
    )


def load_launch_intent(path: str | Path) -> LaunchIntent:
    raw = _versioned(
        _read_json(path),
        {
            "intent_id",
            "stage",
            "run_name",
            "gpu_type",
            "datacenter_id",
            "gpu_count",
            "vcpu",
            "ram_gb",
            "container_disk_gb",
            "image_ref",
            "template_id",
            "registry_auth_id",
            "volume_id",
            "volume_size_gb",
            "volume_mount_path",
            "ports",
            "maximum_duration_seconds",
        },
        "launch intent",
    )
    stage = _text(raw["stage"], "intent.stage")
    if stage not in {"qualification", "production", "recovery"}:
        raise RunpodConfigError("intent.stage must be qualification, production, or recovery")
    intent = LaunchIntent(
        schema_version=1,
        intent_id=_text(raw["intent_id"], "intent.intent_id"),
        stage=stage,  # type: ignore[arg-type]
        run_name=_text(raw["run_name"], "intent.run_name"),
        gpu_type=_text(raw["gpu_type"], "intent.gpu_type"),
        datacenter_id=_text(raw["datacenter_id"], "intent.datacenter_id"),
        gpu_count=_integer(raw["gpu_count"], "intent.gpu_count"),
        vcpu=_integer(raw["vcpu"], "intent.vcpu"),
        ram_gb=_integer(raw["ram_gb"], "intent.ram_gb"),
        container_disk_gb=_integer(raw["container_disk_gb"], "intent.container_disk_gb"),
        image_ref=_text(raw["image_ref"], "intent.image_ref"),
        template_id=_text(raw["template_id"], "intent.template_id"),
        registry_auth_id=_optional_text(raw["registry_auth_id"], "intent.registry_auth_id"),
        volume_id=_text(raw["volume_id"], "intent.volume_id"),
        volume_size_gb=_integer(raw["volume_size_gb"], "intent.volume_size_gb"),
        volume_mount_path=_text(raw["volume_mount_path"], "intent.volume_mount_path"),
        ports=_strings(raw["ports"], "intent.ports"),
        maximum_duration_seconds=_integer(
            raw["maximum_duration_seconds"], "intent.maximum_duration_seconds"
        ),
    )
    if not re.search(r"@sha256:[0-9a-f]{64}$", intent.image_ref):
        raise RunpodConfigError("intent.image_ref must use an immutable SHA-256 digest")
    if not intent.volume_mount_path.startswith("/") or ".." in intent.volume_mount_path.split("/"):
        raise RunpodConfigError("intent.volume_mount_path must be an absolute safe path")
    if intent.ports != ("22/tcp",):
        raise RunpodConfigError("intent.ports must be exactly [22/tcp]")
    return intent


def load_inventory_snapshot(path: str | Path) -> InventorySnapshot:
    raw = _versioned(
        _read_json(path),
        {"observed_at", "account_balance_usd", "offers", "volumes", "live_pods"},
        "inventory snapshot",
    )
    if not isinstance(raw["offers"], list):
        raise RunpodConfigError("inventory.offers must be an array")
    offers = []
    for index, item in enumerate(raw["offers"]):
        offer = _exact(
            item,
            {
                "gpu_type",
                "datacenter_id",
                "price_usd_per_gpu_hour",
                "available",
                "cloud_type",
            },
            f"inventory.offers[{index}]",
        )
        offers.append(
            InventoryOffer(
                gpu_type=_text(offer["gpu_type"], f"inventory.offers[{index}].gpu_type"),
                datacenter_id=_text(
                    offer["datacenter_id"], f"inventory.offers[{index}].datacenter_id"
                ),
                price_usd_per_gpu_hour=_decimal(
                    offer["price_usd_per_gpu_hour"],
                    f"inventory.offers[{index}].price_usd_per_gpu_hour",
                ),
                available=_boolean(offer["available"], f"inventory.offers[{index}].available"),
                cloud_type=cast(
                    Literal["secure", "community"],
                    _choice(
                        offer["cloud_type"],
                        f"inventory.offers[{index}].cloud_type",
                        {"secure", "community"},
                    ),
                ),
            )
        )
    if not isinstance(raw["volumes"], list):
        raise RunpodConfigError("inventory.volumes must be an array")
    volumes = []
    for index, item in enumerate(raw["volumes"]):
        volume = _exact(
            item,
            {"volume_id", "datacenter_id", "size_gb", "free_bytes"},
            f"inventory.volumes[{index}]",
        )
        parsed_volume = InventoryVolume(
            volume_id=_text(volume["volume_id"], f"inventory.volumes[{index}].volume_id"),
            datacenter_id=_text(
                volume["datacenter_id"],
                f"inventory.volumes[{index}].datacenter_id",
            ),
            size_gb=_integer(volume["size_gb"], f"inventory.volumes[{index}].size_gb"),
            free_bytes=_integer(
                volume["free_bytes"],
                f"inventory.volumes[{index}].free_bytes",
                minimum=0,
            ),
        )
        if parsed_volume.free_bytes > parsed_volume.size_gb * 1_000_000_000:
            raise RunpodConfigError(
                f"inventory.volumes[{index}].free_bytes cannot exceed declared capacity"
            )
        volumes.append(parsed_volume)
    live_pods = raw["live_pods"]
    if not isinstance(live_pods, list) or any(not isinstance(item, str) for item in live_pods):
        raise RunpodConfigError("inventory.live_pods must be an array of strings")
    return InventorySnapshot(
        schema_version=1,
        observed_at=_timestamp(raw["observed_at"], "inventory.observed_at"),
        account_balance_usd=_decimal(raw["account_balance_usd"], "inventory.account_balance_usd"),
        offers=tuple(offers),
        volumes=tuple(volumes),
        live_pods=tuple(live_pods),
    )


def load_spend_ledger(path: str | Path) -> SpendLedger:
    raw = _versioned(
        _read_json(path),
        {
            "actual_spend_usd",
            "reserved_spend_usd",
            "bucket_actual_spend_usd",
            "bucket_reserved_spend_usd",
            "active_reservations",
            "consumed_authorization_digests",
        },
        "spend ledger",
    )
    buckets = {"storage", "qualification", "production", "recovery"}
    actual = _exact(raw["bucket_actual_spend_usd"], buckets, "ledger bucket actual spend")
    reserved = _exact(raw["bucket_reserved_spend_usd"], buckets, "ledger bucket reserved spend")
    reservations = raw["active_reservations"]
    consumed = raw["consumed_authorization_digests"]
    if not isinstance(reservations, list) or any(
        not isinstance(item, str) for item in reservations
    ):
        raise RunpodConfigError("ledger.active_reservations must be an array of strings")
    if not isinstance(consumed, list) or any(not isinstance(item, str) for item in consumed):
        raise RunpodConfigError("ledger.consumed_authorization_digests must be an array of strings")
    ledger = SpendLedger(
        schema_version=1,
        actual_spend_usd=_decimal(raw["actual_spend_usd"], "ledger.actual_spend_usd"),
        reserved_spend_usd=_decimal(raw["reserved_spend_usd"], "ledger.reserved_spend_usd"),
        bucket_actual_spend_usd={
            key: _decimal(value, f"ledger.bucket_actual_spend_usd.{key}")
            for key, value in actual.items()
        },
        bucket_reserved_spend_usd={
            key: _decimal(value, f"ledger.bucket_reserved_spend_usd.{key}")
            for key, value in reserved.items()
        },
        active_reservations=tuple(reservations),
        consumed_authorization_digests=tuple(consumed),
    )
    if ledger.actual_spend_usd != sum(ledger.bucket_actual_spend_usd.values(), start=Decimal("0")):
        raise RunpodConfigError(
            "ledger.actual_spend_usd must equal the sum of bucket_actual_spend_usd"
        )
    if ledger.reserved_spend_usd != sum(
        ledger.bucket_reserved_spend_usd.values(), start=Decimal("0")
    ):
        raise RunpodConfigError(
            "ledger.reserved_spend_usd must equal the sum of bucket_reserved_spend_usd"
        )
    return ledger


def load_launch_authorization(path: str | Path) -> LaunchAuthorization:
    raw = _versioned(
        _read_json(path),
        {
            "authorization_id",
            "subject_digest",
            "authorized_at",
            "expires_at",
            "maximum_projected_spend_usd",
            "approver",
            "autopay_disabled",
            "ordinary_launch_cutoff_usd",
            "campaign_ceiling_usd",
            "recovery_authorized",
        },
        "launch authorization",
    )
    return LaunchAuthorization(
        schema_version=1,
        authorization_id=_text(raw["authorization_id"], "authorization.authorization_id"),
        subject_digest=_text(raw["subject_digest"], "authorization.subject_digest"),
        authorized_at=_timestamp(raw["authorized_at"], "authorization.authorized_at"),
        expires_at=_timestamp(raw["expires_at"], "authorization.expires_at"),
        maximum_projected_spend_usd=_decimal(
            raw["maximum_projected_spend_usd"],
            "authorization.maximum_projected_spend_usd",
        ),
        approver=_text(raw["approver"], "authorization.approver"),
        autopay_disabled=_boolean(raw["autopay_disabled"], "authorization.autopay_disabled"),
        ordinary_launch_cutoff_usd=_decimal(
            raw["ordinary_launch_cutoff_usd"],
            "authorization.ordinary_launch_cutoff_usd",
        ),
        campaign_ceiling_usd=_decimal(
            raw["campaign_ceiling_usd"], "authorization.campaign_ceiling_usd"
        ),
        recovery_authorized=_boolean(
            raw["recovery_authorized"], "authorization.recovery_authorized"
        ),
    )


def parse_timestamp(value: str) -> datetime:
    return _timestamp(value, "evaluated_at")
