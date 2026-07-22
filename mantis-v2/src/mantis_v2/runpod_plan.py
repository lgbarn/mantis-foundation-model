"""Pure RunPod launch policy and durable decision writer."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any, cast

from mantis_v2.runpod_config import (
    ExperimentConfig,
    InventoryOffer,
    InventorySnapshot,
    InventoryVolume,
    LaunchAuthorization,
    LaunchIntent,
    LocalConfig,
    PlatformConfig,
    SpendLedger,
    canonical_digest,
    canonical_json,
    load_experiment_config,
    load_inventory_snapshot,
    load_launch_authorization,
    load_launch_intent,
    load_local_config,
    load_platform_config,
    load_spend_ledger,
    parse_timestamp,
)


class LaunchPlanError(ValueError):
    """Raised when a launch decision cannot be written safely."""


@dataclass(frozen=True)
class LaunchDecision:
    schema_version: int
    allowed: bool
    reasons: tuple[str, ...]
    evaluated_at: datetime
    platform_digest: str
    local_digest: str
    experiment_digest: str
    intent_digest: str
    inventory_digest: str
    ledger_digest: str
    authorization_subject_digest: str
    authorization_digest: str | None
    inventory_observed_at: datetime
    provider_price_usd_per_gpu_hour: Decimal | None
    maximum_duration_seconds: int
    projected_spend_usd: Decimal | None
    authorization_expires_at: datetime | None
    run_name: str
    stage: str
    gpu_type: str
    gpu_count: int
    vcpu: int
    ram_gb: int
    datacenter_id: str
    container_disk_gb: int
    image_ref: str
    template_id: str
    registry_auth_id: str
    volume_id: str
    volume_mount_path: str
    ports: tuple[str, ...]
    observed_price_usd_per_gpu_hour: Decimal | None
    openapi_identity: str
    openapi_version: str
    openapi_sha256: str
    decision_digest: str = ""

    def as_json_data(self) -> dict[str, Any]:
        data = asdict(self)
        if not self.decision_digest:
            data.pop("decision_digest")
        return cast(dict[str, Any], json.loads(canonical_json(data)))


def authorization_subject_digest(
    platform: PlatformConfig,
    local: LocalConfig,
    experiment: ExperimentConfig,
    intent: LaunchIntent,
    inventory: InventorySnapshot,
    ledger: SpendLedger,
) -> str:
    """Return the exact pre-authorization identity a human approves."""
    return canonical_digest(
        {
            "schema_version": 1,
            "platform_digest": platform.digest,
            "local_digest": local.digest,
            "experiment_digest": experiment.digest,
            "intent_digest": intent.digest,
            "inventory_digest": inventory.digest,
            "ledger_digest": ledger.digest,
        }
    )


def _matching_offer(intent: LaunchIntent, inventory: InventorySnapshot) -> InventoryOffer | None:
    matches = tuple(
        offer
        for offer in inventory.offers
        if offer.gpu_type == intent.gpu_type and offer.datacenter_id == intent.datacenter_id
    )
    return matches[0] if len(matches) == 1 else None


def _matching_volume(intent: LaunchIntent, inventory: InventorySnapshot) -> InventoryVolume | None:
    matches = tuple(
        volume
        for volume in inventory.volumes
        if volume.volume_id == intent.volume_id
        and volume.datacenter_id == intent.datacenter_id
        and volume.size_gb == intent.volume_size_gb
    )
    return matches[0] if len(matches) == 1 else None


def _projected_spend(
    platform: PlatformConfig, intent: LaunchIntent, price_per_gpu_hour: Decimal
) -> Decimal:
    hours = Decimal(intent.maximum_duration_seconds) / Decimal(3600)
    compute = price_per_gpu_hour * Decimal(intent.gpu_count) * hours
    container = (
        platform.billing.container_disk_usd_per_gb_month
        * Decimal(intent.container_disk_gb)
        * hours
        / Decimal(platform.billing.billing_month_hours)
    )
    return (compute + container).quantize(Decimal("0.01"), rounding=ROUND_CEILING)


def _policy_reasons(
    platform: PlatformConfig,
    experiment: ExperimentConfig,
    intent: LaunchIntent,
    inventory: InventorySnapshot,
    ledger: SpendLedger,
    authorization: LaunchAuthorization | None,
    evaluated_at: datetime,
    subject_digest: str,
    projected_spend: Decimal | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    offer = _matching_offer(intent, inventory)
    volume = _matching_volume(intent, inventory)
    inventory_age = evaluated_at - inventory.observed_at
    if inventory_age < timedelta(0) or inventory_age > timedelta(
        seconds=platform.lifecycle.maximum_inventory_age_seconds
    ):
        reasons.append("inventory_stale")
    if experiment.experiment.stage != intent.stage:
        reasons.append("experiment_stage_mismatch")
    if experiment.experiment.sealed_holdout:
        reasons.append("sealed_holdout_forbidden")
    if intent.gpu_type not in platform.provider.allowed_gpu_types:
        reasons.append("gpu_not_allowed")
    if intent.datacenter_id not in platform.provider.allowed_datacenters:
        reasons.append("datacenter_not_allowed")
    if offer is None:
        reasons.append("offer_not_found")
    elif not offer.available:
        reasons.append("offer_unavailable")
    if offer is not None and offer.cloud_type != "secure":
        reasons.append("secure_cloud_required")
    if intent.vcpu < platform.provider.minimum_vcpu:
        reasons.append("insufficient_vcpu")
    if intent.ram_gb < platform.provider.minimum_ram_gb:
        reasons.append("insufficient_ram")
    if intent.container_disk_gb != platform.provider.container_disk_gb:
        reasons.append("container_disk_mismatch")
    if intent.volume_size_gb != platform.storage.volume_gb:
        reasons.append("volume_size_mismatch")
    if intent.maximum_duration_seconds > platform.lifecycle.maximum_duration_seconds:
        reasons.append("duration_exceeds_limit")
    if volume is None:
        reasons.append("volume_not_found")
    else:
        required_free_bytes = max(
            platform.storage.minimum_free_bytes,
            intent.volume_size_gb * 1_000_000_000 - platform.storage.high_water_bytes,
        )
        if volume.free_bytes < required_free_bytes:
            reasons.append("insufficient_storage")
    if inventory.live_pods:
        reasons.append("live_pod_exists")
    if ledger.active_reservations:
        reasons.append("reserved_spend_overlap")
    committed_spend = ledger.actual_spend_usd + ledger.reserved_spend_usd
    if intent.stage != "recovery":
        ordinary_spend = committed_spend + (projected_spend or Decimal("0"))
        if ordinary_spend >= platform.budget.ordinary_launch_cutoff_usd:
            reasons.append("ordinary_launch_cutoff_reached")
    bucket_limit = {
        "qualification": platform.budget.qualification_usd,
        "production": platform.budget.production_usd,
        "recovery": platform.budget.protected_recovery_usd,
    }[intent.stage]
    bucket_committed = (
        ledger.bucket_actual_spend_usd[intent.stage]
        + ledger.bucket_reserved_spend_usd[intent.stage]
    )
    if projected_spend is not None:
        if bucket_committed + projected_spend > bucket_limit:
            reasons.append("stage_budget_exceeded")
        if committed_spend + projected_spend > platform.budget.account_ceiling_usd:
            reasons.append("account_ceiling_exceeded")
        if inventory.account_balance_usd < projected_spend:
            reasons.append("insufficient_balance")
    if authorization is None:
        reasons.append("authorization_required")
    else:
        if authorization.subject_digest != subject_digest:
            reasons.append("authorization_subject_mismatch")
        if authorization.authorized_at > evaluated_at:
            reasons.append("authorization_not_yet_valid")
        if authorization.expires_at <= evaluated_at:
            reasons.append("authorization_expired")
        if authorization.digest in ledger.consumed_authorization_digests:
            reasons.append("authorization_replayed")
        if (
            projected_spend is not None
            and authorization.maximum_projected_spend_usd < projected_spend
        ):
            reasons.append("authorization_spend_exceeded")
    return tuple(reasons)


def plan_launch(
    platform: PlatformConfig,
    local: LocalConfig,
    experiment: ExperimentConfig,
    intent: LaunchIntent,
    inventory: InventorySnapshot,
    ledger: SpendLedger,
    authorization: LaunchAuthorization | None,
    evaluated_at: datetime,
) -> LaunchDecision:
    """Evaluate launch policy without I/O or provider mutation."""
    subject_digest = authorization_subject_digest(
        platform, local, experiment, intent, inventory, ledger
    )
    authorization_digest = authorization.digest if authorization is not None else None
    offer = _matching_offer(intent, inventory)
    price = offer.price_usd_per_gpu_hour if offer is not None else None
    projected_spend = _projected_spend(platform, intent, price) if price is not None else None
    reasons = _policy_reasons(
        platform,
        experiment,
        intent,
        inventory,
        ledger,
        authorization,
        evaluated_at,
        subject_digest,
        projected_spend,
    )
    initial = LaunchDecision(
        schema_version=1,
        allowed=not reasons,
        reasons=reasons,
        evaluated_at=evaluated_at,
        platform_digest=platform.digest,
        local_digest=local.digest,
        experiment_digest=experiment.digest,
        intent_digest=intent.digest,
        inventory_digest=inventory.digest,
        ledger_digest=ledger.digest,
        authorization_subject_digest=subject_digest,
        authorization_digest=authorization_digest,
        inventory_observed_at=inventory.observed_at,
        provider_price_usd_per_gpu_hour=price,
        maximum_duration_seconds=intent.maximum_duration_seconds,
        projected_spend_usd=projected_spend,
        authorization_expires_at=authorization.expires_at if authorization else None,
        run_name=intent.run_name,
        stage=intent.stage,
        gpu_type=intent.gpu_type,
        gpu_count=intent.gpu_count,
        vcpu=intent.vcpu,
        ram_gb=intent.ram_gb,
        datacenter_id=intent.datacenter_id,
        container_disk_gb=intent.container_disk_gb,
        image_ref=intent.image_ref,
        template_id=intent.template_id,
        registry_auth_id=intent.registry_auth_id,
        volume_id=intent.volume_id,
        volume_mount_path=intent.volume_mount_path,
        ports=intent.ports,
        observed_price_usd_per_gpu_hour=price,
        openapi_identity=platform.adapter.openapi_identity,
        openapi_version=platform.adapter.openapi_version,
        openapi_sha256=platform.adapter.openapi_sha256,
    )
    return replace(initial, decision_digest=canonical_digest(initial.as_json_data()))


def _publish_no_overwrite(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise LaunchPlanError(f"launch decision already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_launch_decision(
    *,
    platform_path: Path,
    local_path: Path,
    experiment_path: Path,
    intent_path: Path,
    inventory_path: Path,
    ledger_path: Path,
    authorization_path: Path | None,
    evaluated_at: str,
    output_path: Path,
) -> dict[str, str]:
    """Load explicit inputs, plan once, and publish one canonical decision."""
    decision = plan_launch(
        load_platform_config(platform_path),
        load_local_config(local_path),
        load_experiment_config(experiment_path),
        load_launch_intent(intent_path),
        load_inventory_snapshot(inventory_path),
        load_spend_ledger(ledger_path),
        load_launch_authorization(authorization_path) if authorization_path else None,
        parse_timestamp(evaluated_at),
    )
    _publish_no_overwrite(output_path, canonical_json(decision) + "\n")
    return {"decision_path": str(output_path), "decision_digest": decision.decision_digest}
