"""Fail-closed statistics and promotion gates for Topstep test replay."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import TypeVar, cast

import numpy as np
import sklearn
import torch

from mantis_v2.downstream_config import load_downstream_config
from mantis_v2.rl_baselines import (
    HistoricalLogisticPolicy,
    fit_hist_gradient_boosting_baseline,
)
from mantis_v2.rl_config import RlConfig
from mantis_v2.rl_confirmation import (
    ConfirmationRequest,
    ConfirmationRunner,
    _atomic_no_overwrite,
    _lineage_passed,
    _load_object,
    _production_campaign_runner,
    _result_passed,
    _sha256,
    _validated_candidate,
    _validated_plan_pairs,
)
from mantis_v2.rl_environment import EntryOrder, EnvironmentEpisode, TopstepEntryEnvironment
from mantis_v2.rl_policy import PROFILES, TICKERS, EntryActorCritic, PolicyVariant
from mantis_v2.rl_validation import (
    LoadedEpisodes,
    _supervised_rows,
    historical_logistic_artifact,
    load_episode_manifest,
    load_test_episode_manifest,
)


class EvaluationContractError(ValueError):
    """Raised when replay evidence cannot support a promotion claim."""


@dataclass(frozen=True)
class EvaluationRequest:
    ordinal: int
    fold: int
    seed: int
    policy: str
    stress: str
    evaluation_plan_sha256: str
    test_manifest_path: Path
    test_manifest_sha256: str
    output: Path
    candidate_attempt_path: Path | None = None


EvaluationRunner = Callable[[EvaluationRequest], Sequence[Mapping[str, object]]]


_OUTCOMES = {"PASS", "BLOW", "TIMEOUT"}
_METRICS = (
    "trading_days",
    "calendar_days",
    "accepted_trades",
    "eligible_entries",
    "net_pnl",
    "costs",
    "commission_costs",
    "slippage_costs",
    "gap_costs",
    "expectancy",
    "maximum_drawdown",
    "minimum_mll_cushion",
    "best_day_profit",
    "consistency_ratio",
    "action_entropy",
    "ambiguity_count",
    "gap_adjusted_fills",
    "latency_cancellations",
    "missed_fill_count",
)
_NUMERIC_STRESSES = {
    "primary",
    "two_tick",
}
_SENSITIVITY_STRESSES = {
    "fee_2x",
    "gap_adverse_tick",
    "latency_1bar",
    "missed_fill_10pct",
    "same_bar_adverse",
    "overlapping_starts",
}
_REQUIRED_STRESSES = _NUMERIC_STRESSES | _SENSITIVITY_STRESSES
_HEADLINE = "primary"
_POLICIES = {
    "candidate",
    "reject_all",
    "take_all",
    "matched_random_take",
    "historical_rejected_logistic_head",
    "hist_gradient_boosting",
    "independent_ticker_ppo",
}
_ATTEMPT_FIELDS = {
    "fold",
    "seed",
    "ticker",
    "profile",
    "regime_block",
    "calendar_block",
    "episode_id",
    "start_ts",
    "end_ts",
    "status",
    "trading_days",
    "calendar_days",
    "accepted_trades",
    "eligible_entries",
    "net_pnl",
    "costs",
    "commission_costs",
    "slippage_costs",
    "gap_costs",
    "expectancy",
    "maximum_drawdown",
    "minimum_mll_cushion",
    "best_day_profit",
    "consistency_ratio",
    "action_entropy",
    "ambiguity_count",
    "gap_adjusted_fills",
    "latency_cancellations",
    "missed_fill_count",
    "finite",
    "accepted_opportunity_ids",
    "stress_identity_sha256",
    "stress_invariants_valid",
}


def _expected_risk_shield_identities(config: RlConfig) -> dict[str, dict[str, str]]:
    """Derive shield trust anchors independently of the supplied shield artifact."""
    repository = Path(__file__).resolve().parents[3]
    rule_path = config.upstream.rule_contract_path
    if not rule_path.is_absolute():
        rule_path = repository / rule_path
    paths = {
        "mask_source": repository / "mantis-v2/src/mantis_v2/rl_risk_shield.py",
        "observation_schema": Path(__file__).resolve().with_name("rl_environment.py"),
        "rule_snapshot": rule_path,
        "calendar_snapshot": repository / "mantis-v2/configs/topstep-holiday-calendar.json",
    }
    try:
        identities = {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()
        }
    except OSError as exc:
        raise EvaluationContractError(
            "issue #11 RiskShield or calendar authority is not installed"
        ) from exc
    if identities["rule_snapshot"]["sha256"] != config.upstream.rule_contract_sha256:
        raise EvaluationContractError("configured Topstep rule authority changed")
    identities["fee_snapshot"] = {
        "identity": config.fees.snapshot,
        "sha256": config.fee_digest,
    }
    return identities


def wilson_one_sided(
    successes: int, attempts: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Return one-sided Wilson lower and upper bounds for a binomial rate."""
    if (
        type(successes) is not int
        or type(attempts) is not int
        or attempts < 1
        or successes < 0
        or successes > attempts
        or not 0.5 < confidence < 1.0
    ):
        raise EvaluationContractError("Wilson inputs are invalid")
    z = NormalDist().inv_cdf(confidence)
    rate = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = (rate + z * z / (2.0 * attempts)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / attempts + z * z / (4.0 * attempts * attempts))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _distribution(values: Sequence[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise EvaluationContractError("metric values must be finite")
    ordered = np.sort(array)
    lower = len(ordered) * 0.25
    upper = len(ordered) * 0.75
    weights = np.asarray(
        [
            max(0.0, min(float(index + 1), upper) - max(float(index), lower))
            for index in range(len(ordered))
        ],
        dtype=np.float64,
    )
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "interquartile_mean": float(np.average(ordered, weights=weights)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "interval_95": [
            float(np.quantile(array, 0.025, method="inverted_cdf")),
            float(np.quantile(array, 0.975, method="inverted_cdf")),
        ],
    }


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise EvaluationContractError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise EvaluationContractError(f"{field} must be an ISO-8601 string") from exc
    if parsed.tzinfo is None:
        raise EvaluationContractError(f"{field} must include a timezone")
    return parsed


def _validated_attempts(
    rows: Sequence[Mapping[str, object]], *, allow_overlap: bool = False
) -> list[Mapping[str, object]]:
    if not rows:
        raise EvaluationContractError("attempt rows are missing")
    identities: set[tuple[object, ...]] = set()
    previous: dict[tuple[object, ...], datetime] = {}
    required = _ATTEMPT_FIELDS
    for row in rows:
        if set(row) != required:
            raise EvaluationContractError("attempt row schema is invalid")
        if row["status"] not in _OUTCOMES:
            raise EvaluationContractError("attempt status is invalid")
        if row["finite"] is not True:
            raise EvaluationContractError("attempt must be finite")
        if row["stress_invariants_valid"] is not True:
            raise EvaluationContractError("stress economic invariants failed")
        if (
            not isinstance(row["stress_identity_sha256"], str)
            or len(row["stress_identity_sha256"]) != 64
        ):
            raise EvaluationContractError("stress identity is invalid")
        opportunities = row["accepted_opportunity_ids"]
        if (
            not isinstance(opportunities, list)
            or any(not isinstance(value, str) or not value for value in opportunities)
            or len(opportunities) != len(set(opportunities))
        ):
            raise EvaluationContractError("accepted opportunity identities are invalid")
        for metric in _METRICS:
            value = row[metric]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise EvaluationContractError(f"{metric} must be finite")
            if (
                metric
                in {
                    "trading_days",
                    "calendar_days",
                    "accepted_trades",
                    "eligible_entries",
                    "costs",
                    "commission_costs",
                    "slippage_costs",
                    "gap_costs",
                    "maximum_drawdown",
                    "best_day_profit",
                    "consistency_ratio",
                    "action_entropy",
                    "ambiguity_count",
                    "gap_adjusted_fills",
                    "latency_cancellations",
                    "missed_fill_count",
                }
                and value < 0
            ):
                raise EvaluationContractError(f"{metric} must be non-negative")
        if cast(int | float, row["accepted_trades"]) > cast(int | float, row["eligible_entries"]):
            raise EvaluationContractError("accepted trades exceed eligible entries")
        if len(opportunities) != row["accepted_trades"]:
            raise EvaluationContractError("accepted opportunity identities do not match trades")
        start = _timestamp(row["start_ts"], "start_ts")
        end = _timestamp(row["end_ts"], "end_ts")
        if start >= end:
            raise EvaluationContractError("attempt interval is invalid")
        identity = (
            row["fold"],
            row["seed"],
            row["ticker"],
            row["profile"],
            row["episode_id"],
        )
        if identity in identities:
            raise EvaluationContractError("attempt identities must be unique")
        identities.add(identity)
        stream = (row["fold"], row["seed"], row["ticker"], row["profile"])
        if stream in previous and start <= previous[stream]:
            raise EvaluationContractError("attempts must be strictly chronological")
        previous[stream] = start if allow_overlap else end
    return list(rows)


def summarize_attempts(
    rows: Sequence[Mapping[str, object]], *, allow_overlap: bool = False
) -> dict[str, object]:
    """Summarize one policy/stress slice without hiding raw account outcomes."""
    attempts = _validated_attempts(rows, allow_overlap=allow_overlap)
    market_keys = {
        (row["fold"], row["episode_id"], row["ticker"], row["profile"]) for row in attempts
    }
    seeds = {row["seed"] for row in attempts}
    headline_seed = 42 if 42 in seeds else next(iter(seeds)) if len(seeds) == 1 else None
    if headline_seed is None:
        raise EvaluationContractError("serving seed 42 headline outcomes are missing")
    headline = [row for row in attempts if row["seed"] == headline_seed]
    headline_keys = {
        (row["fold"], row["episode_id"], row["ticker"], row["profile"]) for row in headline
    }
    if headline_keys != market_keys or len(headline) != len(market_keys):
        raise EvaluationContractError("headline seed does not own every unique attempt")
    counts = {outcome: sum(row["status"] == outcome for row in headline) for outcome in _OUTCOMES}
    total = len(headline)
    pass_lcb, pass_ucb = wilson_one_sided(counts["PASS"], total)
    blow_lcb, blow_ucb = wilson_one_sided(counts["BLOW"], total)
    result: dict[str, object] = {
        "attempt_count": len(market_keys),
        "replay_outcomes": len(attempts),
        "headline_seed": headline_seed,
        "outcomes": {outcome: counts[outcome] for outcome in ("PASS", "BLOW", "TIMEOUT")},
        "rates": {
            "pass": counts["PASS"] / total,
            "blow": counts["BLOW"] / total,
            "timeout": counts["TIMEOUT"] / total,
            "pass_wilson_lcb_95": pass_lcb,
            "pass_wilson_ucb_95": pass_ucb,
            "blow_wilson_lcb_95": blow_lcb,
            "blow_wilson_ucb_95": blow_ucb,
        },
    }
    for metric in _METRICS:
        result[metric] = _distribution([float(cast(int | float, row[metric])) for row in attempts])
    participation = [
        float(cast(int | float, row["accepted_trades"]))
        / float(cast(int | float, row["eligible_entries"]))
        if row["eligible_entries"]
        else 0.0
        for row in attempts
    ]
    result["participation_rate"] = _distribution(participation)
    result["skip_rate"] = _distribution([1.0 - value for value in participation])
    by_seed: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in attempts:
        seed = row["seed"]
        if type(seed) is not int:
            raise EvaluationContractError("seed must be an integer")
        by_seed[seed].append(row)
    seed_rates = {
        seed: sum(row["status"] == "PASS" for row in seed_rows) / len(seed_rows)
        for seed, seed_rows in by_seed.items()
    }
    robust = _distribution(list(seed_rates.values()))
    result["seed_robustness"] = {
        "pass_rates": {str(seed): seed_rates[seed] for seed in sorted(seed_rates)},
        "median_pass_rate": robust["median"],
        "interquartile_mean_pass_rate": robust["interquartile_mean"],
        "worst_seed": min(seed_rates, key=lambda seed: (seed_rates[seed], seed)),
        "worst_pass_rate": min(seed_rates.values()),
        "seed_interval_95": robust["interval_95"],
    }
    return result


def paired_calendar_block_lcb(
    candidate: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
    *,
    evaluation_plan_payload_sha256: str,
    expected_seeds: Sequence[int],
    replicates: int = 100_000,
    ordered_block_ids: Sequence[object] | None = None,
    synchronized_indices: np.ndarray | None = None,
) -> dict[str, object]:
    """Bootstrap paired pass differences by synchronized calendar blocks."""
    if (
        not isinstance(evaluation_plan_payload_sha256, str)
        or len(evaluation_plan_payload_sha256) != 64
        or len(expected_seeds) != 10
        or len(set(expected_seeds)) != 10
        or any(type(seed) is not int for seed in expected_seeds)
        or type(replicates) is not int
        or replicates < 1
    ):
        raise EvaluationContractError("bootstrap controls are invalid")
    key_fields = tuple(sorted(set(candidate[0]) - {"passed"})) if candidate else ()
    if not candidate or len(candidate) != len(baseline) or not key_fields:
        raise EvaluationContractError("paired bootstrap rows are incomplete")
    candidate_by_key = {tuple(row[field] for field in key_fields): row for row in candidate}
    baseline_by_key = {tuple(row[field] for field in key_fields): row for row in baseline}
    if len(candidate_by_key) != len(candidate) or set(candidate_by_key) != set(baseline_by_key):
        raise EvaluationContractError("paired bootstrap identities do not align")
    by_seed_block: dict[tuple[int, object], list[float]] = defaultdict(list)
    for key, row in candidate_by_key.items():
        block = row.get("calendar_block")
        if not isinstance(row.get("passed"), bool) or not isinstance(
            baseline_by_key[key].get("passed"), bool
        ):
            raise EvaluationContractError("paired outcomes must be boolean")
        seed_value = row.get("seed")
        if type(seed_value) is not int:
            raise EvaluationContractError("paired outcomes require integer seeds")
        by_seed_block[(seed_value, block)].append(
            float(cast(bool, row["passed"])) - float(cast(bool, baseline_by_key[key]["passed"]))
        )
    available_blocks = sorted({key[1] for key in by_seed_block}, key=str)
    blocks = available_blocks
    if ordered_block_ids is not None:
        blocks = list(ordered_block_ids)
        if (
            len(blocks) != len(set(blocks))
            or blocks != sorted(blocks, key=str)
            or not set(blocks).issubset(available_blocks)
        ):
            raise EvaluationContractError("paired view lacks a synchronized calendar block")
    if len(blocks) < 20:
        raise EvaluationContractError("at least 20 complete calendar blocks are required")
    if any((seed, block) not in by_seed_block for seed in expected_seeds for block in blocks):
        raise EvaluationContractError("every block and seed require a matched attempt")
    block_sums = np.asarray(
        [
            sum(value for seed in expected_seeds for value in by_seed_block[(seed, block)])
            for block in blocks
        ],
        dtype=np.float64,
    )
    block_counts = np.asarray(
        [sum(len(by_seed_block[(seed, block)]) for seed in expected_seeds) for block in blocks],
        dtype=np.float64,
    )
    if np.any(block_counts <= 0):
        raise EvaluationContractError("calendar block paired denominator is zero")
    point = float(block_sums.sum() / block_counts.sum())
    seed_effects = {
        seed: float(
            np.mean(
                [
                    value
                    for (owner, block), values in by_seed_block.items()
                    if owner == seed and block in blocks
                    for value in values
                ]
            )
        )
        for seed in expected_seeds
    }
    effects = np.asarray([seed_effects[seed] for seed in expected_seeds], dtype=np.float64)
    assignments = np.asarray(
        [[1.0 if mask & (1 << index) else -1.0 for index in range(10)] for mask in range(1_024)],
        dtype=np.float64,
    )
    seed_distribution = (assignments * effects).mean(axis=1)
    if not np.isfinite(seed_distribution).all() or float(np.ptp(seed_distribution)) == 0.0:
        raise EvaluationContractError("seed sign distribution is degenerate")
    seed = int.from_bytes(
        hashlib.sha256(
            f"{evaluation_plan_payload_sha256}:test-market-bootstrap-v1".encode()
        ).digest()[:8],
        "big",
    )
    if synchronized_indices is None:
        generator = np.random.Generator(np.random.PCG64(seed))
        synchronized_indices = generator.integers(
            0, len(blocks), size=(replicates, len(blocks)), dtype=np.uint32
        )
    if synchronized_indices.shape != (replicates, len(blocks)):
        raise EvaluationContractError("synchronized bootstrap index shape is invalid")
    if np.any(synchronized_indices >= len(blocks)):
        raise EvaluationContractError("synchronized bootstrap index is out of range")
    sampled_sums = block_sums[synchronized_indices].sum(axis=1)
    sampled_counts = block_counts[synchronized_indices].sum(axis=1)
    if np.any(sampled_counts <= 0):
        raise EvaluationContractError("bootstrap paired denominator is zero")
    samples = sampled_sums / sampled_counts
    index_digest = hashlib.sha256(synchronized_indices.tobytes())
    if not np.isfinite(samples).all() or float(np.ptp(samples)) == 0.0:
        raise EvaluationContractError("market bootstrap distribution is degenerate")
    observed_seed_effect = float(effects.mean())
    upper_null = float(np.quantile(seed_distribution, 0.95, method="inverted_cdf"))
    return {
        "point_difference": point,
        "lcb_95": float(np.quantile(samples, 0.05, method="inverted_cdf")),
        "replicates": replicates,
        "rng": "numpy.PCG64",
        "resampling_unit": "synchronized_calendar_block",
        "calendar_blocks": len(blocks),
        "ordered_block_ids": blocks,
        "numpy_version": np.__version__,
        "quantile_method": "inverted_cdf",
        "rng_seed": seed,
        "index_matrix_sha256": index_digest.hexdigest(),
        "seed_uncertainty": {
            "method": "exact_paired_sign_assignments_v1",
            "assignments": 1_024,
            "seed_effects": {str(seed): seed_effects[seed] for seed in expected_seeds},
            "point_difference": observed_seed_effect,
            "lower_bound_95": observed_seed_effect - upper_null,
            "assignment_matrix_sha256": hashlib.sha256(assignments.tobytes()).hexdigest(),
            "promotion_gate": False,
        },
    }


def _core_clauses(scope: str, values: Mapping[str, object], *, require_attempts: bool) -> list[str]:
    failed: list[str] = []
    attempts = cast(int | float, values.get("attempts", 0))
    pass_rate = cast(int | float, values.get("pass_rate", -math.inf))
    pass_lcb = cast(int | float, values.get("pass_lcb", -math.inf))
    blow_ucb = cast(int | float, values.get("blow_ucb", math.inf))
    checks = {
        "minimum_attempts": not require_attempts or attempts >= 300,
        "raw_pass_rate": pass_rate >= 0.60,
        "pass_lcb_95": pass_lcb >= 0.50,
        "zero_observed_blows": values.get("blows") == 0,
        "blow_ucb_95": blow_ucb <= 0.01,
    }
    for name, passed in checks.items():
        if not passed:
            failed.append(f"{scope}:{name}")
    return failed


def promotion_verdict(
    *,
    numeric_stresses: Mapping[str, Mapping[str, object]],
    sensitivity_stresses: Mapping[str, Mapping[str, object]],
    required_seeds: set[int],
    required_transfer_cells: set[tuple[str, str]],
    required_artifacts_complete: bool,
) -> dict[str, object]:
    """Apply every preregistered clause; absent or malformed evidence fails closed."""
    failed: list[str] = []
    if set(numeric_stresses) != _NUMERIC_STRESSES:
        failed.append("numeric_stresses:complete")
    required_views = {"pooled", "one_mini", "ten_micros"}
    required_transfer = required_transfer_cells | {("POOLED", "pooled")}
    for stress in sorted(_NUMERIC_STRESSES):
        evidence = numeric_stresses.get(stress)
        if evidence is None:
            continue
        profiles = evidence.get("profiles")
        pooled = evidence.get("pooled")
        seed_rates = evidence.get("seed_pass_rates")
        seed_blows = evidence.get("seed_blow_counts")
        baseline_effects = evidence.get("baseline_effects")
        transfer_effects = evidence.get("transfer_effects")
        if not isinstance(pooled, Mapping) or not isinstance(profiles, Mapping):
            failed.append(f"{stress}:views:complete")
            continue
        failed.extend(_core_clauses(f"{stress}:pooled", pooled, require_attempts=True))
        if set(profiles) != {"one_mini", "ten_micros"}:
            failed.append(f"{stress}:profiles:complete")
        for profile in ("one_mini", "ten_micros"):
            profile_values = profiles.get(profile)
            if isinstance(profile_values, Mapping):
                failed.extend(
                    _core_clauses(
                        f"{stress}:profile:{profile}", profile_values, require_attempts=True
                    )
                )
        if not isinstance(seed_rates, Mapping) or set(seed_rates) != required_views:
            failed.append(f"{stress}:seeds:complete")
        else:
            for scope in sorted(required_views):
                rates = seed_rates[scope]
                if not isinstance(rates, Mapping) or set(rates) != required_seeds:
                    failed.append(f"{stress}:{scope}:seeds:complete")
                    continue
                for seed, rate in rates.items():
                    if not isinstance(rate, int | float) or not math.isfinite(rate) or rate <= 0.50:
                        failed.append(f"{stress}:{scope}:seed:{seed}:raw_pass_rate")
        if not isinstance(seed_blows, Mapping) or set(seed_blows) != required_views:
            failed.append(f"{stress}:seed_blows:complete")
        else:
            for scope in sorted(required_views):
                blows = seed_blows[scope]
                if not isinstance(blows, Mapping) or set(blows) != required_seeds:
                    failed.append(f"{stress}:{scope}:seed_blows:complete")
                    continue
                for seed, count in blows.items():
                    if type(count) is not int or count != 0:
                        failed.append(f"{stress}:{scope}:seed:{seed}:zero_observed_blows")
        if not isinstance(baseline_effects, Mapping) or set(baseline_effects) != required_views:
            failed.append(f"{stress}:baselines:complete")
        else:
            for scope in sorted(required_views):
                effects = baseline_effects[scope]
                if not isinstance(effects, Mapping):
                    failed.append(f"{stress}:{scope}:baselines:complete")
                    continue
                for baseline in ("take_all", "matched_random_take"):
                    effect = effects.get(baseline)
                    if not isinstance(effect, Mapping):
                        failed.append(f"{stress}:{scope}:baseline:{baseline}:complete")
                    elif effect.get("point_difference", -math.inf) <= 0:
                        failed.append(
                            f"{stress}:{scope}:baseline:{baseline}:strict_point_improvement"
                        )
                    elif effect.get("lcb_95", -math.inf) <= 0:
                        failed.append(f"{stress}:{scope}:baseline:{baseline}:positive_lcb_95")
        if not isinstance(transfer_effects, Mapping) or set(transfer_effects) != required_transfer:
            failed.append(f"{stress}:transfer:complete")
        else:
            for ticker, profile in sorted(required_transfer):
                effect = transfer_effects[(ticker, profile)]
                if not isinstance(effect, Mapping):
                    failed.append(f"{stress}:transfer:{ticker}:{profile}:complete")
                elif effect.get("point_difference", -math.inf) < 0:
                    failed.append(f"{stress}:transfer:{ticker}:{profile}:point_difference")
                elif effect.get("lcb_95", -math.inf) < 0:
                    failed.append(f"{stress}:transfer:{ticker}:{profile}:lcb_95")
    if set(sensitivity_stresses) != _SENSITIVITY_STRESSES:
        failed.append("sensitivity_stresses:complete")
    for stress in sorted(_SENSITIVITY_STRESSES):
        result = sensitivity_stresses.get(stress)
        if result is None or result.get("complete") is not True:
            failed.append(f"sensitivity:{stress}:complete")
    if not required_artifacts_complete:
        failed.append("artifacts:complete")
    return {
        "promoted": not failed,
        "decision": "PROMOTE" if not failed else "DO_NOT_PROMOTE",
        "failed_clauses": failed,
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_bytes_no_overwrite(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise EvaluationContractError(
            f"immutable artifact already exists with different bytes: {path}"
        )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise EvaluationContractError(f"immutable artifact already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _array_identity(values: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(values)
    descriptor = _canonical_bytes({"shape": list(contiguous.shape), "dtype": contiguous.dtype.str})
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(descriptor + contiguous.tobytes()).hexdigest(),
    }


def write_independent_baseline_campaign_plan(
    config: RlConfig,
    candidate_path: Path,
    serving_freeze_path: Path,
    training_manifest_paths: Sequence[Path],
    validation_manifest_paths: Sequence[Path],
    output: Path,
) -> dict[str, object]:
    """Freeze the fresh final-budget independent-actor campaign before training it."""
    candidate, candidate_sha256 = _validated_candidate(config, candidate_path)
    serving, serving_sha256 = _validate_serving_freeze(config, serving_freeze_path)
    if serving.get("candidate_sha256") != candidate_sha256:
        raise EvaluationContractError("independent campaign serving lineage mismatch")
    if not training_manifest_paths or len(training_manifest_paths) != len(
        validation_manifest_paths
    ):
        raise EvaluationContractError("independent campaign manifest pairs are incomplete")
    architecture_plan = _load_object(
        Path(cast(str, candidate["architecture_plan_path"])), "candidate architecture plan"
    )
    frozen_pairs = _validated_plan_pairs(config, architecture_plan)
    supplied_pairs: list[dict[str, object]] = []
    for training_path, validation_path in zip(
        training_manifest_paths, validation_manifest_paths, strict=True
    ):
        training = load_episode_manifest(config, training_path)
        validation = load_episode_manifest(config, validation_path)
        if (
            training.partition != "training"
            or validation.partition != "validation"
            or training.fold != validation.fold
        ):
            raise EvaluationContractError(
                "independent campaign requires same-fold train/validation"
            )
        supplied_pairs.append(
            {
                "fold": training.fold,
                "training_manifest_path": str(training_path.resolve()),
                "training_manifest_sha256": _sha256(training_path),
                "validation_manifest_path": str(validation_path.resolve()),
                "validation_manifest_sha256": _sha256(validation_path),
            }
        )
    supplied_pairs.sort(key=lambda item: cast(int, item["fold"]))
    if supplied_pairs != frozen_pairs:
        raise EvaluationContractError("independent campaign schedules differ from issue #9")
    qualification_path = Path(cast(str, candidate["qualification_report_path"]))
    qualification = _load_object(qualification_path, "architecture qualification report")
    report_references = qualification.get("reports")
    if not isinstance(report_references, list):
        raise EvaluationContractError("independent architecture lineage is missing")
    independent_lineage: list[dict[str, object]] = []
    for reference in report_references:
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            raise EvaluationContractError("independent architecture reference is invalid")
        report_path = Path(cast(str, reference["path"]))
        if _sha256(report_path) != reference.get("sha256"):
            raise EvaluationContractError("independent architecture reference changed")
        report = _load_object(report_path, "independent architecture evidence")
        if report.get("variant") == "independent_actor":
            independent_lineage.append(
                {
                    "fold": report.get("fold"),
                    "seed": report.get("seed"),
                    "path": str(report_path.resolve()),
                    "sha256": reference["sha256"],
                }
            )
    expected_development = {
        (cast(int, pair["fold"]), seed)
        for pair in supplied_pairs
        for seed in config.training.development_seeds
    }
    if {(item["fold"], item["seed"]) for item in independent_lineage} != expected_development:
        raise EvaluationContractError("issue #9 independent architecture lineage is incomplete")
    final_timesteps = serving.get("final_timesteps")
    if final_timesteps not in {
        config.training.confirmation_timesteps_per_seed,
        config.training.maximum_timesteps_per_seed,
    }:
        raise EvaluationContractError("independent campaign final budget is invalid")
    expected_ledger = [
        {
            "ordinal": ordinal,
            "fold": cast(int, pair["fold"]),
            "seed": seed,
            "variant": "independent_actor",
            "timesteps": final_timesteps,
        }
        for ordinal, (pair, seed) in enumerate(
            (
                (pair, seed)
                for pair in supplied_pairs
                for seed in config.training.confirmation_seeds
            ),
            start=1,
        )
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "independent-baseline-campaign-plan-v1",
        "status": "frozen_before_training",
        "config_sha256": config.digest,
        "source_sha256": config.upstream.source_digest,
        "dependency_lock_sha256": config.upstream.lock_digest,
        "candidate_path": str(candidate_path.resolve()),
        "candidate_sha256": candidate_sha256,
        "serving_freeze_path": str(serving_freeze_path.resolve()),
        "serving_freeze_sha256": serving_sha256,
        "continuation_decision_path": serving["continuation_decision_path"],
        "continuation_decision_sha256": serving["continuation_decision_sha256"],
        "final_timesteps": final_timesteps,
        "parameters": candidate["optuna_parameters"],
        "manifest_pairs": supplied_pairs,
        "issue9_independent_lineage": sorted(
            independent_lineage, key=lambda item: (cast(int, item["fold"]), cast(int, item["seed"]))
        ),
        "expected_append_only_ledger": expected_ledger,
        "test_accessed": False,
        "sealed_holdout_accessed": False,
    }
    _validated_independent_campaign_plan_constraints(config, payload)
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    path = output / f"independent-baseline-campaign-plan-{digest}.json"
    _atomic_no_overwrite(path, payload)
    return {"campaign_plan_path": str(path), "campaign_plan_sha256": digest, **payload}


def _validated_independent_campaign_plan_constraints(
    config: RlConfig, plan: Mapping[str, object]
) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Recompute every reusable independent-campaign constraint from issue #9."""
    candidate_path = plan.get("candidate_path")
    serving_path = plan.get("serving_freeze_path")
    if not isinstance(candidate_path, str) or not isinstance(serving_path, str):
        raise EvaluationContractError("independent campaign parent paths are missing")
    candidate, candidate_sha256 = _validated_candidate(config, Path(candidate_path))
    serving, serving_sha256 = _validate_serving_freeze(config, Path(serving_path))
    architecture_path = Path(cast(str, candidate["architecture_plan_path"]))
    architecture_plan = _load_object(architecture_path, "candidate architecture plan")
    frozen_pairs = _validated_plan_pairs(config, architecture_plan)
    final_timesteps = serving.get("final_timesteps")
    if (
        plan.get("schema_version") != 1
        or plan.get("stage") != "independent-baseline-campaign-plan-v1"
        or plan.get("status") != "frozen_before_training"
        or plan.get("config_sha256") != config.digest
        or plan.get("source_sha256") != config.upstream.source_digest
        or plan.get("dependency_lock_sha256") != config.upstream.lock_digest
        or plan.get("candidate_sha256") != candidate_sha256
        or plan.get("serving_freeze_sha256") != serving_sha256
        or serving.get("candidate_sha256") != candidate_sha256
        or plan.get("continuation_decision_path") != serving.get("continuation_decision_path")
        or plan.get("continuation_decision_sha256") != serving.get("continuation_decision_sha256")
        or final_timesteps
        not in {
            config.training.confirmation_timesteps_per_seed,
            config.training.maximum_timesteps_per_seed,
        }
        or plan.get("final_timesteps") != final_timesteps
        or plan.get("parameters") != candidate.get("optuna_parameters")
        or plan.get("manifest_pairs") != frozen_pairs
        or plan.get("test_accessed") is not False
        or plan.get("sealed_holdout_accessed") is not False
    ):
        raise EvaluationContractError("independent campaign plan constraint mismatch")

    qualification_path = Path(cast(str, candidate["qualification_report_path"]))
    qualification = _load_object(qualification_path, "architecture qualification report")
    references = qualification.get("reports")
    if not isinstance(references, list):
        raise EvaluationContractError("independent architecture lineage is missing")
    expected_lineage: list[dict[str, object]] = []
    pair_by_fold = {cast(int, pair["fold"]): pair for pair in frozen_pairs}
    for reference in references:
        if (
            not isinstance(reference, dict)
            or not isinstance(reference.get("path"), str)
            or not isinstance(reference.get("sha256"), str)
        ):
            raise EvaluationContractError("independent architecture reference is invalid")
        report_path = Path(cast(str, reference["path"]))
        if _sha256(report_path) != reference["sha256"]:
            raise EvaluationContractError("independent architecture reference changed")
        report = _load_object(report_path, "independent architecture evidence")
        if report.get("variant") != "independent_actor":
            continue
        fold = report.get("fold")
        seed = report.get("seed")
        pair = pair_by_fold.get(fold) if type(fold) is int else None
        if (
            pair is None
            or type(seed) is not int
            or seed not in config.training.development_seeds
            or report.get("partition") != "validation"
            or report.get("validation_manifest_sha256") != pair["validation_manifest_sha256"]
            or report.get("test_accessed") is not False
            or report.get("sealed_holdout_accessed") is not False
        ):
            raise EvaluationContractError("independent architecture lineage is unqualified")
        expected_lineage.append(
            {
                "fold": fold,
                "seed": seed,
                "path": str(report_path.resolve()),
                "sha256": reference["sha256"],
            }
        )
    expected_lineage.sort(key=lambda item: (cast(int, item["fold"]), cast(int, item["seed"])))
    expected_development = [
        (cast(int, pair["fold"]), seed)
        for pair in frozen_pairs
        for seed in config.training.development_seeds
    ]
    if [
        (cast(int, item["fold"]), cast(int, item["seed"])) for item in expected_lineage
    ] != expected_development or plan.get("issue9_independent_lineage") != expected_lineage:
        raise EvaluationContractError("issue #9 independent architecture lineage is incomplete")
    expected_ledger = [
        {
            "ordinal": ordinal,
            "fold": cast(int, pair["fold"]),
            "seed": seed,
            "variant": "independent_actor",
            "timesteps": final_timesteps,
        }
        for ordinal, (pair, seed) in enumerate(
            ((pair, seed) for pair in frozen_pairs for seed in config.training.confirmation_seeds),
            start=1,
        )
    ]
    if plan.get("expected_append_only_ledger") != expected_ledger:
        raise EvaluationContractError("independent campaign ledger schedule mismatch")
    return candidate, serving, frozen_pairs, expected_ledger


def _validated_independent_campaign_plan(
    config: RlConfig, path: Path
) -> tuple[dict[str, object], str]:
    plan = _load_object(path, "independent baseline campaign plan")
    digest = _sha256(path)
    if (
        path.name != f"independent-baseline-campaign-plan-{digest}.json"
        or plan.get("stage") != "independent-baseline-campaign-plan-v1"
        or plan.get("status") != "frozen_before_training"
        or plan.get("config_sha256") != config.digest
        or plan.get("source_sha256") != config.upstream.source_digest
        or plan.get("dependency_lock_sha256") != config.upstream.lock_digest
        or plan.get("test_accessed") is not False
        or plan.get("sealed_holdout_accessed") is not False
    ):
        raise EvaluationContractError("independent campaign plan provenance mismatch")
    _validated_independent_campaign_plan_constraints(config, plan)
    return plan, digest


def _validated_independent_attempt(
    config: RlConfig,
    plan: Mapping[str, object],
    plan_sha256: str,
    record: Mapping[str, object],
    *,
    ordinal: int,
    campaign_root: Path,
) -> tuple[ConfirmationRequest, Mapping[str, object]]:
    """Recursively validate one immutable independent result and its exact request."""
    pairs = cast(list[dict[str, object]], plan["manifest_pairs"])
    expected = cast(list[dict[str, object]], plan["expected_append_only_ledger"])
    if ordinal < 1 or ordinal > len(expected):
        raise EvaluationContractError("independent campaign attempt ordinal is invalid")
    request_identity = expected[ordinal - 1]
    execution = record.get("execution_request")
    result = record.get("result")
    if not isinstance(execution, dict) or not isinstance(result, dict):
        raise EvaluationContractError("independent campaign attempt evidence is incomplete")
    fold = cast(int, request_identity["fold"])
    seed = cast(int, request_identity["seed"])
    pair_by_fold = {cast(int, pair["fold"]): pair for pair in pairs}
    pair = pair_by_fold[fold]
    expected_output = campaign_root / "runs" / f"fold-{fold}" / f"seed-{seed}"
    if (
        record.get("schema_version") != 1
        or record.get("stage") != "independent-baseline-attempt-v1"
        or record.get("campaign_plan_sha256") != plan_sha256
        or record.get("request") != request_identity
        or record.get("test_accessed") is not False
        or record.get("sealed_holdout_accessed") is not False
        or execution.get("phase") != "independent_final"
        or execution.get("fold") != fold
        or execution.get("training_manifest_sha256") != pair["training_manifest_sha256"]
        or execution.get("validation_manifest_sha256") != pair["validation_manifest_sha256"]
        or execution.get("seed") != seed
        or execution.get("timesteps") != plan["final_timesteps"]
        or execution.get("variant") != "independent_actor"
        or execution.get("parameters") != plan["parameters"]
        or execution.get("candidate_sha256") != plan["candidate_sha256"]
        or execution.get("output") != str(expected_output)
        or type(execution.get("resume")) is not bool
        or execution.get("parent_artifact_sha256") is not None
        or execution.get("required_milestone_timesteps")
        != config.training.development_timesteps_per_seed
    ):
        raise EvaluationContractError("independent campaign attempt request mismatch")
    request = ConfirmationRequest(
        phase="independent_final",
        fold=fold,
        training_manifest_sha256=cast(str, pair["training_manifest_sha256"]),
        validation_manifest_sha256=cast(str, pair["validation_manifest_sha256"]),
        seed=seed,
        timesteps=cast(int, plan["final_timesteps"]),
        variant="independent_actor",
        parameters=cast(Mapping[str, object], plan["parameters"]),
        candidate_sha256=cast(str, plan["candidate_sha256"]),
        output=expected_output,
        resume=cast(bool, execution["resume"]),
        parent_artifact_sha256=None,
        required_milestone_timesteps=config.training.development_timesteps_per_seed,
    )
    if (
        not _result_passed(result)
        or not _lineage_passed(request, result)
        or result.get("completed_timesteps") != request.timesteps
        or result.get("lineage_parent_sha256") is not None
    ):
        raise EvaluationContractError("independent campaign attempt lineage mismatch")
    return request, result


def _independent_artifact_from_result(
    request_identity: Mapping[str, object], result: Mapping[str, object]
) -> dict[str, object]:
    bundle_path = Path(cast(str, result["checkpoint_bundle_path"]))
    checkpoint_path = bundle_path.parent / "checkpoint.pt"
    if _sha256(checkpoint_path) != result["artifact_sha256"]:
        raise EvaluationContractError("independent campaign checkpoint bytes changed")
    return {
        "fold": request_identity["fold"],
        "seed": request_identity["seed"],
        "variant": "independent_actor",
        "timesteps": result["completed_timesteps"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": result["artifact_sha256"],
        "training_manifest_path": result["training_manifest_path"],
        "training_manifest_sha256": result["training_manifest_sha256"],
        "checkpoint_bundle_path": result["checkpoint_bundle_path"],
        "checkpoint_bundle_sha256": result["checkpoint_bundle_sha256"],
        "validation_report_path": result["validation_report_path"],
        "validation_report_sha256": result["validation_report_sha256"],
        "finite": result["finite"],
        "action_collapsed": result["action_collapsed"],
        "blows": result["blows"],
    }


def run_independent_baseline_campaign(
    config: RlConfig,
    campaign_plan_path: Path,
    output: Path,
    *,
    runner: ConfirmationRunner | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Train every independent actor fresh at the serving budget and freeze validation evidence."""
    plan, plan_sha256 = _validated_independent_campaign_plan(config, campaign_plan_path)
    pairs = cast(list[dict[str, object]], plan["manifest_pairs"])
    if runner is None:
        runner = _production_campaign_runner(config, pairs, output)
    ledger = output / "ledger"
    attempts: list[dict[str, object]] = []
    expected = cast(list[dict[str, object]], plan["expected_append_only_ledger"])
    if resume and ledger.is_dir():
        ledger_paths = sorted(ledger.glob("attempt-*.json"))
        if len(ledger_paths) > len(expected):
            raise EvaluationContractError("independent campaign resume has extra attempts")
        for ordinal, ledger_path in enumerate(ledger_paths, start=1):
            record = _load_object(ledger_path, "independent campaign attempt")
            if ledger_path.name != f"attempt-{ordinal:04d}.json":
                raise EvaluationContractError("independent campaign resume mismatch")
            _validated_independent_attempt(
                config,
                plan,
                plan_sha256,
                record,
                ordinal=ordinal,
                campaign_root=output,
            )
            attempts.append(record)
    pair_by_fold = {cast(int, pair["fold"]): pair for pair in pairs}
    for request_identity in expected[len(attempts) :]:
        fold = cast(int, request_identity["fold"])
        seed = cast(int, request_identity["seed"])
        pair = pair_by_fold[fold]
        request = ConfirmationRequest(
            phase="independent_final",
            fold=fold,
            training_manifest_sha256=cast(str, pair["training_manifest_sha256"]),
            validation_manifest_sha256=cast(str, pair["validation_manifest_sha256"]),
            seed=seed,
            timesteps=cast(int, plan["final_timesteps"]),
            variant="independent_actor",
            parameters=cast(Mapping[str, object], plan["parameters"]),
            candidate_sha256=cast(str, plan["candidate_sha256"]),
            output=output / "runs" / f"fold-{fold}" / f"seed-{seed}",
            resume=resume,
            parent_artifact_sha256=None,
            required_milestone_timesteps=config.training.development_timesteps_per_seed,
        )
        try:
            result = dict(runner(request))
        except Exception as exc:
            result = {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        new_record: dict[str, object] = {
            "schema_version": 1,
            "stage": "independent-baseline-attempt-v1",
            "campaign_plan_sha256": plan_sha256,
            "request": request_identity,
            "execution_request": {**asdict(request), "output": str(request.output)},
            "result": result,
            "test_accessed": False,
            "sealed_holdout_accessed": False,
        }
        try:
            _validated_independent_attempt(
                config,
                plan,
                plan_sha256,
                new_record,
                ordinal=cast(int, request_identity["ordinal"]),
                campaign_root=output,
            )
        except EvaluationContractError:
            passed = False
        else:
            passed = True
        _atomic_no_overwrite(
            ledger / f"attempt-{cast(int, request_identity['ordinal']):04d}.json", new_record
        )
        attempts.append(new_record)
        if not passed:
            failure_payload = {
                "schema_version": 1,
                "stage": "independent-baseline-campaign-failure-v1",
                "status": "failed",
                "campaign_plan_sha256": plan_sha256,
                "request": request_identity,
                "result": result,
                "test_accessed": False,
                "sealed_holdout_accessed": False,
            }
            failure_sha256 = hashlib.sha256(_canonical_bytes(failure_payload)).hexdigest()
            _atomic_no_overwrite(
                output / "failures" / f"independent-baseline-failure-{failure_sha256}.json",
                failure_payload,
            )
            raise EvaluationContractError(
                f"independent baseline fold {fold} seed {seed} failed validation"
            )
    artifacts: list[dict[str, object]] = []
    for artifact_record in attempts:
        request_identity = cast(dict[str, object], artifact_record["request"])
        result = cast(dict[str, object], artifact_record["result"])
        artifacts.append(_independent_artifact_from_result(request_identity, result))
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "independent-baseline-campaign-v1",
        "status": "complete",
        "campaign_plan_path": str(campaign_plan_path.resolve()),
        "campaign_plan_sha256": plan_sha256,
        "config_sha256": config.digest,
        "source_sha256": config.upstream.source_digest,
        "dependency_lock_sha256": config.upstream.lock_digest,
        "final_timesteps": plan["final_timesteps"],
        "artifacts": artifacts,
        "attempt_ledger": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in sorted(ledger.glob("attempt-*.json"))
        ],
        "test_accessed": False,
        "sealed_holdout_accessed": False,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    path = output / f"independent-baseline-campaign-{digest}.json"
    _atomic_no_overwrite(path, payload)
    return {"campaign_freeze_path": str(path), "campaign_freeze_sha256": digest, **payload}


def _validated_independent_campaign(config: RlConfig, path: Path) -> tuple[dict[str, object], str]:
    campaign = _load_object(path, "independent baseline campaign")
    digest = _sha256(path)
    if (
        path.name != f"independent-baseline-campaign-{digest}.json"
        or campaign.get("stage") != "independent-baseline-campaign-v1"
        or campaign.get("status") != "complete"
        or campaign.get("config_sha256") != config.digest
        or campaign.get("source_sha256") != config.upstream.source_digest
        or campaign.get("dependency_lock_sha256") != config.upstream.lock_digest
        or campaign.get("test_accessed") is not False
        or campaign.get("sealed_holdout_accessed") is not False
    ):
        raise EvaluationContractError("independent campaign freeze provenance mismatch")
    plan_path = campaign.get("campaign_plan_path")
    if not isinstance(plan_path, str):
        raise EvaluationContractError("independent campaign plan reference is missing")
    plan, plan_sha256 = _validated_independent_campaign_plan(config, Path(plan_path))
    if plan_sha256 != campaign.get("campaign_plan_sha256") or plan.get(
        "final_timesteps"
    ) != campaign.get("final_timesteps"):
        raise EvaluationContractError("independent campaign plan identity mismatch")
    ledgers = campaign.get("attempt_ledger")
    artifacts = campaign.get("artifacts")
    if not isinstance(ledgers, list) or not isinstance(artifacts, list):
        raise EvaluationContractError("independent campaign evidence is incomplete")
    if len(ledgers) != len(cast(list[object], plan["expected_append_only_ledger"])):
        raise EvaluationContractError("independent campaign ledger is incomplete")
    expected_artifacts: list[dict[str, object]] = []
    for ordinal, reference in enumerate(ledgers, start=1):
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            raise EvaluationContractError("independent campaign ledger reference is invalid")
        ledger_path = Path(cast(str, reference["path"]))
        record = _load_object(ledger_path, "independent campaign attempt")
        if (
            ledger_path.parent != path.parent / "ledger"
            or ledger_path.name != f"attempt-{ordinal:04d}.json"
            or _sha256(ledger_path) != reference.get("sha256")
        ):
            raise EvaluationContractError("independent campaign ledger provenance mismatch")
        _request, result = _validated_independent_attempt(
            config,
            plan,
            plan_sha256,
            record,
            ordinal=ordinal,
            campaign_root=path.parent,
        )
        expected_artifacts.append(
            _independent_artifact_from_result(cast(dict[str, object], record["request"]), result)
        )
    if artifacts != expected_artifacts:
        raise EvaluationContractError("independent campaign artifacts differ from ledger")
    return campaign, digest


def _hgb_freeze_material(
    config: RlConfig, training: LoadedEpisodes, validation: LoadedEpisodes
) -> tuple[bytes, dict[str, object], dict[str, object]]:
    training_rows = _supervised_rows(config, training)
    validation_rows = _supervised_rows(config, validation)
    _policy, estimator, evidence = fit_hist_gradient_boosting_baseline(
        training_rows, validation_rows, seed=config.run.seed
    )
    estimator_bytes = pickle.dumps(estimator, protocol=5)
    input_lineage: dict[str, object] = {
        "training_features": _array_identity(training_rows.features),
        "training_labels": _array_identity(training_rows.labels),
        "validation_features": _array_identity(validation_rows.features),
        "validation_labels": _array_identity(validation_rows.labels),
    }
    baseline_code = Path(__file__).resolve().with_name("rl_baselines.py")
    evidence_payload = asdict(evidence)
    artifact_contract: dict[str, object] = {
        "estimator_schema": {
            "format": "python_pickle",
            "pickle_protocol": 5,
            "module": type(estimator).__module__,
            "class": type(estimator).__qualname__,
            "library": "scikit-learn",
            "library_version": sklearn.__version__,
        },
        "fit_code_path": str(baseline_code),
        "fit_code_sha256": _sha256(baseline_code),
        "fit_evidence": evidence_payload,
        "threshold_selection": {
            "source": evidence.threshold_source,
            "method": evidence.threshold_selection,
            "threshold": evidence.threshold,
            "validation_features_sha256": cast(
                dict[str, object], input_lineage["validation_features"]
            )["sha256"],
            "validation_labels_sha256": cast(dict[str, object], input_lineage["validation_labels"])[
                "sha256"
            ],
        },
        "input_lineage": input_lineage,
        "test_labels_accessed": False,
    }
    return estimator_bytes, input_lineage, artifact_contract


def _historical_logistic_contract(config: RlConfig, fold: int) -> dict[str, object]:
    policy, artifact_path, artifact_sha256 = historical_logistic_artifact(config, fold)
    source_code = Path(__file__).resolve().with_name("rl_validation.py")
    return {
        "path": str(artifact_path.resolve()),
        "sha256": artifact_sha256,
        "threshold": policy.head.threshold,
        "estimator_schema": {
            "format": "portable_rejected_logistic_head",
            "class": type(policy).__qualname__,
        },
        "source_code_path": str(source_code),
        "source_code_sha256": _sha256(source_code),
    }


def _independent_ppo_baseline_contract(
    artifact: Mapping[str, object],
    campaign: Mapping[str, object],
    campaign_sha256: str,
    plan: Mapping[str, object],
    plan_sha256: str,
) -> dict[str, object]:
    """Project one recursively validated campaign artifact without losing identity."""
    return {
        "fold": artifact["fold"],
        "seed": artifact["seed"],
        "path": artifact["checkpoint_path"],
        "sha256": artifact["checkpoint_sha256"],
        "variant": artifact["variant"],
        "timesteps": artifact["timesteps"],
        "validation_report_path": artifact["validation_report_path"],
        "validation_report_sha256": artifact["validation_report_sha256"],
        "campaign_artifact": dict(artifact),
        "campaign_sha256": campaign_sha256,
        "campaign_plan_sha256": plan_sha256,
        "campaign_phase": "independent_final",
        "parameters": plan["parameters"],
        "config_sha256": campaign["config_sha256"],
        "source_sha256": campaign["source_sha256"],
        "dependency_lock_sha256": campaign["dependency_lock_sha256"],
        "final_timesteps": campaign["final_timesteps"],
    }


def freeze_evaluation_baselines(
    config: RlConfig,
    architecture_candidate_path: Path,
    training_manifest_paths: Sequence[Path],
    validation_manifest_paths: Sequence[Path],
    independent_campaign_path: Path,
    output: Path,
) -> dict[str, object]:
    """Fit and freeze all validation-owned descriptive baselines before test access."""
    if (
        len(training_manifest_paths) != len(validation_manifest_paths)
        or not training_manifest_paths
    ):
        raise EvaluationContractError("baseline manifest pairs are incomplete")
    candidate, candidate_sha256 = _validated_candidate(config, architecture_candidate_path)
    independent_campaign, independent_campaign_sha256 = _validated_independent_campaign(
        config, independent_campaign_path
    )
    independent_plan, independent_plan_sha256 = _validated_independent_campaign_plan(
        config, Path(cast(str, independent_campaign["campaign_plan_path"]))
    )
    if independent_plan.get("candidate_sha256") != candidate_sha256:
        raise EvaluationContractError("independent campaign candidate lineage mismatch")
    supplied_pairs = [
        {
            "fold": load_episode_manifest(config, training_path).fold,
            "training_manifest_path": str(training_path.resolve()),
            "training_manifest_sha256": _sha256(training_path),
            "validation_manifest_path": str(validation_path.resolve()),
            "validation_manifest_sha256": _sha256(validation_path),
        }
        for training_path, validation_path in zip(
            training_manifest_paths, validation_manifest_paths, strict=True
        )
    ]
    if supplied_pairs != independent_plan.get("manifest_pairs"):
        raise EvaluationContractError("baseline manifests differ from independent campaign plan")
    campaign_artifacts = cast(list[dict[str, object]], independent_campaign["artifacts"])
    artifact_keys = [
        (cast(int, item["fold"]), cast(int, item["seed"])) for item in campaign_artifacts
    ]
    expected_artifact_keys = [
        (cast(int, pair["fold"]), seed)
        for pair in cast(list[dict[str, object]], independent_plan["manifest_pairs"])
        for seed in config.training.confirmation_seeds
    ]
    if artifact_keys != expected_artifact_keys:
        raise EvaluationContractError("independent campaign artifact order or set is invalid")
    independent = {
        key: _independent_ppo_baseline_contract(
            artifact,
            independent_campaign,
            independent_campaign_sha256,
            independent_plan,
            independent_plan_sha256,
        )
        for key, artifact in zip(artifact_keys, campaign_artifacts, strict=True)
    }
    folds: list[dict[str, object]] = []
    prior_fold = -1
    for training_path, validation_path in zip(
        training_manifest_paths, validation_manifest_paths, strict=True
    ):
        training = load_episode_manifest(config, training_path)
        validation = load_episode_manifest(config, validation_path)
        if (
            training.partition != "training"
            or validation.partition != "validation"
            or training.fold != validation.fold
            or training.fold <= prior_fold
        ):
            raise EvaluationContractError("baseline manifests must be ordered same-fold pairs")
        prior_fold = training.fold
        expected_independent = {
            (training.fold, seed) for seed in config.training.confirmation_seeds
        }
        if not expected_independent.issubset(independent):
            raise EvaluationContractError(
                f"independent PPO checkpoints are incomplete for fold {training.fold}"
            )
        estimator_bytes, input_lineage, hgb_contract = _hgb_freeze_material(
            config, training, validation
        )
        estimator_sha256 = hashlib.sha256(estimator_bytes).hexdigest()
        estimator_path = output / "estimators" / f"hgb-fold-{training.fold}-{estimator_sha256}.pkl"
        _atomic_bytes_no_overwrite(estimator_path, estimator_bytes)
        folds.append(
            {
                "fold": training.fold,
                "training_manifest_path": str(training_path.resolve()),
                "training_manifest_sha256": _sha256(training_path),
                "validation_manifest_path": str(validation_path.resolve()),
                "validation_manifest_sha256": _sha256(validation_path),
                **input_lineage,
                "historical_logistic": _historical_logistic_contract(config, training.fold),
                "hist_gradient_boosting": {
                    "path": str(estimator_path.resolve()),
                    "sha256": estimator_sha256,
                    **hgb_contract,
                },
                "independent_ppo": [
                    independent[(training.fold, seed)]
                    for seed in config.training.confirmation_seeds
                ],
            }
        )
    if list(independent) != expected_artifact_keys:
        raise EvaluationContractError("independent PPO checkpoint set has unexpected entries")
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "baseline-freeze-v1",
        "status": "complete",
        "config_sha256": config.digest,
        "source_sha256": config.upstream.source_digest,
        "dependency_lock_sha256": config.upstream.lock_digest,
        "architecture_candidate_path": str(architecture_candidate_path.resolve()),
        "architecture_candidate_sha256": candidate_sha256,
        "independent_campaign_path": str(independent_campaign_path.resolve()),
        "independent_campaign_sha256": independent_campaign_sha256,
        "evaluation_code_path": str(Path(__file__).resolve()),
        "evaluation_code_sha256": _sha256(Path(__file__)),
        "environment_code_path": str(Path(__file__).resolve().with_name("rl_environment.py")),
        "environment_code_sha256": _sha256(Path(__file__).resolve().with_name("rl_environment.py")),
        "baseline_code_path": str(Path(__file__).resolve().with_name("rl_baselines.py")),
        "baseline_code_sha256": _sha256(Path(__file__).resolve().with_name("rl_baselines.py")),
        "folds": folds,
        "test_accessed": False,
        "sealed_holdout_accessed": False,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    path = output / f"baseline-freeze-{digest}.json"
    _atomic_no_overwrite(path, payload)
    return {"baseline_freeze_path": str(path), "baseline_freeze_sha256": digest, **payload}


def _validated_baseline_freeze(config: RlConfig, path: Path) -> tuple[dict[str, object], str]:
    baseline = _load_object(path, "baseline freeze")
    digest = _sha256(path)
    if (
        path.name != f"baseline-freeze-{digest}.json"
        or baseline.get("schema_version") != 1
        or baseline.get("stage") != "baseline-freeze-v1"
        or baseline.get("status") != "complete"
        or baseline.get("config_sha256") != config.digest
        or baseline.get("source_sha256") != config.upstream.source_digest
        or baseline.get("dependency_lock_sha256") != config.upstream.lock_digest
        or baseline.get("test_accessed") is not False
        or baseline.get("sealed_holdout_accessed") is not False
    ):
        raise EvaluationContractError("baseline freeze provenance mismatch")
    candidate_path = baseline.get("architecture_candidate_path")
    if not isinstance(candidate_path, str) or _sha256(Path(candidate_path)) != baseline.get(
        "architecture_candidate_sha256"
    ):
        raise EvaluationContractError("baseline freeze architecture lineage mismatch")
    _candidate, validated_candidate_sha256 = _validated_candidate(config, Path(candidate_path))
    if validated_candidate_sha256 != baseline["architecture_candidate_sha256"]:
        raise EvaluationContractError("baseline freeze candidate validation mismatch")
    campaign_path = baseline.get("independent_campaign_path")
    if not isinstance(campaign_path, str):
        raise EvaluationContractError("baseline freeze independent campaign is missing")
    campaign, campaign_sha256 = _validated_independent_campaign(config, Path(campaign_path))
    if campaign_sha256 != baseline.get("independent_campaign_sha256"):
        raise EvaluationContractError("baseline freeze independent campaign changed")
    plan, plan_sha256 = _validated_independent_campaign_plan(
        config, Path(cast(str, campaign["campaign_plan_path"]))
    )
    folds = baseline.get("folds")
    if not isinstance(folds, list) or not folds:
        raise EvaluationContractError("baseline freeze folds are missing")
    frozen_pairs = cast(list[dict[str, object]], plan["manifest_pairs"])
    baseline_pairs = [
        {
            "fold": item.get("fold"),
            "training_manifest_path": item.get("training_manifest_path"),
            "training_manifest_sha256": item.get("training_manifest_sha256"),
            "validation_manifest_path": item.get("validation_manifest_path"),
            "validation_manifest_sha256": item.get("validation_manifest_sha256"),
        }
        for item in folds
        if isinstance(item, dict)
    ]
    if baseline_pairs != frozen_pairs:
        raise EvaluationContractError("baseline manifests differ from independent campaign plan")
    expected_code = {
        "evaluation_code_path": str(Path(__file__).resolve()),
        "environment_code_path": str(Path(__file__).resolve().with_name("rl_environment.py")),
        "baseline_code_path": str(Path(__file__).resolve().with_name("rl_baselines.py")),
    }
    if any(
        baseline.get(path_field) != expected_path
        or baseline.get(path_field.replace("_path", "_sha256")) != _sha256(Path(expected_path))
        for path_field, expected_path in expected_code.items()
    ):
        raise EvaluationContractError("baseline freeze source identity mismatch")
    campaign_artifacts = cast(list[dict[str, object]], campaign.get("artifacts"))
    expected_independent = [
        _independent_ppo_baseline_contract(artifact, campaign, campaign_sha256, plan, plan_sha256)
        for artifact in campaign_artifacts
    ]
    observed_independent: list[dict[str, object]] = []
    prior_fold = -1
    for item in folds:
        if not isinstance(item, dict) or type(item.get("fold")) is not int:
            raise EvaluationContractError("baseline freeze fold is invalid")
        fold = cast(int, item["fold"])
        if fold <= prior_fold:
            raise EvaluationContractError("baseline freeze folds are not ordered")
        prior_fold = fold
        for name in ("training_manifest", "validation_manifest"):
            reference = item.get(f"{name}_path")
            if not isinstance(reference, str) or _sha256(Path(reference)) != item.get(
                f"{name}_sha256"
            ):
                raise EvaluationContractError("baseline freeze schedule digest mismatch")
        training = load_episode_manifest(config, Path(cast(str, item["training_manifest_path"])))
        validation = load_episode_manifest(
            config, Path(cast(str, item["validation_manifest_path"]))
        )
        if (
            training.partition != "training"
            or validation.partition != "validation"
            or training.fold != fold
            or validation.fold != fold
        ):
            raise EvaluationContractError("baseline estimator input ownership mismatch")
        estimator_bytes, input_lineage, hgb_contract = _hgb_freeze_material(
            config, training, validation
        )
        if any(item.get(name) != identity for name, identity in input_lineage.items()):
            raise EvaluationContractError("baseline estimator input lineage mismatch")
        historical = item.get("historical_logistic")
        if historical != _historical_logistic_contract(config, fold):
            raise EvaluationContractError("historical baseline contract mismatch")
        hgb = item.get("hist_gradient_boosting")
        if not isinstance(hgb, dict) or not isinstance(hgb.get("path"), str):
            raise EvaluationContractError("HGB baseline contract is invalid")
        estimator_path = Path(cast(str, hgb["path"]))
        estimator_sha256 = hashlib.sha256(estimator_bytes).hexdigest()
        expected_hgb = {
            "path": str(estimator_path.resolve()),
            "sha256": estimator_sha256,
            **hgb_contract,
        }
        if (
            hgb != expected_hgb
            or estimator_path.name != f"hgb-fold-{fold}-{estimator_sha256}.pkl"
            or _sha256(estimator_path) != estimator_sha256
            or estimator_path.read_bytes() != estimator_bytes
        ):
            raise EvaluationContractError("HGB baseline recomputation mismatch")
        independent = item.get("independent_ppo")
        if not isinstance(independent, list) or not all(
            isinstance(entry, dict) for entry in independent
        ):
            raise EvaluationContractError("independent PPO baseline set is incomplete")
        observed_independent.extend(cast(list[dict[str, object]], independent))
    if observed_independent != expected_independent:
        raise EvaluationContractError(
            "independent PPO baseline differs from validated campaign artifacts"
        )
    return baseline, digest


def _validated_pre_promotion_serving_contract(
    config: RlConfig,
    serving: Mapping[str, object],
    serving_sha256: str,
    deployment_selector_path: Path,
    risk_shield_path: Path,
) -> dict[str, object]:
    """Validate the #11 selector and shared shield before any test metadata opens."""
    selector = _load_object(deployment_selector_path, "deployment checkpoint selection")
    selector_sha256 = _sha256(deployment_selector_path)
    artifacts = serving.get("seed_artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(item, dict) for item in artifacts):
        raise EvaluationContractError("serving artifacts are unavailable for deployment selection")
    eligible = sorted(
        (
            cast(dict[str, object], item)
            for item in artifacts
            if cast(dict[str, object], item).get("seed") == config.training.serving_seed
        ),
        key=lambda item: cast(int, item["fold"]),
    )
    folds = [item.get("fold") for item in eligible]
    if (
        not eligible
        or any(type(fold) is not int for fold in folds)
        or folds != list(range(cast(int, folds[0]), cast(int, folds[-1]) + 1))
    ):
        raise EvaluationContractError("deployment folds are incomplete or nonchronological")
    selected = eligible[-1]
    if (
        deployment_selector_path.name != f"deployment-checkpoint-selection-{selector_sha256}.json"
        or selector.get("schema_version") != 1
        or selector.get("stage") != "deployment-checkpoint-selection-v1"
        or selector.get("status") != "complete"
        or selector.get("selector") != "deployment-checkpoint-selector-v1"
        or selector.get("serving_seed") != config.training.serving_seed
        or selector.get("selected_fold") != selected.get("fold")
        or selector.get("serving_freeze_sha256") != serving_sha256
        or selector.get("checkpoint_bundle_path")
        != str(Path(cast(str, selected["checkpoint_bundle_path"])).resolve())
        or selector.get("checkpoint_bundle_sha256") != selected.get("checkpoint_bundle_sha256")
        or selector.get("config_sha256") != config.digest
        or selector.get("source_sha256") != config.upstream.source_digest
        or selector.get("dependency_lock_sha256") != config.upstream.lock_digest
        or selector.get("test_accessed") is not False
        or selector.get("sealed_holdout_accessed") is not False
    ):
        raise EvaluationContractError("deployment checkpoint selection is not authorized")

    shield = _load_object(risk_shield_path, "RiskShield contract")
    shield_sha256 = _sha256(risk_shield_path)
    if (
        risk_shield_path.name != f"risk-shield-contract-{shield_sha256}.json"
        or shield.get("schema_version") != 1
        or shield.get("stage") != "risk-shield-contract-v1"
        or shield.get("status") != "complete"
        or shield.get("version") != "RiskShieldV1"
        or shield.get("config_sha256") != config.digest
        or shield.get("source_sha256") != config.upstream.source_digest
        or shield.get("dependency_lock_sha256") != config.upstream.lock_digest
        or shield.get("consumers") != ["training", "evaluation", "serving", "parity", "benchmark"]
        or shield.get("test_accessed") is not False
        or shield.get("sealed_holdout_accessed") is not False
    ):
        raise EvaluationContractError("RiskShield contract is not authorized")
    references = _expected_risk_shield_identities(config)
    if any(shield.get(name) != reference for name, reference in references.items()):
        raise EvaluationContractError("RiskShield authority differs from approved identities")
    return {
        "deployment_selector": {
            "path": str(deployment_selector_path.resolve()),
            "sha256": selector_sha256,
            "selector": "deployment-checkpoint-selector-v1",
            "selected_fold": selected["fold"],
            "checkpoint_bundle_sha256": selected["checkpoint_bundle_sha256"],
        },
        "risk_shield": {
            "path": str(risk_shield_path.resolve()),
            "sha256": shield_sha256,
            "version": "RiskShieldV1",
            **references,
        },
    }


def write_evaluation_access_plan(
    config: RlConfig,
    serving_freeze_path: Path,
    baseline_freeze_path: Path,
    deployment_selector_path: Path,
    risk_shield_path: Path,
    output: Path,
    *,
    run_identity: str,
    created_at: str,
) -> dict[str, object]:
    """Authorize metadata-only test schedule construction before any test file opens."""
    if not run_identity:
        raise EvaluationContractError("evaluation access run identity is required")
    _timestamp(created_at, "created_at")
    serving, serving_sha256 = _validate_serving_freeze(config, serving_freeze_path)
    _baseline, baseline_sha256 = _validated_baseline_freeze(config, baseline_freeze_path)
    pre_promotion_serving = _validated_pre_promotion_serving_contract(
        config,
        serving,
        serving_sha256,
        deployment_selector_path,
        risk_shield_path,
    )
    repository = Path(__file__).resolve().parents[3]
    embedding_path = config.upstream.embedding_manifest_path
    corpus_path = config.upstream.corpus_manifest_path
    if not embedding_path.is_absolute():
        embedding_path = repository / embedding_path
    if not corpus_path.is_absolute():
        corpus_path = repository / corpus_path
    if (
        _sha256(embedding_path) != config.upstream.embedding_manifest_sha256
        or _sha256(corpus_path) != config.upstream.corpus_manifest_sha256
    ):
        raise EvaluationContractError("evaluation access upstream descriptor changed")
    embedding = _load_object(embedding_path, "embedding descriptor")
    corpus = _load_object(corpus_path, "corpus descriptor")
    downstream_path = config.upstream.downstream_config_path
    if not downstream_path.is_absolute():
        downstream_path = repository / downstream_path
    downstream = load_downstream_config(downstream_path)
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "evaluation-access-plan-v1",
        "status": "frozen_before_test_metadata",
        "created_at": created_at,
        "run_identity": run_identity,
        "config_sha256": config.digest,
        "source_sha256": config.upstream.source_digest,
        "dependency_lock_sha256": config.upstream.lock_digest,
        "serving_freeze_path": str(serving_freeze_path.resolve()),
        "serving_freeze_sha256": serving_sha256,
        "baseline_freeze_path": str(baseline_freeze_path.resolve()),
        "baseline_freeze_sha256": baseline_sha256,
        "pre_promotion_serving": pre_promotion_serving,
        "final_timesteps": serving["final_timesteps"],
        "test_descriptors": {
            "embedding_manifest": {
                "path": str(embedding_path.resolve()),
                "sha256": config.upstream.embedding_manifest_sha256,
                "outputs_sha256": hashlib.sha256(
                    _canonical_bytes(embedding.get("outputs"))
                ).hexdigest(),
            },
            "corpus_manifest": {
                "path": str(corpus_path.resolve()),
                "sha256": config.upstream.corpus_manifest_sha256,
                "outputs_sha256": hashlib.sha256(
                    _canonical_bytes(corpus.get("outputs"))
                ).hexdigest(),
            },
        },
        "metadata_columns": [
            "symbol",
            "decision_ts",
            *[f"{timeframe}_index" for timeframe in downstream.data.timeframes],
        ],
        "forbidden_scheduler_inputs": [
            "feature_shards",
            "labels",
            "ohlc_outcomes",
            "policy_outputs",
        ],
        "schedule_construction": {
            "algorithm": "chronological_greedy_nonoverlap_v1",
            "overlap_coverage_algorithm": "all_rejected_overlapping_starts_v1",
            "ticker_local": True,
            "scheduler_code_path": str(Path(__file__).resolve().with_name("rl_episodes.py")),
            "scheduler_code_sha256": _sha256(Path(__file__).resolve().with_name("rl_episodes.py")),
        },
        "pair_key": ["fold", "calendar_block", "episode_id", "ticker", "profile", "seed"],
        "thresholds": {
            "minimum_raw_pass_rate": config.evaluation.minimum_raw_pass_rate,
            "minimum_seed_raw_pass_rate": config.evaluation.minimum_seed_raw_pass_rate,
            "minimum_pass_rate_lcb_95": config.evaluation.minimum_pass_rate_lcb_95,
            "maximum_observed_blows": config.evaluation.maximum_observed_blows,
            "maximum_blow_rate_ucb_95": config.evaluation.maximum_blow_rate_ucb_95,
            "minimum_chronological_attempts": config.evaluation.minimum_chronological_attempts,
        },
        "stress_names": sorted({_HEADLINE, *_REQUIRED_STRESSES}),
        "bootstrap": {
            "replicates": 100_000,
            "rng": "numpy.PCG64",
            "quantile": 0.05,
            "quantile_method": "inverted_cdf",
        },
        "authorized_identities": {
            "config": {"sha256": config.digest},
            "source": {"sha256": config.upstream.source_digest},
            "lock": {"sha256": config.upstream.lock_digest},
            "corpus": {"sha256": config.upstream.corpus_manifest_sha256},
            "embedding": {"sha256": config.upstream.embedding_manifest_sha256},
        },
        "test_metadata_accessed": False,
        "test_features_accessed": False,
        "sealed_holdout_accessed": False,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    path = output / f"evaluation-access-plan-{digest}.json"
    _atomic_no_overwrite(path, payload)
    return {"access_plan_path": str(path), "access_plan_sha256": digest, **payload}


def _validated_evaluation_access_plan(
    config: RlConfig, path: Path
) -> tuple[dict[str, object], str]:
    plan = _load_object(path, "evaluation access plan")
    digest = _sha256(path)
    if (
        path.name != f"evaluation-access-plan-{digest}.json"
        or plan.get("stage") != "evaluation-access-plan-v1"
        or plan.get("status") != "frozen_before_test_metadata"
        or plan.get("config_sha256") != config.digest
        or plan.get("source_sha256") != config.upstream.source_digest
        or plan.get("dependency_lock_sha256") != config.upstream.lock_digest
        or plan.get("test_metadata_accessed") is not False
        or plan.get("test_features_accessed") is not False
        or plan.get("sealed_holdout_accessed") is not False
    ):
        raise EvaluationContractError("evaluation access plan provenance mismatch")
    serving_path = plan.get("serving_freeze_path")
    baseline_path = plan.get("baseline_freeze_path")
    if not isinstance(serving_path, str) or not isinstance(baseline_path, str):
        raise EvaluationContractError("evaluation access parent references are missing")
    serving, serving_sha256 = _validate_serving_freeze(config, Path(serving_path))
    _baseline, baseline_sha256 = _validated_baseline_freeze(config, Path(baseline_path))
    if (
        serving_sha256 != plan.get("serving_freeze_sha256")
        or baseline_sha256 != plan.get("baseline_freeze_sha256")
        or serving.get("final_timesteps") != plan.get("final_timesteps")
    ):
        raise EvaluationContractError("evaluation access parent identity mismatch")
    pre_promotion = plan.get("pre_promotion_serving")
    if not isinstance(pre_promotion, dict):
        raise EvaluationContractError("pre-promotion serving contract is missing")
    selector = pre_promotion.get("deployment_selector")
    shield = pre_promotion.get("risk_shield")
    if (
        not isinstance(selector, dict)
        or not isinstance(selector.get("path"), str)
        or not isinstance(shield, dict)
        or not isinstance(shield.get("path"), str)
    ):
        raise EvaluationContractError("pre-promotion serving references are missing")
    expected_pre_promotion = _validated_pre_promotion_serving_contract(
        config,
        serving,
        serving_sha256,
        Path(cast(str, selector["path"])),
        Path(cast(str, shield["path"])),
    )
    if pre_promotion != expected_pre_promotion:
        raise EvaluationContractError("pre-promotion serving contract changed")
    descriptors = plan.get("test_descriptors")
    if not isinstance(descriptors, dict):
        raise EvaluationContractError("evaluation access descriptors are missing")
    for descriptor in descriptors.values():
        if (
            not isinstance(descriptor, dict)
            or not isinstance(descriptor.get("path"), str)
            or _sha256(Path(cast(str, descriptor["path"]))) != descriptor.get("sha256")
        ):
            raise EvaluationContractError("evaluation access descriptor changed")
    construction = plan.get("schedule_construction")
    if (
        not isinstance(construction, dict)
        or not isinstance(construction.get("scheduler_code_path"), str)
        or _sha256(Path(cast(str, construction["scheduler_code_path"])))
        != construction.get("scheduler_code_sha256")
    ):
        raise EvaluationContractError("evaluation access scheduler changed")
    return plan, digest


def _rederive_test_schedule_from_authorized_metadata(
    config: RlConfig, schedule: Mapping[str, object]
) -> dict[str, object]:
    """Rebuild the complete test schedule from its frozen metadata authority."""
    proof = schedule.get("schedule_proof")
    fold = schedule.get("fold")
    if (
        not isinstance(proof, dict)
        or not isinstance(proof.get("access_plan_path"), str)
        or not isinstance(proof.get("access_plan_sha256"), str)
        or type(fold) is not int
    ):
        raise EvaluationContractError("test schedule metadata authority is missing")
    access_plan_path = Path(cast(str, proof["access_plan_path"]))
    _access_plan, access_plan_sha256 = _validated_evaluation_access_plan(config, access_plan_path)
    if access_plan_sha256 != proof["access_plan_sha256"]:
        raise EvaluationContractError("test schedule metadata authority changed")
    from mantis_v2.rl_episodes import EpisodeContractError, build_episode_manifest

    try:
        return build_episode_manifest(
            config,
            fold_number=fold,
            partition_name="test",
            episode_count=1,
            evaluation=True,
            access_plan_path=access_plan_path,
            publish=False,
        )
    except EpisodeContractError as exc:
        raise EvaluationContractError(
            "test schedule cannot be rederived from authorized metadata"
        ) from exc


def _validated_test_schedule_reference(
    config: RlConfig, path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    schedule = _load_object(path, "test schedule")
    digest = _sha256(path)
    partition = schedule.get("partition")
    configured_identities = schedule.get("identities")
    configured = (
        configured_identities.get("config") if isinstance(configured_identities, dict) else None
    )
    episodes = schedule.get("episodes")
    if (
        path.name != f"evaluation-schedule-{digest}.json"
        or schedule.get("schema_version") != 1
        or schedule.get("stage") != "rl-episode-schedule"
        or type(schedule.get("fold")) is not int
        or not isinstance(partition, dict)
        or partition.get("name") != "test"
        or not isinstance(partition.get("start"), str)
        or not isinstance(partition.get("end"), str)
        or _timestamp(partition["start"], "test partition start")
        >= _timestamp(partition["end"], "test partition end")
        or _timestamp(partition["end"], "test partition end")
        >= config.evaluation.sealed_holdout_start
        or schedule.get("schedule_mode") != "chronological_greedy_nonoverlap_v1"
        or schedule.get("overlapping_starts") is not False
        or not isinstance(configured, dict)
        or configured.get("sha256") != config.digest
        or not isinstance(episodes, list)
        or not episodes
    ):
        raise EvaluationContractError("test schedule provenance mismatch")
    partition_start = _timestamp(partition["start"], "test partition start")
    partition_end = _timestamp(partition["end"], "test partition end")
    selected_identities: list[dict[str, object]] = []
    observed_numbers: set[int] = set()
    prior_terminal: dict[tuple[str, str], datetime] = {}
    for raw in episodes:
        if not isinstance(raw, dict):
            raise EvaluationContractError("test schedule episode is invalid")
        number = raw.get("number")
        ticker = raw.get("ticker")
        profile = raw.get("profile")
        if (
            type(number) is not int
            or number in observed_numbers
            or ticker not in TICKERS
            or profile not in PROFILES
            or (ticker == "ZB" and profile != "one_mini")
        ):
            raise EvaluationContractError("test schedule episode identity is invalid")
        observed_numbers.add(number)
        lookback = _timestamp(raw.get("lookback_start"), "episode lookback_start")
        decision = _timestamp(raw.get("decision_start"), "episode decision_start")
        exit_end = _timestamp(raw.get("exit_end"), "episode exit_end")
        terminal = _timestamp(raw.get("terminal_end"), "episode terminal_end")
        if not (partition_start <= lookback <= decision <= exit_end <= terminal < partition_end):
            raise EvaluationContractError("test episode crosses its owning partition")
        stream = (cast(str, ticker), cast(str, profile))
        if stream in prior_terminal and decision <= prior_terminal[stream]:
            raise EvaluationContractError("test schedule episodes overlap")
        prior_terminal[stream] = terminal
        selected_identities.append(
            {
                "number": number,
                "ticker": ticker,
                "profile": profile,
                "lookback_start": raw["lookback_start"],
                "decision_start": raw["decision_start"],
                "exit_end": raw["exit_end"],
                "terminal_end": raw["terminal_end"],
            }
        )
    stress_coverage = schedule.get("stress_coverage")
    overlap = (
        stress_coverage.get("overlapping_starts") if isinstance(stress_coverage, dict) else None
    )
    overlap_episodes = overlap.get("episodes") if isinstance(overlap, dict) else None
    if (
        not isinstance(overlap, dict)
        or overlap.get("stage") != "overlapping-start-coverage-v1"
        or overlap.get("promotion_denominator") is not False
        or overlap.get("overlapping_starts") is not True
        or not isinstance(overlap_episodes, list)
        or not overlap_episodes
        or overlap.get("episode_count") != len(overlap_episodes)
        or overlap.get("identity_sha256")
        != hashlib.sha256(_canonical_bytes(overlap_episodes)).hexdigest()
    ):
        raise EvaluationContractError("overlapping-start stress coverage is invalid")
    headline_intervals: dict[tuple[str, str], list[tuple[datetime, datetime]]] = defaultdict(list)
    for raw in episodes:
        assert isinstance(raw, dict)
        headline_intervals[(cast(str, raw["ticker"]), cast(str, raw["profile"]))].append(
            (
                _timestamp(raw["decision_start"], "headline decision_start"),
                _timestamp(raw["terminal_end"], "headline terminal_end"),
            )
        )
    overlap_numbers: set[int] = set()
    for raw in overlap_episodes:
        if not isinstance(raw, dict):
            raise EvaluationContractError("overlapping-start episode is invalid")
        number = raw.get("number")
        ticker = raw.get("ticker")
        profile = raw.get("profile")
        decision = _timestamp(raw.get("decision_start"), "overlap decision_start")
        terminal = _timestamp(raw.get("terminal_end"), "overlap terminal_end")
        lookback = _timestamp(raw.get("lookback_start"), "overlap lookback_start")
        exit_end = _timestamp(raw.get("exit_end"), "overlap exit_end")
        stream = (cast(str, ticker), cast(str, profile))
        if (
            type(number) is not int
            or number in overlap_numbers
            or ticker not in TICKERS
            or profile not in PROFILES
            or (ticker == "ZB" and profile != "one_mini")
            or not (partition_start <= lookback <= decision <= exit_end <= terminal < partition_end)
            or not any(
                decision <= headline_terminal and terminal >= headline_decision
                for headline_decision, headline_terminal in headline_intervals.get(stream, [])
            )
        ):
            raise EvaluationContractError("overlapping-start episode is not valid coverage")
        overlap_numbers.add(number)
    proof = schedule.get("schedule_proof")
    identity_sha256 = hashlib.sha256(_canonical_bytes(selected_identities)).hexdigest()
    if (
        not isinstance(proof, dict)
        or proof.get("algorithm") != "chronological_greedy_nonoverlap_v1"
        or proof.get("selected_identity_sha256") != identity_sha256
        or not isinstance(proof.get("builder_code_path"), str)
        or _sha256(Path(cast(str, proof["builder_code_path"]))) != proof.get("builder_code_sha256")
    ):
        raise EvaluationContractError("test schedule greedy construction proof mismatch")
    canonical = _rederive_test_schedule_from_authorized_metadata(config, schedule)
    if canonical.get("schedule_sha256") != digest:
        raise EvaluationContractError("test schedule differs from canonical authorized metadata")
    return schedule, {"path": str(path.resolve()), "sha256": digest}


def _evaluation_contract_fields(
    *,
    serving_seed: int,
    checkpoints: list[dict[str, object]],
    profile_counts: dict[str, int],
    attempt_identity_sha256: str,
    expected_ledger: list[dict[str, object]],
) -> dict[str, object]:
    pair_key = ["fold", "calendar_block", "episode_id", "ticker", "profile", "seed"]
    stress_definitions: dict[str, object] = {
        "primary": {
            "adverse_slippage_ticks_per_side": 1.0,
            "fee": "pinned_product_round_turn_snapshot",
            "numeric_promotion_gates": True,
        },
        "two_tick": {
            "adverse_slippage_ticks_per_side": 2.0,
            "fee": "pinned_product_round_turn_snapshot",
            "numeric_promotion_gates": True,
        },
        "fee_2x": {
            "adverse_slippage_ticks_per_side": 1.0,
            "fee_multiplier": 2.0,
            "numeric_promotion_gates": False,
        },
        "gap_adverse_tick": {
            "base": "primary",
            "extra_adverse_ticks": 1.0,
            "condition": "historical_next_open_gaps_adversely_from_decision_close",
            "numeric_promotion_gates": False,
        },
        "latency_1bar": {
            "base": "primary",
            "entry_defer_eligible_bars": 1,
            "fill": "deferred_bar_next_open",
            "cancel_boundaries": ["session_discontinuity", "partition_end", "terminal"],
            "numeric_promotion_gates": False,
        },
        "missed_fill_10pct": {
            "base": "primary",
            "cancel_without_carry": "every_tenth_eligible_entry_order",
            "offset_derivation": (
                "first_u64(SHA256(evaluation_plan_payload_sha256 + pair_key + "
                "':missed-fill-v1')) mod 10"
            ),
            "numeric_promotion_gates": False,
        },
        "same_bar_adverse": {
            "base": "primary",
            "ambiguous_stop_target_resolution": "stop_first",
            "record_ambiguity_count": True,
            "numeric_promotion_gates": False,
        },
        "overlapping_starts": {
            "base": "primary",
            "schedule": "all_rejected_overlapping_starts_v1",
            "promotion_denominator": False,
            "numeric_promotion_gates": False,
        },
    }
    return {
        "checkpoint_bundles": checkpoints,
        "test_metadata_accessed": True,
        "test_features_accessed": False,
        "schedule_construction": "deterministic_chronological_greedy_nonoverlap_v1",
        "serving_seed": serving_seed,
        "attempt_identity": ["fold", "episode_id", "ticker", "profile"],
        "minimum_attempts_per_profile": 300,
        "frozen_attempt_counts": profile_counts,
        "frozen_attempt_identity_sha256": attempt_identity_sha256,
        "profiles": {
            "one_mini": "required",
            "ten_micros": "required",
            "ZB:ten_micros": "not_applicable",
        },
        "policies": sorted(_POLICIES),
        "baseline_definitions": {
            "take_all": "enter_every_legal_masked_opportunity",
            "matched_random_take": {
                "participation_owner": "candidate_accepted_entry_count_per_seed_attempt",
                "sampling": "sha256_rank_without_replacement",
                "rank_derivation": (
                    "SHA256(evaluation_plan_payload_sha256 + canonical_pair_key + "
                    "opportunity_identity + ':matched-random-v1')"
                ),
                "fallback": None,
            },
            "historical_rejected_logistic_head": "baseline_freeze_exact_bytes",
            "hist_gradient_boosting": "baseline_freeze_training_fit_validation_threshold",
            "independent_ticker_ppo": "baseline_freeze_exact_checkpoint_bytes",
        },
        "stress_definitions": stress_definitions,
        "numeric_stresses": sorted(_NUMERIC_STRESSES),
        "report_only_stresses": sorted(_SENSITIVITY_STRESSES),
        "thresholds": {
            "minimum_raw_pass_rate": 0.60,
            "minimum_seed_raw_pass_rate_exclusive": 0.50,
            "minimum_pass_rate_lcb_95": 0.50,
            "maximum_observed_blows": 0,
            "maximum_blow_rate_ucb_95": 0.01,
            "minimum_complete_calendar_blocks": 20,
        },
        "wilson": {
            "method": "uncorrected_one_sided_wilson_score",
            "z": 1.6448536269514715,
            "continuity_correction": False,
        },
        "bootstrap": {
            "pair_key": pair_key,
            "pair_key_serialization": "canonical_json_array_utf8",
            "replicates": 100_000,
            "rng": "numpy.PCG64",
            "seed_derivation": (
                "first_u64_be(SHA256(evaluation_plan_payload_sha256 + ':test-market-bootstrap-v1'))"
            ),
            "quantile": 0.05,
            "quantile_method": "inverted_cdf",
            "block": "fixed_exchange_calendar_week",
            "synchronized_across_all_views": True,
            "index_matrix_identity": "sha256_uint32_c_order_replicates_by_ordered_blocks",
            "seed_assignments": 1_024,
        },
        "required_views": {
            "numeric": ["pooled", "one_mini", "ten_micros"],
            "transfer": "pooled_and_all_13_supported_ticker_profile_cells",
            "metrics": ["pooled", "fold", "ticker", "profile", "regime_block", "seed", "cell"],
        },
        "artifact_schemas": [
            "baseline-freeze-v1",
            "evaluation-attempt-v1",
            "evaluation-report-v1",
            "promotion-verdict-v1",
        ],
        "expected_append_only_ledger": expected_ledger,
    }


def write_evaluation_plan(
    config: RlConfig,
    access_plan_path: Path,
    test_manifest_paths: Sequence[Path],
    output: Path,
    *,
    run_identity: str,
    created_at: str,
) -> dict[str, object]:
    """Freeze the complete preregistered test contract before reading test outcomes."""
    if not run_identity:
        raise EvaluationContractError("evaluation plan run identity or creation time is invalid")
    _timestamp(created_at, "created_at")
    access_plan, access_plan_sha256 = _validated_evaluation_access_plan(config, access_plan_path)
    serving_freeze_path = Path(cast(str, access_plan["serving_freeze_path"]))
    baseline_freeze_path = Path(cast(str, access_plan["baseline_freeze_path"]))
    serving, serving_sha256 = _validate_serving_freeze(config, serving_freeze_path)
    _baseline, baseline_sha256 = _validated_baseline_freeze(config, baseline_freeze_path)
    if not test_manifest_paths:
        raise EvaluationContractError("test manifests are missing")
    schedules = [_validated_test_schedule_reference(config, path) for path in test_manifest_paths]
    if any(
        cast(dict[str, object], schedule[0]["schedule_proof"]).get("access_plan_sha256")
        != access_plan_sha256
        for schedule in schedules
    ):
        raise EvaluationContractError("test schedule was not authorized by the access plan")
    folds = [cast(int, schedule[0]["fold"]) for schedule in schedules]
    if folds != sorted(set(folds)):
        raise EvaluationContractError("test manifests must have unique ordered folds")
    attempt_identities = {
        (
            cast(int, schedule["fold"]),
            episode["number"],
            episode["ticker"],
            episode["profile"],
        )
        for schedule, _reference in schedules
        for episode in cast(list[dict[str, object]], schedule["episodes"])
    }
    profile_counts = {
        profile: sum(identity[3] == profile for identity in attempt_identities)
        for profile in PROFILES
    }
    if any(count < 300 for count in profile_counts.values()):
        raise EvaluationContractError("evaluation plan requires 300 unique attempts per profile")
    observed_cells = {(cast(str, value[2]), cast(str, value[3])) for value in attempt_identities}
    required_cells = {
        (ticker, profile)
        for ticker in TICKERS
        for profile in (("one_mini",) if ticker == "ZB" else PROFILES)
    }
    if observed_cells != required_cells:
        raise EvaluationContractError("evaluation plan ticker/profile cells are incomplete")
    artifacts = cast(list[dict[str, object]], serving["seed_artifacts"])
    checkpoints = [
        {
            "fold": item["fold"],
            "seed": item["seed"],
            "checkpoint_bundle_path": str(
                Path(cast(str, item["checkpoint_bundle_path"])).resolve()
            ),
            "checkpoint_bundle_sha256": item["checkpoint_bundle_sha256"],
        }
        for item in sorted(artifacts, key=lambda value: (value["fold"], value["seed"]))
    ]
    expected_ledger = [
        {
            "ordinal": ordinal,
            "fold": fold,
            "seed": seed,
            "policy": policy,
            "stress": stress,
        }
        for ordinal, (fold, seed, policy, stress) in enumerate(
            (
                (fold, seed, policy, stress)
                for fold in folds
                for seed in config.training.confirmation_seeds
                for policy in sorted(_POLICIES)
                for stress in sorted(_REQUIRED_STRESSES)
            ),
            start=1,
        )
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "evaluation-plan-v1",
        "status": "frozen_before_test",
        "created_at": created_at,
        "run_identity": run_identity,
        "config_sha256": config.digest,
        "source_sha256": config.upstream.source_digest,
        "dependency_lock_sha256": config.upstream.lock_digest,
        "evaluation_code_path": str(Path(__file__).resolve()),
        "evaluation_code_sha256": _sha256(Path(__file__)),
        "environment_code_path": str(Path(__file__).resolve().with_name("rl_environment.py")),
        "environment_code_sha256": _sha256(Path(__file__).resolve().with_name("rl_environment.py")),
        "access_plan_path": str(access_plan_path.resolve()),
        "access_plan_sha256": access_plan_sha256,
        "serving_freeze_path": str(serving_freeze_path.resolve()),
        "serving_freeze_sha256": serving_sha256,
        "baseline_freeze_path": str(baseline_freeze_path.resolve()),
        "baseline_freeze_sha256": baseline_sha256,
        "pre_promotion_serving": access_plan["pre_promotion_serving"],
        "test_schedules": [reference for _schedule, reference in schedules],
        **_evaluation_contract_fields(
            serving_seed=cast(int, serving["serving_seed"]),
            checkpoints=checkpoints,
            profile_counts=profile_counts,
            attempt_identity_sha256=hashlib.sha256(
                _canonical_bytes(
                    sorted(attempt_identities, key=lambda value: tuple(map(str, value)))
                )
            ).hexdigest(),
            expected_ledger=expected_ledger,
        ),
        "test_accessed": False,
        "sealed_holdout_accessed": False,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    path = output / f"evaluation-plan-{digest}.json"
    _atomic_no_overwrite(path, payload)
    return {
        "evaluation_plan_path": str(path),
        "evaluation_plan_sha256": digest,
        **payload,
    }


def _attempt_slice(row: Mapping[str, object]) -> dict[str, object]:
    return {field: row[field] for field in _ATTEMPT_FIELDS}


def _group_summaries(
    rows: Sequence[Mapping[str, object]], *, allow_overlap: bool = False
) -> dict[str, object]:
    dimensions = {
        "fold": ("fold",),
        "ticker": ("ticker",),
        "profile": ("profile",),
        "regime_block": ("regime_block",),
        "seed": ("seed",),
        "cell": ("fold", "ticker", "profile", "regime_block", "seed"),
    }
    result: dict[str, object] = {
        "pooled": summarize_attempts(
            [_attempt_slice(row) for row in rows], allow_overlap=allow_overlap
        )
    }
    for name, fields in dimensions.items():
        grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[tuple(row[field] for field in fields)].append(row)
        result[name] = [
            {
                **{field: key[index] for index, field in enumerate(fields)},
                "summary": summarize_attempts(
                    [_attempt_slice(row) for row in grouped[key]],
                    allow_overlap=allow_overlap,
                ),
            }
            for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item))
        ]
    return result


def _effect_rows(
    rows: Sequence[Mapping[str, object]], policy: str, stress: str
) -> list[dict[str, object]]:
    return [
        {
            "fold": row["fold"],
            "seed": row["seed"],
            "ticker": row["ticker"],
            "profile": row["profile"],
            "regime_block": row["regime_block"],
            "calendar_block": row["calendar_block"],
            "episode_id": row["episode_id"],
            "passed": row["status"] == "PASS",
        }
        for row in rows
        if row["policy"] == policy and row["stress"] == stress
    ]


def _point_and_lcb(
    rows: Sequence[Mapping[str, object]],
    candidate_policy: str,
    baseline_policy: str,
    stress: str,
    evaluation_plan_payload_sha256: str,
    expected_seeds: Sequence[int],
    ordered_block_ids: Sequence[object],
    synchronized_indices: np.ndarray,
) -> dict[str, object]:
    return paired_calendar_block_lcb(
        _effect_rows(rows, candidate_policy, stress),
        _effect_rows(rows, baseline_policy, stress),
        evaluation_plan_payload_sha256=evaluation_plan_payload_sha256,
        expected_seeds=expected_seeds,
        ordered_block_ids=ordered_block_ids,
        synchronized_indices=synchronized_indices,
    )


def _gate_values(summary: Mapping[str, object]) -> dict[str, object]:
    rates = cast(Mapping[str, object], summary["rates"])
    outcomes = cast(Mapping[str, object], summary["outcomes"])
    return {
        "attempts": summary["attempt_count"],
        "pass_rate": rates["pass"],
        "pass_lcb": rates["pass_wilson_lcb_95"],
        "blows": outcomes["BLOW"],
        "blow_ucb": rates["blow_wilson_ucb_95"],
    }


def _validate_serving_freeze(config: RlConfig, path: Path) -> tuple[dict[str, object], str]:
    serving = _load_object(path, "serving freeze")
    digest = _sha256(path)
    if (
        path.name != f"serving-freeze-{digest}.json"
        or serving.get("stage") != "serving-freeze-v1"
        or serving.get("status") != "complete"
        or serving.get("selected_variant") != "shared_ticker_value"
        or serving.get("serving_seed") != config.training.serving_seed
        or serving.get("serving_seed_frozen_before_test") is not True
        or serving.get("test_accessed") is not False
        or serving.get("sealed_holdout_accessed") is not False
    ):
        raise EvaluationContractError("serving freeze provenance mismatch")
    candidate_path = serving.get("candidate_path")
    if not isinstance(candidate_path, str) or _sha256(Path(candidate_path)) != serving.get(
        "candidate_sha256"
    ):
        raise EvaluationContractError("serving freeze candidate mismatch")
    candidate = _load_object(Path(candidate_path), "candidate freeze")
    if (
        candidate.get("config_sha256") != config.digest
        or candidate.get("stage") != "candidate-freeze-v1"
        or candidate.get("status") != "qualified"
        or candidate.get("test_accessed") is not False
        or candidate.get("sealed_holdout_accessed") is not False
    ):
        raise EvaluationContractError("candidate freeze provenance mismatch")
    _validated_candidate(config, Path(candidate_path))
    decision_path = serving.get("continuation_decision_path")
    decision_sha256 = serving.get("continuation_decision_sha256")
    if not isinstance(decision_path, str) or _sha256(Path(decision_path)) != decision_sha256:
        raise EvaluationContractError("serving freeze continuation decision mismatch")
    decision = _load_object(Path(decision_path), "serving continuation decision")
    if (
        decision.get("stage") != "rl-continuation-decision-v1"
        or decision.get("candidate_sha256") != serving.get("candidate_sha256")
        or decision.get("final_timesteps") != serving.get("final_timesteps")
        or decision.get("test_accessed") is not False
        or decision.get("sealed_holdout_accessed") is not False
    ):
        raise EvaluationContractError("serving continuation decision provenance mismatch")
    artifacts = serving.get("seed_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise EvaluationContractError("serving seed artifacts are missing")
    observed = {
        (item.get("fold"), item.get("seed")) for item in artifacts if isinstance(item, dict)
    }
    candidate_folds = candidate.get("folds")
    if not isinstance(candidate_folds, list):
        raise EvaluationContractError("candidate freeze fold identities are missing")
    folds = set(candidate_folds)
    expected = {(fold, seed) for fold in folds for seed in config.training.confirmation_seeds}
    if observed != expected:
        raise EvaluationContractError("serving seed artifacts are incomplete")
    for item in artifacts:
        assert isinstance(item, dict)
        for prefix in ("training_manifest", "checkpoint_bundle", "validation_report"):
            reference = item.get(f"{prefix}_path")
            if not isinstance(reference, str) or _sha256(Path(reference)) != item.get(
                f"{prefix}_sha256"
            ):
                raise EvaluationContractError("serving seed artifact digest mismatch")
        bundle_path = Path(cast(str, item["checkpoint_bundle_path"]))
        bundle = _load_object(bundle_path, "serving checkpoint bundle")
        checkpoint_path = bundle_path.parent / "checkpoint.pt"
        checkpoint_sha256 = _sha256(checkpoint_path)
        if (
            bundle.get("checkpoint_sha256") != checkpoint_sha256
            or item.get("artifact_sha256") != checkpoint_sha256
        ):
            raise EvaluationContractError("serving checkpoint bytes differ from frozen artifact")
    return serving, digest


def _validated_evaluation_plan(config: RlConfig, path: Path) -> tuple[dict[str, object], str]:
    plan = _load_object(path, "evaluation plan")
    digest = _sha256(path)
    if (
        path.name != f"evaluation-plan-{digest}.json"
        or plan.get("schema_version") != 1
        or plan.get("stage") != "evaluation-plan-v1"
        or plan.get("status") != "frozen_before_test"
        or plan.get("config_sha256") != config.digest
        or plan.get("source_sha256") != config.upstream.source_digest
        or plan.get("dependency_lock_sha256") != config.upstream.lock_digest
        or not isinstance(plan.get("evaluation_code_path"), str)
        or _sha256(Path(cast(str, plan["evaluation_code_path"])))
        != plan.get("evaluation_code_sha256")
        or not isinstance(plan.get("environment_code_path"), str)
        or _sha256(Path(cast(str, plan["environment_code_path"])))
        != plan.get("environment_code_sha256")
        or plan.get("test_accessed") is not False
        or plan.get("sealed_holdout_accessed") is not False
    ):
        raise EvaluationContractError("evaluation plan provenance mismatch")
    if not isinstance(plan.get("run_identity"), str) or not cast(str, plan["run_identity"]):
        raise EvaluationContractError("evaluation plan run identity is invalid")
    _timestamp(plan.get("created_at"), "evaluation plan created_at")
    serving_path = plan.get("serving_freeze_path")
    baseline_path = plan.get("baseline_freeze_path")
    access_path = plan.get("access_plan_path")
    if (
        not isinstance(serving_path, str)
        or not isinstance(baseline_path, str)
        or not isinstance(access_path, str)
    ):
        raise EvaluationContractError("evaluation plan artifact graph is incomplete")
    access, access_sha256 = _validated_evaluation_access_plan(config, Path(access_path))
    serving, serving_sha256 = _validate_serving_freeze(config, Path(serving_path))
    _baseline, baseline_sha256 = _validated_baseline_freeze(config, Path(baseline_path))
    if (
        access_sha256 != plan.get("access_plan_sha256")
        or access.get("serving_freeze_sha256") != serving_sha256
        or access.get("baseline_freeze_sha256") != baseline_sha256
        or access.get("pre_promotion_serving") != plan.get("pre_promotion_serving")
        or serving_sha256 != plan.get("serving_freeze_sha256")
        or baseline_sha256 != plan.get("baseline_freeze_sha256")
    ):
        raise EvaluationContractError("evaluation plan parent artifact mismatch")
    references = plan.get("test_schedules")
    if not isinstance(references, list) or not references:
        raise EvaluationContractError("evaluation plan test schedules are missing")
    folds: list[int] = []
    for reference in references:
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            raise EvaluationContractError("evaluation plan test schedule reference is invalid")
        schedule, verified = _validated_test_schedule_reference(
            config, Path(cast(str, reference["path"]))
        )
        proof = schedule.get("schedule_proof")
        if (
            verified != reference
            or not isinstance(proof, dict)
            or proof.get("access_plan_sha256") != access_sha256
            or proof.get("test_metadata_accessed") is not True
            or proof.get("test_features_accessed") is not False
        ):
            raise EvaluationContractError("evaluation plan test schedule digest mismatch")
        folds.append(cast(int, schedule["fold"]))
    if folds != sorted(set(folds)):
        raise EvaluationContractError("evaluation plan folds are not unique and ordered")
    attempt_identities = {
        (
            cast(int, schedule["fold"]),
            episode["number"],
            episode["ticker"],
            episode["profile"],
        )
        for reference in references
        for schedule in [_load_object(Path(cast(str, reference["path"])), "test schedule")]
        for episode in cast(list[dict[str, object]], schedule["episodes"])
    }
    counts = {
        profile: sum(identity[3] == profile for identity in attempt_identities)
        for profile in PROFILES
    }
    identities_sha256 = hashlib.sha256(
        _canonical_bytes(sorted(attempt_identities, key=lambda value: tuple(map(str, value))))
    ).hexdigest()
    observed_cells = {(cast(str, value[2]), cast(str, value[3])) for value in attempt_identities}
    required_cells = {
        (ticker, profile)
        for ticker in TICKERS
        for profile in (("one_mini",) if ticker == "ZB" else PROFILES)
    }
    if (
        counts != plan.get("frozen_attempt_counts")
        or any(count < 300 for count in counts.values())
        or identities_sha256 != plan.get("frozen_attempt_identity_sha256")
        or observed_cells != required_cells
    ):
        raise EvaluationContractError("evaluation plan frozen attempt identity mismatch")
    expected = [
        {
            "ordinal": ordinal,
            "fold": fold,
            "seed": seed,
            "policy": policy,
            "stress": stress,
        }
        for ordinal, (fold, seed, policy, stress) in enumerate(
            (
                (fold, seed, policy, stress)
                for fold in folds
                for seed in config.training.confirmation_seeds
                for policy in sorted(_POLICIES)
                for stress in sorted(_REQUIRED_STRESSES)
            ),
            start=1,
        )
    ]
    artifacts = cast(list[dict[str, object]], serving["seed_artifacts"])
    checkpoints = [
        {
            "fold": item["fold"],
            "seed": item["seed"],
            "checkpoint_bundle_path": str(
                Path(cast(str, item["checkpoint_bundle_path"])).resolve()
            ),
            "checkpoint_bundle_sha256": item["checkpoint_bundle_sha256"],
        }
        for item in sorted(artifacts, key=lambda value: (value["fold"], value["seed"]))
    ]
    expected_contract = _evaluation_contract_fields(
        serving_seed=cast(int, serving["serving_seed"]),
        checkpoints=checkpoints,
        profile_counts=counts,
        attempt_identity_sha256=identities_sha256,
        expected_ledger=expected,
    )
    if any(plan.get(field) != value for field, value in expected_contract.items()):
        raise EvaluationContractError("evaluation plan preregistered contract mismatch")
    return plan, digest


def _terminal_failure(
    output: Path,
    plan_sha256: str,
    request: EvaluationRequest,
    expected: Mapping[str, object],
    exception: Exception,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "evaluation-terminal-failure-v1",
        "status": "not_promoted",
        "evaluation_plan_sha256": plan_sha256,
        "request": {
            **asdict(request),
            "test_manifest_path": str(request.test_manifest_path),
            "output": str(request.output),
        },
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "test_accessed": True,
        "sealed_holdout_accessed": False,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    failure_path = output / "failures" / f"terminal-failure-{digest}.json"
    _atomic_no_overwrite(failure_path, payload)
    attempt_payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "evaluation-attempt-v1",
        "status": "failed",
        "evaluation_plan_sha256": plan_sha256,
        "request": dict(expected),
        "terminal_failure_path": str(failure_path),
        "terminal_failure_sha256": digest,
        "test_accessed": True,
        "sealed_holdout_accessed": False,
    }
    attempt_sha256 = hashlib.sha256(_canonical_bytes(attempt_payload)).hexdigest()
    attempt_path = output / "attempts" / f"evaluation-attempt-{attempt_sha256}.json"
    _atomic_no_overwrite(attempt_path, attempt_payload)
    ledger_payload = {
        "schema_version": 1,
        "stage": "evaluation-ledger-v1",
        "ordinal": request.ordinal,
        "evaluation_plan_sha256": plan_sha256,
        "attempt": {"path": str(attempt_path), "sha256": attempt_sha256},
    }
    _atomic_no_overwrite(output / "ledger" / f"attempt-{request.ordinal:04d}.json", ledger_payload)
    verdict_payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "promotion-verdict-v1",
        "status": "not_promoted",
        "promoted": False,
        "failed_clauses": ["terminal_replay_failure"],
        "evaluation_plan_sha256": plan_sha256,
        "terminal_failure_path": str(failure_path),
        "terminal_failure_sha256": digest,
        "test_accessed": True,
        "sealed_holdout_accessed": False,
    }
    verdict_sha256 = hashlib.sha256(_canonical_bytes(verdict_payload)).hexdigest()
    verdict_path = output / f"promotion-verdict-{verdict_sha256}.json"
    _atomic_no_overwrite(verdict_path, verdict_payload)
    return {
        "verdict_path": str(verdict_path),
        "verdict_sha256": verdict_sha256,
        **verdict_payload,
    }


def _finalization_failure(
    output: Path,
    plan_sha256: str,
    replay_path: Path,
    exception: Exception,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "evaluation-terminal-failure-v1",
        "status": "not_promoted",
        "failure_phase": "post_replay_estimation_or_report",
        "evaluation_plan_sha256": plan_sha256,
        "replay_path": str(replay_path.resolve()),
        "replay_sha256": _sha256(replay_path),
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "test_accessed": True,
        "sealed_holdout_accessed": False,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    failure_path = output / "failures" / f"terminal-failure-{digest}.json"
    _atomic_no_overwrite(failure_path, payload)
    verdict_payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "promotion-verdict-v1",
        "status": "not_promoted",
        "promoted": False,
        "failed_clauses": ["post_replay_non_estimable_or_report_failure"],
        "evaluation_plan_sha256": plan_sha256,
        "terminal_failure_path": str(failure_path.resolve()),
        "terminal_failure_sha256": digest,
        "replay_path": str(replay_path.resolve()),
        "replay_sha256": _sha256(replay_path),
        "test_accessed": True,
        "sealed_holdout_accessed": False,
    }
    verdict_sha256 = hashlib.sha256(_canonical_bytes(verdict_payload)).hexdigest()
    verdict_path = output / f"promotion-verdict-{verdict_sha256}.json"
    _atomic_no_overwrite(verdict_path, verdict_payload)
    return {"verdict_path": str(verdict_path), "verdict_sha256": verdict_sha256, **verdict_payload}


def _loaded_actor(path: Path, variant: PolicyVariant, expected_sha256: str) -> EntryActorCritic:
    if _sha256(path) != expected_sha256:
        raise EvaluationContractError("policy checkpoint changed after freeze")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise EvaluationContractError("policy checkpoint is unreadable") from exc
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise EvaluationContractError("policy checkpoint model state is missing")
    if variant is PolicyVariant.INDEPENDENT_ACTOR:
        first = state.get("independent_actor_trunks.ES.0.weight")
        offset = 2
    else:
        first = state.get("shared_actor_trunk.0.weight")
        offset = 10
    if not isinstance(first, torch.Tensor) or first.ndim != 2 or first.shape[1] <= offset:
        raise EvaluationContractError("policy checkpoint architecture is invalid")
    model = EntryActorCritic(
        int(first.shape[1]) - offset,
        variant,
        hidden_width=int(first.shape[0]),
    )
    model.load_state_dict(cast(dict[str, object], state), strict=True)
    model.eval()
    return model


def _stress_config(config: RlConfig, stress: str) -> RlConfig:
    slippage = 2.0 if stress == "two_tick" else 1.0
    fees = config.fees
    if stress == "fee_2x":
        fees = replace(
            fees,
            es=fees.es * 2.0,
            mes=fees.mes * 2.0,
            nq=fees.nq * 2.0,
            mnq=fees.mnq * 2.0,
            rty=fees.rty * 2.0,
            m2k=fees.m2k * 2.0,
            ym=fees.ym * 2.0,
            mym=fees.mym * 2.0,
            gc=fees.gc * 2.0,
            mgc=fees.mgc * 2.0,
            cl=fees.cl * 2.0,
            mcl=fees.mcl * 2.0,
            zb=fees.zb * 2.0,
        )
    return replace(
        config,
        execution=replace(config.execution, adverse_slippage_ticks_per_side=slippage),
        fees=fees,
    )


def _stress_contract_valid(config: RlConfig, stressed: RlConfig, stress: str) -> bool:
    if stress == "two_tick":
        return stressed.execution.adverse_slippage_ticks_per_side == 2.0
    if stress == "fee_2x":
        fields = (
            "es",
            "mes",
            "nq",
            "mnq",
            "rty",
            "m2k",
            "ym",
            "mym",
            "gc",
            "mgc",
            "cl",
            "mcl",
            "zb",
        )
        return stressed.execution.adverse_slippage_ticks_per_side == 1.0 and all(
            math.isclose(
                float(getattr(stressed.fees, field)),
                2.0 * float(getattr(config.fees, field)),
            )
            for field in fields
        )
    if stress == "same_bar_adverse":
        return stressed.exit.same_bar_policy == "prior_stop_first"
    return stressed.execution.adverse_slippage_ticks_per_side == 1.0


def _binary_entropy(takes: int, opportunities: int) -> float:
    if opportunities == 0 or takes in {0, opportunities}:
        return 0.0
    probability = takes / opportunities
    return -probability * math.log(probability) - (1.0 - probability) * math.log(1.0 - probability)


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvaluationContractError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationContractError(f"{field} must be finite")
    return result


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise EvaluationContractError(f"{field} must be an integer")
    return value


def _sha_ranked_exact_selection(
    ordered_ranks: Sequence[int],
    target_count: int,
    accepted_count: Callable[[frozenset[int]], int],
) -> frozenset[int]:
    """Admit SHA ranks until the dynamically legal replay accepts exactly the target."""
    if target_count < 0 or target_count > len(ordered_ranks):
        raise EvaluationContractError("matched random count is impossible")
    admitted = target_count
    while admitted <= len(ordered_ranks):
        selected = frozenset(ordered_ranks[:admitted])
        accepted = accepted_count(selected)
        if accepted == target_count:
            return selected
        if accepted > target_count:
            raise EvaluationContractError("matched random participation overshot target")
        admitted += max(1, target_count - accepted)
    raise EvaluationContractError("matched random dynamic legality cannot match participation")


_LatencyPayload = TypeVar("_LatencyPayload")


def _latency_one_bar_action(
    proposed_action: int,
    opportunity: _LatencyPayload | None,
    pending_opportunity: _LatencyPayload | None,
) -> tuple[int, _LatencyPayload | None, _LatencyPayload | None]:
    """Queue a new take or release the prior legal-bar submission deterministically."""
    if pending_opportunity is not None:
        return 1, None, pending_opportunity
    if proposed_action:
        if opportunity is None:
            raise EvaluationContractError("latency order payload is missing")
        return 0, opportunity, None
    return 0, None, None


@dataclass
class _StressTransitionState:
    stress: str
    missed_offset: int
    pending_order: EntryOrder | None = None
    opportunity_number: int = 0
    proposed_takes: int = 0
    missed_fill_count: int = 0
    latency_cancellations: int = 0


def _prepare_stressed_action(
    environment: TopstepEntryEnvironment,
    state: _StressTransitionState,
    action: int,
    opportunity: str,
) -> tuple[int, EntryOrder | None, str | None]:
    """Apply the shared deterministic order stress before one environment step."""
    if state.stress == "latency_1bar":
        state.proposed_takes += int(bool(action) and state.pending_order is None)
        proposed = environment.capture_entry_order(opportunity) if action else None
        action, state.pending_order, submitted = _latency_one_bar_action(
            action, proposed, state.pending_order
        )
        return (
            action,
            submitted,
            submitted.opportunity_identity if submitted is not None else None,
        )
    if not action:
        return 0, None, None
    state.proposed_takes += 1
    if state.stress == "missed_fill_10pct" and state.opportunity_number % 10 == state.missed_offset:
        action = 0
        state.missed_fill_count += 1
    state.opportunity_number += 1
    return action, None, opportunity if action else None


def _finish_stressed_step(
    environment: TopstepEntryEnvironment,
    state: _StressTransitionState,
    prior_session: str,
    submitted_order: EntryOrder | None,
    info: Mapping[str, object],
    terminated: bool,
    truncated: bool,
) -> None:
    """Update shared stress state after one environment transition."""
    if state.pending_order is not None and (
        environment._session != prior_session or info.get("event") == "DISCONTINUITY_RESET"
    ):
        state.pending_order = None
        state.latency_cancellations += 1
    if (
        state.stress == "latency_1bar"
        and submitted_order is not None
        and "booked_round_trip_fee" not in info
    ):
        state.latency_cancellations += 1
    if (terminated or truncated) and state.pending_order is not None:
        state.pending_order = None
        state.latency_cancellations += 1


def _canonical_pair_key(
    fold: int,
    calendar_block: str,
    episode_id: str,
    ticker: str,
    profile: str,
    seed: int,
) -> str:
    return _canonical_bytes([fold, calendar_block, episode_id, ticker, profile, seed]).decode()


def _production_evaluation_runner(config: RlConfig, plan: Mapping[str, object]) -> EvaluationRunner:
    serving = _load_object(Path(cast(str, plan["serving_freeze_path"])), "serving freeze")
    baseline = _load_object(Path(cast(str, plan["baseline_freeze_path"])), "baseline freeze")
    serving_artifacts = {
        (cast(int, item["fold"]), cast(int, item["seed"])): item
        for item in cast(list[dict[str, object]], serving["seed_artifacts"])
    }
    baseline_folds = {
        cast(int, item["fold"]): item for item in cast(list[dict[str, object]], baseline["folds"])
    }
    loaded_schedules: dict[
        tuple[Path, str, str | None], tuple[LoadedEpisodes, list[dict[str, object]]]
    ] = {}
    actor_cache: dict[tuple[str, int, int], EntryActorCritic] = {}

    def actor(policy: str, fold: int, seed: int) -> EntryActorCritic:
        key = (policy, fold, seed)
        if key in actor_cache:
            return actor_cache[key]
        if policy == "candidate":
            bundle_path = Path(cast(str, serving_artifacts[(fold, seed)]["checkpoint_bundle_path"]))
            checkpoint_path = bundle_path.parent / "checkpoint.pt"
            result = _loaded_actor(
                checkpoint_path,
                PolicyVariant.SHARED_TICKER_VALUE,
                cast(str, serving_artifacts[(fold, seed)]["artifact_sha256"]),
            )
        else:
            independent = {
                cast(int, item["seed"]): item
                for item in cast(list[dict[str, object]], baseline_folds[fold]["independent_ppo"])
            }[seed]
            result = _loaded_actor(
                Path(cast(str, independent["path"])),
                PolicyVariant.INDEPENDENT_ACTOR,
                cast(str, independent["sha256"]),
            )
        actor_cache[key] = result
        return result

    def evaluate(request: EvaluationRequest) -> Sequence[Mapping[str, object]]:
        stress_definition = cast(
            Mapping[str, object],
            cast(Mapping[str, object], plan["stress_definitions"])[request.stress],
        )
        stress_identity_sha256 = hashlib.sha256(_canonical_bytes(stress_definition)).hexdigest()
        coverage = "overlapping_starts" if request.stress == "overlapping_starts" else None
        schedule_key = (
            request.test_manifest_path.resolve(),
            request.test_manifest_sha256,
            coverage,
        )
        if schedule_key not in loaded_schedules:
            if _sha256(request.test_manifest_path) != request.test_manifest_sha256:
                raise EvaluationContractError("test schedule changed before replay")
            loaded = load_test_episode_manifest(
                config, request.test_manifest_path, coverage=coverage
            )
            raw = _load_object(request.test_manifest_path, "test schedule")
            if (
                loaded.manifest_sha256 != request.test_manifest_sha256
                or _sha256(request.test_manifest_path) != request.test_manifest_sha256
            ):
                raise EvaluationContractError("loaded test schedule digest mismatch")
            if coverage is None:
                identities = raw.get("episodes")
            else:
                stress_coverage = raw.get("stress_coverage")
                selected = (
                    stress_coverage.get(coverage) if isinstance(stress_coverage, dict) else None
                )
                identities = selected.get("episodes") if isinstance(selected, dict) else None
            if not isinstance(identities, list) or len(identities) != len(loaded.episodes):
                raise EvaluationContractError("test episode identities are invalid")
            loaded_schedules[schedule_key] = (
                loaded,
                cast(list[dict[str, object]], identities),
            )
        loaded, identities = loaded_schedules[schedule_key]
        if loaded.manifest_sha256 != request.test_manifest_sha256:
            raise EvaluationContractError("cached test schedule digest mismatch")
        episodes: Sequence[EnvironmentEpisode] = loaded.episodes
        fold_baseline = baseline_folds[request.fold]
        historical: HistoricalLogisticPolicy | None = None
        hgb_model: object | None = None
        hgb_threshold = 0.0
        policy_model: EntryActorCritic | None = None
        if request.policy in {"candidate", "independent_ticker_ppo"}:
            policy_model = actor(request.policy, request.fold, request.seed)
        elif request.policy == "historical_rejected_logistic_head":
            historical, _path, _digest = historical_logistic_artifact(config, request.fold)
        elif request.policy == "hist_gradient_boosting":
            hgb = cast(dict[str, object], fold_baseline["hist_gradient_boosting"])
            if _sha256(Path(cast(str, hgb["path"]))) != hgb["sha256"]:
                raise EvaluationContractError("HGB estimator changed after freeze")
            with Path(cast(str, hgb["path"])).open("rb") as handle:
                hgb_model = pickle.load(handle)
            evidence = cast(dict[str, object], hgb["fit_evidence"])
            hgb_threshold = _finite_float(evidence["threshold"], "HGB threshold")
        stressed_config = _stress_config(config, request.stress)
        stress_contract_valid = _stress_contract_valid(config, stressed_config, request.stress)
        matched_counts: dict[str, int] = {}
        if request.policy == "matched_random_take":
            if request.candidate_attempt_path is None:
                raise EvaluationContractError("matched random candidate attempt is missing")
            candidate_attempt = _load_object(
                request.candidate_attempt_path, "matched random candidate attempt"
            )
            if (
                candidate_attempt.get("stage") != "evaluation-attempt-v1"
                or candidate_attempt.get("status") != "complete"
                or candidate_attempt.get("evaluation_plan_sha256") != request.evaluation_plan_sha256
            ):
                raise EvaluationContractError("matched random candidate attempt is invalid")
            candidate_rows = candidate_attempt.get("rows")
            if not isinstance(candidate_rows, list):
                raise EvaluationContractError("matched random candidate rows are missing")
            matched_counts = {
                cast(str, row["episode_id"]): cast(int, row["accepted_trades"])
                for row in candidate_rows
                if isinstance(row, dict)
            }
        rows: list[dict[str, object]] = []
        for episode, identity in zip(episodes, identities, strict=True):
            active_episode = episode
            environment = TopstepEntryEnvironment(
                stressed_config,
                active_episode,
                gap_adverse_extra_ticks=(1.0 if request.stress == "gap_adverse_tick" else 0.0),
            )
            observation, _ = environment.reset(seed=request.seed)
            eligible = 0
            accepted_opportunities: list[str] = []
            peak_equity = _finite_float(environment.account_state["equity"], "account equity")
            maximum_drawdown = 0.0
            minimum_cushion = _finite_float(
                environment.account_state["equity"], "account equity"
            ) - _finite_float(environment.account_state["mll_floor"], "account MLL floor")
            commission_costs = 0.0
            gap_costs = 0.0
            gap_adjusted_fills = 0
            episode_id = str(identity.get("number"))
            episode_start = active_episode.bars[0].timestamp
            episode_iso = episode_start.isocalendar()
            calendar_block = f"{episode_iso.year}-W{episode_iso.week:02d}"
            pair_key = _canonical_pair_key(
                request.fold,
                calendar_block,
                episode_id,
                episode.ticker,
                episode.profile,
                request.seed,
            )
            matched_ranks: set[int] = set()
            offset = (
                int.from_bytes(
                    hashlib.sha256(
                        f"{request.evaluation_plan_sha256}{pair_key}:missed-fill-v1".encode()
                    ).digest()[:8],
                    "big",
                )
                % 10
            )
            stress_state = _StressTransitionState(request.stress, offset)
            if request.policy == "matched_random_take":
                target_count = matched_counts.get(str(identity.get("number")))
                if target_count is None:
                    raise EvaluationContractError("matched random target count is missing")
                opportunities: list[int] = []
                probe = TopstepEntryEnvironment(stressed_config, active_episode)
                probe.reset(seed=request.seed)
                while True:
                    probe_mask = probe.action_mask()
                    if bool(probe_mask[1]):
                        opportunity = probe.episode.bars[probe._index].timestamp.isoformat()
                        opportunities.append(
                            int.from_bytes(
                                hashlib.sha256(
                                    f"{request.evaluation_plan_sha256}{pair_key}{opportunity}:matched-random-v1".encode()
                                ).digest(),
                                "big",
                            )
                        )
                    _, _, probe_done, probe_truncated, _ = probe.step(0)
                    if probe_done or probe_truncated:
                        break
                if target_count > len(opportunities):
                    raise EvaluationContractError("matched random count is impossible")
                ordered_ranks = sorted(opportunities)

                def accepted_for(
                    selected: frozenset[int],
                    episode_for_replay: EnvironmentEpisode = active_episode,
                    pair_key_for_replay: str = pair_key,
                    missed_offset: int = offset,
                ) -> int:
                    replay = TopstepEntryEnvironment(
                        stressed_config,
                        episode_for_replay,
                        gap_adverse_extra_ticks=(
                            1.0 if request.stress == "gap_adverse_tick" else 0.0
                        ),
                    )
                    replay.reset(seed=request.seed)
                    replay_stress = _StressTransitionState(request.stress, missed_offset)
                    while True:
                        replay_mask = replay.action_mask()
                        replay_action = 0
                        replay_submitted: EntryOrder | None = None
                        if bool(replay_mask[1]):
                            replay_opportunity = replay.episode.bars[
                                replay._index
                            ].timestamp.isoformat()
                            replay_rank = int.from_bytes(
                                hashlib.sha256(
                                    f"{request.evaluation_plan_sha256}{pair_key_for_replay}{replay_opportunity}:matched-random-v1".encode()
                                ).digest(),
                                "big",
                            )
                            replay_action = int(replay_rank in selected)
                            replay_action, replay_submitted, _ = _prepare_stressed_action(
                                replay,
                                replay_stress,
                                replay_action,
                                replay_opportunity,
                            )
                        prior_replay_session = replay._session
                        _, _, replay_done, replay_truncated, replay_info = replay.step(
                            replay_action, entry_order=replay_submitted
                        )
                        _finish_stressed_step(
                            replay,
                            replay_stress,
                            prior_replay_session,
                            replay_submitted,
                            replay_info,
                            replay_done,
                            replay_truncated,
                        )
                        if replay_done or replay_truncated:
                            break
                    return _integer(
                        replay.account_state["accepted_trades"], "matched accepted trades"
                    )

                matched_ranks = set(
                    _sha_ranked_exact_selection(ordered_ranks, target_count, accepted_for)
                )
            while True:
                mask = environment.action_mask()
                action = 0
                submitted_opportunity: str | None = None
                submitted_order: EntryOrder | None = None
                if bool(mask[1]):
                    eligible += 1
                    opportunity = (
                        f"{environment.episode.bars[environment._index].timestamp.isoformat()}"
                    )
                    if request.policy == "take_all":
                        action = 1
                    elif request.policy == "reject_all":
                        action = 0
                    elif request.policy == "historical_rejected_logistic_head":
                        assert historical is not None
                        action = historical.action(observation, mask.copy())
                    elif request.policy == "hist_gradient_boosting":
                        assert hgb_model is not None
                        probability = float(
                            hgb_model.predict_proba(observation.vector[None, :])[0, 1]  # type: ignore[attr-defined]
                        )
                        action = int(probability >= hgb_threshold)
                    elif request.policy in {"candidate", "independent_ticker_ppo"}:
                        assert policy_model is not None
                        with torch.no_grad():
                            logits, values = policy_model(
                                torch.from_numpy(observation.vector.copy()).unsqueeze(0),
                                torch.tensor([TICKERS.index(episode.ticker)]),
                                torch.tensor([PROFILES.index(episode.profile)]),
                            )
                            if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
                                raise EvaluationContractError("policy inference is non-finite")
                            action = int(
                                logits.masked_fill(
                                    ~torch.from_numpy(mask).unsqueeze(0),
                                    torch.finfo(logits.dtype).min,
                                )
                                .argmax(dim=1)
                                .item()
                            )
                    elif request.policy == "matched_random_take":
                        rank = int.from_bytes(
                            hashlib.sha256(
                                f"{request.evaluation_plan_sha256}{pair_key}{opportunity}:matched-random-v1".encode()
                            ).digest(),
                            "big",
                        )
                        # The cutoff is frozen from all legal opportunities below.
                        action = int(rank in matched_ranks)
                    action, submitted_order, submitted_opportunity = _prepare_stressed_action(
                        environment,
                        stress_state,
                        action,
                        opportunity,
                    )
                prior_session = environment._session
                observation, reward, terminated, truncated, info = environment.step(
                    action, entry_order=submitted_order
                )
                _finish_stressed_step(
                    environment,
                    stress_state,
                    prior_session,
                    submitted_order,
                    info,
                    terminated,
                    truncated,
                )
                if not np.isfinite(observation.vector).all() or not math.isfinite(float(reward)):
                    raise EvaluationContractError("evaluation transition is non-finite")
                if "booked_round_trip_fee" in info:
                    commission_costs += _finite_float(info["booked_round_trip_fee"], "booked fee")
                    gap_adjusted_fills += int(info.get("gap_adjusted_fill") is True)
                    gap_costs += _finite_float(info.get("gap_extra_cost", 0.0), "gap extra cost")
                    if submitted_opportunity is None:
                        raise EvaluationContractError("fill has no entry opportunity identity")
                    accepted_opportunities.append(submitted_opportunity)
                state = environment.account_state
                equity = _finite_float(state["equity"], "account equity")
                peak_equity = max(peak_equity, equity)
                maximum_drawdown = max(maximum_drawdown, peak_equity - equity)
                minimum_cushion = min(
                    minimum_cushion,
                    equity - _finite_float(state["mll_floor"], "account MLL floor"),
                )
                if terminated or truncated:
                    break
            state = environment.account_state
            accepted = _integer(state["accepted_trades"], "accepted trades")
            if (
                request.policy == "matched_random_take"
                and accepted != matched_counts[str(identity.get("number"))]
            ):
                raise EvaluationContractError("matched random participation did not match")
            starting = config.episode.account_start
            net_pnl = _finite_float(state["balance"], "account balance") - starting
            slippage_cost = (
                accepted
                * 2.0
                * stressed_config.execution.adverse_slippage_ticks_per_side
                * float(environment._tick_value())
            )
            start = episode_start
            end = active_episode.bars[-1].timestamp
            rows.append(
                {
                    "fold": request.fold,
                    "seed": request.seed,
                    "ticker": episode.ticker,
                    "profile": episode.profile,
                    "regime_block": f"{start.year}-Q{(start.month - 1) // 3 + 1}",
                    "calendar_block": calendar_block,
                    "episode_id": episode_id,
                    "start_ts": start.isoformat(),
                    "end_ts": end.isoformat(),
                    "status": str(state["status"]),
                    "trading_days": _finite_float(state["trading_days"], "trading days"),
                    "calendar_days": float((end.date() - start.date()).days + 1),
                    "accepted_trades": accepted,
                    "eligible_entries": eligible,
                    "net_pnl": net_pnl,
                    "costs": commission_costs + slippage_cost + gap_costs,
                    "commission_costs": commission_costs,
                    "slippage_costs": slippage_cost,
                    "gap_costs": gap_costs,
                    "expectancy": net_pnl / accepted if accepted else 0.0,
                    "maximum_drawdown": maximum_drawdown,
                    "minimum_mll_cushion": minimum_cushion,
                    "best_day_profit": _finite_float(state["best_day_profit"], "best day profit"),
                    "consistency_ratio": _finite_float(
                        state["consistency_ratio"], "consistency ratio"
                    ),
                    "action_entropy": _binary_entropy(stress_state.proposed_takes, eligible),
                    "ambiguity_count": _integer(state["ambiguity_count"], "ambiguity count"),
                    "gap_adjusted_fills": gap_adjusted_fills,
                    "latency_cancellations": stress_state.latency_cancellations,
                    "missed_fill_count": stress_state.missed_fill_count,
                    "finite": True,
                    "accepted_opportunity_ids": accepted_opportunities,
                    "stress_identity_sha256": stress_identity_sha256,
                    "stress_invariants_valid": stress_contract_valid,
                }
            )
        return rows

    return evaluate


def run_topstep_evaluation(
    config: RlConfig,
    evaluation_plan_path: Path,
    output: Path,
    *,
    runner: EvaluationRunner | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Run the frozen append-only replay schedule and publish report plus verdict."""
    plan, plan_sha256 = _validated_evaluation_plan(config, evaluation_plan_path)
    if runner is None:
        runner = _production_evaluation_runner(config, plan)
    if output.exists() and any(output.iterdir()) and not resume:
        raise EvaluationContractError("evaluation output already exists; resume is required")
    failure_files = sorted((output / "failures").glob("terminal-failure-*.json"))
    if failure_files:
        failure_digests: set[str] = set()
        for failure_path in failure_files:
            failure = _load_object(failure_path, "evaluation terminal failure")
            if (
                failure_path.name != f"terminal-failure-{_sha256(failure_path)}.json"
                or failure.get("stage") != "evaluation-terminal-failure-v1"
                or failure.get("status") != "not_promoted"
                or failure.get("evaluation_plan_sha256") != plan_sha256
                or failure.get("sealed_holdout_accessed") is not False
            ):
                raise EvaluationContractError("evaluation terminal failure provenance mismatch")
            failure_digests.add(_sha256(failure_path))
        verdict_files = sorted(output.glob("promotion-verdict-*.json"))
        if len(verdict_files) != 1:
            raise EvaluationContractError("terminal failure promotion verdict is missing")
        verdict = _load_object(verdict_files[0], "terminal promotion verdict")
        if (
            verdict_files[0].name != f"promotion-verdict-{_sha256(verdict_files[0])}.json"
            or verdict.get("stage") != "promotion-verdict-v1"
            or verdict.get("status") != "not_promoted"
            or verdict.get("promoted") is not False
            or verdict.get("evaluation_plan_sha256") != plan_sha256
            or verdict.get("terminal_failure_sha256") not in failure_digests
        ):
            raise EvaluationContractError("terminal failure promotion verdict mismatch")
        return {
            "verdict_path": str(verdict_files[0]),
            "verdict_sha256": _sha256(verdict_files[0]),
            **verdict,
        }
    schedule_by_fold = {
        cast(int, _load_object(Path(cast(str, item["path"])), "test schedule")["fold"]): item
        for item in cast(list[dict[str, object]], plan["test_schedules"])
    }
    mechanics_sha256 = hashlib.sha256(
        _canonical_bytes(
            {
                "rule_digest": config.rule_digest,
                "fee_digest": config.fee_digest,
                "environment_code_sha256": _sha256(
                    Path(__file__).resolve().with_name("rl_environment.py")
                ),
            }
        )
    ).hexdigest()
    ledger = output / "ledger"
    attempt_directory = output / "attempts"
    rows: list[dict[str, object]] = []
    attempt_paths: dict[tuple[int, int, str, str], Path] = {}
    completed = 0
    if resume and ledger.is_dir():
        ledger_paths = sorted(ledger.glob("attempt-*.json"))
        if len(ledger_paths) > len(cast(list[object], plan["expected_append_only_ledger"])):
            raise EvaluationContractError("evaluation resume has unexpected ledger entries")
        for ordinal, ledger_path in enumerate(ledger_paths, start=1):
            entry = _load_object(ledger_path, "evaluation ledger")
            reference = entry.get("attempt")
            if (
                ledger_path.name != f"attempt-{ordinal:04d}.json"
                or entry.get("stage") != "evaluation-ledger-v1"
                or entry.get("ordinal") != ordinal
                or entry.get("evaluation_plan_sha256") != plan_sha256
                or not isinstance(reference, dict)
                or not isinstance(reference.get("path"), str)
                or _sha256(Path(cast(str, reference["path"]))) != reference.get("sha256")
            ):
                raise EvaluationContractError("evaluation resume ledger mismatch")
            attempt = _load_object(Path(cast(str, reference["path"])), "evaluation attempt")
            expected = cast(list[dict[str, object]], plan["expected_append_only_ledger"])[
                ordinal - 1
            ]
            if (
                attempt.get("stage") != "evaluation-attempt-v1"
                or attempt.get("status") != "complete"
                or attempt.get("request") != expected
                or attempt.get("evaluation_plan_sha256") != plan_sha256
                or attempt.get("sealed_holdout_accessed") is not False
                or not isinstance(attempt.get("rows"), list)
            ):
                raise EvaluationContractError("evaluation resume attempt mismatch")
            rows.extend(cast(list[dict[str, object]], attempt["rows"]))
            attempt_paths[
                (
                    cast(int, expected["fold"]),
                    cast(int, expected["seed"]),
                    cast(str, expected["policy"]),
                    cast(str, expected["stress"]),
                )
            ] = Path(cast(str, reference["path"]))
            completed = ordinal
    expected_ledger = cast(list[dict[str, object]], plan["expected_append_only_ledger"])
    for expected in expected_ledger[completed:]:
        fold = cast(int, expected["fold"])
        schedule = schedule_by_fold[fold]
        request = EvaluationRequest(
            ordinal=cast(int, expected["ordinal"]),
            fold=fold,
            seed=cast(int, expected["seed"]),
            policy=cast(str, expected["policy"]),
            stress=cast(str, expected["stress"]),
            evaluation_plan_sha256=plan_sha256,
            test_manifest_path=Path(cast(str, schedule["path"])),
            test_manifest_sha256=cast(str, schedule["sha256"]),
            output=output
            / "runs"
            / f"fold-{fold}"
            / f"seed-{expected['seed']}"
            / cast(str, expected["policy"])
            / cast(str, expected["stress"]),
            candidate_attempt_path=(
                attempt_paths.get(
                    (fold, cast(int, expected["seed"]), "candidate", cast(str, expected["stress"]))
                )
                if expected["policy"] == "matched_random_take"
                else None
            ),
        )
        try:
            raw_rows = runner(request)
            attempt_rows = []
            for raw in raw_rows:
                if set(raw) != _ATTEMPT_FIELDS:
                    raise EvaluationContractError("evaluation runner row schema is invalid")
                attempt_rows.append(
                    {
                        **raw,
                        "policy": request.policy,
                        "stress": request.stress,
                        "schedule_sha256": request.test_manifest_sha256,
                        "mechanics_sha256": mechanics_sha256,
                    }
                )
            if not attempt_rows:
                raise EvaluationContractError("evaluation runner returned no attempts")
            _validated_attempts(
                [_attempt_slice(row) for row in attempt_rows],
                allow_overlap=request.stress == "overlapping_starts",
            )
        except Exception as exc:
            _terminal_failure(output, plan_sha256, request, expected, exc)
            raise EvaluationContractError(
                f"evaluation failed for fold {request.fold} seed {request.seed} "
                f"{request.policy} {request.stress}"
            ) from exc
        attempt_payload: dict[str, object] = {
            "schema_version": 1,
            "stage": "evaluation-attempt-v1",
            "status": "complete",
            "evaluation_plan_sha256": plan_sha256,
            "request": expected,
            "rows": attempt_rows,
            "test_accessed": True,
            "sealed_holdout_accessed": False,
        }
        attempt_sha256 = hashlib.sha256(_canonical_bytes(attempt_payload)).hexdigest()
        attempt_path = attempt_directory / f"evaluation-attempt-{attempt_sha256}.json"
        _atomic_no_overwrite(attempt_path, attempt_payload)
        ledger_payload = {
            "schema_version": 1,
            "stage": "evaluation-ledger-v1",
            "ordinal": request.ordinal,
            "evaluation_plan_sha256": plan_sha256,
            "attempt": {"path": str(attempt_path), "sha256": attempt_sha256},
        }
        _atomic_no_overwrite(ledger / f"attempt-{request.ordinal:04d}.json", ledger_payload)
        attempt_paths[(fold, request.seed, request.policy, request.stress)] = attempt_path
        rows.extend(attempt_rows)
    replay_payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "rl-topstep-test-replay-v1",
        "status": "complete",
        "partition": "test",
        "config_sha256": config.digest,
        "evaluation_plan_path": str(evaluation_plan_path.resolve()),
        "evaluation_plan_sha256": plan_sha256,
        "serving_freeze_sha256": plan["serving_freeze_sha256"],
        "test_schedules": plan["test_schedules"],
        "mechanics_sha256": mechanics_sha256,
        "market_uncertainty": "synchronized_calendar_block_bootstrap",
        "overlapping_start_coverage": {
            "replayed": True,
            "promotion_denominator": False,
        },
        "attempt_ledger": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in sorted(ledger.glob("attempt-*.json"))
        ],
        "rows": rows,
        "test_accessed": True,
        "sealed_holdout_accessed": False,
    }
    replay_sha256 = hashlib.sha256(_canonical_bytes(replay_payload)).hexdigest()
    replay_path = output / f"test-replay-{replay_sha256}.json"
    _atomic_no_overwrite(replay_path, replay_payload)
    try:
        return evaluate_topstep_promotion(
            config,
            Path(cast(str, plan["serving_freeze_path"])),
            replay_path,
            output,
        )
    except Exception as exc:
        _finalization_failure(output, plan_sha256, replay_path, exc)
        raise EvaluationContractError("evaluation finalization failed after test replay") from exc


def _validated_replay(
    config: RlConfig, replay_path: Path, serving_sha256: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    replay = _load_object(replay_path, "Topstep replay")
    plan_path = replay.get("evaluation_plan_path")
    if not isinstance(plan_path, str):
        raise EvaluationContractError("test replay evaluation plan is missing")
    plan, plan_sha256 = _validated_evaluation_plan(config, Path(plan_path))
    if (
        replay.get("schema_version") != 1
        or replay.get("stage") != "rl-topstep-test-replay-v1"
        or replay.get("status") != "complete"
        or replay.get("partition") != "test"
        or replay.get("config_sha256") != config.digest
        or replay.get("evaluation_plan_sha256") != plan_sha256
        or replay.get("serving_freeze_sha256") != serving_sha256
        or replay.get("test_accessed") is not True
        or replay.get("sealed_holdout_accessed") is not False
        or replay.get("overlapping_start_coverage")
        != {"replayed": True, "promotion_denominator": False}
        or replay.get("market_uncertainty") != "synchronized_calendar_block_bootstrap"
    ):
        raise EvaluationContractError("test replay provenance mismatch")
    schedules = replay.get("test_schedules")
    if not isinstance(schedules, list) or not schedules:
        raise EvaluationContractError("test schedules are missing")
    if schedules != plan.get("test_schedules"):
        raise EvaluationContractError("test replay schedules differ from frozen plan")
    schedule_digests: set[str] = set()
    scheduled_attempts: dict[tuple[str, int, str, str], tuple[str, str]] = {}
    overlapping_attempts: dict[tuple[str, int, str, str], tuple[str, str]] = {}
    for reference in schedules:
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            raise EvaluationContractError("test schedule reference is invalid")
        schedule_path = Path(cast(str, reference["path"]))
        digest = _sha256(schedule_path)
        schedule = _load_object(schedule_path, "test schedule")
        partition = schedule.get("partition")
        if (
            digest != reference.get("sha256")
            or schedule.get("stage") != "rl-episode-schedule"
            or not isinstance(partition, dict)
            or partition.get("name") != "test"
            or datetime.fromisoformat(cast(str, partition.get("end")))
            >= config.evaluation.sealed_holdout_start
        ):
            raise EvaluationContractError("test schedule provenance mismatch")
        schedule_digests.add(digest)
        for episode in cast(list[dict[str, object]], schedule["episodes"]):
            key = (
                digest,
                cast(int, episode["number"]),
                cast(str, episode["ticker"]),
                cast(str, episode["profile"]),
            )
            scheduled_attempts[key] = (
                cast(str, episode["decision_start"]),
                cast(str, episode["terminal_end"]),
            )
        overlap = cast(
            dict[str, object],
            cast(dict[str, object], schedule["stress_coverage"])["overlapping_starts"],
        )
        for episode in cast(list[dict[str, object]], overlap["episodes"]):
            key = (
                digest,
                cast(int, episode["number"]),
                cast(str, episode["ticker"]),
                cast(str, episode["profile"]),
            )
            overlapping_attempts[key] = (
                cast(str, episode["decision_start"]),
                cast(str, episode["terminal_end"]),
            )
    raw_rows = replay.get("rows")
    if (
        not isinstance(raw_rows, list)
        or not raw_rows
        or any(not isinstance(row, dict) for row in raw_rows)
    ):
        raise EvaluationContractError("test replay rows are missing")
    rows = cast(list[dict[str, object]], raw_rows)
    required = _ATTEMPT_FIELDS | {"policy", "stress", "schedule_sha256", "mechanics_sha256"}
    mechanics: set[object] = set()
    stress_definitions = cast(Mapping[str, object], plan["stress_definitions"])
    for row in rows:
        if set(row) != required:
            raise EvaluationContractError("test replay row schema is invalid")
        if row["policy"] not in _POLICIES or row["stress"] not in {_HEADLINE, *_REQUIRED_STRESSES}:
            raise EvaluationContractError("test replay policy or stress is invalid")
        if row["schedule_sha256"] not in schedule_digests:
            raise EvaluationContractError("test replay schedule identity mismatch")
        expected_stress_sha256 = hashlib.sha256(
            _canonical_bytes(stress_definitions[row["stress"]])
        ).hexdigest()
        if row["stress_identity_sha256"] != expected_stress_sha256:
            raise EvaluationContractError("test replay stress identity mismatch")
        schedule_key = (
            row["schedule_sha256"],
            int(cast(str, row["episode_id"])),
            cast(str, row["ticker"]),
            cast(str, row["profile"]),
        )
        expected_attempts = (
            overlapping_attempts if row["stress"] == "overlapping_starts" else scheduled_attempts
        )
        if expected_attempts.get(schedule_key) != (row["start_ts"], row["end_ts"]):
            raise EvaluationContractError("test replay row differs from frozen schedule")
        mechanics.add(row["mechanics_sha256"])
    if len(mechanics) != 1 or replay.get("mechanics_sha256") not in mechanics:
        raise EvaluationContractError("policies and stresses did not share account mechanics")
    combinations = {(cast(str, row["policy"]), cast(str, row["stress"])) for row in rows}
    if combinations != {
        (policy, stress) for policy in _POLICIES for stress in {_HEADLINE, *_REQUIRED_STRESSES}
    }:
        raise EvaluationContractError("policy and stress replay matrix is incomplete")
    expected_keys: dict[str, set[tuple[object, ...]]] = {}
    for policy, stress in sorted(combinations):
        subset = [row for row in rows if row["policy"] == policy and row["stress"] == stress]
        _validated_attempts(
            [_attempt_slice(row) for row in subset],
            allow_overlap=stress == "overlapping_starts",
        )
        keys: set[tuple[object, ...]] = {
            (row["fold"], row["seed"], row["ticker"], row["profile"], row["episode_id"])
            for row in subset
        }
        coverage_class = "overlap" if stress == "overlapping_starts" else "headline"
        if coverage_class not in expected_keys:
            expected_keys[coverage_class] = keys
        elif keys != expected_keys[coverage_class]:
            raise EvaluationContractError("policies and stresses must replay identical attempts")
    if {cast(int, row["seed"]) for row in rows} != set(config.training.confirmation_seeds):
        raise EvaluationContractError("test replay confirmation seeds are incomplete")
    observed_cells = {(cast(str, row["ticker"]), cast(str, row["profile"])) for row in rows}
    required_cells = {
        (ticker, profile)
        for ticker in TICKERS
        for profile in (("one_mini",) if ticker == "ZB" else PROFILES)
    }
    if observed_cells != required_cells:
        raise EvaluationContractError("test replay ticker/profile cells are incomplete")
    ledger = replay.get("attempt_ledger")
    if not isinstance(ledger, list) or len(ledger) != len(
        cast(list[object], plan["expected_append_only_ledger"])
    ):
        raise EvaluationContractError("test replay ledger is incomplete")
    for reference in ledger:
        if (
            not isinstance(reference, dict)
            or not isinstance(reference.get("path"), str)
            or _sha256(Path(cast(str, reference["path"]))) != reference.get("sha256")
        ):
            raise EvaluationContractError("test replay ledger digest mismatch")
    return replay, rows


def _complete_calendar_blocks(
    rows: Sequence[Mapping[str, object]],
    required_cells: set[tuple[str, str]],
    expected_seeds: Sequence[int],
) -> list[object]:
    """Return the ordered week intersection shared by every required cell and seed."""
    blocks_by_owner: dict[tuple[str, str, int], set[object]] = defaultdict(set)
    for row in rows:
        if row.get("policy") != "candidate" or row.get("stress") != "primary":
            continue
        ticker = row.get("ticker")
        profile = row.get("profile")
        seed = row.get("seed")
        if (
            isinstance(ticker, str)
            and isinstance(profile, str)
            and type(seed) is int
            and seed in expected_seeds
        ):
            blocks_by_owner[(ticker, profile, seed)].add(row.get("calendar_block"))
    owners = {
        (ticker, profile, seed) for ticker, profile in required_cells for seed in expected_seeds
    }
    if set(blocks_by_owner) != owners or any(not blocks_by_owner[owner] for owner in owners):
        raise EvaluationContractError("calendar block ownership is incomplete")
    complete = set.intersection(*(blocks_by_owner[owner] for owner in sorted(owners)))
    blocks = sorted(complete, key=str)
    if len(blocks) < 20:
        raise EvaluationContractError(
            "global evaluation has fewer than 20 complete calendar blocks"
        )
    return blocks


def _baseline_result_differences(
    rows: Sequence[Mapping[str, object]], stress: str
) -> list[dict[str, object]]:
    """Report matched candidate-minus-baseline outcomes for every required view."""
    pair_fields = ("fold", "seed", "ticker", "profile", "episode_id")
    dimensions = {
        "pooled": (),
        "fold": ("fold",),
        "ticker": ("ticker",),
        "profile": ("profile",),
        "regime_block": ("regime_block",),
        "seed": ("seed",),
        "cell": ("fold", "ticker", "profile", "regime_block", "seed"),
    }
    candidate = [
        row for row in rows if row.get("policy") == "candidate" and row.get("stress") == stress
    ]
    candidate_by_key = {tuple(row[field] for field in pair_fields): row for row in candidate}
    if not candidate or len(candidate_by_key) != len(candidate):
        raise EvaluationContractError("candidate result differences are not uniquely paired")
    reports: list[dict[str, object]] = []
    for baseline in sorted(_POLICIES - {"candidate"}):
        baseline_rows = [
            row for row in rows if row.get("policy") == baseline and row.get("stress") == stress
        ]
        baseline_by_key = {tuple(row[field] for field in pair_fields): row for row in baseline_rows}
        if len(baseline_by_key) != len(baseline_rows) or set(baseline_by_key) != set(
            candidate_by_key
        ):
            raise EvaluationContractError(f"{baseline} result differences are not paired")
        views: dict[str, list[dict[str, object]]] = {}
        for name, fields in dimensions.items():
            grouped: dict[
                tuple[object, ...], list[tuple[Mapping[str, object], Mapping[str, object]]]
            ] = defaultdict(list)
            for key, candidate_row in candidate_by_key.items():
                group = tuple(candidate_row[field] for field in fields)
                grouped[group].append((candidate_row, baseline_by_key[key]))
            views[name] = []
            for group in sorted(grouped, key=lambda value: tuple(map(str, value))):
                pairs = grouped[group]

                def mean_difference(
                    field: str,
                    selected_pairs: list[tuple[Mapping[str, object], Mapping[str, object]]] = pairs,
                ) -> float:
                    return float(
                        np.mean(
                            [
                                _finite_float(left[field], field)
                                - _finite_float(right[field], field)
                                for left, right in selected_pairs
                            ]
                        )
                    )

                views[name].append(
                    {
                        **{field: group[index] for index, field in enumerate(fields)},
                        "attempts": len(pairs),
                        "pass_rate_difference": float(
                            np.mean(
                                [
                                    float(left["status"] == "PASS")
                                    - float(right["status"] == "PASS")
                                    for left, right in pairs
                                ]
                            )
                        ),
                        "blow_rate_difference": float(
                            np.mean(
                                [
                                    float(left["status"] == "BLOW")
                                    - float(right["status"] == "BLOW")
                                    for left, right in pairs
                                ]
                            )
                        ),
                        "mean_net_pnl_difference": mean_difference("net_pnl"),
                        "mean_cost_difference": mean_difference("costs"),
                        "mean_maximum_drawdown_difference": mean_difference("maximum_drawdown"),
                    }
                )
        reports.append({"baseline": baseline, "views": views})
    return reports


def evaluate_topstep_promotion(
    config: RlConfig, serving_freeze_path: Path, replay_path: Path, output: Path
) -> dict[str, object]:
    """Evaluate a complete fixed test replay and write one immutable promotion report."""
    _serving, serving_sha256 = _validate_serving_freeze(config, serving_freeze_path)
    replay, rows = _validated_replay(config, replay_path, serving_sha256)
    plan_sha256 = replay.get("evaluation_plan_sha256")
    if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
        raise EvaluationContractError("test replay evaluation plan identity is missing")
    expected_seeds = tuple(config.training.confirmation_seeds)
    required_cells = {
        (ticker, profile)
        for ticker in TICKERS
        for profile in (("one_mini",) if ticker == "ZB" else PROFILES)
    }
    global_blocks = _complete_calendar_blocks(rows, required_cells, expected_seeds)
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(f"{plan_sha256}:test-market-bootstrap-v1".encode()).digest()[:8],
        "big",
    )
    bootstrap_indices = np.random.Generator(np.random.PCG64(bootstrap_seed)).integers(
        0,
        len(global_blocks),
        size=(100_000, len(global_blocks)),
        dtype=np.uint32,
    )
    stress_metrics: dict[str, object] = {}
    numeric_evidence: dict[str, dict[str, object]] = {}
    sensitivity_evidence: dict[str, dict[str, object]] = {}
    bootstrap_hashes: set[str] = set()
    for stress in sorted(_REQUIRED_STRESSES):
        stress_report: dict[str, object] = {
            policy: _group_summaries(
                [row for row in rows if row["policy"] == policy and row["stress"] == stress],
                allow_overlap=stress == "overlapping_starts",
            )
            for policy in sorted(_POLICIES)
        }
        stress_report["result_differences"] = _baseline_result_differences(rows, stress)
        stress_metrics[stress] = stress_report
        candidate_rows = [
            row for row in rows if row["policy"] == "candidate" and row["stress"] == stress
        ]
        summaries = cast(
            dict[str, object], cast(dict[str, object], stress_metrics[stress])["candidate"]
        )
        if stress in _SENSITIVITY_STRESSES:
            sensitivity_evidence[stress] = {"complete": True}
            continue
        profile_summaries = {
            cast(str, item["profile"]): _gate_values(cast(Mapping[str, object], item["summary"]))
            for item in cast(list[dict[str, object]], summaries["profile"])
        }
        seed_pass_rates: dict[str, dict[int, float]] = {}
        seed_blow_counts: dict[str, dict[int, int]] = {}
        for scope, profile in (
            ("pooled", None),
            ("one_mini", "one_mini"),
            ("ten_micros", "ten_micros"),
        ):
            selected = (
                candidate_rows
                if profile is None
                else [row for row in candidate_rows if row["profile"] == profile]
            )
            seed_pass_rates[scope] = {
                seed: float(
                    np.mean([row["status"] == "PASS" for row in selected if row["seed"] == seed])
                )
                for seed in expected_seeds
            }
            seed_blow_counts[scope] = {
                seed: sum(row["status"] == "BLOW" for row in selected if row["seed"] == seed)
                for seed in expected_seeds
            }
        baseline_effects: dict[str, dict[str, object]] = {}
        for scope, profile in (
            ("pooled", None),
            ("one_mini", "one_mini"),
            ("ten_micros", "ten_micros"),
        ):
            selected = (
                rows if profile is None else [row for row in rows if row["profile"] == profile]
            )
            baseline_effects[scope] = {}
            for baseline in ("take_all", "matched_random_take"):
                effect = _point_and_lcb(
                    selected,
                    "candidate",
                    baseline,
                    stress,
                    plan_sha256,
                    expected_seeds,
                    global_blocks,
                    bootstrap_indices,
                )
                baseline_effects[scope][baseline] = effect
                bootstrap_hashes.add(cast(str, effect["index_matrix_sha256"]))
        transfer_effects: dict[tuple[str, str], Mapping[str, object]] = {}
        pooled_transfer = _point_and_lcb(
            rows,
            "candidate",
            "independent_ticker_ppo",
            stress,
            plan_sha256,
            expected_seeds,
            global_blocks,
            bootstrap_indices,
        )
        transfer_effects[("POOLED", "pooled")] = pooled_transfer
        bootstrap_hashes.add(cast(str, pooled_transfer["index_matrix_sha256"]))
        for ticker, profile in sorted(required_cells):
            cell_rows = [
                row for row in rows if row["ticker"] == ticker and row["profile"] == profile
            ]
            effect = _point_and_lcb(
                cell_rows,
                "candidate",
                "independent_ticker_ppo",
                stress,
                plan_sha256,
                expected_seeds,
                global_blocks,
                bootstrap_indices,
            )
            transfer_effects[(ticker, profile)] = effect
            bootstrap_hashes.add(cast(str, effect["index_matrix_sha256"]))
        numeric_evidence[stress] = {
            "pooled": _gate_values(cast(Mapping[str, object], summaries["pooled"])),
            "profiles": profile_summaries,
            "seed_pass_rates": seed_pass_rates,
            "seed_blow_counts": seed_blow_counts,
            "baseline_effects": baseline_effects,
            "transfer_effects": transfer_effects,
        }
    if len(bootstrap_hashes) != 1:
        raise EvaluationContractError("gated views did not share bootstrap indices")
    verdict = promotion_verdict(
        numeric_stresses=numeric_evidence,
        sensitivity_stresses=sensitivity_evidence,
        required_seeds=set(expected_seeds),
        required_transfer_cells=required_cells,
        required_artifacts_complete=True,
    )
    serial_numeric_evidence: dict[str, object] = {}
    for stress, evidence in numeric_evidence.items():
        serial_numeric_evidence[stress] = {
            **evidence,
            "transfer_effects": [
                {"ticker": ticker, "profile": profile, **effect}
                for (ticker, profile), effect in sorted(
                    cast(
                        Mapping[tuple[str, str], Mapping[str, object]],
                        evidence["transfer_effects"],
                    ).items()
                )
            ],
        }
    report_payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "evaluation-report-v1",
        "status": "complete",
        "config_sha256": config.digest,
        "serving_freeze_path": str(serving_freeze_path.resolve()),
        "serving_freeze_sha256": serving_sha256,
        "evaluation_plan_sha256": plan_sha256,
        "replay_path": str(replay_path.resolve()),
        "replay_sha256": _sha256(replay_path),
        "test_schedule_sha256": replay["test_schedules"],
        "mechanics_sha256": replay["mechanics_sha256"],
        "stress_metrics": stress_metrics,
        "numeric_gate_evidence": serial_numeric_evidence,
        "bootstrap": {
            "ordered_block_ids": global_blocks,
            "rng_seed": bootstrap_seed,
            "replicates": 100_000,
            "rng": "numpy.PCG64",
            "one_sided_quantile": 0.05,
            "unit": "synchronized_calendar_block",
            "index_matrix_sha256": next(iter(bootstrap_hashes)),
        },
        "test_accessed": True,
        "sealed_holdout_accessed": False,
        "quality_claim": False,
    }
    report_digest = hashlib.sha256(_canonical_bytes(report_payload)).hexdigest()
    report_path = output / f"evaluation-report-{report_digest}.json"
    _atomic_no_overwrite(report_path, report_payload)
    verdict_payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "promotion-verdict-v1",
        "status": "promoted" if verdict["promoted"] else "not_promoted",
        "promoted": verdict["promoted"],
        "failed_clauses": verdict["failed_clauses"],
        "config_sha256": config.digest,
        "serving_freeze_sha256": serving_sha256,
        "evaluation_plan_sha256": plan_sha256,
        "evaluation_report_path": str(report_path.resolve()),
        "evaluation_report_sha256": report_digest,
        "test_accessed": True,
        "sealed_holdout_accessed": False,
    }
    verdict_digest = hashlib.sha256(_canonical_bytes(verdict_payload)).hexdigest()
    verdict_path = output / f"promotion-verdict-{verdict_digest}.json"
    _atomic_no_overwrite(verdict_path, verdict_payload)
    return {
        "report_path": str(report_path),
        "report_sha256": report_digest,
        "verdict_path": str(verdict_path),
        "verdict_sha256": verdict_digest,
        **verdict_payload,
    }
