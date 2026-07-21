"""Manifest-backed deterministic environment qualification."""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import sklearn

from mantis_v2.downstream_config import TrendMagicStrategyConfig, load_downstream_config
from mantis_v2.rl_baselines import (
    EntryBaseline,
    HistoricalLogisticPolicy,
    MatchedRandomPolicy,
    RejectAllPolicy,
    SupervisedRows,
    TakeAllPolicy,
    fit_supervised_baselines,
    replay_policy,
)
from mantis_v2.rl_config import RlConfig
from mantis_v2.rl_environment import (
    BarData,
    CandidateData,
    EnvironmentEpisode,
    TopstepEntryEnvironment,
)
from mantis_v2.rl_episodes import _verified_path, read_observation
from mantis_v2.rl_provenance import sha256_file
from mantis_v2.strategy import load_market_frame, trend_magic_state
from mantis_v2.walk_forward import PortableHead


class EnvironmentValidationError(RuntimeError):
    """Raised when environment qualification cannot prove its contract."""


_ORACLE_ECONOMICS = (
    ("ES", "one_mini", 1, 50.0, 12.5, 3.78),
    ("ES", "ten_micros", 10, 50.0, 12.5, 12.20),
    ("NQ", "one_mini", 1, 20.0, 5.0, 3.78),
    ("NQ", "ten_micros", 10, 20.0, 5.0, 12.20),
    ("RTY", "one_mini", 1, 50.0, 5.0, 3.78),
    ("RTY", "ten_micros", 10, 50.0, 5.0, 12.20),
    ("YM", "one_mini", 1, 5.0, 5.0, 3.78),
    ("YM", "ten_micros", 10, 5.0, 5.0, 12.20),
    ("GC", "one_mini", 1, 100.0, 10.0, 4.32),
    ("GC", "ten_micros", 10, 100.0, 10.0, 19.20),
    ("CL", "one_mini", 1, 1000.0, 10.0, 4.02),
    ("CL", "ten_micros", 10, 1000.0, 10.0, 15.20),
    ("ZB", "one_mini", 1, 1000.0, 31.25, 2.76),
)


def _oracle_candidate() -> CandidateData:
    return CandidateData(np.array([1.0, -1.0], dtype=np.float32), 1, 99.0, 2.0, 1, 1)


def _independent_replay_oracles(config: RlConfig) -> dict[str, Any]:
    """Compare production transitions with literal, independently calculated fixtures."""
    origin = datetime(2025, 1, 6, 15, 0, tzinfo=UTC)
    cases: list[dict[str, Any]] = []
    for ticker, profile, quantity, multiplier, tick_value, fee in _ORACLE_ECONOMICS:
        bars = (
            BarData(origin, 100.0, 100.2, 99.8, 100.0, 1.0, _oracle_candidate()),
            BarData(origin + timedelta(minutes=3), 100.0, 100.5, 99.5, 100.0, 1.0),
            BarData(origin + timedelta(minutes=6), 100.0, 101.9, 100.0, 101.5, 1.0),
            BarData(origin + timedelta(minutes=9), 101.5, 101.6, 99.5, 100.0, 1.0),
            BarData(origin + timedelta(minutes=12), 100.0, 103.1, 100.0, 103.0, 1.0),
            BarData(origin + timedelta(minutes=15), 103.0, 103.0, 102.0, 102.5, 1.0),
        )
        environment = TopstepEntryEnvironment(config, EnvironmentEpisode(ticker, profile, bars))
        reset_observation, _ = environment.reset(seed=7)
        actions = (1, 0, 0, 0, 0)
        submitted_actions: list[int] = []
        observations = []
        infos = []
        for action in actions:
            observation, _, terminated, truncated, info = environment.step(action)
            submitted_actions.append(action)
            observations.append(observation)
            infos.append(info)
            if terminated or truncated:
                break
        expected = 100_000.0 + 2.35 * multiplier - 2.0 * tick_value - fee
        state = environment.account_state
        checks = {
            "quantity": math.isclose(
                float(reset_observation.vector[environment.observation_schema.index("quantity")]),
                float(quantity),
                abs_tol=1e-6,
            ),
            "actions": tuple(submitted_actions) == actions,
            "accepted_trades": state["accepted_trades"] == 1,
            "terminal_status": state["status"] == "TIMEOUT",
            "fill_timestamp": infos[0].get("fill_timestamp") == bars[1].timestamp.isoformat(),
            "fill_price": infos[0].get("fill_price") == 100.0,
            "fee": infos[0].get("booked_round_trip_fee") == fee,
            "pre_activation_retrace": observations[2].positioned == 1.0,
            "trail_exit": infos[-1].get("event") == "STOP",
            "balance": math.isclose(cast(float, state["balance"]), expected, abs_tol=1e-6),
            "equity": math.isclose(cast(float, state["equity"]), expected, abs_tol=1e-6),
        }
        if not all(checks.values()):
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            raise EnvironmentValidationError(
                f"independent replay oracle failed: {ticker} {profile}: {failed}"
            )
        cases.append(
            {
                "ticker": ticker,
                "profile": profile,
                "quantity": quantity,
                "actions": submitted_actions,
                "accepted_trades": state["accepted_trades"],
                "status": state["status"],
                "checks": checks,
            }
        )

    def lifecycle(name: str, episode: EnvironmentEpisode, expected: dict[str, Any]) -> None:
        environment = TopstepEntryEnvironment(config, episode)
        environment.reset(seed=9)
        infos = []
        while True:
            action = int(environment.action_mask()[1])
            _, _, terminated, truncated, info = environment.step(action)
            infos.append(info)
            if terminated or truncated:
                break
        state = environment.account_state
        if expected.get("status") is not None and state["status"] != expected["status"]:
            raise EnvironmentValidationError(f"independent replay oracle failed: {name}")
        if expected.get("event") is not None and expected["event"] not in {
            item.get("event") for item in infos
        }:
            raise EnvironmentValidationError(f"independent replay oracle failed: {name}")
        for key in ("balance", "equity", "best_day_profit", "consistency_ratio"):
            if key in expected and not math.isclose(
                cast(float, state[key]), float(expected[key]), abs_tol=1e-6
            ):
                raise EnvironmentValidationError(f"independent replay oracle failed: {name}")
        cases.append(
            {"name": name, "status": state["status"], "events": [i.get("event") for i in infos]}
        )

    discontinuity = EnvironmentEpisode(
        "ES",
        "one_mini",
        (
            BarData(origin, 100, 100, 100, 100, 1, _oracle_candidate()),
            BarData(origin + timedelta(minutes=3), 100, 100, 100, 100, 1, discontinuity=True),
        ),
    )
    lifecycle(
        "rollover_discontinuity",
        discontinuity,
        {"status": "TIMEOUT", "event": "DISCONTINUITY_RESET"},
    )
    mll = EnvironmentEpisode(
        "NQ",
        "one_mini",
        (
            BarData(origin, 100, 100, 100, 100, 1, _oracle_candidate()),
            BarData(origin + timedelta(minutes=3), 100, 100, 100, 100, 1),
            BarData(origin + timedelta(minutes=6), -100, -100, -100, -100, 1),
        ),
    )
    lifecycle("mll_blow", mll, {"status": "BLOW", "event": "MLL_BLOW", "balance": 95_986.22})
    dll = EnvironmentEpisode(
        "ES",
        "one_mini",
        (
            BarData(origin, 100, 100, 100, 100, 1, _oracle_candidate()),
            BarData(origin + timedelta(minutes=3), 100, 100, 100, 100, 1),
            BarData(origin + timedelta(minutes=6), 55, 55, 55, 55, 1),
        ),
    )
    lifecycle(
        "dll_lockout", dll, {"status": "TIMEOUT", "event": "DLL_LOCKOUT", "balance": 97_721.22}
    )
    flat_origin = datetime(2025, 1, 6, 21, 6, tzinfo=UTC)
    session_flat = EnvironmentEpisode(
        "ES",
        "one_mini",
        (
            BarData(flat_origin, 100, 100, 100, 100, 1, _oracle_candidate()),
            BarData(flat_origin + timedelta(minutes=3), 100, 100.5, 99.5, 100, 1),
            BarData(flat_origin + timedelta(minutes=6), 101, 101, 101, 101, 1),
        ),
    )
    lifecycle(
        "session_flatten",
        session_flat,
        {
            "status": "TIMEOUT",
            "event": "SESSION_FLATTEN",
            "balance": 100_021.22,
            "equity": 100_021.22,
            "best_day_profit": 21.22,
            "consistency_ratio": 1.0,
        },
    )
    return {"passed": True, "cases": cases, "case_count": len(cases)}


@dataclass(frozen=True)
class FeatureRef:
    path: Path
    row: int
    width: int


@dataclass(frozen=True)
class LoadedEpisodes:
    episodes: tuple[EnvironmentEpisode, ...]
    feature_refs: tuple[FeatureRef, ...]
    manifest_sha256: str
    partition: str
    fold: int = 0


def _manifest(path: Path, config: RlConfig) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentValidationError(f"invalid episode manifest: {path}") from exc
    if raw.get("schema_version") != 1 or raw.get("stage") != "rl-episode-schedule":
        raise EnvironmentValidationError("episode manifest contract is invalid")
    partition = raw.get("partition")
    if not isinstance(partition, dict) or partition.get("name") not in {
        "training",
        "validation",
    }:
        raise EnvironmentValidationError("environment validation accepts training/validation only")
    end = pd.Timestamp(partition.get("end"))
    if end.tzinfo is None or end >= pd.Timestamp(config.evaluation.sealed_holdout_start):
        raise EnvironmentValidationError("episode manifest reaches the sealed holdout")
    identities = raw.get("identities")
    if not isinstance(identities, dict):
        raise EnvironmentValidationError("episode manifest identities are missing")
    configured = identities.get("config")
    if not isinstance(configured, dict) or configured.get("sha256") != config.digest:
        raise EnvironmentValidationError("episode manifest config identity mismatch")
    embedding = identities.get("embedding")
    if (
        not isinstance(embedding, dict)
        or embedding.get("sha256") != config.upstream.embedding_manifest_sha256
    ):
        raise EnvironmentValidationError("episode manifest embedding identity mismatch")
    return cast(dict[str, Any], raw)


def _direction_age(direction: np.ndarray) -> np.ndarray:
    ages = np.zeros(len(direction), dtype=np.int64)
    for index in range(1, len(direction)):
        ages[index] = ages[index - 1] + 1 if direction[index] == direction[index - 1] else 0
    return ages


def load_episode_manifest(
    config: RlConfig, path: Path, repository_root: Path | None = None
) -> LoadedEpisodes:
    """Load complete bar episodes and mmap candidate embeddings from a schedule manifest."""
    root = repository_root.resolve() if repository_root else Path(__file__).resolve().parents[3]
    raw = _manifest(path, config)
    downstream_path = config.upstream.downstream_config_path
    if not downstream_path.is_absolute():
        downstream_path = root / downstream_path
    downstream = load_downstream_config(downstream_path)
    strategy = downstream.strategy
    if not isinstance(strategy, TrendMagicStrategyConfig):
        raise EnvironmentValidationError("entry environment requires Trend Magic")
    embedding = raw.get("embedding")
    if not isinstance(embedding, dict) or not isinstance(embedding.get("outputs"), list):
        raise EnvironmentValidationError("episode manifest embedding outputs are missing")
    width = embedding.get("feature_width")
    if not isinstance(width, int) or width < 1:
        raise EnvironmentValidationError("episode manifest feature width is invalid")
    outputs = {int(item["number"]): item for item in embedding["outputs"]}
    market_cache: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    episodes: list[EnvironmentEpisode] = []
    feature_refs: list[FeatureRef] = []
    raw_episodes = raw.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise EnvironmentValidationError("episode manifest has no episodes")
    for episode in raw_episodes:
        if not isinstance(episode, dict):
            raise EnvironmentValidationError("episode entry must be an object")
        ticker = str(episode.get("ticker"))
        if ticker not in market_cache:
            frame = load_market_frame(downstream, ticker, downstream.data.decision_timeframe)
            line, direction = trend_magic_state(
                frame,
                strategy.cci_period,
                strategy.trend_magic_atr_period,
                strategy.trend_magic_multiplier,
            )
            market_cache[ticker] = (frame, line, _direction_age(direction))
        frame, line, ages = market_cache[ticker]
        candidates: dict[int, CandidateData] = {}
        spans = episode.get("observation_spans")
        if not isinstance(spans, list) or not spans:
            raise EnvironmentValidationError("episode has no observation spans")
        for span in spans:
            shard = int(span["shard"])
            first = int(span["first_row"])
            count = int(span["row_count"])
            output = outputs.get(shard)
            if output is None:
                raise EnvironmentValidationError(f"unknown embedding shard: {shard}")
            features_path = _verified_path(
                output["features"], path.parent, f"features shard {shard}"
            )
            metadata_path = _verified_path(
                output["metadata"], path.parent, f"metadata shard {shard}"
            )
            metadata = pd.read_parquet(metadata_path).iloc[first : first + count]
            features = np.load(features_path, mmap_mode="r", allow_pickle=False)
            if len(metadata) != count or features.shape[1] != width:
                raise EnvironmentValidationError("episode observation span is out of range")
            for offset, row in enumerate(metadata.itertuples(index=False)):
                feature_row = first + offset
                decision_index = int(row.decision_index)
                candidates[decision_index] = CandidateData(
                    np.asarray(features[feature_row], dtype=np.float32),
                    int(row.direction),
                    float(line[decision_index]),
                    float(row.atr),
                    int(ages[decision_index]),
                    int(row.label),
                )
                feature_refs.append(FeatureRef(features_path, feature_row, width))
        start = pd.Timestamp(episode["decision_start"])
        end = pd.Timestamp(episode["terminal_end"])
        selected = frame[(frame["close_timestamp"] >= start) & (frame["close_timestamp"] <= end)]
        if selected.empty:
            raise EnvironmentValidationError("episode has no owned 3-minute bars")
        bars = tuple(
            BarData(
                pd.Timestamp(row.close_timestamp).to_pydatetime(),
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume),
                candidates.get(int(index)),
                bool(row["_discontinuity"]),
            )
            for index, row in selected.iterrows()
        )
        if not any(bar.candidate is not None for bar in bars):
            raise EnvironmentValidationError("episode bars do not contain scheduled candidates")
        episodes.append(EnvironmentEpisode(ticker, str(episode.get("profile")), bars))
    return LoadedEpisodes(
        tuple(episodes),
        tuple(feature_refs),
        sha256_file(path),
        str(raw["partition"]["name"]),
        int(raw.get("fold", -1)),
    )


def load_historical_logistic_policy(
    config: RlConfig, fold: int, repository_root: Path | None = None
) -> tuple[HistoricalLogisticPolicy, str]:
    """Load the exact rejected Trend Magic logistic head for one fold."""
    root = repository_root.resolve() if repository_root else Path(__file__).resolve().parents[3]
    downstream_path = config.upstream.downstream_config_path
    if not downstream_path.is_absolute():
        downstream_path = root / downstream_path
    downstream = load_downstream_config(downstream_path)
    manifest_path = (
        downstream.run.artifact_root / downstream.run.name / "walk-forward" / "manifest.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentValidationError("historical logistic manifest is invalid") from exc
    if manifest.get("stage") != "walk-forward" or not isinstance(manifest.get("folds"), list):
        raise EnvironmentValidationError("historical logistic manifest contract is invalid")
    matching = [item for item in manifest["folds"] if item.get("fold", {}).get("number") == fold]
    if len(matching) != 1:
        raise EnvironmentValidationError(f"historical logistic head missing for fold {fold}")
    identity = matching[0].get("head")
    head_path = _verified_path(identity, manifest_path.parent, f"historical head fold {fold}")
    with np.load(head_path, allow_pickle=False) as values:
        required = {"scaler_mean", "scaler_scale", "coefficients", "intercept", "threshold"}
        if set(values.files) != required:
            raise EnvironmentValidationError("historical logistic head contract is invalid")
        head = PortableHead(
            np.asarray(values["scaler_mean"], dtype=np.float64),
            np.asarray(values["scaler_scale"], dtype=np.float64),
            np.asarray(values["coefficients"], dtype=np.float64),
            np.asarray(values["intercept"], dtype=np.float64),
            float(values["threshold"][0]),
        )
    return HistoricalLogisticPolicy(head), sha256_file(head_path)


def _supervised_rows(config: RlConfig, loaded: LoadedEpisodes) -> SupervisedRows:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for episode in loaded.episodes:
        environment = TopstepEntryEnvironment(config, episode)
        observation, _ = environment.reset(seed=config.run.seed)
        while True:
            if environment.action_mask()[1]:
                label = environment.current_label
                assert label is not None
                features.append(observation.vector)
                labels.append(label)
            observation, _, terminated, truncated, _ = environment.step(0)
            if terminated or truncated:
                break
    return SupervisedRows(
        np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int8), loaded.partition
    )


def _mmap_benchmark(refs: tuple[FeatureRef, ...]) -> dict[str, float | int]:
    if not refs:
        raise EnvironmentValidationError("no mmap feature references to benchmark")
    timings: list[float] = []
    for ref in (refs * (1000 // len(refs) + 1))[:1000]:
        started = time.perf_counter_ns()
        read_observation(ref.path, ref.row, ref.width)
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    p95 = float(np.percentile(timings, 95))
    return {
        "samples": len(timings),
        "p95_ms": p95,
        "decision_interval_ms": 180_000.0,
        "headroom_x": 180_000.0 / p95,
    }


def _throughput_benchmark(config: RlConfig, episode: EnvironmentEpisode) -> dict[str, float | int]:
    transitions = 0
    started = time.perf_counter()
    while transitions < 10_000:
        environment = TopstepEntryEnvironment(config, episode)
        _, _ = environment.reset(seed=config.run.seed)
        while transitions < 10_000:
            _, _, terminated, truncated, _ = environment.step(0)
            transitions += 1
            if terminated or truncated:
                break
    elapsed = time.perf_counter() - started
    return {
        "steps": transitions,
        "elapsed_seconds": elapsed,
        "steps_per_second": transitions / elapsed,
    }


def validate_environment(
    config: RlConfig,
    training: LoadedEpisodes,
    validation: LoadedEpisodes,
    historical_policy: EntryBaseline | None = None,
    historical_head_sha256: str | None = None,
) -> dict[str, Any]:
    """Qualify finite observations, causality, baselines, performance, and provenance."""
    if training.partition != "training" or validation.partition != "validation":
        raise EnvironmentValidationError("environment validation requires training and validation")
    independent_oracles = _independent_replay_oracles(config)
    training_rows = _supervised_rows(config, training)
    validation_rows = _supervised_rows(config, validation)
    logistic, hist, fit = fit_supervised_baselines(
        training_rows, validation_rows, seed=config.run.seed
    )
    historical = historical_policy if historical_policy is not None else logistic
    baseline_results: list[dict[str, Any]] = []
    mask_comparisons = 0
    mask_mismatches = 0
    for episode in validation.episodes:
        learned = replay_policy(config, episode, hist)
        matched = None
        for offset in range(100):
            candidate = replay_policy(
                config,
                episode,
                MatchedRandomPolicy(
                    take_count=learned.accepted_trades,
                    legal_opportunities=sum(bar.candidate is not None for bar in episode.bars),
                    seed=config.run.seed + offset,
                ),
            )
            if candidate.accepted_trades == learned.accepted_trades:
                matched = candidate
                break
        if matched is None:
            matched = replay_policy(
                config,
                episode,
                MatchedRandomPolicy(
                    take_count=learned.accepted_trades,
                    legal_opportunities=sum(bar.candidate is not None for bar in episode.bars),
                    seed=config.run.seed,
                    probability=1.0,
                ),
            )
        if matched.accepted_trades != learned.accepted_trades:
            raise EnvironmentValidationError("random baseline cannot match learned participation")
        policies = [
            RejectAllPolicy(),
            TakeAllPolicy(),
            historical,
            hist,
        ]
        replays = [asdict(replay_policy(config, episode, policy)) for policy in policies]
        replays.insert(2, asdict(matched))
        baseline_results.append(
            {
                "ticker": episode.ticker,
                "profile": episode.profile,
                "results": replays,
            }
        )
        environment = TopstepEntryEnvironment(config, episode)
        observation, _ = environment.reset(seed=config.run.seed)
        while True:
            mask = environment.action_mask()
            mask_comparisons += 1
            mask_mismatches += int(not np.array_equal(observation.action_mask, mask))
            observation, _, terminated, truncated, _ = environment.step(0)
            if terminated or truncated:
                break
    first = validation.episodes[0]
    original, _ = TopstepEntryEnvironment(config, first).reset(seed=config.run.seed)
    changed_last = replace(first.bars[-1], close=first.bars[-1].close * 1.5)
    changed = replace(first, bars=(*first.bars[:-1], changed_last))
    altered, _ = TopstepEntryEnvironment(config, changed).reset(seed=config.run.seed)
    causal = bool(np.array_equal(original.vector, altered.vector))
    mmap = _mmap_benchmark(validation.feature_refs)
    throughput = _throughput_benchmark(config, first)
    if not causal:
        raise EnvironmentValidationError("future-bar mutation changed a prior observation")
    if mmap["headroom_x"] < 100.0:
        raise EnvironmentValidationError("mmap observation latency lacks 100x headroom")
    if throughput["steps_per_second"] < 5_000.0:
        raise EnvironmentValidationError("environment throughput is below 5,000 steps/second")
    return {
        "schema_version": 1,
        "stage": "rl-environment-validation",
        "sealed_holdout_accessed": False,
        "inputs": {
            "training_manifest_sha256": training.manifest_sha256,
            "validation_manifest_sha256": validation.manifest_sha256,
            "config_sha256": config.digest,
        },
        "finite_observations": bool(
            np.isfinite(training_rows.features).all()
            and np.isfinite(validation_rows.features).all()
        ),
        "causal_prefix": causal,
        "shared_action_mask": mask_mismatches == 0,
        "action_mask_parity": {"comparisons": mask_comparisons, "mismatches": mask_mismatches},
        "baseline_fit": asdict(fit),
        "independent_replay_oracles": independent_oracles,
        "historical_head_sha256": historical_head_sha256,
        "baseline_replays": baseline_results,
        "benchmark": {"mmap_fetch": mmap, "environment": throughput},
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def write_environment_validation(
    config: RlConfig,
    training_manifest: Path,
    validation_manifest: Path,
    output: Path,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Load manifest-backed episodes and atomically write qualification evidence."""
    training = load_episode_manifest(config, training_manifest, repository_root)
    validation = load_episode_manifest(config, validation_manifest, repository_root)
    if training.fold != validation.fold:
        raise EnvironmentValidationError("training and validation manifests use different folds")
    historical, historical_sha = load_historical_logistic_policy(
        config, training.fold, repository_root
    )
    payload = validate_environment(
        config,
        training,
        validation,
        historical_policy=historical,
        historical_head_sha256=historical_sha,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise EnvironmentValidationError(f"validation output already exists: {output}")
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
    except FileExistsError as exc:
        raise EnvironmentValidationError(f"validation output already exists: {output}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return payload
