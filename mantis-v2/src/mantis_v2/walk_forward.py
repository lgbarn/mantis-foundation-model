"""Purged chronological folds and training-only logistic probes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from mantis_v2.downstream_config import DownstreamConfig


class WalkForwardContractError(ValueError):
    """Raised when a fold cannot be built or evaluated safely."""


@dataclass(frozen=True)
class Fold:
    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class PortableHead:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    threshold: float


def build_folds(metadata: pd.DataFrame, config: DownstreamConfig) -> list[Fold]:
    """Build fixed rolling month windows before the sealed holdout."""
    if metadata.empty:
        raise WalkForwardContractError("embedding metadata is empty")
    decisions = pd.to_datetime(metadata["decision_ts"], utc=True, errors="raise")
    non_holdout = decisions[decisions < pd.Timestamp(config.data.holdout_start)]
    if non_holdout.empty:
        raise WalkForwardContractError("no pre-holdout rows are available")
    start = non_holdout.min().to_period("M").start_time.tz_localize("UTC")
    limit = min(
        non_holdout.max() + pd.Timedelta(nanoseconds=1), pd.Timestamp(config.data.holdout_start)
    )
    walk = config.walk_forward
    folds: list[Fold] = []
    number = 0
    while True:
        train_end = start + pd.DateOffset(months=walk.train_months)
        validation_end = train_end + pd.DateOffset(months=walk.validation_months)
        test_end = validation_end + pd.DateOffset(months=walk.test_months)
        if test_end > limit:
            break
        folds.append(
            Fold(
                number=number,
                train_start=start,
                train_end=train_end,
                validation_start=train_end,
                validation_end=validation_end,
                test_start=validation_end,
                test_end=test_end,
            )
        )
        number += 1
        start += pd.DateOffset(months=walk.stride_months)
    if not folds:
        raise WalkForwardContractError(
            "data span is too short for the configured walk-forward fold"
        )
    return folds


def fold_masks(
    metadata: pd.DataFrame, fold: Fold, config: DownstreamConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply event-span purging and a context embargo at each fold boundary."""
    decision = pd.to_datetime(metadata["decision_ts"], utc=True, errors="raise")
    label_end = pd.to_datetime(metadata["label_end_ts"], utc=True, errors="raise")
    embargo = timedelta(seconds=180 * int(config.walk_forward.embargo_bars))
    train = (
        (decision >= fold.train_start)
        & (decision < fold.train_end - embargo)
        & (label_end < fold.train_end - embargo)
    )
    validation = (
        (decision >= fold.validation_start + embargo)
        & (decision < fold.validation_end - embargo)
        & (label_end < fold.validation_end - embargo)
    )
    test = (
        (decision >= fold.test_start + embargo)
        & (decision < fold.test_end)
        & (label_end < fold.test_end)
    )
    return train.to_numpy(), validation.to_numpy(), test.to_numpy()


def deterministic_cap(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """Select a stable, class-agnostic subset without changing temporal ownership."""
    if len(indices) <= maximum:
        return indices
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(indices, size=maximum, replace=False))


def fit_logistic_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    config: DownstreamConfig,
) -> tuple[PortableHead, dict[str, float | None]]:
    """Fit scaler/classifier on train only and choose a threshold on validation only."""
    if len(np.unique(train_labels)) != 2:
        raise WalkForwardContractError("training fold must contain both label classes")
    if not np.isfinite(train_features).all() or not np.isfinite(validation_features).all():
        raise WalkForwardContractError("embedding features must be finite")
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    class_weight = None if config.walk_forward.class_weight == "none" else "balanced"
    classifier = LogisticRegression(
        max_iter=config.walk_forward.max_iter,
        class_weight=class_weight,
        random_state=config.run.seed,
    )
    classifier.fit(scaled_train, train_labels)
    probability = classifier.predict_proba(scaler.transform(validation_features))[:, 1]
    threshold = float(np.percentile(probability, config.walk_forward.threshold_percentile))
    metrics = probability_metrics(validation_labels, probability)
    metrics["threshold"] = threshold
    head = PortableHead(
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        coefficients=np.asarray(classifier.coef_, dtype=np.float64),
        intercept=np.asarray(classifier.intercept_, dtype=np.float64),
        threshold=threshold,
    )
    return head, metrics


def predict_head(head: PortableHead, features: np.ndarray) -> np.ndarray:
    """Apply a portable binary logistic head without pickle."""
    scaled = (features - head.scaler_mean) / head.scaler_scale
    logits = scaled @ head.coefficients[0] + head.intercept[0]
    positive = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    return np.asarray(positive, dtype=np.float64)


def probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    """Return primary proper scores and diagnostic ranking metrics."""
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise WalkForwardContractError("predicted probabilities must be finite and in [0, 1]")
    labels = np.asarray(labels, dtype=np.int8)
    counts = np.bincount(labels, minlength=2)
    sample_weight = np.ones(len(labels), dtype=np.float64)
    for label, count in enumerate(counts):
        if count:
            sample_weight[labels == label] = len(labels) / (2.0 * count)
    result = {
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "brier": float(np.mean((probability - labels) ** 2)),
        "weighted_log_loss": float(
            log_loss(labels, probability, labels=[0, 1], sample_weight=sample_weight)
        ),
        "weighted_brier": float(np.average((probability - labels) ** 2, weights=sample_weight)),
        "roc_auc": None,
        "pr_auc": None,
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, probability))
        result["pr_auc"] = float(average_precision_score(labels, probability))
    return result
