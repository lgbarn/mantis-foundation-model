"""Deterministic entry baselines routed through the shared environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mantis_v2.rl_config import RlConfig
from mantis_v2.rl_environment import (
    EntryObservation,
    EnvironmentEpisode,
    TopstepEntryEnvironment,
)
from mantis_v2.walk_forward import PortableHead, predict_head


class EntryBaseline(Protocol):
    name: str

    def reset(self) -> None: ...

    def action(self, observation: EntryObservation, mask: np.ndarray) -> int: ...


class BaselineContractError(ValueError):
    """Raised when a supervised baseline would violate partition ownership."""


@dataclass(frozen=True)
class SupervisedRows:
    features: np.ndarray
    labels: np.ndarray
    partition: str


@dataclass(frozen=True)
class BaselineFitEvidence:
    training_rows: int
    validation_rows: int
    threshold_source: str
    logistic_threshold: float
    hist_gradient_boosting_threshold: float


@dataclass(frozen=True)
class HistGradientBoostingFitEvidence:
    training_rows: int
    validation_rows: int
    threshold_source: str
    threshold_selection: str
    threshold: float
    hyperparameters: dict[str, object]
    random_state: int


@dataclass(frozen=True)
class ReplayResult:
    policy: str
    actions: tuple[int, ...]
    accepted_trades: int
    status: str
    ending_balance: float


class RejectAllPolicy:
    name = "reject_all"

    def reset(self) -> None:
        return None

    def action(self, observation: EntryObservation, mask: np.ndarray) -> int:
        return 0


class TakeAllPolicy(RejectAllPolicy):
    name = "take_all"

    def action(self, observation: EntryObservation, mask: np.ndarray) -> int:
        return int(bool(mask[1]))


class MatchedRandomPolicy(RejectAllPolicy):
    name = "matched_random_take"

    def __init__(
        self, *, take_count: int, legal_opportunities: int, seed: int, probability: float = 0.75
    ) -> None:
        if take_count < 0 or legal_opportunities < 0 or take_count > legal_opportunities:
            raise ValueError("matched random counts must satisfy 0 <= take <= opportunities")
        self.take_count = take_count
        self.legal_opportunities = legal_opportunities
        self.seed = seed
        self.probability = probability

    def reset(self) -> None:
        generator = np.random.default_rng(self.seed)
        self._generator = generator
        self._accepted = 0

    def action(self, observation: EntryObservation, mask: np.ndarray) -> int:
        if not bool(mask[1]):
            return 0
        if self._accepted >= self.take_count:
            return 0
        result = int(self._generator.random() < self.probability)
        if result:
            self._accepted += 1
        return result


class _ProbabilityPolicy:
    def __init__(self, name: str, predict: object, threshold: float) -> None:
        self.name = name
        self._predict = predict
        self.threshold = threshold

    def reset(self) -> None:
        return None

    def action(self, observation: EntryObservation, mask: np.ndarray) -> int:
        if not bool(mask[1]):
            return 0
        probability = float(self._predict(np.asarray([observation.vector]))[0])  # type: ignore[operator]
        if not math.isfinite(probability):
            raise BaselineContractError(f"{self.name} produced a non-finite probability")
        return int(probability >= self.threshold)


class HistoricalLogisticPolicy:
    """Replay one immutable rejected downstream logistic head."""

    name = "historical_rejected_logistic_head"

    def __init__(self, head: PortableHead) -> None:
        self.head = head

    def reset(self) -> None:
        return None

    def action(self, observation: EntryObservation, mask: np.ndarray) -> int:
        if not bool(mask[1]):
            return 0
        width = len(self.head.scaler_mean)
        probability = float(predict_head(self.head, observation.vector[None, :width])[0])
        return int(probability >= self.head.threshold)


def _validated_rows(rows: SupervisedRows, expected: str) -> tuple[np.ndarray, np.ndarray]:
    if rows.partition != expected:
        raise BaselineContractError(f"expected {expected} rows, got {rows.partition}")
    features = np.asarray(rows.features, dtype=np.float64)
    labels = np.asarray(rows.labels, dtype=np.int8)
    if features.ndim != 2 or labels.shape != (len(features),) or not len(features):
        raise BaselineContractError(f"{expected} rows have invalid shapes")
    if not np.isfinite(features).all() or set(np.unique(labels)) - {0, 1}:
        raise BaselineContractError(f"{expected} rows must be finite with binary labels")
    if len(np.unique(labels)) != 2:
        raise BaselineContractError(f"{expected} rows must contain both classes")
    return features, labels


def _validation_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(probability)
    best_score = -math.inf
    best_threshold = 1.0
    for threshold in candidates:
        prediction = probability >= threshold
        positive = labels == 1
        negative = ~positive
        score = 0.5 * (float(prediction[positive].mean()) + float((~prediction[negative]).mean()))
        if score > best_score or (math.isclose(score, best_score) and threshold > best_threshold):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def fit_supervised_baselines(
    training: SupervisedRows,
    validation: SupervisedRows,
    *,
    seed: int,
) -> tuple[EntryBaseline, EntryBaseline, BaselineFitEvidence]:
    """Fit fixed contextual baselines on training and select thresholds on validation."""
    train_x, train_y = _validated_rows(training, "training")
    validation_x, validation_y = _validated_rows(validation, "validation")
    if train_x.shape[1] != validation_x.shape[1]:
        raise BaselineContractError("training and validation feature widths differ")
    scaler = StandardScaler().fit(train_x)
    logistic = LogisticRegression(
        C=0.0001,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    ).fit(scaler.transform(train_x), train_y)

    def logistic_predict(values: np.ndarray) -> np.ndarray:
        return np.asarray(logistic.predict_proba(scaler.transform(values))[:, 1], dtype=np.float64)

    logistic_probability = logistic_predict(validation_x)
    logistic_threshold = _validation_threshold(validation_y, logistic_probability)
    hist = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=2,
        l2_regularization=1.0,
        random_state=seed,
    ).fit(train_x, train_y)

    def hist_predict(values: np.ndarray) -> np.ndarray:
        return np.asarray(hist.predict_proba(values)[:, 1], dtype=np.float64)

    hist_probability = hist_predict(validation_x)
    hist_threshold = _validation_threshold(validation_y, hist_probability)
    return (
        _ProbabilityPolicy(
            "historical_rejected_logistic_head", logistic_predict, logistic_threshold
        ),
        _ProbabilityPolicy("hist_gradient_boosting_contextual", hist_predict, hist_threshold),
        BaselineFitEvidence(
            len(train_x),
            len(validation_x),
            "validation",
            logistic_threshold,
            hist_threshold,
        ),
    )


def fit_hist_gradient_boosting_baseline(
    training: SupervisedRows,
    validation: SupervisedRows,
    *,
    seed: int,
) -> tuple[EntryBaseline, HistGradientBoostingClassifier, HistGradientBoostingFitEvidence]:
    """Fit the fixed HGB on training and choose its threshold on validation only."""
    train_x, train_y = _validated_rows(training, "training")
    validation_x, validation_y = _validated_rows(validation, "validation")
    if train_x.shape[1] != validation_x.shape[1]:
        raise BaselineContractError("training and validation feature widths differ")
    hyperparameters: dict[str, object] = {
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 2,
        "l2_regularization": 1.0,
    }
    model = HistGradientBoostingClassifier(
        **hyperparameters,
        random_state=seed,
    ).fit(train_x, train_y)

    def predict(values: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict_proba(values)[:, 1], dtype=np.float64)

    threshold = _validation_threshold(validation_y, predict(validation_x))
    return (
        _ProbabilityPolicy("hist_gradient_boosting", predict, threshold),
        model,
        HistGradientBoostingFitEvidence(
            training_rows=len(train_x),
            validation_rows=len(validation_x),
            threshold_source="validation",
            threshold_selection="maximum_balanced_accuracy_then_highest_threshold_v1",
            threshold=threshold,
            hyperparameters=hyperparameters,
            random_state=seed,
        ),
    )


def replay_policy(
    config: RlConfig, episode: EnvironmentEpisode, policy: EntryBaseline
) -> ReplayResult:
    environment = TopstepEntryEnvironment(config, episode)
    observation, _ = environment.reset(seed=config.run.seed)
    policy.reset()
    actions: list[int] = []
    while True:
        mask = environment.action_mask()
        action = policy.action(observation, mask.copy())
        actions.append(action)
        observation, _, terminated, truncated, _ = environment.step(action)
        if terminated or truncated:
            break
    state = environment.account_state
    return ReplayResult(
        policy.name,
        tuple(actions),
        cast(int, state["accepted_trades"]),
        str(state["status"]),
        cast(float, state["balance"]),
    )
