"""Small supervised continuation/reversal entry head for frozen embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from mantis_v2.downstream_config import DownstreamConfig
from mantis_v2.walk_forward import WalkForwardContractError, probability_metrics


@dataclass(frozen=True)
class SupervisedExpertHead:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    trunk_weight: np.ndarray
    trunk_bias: np.ndarray
    continuation_weight: np.ndarray
    continuation_bias: float
    reversal_weight: np.ndarray
    reversal_bias: float
    risk_weight: np.ndarray
    risk_bias: float
    thresholds: np.ndarray
    risk_penalty: float
    symbols: tuple[str, ...]

    @property
    def threshold(self) -> float:
        """Return the mean threshold for summary-only compatibility."""
        return float(np.mean(self.thresholds))


@dataclass(frozen=True)
class SupervisedFitDiagnostics:
    converged: bool
    epochs_ran: int
    best_epoch: int
    best_validation_loss: float
    device: str


class _ExpertNetwork(nn.Module):
    def __init__(self, input_width: int, hidden_width: int) -> None:
        super().__init__()
        self.trunk = nn.Linear(input_width, hidden_width)
        self.continuation = nn.Linear(hidden_width, 1)
        self.reversal = nn.Linear(hidden_width, 1)
        self.risk = nn.Linear(hidden_width, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = torch.relu(self.trunk(values))
        return (
            self.continuation(hidden).squeeze(1),
            self.reversal(hidden).squeeze(1),
            self.risk(hidden).squeeze(1),
        )


def build_context_features(
    metadata: pd.DataFrame, symbols: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Build causal row context and the exact contiguous direction-flip mask."""
    annotated = annotate_supervised_context(metadata)
    symbol = annotated["symbol"].astype(str)
    direction = annotated["direction"].to_numpy(dtype=np.float32)
    reversal = annotated["_expert_reversal"].to_numpy(dtype=bool)
    age = annotated["_expert_age"].to_numpy(dtype=np.float32)
    timestamp = pd.to_datetime(annotated["decision_ts"], utc=True, errors="raise")
    minute = (timestamp.dt.hour * 60 + timestamp.dt.minute).to_numpy(dtype=np.float32)
    angle = minute * (2.0 * np.pi / 1440.0)
    one_hot = np.column_stack([(symbol == item).to_numpy(dtype=np.float32) for item in symbols])
    context = np.column_stack(
        (
            direction,
            reversal.astype(np.float32),
            age,
            np.sin(angle),
            np.cos(angle),
            one_hot,
        )
    )
    return np.asarray(context, dtype=np.float32), reversal


def annotate_supervised_context(metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach causal role state before deterministic row capping or sharding."""
    required = {"symbol", "decision_index", "decision_ts", "direction"}
    missing = required - set(metadata.columns)
    if missing:
        raise WalkForwardContractError(f"supervised metadata is missing: {sorted(missing)}")
    if {"_expert_reversal", "_expert_age"} <= set(metadata.columns):
        return metadata
    annotated = metadata.copy()
    symbol = metadata["symbol"].astype(str)
    direction = metadata["direction"].to_numpy(dtype=np.float32)
    index = metadata["decision_index"].to_numpy(dtype=np.int64)
    same_symbol = symbol.eq(symbol.shift(1)).to_numpy()
    contiguous = np.zeros(len(metadata), dtype=bool)
    if len(metadata) > 1:
        contiguous[1:] = index[1:] == index[:-1] + 1
    reversal = np.zeros(len(metadata), dtype=bool)
    if len(metadata) > 1:
        reversal[1:] = same_symbol[1:] & contiguous[1:] & (direction[1:] != direction[:-1])

    flip_group = pd.Series(reversal, index=metadata.index).groupby(symbol, sort=False).cumsum()
    age = metadata.groupby([symbol, flip_group], sort=False).cumcount().to_numpy(dtype=np.float32)
    age = np.log1p(np.minimum(age, 256.0)) / np.log(257.0)
    annotated["_expert_reversal"] = reversal
    annotated["_expert_age"] = age
    return annotated


def early_stop_labels(metadata: pd.DataFrame, bars: int) -> np.ndarray:
    required = {"entry_ts", "label_end_ts", "reward_r"}
    missing = required - set(metadata.columns)
    if missing:
        raise WalkForwardContractError(f"supervised metadata is missing: {sorted(missing)}")
    entry = pd.to_datetime(metadata["entry_ts"], utc=True, errors="raise")
    end = pd.to_datetime(metadata["label_end_ts"], utc=True, errors="raise")
    fast = (end - entry) <= pd.to_timedelta(3 * bars, unit="min")
    stopped = metadata["reward_r"].to_numpy(dtype=np.float32) <= -1.0
    return np.asarray(fast.to_numpy() & stopped, dtype=np.float32)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise WalkForwardContractError("supervised_head.device=cuda is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise WalkForwardContractError("supervised_head.device=mps is unavailable")
    return torch.device(name)


def _balanced_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positives = torch.count_nonzero(labels).to(dtype=torch.float32)
    negatives = labels.numel() - positives
    weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), 0.25, 4.0)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)


def _loss(
    outputs: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    reversal: torch.Tensor,
    risk: torch.Tensor,
) -> torch.Tensor:
    continuation_logits, reversal_logits, risk_logits = outputs
    continuation_mask = ~reversal
    pieces = [_balanced_loss(risk_logits, risk)]
    if torch.any(continuation_mask):
        pieces.append(
            _balanced_loss(continuation_logits[continuation_mask], labels[continuation_mask])
        )
    if torch.any(reversal):
        pieces.append(_balanced_loss(reversal_logits[reversal], labels[reversal]))
    return torch.stack(pieces).mean()


def fit_supervised_expert_head(
    train_embeddings: np.ndarray,
    train_metadata: pd.DataFrame,
    validation_embeddings: np.ndarray,
    validation_metadata: pd.DataFrame,
    config: DownstreamConfig,
) -> tuple[SupervisedExpertHead, dict[str, float | None], SupervisedFitDiagnostics]:
    """Fit the compact expert head on train rows and select on validation rows."""
    settings = config.supervised_head
    if settings is None:
        raise WalkForwardContractError("supervised head configuration is required")
    train_context, train_reversal = build_context_features(train_metadata, config.data.symbols)
    validation_context, validation_reversal = build_context_features(
        validation_metadata, config.data.symbols
    )
    train = np.column_stack((train_embeddings, train_context)).astype(np.float32)
    validation = np.column_stack((validation_embeddings, validation_context)).astype(np.float32)
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise WalkForwardContractError("supervised head features must be finite")
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-7] = 1.0
    train = (train - mean) / scale
    validation = (validation - mean) / scale

    device = _device(settings.device)
    torch.manual_seed(config.run.seed)
    model = _ExpertNetwork(train.shape[1], settings.hidden_width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    train_x = torch.from_numpy(train)
    train_y = torch.from_numpy(train_metadata["label"].to_numpy(dtype=np.float32))
    train_role = torch.from_numpy(train_reversal)
    train_risk = torch.from_numpy(early_stop_labels(train_metadata, settings.early_stop_bars))
    validation_x = torch.from_numpy(validation).to(device)
    validation_y = torch.from_numpy(validation_metadata["label"].to_numpy(dtype=np.float32)).to(
        device
    )
    validation_role = torch.from_numpy(validation_reversal).to(device)
    validation_risk = torch.from_numpy(
        early_stop_labels(validation_metadata, settings.early_stop_bars)
    ).to(device)
    generator = torch.Generator().manual_seed(config.run.seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    epochs_ran = 0
    for epoch in range(settings.epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), settings.batch_size):
            selected = order[start : start + settings.batch_size]
            batch_x = train_x[selected].to(device)
            batch_y = train_y[selected].to(device)
            batch_role = train_role[selected].to(device)
            batch_risk = train_risk[selected].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(batch_x), batch_y, batch_role, batch_risk)
            torch.autograd.backward(loss)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                _loss(model(validation_x), validation_y, validation_role, validation_risk).item()
            )
        epochs_ran = epoch + 1
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if best_state is None:
        raise WalkForwardContractError("supervised head produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    state = model.state_dict()
    provisional = SupervisedExpertHead(
        scaler_mean=mean,
        scaler_scale=scale,
        trunk_weight=state["trunk.weight"].detach().cpu().numpy(),
        trunk_bias=state["trunk.bias"].detach().cpu().numpy(),
        continuation_weight=state["continuation.weight"].detach().cpu().numpy()[0],
        continuation_bias=float(state["continuation.bias"].detach().cpu().numpy()[0]),
        reversal_weight=state["reversal.weight"].detach().cpu().numpy()[0],
        reversal_bias=float(state["reversal.bias"].detach().cpu().numpy()[0]),
        risk_weight=state["risk.weight"].detach().cpu().numpy()[0],
        risk_bias=float(state["risk.bias"].detach().cpu().numpy()[0]),
        thresholds=np.full(len(config.data.symbols), 0.5, dtype=np.float64),
        risk_penalty=settings.risk_penalty,
        symbols=config.data.symbols,
    )
    probability = predict_supervised_expert_head(
        provisional, validation_embeddings, validation_metadata
    )
    thresholds = select_supervised_thresholds(
        probability,
        validation_metadata,
        config.data.symbols,
        settings.target_trades_per_symbol_day,
    )
    head = SupervisedExpertHead(**{**provisional.__dict__, "thresholds": thresholds})
    metrics = probability_metrics(validation_metadata["label"].to_numpy(dtype=np.int8), probability)
    metrics["threshold"] = head.threshold
    diagnostics = SupervisedFitDiagnostics(
        converged=True,
        epochs_ran=epochs_ran,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        device=str(device),
    )
    return head, metrics, diagnostics


def predict_supervised_expert_head(
    head: SupervisedExpertHead, embeddings: np.ndarray, metadata: pd.DataFrame
) -> np.ndarray:
    """Run the portable expert head without pickle or a live torch module."""
    context, reversal = build_context_features(metadata, head.symbols)
    values = np.column_stack((embeddings, context)).astype(np.float32)
    values = (values - head.scaler_mean) / head.scaler_scale
    hidden = np.maximum(values @ head.trunk_weight.T + head.trunk_bias, 0.0)
    continuation = hidden @ head.continuation_weight + head.continuation_bias
    reversed_logits = hidden @ head.reversal_weight + head.reversal_bias
    logits = np.where(reversal, reversed_logits, continuation)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    risk_logits = hidden @ head.risk_weight + head.risk_bias
    risk = 1.0 / (1.0 + np.exp(-np.clip(risk_logits, -50.0, 50.0)))
    probability *= 1.0 - head.risk_penalty * risk
    return np.asarray(np.clip(probability, 0.0, 1.0), dtype=np.float64)


def supervised_thresholds(head: SupervisedExpertHead, metadata: pd.DataFrame) -> np.ndarray:
    """Return each row's validation-owned instrument threshold."""
    lookup = dict(zip(head.symbols, head.thresholds, strict=True))
    mapped = metadata["symbol"].astype(str).map(lookup)
    if mapped.isna().any():
        raise WalkForwardContractError("supervised threshold symbol is unknown")
    return np.asarray(mapped.to_list(), dtype=np.float64)


def select_supervised_thresholds(
    probability: np.ndarray,
    metadata: pd.DataFrame,
    symbols: tuple[str, ...],
    target_trades_per_symbol_day: float,
) -> np.ndarray:
    """Select an independent validation threshold for each configured symbol."""
    if len(probability) != len(metadata) or not np.isfinite(probability).all():
        raise WalkForwardContractError("threshold inputs must be aligned and finite")
    day = pd.to_datetime(metadata["decision_ts"], utc=True).dt.date
    thresholds = np.ones(len(symbols), dtype=np.float64)
    for number, symbol in enumerate(symbols):
        mask = metadata["symbol"].astype(str).eq(symbol).to_numpy()
        if not np.any(mask):
            continue
        symbol_days = len(pd.unique(day[mask]))
        desired = max(1, int(np.ceil(symbol_days * target_trades_per_symbol_day)))
        symbol_probability = probability[mask]
        rank = min(desired, len(symbol_probability))
        thresholds[number] = float(
            np.partition(symbol_probability, len(symbol_probability) - rank)[
                len(symbol_probability) - rank
            ]
        )
    return thresholds
