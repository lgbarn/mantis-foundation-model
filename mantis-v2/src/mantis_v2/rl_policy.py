"""Ticker- and profile-conditioned entry-only actor/critic modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum

import numpy as np
import torch
from torch import nn

TICKERS = ("ES", "NQ", "RTY", "YM", "GC", "CL", "ZB")
PROFILES = ("one_mini", "ten_micros")


class PolicyVariant(StrEnum):
    """Preregistered actor/critic variants sharing one training seam."""

    INDEPENDENT_ACTOR = "independent_actor"
    SHARED_CRITIC = "shared_critic"
    SHARED_TICKER_VALUE = "shared_ticker_value"


class ReturnNormalizers:
    """Independent Welford return statistics owned by ticker."""

    def __init__(self, tickers: Iterable[str] = TICKERS) -> None:
        self._state = {ticker: [0, 0.0, 0.0] for ticker in tickers}

    def update(self, ticker: str, values: np.ndarray) -> None:
        if ticker not in self._state:
            raise ValueError(f"unknown return normalizer owner: {ticker}")
        for value in np.asarray(values, dtype=np.float64).reshape(-1):
            if not np.isfinite(value):
                raise ValueError("critic returns must be finite")
            state = self._state[ticker]
            state[0] += 1
            delta = float(value) - state[1]
            state[1] += delta / state[0]
            state[2] += delta * (float(value) - state[1])

    def normalize(self, ticker: str, values: torch.Tensor) -> torch.Tensor:
        count, mean, squared = self._state[ticker]
        variance = squared / max(count, 1)
        return (values - mean) / max(float(np.sqrt(variance)), 1e-6)

    def count(self, ticker: str) -> int:
        return int(self._state[ticker][0])

    def state_dict(self) -> dict[str, list[float | int]]:
        return {ticker: list(values) for ticker, values in self._state.items()}

    def load_state_dict(self, raw: Mapping[str, object]) -> None:
        if set(raw) != set(self._state):
            raise ValueError("return normalizer ticker ownership mismatch")
        restored: dict[str, list[float | int]] = {}
        for ticker, value in raw.items():
            if (
                not isinstance(value, list)
                or len(value) != 3
                or type(value[0]) is not int
                or not all(isinstance(item, int | float) for item in value[1:])
            ):
                raise ValueError("return normalizer state is invalid")
            restored[ticker] = [int(value[0]), float(value[1]), float(value[2])]
        self._state = restored


class EntryActorCritic(nn.Module):
    """Binary entry actor with the three preregistered value ablations."""

    action_names = ("skip", "enter")

    def __init__(
        self,
        observation_width: int,
        variant: PolicyVariant | str = PolicyVariant.SHARED_TICKER_VALUE,
        *,
        hidden_width: int = 64,
    ) -> None:
        super().__init__()
        if observation_width < 1 or hidden_width < 1:
            raise ValueError("policy widths must be positive")
        self.observation_width = observation_width
        self.hidden_width = hidden_width
        self.variant = PolicyVariant(variant)
        self.ticker_embedding = nn.Embedding(len(TICKERS), 8)
        self.profile_embedding = nn.Embedding(len(PROFILES), 2)
        conditioned_width = observation_width + 10

        def trunk() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(conditioned_width, hidden_width),
                nn.Tanh(),
                nn.Linear(hidden_width, hidden_width),
                nn.Tanh(),
            )

        self.critic_trunk = trunk()
        self.cost_critic_trunk = trunk()
        self.shared_actor_trunk = trunk()
        self.independent_actor_trunks = nn.ModuleDict()
        self.independent_profile_embeddings = nn.ModuleDict()
        if self.variant is PolicyVariant.INDEPENDENT_ACTOR:
            self.independent_profile_embeddings = nn.ModuleDict(
                {ticker: nn.Embedding(len(PROFILES), 2) for ticker in TICKERS}
            )

            def independent_trunk() -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(observation_width + 2, hidden_width),
                    nn.Tanh(),
                    nn.Linear(hidden_width, hidden_width),
                    nn.Tanh(),
                )

            self.independent_actor_trunks = nn.ModuleDict(
                {ticker: independent_trunk() for ticker in TICKERS}
            )
        self.actor_head: nn.Linear | None
        if self.variant is PolicyVariant.INDEPENDENT_ACTOR:
            self.actor_heads = nn.ModuleDict(
                {ticker: nn.Linear(hidden_width, 2) for ticker in TICKERS}
            )
            self.actor_head = None
        else:
            self.actor_heads = nn.ModuleDict()
            self.actor_head = nn.Linear(hidden_width, 2)
        self.value_head: nn.Linear | None
        if self.variant is PolicyVariant.SHARED_CRITIC:
            self.value_head = nn.Linear(hidden_width, 1)
            self.value_heads = nn.ModuleDict()
        else:
            self.value_head = None
            self.value_heads = nn.ModuleDict(
                {ticker: nn.Linear(hidden_width, 1) for ticker in TICKERS}
            )
        self.cost_value_heads = nn.ModuleDict(
            {ticker: nn.Linear(hidden_width, 1) for ticker in TICKERS}
        )

    def _conditioned(
        self, observations: torch.Tensor, tickers: torch.Tensor, profiles: torch.Tensor
    ) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.observation_width:
            raise ValueError("policy observation width mismatch")
        return torch.cat(
            (
                observations,
                self.ticker_embedding(tickers),
                self.profile_embedding(profiles),
            ),
            dim=1,
        )

    @staticmethod
    def _owned_output(
        hidden: torch.Tensor,
        owners: torch.Tensor,
        heads: nn.ModuleDict,
        width: int,
    ) -> torch.Tensor:
        output = torch.empty((len(hidden), width), dtype=hidden.dtype, device=hidden.device)
        for index, ticker in enumerate(TICKERS):
            mask = owners == index
            if bool(mask.any()):
                output[mask] = heads[ticker](hidden[mask])
        return output

    def forward(
        self, observations: torch.Tensor, tickers: torch.Tensor, profiles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioned = self._conditioned(observations, tickers, profiles)
        if self.actor_head is None:
            actor_input = torch.empty(
                (len(observations), self.observation_width + 2),
                dtype=observations.dtype,
                device=observations.device,
            )
            for index, ticker in enumerate(TICKERS):
                owned = tickers == index
                if bool(owned.any()):
                    actor_input[owned] = torch.cat(
                        (
                            observations[owned],
                            self.independent_profile_embeddings[ticker](profiles[owned]),
                        ),
                        dim=1,
                    )
            actor_hidden = self._owned_output(
                actor_input, tickers, self.independent_actor_trunks, self.hidden_width
            )
            logits = self._owned_output(actor_hidden, tickers, self.actor_heads, 2)
        else:
            logits = self.actor_head(self.shared_actor_trunk(conditioned))
        critic_hidden = self.critic_trunk(conditioned)
        if self.value_head is None:
            values = self._owned_output(critic_hidden, tickers, self.value_heads, 1).squeeze(1)
        else:
            values = self.value_head(critic_hidden).squeeze(1)
        return logits, values

    def forward_with_cost(
        self, observations: torch.Tensor, tickers: torch.Tensor, profiles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return actor logits plus separate reward and cost values."""
        logits, reward_values = self.forward(observations, tickers, profiles)
        conditioned = self._conditioned(observations, tickers, profiles)
        cost_hidden = self.cost_critic_trunk(conditioned)
        cost_values = self._owned_output(cost_hidden, tickers, self.cost_value_heads, 1).squeeze(1)
        return logits, reward_values, cost_values

    def value_parameters(self, ticker: str) -> tuple[nn.Parameter, ...]:
        if ticker not in TICKERS:
            raise ValueError(f"unknown ticker: {ticker}")
        head = self.value_head if self.value_head is not None else self.value_heads[ticker]
        return tuple(head.parameters())

    def cost_value_parameters(self, ticker: str) -> tuple[nn.Parameter, ...]:
        if ticker not in TICKERS:
            raise ValueError(f"unknown ticker: {ticker}")
        return tuple(self.cost_value_heads[ticker].parameters())

    def actor_parameters(self) -> tuple[nn.Parameter, ...]:
        heads: Iterable[nn.Parameter]
        if self.actor_head is None:
            heads = self.actor_heads.parameters()
            trunks = self.independent_actor_trunks.parameters()
            return (
                *self.independent_profile_embeddings.parameters(),
                *trunks,
                *heads,
            )
        else:
            heads = self.actor_head.parameters()
            trunks = self.shared_actor_trunk.parameters()
        return (
            *self.ticker_embedding.parameters(),
            *self.profile_embedding.parameters(),
            *trunks,
            *heads,
        )

    def owned_actor_parameters(self, ticker: str) -> tuple[nn.Parameter, ...]:
        """Return actor parameters reachable from one ticker's policy."""
        if ticker not in TICKERS:
            raise ValueError(f"unknown ticker: {ticker}")
        if self.actor_head is not None:
            return self.actor_parameters()
        return (
            *self.independent_profile_embeddings[ticker].parameters(),
            *self.independent_actor_trunks[ticker].parameters(),
            *self.actor_heads[ticker].parameters(),
        )
