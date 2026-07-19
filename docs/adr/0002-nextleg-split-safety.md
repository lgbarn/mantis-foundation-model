# ADR 0002: Preserve complete NextLeg targets within each split

## Status

Accepted.

## Context

NextLeg predicts two consecutive leg durations plus shorter candle horizons. The legacy reference correctly declares a 712-bar reserve (`max_context + 2 * leg_cap`) but an inherited field resets its runtime parent span to 225 (`max_context + max_candle_horizon`). Its assertion therefore does not prove that both legs remain inside a split. The reference also detects pivots after concatenating streams, which can create boundary artifacts.

## Decision

- Detect centered fractal pivots independently in every symbol/interval stream.
- Confirm a pivot at `origin + k`, discard repeated directions, and require two positive following leg durations no greater than `leg_cap`.
- Admit a sample only when the longest context, longest candle horizon, and second following pivot remain inside the same half-open split.
- Keep `target_reserve` and candle batch lookahead as separate configuration concepts.
- Enforce `target_reserve >= max(context_lengths) + 2 * leg_cap` during config loading.
- Normalize future targets with context-only statistics; future values never influence input normalization.

## Consequences

Some boundary samples are discarded. This is intentional: the smaller sample count is preferable to label leakage. Tests exercise two-leg reservation, per-stream construction, and causal normalization.
