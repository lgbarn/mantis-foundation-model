---
tags: [trading, ml]
---

# Trend Magic downstream strategy specification

## Decision

The MantisV2 downstream pipeline supports Trend Magic as a separate, explicit
strategy profile. It does not mutate or reinterpret historical Supertrend
configs or artifacts.

There is no uniquely authenticated canonical Trend Magic formula. This repo
therefore pins the reproducible public `20/5/1` profile:

- Decision bars: completed 3-minute bars only.
- CCI: period 20, close source, zero included in the bullish state.
- Volatility range: period-5 simple moving average of true range.
- Multiplier: 1.0.
- Bullish raw line: `low - multiplier * range`.
- Bearish raw line: `high + multiplier * range`.
- Bullish recursion: the line may only rise.
- Bearish recursion: the line may only fall.
- On a state change, the active line is seeded from the prior active line before
  applying the directional clamp.
- Warmup: no state until both CCI and the range average are finite.
- Discontinuities: CCI, range, line, and direction reset independently at every
  detected rollover, repaired-data boundary, or other discontinuity.

The strategy emits a candidate on every eligible warmed bar, not only on color
flips. Direction is `+1` when CCI is greater than or equal to zero and `-1`
otherwise. A signal formed at bar close cannot enter before the next eligible
bar open. The existing barrier labels, session close, costs, and Topstep
simulation rules remain controlled by TOML.

## Why this profile

The 2017 open-source TradingView conversion identifies the indicator as an MT4
conversion and documents the ATR multiplier and its CCI/buffer coloring mode.
The 2021 open-source implementation independently describes Trend Magic as ATR
trailing-stop logic whose state/color is controlled by CCI sign. Public
implementations disagree on CCI source, CCI period, and ATR smoothing, so those
choices are part of the experiment identity rather than hidden defaults.

Sources:

- [glaz Trend Magic, TradingView](https://www.tradingview.com/script/Wpy6qdyl-Trend-Magic/)
- [Kivanc Ozbilgic Trend Magic, TradingView](https://www.tradingview.com/script/kRIjThLZ-Trend-Magic/)
- [MetaQuotes TrendMagic code entry](https://www.mql5.com/en/code/284)
- [TradingView CCI definition](https://www.tradingview.com/support/solutions/43000502001-commodity-channel-index-cci/)
- [TradingView ATR definition](https://www.tradingview.com/support/solutions/43000501823-average-true-range-atr/)

## Configuration contract

Trend Magic requires these strategy keys:

```toml
[strategy]
kind = "trend_magic"
atr_period = 20
cci_period = 20
trend_magic_atr_period = 5
trend_magic_multiplier = 1.0
```

`atr_period` is deliberately separate: it controls trade risk distance, while
`trend_magic_atr_period` controls the indicator. Mixing the two changes labels
and candidate direction and therefore invalidates embedding reuse. The full
strategy config participates in the embedding contract digest.

The production profile is
`mantis-v2/configs/trend-magic-topstep-100k.toml`. It binds the repaired Parquet
corpus, the validated MantisV2 export, 1m/3m/15m contexts, a January 1, 2026
sealed holdout, and the existing 24/3/3-month purged walk-forward definition.

Run each recoverable stage separately:

```bash
just downstream-prepare mantis-v2/configs/trend-magic-topstep-100k.toml
just downstream-embed mantis-v2/configs/trend-magic-topstep-100k.toml
just downstream-walk-forward mantis-v2/configs/trend-magic-topstep-100k-head-c0001-v2.toml
just downstream-simulate mantis-v2/configs/trend-magic-topstep-100k-head-c0001-v2.toml
```

The original `C=1.0`, 500-iteration head failed closed on fold 0. An exact-fold
diagnostic changed one variable and found that `C=0.0001` converged in 56 of 500
LBFGS iterations; the 3,840 features contained no zero-variance dimensions.
The production consumer therefore pins `C=0.0001`, a 1,000-iteration safety
ceiling, and the exact completed embed and producer-config hashes under a new
run identity. This repairs optimizer conditioning only. It does not relax the
full-fold proper-score gate or establish strategy quality.

## Validation and interpretation

The formula fixture, append-invariance test, discontinuity-reset test, strict
config tests, provenance mismatch test, and end-to-end smoke must pass before a
production prepare. Preparation must validate the repaired corpus before it
writes candidates. Every later stage must verify the prior manifest and hashes.

The chosen parameters establish reproducibility, not profitability. No rigorous
public evidence validates Trend Magic on 3-minute futures. Primary model metrics
remain class-balanced log loss and Brier score; ROC AUC and PR AUC are
diagnostics. The sealed 2026 holdout stays unavailable until the strategy,
classifier, threshold, and acceptance criteria are frozen.
