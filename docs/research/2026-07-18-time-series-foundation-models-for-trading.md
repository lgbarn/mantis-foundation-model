Research report \| Accessed 2026-07-18

# Time-Series Foundation Models for Trading

What Chronos, TimesFM, Moirai, MantisV2, and finance-native models actually pretrain on; how their timeframes translate into market bars; and what a defensible pretraining sequence looks like for liquid futures.

| Measure | Value |
| --- | ---: |
| Verified public PDFs | 12 |
| Core pretraining families | 9 |
| Finance share of LOTSA | 0.10% |
| Complete intraday futures alpha proofs | 0 |

**Bottom line.** The general models are useful design references, not pretrained trading models. Their largest corpora contain little or effectively no finance. The best current evidence says domain-specific financial pretraining can help forecasting, but forecast gains often disappear after realistic costs. No reviewed paper closes the full chain from leakage-safe intraday futures pretraining to executable, net out-of-sample alpha.

## 1. The answer in five points

1.  **Chronos learns a numeric language.** It mean-scales each univariate context, quantizes every value into one of 4,094 numeric bins, and trains T5 with next-token cross-entropy. Its standard sample is 512 context points and 64 targets, with 90% mixed real data and 10% Gaussian-process synthetic data.
2.  **TimesFM and Moirai learn patches.** TimesFM uses 32-point input patches and predicts 128-point output patches. Moirai varies total sequence length from 2 to 512, assigns 15%-50% to the forecast region, and selects patch sizes according to frequency.
3.  **Corpus size is not market coverage.** Moirai's 27.6B-time-step LOTSA corpus is only 0.10% economic/financial data. Time-MoE stores 309B points but has only 413,696 finance points, about 0.0001%.
4.  **Finance-native models are closer to the target.** Kronos pretrains on more than 12B OHLCVA bars across 45 exchanges and includes 75 Chinese futures series at 1-, 5-, 15-minute, and daily frequencies. FinCast and Delphyne add broader multi-frequency finance corpora and explicit domain balancing.
5.  **Trading validation remains the weak link.** The strongest finance study found that finance-only pretraining improved forecasts and gross portfolios, but every tested long-short strategy had negative Sharpe at 20 bps costs.

## 2. Evidence tiers

| Tier | What it establishes | Required evidence |
|----|----|----|
| Tier 1: trading | A model produced an executable economic result. | Causal inputs, untouched OOS period, next-eligible-bar fills, costs, slippage, rolls/borrow, and a fully specified portfolio rule. |
| Tier 2: finance forecast | A model predicts a financial target better under a documented split. | Point-in-time data, chronological evaluation, strong forecast baseline, and contamination audit. PnL is not implied. |
| Tier 3: transferable method | A pretraining technique works on general time series. | Reproducible corpus, objective, sampling, and training recipe. Market transfer remains an untested hypothesis. |

## 3. Core model pretraining map

| Model | Corpus and timeframe | Training sample | Representation and objective | Scale | Trading relevance |
|----|----|----|----|----|----|
| [Chronos](https://arxiv.org/abs/2403.07815) | Public forecasting datasets plus 1M synthetic GP series; mixed frequencies but frequency metadata is ignored. | 512 context + 64 labels; 90% TSMixup, 10% synthetic. | One scaled value per token; 4,096-token vocabulary; T5 cross-entropy. | 20M-710M params; 200K steps; batch 256; 8 A100 GPUs. | Generic prior. Quantization may erase small returns on high price levels. |
| [Chronos-Bolt](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/) | Nearly 100B observations claimed; inventory and mixture undisclosed. | Released base config: 2,048 context, patch 16, direct 64-step output. | Raw-value patches; direct marginal quantiles. | 9M-205M params. | Faster design, but pretraining is not publicly reproducible. |
| [TimesFM 1.0](https://proceedings.mlr.press/v235/das24c.html) | Wikipedia, Google Trends, public benchmarks, and 3M synthetic series; 10-minute to yearly. | Up to 512 points; weekly 256; monthly or slower 64. Input patch 32, output patch 128. | Decoder-only next-patch MSE; 80% real, 20% synthetic; equal real frequency groups. | 17M-200M; 1.5M iterations; batch 4,096; 16 TPUv5e cores. | Excellent sampling reference; no market-specific corpus. |
| [Moirai](https://proceedings.mlr.press/v235/woo24a.html) | LOTSA: 27.646B steps, seconds to yearly; finance 24.92M points or 0.10%. | Total length 2-512; forecast 15%-50%; frequency-dependent patches 8-128. | Masked encoder; instance normalization; any-variate attention; mixture NLL. | 14M-311M; up to 1M steps; batch 256. | Best open cross-frequency sampler; finance share is too small for market semantics. |
| [Lag-Llama](https://arxiv.org/abs/2310.08278) | 27 datasets, 7,965 series, about 352M points/windows; minute through daily. | Best sampled context 32 plus additional lag history; autoregressive horizon. | Lag and calendar covariates; robust median/IQR scaling; Student-t NLL. | 2.45M params; one P100 GPU. | Frequency-aware lags transfer well conceptually; corpus has no meaningful trading data. |
| [Timer](https://arxiv.org/abs/2402.02368) | UTSD up to about 1B original points; yearly through millisecond. Later checkpoints mix larger corpora. | 15 tokens x 96 raw points = 1,440 points. | Univariate pool; decoder-only next-segment MSE. | 1M-67M in paper; 10 epochs; batch 8,192. | Simple generative baseline; channel relationships are discarded. |
| [MOMENT](https://proceedings.mlr.press/v235/goswami24a.html) | Time Series Pile: 1.13B+ observations, about 13M series, 13 domains. | 512 points, 64 patches of 8; 30% patch masking. | Masked-patch reconstruction MSE; channel independent. | 40M-385M; 2 epochs; batch 2,048 sequences. | Strong representation-learning comparator for frozen embeddings. |
| [Time-MoE](https://openreview.net/forum?id=e1wDDFmlVu) | 309.09B stored, about 117B sampled; seconds to yearly; finance 0.0001%. | Packed sequences up to 4,096; direct heads 1/8/32/64. | Point tokens; sparse MoE; multi-resolution Huber loss. | 113M-2.4B total; 128 A100-80GB GPUs. | Useful scaling ceiling, not an appropriate first trading experiment. |
| [MantisV2](https://arxiv.org/abs/2602.17868) | 2M CauKer synthetic sequences; no finance or bar timeframe. | Inputs resized to 512; fixed 32 tokens. | ViT encoder; Random Crop Resize; contrastive InfoNCE; classification embeddings. | Lightweight encoder; 200 pretraining epochs. | Natural repository backbone, but not a forecasting or trading pretraining result. |

## 4. Timeframes: what the paper numbers mean in markets

Foundation-model papers usually state context in observations, not elapsed market time. The same 512 observations encode different regimes and economic horizons:

| Bar frequency | 512 observations represent | 64-step horizon represents | Trading interpretation |
|----|----|----|----|
| 1 minute | 8 hours 32 minutes | 1 hour 4 minutes | About one regular futures session of context; sensitive to microstructure and session boundaries. |
| 3 minute | 25 hours 36 minutes | 3 hours 12 minutes | Several sessions; overnight and regular-session dynamics are mixed unless encoded. |
| 5 minute | 42 hours 40 minutes | 5 hours 20 minutes | Roughly 6.6 US equity regular sessions; fewer full futures sessions. |
| 15 minute | 128 hours | 16 hours | About 20 US equity regular sessions; now closer to a monthly market regime. |
| Hourly | 512 hours | 64 hours | Several weeks of continuous-market behavior. |
| Daily | About 2 trading years | About 3 trading months | Macro/regime context rather than intraday execution structure. |

Elapsed durations are raw bar-time equivalents. Session calendars, holidays, and overnight breaks change calendar duration. This is precisely why frequency, session, and elapsed-time metadata should be explicit.

### Published finance-native timeframe details

| Model | Financial pretraining | Context and horizon examples | Key split fact |
|----|----|----|----|
| [Kronos](https://arxiv.org/abs/2508.02739) | 12B+ OHLCVA bars, 45+ exchanges, seven main granularities. Includes 75 Chinese futures series with 63.3M observations from 2010 at 1m, 5m, 15m, and daily. | Max paper context 512. Evaluation: 5m 480/96, 15m 160/32, 1h 80/12, daily 40/12. | Pretraining ends June 2024; tests begin July 2024. No purge gap disclosed. |
| [FinCast](https://arxiv.org/abs/2508.19609) | 20B points: stocks 9.1B, FX 3.27B, crypto 1.78B, futures 1.71B; seconds to months. | Max 1,024 for seconds-daily; 256 weekly-monthly. Evaluation context 128, horizons 10/30/60. | Test series said to be excluded, but corpus identities, dates, and hashes are not published. |
| [Delphyne](https://arxiv.org/abs/2506.06288) | 15% finance plus 85% LOTSA. Finance includes 15,817 securities at 5-minute OHLC, volume, and trades. | Patch 32; 512 patches = 16,384 raw steps. Intraday task uses 15 days x 78 five-minute bars to predict 78 bars. | Finance pretraining stops 2019-12-31; downstream tests occur later. |
| [Financial TimesFM](https://arxiv.org/abs/2412.09880) | About 80M-90M used points across 100K+ series; hourly stocks/crypto and daily stocks/FX/commodities/indices. | Random context 128-512; fixed output 128. | Pre-2023 train/validation; 2023+ test. No overlap purge disclosed. |

## 5. What finance studies actually found

| Study | Data and protocol | Finding | Evidence tier |
|----|----|----|----|
| [Re(Visiting) TSFMs in Finance](https://arxiv.org/abs/2511.18578) | Daily firm excess returns, 1990-2023, 94 countries; 23 expanding annual models; windows 5/21/252/512. | Off-the-shelf zero-shot and most fine-tuning performed poorly. Finance-native from-scratch pretraining improved forecasts and gross portfolios. All long-short strategies had negative Sharpe at 20 bps costs. | Tier 1 cost stress |
| [Multivariate Financial TSFMs](https://arxiv.org/abs/2507.07296) | US 10-year yield changes, EUR/USD volatility, and equity spreads. | Pretrained TTM needed roughly 3-10 fewer years of task data, but specialized models matched or beat TSFMs in two of three tasks. | Tier 2 |
| [Pretrained TSFMs for Return Forecasting](https://arxiv.org/abs/2606.27100) | Daily returns for AAPL, AMZN, GOOG, JPM, META; equalized context and rolling origin. | TSFMs won most rankings, but gains over random walk were small. Only Chronos/AMZN and Moirai/GOOG passed the reported one-sided Diebold-Mariano tests. | Tier 2 |
| [Chronos-2 Multivariate Finance](https://arxiv.org/abs/2605.21504) | Magnificent-7 equities and US Treasury rates; rolling monthly evaluation, 2000-2025. | Related multivariate inputs improved forecast errors; mixing equities and rates could degrade accuracy. No trading layer. | Tier 2 |
| [Limits of Kronos Integration](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6849321) | About 5,000 Chinese A-shares, daily volatility, 2023-2025, lagged-only protocol. | Zero-shot Kronos ranked last; MAE was 2.4x-2.6x HAR-RV. Fine-tuning also exposed an implementation code-path failure. | Tier 2 preprint |

**The most important negative result:** pretraining can make the model statistically better and economically useless at the same time. Trading evaluation must be a separate downstream stage, never an inference from MSE, MAE, CRPS, or information coefficient.

## 6. A reproducible pretraining sequence for MantisV2

This is a research-derived experiment sequence, not an implementation specification. It preserves the repository's existing NextLeg pipeline as the supervised baseline.

1. **Freeze the evaluation contract first.** Reserve the final chronological holdout and at least one instrument holdout before resampling, normalizing, or windowing. Create all bar frequencies inside each split. Reject any sample whose context or target crosses a boundary.
2. **Build a corpus manifest.** Record unique timestamps, scalar observations, instrument-years, effective sampled windows, bar frequency, session, roll mapping, source license, hashes, and sampling weights. Papers routinely blur archive points, usable points, and actual training exposure.
3. **Start with the current nine futures roots and four intraday frequencies.** Keep ES, NQ, RTY, YM, GC, SI, CL, ZB, and ZN streams independent during label and window construction. Balance by instrument-time and frequency, not raw row count, so 1-minute equity-index data do not dominate.
4. **Compare representations at a small parameter budget.** Hold data and compute fixed while testing return/relative-OHLC point tokens, 8/16/32-point patches, and the native MantisV2 32-token encoder. Use context-only normalization. Add instrument, frequency, session, and elapsed-time conditioning.
5. **Compare objectives separately.** Test masked reconstruction, next-patch Huber, direct multi-horizon Huber/quantile loss, and contrastive representation learning. Random Crop Resize should be an ablation because MantisV2 itself notes that it changes measurement units and is less suitable for forecasting.
6. **Add synthetic data conservatively.** Start with 10%-20%, following Chronos and TimesFM. Generate volatility regimes, jumps, gaps, missing sessions, intraday seasonality, and contract rolls. Keep it only if it improves every untouched real-market fold.
7. **Domain-adapt before scaling.** The finance evidence favors in-domain pretraining over a giant generic mixture. Scale data or parameters only after a small model beats train-from-scratch, generic-checkpoint, and simple statistical baselines under identical splits.
8. **Gate trading independently.** Require next-eligible-bar fills, spread/slippage, commissions, futures point values, rolls, session calendars, and frozen sizing. Keep the 2026 holdout untouched by objective choice, hyperparameters, and walk-forward selection.

### Recommended first experiment matrix

| Axis | Initial choices | Reason |
|----|----|----|
| Context | 512 and 1,024 raw bars | Matches MantisV2 and common TSFM scales without jumping to long-context compute. |
| Patch | 8, 16, 32 | Covers MOMENT, Chronos-Bolt, TimesFM, and Moirai's most common range. |
| Input | Log-relative OHLC, causal scale features, volume/activity with controlled dropout | Cross-contract comparability while retaining liquidity information. |
| Objective | Masked MSE; next-patch Huber; contrastive InfoNCE | Direct comparison of representation, causal forecast, and native Mantis-style learning. |
| Frequency curriculum | 1m, 3m, 5m, 15m balanced by instrument-time | Prevents point-count domination and matches the current repository data contract. |
| Synthetic share | 0%, 10%, 20% | Tests the range supported by Chronos/TimesFM without assuming synthetic transfer. |
| Scale gate | Small encoder first | TTM and Lag-Llama show useful transfer can be tested near 1M-10M parameters. |

## 7. Failure modes to prevent

- **Frequency leakage:** resampling one raw stream into several frequencies, then splitting windows randomly, puts the same underlying price events in train and test.
- **Boundary overlap:** a date cutoff is insufficient if a 512-bar context or its target crosses the cutoff. Purge by full sample span.
- **Normalization leakage:** scaling with a whole window or series can expose future volatility. Statistics must come from the observed context only.
- **Checkpoint contamination:** a public zero-shot model may have seen the evaluation dates. "Zero-shot" describes adaptation, not temporal cleanliness.
- **Corpus-count inflation:** distinguish stored points, scalar variate values, unique timestamps, unique windows, and actually sampled training exposure.
- **Path mismatch:** independent multi-horizon quantiles do not create coherent price paths for stops, excursions, or path-dependent labels.
- **Forecast-to-PnL substitution:** favorable MSE, MAE, rank correlation, or directional accuracy does not establish net profitability.

## 8. What remains unknown

- No public paper supplies a fully reproducible intraday futures corpus with point-in-time contract rolls and a net execution backtest.
- Kronos includes intraday futures but tests trading on daily Chinese equities, with incomplete fill and market-friction details.
- FinCast reports 1.71B futures points but not the contracts, exchanges, dates, cleaning, rolls, or outcomes.
- Chronos-Bolt and several later checkpoints disclose capabilities without a complete peer-reviewed pretraining recipe.
- Most general TSFM papers do not purge overlapping market-style windows or embargo neighboring examples.
- The correct mix of generic synthetic structure and finance-native data remains empirical; Delphyne reports that broad nonfinancial data can cause negative transfer.

## 9. Source bundle and reading order

The verified bundle contains 12 public PDFs. Every file passed a PDF-signature check, plausible-size check, and first-page title/author verification.

1.  **Design axes:** Chronos, TimesFM, Moirai, and MantisV2.
2.  **Finance-domain evidence:** Re(Visiting) Time Series Foundation Models in Finance.
3.  **Small/efficient baselines:** Tiny Time Mixers, MOMENT, Timer, and Lag-Llama.
4.  **Scaling ceiling:** Time-MoE.
5.  **Adverse validation:** the two remaining financial forecasting studies.

**Local bundle:** `/Users/lgbarn/Downloads/research-sources/2026-07-18_time-series-foundation-models-trading/`
**Manifest:** `sources.tsv`

## 10. Primary sources

1.  Ansari et al. (2024), ["Chronos: Learning the Language of Time Series."](https://arxiv.org/abs/2403.07815)
2.  Das et al. (2024), ["A Decoder-only Foundation Model for Time-series Forecasting."](https://proceedings.mlr.press/v235/das24c.html)
3.  Woo et al. (2024), ["Unified Training of Universal Time Series Forecasting Transformers."](https://proceedings.mlr.press/v235/woo24a.html)
4.  Rasul et al. (2024), ["Lag-Llama."](https://arxiv.org/abs/2310.08278)
5.  Liu et al. (2024), ["Timer."](https://arxiv.org/abs/2402.02368)
6.  Goswami et al. (2024), ["MOMENT."](https://proceedings.mlr.press/v235/goswami24a.html)
7.  Ekambaram et al. (2024), ["Tiny Time Mixers."](https://arxiv.org/abs/2401.03955)
8.  Shi et al. (2025), ["Time-MoE."](https://openreview.net/forum?id=e1wDDFmlVu)
9.  Feofanov et al. (2026), ["MantisV2."](https://arxiv.org/abs/2602.17868)
10. Shi et al. (2025/2026), ["Kronos: A Foundation Model for the Language of Financial Markets."](https://arxiv.org/abs/2508.02739)
11. Zhu et al. (2025), ["FinCast."](https://arxiv.org/abs/2508.19609)
12. Ding, Mittal, and Gopal (2025), ["A Pre-Trained Model for General and Financial Time Series" (Delphyne).](https://arxiv.org/abs/2506.06288)
13. Fu, Hirano, and Imajo (2024), ["Financial Fine-tuning a Large Time Series Model."](https://arxiv.org/abs/2412.09880)
14. Rahimikia, Ni, and Wang (2025), ["Re(Visiting) Time Series Foundation Models in Finance,"](https://arxiv.org/abs/2511.18578) also [SSRN 5770562](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5770562).
15. Marconi (2025), ["Time Series Foundation Models for Multivariate Financial Time Series Forecasting."](https://arxiv.org/abs/2507.07296)
16. Noguer i Alonso and Franklin (2026), ["Pretrained Time-Series Foundation Models for Financial Return Forecasting."](https://arxiv.org/abs/2606.27100)

------------------------------------------------------------------------

**Provenance.** Research used public primary papers, official repositories/model cards where needed, arXiv, PMLR, OpenReview, and SSRN metadata. Accessed 2026-07-18. No paywall, CAPTCHA, login wall, or access control was bypassed. Source claims are separated from derived recommendations. Model and checkpoint version drift is noted where public releases no longer match the peer-reviewed recipe.
