# ADR 0004: Rebuild a contract-aware futures corpus

## Status

Accepted.

## Context

The first MantisV2 NextLeg run trained directly on the `FFM_NEXTLEG` continuous
CSV corpus. A failed downstream CL trade exposed alternating contract rows,
including an `89.55 -> 80.93 -> 89.54` sequence on 2023-10-03. The CSV generator
selected the highest-volume contract independently for each minute and then
discarded the contract symbol. Deterministic sampler replay proved that the
exported foundation checkpoint consumed affected CL windows at every configured
timeframe.

The original local Databento DBN archives remain available and preserve the
exact contract symbol on every bar. Databento's official Python SDK can decode
these archives offline. The SDK is the authoritative file decoder; the roll
selection and adjustment policy remains an explicit Mantis pipeline decision.

## Decision

- Decode local DBN archives with the pinned official `databento` SDK.
- Pin the SHA-256 of every DBN archive and content-verify every decoded cache
  shard before reuse.
- Retain canonical outright futures and preserve the exact source contract on
  every output bar.
- Select a new front contract only after it exceeds the current contract's
  aggregate volume for two consecutive CME sessions.
- Activate the confirmed roll on the next session. Do not switch contracts
  independently from minute to minute.
- Ratio-back-adjust earlier OHLC prices at each roll and record the roll date,
  old and new contracts, ratio, and cumulative factors.
- Classify isolated bad prints that make an extreme move and revert within
  three minutes. Preserve the source bar, record it in the anomaly ledger, and
  propagate a quality flag so causal loaders exclude affected windows. Do not
  fabricate replacement prices. Persistent moves are not classified as bad
  prints.
- Publish adjusted 1-minute Parquet first. Derive 3-minute, 5-minute, and
  15-minute Parquet only from that repaired 1-minute series.
- Publish raw selected 1-minute bars, adjusted market bars, roll ledgers, repair
  ledgers, anomaly ledgers, and a content-addressed manifest under a new
  immutable corpus ID.
- Never overwrite an existing corpus ID or patch a published Parquet file in
  place.
- Validate timestamps, OHLC envelopes, finite values, unique rows, adjustment
  reconciliation, full date coverage, contract transitions, persisted resample
  equality, output hashes, row counts, and remaining discontinuities before the
  corpus is atomically published or eligible for training.
- Keep the discontinuity detector in foundation and downstream loaders as a
  fail-safe. It cannot replace contract-aware corpus construction.
- Fail publication on every unclassified same-contract dislocation. A reviewed
  real market discontinuity requires an exact symbol, UTC timestamp, and reason
  in versioned corpus config; stale acceptances also fail.
- Bind the validated corpus manifest digest into production configuration and
  training provenance. A manifest or file mismatch must fail closed before
  training.

## Consequences

The contaminated foundation checkpoint and all embeddings, heads, and
simulations derived from it are diagnostic only. Production qualification must
restart from a validated repaired corpus and regenerate the foundation export,
embeddings, walk-forward heads, simulations, and holdout evaluation.

The old continuous CSV files remain immutable incident evidence, not production
training inputs. CL remains in the nine-symbol corpus because exact-contract DBN
data makes deterministic repair possible. Any future change to source archives,
roll policy, repair policy, date bounds, symbols, or timeframe aggregation
requires a new corpus ID and full downstream invalidation.
