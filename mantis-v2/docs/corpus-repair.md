# Contract-aware corpus repair

## Purpose

This is the mandatory entry point for MantisV2 futures data. It prevents the
minute-by-minute contract interleaving that contaminated the original
`FFM_NEXTLEG` CSV corpus.

Databento's official Python SDK decodes the local DBN archives. The decoder
preserves exact contract symbols but does not decide this repository's
continuous-contract policy. Mantis applies the versioned policy in
`configs/corpus-repair-v1.toml` and records every decision in Parquet ledgers and
the corpus manifest.

## Fixed policy in version 1

1. Decode the three local DBN archives offline with the pinned `databento` SDK.
2. Keep canonical outright contracts for ES, NQ, RTY, YM, GC, SI, CL, ZB, and
   ZN.
3. Aggregate volume by CME session. A candidate must beat the current contract
   for two consecutive sessions.
4. Activate the roll at the first bar of the following session.
5. Ratio-back-adjust all earlier OHLC prices and preserve volume unchanged.
6. Classify an extreme isolated print only if prices recover within three bars.
   Preserve the source row, write it to the anomaly ledger, and exclude affected
   model windows through a propagated Parquet `quality_flag` rather than
   inventing prices.
7. Write the selected raw 1-minute series and the adjusted 1-minute market
   series as Parquet.
8. Derive 3-minute, 5-minute, and 15-minute bars only from adjusted 1-minute
   Parquet semantics: first open, maximum high, minimum low, last close, and sum
   volume.
9. Hash every output and publish the directory atomically under a unique corpus
   ID. Existing IDs cannot be overwritten.

Each source archive has an expected SHA-256 in the config. The builder refuses
to decode a changed archive under the existing corpus ID. Decode-cache shards
also carry their own sizes, row counts, SHA-256 values, decoding-policy digest,
SDK/dataframe/Parquet versions, and a digest of the targeted decode and partition
writer implementations; cache reuse fails closed on any mismatch. The cache
directory identity includes that policy digest, so reconstruction-only changes
create a new verified cache alongside older caches instead of overwriting them.

Databento preserves exchange contract symbols. CME one-digit year codes repeat
every decade, so the builder resolves the expiry decade from each contract's
first observation. It initializes a continuous series with the highest-volume
contract in the first configured CME trade session; an isolated stale or
far-dated print cannot become the initial front contract.

## Commands

From the repository root:

```bash
uv sync --all-packages
uv run python -c 'import databento; print(databento.__version__)'
just repair-corpus mantis-v2/configs/corpus-repair-v1.toml
just validate-corpus mantis-v2/configs/corpus-repair-v1.toml
```

The configured publication root is:

```text
/Volumes/Storage/trading-research/data/MANTIS_NEXTLEG_PARQUET_V1/
|-- manifest.json
|-- raw/       # selected source-contract 1-minute bars
|-- market/    # adjusted 1m, 3m, 5m, and 15m bars
|-- rolls/     # roll dates, contracts, ratios, and cumulative factors
`-- repairs/   # classified source anomalies; empty files are still published
```

The source-hash-and-policy-keyed decode cache is stored beside the corpus under
`.mantis-dbn-cache/`. It is reproducible scratch data and is not part of the
published corpus identity. DBN decode is chunked to disk and reconstruction is
bounded to one root symbol at a time, rather than materializing all nine symbols
at once.

## Validation gate

`validate-corpus` must succeed before any training config may reference the
corpus. It verifies the manifest digest and the size, SHA-256 digest, and Parquet
row count of every published file. The repair stage also verifies:

- sorted, unique UTC timestamps;
- finite, positive OHLC and nonnegative volume;
- valid OHLC envelopes;
- exact raw-price times adjustment-factor reconciliation;
- a roll and repair ledger for every symbol;
- a discontinuity report for every adjusted 1-minute stream;
- exact higher-timeframe derivation from the repaired 1-minute source.

Every same-contract price dislocation above the configured threshold must be
either a classified isolated anomaly or an explicit config entry with symbol,
UTC timestamp, and review reason. Unclassified dislocations fail publication.
Stale allowlist entries also fail, so an acceptance cannot silently carry into
a different corpus.

### Reviewed market dislocations

The version 1 config explicitly accepts the SIH6 event at 2026-02-01 23:00 UTC.
The active contract closed the prior session at 85.250, then its first Sunday
minute traded 512 contracts from 84.300 to 80.865 with an 80.680 low. Subsequent
minutes continued trading at high volume, and no contract transition occurred;
the event is real weekend price discovery, not a rollover or isolated print.
External evidence also records the late-January silver crash and emergency CME
margin increase ([S&P Global](https://www.spglobal.com/market-intelligence/en/news-insights/research/2026/02/the-rise-and-fall-of-the-silver-price),
[BIS Quarterly Review](https://www.bis.org/publ/qtrpdf/r_qt2603.pdf)). The source
bar remains unchanged in Parquet. Its exact timestamp and review reason are
recorded in the manifest quality report; stale acceptances fail publication.

The repair function runs that complete audit against its partial directory
before the atomic rename. A failed audit never creates the final immutable
corpus directory.

Production foundation and downstream configs must point at `market/`, select
Parquet input, and pin the exact `manifest.json` digest. A missing manifest,
modified file, wrong digest, or unvalidated manifest is a hard error. Do not
replace a manifest hash in an existing run identity.

Until those configs exist, the root `just` training and downstream recipes
require an explicit config argument. They intentionally have no legacy CSV
default.

## Incident and invalidation rule

The original foundation checkpoint consumed contaminated CL windows. Its
foundation export, embeddings, tuned heads, walk-forward outputs, simulations,
and holdout results are not production artifacts.

Any change to the raw archives or to roll, repair, adjustment, aggregation, date,
symbol, or timeframe policy creates a different dataset. Publish a new corpus
ID, validate it, create new foundation and downstream run IDs, retrain the
foundation, and regenerate every downstream artifact.

## Failure recovery

- Do not edit raw DBN archives.
- Do not patch a published Parquet or manifest in place.
- A failed build leaves completed decode-cache shards and a PID-specific partial
  publication. Rerunning the same config reuses the source-hash cache and starts
  a fresh partial publication.
- If a published corpus fails validation, preserve it for diagnosis and publish
  a corrected policy under a new corpus ID.
- If the external drive is missing or a DBN source hash changes, stop. Restore
  the expected source or intentionally version a new corpus; never silently
  continue.
