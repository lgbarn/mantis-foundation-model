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
5. Price the calendar spread from the latest timestamp shared by both contracts
   at or before activation, with a three-day maximum age. Record that pricing
   timestamp, ratio-back-adjust all earlier OHLC prices, and preserve volume.
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

Every price dislocation above the configured threshold must be either a
classified isolated anomaly or an explicit config entry with symbol, UTC
timestamp, event kind, exact contract identity, and review reason. Roll events
bind both the old and new contracts; same-contract events bind the active
contract. Unclassified, wrong-kind, wrong-contract, and stale entries fail
publication, so an acceptance cannot silently carry into a different event.

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

The version 1 config also accepts the CLK6-to-CLM6 boundary at 2026-04-19
22:00 UTC. The same-time roll ratio correctly removes the calendar spread:
CLK6 closed 90.91 while CLM6 closed 89.46. Both contracts independently rose
from their Friday closes at the Sunday reopen, so using prior-session prices
would incorrectly erase a real market move. CME reported WTI approaching $86
as geopolitical tensions escalated, and S&P Global recorded the April 20 rise
after renewed Strait of Hormuz disruption
([CME Group](https://www.cmegroup.com/videos/2026/04/20/geopolitical-tensions-push-crude-higher-4-20-26.html),
[S&P Global](https://www.spglobal.com/energy/en/news-research/latest-news/crude-oil/042026-factbox-oil-prices-rise-on-us-iran-hormuz-standoff-ahead-of-ceasefire-expiry)).
The manifest records this separately as an accepted roll dislocation.

CL has ten additional reviewed same-contract events. Each event was reproduced
in the next listed contract with a similar move, began with 2,970-10,134
contracts in the selected bar, and remained materially displaced afterward.
The builder preserves every source value and records each acceptance in the
manifest.

| UTC timestamp | Contract | Maximum excursion | Review |
| --- | --- | ---: | --- |
| 2022-02-27 23:00 | CLJ2 | 6.1236% | Ukraine invasion supply shock |
| 2022-03-06 23:00 | CLJ2 | 9.6522% | Russia sanctions and supply fears |
| 2023-04-02 22:00 | CLK3 | 7.9128% | Surprise OPEC+ production cuts |
| 2024-10-27 22:00 | CLZ4 | 5.2309% | Israel spared Iranian energy facilities |
| 2025-06-15 22:00 | CLN5 | 5.8896% | Israel-Iran escalation |
| 2025-06-22 22:00 | CLQ5 | 5.8887% | US strikes on Iranian nuclear sites |
| 2026-03-01 23:02 | CLJ6 | 11.9483% | US-Israeli strikes and shipping disruption |
| 2026-03-08 22:00 | CLJ6 | 10.9346% | Iran war and Strait of Hormuz disruption |
| 2026-03-23 11:08 | CLK6 | 6.0275% | Middle East negotiation-headline reversal |
| 2026-04-12 22:00 | CLK6 | 10.0596% | Failed talks and blockade risk |

Authoritative context includes the
[EIA's Ukraine analysis](https://www.eia.gov/todayinenergy/detail.php?id=51498),
the [OPEC April 2023 announcement](https://www.opec.org/pr-detail/63-03-apr-2023.html),
and [CME's March 23, 2026 WTI report](https://www.cmegroup.com/videos/2026/03/23/wti-crude-oil-futures-drop-10-as-middle-east-talks-continue-3-.html).
The complete post-fix rates audit found no 5% events: ZB contained 1,543,998
rows with a 1.5848% maximum excursion, and ZN contained 1,667,561 rows with a
1.0168% maximum; both span the configured five years.

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
