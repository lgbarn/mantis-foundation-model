---
tags: [trading, ml]
---

# MantisV2 Supertrend and Topstep 100K Pipeline Specification

## Decision

Build a downstream, config-driven pipeline that consumes a validated MantisV2
export, creates a candidate at every eligible closed 3-minute bar, and uses the
current Supertrend state for direction. It must not require a Supertrend flip.
The first production context set is 1-minute, 3-minute, and 15-minute data,
aligned causally to the 3-minute decision close.

The classifier and the account simulator are separate stages. The classifier
is selected with purged chronological walk-forward folds. The simulator then
replays the resulting out-of-sample predictions under a configurable Topstep
100K Combine rule set. This separation prevents changes to account rules from
forcing model retraining.

## Authoritative evidence

- MantisV2 paper: <https://arxiv.org/abs/2602.17868>
- MantisV2 source tag `v1.0.0`, commit
  `0c94f8ceb9f1d1421dd292ed917090df8c31605b`
- Pinned MantisV2 weights revision
  `99fe0f548960e272fbfa4b82fd9b5b5956779dfd`
- Topstep Trading Combine parameters:
  <https://help.topstep.com/en/articles/8284197-trading-combine-parameters>
- Topstep Maximum Loss Limit:
  <https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit>
- Topstep Daily Loss Limit:
  <https://help.topstep.com/en/articles/10490293-daily-loss-limit-in-the-trading-combine-and-express-funded-account>
- Topstep consistency rule:
  <https://help.topstep.com/en/articles/8284208-consistency-at-topstep>
- Local behavior references:
  `Futures-Foundation-Model/scripts/supertrend_mantis.py` and
  `futures-combine-trainer/combine_trainer`.

## Data and candidates

The production pool is `ES`, `NQ`, `RTY`, `YM`, `GC`, `CL`, and `ZB`. Inputs
are the existing continuous CSV streams on the external drive; normalized
pipeline tables are Parquet. Every input file is content-hashed.

For each root:

1. Compute Wilder ATR(20) and Supertrend(10, 3.0) on closed 3-minute bars.
2. Emit a candidate on every eligible 3-minute close after indicators and all
   contexts are warm. Direction is the current Supertrend state.
3. Align each context with the rightmost completed timeframe bar whose close is
   less than or equal to the decision timestamp. A forming 15-minute bar is
   never visible to a 3-minute decision.
4. Enter at the next 3-minute open. The stop is 0.5 ATR and the primary target
   is 3R. Label the first stop/target touch through a 120-bar horizon. If both
   touch in one bar, count the stop first. Force-close at the final completed
   bar before the 3:10 PM CT session boundary so entry eligibility never
   depends on a future realized outcome. Store `label_end_ts` for purging.
5. Apply 0.03R round-trip cost. Also retain 2R, 4R, and 6R label columns for
   later analysis without redefining the primary training objective.

The candidate table remains every-bar. A 20-bar cooldown, flat-only execution,
and no pyramiding are applied only after a prediction passes its validation-
owned threshold.

## MantisV2 representations

The production representation uses the validated fine-tuned NextLeg encoder's
final-layer CLS embedding. Its input preprocessing must match NextLeg training:
per-context channel standardization, clamp to `[-10, 10]`, and linear resize to
512 with `align_corners=false`.

The researched upstream benchmark is transformer layer index 2 (the third
layer) with the combined CLS-plus-mean token and official input normalization.
It is deliberately deferred from this production pipeline because it requires
the original upstream weights and a separate preprocessing contract; applying
those settings to the fine-tuned export would not be a valid benchmark. The
production config therefore allows only the validated NextLeg preprocessing.

Embedding extraction is batch bounded and written to immutable `.npy` shards
with Parquet metadata. A stage manifest records config, source, model, input,
and output hashes. The trusted foundation safetensors SHA-256 is an explicit
config value even for legacy manifests. No pickle format is used for released
heads.

## Classifier and walk-forward protocol

The primary head is `StandardScaler` followed by logistic regression with
`max_iter=500`. Both are fit on each fold's training rows only. An MLP is not a
production option until it beats the logistic baseline on untouched
walk-forward test folds.

Default folds use 24 train months, 3 validation months, 3 test months, and a
3-month stride. A 12/2/2 sensitivity profile is also configurable. Purge any
training or validation event whose `[decision_ts, label_end_ts]` overlaps the
next partition; add a configurable context embargo at both boundaries.

The validation partition selects a probability threshold by percentile,
defaulting to the top 50 percent. Persist the resulting numeric threshold and
apply that exact number to test. Never recompute a percentile on test or the
sealed final holdout.

Primary model metrics are weighted log loss and Brier score. ROC AUC and PR AUC
are diagnostics, along with trade count, expectancy in R, and calibration.
Final holdout access is locked by default and requires both a config opt-in and
an explicit holdout command.

## Topstep 100K Combine simulator

The default account profile is:

- Starting balance: $100,000
- Profit target: $6,000
- Maximum Loss Limit distance: $3,000
- Initial loss floor: $97,000
- MLL ratchets from highest end-of-day equity and stops at $100,000
- Intraday realized plus unrealized equity touching the MLL fails the account
- Best-day consistency: best day divided by total profit must remain below 50%
- Maximum size: 10 minis or 100 micros, with 10 micros equal to one mini
- Minimum passing duration: 2 trading days
- Trading day: 5:00 PM CT through 3:10 PM CT the next day

The base Combine has no mandatory Daily Loss Limit. A separate config variant
may enable the $2,000 Daily Loss Limit associated with the optional checkout
path; it is a soft daily halt, not an account failure. Express Funded and Live
Funded scaling or payout rules are out of scope.

## Stage contract

The command surface is stage-oriented and every parameter or location comes
from TOML or a recorded `--set section.key=value` override:

1. `downstream-prepare`: validate and hash inputs, generate pre-holdout
   candidates and labels, and write Parquet. It never materializes holdout
   labels.
2. `downstream-embed`: verify the foundation export, extract bounded embedding
   shards, and write an embedding manifest.
3. `downstream-walk-forward`: build purged folds, fit and validate the head,
   then write out-of-sample predictions, metrics, and portable head arrays.
4. `downstream-simulate`: replay predictions through execution and the 100K
   account state machine.
5. `downstream-run`: run the four normal stages in order.
6. `downstream-holdout`: explicitly unlock, generate holdout candidates and
   embeddings, and evaluate the sealed holdout once.

Every stage validates its inputs before writing and writes atomically. Export
is allowed only after finite metrics, prediction coverage, probability bounds,
fold ownership, model identity, and simulator invariants pass.

## Acceptance criteria

- A forming higher-timeframe bar cannot enter a lower-timeframe candidate.
- Candidates occur on every eligible 3-minute state bar, not only flips.
- Entry is next-open and same-bar stop/target ambiguity loses conservatively.
- Event-span purge and embargo prevent label leakage across all boundaries.
- Scaler, classifier, calibration, and numeric threshold are training/validation
  owned and never fit or selected on test.
- Topstep MLL, end-of-day ratchet, consistency, optional DLL, size limits, and
  two-day minimum have deterministic boundary tests.
- The sealed holdout command fails closed without both unlock controls.
- A smoke run produces validated manifests without CUDA.
- Full local gates and the autofix review pass before commit and push.
