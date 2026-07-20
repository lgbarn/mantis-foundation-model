# ADR 0003: Reuse pinned embeddings for isolated head runs

## Status

Accepted.

## Context

The first production walk-forward run used a class-balanced LBFGS logistic head
with `C=1` and `max_iter=500`. All eight folds reached the iteration ceiling.
The run was artifact-valid but failed its model-quality gate: mean weighted log
loss was 0.869 versus the balanced constant baseline of 0.693, and mean
weighted Brier score was 0.299 versus 0.250.

The original downstream identity coupled every stage to one whole-workflow
digest and one output directory. Changing a head-only optimizer setting would
therefore reject or overwrite the already verified 26 GiB embedding stage.
Patching the old manifests would destroy provenance.

## Decision

- Permit walk-forward training to consume an explicit embed manifest only when
  both that manifest and its producer config are pinned by full SHA-256.
- Accept the producer's current workflow digest and the explicitly supported
  legacy workflow digest. Newly generated manifests record the current digest;
  the legacy alternative exists only for artifacts created before the digest
  split. All other identity and semantic checks remain unchanged.
- Require the producer's original workflow identity and its ordered data,
  labeling, preprocessing, and encoder semantics to match the consumer config.
- Rehash every referenced feature and metadata shard before fitting. Preserve
  the embed manifest's original source, lock, foundation, and data identities.
- Require a new `run.name` for a head rerun and record a separate head digest
  from the seed, head configuration, and exact embed-manifest SHA-256.
- Configure solver, inverse regularization strength, tolerance, iteration
  ceiling, class weighting, and convergence policy in TOML. Baseline gating is
  mandatory.
- Capture scikit-learn convergence diagnostics per fold. Production fails
  closed and writes `failure.json` when the selected head does not converge.
- Require mean class-balanced test log loss and Brier score to beat the constant
  0.5 probability baselines before simulation or sealed holdout evaluation.
- Use `C=0.0001`, balanced classes, LBFGS, tolerance `0.0001`, and an iteration
  ceiling of 1,000 for the second production head run. This value was selected
  on fold-0 validation only; it converged in 47 iterations and reduced weighted
  validation log loss from 0.914 to 0.688.

## Consequences

Verified embeddings can support multiple reproducible head experiments without
re-embedding or weakening their identity. Each experiment has separate output
and cannot silently become eligible for simulation. The first walk-forward run
remains an immutable diagnostic artifact. Strong regularization improves the
linear probe, but the full eight-fold rerun must still pass the primary-loss
gate; fold-0 improvement is not treated as production qualification.

The same decision applies to the Trend Magic production embeddings. Its first
fold reached 500 iterations at `C=1.0`; changing only `C` to `0.0001` converged
in 56 iterations on the exact 25,000-row capped fold with no zero-variance
features. The replacement consumer pins the original Trend Magic embed manifest
and producer config, uses a new run identity, and leaves the failed artifact
intact. Its weak fold-0 proper scores remain subject to the unchanged full-fold
quality gate.
