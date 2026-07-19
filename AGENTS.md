# AGENTS.md

## Mission

This repository builds and trains foundation models in the Mantis family. It will support Mantis, Mantis+, and MantisV2, but MantisV2 is the only active implementation until its end-to-end workflow is reproducible.

## Repository shape

Each major model family owns its implementation, configuration, tests, and documentation:

```text
mantis-foundation-model/
|-- mantis-v2/          # Active first implementation
|-- mantis-plus/        # Reserved; do not add speculative code
|-- mantis/             # Reserved; do not add speculative code
|-- shared/             # Only genuinely version-neutral code
|-- docs/               # Repository-level architecture and operating guides
|-- justfile             # Root orchestration across model families
|-- pyproject.toml       # Root development tooling
`-- uv.lock              # Reproducible Python environment
```

Within `mantis-v2/`, prefer a staged, config-driven package:

```text
mantis-v2/
|-- configs/            # Versioned experiment and training configuration
|-- src/mantis_v2/      # Data, model, training, evaluation, and export stages
|-- tests/              # Unit and end-to-end contract tests
|-- docs/               # MantisV2-specific decisions and guides
`-- README.md            # Supported workflows and exact commands
```

Do not couple model-family directories through mutable implementation details. Shared code must be stable, version-neutral, tested at the root, and justified by at least two real consumers. Otherwise keep it inside the owning model version.

## Current source of truth

- Treat `/Users/lgbarn/Personal/Trading/futures-combine-trainer` as a structural reference for staged packages, strict configuration, provenance, tests, and `uv`/`just` workflows.
- Treat `/Users/lgbarn/Personal/Trading/Futures-Foundation-Model/scripts/supertrend_mantis.py` as behavioral evidence for the current Mantis training harness.
- Treat `/Users/lgbarn/Personal/Trading/Futures-Foundation-Model/.scratch` as historical inspiration only. Its cached script identifies version 2.0 while the maintained script identifies version 3.0.
- Pin MantisV2 semantics to paper arXiv 2602.17868, code to `vfeofanov/mantis` tag `v1.0.0` at commit `0c94f8ceb9f1d1421dd292ed917090df8c31605b`, and weights to `paris-noah/MantisV2` revision `99fe0f548960e272fbfa4b82fd9b5b5956779dfd`. Do not follow floating branches or Hub revisions.
- Upstream license declarations conflict between Apache-2.0 and MIT. Do not redistribute upstream source or weights or claim a unified license until that conflict is clarified.

## MantisV2 invariants

- Require a real, explicitly configured foundation checkpoint. Fail closed; never silently substitute a vanilla encoder or randomly initialized model.
- Keep raw input and label construction causal. Outcomes may use future bars only as labels, never as model inputs.
- Define execution timing explicitly. For trading experiments, signals formed at a bar close must not receive an earlier fill than the next eligible bar.
- Purge overlapping lookback and label horizons at train, validation, test, and walk-forward boundaries.
- For NextLeg, preserve `max_context + 2 * leg_cap` split reservation. Keep this separate from the shorter candle-horizon batch span; the legacy reference accidentally conflates them.
- Detect pivots independently within each instrument/interval stream. Never concatenate streams before label construction.
- Keep a reserved out-of-sample holdout untouched by tuning, model selection, and walk-forward iteration.
- Make long training and walk-forward runs resumable from provenance-checked checkpoints.
- Preserve separate smoke, walk-forward, and production workflows. Smoke mode may reduce scale, not weaken correctness checks.
- Verify exported models against the native model on fixed fixtures and numerical tolerances before declaring export parity.
- Record enough provenance to reproduce every artifact: source revision, config hash, dataset identity, checkpoint identity, dependency lock, seeds, inputs, and outputs.

## Data and artifacts

Source code, configs, small deterministic fixtures, and documentation belong in git. Raw datasets, transformed datasets, checkpoints, trained weights, reports, caches, secrets, and local environment files do not.

Generated paths must be configurable and ignored before the first run. Never overwrite a checkpoint or result directory without an explicit run identity. A downstream stage must reject stale, modified, or mismatched upstream artifacts.

Do not add or enable a paid, metered, token-requiring, or account-requiring dependency without explicit user approval.

## Configuration

- Use typed, versioned configuration rather than scattered environment variables or magic constants.
- Reject unknown keys, missing required values, invalid ranges, and incompatible combinations.
- Keep secrets and machine-local paths outside committed configs.
- Environment variables may select runtime resources or override documented operational settings, but must not hide the experiment definition.

## Development workflow

Use `uv` for Python environments and dependency locking, and `just` for discoverable repository commands. Do not install packages without explicit user approval.

The root `justfile` should eventually expose at least setup, format, lint, test, smoke, train, evaluate, and export commands, with model-version-specific recipes where behavior differs. Until those files exist, do not claim any command or test passes.

For implementation work:

1. Read the owning model version's README, configs, and tests before editing.
2. Make the smallest complete change within one model-family boundary.
3. Add or update tests for configuration, data leakage, checkpoint provenance, resume behavior, and export parity as applicable.
4. Run the narrowest relevant checks, then the repository gate before completion.
5. Report the exact commands run and distinguish verified results from inference.

## Scope discipline

- Build MantisV2 first. Do not populate `mantis/` or `mantis-plus/` by copying MantisV2.
- Do not refactor across versions merely to make directories look uniform.
- Do not weaken leakage, holdout, provenance, or parity checks to make a run pass.
- Do not commit generated models or data.
- Record durable architecture decisions in `docs/adr/` and keep model-specific operational details beside that model version.

## Planning state

The active decision map is `.scratch/mantis-family-repo/map.md`. It is not an implementation backlog. The upstream contract is resolved; production hardware, tracking, retention, and final target decisions remain configurable pending user confirmation.
