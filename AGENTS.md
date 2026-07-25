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

## Minimal implementation and command-first workflows

Use the least code and the least abstraction that complete the task safely. Code
complexity must be earned by a current requirement, not by possible future reuse.
Prefer direct, readable control flow over frameworks, wrapper layers, generic
orchestration, and configuration that merely moves simple logic out of sight.
Every additional module, abstraction, dependency, and command wrapper must make
the workflow materially safer, clearer, more reproducible, or easier to test.

Before writing custom code, look for an official, maintained CLI or documented
command supplied by the owning project. Prefer that interface when it supports
the required behavior, can be version-pinned or provenance-recorded, works in
the qualified runtime, and does not introduce an unapproved cost or dependency.
Verify commands and flags against official documentation or the installed
tool's help output; never invent an interface. Use `just` as a discoverable
entry point when a repository command needs stable arguments or combines real
workflow stages, but do not hide a single clear command behind unnecessary
shell or Python layers.

Treat model training as an explicit sequence of tasks. The workflow should be
understandable from the commands it runs and the artifacts passed between them.
Documentation for each supported workflow must state:

1. The exact commands, in execution order.
2. Why that order is required and which invariant each stage protects.
3. The inputs, outputs, run identity, and provenance expected at each boundary.
4. The safe resume, retry, and failure behavior.
5. Any justified exception to the normal order, including when it applies and
   what additional validation it requires.

Assume existing code may be more complex than necessary and actively look for
safe simplifications while working in the relevant area. Refactor complexity
when it obscures the command sequence, duplicates an official tool, creates
unused flexibility, or makes behavior difficult to explain or test. Keep such
refactoring scoped to the task, establish behavior with tests before changing
it, and verify the same behavior afterward. Do not perform unrelated rewrites,
and do not refactor files used by an active or resumable run.

Minimal does not mean incomplete. Never remove validation, causal boundaries,
holdout isolation, checkpoint provenance, resume safety, determinism controls,
or export parity merely to reduce line count. The target is the smallest
complete implementation whose behavior, commands, and failure modes can be
explained plainly.

## Development workflow

Use `uv` for Python environments and dependency locking, and `just` for discoverable repository commands. Do not install packages without explicit user approval.

### Pause and stop notifications

Never leave the user guessing whether work is active. Immediately send a concise
status notification whenever the current process stops, work pauses, execution
waits on a test, subagent, provider, approval, or external condition, or no task
process is running. State what stopped or is waiting, why, the exact condition
that resumes work, and whether any billed resource remains active. Send another
notification as soon as execution resumes. Do not silently wait between stages.

RunPod's supported PyTorch image includes `uv` but not `just`. Remote Pod commands
must therefore invoke `uv run mantis-v2 ...` directly after `uv sync --frozen`.
Rsyncing to a RunPod network volume must use `--no-owner --no-group`; the mounted
volume rejects ownership changes. Before any source transfer or corpus work, smoke
the image's CUDA runtime. After the frozen sync, smoke the resolved `uv` runtime
again before inspecting data or starting training.
The resolved remote virtual environment must live on the Pod's ephemeral `/tmp`,
not the shared network volume, so failed installs cannot contaminate later runs.
Use the network volume only for immutable staged data, manifests, checkpoints,
exports, and TensorBoard logs. Copying data requires neither a CUDA image nor a
GPU: stage it from the local machine through RunPod's S3-compatible API, or use
a CPU-only Pod only when remote transformation is actually required. Reserve a
CUDA Pod for the CUDA smoke, training, and export stages.
For remote source provenance, transfer a Git bundle created from stable branch
refs rather than copying the live `.git` directory, whose transient agent refs
can change while a Pod is starting.

The root `justfile` should eventually expose at least setup, format, lint, test, smoke, train, evaluate, and export commands, with model-version-specific recipes where behavior differs. Until those files exist, do not claim any command or test passes.

For implementation work:

1. Read the owning model version's README, configs, and tests before editing.
2. Make the smallest complete change within one model-family boundary.
3. Add or update tests for configuration, data leakage, checkpoint provenance, resume behavior, and export parity as applicable.
4. Run the narrowest relevant checks, then the repository gate before completion.
5. Report the exact commands run and distinguish verified results from inference.

For operating an existing workflow rather than changing implementation, read
`docs/agent-runbook.md` and `docs/workflow.md`. During an active or resumable run,
do not edit its source, dependency lock, config, data, or artifact files; do not
start a second command with the same run identity; and do not kill, signal,
suspend, or restart its processes. Documentation-only work must remain outside
active data and artifact paths.

## Agent skills

### Issue tracker

Track implementation work in GitHub Issues for `lgbarn/mantis-foundation-model`.
External pull requests are not a triage request surface. See
`docs/agents/issue-tracker.md`.

### Triage labels

Use the canonical `needs-triage`, `needs-info`, `ready-for-agent`,
`ready-for-human`, and `wontfix` states plus the orthogonal `tdd` modifier. See
`docs/agents/triage-labels.md`.

### Domain docs

Use a single repository context with root ADRs in `docs/adr/`. See
`docs/agents/domain.md`.

## Scope discipline

- Build MantisV2 first. Do not populate `mantis/` or `mantis-plus/` by copying MantisV2.
- Do not refactor across versions merely to make directories look uniform.
- Do not weaken leakage, holdout, provenance, or parity checks to make a run pass.
- Do not commit generated models or data.
- Record durable architecture decisions in `docs/adr/` and keep model-specific operational details beside that model version.

## Planning state

The map at `.scratch/mantis-family-repo/map.md` is historical decision context,
not an implementation backlog or operating guide. Current commands and behavior
are defined by `justfile`, committed TOML configs, package tests,
`docs/workflow.md`, and `mantis-v2/README.md`. The qualified foundation-model
production path is Apple MPS with local JSON/provenance artifacts; paths and
future run identities remain machine-specific configuration. The accepted
Topstep RL specification qualifies its policy-training path on CPU separately;
do not treat MPS as an RL target without its own determinism and throughput
qualification. CPU and CUDA foundation-model production recipes are not
qualified end to end.
