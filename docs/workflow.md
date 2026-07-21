# End-to-end workflow

This runbook takes a human operator from a cloned repository to a validated
MantisV2 export and downstream trading simulation. It describes the current
implementation, not a proposed workflow.

## Read this before running commands

- Only MantisV2 is implemented.
- The foundation stage is supervised futures-specific fine-tuning, not MantisV2
  pretraining.
- The committed production configs are machine-specific records.
- Smoke runs prove orchestration, not production quality.
- Validation selects models and thresholds. The holdout is not routine test data.
- A passing export is not evidence of profitable trading.
- Installing dependencies, downloading weights, scanning 40M rows, training, and
  holdout access have different cost and authorization boundaries.

## Protect an active run

When any foundation or downstream stage is active or resumable:

1. Do not kill, signal, suspend, renice, or restart its processes.
2. Do not edit Python source, `uv.lock`, the active TOML, source data files, or any
   file beneath the active artifact directory.
3. Do not start another command with the same `run.name` and artifact root.
4. Do not run `just gate` casually on a resource-constrained host. It launches
   tests plus both smoke workflows.
5. Use existing JSON metrics and manifests for status instead of loading mutable
   checkpoints.
6. Limit concurrent documentation work to Markdown and read-only checks.
7. Give every new experiment a new config filename, run name, and artifact path.

Source, config, lock, data, and upstream identities are provenance-bound. Editing
them can make a checkpoint intentionally non-resumable.

## Understand command effects

| Command | Main work | Writes | Typical cost or risk |
| --- | --- | --- | --- |
| `just --list` | Lists recipes | Nothing | Negligible |
| `just sync` | Resolves and installs locked packages | `.venv` and caches | Network and environment change |
| `just gate` | Format check, lint, types, tests, 2 smokes | Disposable smoke artifacts | CPU/device load; avoid during production |
| `just verify-upstream` | Fetches/verifies official weights on CPU | Hub cache | Network on first use |
| `just inspect-data <config>` | Reads every configured stream and builds anchors | Nothing | Full corpus scan |
| `just probe-mps` | 32 real optimizer updates and one validation batch | Disposable probe artifacts | MPS allocation and weight fetch |
| `just train <config>` | Trains or resumes | Checkpoints, metrics, provenance | Long accelerator run; mutates run state |
| `just evaluate <config>` | Evaluates `best.pt` | `evaluation.json` | Reads data/checkpoint; writes authorization |
| `just export <config>` | Exports after prior evaluation | Export directory | Requires exact current evaluation |
| `just validated-export <config>` | Evaluates and exports one snapshot | Evaluation and export | Preferred release path |
| `just downstream-prepare <config>` | Builds candidates and labels | Parquet plus manifest | Full data preparation |
| `just downstream-verify <config>` | Validates and prints the downstream contract | Nothing | Read-only config check |
| `just downstream-embed <config>` | Runs frozen encoder | NPY shards, Parquet, manifest | Long MPS/CPU inference |
| `just downstream-walk-forward <config>` | Fits scaler and logistic folds | NPZ, predictions, metrics, manifest | CPU and memory |
| `just downstream-simulate <config>` | Replays trades and account rules | Results and manifest | CPU |
| `just downstream-run <config>` | Runs all downstream stages | All downstream outputs | Long, multi-stage mutation |
| `just downstream-holdout ...` | Opens sealed 2026 downstream data | Holdout artifacts | One-time governance action |
| `just rl-dry-run <config>` | Validates the locked RL identity | Atomic dry-run manifest | CPU; no holdout access |
| `just rl-account-replay <input> <output> <config>` | Replays marked-equity account fixtures | Atomic replay manifest | CPU; refuses overwrite |

## Phase 1: inspect the repository

From the repository root:

```bash
just --list
git status --short
```

Read:

1. `README.md`
2. `AGENTS.md` if you are an AI agent
3. The selected TOML config
4. `mantis-v2/README.md`
5. Both accepted architecture decisions

Do not treat `.scratch` planning files as current commands.

## Phase 2: prepare a portable config

Do not use the committed production config as a generic template. Copy it:

```bash
cp mantis-v2/configs/nextleg-parquet-v2.toml mantis-v2/configs/nextleg-myhost-mps.toml
```

In the copy, set:

- A new `run.name`
- Your `run.artifact_root`
- Your `data.root`
- Your intended `run.device`
- An appropriate `require_accelerator`
- `allow_overwrite=false`

Keep immutable source, Hub revision, and weight digest pins unless you are
performing a separately reviewed upstream upgrade.

For downstream work, copy the production downstream config and point
`foundation.manifest_path` to your validated export. Pin the exact exported
weights SHA-256.

## Phase 3: synchronize dependencies

With installation authority:

```bash
just sync
```

This materializes all workspace packages from `uv.lock`. Do not run it during a
protected training run unless changing the environment is explicitly intended.

## Phase 4: run synthetic correctness checks

Run the complete gate when resources permit:

```bash
just gate
```

The gate performs:

1. Formatting check
2. Ruff lint
3. Strict mypy
4. Pytest
5. Foundation synthetic smoke
6. Downstream synthetic smoke

The foundation smoke trains a scratch architecture for 1 epoch with 2 training
and 2 validation batches, then evaluates and verifies export parity. It does not
download official weights or use market data.

The downstream smoke runs the 4 stages on deterministic synthetic data using
CPU. It does not qualify production embeddings or trading results.

## Phase 5: verify official MantisV2

```bash
just verify-upstream
```

This command:

1. Downloads or reuses the exact pinned Hub revision.
2. Verifies the model weight digest.
3. Constructs the upstream adapter on CPU.
4. Runs a fixed `[1,5,512]` fixture.
5. Requires a `[1,1280]` concatenated embedding.

It does not train or inspect market data.

## Phase 6: inspect real data

```bash
just inspect-data mantis-v2/configs/nextleg-myhost-mps.toml
```

Record:

- Every stream name
- First and last timestamps
- Row counts
- Train, validation, and holdout anchor counts
- Any rejected stream or invariant

This is read-only but scans the complete configured corpus. It can take time and
consume I/O bandwidth.

## Phase 7: qualify the device

### Apple MPS

The committed probe is guarded to exactly 32 real optimizer updates and one
real validation batch across all configured production streams:

```bash
just probe-mps
```

For another host, copy the probe config, update paths and its disposable run
identity, and invoke the copied config directly:

```bash
uv run mantis-v2 probe --config mantis-v2/configs/nextleg-myhost-mps-probe.toml
```

Run the matching bounded probe before a new long MPS recipe. It writes disposable
artifacts and permits overwrite only for its probe identity.

### CPU

No generic production CPU config is committed. CPU smoke and upstream
verification establish correctness only. Create a separate config with
`require_accelerator=false` and a new run identity for bounded CPU experiments.

### CUDA

The foundation code path exists, but there is no committed CUDA probe or
production recipe. Downstream embedding does not support CUDA. A CUDA
qualification effort is a new implementation and validation scope, not a config
toggle that makes the whole workflow supported.

## Phase 8: train or resume NextLeg

```bash
just train mantis-v2/configs/nextleg-myhost-mps.toml
```

The same command resumes when `training.resume=true` and valid `latest.pt` or
`pending.pt` state exists.

Production training uses:

- Official pinned weights
- Full encoder fine-tuning
- 120 configured epochs
- 200 training batches per epoch
- Batch size 128
- 20 validation batches per epoch
- 10-epoch linear warmup
- Cosine learning-rate decay
- Gradient clipping at norm 1.0
- Early stop after 8 non-improving validation epochs

An epoch draws 25,600 anchors with replacement. It is not a complete corpus pass.

### Interpret status

Monitor `metrics.json`. Key fields are:

- Epoch and global step
- Learning rate
- Training total, candle, and leg losses
- Validation total, candle, and leg losses
- Best validation total
- Whether `best.pt` changed

Interpret them as trends:

- Healthy learning: validation total and both components generally decline.
- Overfitting: training falls while validation stalls or rises repeatedly.
- Objective imbalance: one component improves while the other degrades.
- First best checkpoint: expected and not evidence of convergence.
- Early warmup: too early for a final quality judgment.

Candle MSE is measured in context-standardized OHLCV units. Leg SmoothL1 is
measured on `log1p` durations. Neither is price error, directional accuracy, or
profit.

### Understand data exposure

Compute anchor draws as:

```text
global_step x batch_size
```

Because sampling uses replacement, draws are not unique anchors. Neighboring
anchors also reuse many raw bars. A status report should state:

- Total draws
- Reconstructed or estimated unique anchors
- Total eligible anchors
- Unique-anchor coverage

### Resume after interruption

Resume only when all load-bearing identities remain unchanged. Use the same
training command. Do not manually copy or rename checkpoint files.

If provenance fails, choose one of 2 honest paths:

1. Restore the exact original source, config, lock, and data.
2. Start a new run identity from the intended inputs.

Do not weaken provenance checks or enable overwrite to force resume.

## Phase 9: evaluate and export

Preferred command:

```bash
just validated-export mantis-v2/configs/nextleg-myhost-mps.toml
```

This evaluates and exports one immutable checkpoint snapshot. Use separate stages
only for diagnosis:

```bash
just evaluate mantis-v2/configs/nextleg-myhost-mps.toml
just export mantis-v2/configs/nextleg-myhost-mps.toml
```

Completion requires:

- Validation metrics are finite.
- Evaluation is bound to exact `best.pt`.
- `best.pt` is the minimum-loss durable epoch.
- Config, data, source, lock, and upstream identities match.
- All tensors reload exactly from safetensors.
- Fixed-fixture native and export outputs match tolerances.

Record the export manifest, model hash, evaluation hash, best epoch, global step,
and metrics. Do not claim holdout or trading performance.

## Phase 10: run downstream stages

Copy and review the downstream config first. Confirm its foundation manifest path
and weights digest match the export.

Run stages independently for bounded recovery:

```bash
just trend-magic-verify
just trend-magic-prepare
just trend-magic-embed
just trend-magic-head
```

`trend-magic-topstep-100k-head-c0001-v2.toml` is the production consumer for
walk-forward. It pins and reuses the completed embeddings from
the producer config, changes only head fitting, and writes under a distinct run
identity. Do not run the all-stage chain with this consumer config because its
purpose is isolated head fitting against immutable embeddings.

`trend-magic-verify` is read-only. It prints the named
`trend_magic_fixed_3r_v1` contract and its workflow identities without reading
market data or writing artifacts. The loader rejects drift from every eligible
closed 3m Trend Magic state bar, next-open entry, ATR(20) risk at 0.5 ATR,
strict 3R-before-stop labels, the 2/3/4/6R analysis ladder, 120 session-bounded
bars, 0.03R cost, and stop-first ties. The separate 2R trail belongs to
execution replay and never changes this supervised label.

No all-stage Trend Magic command is currently qualified. A future producer
config may use `downstream-run` only after its embedded head settings have
passed the exact-fold convergence diagnostic and that config has its own
documented run identity.

Each stage verifies upstream hashes and writes a manifest. Production stages fail
closed on partial output when overwrite is false. Do not delete partial outputs
before diagnosing which durable stage completed.

### Qualify the dependency-light RL entry environment

Build training and validation schedules for the same fold, then validate them:

```bash
just rl-build-episodes mantis-v2/configs/rl-entry-topstep-100k.toml 0 training 21
just rl-build-episodes mantis-v2/configs/rl-entry-topstep-100k.toml 0 validation 21
just rl-validate-environment \
  /Volumes/Storage/trading-research/artifacts/mantis-foundation-model/rl-entry-topstep-100k-v1/episodes/fold-00-training-seed-42.json \
  /Volumes/Storage/trading-research/artifacts/mantis-foundation-model/rl-entry-topstep-100k-v1/episodes/fold-00-validation-seed-42.json \
  /Volumes/Storage/trading-research/artifacts/mantis-foundation-model/rl-entry-topstep-100k-v1/environment-validation.json \
  mantis-v2/configs/rl-entry-topstep-100k.toml
```

The validator accepts only training and validation manifests below the sealed
holdout boundary. It rechecks manifest/config/embedding identities, loads frozen
features with mmap, replays complete 3-minute bar episodes, and writes a new
atomic result. Completion requires finite causal observations, identical shared
action-mask mechanics for all policies, exact deterministic baseline replay,
mmap p95 latency with at least 100x headroom against three minutes, and at least
5,000 environment steps per second on the recorded host. The historical
logistic baseline is loaded from the exact rejected fold head. The fixed
HistGradientBoosting baseline fits training only and obtains its threshold from
validation only. Test schedules and January-July 2026 holdout data are rejected.

### Prepare

Verify:

- Every eligible closed 3-minute state bar becomes a candidate.
- Candidate direction uses the configured Supertrend or Trend Magic state.
- Entry is next-open, never same-close.
- Stop, target, cost, tie, horizon, and session-close policies match the config.
- Normal preparation excludes holdout labels.

### Embed

Verify:

- Foundation manifest and weights digest match.
- Preprocessing matches the exported foundation contract.
- Shard metadata is complete.
- Expected float16 `.npy` shards and Parquet metadata exist.
- The manifest binds source rows to shard outputs.

### Walk forward

The head must converge on every fold and beat both class-balanced constant
baselines before simulation. A convergence failure is durable evidence: retain
`failure.json`, diagnose on the exact capped fold, and create a new run identity
for any changed head parameters. Never patch an old manifest or copy embeddings
into a new directory. Reuse requires exact embed-manifest and producer-config
paths and SHA-256 values in the consumer TOML.

The Trend Magic `head-c0001-v2` production run is a recorded rejection, not a
simulation candidate: all eight folds converged, but mean test weighted log loss
0.695306 and Brier 0.251044 failed their 0.693147 and 0.250000 baselines. Do not
run simulation or holdout evaluation for that run identity.

Verify:

- Training, validation, and test windows are chronological.
- Event spans are purged.
- Context is embargoed.
- Scaler and logistic model fit on train only.
- Validation owns the persisted probability threshold.
- Test uses the unchanged threshold.

### Simulate

Verify:

- Contract multipliers are correct.
- Costs and position sizing match config.
- Maximum Loss Limit and consistency rules match the selected profile.
- Session dates use America/Chicago.
- Results are not selected using future realized exit time.

## Phase 11: keep the holdout sealed

Routine foundation evaluation cannot access holdout. Do not edit that guard.

Downstream holdout access requires a reviewed config and exact unlock:

```bash
just downstream-holdout \
  mantis-v2/configs/supertrend-reviewed-holdout.toml \
  REVIEWED-ONE-TIME-HOLDOUT
```

Use this only after the final foundation export, classifier definition,
threshold, simulator, and acceptance criteria are frozen. Run it once. Record the
result without tuning against it.

## Completion checklist

### Foundation adaptation is complete when

- Training completed or early-stopped normally.
- Durable metrics are contiguous.
- `best.pt` is validation-selected.
- Evaluation reproduced the selected metrics.
- Safetensors exact reload and inference parity passed.
- Export manifest hashes match all inputs.
- The result is documented as validation evidence, not holdout or trading proof.

### Downstream workflow is complete when

- All 4 stage manifests exist and chain correctly.
- Walk-forward folds preserve purge, embargo, and validation ownership.
- Simulation uses explicit execution, cost, and account rules.
- Results are compared with required traditional baselines.
- Holdout remains sealed unless one-time evaluation was authorized.

### A completion report must include

- Config path and run name
- Device and host-relevant versions
- Source commit and dirty state
- Dataset identity and anchor counts
- Best epoch, step, and component metrics
- Evaluation batch count and exact reproduction status
- Export weights and manifest hashes
- Downstream fold and simulation summaries when applicable
- Holdout status
- Every limitation or unqualified platform path

## Related documentation

- [Setup, dependencies, and hardware](setup-and-hardware.md)
- [System architecture](architecture.md)
- [AI-agent runbook](agent-runbook.md)
- [Troubleshooting](troubleshooting.md)
- [MantisV2 package reference](../mantis-v2/README.md)
