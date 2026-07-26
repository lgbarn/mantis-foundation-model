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
| `just probe-cuda-fp32 ...` | Strict 32-update CUDA FP32 probe and validated export | Unique no-overwrite evidence | Real CUDA, pinned image, data, and official weights |
| `just train <config>` | Trains or resumes | Checkpoints, metrics, provenance | Long accelerator run; mutates run state |
| `just tensorboard <run-root>` | Serves run events on `127.0.0.1:6006` | Nothing | Read-only; use an SSH tunnel remotely |
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
| `just rl-smoke <output> <config> [--resume]` | Trains the bounded synthetic MaskablePPO qualification | Atomic checkpoint, state, metrics, manifest | 50K CPU steps; no production or holdout data |
| `just rl-train <training_manifest> <output> <config> [variant] [--resume]` | Trains all declared development seeds on one production training partition | Atomic fold/seed checkpoints, metrics, seed summary | CPU only; no validation, test, or holdout input |
| `just runpod-official-bootstrap <archive> <receipt>` | Archives a clean commit and emits the official-template provenance receipt | Source tarball and canonical JSON | Local only; default supported RunPod route |
| `just runpod-image-build <image>` | Builds the legacy pinned Linux amd64 CUDA image from a clean commit | Local Docker image | Self-supported fallback; Docker CPU/disk/network use |
| `just runpod-image-scan <image> <output>` | Scans saved image layers and history | Canonical scan JSON | Requires a local Docker daemon |
| `just runpod-image-self-check <image> <output>` | Verifies image architecture, tools, lock, and source before launch | Canonical static image JSON | Docker emulation is supported; CUDA is checked first inside the Pod |
| `just transfer-stage-runpod <config> <local> <decision>` | Rehashes and pre-stages the immutable bundle through RunPod S3 | Network-volume objects | Live storage writes; no paid Pod |
| `just runpod-seal-workload <spec> <output-root>` | Seals all training, data, image, watchdog, and budget identities | Content-addressed workload manifest | Local only |
| `just runpod-bind-workload <manifest> <decision> <pod-path> <time> <output>` | Binds an approved provider decision to one immediate workload | Immutable bound decision | Local only |
| `just runpod-supervise-workload <manifest> <decision> <local> <runpodctl>` | Verifies staging, creates, monitors, terminates, reconciles, and replicates one Pod | Durable receipts and replicated artifacts | The only permitted paid Pod create path |
| `just runpod-launch <decision> <local>` | Rejects unsupervised Pod creation | Nothing | Always fails closed |
| `just runpod-status <pod-id> <run-name> <local>` | Reads allowlisted status for one receipted Pod | Redacted JSON status | Real provider read |
| `just runpod-terminate <pod-id> <run-name> <local>` | Terminates one exact receipted Pod idempotently | Durable termination receipt | Real provider mutation |
| `just runpod-enforce-deadline <pod-id> <local>` | Enforces the durable hard deadline | Durable termination receipt | Real provider mutation after deadline |
| `just transfer-bundle <config>` | Hashes an immutable source snapshot | Canonical no-overwrite manifest | Full local read; no network access |
| `just transfer-stage-dry-run <config> <inventory>` | Exercises injected S3 HEAD/PUT policy | JSON plan on stdout | Zero network and zero object writes |
| `just transfer-promote <config>` | Verifies mounted bytes and atomically renames incoming | Immutable final input directory | Fails closed before promotion |
| `just transfer-backup-verify <config> <artifact-digest>` | Verifies distinct internal and external copies | JSON receipt on stdout | Full read of both copies; performs no copy |

### Official frozen expected-R paid screen

Use `mantis-v2/configs/frozen-paid-control.example.json` as the schema for one
ignored machine-local control config. The supported order is:

1. `just frozen-screen-prepare` creates the immutable candidate/window input
   from accepted Parquet before a Pod exists. It does not audit raw data.
2. `just transfer-bundle` records the prepared input, frozen config, and pinned
   offline Hugging Face cache. Replace the bundle digest in the control config.
3. `just frozen-screen-plan-paid <control> <output>` runs the four focused tests
   in under 60 seconds and emits `preflight.json`, `experiment.toml`,
   `intent.json`, and `workload-experiment.json`. It creates no provider state.
4. Run `just runpod-plan` against a current inventory/ledger, create the exact
   short-lived authorization for its subject digest, then run
   `just runpod-plan-authorized` with the unchanged inputs.
5. `just transfer-stage-runpod` uploads the immutable bundle without a GPU.
6. `just frozen-screen-seal-paid` creates the content-addressed official-template
   workload manifest and binds the approved decision. No workload JSON is
   hand-authored.
7. `just frozen-screen-run` is the sole paid create. It validates the preflight,
   staged input, manifest, decision, source, config, checkpoint, price, budget,
   and deadline before provisioning one L40S.

The paid command runs frozen embedding and corrected CUDA initial-screen plus
three development folds on the same Pod. Its one shell retry uses the same
provenance: complete embedding shards resume, and a complete comparison returns
only after its digest and input/config/weights/precision/parity identities match.
A partial or mismatched comparison fails again and the supervisor terminates.
The supervisor polls every 30 seconds, caps the run at the lesser of six hours
and `$10 / hourly_rate`, replicates the full run root (embedding shards and
manifest plus comparison result/progress) to both configured destinations,
deletes the Pod, verifies provider absence, and reconciles billing. No holdout,
simulation, PPO, fine-tuning, raw-data audit, or TensorBoard runs in this stage.
| `just transfer-retention-check ...` | Re-verifies backups and evaluates exact authorization | JSON decision on stdout | Decision only; never deletes data |

For remote runs, start the fixed localhost-only server on the remote host, then
open an SSH tunnel from the local machine:

```bash
just tensorboard /network/volume/runs/RUN_ID
ssh -N -L 6006:127.0.0.1:6006 root@POD_HOST -p POD_SSH_PORT
```

Open `http://127.0.0.1:6006`. Public and wildcard binds such as `0.0.0.0` are
rejected. TensorBoard is observational; JSON, checkpoints, and manifests remain
authoritative if event writing fails.

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

### RunPod runtime preflight

The production default is RunPod's official PyTorch 2.8 template, not a custom
container. From a clean commit, create the hash-pinned source archive and
deployment receipt with `just runpod-official-bootstrap`, then include both in
the sealed workload. The receipt pins official template `runpod-torch-v280` and
the immutable RunPod image digest. The bootstrap starts the template's supported
`/start.sh`, uses the image-bundled `uv` 0.9.0, and syncs the frozen lock into the
persistent network-volume cache before executing the workload manifest.

The custom image build/scan workflow remains documented in
`infra/runpod/README.md` for recovery comparisons, but it is self-supported and
is not the normal production route. Both routes still require the in-Pod CUDA,
driver, source, lock, and allocation self-check before data promotion or training.

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

The committed FP32 qualification contract is
`mantis-v2/configs/cuda-fp32-qualification.toml`. On the pinned CUDA image, use
a machine-local pipeline config with real pre-holdout paths and a unique absent
run identity:

```bash
just probe-cuda-fp32 \
  mantis-v2/configs/nextleg-parquet-v2-probe.toml \
  mantis-v2/configs/cuda-fp32-qualification.toml \
  cuda-fp32-probe-YYYYMMDDTHHMMSSZ-1234abcd \
  /workspace/artifacts
```

The command fails closed unless it can use explicit CUDA, pinned memory, the
official base checkpoint, exactly 32 real optimizer updates, one validation
batch, no resume or overwrite, and the sealed holdout remains unavailable. It
then evaluates the selected checkpoint and verifies safetensors reload parity
at `atol=1e-5`, `rtol=1e-4`. A CPU host emits a canonical skip record without
opening data or artifact paths.

The same qualification module owns the fixed-input CPU/CUDA output, component
loss, total loss, gradient, and optimizer-update oracle; fresh-process batch
doubling through a classified OOM with both 80% memory caps; exact 32 versus
16+restart equivalence; both fine-tune parameter-count contracts; environment
identity; and stale evaluation/export refusal. CPU CI validates these schemas,
errors, tolerances, and skip reasons. Only a reviewed real-CUDA evidence bundle
can qualify a GPU shape; the committed code and a CPU skip do not.

BF16 is an explicit, fail-closed candidate rather than a default. Set
`training.precision="bf16"` only on explicit CUDA after
`torch.cuda.is_bf16_supported()` succeeds. Forward and loss computation use
BF16 autocast while parameters and optimizer state remain FP32; no loss scaler
or automatic retry is used. Config, provenance, checkpoint, evaluation, export,
TensorBoard, and qualification evidence all record the precision identity.

The preregistered policy is
`mantis-v2/configs/cuda-bf16-qualification.toml`. A reviewed paid run captures
the public fixed fixture in FP32 and BF16, 32 uninterrupted versus 16+restart
BF16 state, native versus reloaded export evidence, and all precision records.
Apply the immutable evidence gate with:

```bash
just qualify-cuda-bf16 fp32.json bf16.json qualification.json
```

For an unsupported operation, OOM, overflow, non-finite value, or other paid
attempt failure, record the rejection instead:

```bash
just reject-cuda-bf16 fp32.json 'CUDA out of memory' qualification.json
```

Both commands write a new no-overwrite decision. A failed comparison selects
FP32 after exactly one attempt without reducing registered tolerances. CPU tests
exercise the policy and fake failures, but only reviewed real-CUDA evidence can
promote BF16. Until that evidence passes, FP32 remains selected.

Downstream embedding accepts only explicit `cpu`, `mps`, or `cuda`; there is no
Linux auto-fallback. The production embed command requires an explicitly
promoted foundation manifest and the ordered `1min, 3min, 5min, 15min`
contract before reading prepared market inputs. It commits each feature/metadata
pair with a final atomic receipt. Restart rehashes every complete pair, resumes
an incomplete matching pair without deleting it, and rejects any export,
producer config, corpus, source, lock, width, row-span, or byte drift.

Pre-promotion matrix scoring uses `export_role="diagnostic_candidate"`; it must
never be relabeled as promoted. After CPU and CUDA fixture files, pair receipts,
and performance evidence have been captured on the authorized host, apply the
zero-cost evidence gate with `just qualify-cuda-embedding ...`. The preregistered
policy is `mantis-v2/configs/cuda-embedding-qualification.toml`; it pins the
official MantisV2 source and weights, requires maximum absolute difference at
most 0.01, per-row cosine at least 0.999, exact metadata order, finite values,
checkpoint-free restart, and a measured four-timeframe footprint. CPU fixtures
exercise this gate in CI; only reviewed real-CUDA evidence qualifies throughput.

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
  /Volumes/Storage/trading-research/artifacts/mantis-foundation-model/rl-entry-topstep-100k-v13/episodes/fold-00-training-seed-42.json \
  /Volumes/Storage/trading-research/artifacts/mantis-foundation-model/rl-entry-topstep-100k-v13/episodes/fold-00-validation-seed-42.json \
  /Volumes/Storage/trading-research/artifacts/mantis-foundation-model/rl-entry-topstep-100k-v13/environment-validation.json \
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
The v4 schedule uses an independent versioned weekday exchange calendar and a
15-minute maximum observed gap. Observation schema v1 records explicit contract,
quantity, stop-risk, tick, fee, best-day, consistency, and action-mask fields.
Unit tests verify deterministic benchmark contracts; real timing thresholds are
enforced only by the target-Mac validation command above.
Validation also executes 17 independent literal replay oracles covering every
supported ticker/profile economics tuple plus discontinuity, MLL, DLL, and
session-flatten account paths. The economics cases read the emitted quantity,
submit and record the literal action sequence, require one accepted trade and a
TIMEOUT terminal status, and include a sub-2R retrace before the 3.1R excursion
that activates the locked 2R/0.75R trail. Any action, quantity, fill, fee, exit,
balance, equity, best-day, consistency, event, or terminal mismatch fails
publication.

### Qualify MaskablePPO on CPU

Use a unique local output directory for the bounded Stage 1 smoke:

```bash
just rl-smoke \
  artifacts/rl-entry-smoke-v1 \
  mantis-v2/configs/rl-entry-smoke.toml
```

If the process stops after an atomic checkpoint, resume the same identity with:

```bash
just rl-smoke \
  artifacts/rl-entry-smoke-v1 \
  mantis-v2/configs/rl-entry-smoke.toml \
  --resume
```

Production entry-policy training starts only from the qualified training
schedule. The default candidate and both preregistered ablations share one seam:

```bash
just rl-train \
  /Volumes/Storage/trading-research/artifacts/mantis-foundation-model/rl-entry-topstep-100k-v13/episodes/fold-00-training-seed-42.json \
  /Volumes/Storage/trading-research/artifacts/mantis-foundation-model/rl-entry-topstep-100k-v13/training/shared-ticker-value \
  mantis-v2/configs/rl-entry-topstep-100k.toml \
  shared_ticker_value
```

Use `independent_actor` or `shared_critic` only for the named architecture
ablation. Training replays complete episodes with separate reward and cost
objectives. PASS reward=1 and BLOW cost=1 only at termination; both use gamma
1.0. Ticker-specific reward and cost advantages are standardized independently;
cost-value targets and dual updates retain raw binary outcomes. The actor surrogate uses
`A_reward - lambda*A_cost`. The projected multiplier starts at
1.0, updates once per complete episode batch from raw BLOW indicators toward a
0.01 cost limit at learning rate 0.01, and is capped at 100. Minimum MLL cushion
is an observation and logged path metric only, never reward shaping or dense
cost. Old policy log probabilities plus reward and cost values are frozen
before the configured PPO epochs. Each minibatch has exactly equal samples per
present ticker, with deterministic oversampling of shorter ticker streams. A
checkpoint is an immutable complete-cycle bundle containing both critics,
optimizer, multiplier/controller state, raw cost statistics, and RNG state; an
atomic pointer selects the latest bundle, and resume recovers a semantically
valid orphan bundle after a pointer crash. Resume with a final `--resume`; any
source revision/dirty state, lock, config, constraint definition, schedule
bytes, loaded episode collection, embedding, corpus, foundation, rule, fee,
seed, fold, cursor, variant, training mode, or requested budget mismatch fails
before state is loaded. Pre-constraint checkpoint schemas are rejected.

For a bounded mechanics qualification, append the final just argument
`'--target-updates 1'`. This still loads every episode in the declared schedule
and reports every configured development seed plus the worst seed, but it does
not satisfy the 2M-step quality budget and retains `quality_claim=false`.

The smoke uses the MIT-licensed, hash-locked local stack: Gymnasium 1.3.0,
Stable-Baselines3 2.9.0, SB3-Contrib 2.9.0, and Optuna 4.9.0. No paid or hosted
service is enabled. The first three provide the local CPU training path; Optuna
is pinned for the later search stage and is not invoked by this smoke.
MaskablePPO runs on
CPU. It trains for exactly 50,000 steps on deterministic synthetic episodes
that exercise the same production observation, transition, and action-mask
adapter. It must beat reject-all, keep every policy value finite, submit no
illegal action, reproduce the seeded schedule, and match deterministic actions
after checkpoint reload. Immutable 10K checkpoint bundles contain model,
optimizer, Python/NumPy/Torch/Gym RNG, schedule trace, and numerical-audit
state. Atomic `state.json` points at the last complete bundle, so a failed save
or pointer swap cannot invalidate it. `manifest.json` binds the checkpoint and
metrics to the exact config, source, lock, schedule, policy, dependency
versions, and seed.
The command never reads a production episode manifest, market corpus, embedding
shard, or sealed-holdout file. Passing this smoke permits later bounded fold
work; it is not evidence of trading quality or permission to start Optuna.

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
- The foundation export role is `promoted`; diagnostic candidates are rejected.
- Preprocessing matches the exported foundation contract.
- Timeframes are ordered exactly as `1min, 3min, 5min, 15min`.
- Shard feature/metadata pairs have complete atomic receipts.
- Expected float16 `.npy` shards and Parquet metadata exist.
- The manifest binds source rows, producer, corpus, source, lock, export, width,
  row spans, performance, and shard bytes to the outputs.

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
