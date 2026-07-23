# RunPod implementation handoff

Status: accepted planning handoff from GitHub issue #20. Nothing in this file is
an existing command or permission to create billable resources unless the
repository already documents it as such.

## Outcome

Implement a config-driven RunPod control plane for the existing MantisV2
workflow. It must preserve causal data handling, immutable provenance, exact
resume, atomic artifacts, validation-owned selection, and the sealed 2026
holdout.

The route uses:

- one on-demand Secure Cloud Pod at a time;
- a provisional NVIDIA H100 SXM 80 GB qualification target;
- one 150 GB Standard network volume;
- a digest-pinned Linux amd64 CUDA image;
- Terraform for stable non-secret resources;
- a pinned `runpodctl` adapter for Pod launch and termination until provider
  parity is verified;
- direct S3-compatible transfer for the network volume;
- localhost-only TensorBoard through SSH; and
- JSON manifests and atomic checkpoints as the authoritative record.

Ansible is not part of the route. Do not add it without a measured need for
mutable long-lived host configuration.

## Non-outcomes

- No RunPod GPU, CPU host, precision mode, batch size, or runtime is qualified
  yet.
- No paid resource has been created.
- The funded $150 is a hard ceiling and the operator has granted standing
  permission for this documented pipeline within that ceiling. Exact launch
  identity, fresh price and stock, and every fail-closed gate still apply.
- This is supervised futures adaptation from the pinned official MantisV2
  checkpoint, not random-weight base pretraining.
- The corpus may span July 2021 through July 2026, but ordinary training,
  tuning, walk-forward selection, Optuna, and Monte Carlo exclude the configured
  2026 holdout.
- Derived weights must not be published to Hugging Face while the upstream
  license declarations remain in conflict.

## Control-plane ownership

| Owner | Responsibility |
| --- | --- |
| Terraform | Standard network volume, private Pod template, non-secret stable resource identity, import/adoption checks |
| Pinned launch adapter | Exact Pod request, one-live-Pod lock, deadline, status, termination, cost reconciliation |
| Container image | CUDA-compatible userspace, Python 3.12, pinned `uv`, locked dependencies, system packages, self-checks |
| `just` | Human-facing preflight, plan, transfer, qualification, monitoring, termination, download, and verification workflows |
| Typed TOML | Experiment definition, runtime resources, paths, precision, matrix arms, seeds, budgets, thresholds |
| Python pipeline | Training, evaluation, export, downstream, RL, manifests, metrics, checkpoint and parity behavior |
| Local controller | Secrets, SSH, S3 credentials, Terraform state, approvals, spend ledger, independent watchdog |

Terraform and the launch adapter must never both create the same Pod. Terraform
owns the template and volume; the adapter consumes their recorded identities.
An import/adoption preflight must reject duplicate or unmanaged live resources.
The image contains no datasets, upstream or derived weights, credentials, or
run artifacts.

The one-live-Pod guard combines a local process lock on the single authorized
controller host with a fresh remote inventory snapshot. Other controller hosts
are unsupported. The lock remains held across the remote check and create call
so concurrent local invocations cannot pass the same check-create window.

## Configuration contract

Implementation must add:

1. A committed, versioned, strict RunPod platform config containing provider
   units, GPU fallback order, minimum CPU/RAM, container disk, volume size,
   high-water threshold, deadlines, budget buckets, alert thresholds, and
   allowed datacenters.
2. Committed portable MantisV2 experiment configs for the CUDA probe,
   full-upstream arm, transformer-only control, LoRA arms, downstream producer
   and consumer, and CPU RL stages.
3. A deterministic matrix renderer for seeds and arm identities. Every resolved
   TOML is stored and hashed in its run directory; command-line switches may
   select a declared arm/seed but may not silently change the experiment.
4. Fixed remote paths below `/workspace/mantis`. Local source/backup paths live
   in an ignored typed local config, not committed experiment TOMLs.
5. Strict rejection of unknown keys, incompatible precision/device pairs,
   missing hashes, reused run names, and path overlap between active runs.

Only secret names may appear in committed configuration:

- `RUNPOD_API_KEY` for the local control plane;
- `RUNPOD_S3_ACCESS_KEY_ID` and `RUNPOD_S3_SECRET_ACCESS_KEY` for the local S3
  transfer process; and
- a scoped RunPod registry-auth object only if a private image is unavoidable.

Do not place secret values in git, `.tfvars`, Terraform state, image layers,
commands, the network volume, TensorBoard, or manifests. The local account
provisioning key and S3 data-plane keys never enter the Pod. RunPod
automatically supplies its own Pod-scoped API key; workloads must not persist,
log, export, or treat that scoped key as account provisioning authority. SSH
private keys never leave the local controller.

## Persistent layout

```text
/workspace/mantis/
|-- inputs/
|   |-- raw/<bundle-sha256>/
|   |-- corpus/<corpus-manifest-sha256>/
|   `-- upstream/<model-revision>/
|-- cache/{huggingface,uv}/
|-- runs/<immutable-run-name>/
|-- exports/<immutable-run-name>/
|-- logs/tensorboard/<immutable-run-name>/
`-- transfer/{incoming,outgoing}/<bundle-sha256>/
```

Use the provider/API value `150 GB`; do not call it 150 GiB. Before every
stage, measure the mounted filesystem in bytes, enforce the 120 GB high-water
policy, and require at least 30,000,000,000 free bytes.

## Execution DAG

### A. Zero-cost readiness

```text
source/spec/issues
  -> local green gate
  -> image build and self-test
  -> Terraform plan and policy checks
  -> transfer-manifest dry run
  -> launch request dry run
  -> explicit spend approval
```

Required implementation includes the pinned image, CUDA probe, typed precision,
TensorBoard writer, resource instrumentation, cross-device parity fixtures,
CUDA embedding, transfer verifier, cost ledger, one-Pod lock, and independent
watchdog. Do not provision before all zero-cost checks pass.

### B. Input transfer

```text
external source
  -> immutable internal-SSD snapshot
  -> relative-path/size/SHA-256 bundle manifest
  -> S3 API incoming directory
  -> in-Pod byte and SHA-256 verification
  -> atomic promotion to content-addressed inputs
```

The measured disaster-recovery input bundle is 1,395,349,697 bytes. It contains
three DBN.ZST sources, the 64-file repaired corpus, and the pinned official
MantisV2 cache. Do not upload the 54.6 GB historical artifact tree wholesale.
S3 ETags are not provenance. Rsync over SSH is a paid-time fallback only.

### C. First paid qualification gate

```text
environment/I-O inventory
  -> FP32 batch envelope
  -> CPU/CUDA parity
  -> BF16 qualification
  -> interrupted-resume equivalence
  -> full-upstream/transformer resource comparison
  -> CPU/CUDA embedding parity and throughput
  -> CPU RL host qualification
  -> TensorBoard/JSON/checkpoint audit
  -> verified download
  -> Pod termination
```

The first H100 SXM Pod has a 7,200-second workload, a 1,800-second startup
allowance, and 120 seconds of shutdown grace, for a 9,120-second wall-clock cap.
At the observed $2.99/hour Secure Cloud price, its maximum projected
qualification spend is $7.60. Sequential L40S or A40 fallback probes are manual
only and require a fresh exact decision. Total qualification compute plus
container-disk overhead remains below $10.

No model accuracy decision uses the smoke benchmark. It selects a safe device,
precision, batch envelope, throughput estimate, and production cost projection.

### D. Foundation accuracy route

```text
pinned official MantisV2 revision and weight SHA-256
  -> compute-matched 3-TF vs 4-TF full-upstream screen
  -> promote/reject 5-minute context on pre-holdout validation
  -> 4-TF preserved-exposure full/transformer/LoRA screen
  -> five-seed confirmation
  -> validation-selected best checkpoint
  -> evaluation bound to best checkpoint
  -> safetensors export and native parity
```

The universe is `ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN` at
`1min,3min,5min,15min`. The 3-minute slice is primary. Start from official Hub
revision `99fe0f548960e272fbfa4b82fd9b5b5956779dfd` and weights SHA-256
`49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1`.
Never fall back to random weights.

`full_finetune` already exists and trains the loaded upstream encoder plus new
NextLeg heads. `transformer_finetune` is the control. LoRA must target MantisV2
`wQKV` and `wO` layers at ranks 8/16, keep base and adapter identities separate,
and pass merged parity before promotion.

Screen seeds 42-44 and confirm seeds 42-46. If four-timeframe training is
promoted, preserve per-stream exposure with 267 training and 27 validation
batches at batch size 128 unless the benchmark selects a different equivalent
effective batch. Promotion requires at least four of five confirmation seeds,
improved median macro 3-minute results, no material instrument-family
regression, and improved downstream log loss and Brier score. A LoRA arm must
also pass the registered non-inferiority margins from issue #28.

### E. Trend Magic downstream route

```text
accepted foundation export
  -> new portable producer config
  -> Trend Magic contract verification
  -> causal candidate/label preparation
  -> frozen-encoder embedding
  -> walk-forward head
  -> convergence/log-loss/Brier gate
  -> Topstep 100K simulation
  -> frozen candidate for later holdout
```

Trend Magic owns direction. Every eligible closed 3-minute state bar is a
candidate; this is not flip-only. Entry is next-open. The supervised label uses
the fixed 3R-before-stop recipe and never uses the execution-only trail after
2R. The execution replay retains the accepted 2R activation and 0.75R trail.

Current Trend Magic artifacts cannot be reused silently. They use seven symbols
and three timeframes and bind a rejected foundation/head identity. The new
producer must explicitly decide and bind the promoted four-timeframe embedding
contract. The existing eight-fold head is rejected because weighted log loss
0.695306 and Brier 0.251044 failed their baselines; simulation and holdout are
prohibited for that identity.

### F. Topstep RL route

```text
accepted pre-holdout embeddings and Trend Magic direction
  -> production trainer review/merge
  -> identity dry run
  -> training/validation episode schedules
  -> environment validation on target CPU host
  -> CPU smoke and exact resume
  -> validation-owned Optuna search
  -> development seeds
  -> frozen chronological confirmation
  -> synchronized-block Monte Carlo
  -> promotion gates
  -> serving export/reload parity
  -> optional reviewed one-time holdout
```

RL stays CPU-only until separately qualified. The actual RunPod host must exceed
5,000 environment steps/second and retain 100x mmap-latency headroom. Trend
Magic owns trade direction; the accepted policy/action contract owns only its
declared entry, exit, and sizing decisions. Do not broaden actions in the
infrastructure work.

PR #19 remains open and must pass the normal `/autofix -> /merge` gate. Existing
issues #7-#12 own production training, Optuna, confirmation, evaluation,
export, and holdout. Do not create duplicate RL issues. Optuna is currently
locked but not operational. Preserve its maximum 30 trials x 500K steps x 3
seeds, followed by 2M-step development and 5M-step confirmation within the
accepted hard caps.

PR #19 must also satisfy the accepted constrained-PPO objective. It may not
collapse terminal blow cost into scalar reward shaping or omit the separately
checkpointed cost-value/constraint state merely because a bounded smoke passes.

Final RL promotion includes at least 300 fixed chronological attempts, pooled
pass rate at least 60%, each-seed pass rate at least 50%, pass-rate lower bound
at least 50%, zero observed blows, blow-rate upper bound at most 1%, paired
bootstrap improvement over baselines, and no hidden ticker/profile failure.

## Observability

TensorBoard binds to `127.0.0.1:6006` and is viewed only through an SSH tunnel.
Event files persist beneath the active run directory. The Run Identity is
immutable, while checkpoints, event logs, and the In-Progress Checkpoint
Pointer evolve atomically until each Completed Artifact is finalized. Record:

- train/validation total, candle, and leg losses;
- learning rate, gradient norm, epoch, and global step;
- examples/updates per second and data wait;
- GPU allocated/reserved memory and utilization;
- host RSS, filesystem free bytes, and checkpoint duration/size;
- initialization mode, base revision/digest, trainable/frozen counts, seed,
  precision, and resume source; and
- best-checkpoint, early-stop, failure, and shutdown events.

TensorBoard is diagnostic. JSON manifests, metrics, checkpoints, and hashes are
authoritative and must be sufficient when TensorBoard events are absent or
damaged.

## Failure and resume

- First launch requires an absent run directory and `allow_overwrite=false`.
- Resume may load only `pending.pt`/`latest.pt` owned by the same exact run
  identity and provenance. Never rename, copy, or patch a checkpoint to force
  resume.
- Checkpoint provenance binds config, data, source, lock, upstream revision and
  weights, contamination state, precision, image, and initialization identity.
- Every stage writes atomically and preserves `failure.json` plus the last valid
  checkpoint. A failed stage does not erase partial evidence.
- A stopped Pod is replaced only by an explicitly approved Pod attached to the
  same volume. There is no automatic paid retry.
- Download every expensive completed stage to the internal SSD and verify its
  transfer manifest, then make and verify the external-drive copy.
- Remote deletion requires both local copies, explicit operator approval, and
  a recorded retention action. Network-volume deletion is last.

## Spend and lifecycle gates

- $15 storage and overhead allocation.
- $10 cumulative qualification allocation.
- $100 production compute pool, unlocked only after qualification acceptance.
- $25 protected recovery reserve requiring new approval.
- One live Mantis Pod maximum.
- Auto-pay disabled and no automatic fallback or retry.
- Append-only cost ledger and reconciliation from RunPod billing records.
- Alerts at $50 and $100 total; reject ordinary launches at $125.
- Re-query price, stock, datacenter, S3 compatibility, and required account
  balance immediately before every launch.
- Independent local/server-side deadline must terminate the Pod even if the
  training process or SSH session fails.

## Required `just` interface

The implementation spec should define, test, and document recipes with these
responsibilities; final names may be refined once without changing code for
ordinary path or parameter changes:

- tool/image/version preflight and image self-test;
- Terraform format/validate/plan and policy checks;
- transfer manifest, upload, in-Pod verify, download, and backup verify;
- launch dry run, approved qualification launch, status, cost, and terminate;
- localhost TensorBoard tunnel instructions;
- CUDA foundation probe/parity/resume matrix;
- foundation arm/seed orchestration and validated export;
- CUDA embedding qualification and downstream stages;
- CPU RL environment/smoke/Optuna/confirmation/Monte Carlo/export stages; and
- final artifact and spend reconciliation.

Every mutating recipe must print its exact run/resource identity and refuse an
ambiguous default. Destructive recipes require exact IDs and an explicit
operator action.

## Acceptance gates

Implementation is ready for the first paid probe only when:

1. The repository gate and image self-test pass from a clean commit.
2. Dependency and tool versions are recorded after installation.
3. Terraform plan contains only the expected stable resources and no secret.
4. Launch dry run proves one-Pod locking, deadlines, budgets, and no public
   TensorBoard port.
5. Transfer dry run and local backup preflight pass.
6. CUDA/precision/resume/embedding fixtures and TensorBoard parser exist.
7. The holdout guard is tested and no unlock artifact exists.
8. The user explicitly approves the current price and bounded first launch.

Production is unlocked only after the paid qualification bundle passes every
gate in issue #24 and its projection fits the $100 pool.

## Handoff to the engineering workflow

Run the canonical sequence next:

```text
/scope-docs
  -> /draft-spec
  -> /slice-issues
  -> execution loop
```

The scope/spec must treat this file and issues #20-#28 as inputs, reconcile
with existing RL issues #7-#12 and open PR #19, and slice at least these
independent implementation areas:

1. strict RunPod platform/local config and secure Terraform state;
2. pinned CUDA image and reproducible tool bootstrap;
3. content-addressed S3 transfer, checksum, and two-copy backup;
4. lifecycle adapter, one-Pod lock, watchdog, and spend ledger;
5. TensorBoard and resource instrumentation;
6. CUDA FP32/BF16/parity/resume probe;
7. CUDA downstream embedding and parity;
8. portable foundation accuracy-matrix orchestration;
9. portable Trend Magic downstream orchestration;
10. RunPod CPU qualification and integration with issues #7-#12;
11. first paid qualification run as an explicit human-approved issue; and
12. production run, artifact repatriation, retention, and final operations
    documentation.

Keep billable launches human-approved. Code, tests, dry runs, and documentation
may remain AFK; provisioning, holdout unlock, and destructive retention are
HITL boundaries.
