# RunPod infrastructure

This directory contains the reproducible RunPod control-plane inputs for the
MantisV2 training pipeline. The decision map is GitHub issue #20. The accepted
implementation route is [IMPLEMENTATION_HANDOFF.md](IMPLEMENTATION_HANDOFF.md).

## Deterministic launch planning

`runpod-plan` is the zero-cost planning boundary. It reads only explicit files,
does not query RunPod, does not invoke a subprocess, and cannot create or modify
a provider resource. Inputs are strict and versioned:

- `configs/platform-v1.toml` is the committed non-secret resource and spend
  policy.
- An Experiment Config binds one portable scientific definition. The example
  file uses a placeholder definition digest and is not production authority.
- `configs/local.example.toml` documents the ignored machine mapping. Copy it
  to `infra/runpod/local.toml` and replace the controller hostname and paths;
  that filename is ignored. Live commands reject every other host before
  reading the API credential.
- Launch Intent, Inventory Snapshot, Spend Ledger, and optional Launch
  Authorization are explicit JSON inputs. Launch Intent pins the digest image,
  template, scoped registry-auth object, volume mount, and sole `22/tcp` port.
  The files under `examples/` are deterministic synthetic fixtures, not current
  inventory or authorization.

The rejection dry run below is deliberately historical and synthetic. It is
safe to execute because the Inventory Snapshot is a file, not a live query:

```bash
just runpod-plan \
  infra/runpod/configs/platform-v1.toml \
  infra/runpod/configs/local.example.toml \
  infra/runpod/configs/experiment-cuda-qualification.example.toml \
  infra/runpod/examples/intent-h100-qualification.json \
  infra/runpod/examples/inventory-synthetic.json \
  infra/runpod/examples/spend-ledger-empty.json \
  2026-07-21T12:02:00Z \
  /tmp/mantis-runpod-plan/launch-decision.json
```

Standard output contains only `decision_path` and `decision_digest`. The file
is canonical JSON and contains `allowed = false`, reason
`authorization_required`, and the exact `authorization_subject_digest` a human
would review. The output path must not already exist. Every target path and the
evaluation timestamp are required; there is no default environment, account,
GPU, datacenter, experiment, inventory, ledger, or output location.

An operator creates a Launch Authorization outside this command after reviewing
the exact subject, provider price, projected spend, maximum duration, and
expiry. Re-evaluate through the same CLI with the explicit authorization:

```bash
just runpod-plan-authorized \
  <platform> <local> <experiment> <intent> <inventory> <ledger> \
  <authorization> <evaluated-at> <new-output>
```

Authorization is exact, expiring, and single-use. The decision binds the
official REST v1 OpenAPI identity, version, and reviewed SHA-256 recorded in
`platform-v1.toml`; runtime schema retrieval is forbidden. A changed config, local
controller mapping, intent, price/inventory observation, ledger/reservation
state, duration, spend ceiling, expiry, or consumed authorization digest rejects
the decision. The lifecycle commands consume the approved decision, while the
planner itself remains side-effect free.

For the paid frozen screen, generate the two live planner inputs immediately
before planning instead of editing the synthetic examples:

```bash
just runpod-provider-snapshot \
  infra/runpod/configs/platform-v1.toml /path/to/runpod-local.toml \
  /path/to/frozen-paid-control.json /path/to/generated-intent.json \
  /path/to/runpodctl /path/to/provider-snapshot
```

This read-only command pins and verifies `runpodctl`, requires zero live Pods
and zero current provider spend, and reconciles every local Pod receipt through
its termination and spend receipts. Its `inventory.json` and
`spend-ledger.json` feed `runpod-plan`; `provenance.json` records the command
output hashes and the reviewed price/storage policy bindings. Existing output,
unreconciled receipts, unavailable secure stock, or mismatched network-volume
identity fail closed.

## Pod lifecycle

`just runpod-supervise-workload <manifest> <decision> <local> <runpodctl>` is
the only create path. It validates the content digest, approved local-config
digest, controller hostname, staged bytes, and unexpired authorization before
provider access. It takes the approved lock, reads fresh Pod inventory, writes
a create attempt, and performs exactly one create request. An unknown outcome
is reconciled by that supervisor; never repeat launch. Direct `runpod-launch`
always fails closed.

Use `runpod-status`, `runpod-terminate`, `runpod-reconcile-termination`,
`runpod-enforce-deadline`, and `runpod-reconcile-spend` with the exact receipt
identity. Receipts under the configured local state root are authoritative.
Termination is idempotent, billing lag remains pending, and provider payloads
are reduced to an allowlist so API keys, registry data, environment variables,
and SSH material cannot enter command output. These commands make real API
requests and launch/termination can change paid resources; fixture tests inject
a no-network transport. A known Pod that fails create-response validation is
recorded separately as `QUARANTINED`: it blocks another launch and remains
eligible for exact deadline enforcement and termination, but is never presented
as an approved Pod Receipt.

The REST adapter accepts the provider response spellings observed for image and
network-volume identity (`image`/`imageName` and nested
`networkVolume.id`/`networkVolumeId`), but rejects conflicting aliases. Requested
vCPU and RAM are provider minima; a larger allocation is valid and both requested
and observed resources are recorded. Provider timestamps are normalized to UTC
before lifecycle code consumes them. A resource-only quarantine may be
reconciled from fresh inventory after full identity, price, and minimum-resource
validation; every other quarantine still requires termination.

## Stable Terraform resources

The stable-resource workflow is documented in
[`terraform/README.md`](terraform/README.md). It pins OpenTofu and the REST
provider, inspects explicit inventory before planning, requires deliberate
imports for unmanaged matches, validates saved-plan JSON, and keeps apply behind
an exact short-lived human authorization. It owns only the 150 GB network volume
and private Pod template; it cannot own a Pod.

## Ownership

- Terraform owns stable non-secret resources such as the network volume and
  private Pod template.
- A pinned container image owns CUDA, Python, `uv`, system packages, and the
  repository runtime environment.
- A pinned official REST v1 adapter owns exact Pod launch, status, billing, and
  termination. It sends the approved GPU, CPU/RAM, registry, storage, template,
  and port fields and normalizes only reviewed response fields.
- Ansible is not part of the baseline. Add it only if a measured requirement
  for mutable, long-lived host configuration emerges.

## Safety boundaries

- Planning and validation must not create billable resources.
- Provisioning requires an explicit approved spend envelope.
- Read the account provisioning credential only from local
  `RUNPOD_API_KEY`; stable-resource recipes pass it through one ephemeral,
  sensitive OpenTofu input that cannot enter state or saved plans. Never put it
  in persisted Terraform variables, state, `.tfvars`, container images,
  committed `.env` files, or Pod environment variables.
  RunPod automatically supplies a distinct Pod-scoped key; never persist, log,
  export, or treat it as account provisioning authority.
- Use the RunPod account SSH public key for interactive Pod access. Never copy
  a private key into the repository, Terraform state, image, or Pod volume.
- Keep the sealed holdout locked during infrastructure and CUDA qualification.
- Preserve immutable run identities, artifact hashes, atomic checkpoints, and
  fail-closed resume checks on remote workers.

## Supported official PyTorch runtime

The production default uses RunPod's official `Runpod Pytorch 2.8.0` template:

```text
template: runpod-torch-v280
image: runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35
registry auth: none
```

[RunPod documents official templates](https://docs.runpod.io/pods/templates/overview)
as RunPod-supported; custom templates are self-supported. Create the source
bundle and deployment receipt only from a clean committed worktree:

```bash
just runpod-official-bootstrap \
  "$HOME/Library/Application Support/mantis-runpod/bootstrap/SOURCE_SHA/source.tar.gz" \
  "$HOME/Library/Application Support/mantis-runpod/bootstrap/SOURCE_SHA/receipt.json"
```

Add a `bootstrap` object to the workload spec with file records for that source
archive, project root `/workspace/mantis/runtime/SOURCE_SHA`, environment root
`/opt/mantis/venv`, and `uv_version = 0.9.0`. Use the
receipt as `image.self_check`. Pre-create staging uploads and downloads the
archive and verifies its size and SHA-256. At container start, the bound command:

1. Starts the official template's `/start.sh` so RunPod owns SSH services.
2. Re-verifies the source archive SHA-256 before extraction.
3. Re-extracts the verified source archive for every launch.
4. Resolves the official image's bundled `uv` 0.9.0 from `PATH` and runs
   `uv sync --frozen --no-dev` against the committed lock. The environment at
   `/opt/mantis/venv` and cache at `/opt/mantis/cache` use fast container disk;
   source inputs, checkpoints, and final artifacts remain on `/workspace`.
5. Executes the same sealed workload and fail-closed CUDA self-check used by the
   legacy image path.

Matrix control files preserve the rendered plan's content-addressed directory
on both the controller and Pod. Stage `matrix-plan.json`, `base-config.toml`, and
the generated cell configs beneath
`/workspace/mantis/control/PLAN_DIGEST/`; flattening them directly into the
control root is rejected before Pod creation.

CUDA utilization is optional telemetry. A missing NVML Python binding records
`cuda_utilization_percent = null` and never interrupts training, checkpointing,
or authoritative metric output.

### 2026-07-23 A100 production evidence

The first supported-template production run used source commit `c7d3ed9`, the
pinned official MantisV2 checkpoint, four timeframes, seed 42, full fine-tuning,
and one A100 SXM 80 GB. It early-stopped after 50 completed epochs and 10,000
steps. Best validation total loss was `2.441334825754166` at epoch 41. The run
produced 10 files, including both checkpoints, provenance, contamination,
metrics, TensorBoard events, telemetry, and `train-result.json`.

Before Pod deletion, all 10 files were copied and SHA-256 verified at both:

```text
~/Library/Application Support/mantis-runpod/artifacts/mantisv2-foundation-accuracy-v1-initial-4tf-full_finetune-s42-cd42b3be4d
/Volumes/Storage/trading-research/artifacts/mantis-foundation-model/mantisv2-foundation-accuracy-v1-initial-4tf-full_finetune-s42-cd42b3be4d
```

The checkpoint identities are:

```text
best.pt    5e6de4cebfbd0fabc84857ed2cbdfb0954c02180466b98a9beb45dd52a364cbb
latest.pt  87bc6f2c93f70fbb82fd56e88b43125017ca47038ac082d56b6a6554f2e495a3
```

Pod `4r4ek5keeyrpe0` was deleted only after both copies matched the remote
manifest, and provider absence was then verified. This ordering is mandatory:
complete -> hash -> copy -> verify both destinations -> delete -> verify
provider absence.

Expose only `22/tcp`. TensorBoard remains bound to `127.0.0.1:6006` and is
reached through an SSH tunnel. The example intent at
`infra/runpod/examples/intent-h100-qualification.json` is the canonical
supported-image identity.

## Legacy custom CUDA image

The self-supported recovery workflow builds Linux amd64 only. The Dockerfile
pins the platform-specific NVIDIA CUDA 13.0.2 cuDNN runtime manifest and the official
`uv` 0.11.30 manifest by SHA-256. It installs Python 3.12 and pinned system
packages, then materializes the committed workspace exclusively with
`uv sync --frozen`. The build refuses a dirty worktree so uncommitted source
cannot enter the declared source identity.

The dependency install uses a BuildKit cache mount for `/root/.cache/uv`; the
cache accelerates rebuilds but is not part of the final image. Do not replace it
with a normal image-layer cache directory, which duplicates several gigabytes
of locked CUDA wheels and materially delays paid Pod startup.

From a clean committed checkout, build and record the two zero-cost contracts:

```bash
just runpod-image-build ghcr.io/lgbarn/mantis-v2-cuda:SOURCE_SHA
just runpod-image-scan ghcr.io/lgbarn/mantis-v2-cuda:SOURCE_SHA reports/image-scan.json
just runpod-image-self-check \
  ghcr.io/lgbarn/mantis-v2-cuda:SOURCE_SHA reports/runtime-inventory.json
```

The scan examines every saved layer plus image history and rejects datasets,
checkpoints, weights, artifacts, private keys, and secret-like assignments. The
local self-check runs without GPU access and writes canonical no-overwrite JSON
containing the base, source tree, source revision, lock, image-contract,
architecture, tool, Torch, and installed-package identities. The Pod executor
then performs the CUDA runtime, driver, and allocation self-check as its first
operation. Missing CUDA, a failed allocation, a non-CUDA-13 runtime, or an
unavailable driver exits nonzero before market data promotion or training.
Rebuilding the same clean commit and lock preserves the declared source, lock,
base, tool, and image-contract identities.

If this recovery route is explicitly selected, use a public registry and
immutable digest reference. If a private pull is unavoidable, configure a
scoped RunPod registry-auth object on the Pod template. Never pass registry
credentials through Pod environment variables.
The image exposes SSH port 22 only. Run TensorBoard inside the Pod on
`127.0.0.1:6006`, then use the existing localhost tunnel contract:

```bash
just tensorboard /network/volume/runs/RUN_ID
ssh -N -L 6006:127.0.0.1:6006 root@POD_HOST -p POD_SSH_PORT
```

At container start, the entrypoint removes any baked `authorized_keys`, installs
only the runtime `PUBLIC_KEY`, validates it with `ssh-keygen`, and uses mode
`0600`. An absent key leaves SSH unavailable without blocking a noninteractive
workload. The public key may enter the Pod environment; private keys never do.

Open `http://127.0.0.1:6006` locally. The command rejects `0.0.0.0`, `::`,
`localhost`, wildcard, and public-interface binds; do not add a public HTTP port.

Do not publish this image or create a Pod until the local scan and static
self-check are clean. A local machine without a Docker daemon can run the
static/unit tests, but it cannot honestly claim image-build or layer-scan
verification. CUDA compatibility is qualified only by the fail-closed runtime
self-check inside the newly created Pod, before input promotion or training.

## Sizing gate

Do not add deployable Terraform resources until these decisions are resolved:

1. Current stage and artifact inventory: issue #21.
2. RunPod hosting, storage, and security topology: issue #22.
3. Data transfer and network-volume capacity: issue #23.
4. Smoke-scale CPU/GPU benchmark matrix: issue #24.
5. User-approved spend and reliability envelope: issue #25.
6. Foundation initialization contract: issue #27.
7. Accuracy-first OHLCV, LoRA, ticker, and timeframe recipe: issue #28.

Issue #26 will synthesize those answers into the implementation contract.

## Initial benchmark envelope

The first CUDA qualification target is deliberately one Pod, not a production
fleet:

- Secure Cloud
- one NVIDIA H100 SXM with 80 GiB VRAM
- at least 8 vCPU and 32 GiB host RAM
- 50 GB container disk
- 150 GB network volume
- 7,200-second smoke workload plus 1,800 seconds of startup allowance and 120
  seconds of shutdown grace, capped at 9,120 wall-clock seconds
- no automatic retry that can create a second Pod

This is a benchmark envelope, not a final production size or authorization to
provision. Re-query price and availability immediately before any plan or
apply. The hard project ceiling recorded in issue #25 is $150.

Budget allocation is fail-closed:

- $15 maximum for the first month of storage and storage overhead.
- $10 cumulative maximum for qualification probes.
- $100 production compute pool after benchmark acceptance.
- $25 protected recovery reserve that requires new user approval.

Keep auto-pay disabled. Permit only one live Mantis Pod, never launch an
automatic paid retry, and reject ordinary launches once cumulative spend
reaches $125.

TensorBoard event files belong under the persistent run directory. Bind the
server to `127.0.0.1:6006` and view it through an SSH tunnel; do not expose a
public TensorBoard HTTP port. JSON manifests and atomic checkpoints remain the
authoritative provenance and resume records. Writer failures are recorded in
`instrumentation/diagnostics.jsonl` and must not invalidate those artifacts.

## Transfer and storage contract

Use one 150 GB Standard network volume and enforce a 120 GB high-water mark
with at least 30,000,000,000 free bytes before a stage starts. At the documented
$0.07/GB/month rate, the volume costs $10.50/month. Select a datacenter that
supports both the chosen Pod and RunPod's S3-compatible network-volume API.

Stage manifest-owned files through the S3 API without a running Pod, then
verify every byte size and SHA-256 inside the attached volume before atomically
promoting the incoming directory. S3 ETags are not provenance. Use rsync over
SSH only as a fallback because it requires paid Pod uptime. Never use a
destructive sync or overwrite an immutable run identity.

The frozen production input bundle is 1,437,579,939 bytes across 75 files. Its
digest is `4d12c3f334dba95fb045905d89654f16d63e75b68f8423ae8726eac06d70fba6`.
It contains three original DBN.ZST archives, the 64-file repaired corpus, the
pinned official MantisV2 cache as regular files, and the frozen diagnostic
fixture. The 24.675 GiB existing continuation bundle is optional
reference state, not an input to the fresh CUDA run. Do not copy the 54.6 GB
historical artifact tree wholesale.

Use this persistent layout:

```text
/workspace/mantis/
|-- inputs/<bundle-sha256>/{dbn,corpus,cache,diagnostic}/
|-- cache/{huggingface,uv}/
|-- runs/<immutable-run-name>/
|-- exports/<immutable-run-name>/
|-- logs/tensorboard/<immutable-run-name>/
`-- transfer/{incoming,outgoing}/<bundle-sha256>/
```

Before cloud upload, make and verify an immutable source snapshot on the Mac's
internal SSD. Download and verify every completed expensive stage to that SSD,
then make a second verified copy on the external drive. The external USB/APFS
drive is excluded from Time Machine and is not a sufficient sole backup.
Deletion remains an explicit operator action and is permitted only after both
copies match the transfer manifest.

### Transfer Bundle commands

Copy `configs/transfer.example.toml` to an ignored machine-local path and
replace every placeholder. The strict config contains paths and immutable
identities only; S3 credentials remain in the local environment and never
enter commands, manifests, receipts, logs, or remote object names.

Build the canonical manifest from an immutable internal-SSD snapshot:

```bash
just transfer-bundle infra/runpod/transfer.local.toml
```

The command rejects unsafe paths, duplicate normalized paths, symlinks,
special files, and source mutation. It publishes the manifest atomically and
refuses overwrite. Exercise the exact staging policy against a synthetic HEAD
inventory before using live credentials:

```bash
just transfer-stage-dry-run \
  infra/runpod/transfer.local.toml \
  infra/runpod/examples/remote-inventory-empty.json
```

This command hashes every source again, uses object size only to plan absent or
size-mismatched PUTs, and performs zero network or object writes. ETag is
accepted as opaque inventory evidence and deliberately ignored. The production
`S3TransferAdapter` seam follows the RunPod
[S3-compatible API](https://docs.runpod.io/storage/s3-api) object-key mapping
and AWS [HeadObject](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html)
size semantics. The production adapter invokes the AWS CLI against the fixed
RunPod endpoint and passes credentials only in its child-process environment.
Set the environment variable names declared by ignored `local.toml`; the
standard names are `RUNPOD_S3_ACCESS_KEY_ID` and
`RUNPOD_S3_SECRET_ACCESS_KEY`. Then pre-stage without creating a Pod:

```bash
just transfer-stage-runpod \
  infra/runpod/transfer.foundation.local.toml \
  infra/runpod/local.toml \
  /path/to/approved-bound-decision.json
```

The command rehashes local inputs, HEAD-checks all objects, uploads only absent
or wrong-size objects, and verifies each resulting size. Never use destructive
sync.

Inside a Pod with the network volume mounted, set the config's mounted paths to
the incoming `files` directory and immutable final parent, then run:

```bash
just transfer-promote infra/runpod/transfer.pod.toml
```

Every mounted byte is checked against SHA-256 before one atomic directory
rename. Missing, corrupt, wrong-size, unexpected, linked, or special content
leaves the final input absent. Repeating the command verifies the existing
final directory and does not replace it.

After an expensive Completed Artifact has been copied to the internal SSD and
separately to the external drive, place a canonical local manifest beside each
copy as configured and verify both:

```bash
just transfer-backup-verify \
  infra/runpod/transfer.local.toml \
  COMPLETED_ARTIFACT_DIGEST
```

The two roots must be distinct and both manifests, bundle digests, file bytes,
and Completed Artifact identity must agree. The command verifies existing
copies; it does not perform a download or copy.

Retention is a decision-only, fail-closed command. First obtain the exact
subject digest from a refusal:

```bash
just transfer-retention-check \
  infra/runpod/transfer.local.toml \
  COMPLETED_ARTIFACT_DIGEST inactive
```

A human may then create an uncommitted authorization JSON with exactly
`schema_version`, `subject_digest`, and `approved_by`, and recheck through
`transfer-retention-check-authorized`. Active run state, absent or mismatched
backups, Completed Artifact drift, or authorization drift deterministically
refuses. The CLI never deletes remote or local data; the isolated deletion
executor exists only behind an explicit verified decision and is covered by a
temporary-directory test.

The following no-I/O dry run validates the older synthetic transfer fixture:

```bash
just transfer-manifest-inspect \
  infra/runpod/examples/measured-input-bundle.fixture.json
```

It reports total size `1395349697` and roots `cache`, `corpus`, and `dbn`.
Fixture entry digests and category sizes are deterministic test oracles, not a
production upload manifest. The production manifest is rebuilt from the
immutable 75-file snapshot described above. Historical continuation
checkpoints and the 54.6 GB artifact tree are not members and must not be added.

### No-idle supervised launch

Direct `runpod-launch` is intentionally rejected. Before a paid create, build,
statically self-check, and scan a clean digest-pinned image, stage the complete input bundle, render
the matrix plan, seal the exact workload, create an approved provider decision,
and bind that decision to the workload. Start the only paid mutation through:

```bash
just runpod-supervise-workload \
  /path/to/sealed/launch-manifest.json \
  /path/to/bound-decision.json \
  infra/runpod/local.toml \
  /path/to/pinned/runpodctl
```

The static image receipt binds architecture, dependencies, source, and lock but
does not claim local CUDA availability. The Pod executor performs the CUDA and
driver allocation self-check as its first operation and starts training only on
success. The supervisor verifies S3 staging before create, requires the first signed
heartbeat within the configured startup allowance (1,800 seconds for the
current cold-image qualification), terminates after four missed 30-second polls or
the hard deadline, captures diagnostics and an emergency checkpoint, verifies
provider absence, reconciles billing, and copies completed artifacts to both
declared backup roots. TensorBoard is reached only through an SSH tunnel.
Image pulling and extraction are part of the startup allowance. A
provider state such as `RUNNING` does not prove workload execution: require a
nonnegative runtime plus the signed heartbeat. If either is absent at the limit,
terminate the Pod and preserve the network volume rather than extending paid
idle time ad hoc.

## Accuracy-first training contract

The RunPod candidate is a fresh supervised futures-adaptation run from the
pinned official MantisV2 checkpoint, not a resume from the existing local
fine-tune and not a randomly initialized encoder. It is not random-weight
base-model pretraining or self-supervised domain-adaptive pretraining. The
verified corpus covers nine instruments
(`ES`, `NQ`, `RTY`, `YM`, `GC`, `SI`, `CL`, `ZB`, `ZN`) at four timeframes
(`1min`, `3min`, `5min`, `15min`). The 3-minute slice is the primary trading
horizon and evaluation slice.

Full-upstream fine-tuning plus new NextLeg heads is the primary accuracy arm.
The current `transformer_finetune` behavior freezes the pretrained tokenizer
and projector and is a separate control, not the full arm. Test LoRA ranks 8
and 16 against MantisV2's `wQKV` and `wO` projections only as non-inferiority
arms; the older FFM Mantis-8M module names are not compatible. Retain base and
adapter hashes separately and merge only after native-versus-merged parity
passes.

Every first launch uses a unique absent run directory, loads and verifies the
pinned official revision, deterministically initializes the new task heads,
and records both identities. A later resume may load only an atomic checkpoint
from that same provenance-bound run. Random `scratch` mode is outside this
route and cannot be substituted when official weights are unavailable.

First compare the current three-timeframe recipe with a compute-matched
four-timeframe recipe. If 5-minute data is promoted, preserve per-stream epoch
exposure with 267 training batches and 27 validation batches at batch size 128,
subject to the measured H100 memory and throughput gate. Screen seeds 42-44 and
confirm with seeds 42-46. Promotion requires improvement on at least four of
five confirmation seeds, improved median macro 3-minute performance, no
material instrument-family regression, and improved downstream log loss and
Brier score. Keep the sealed 2026 holdout unavailable to selection.

Checkpoint selection uses validation total, candle, and leg loss. TensorBoard
also records learning rate, gradient norm, examples per second, data wait, GPU
memory, host RSS, and checkpoint state. JSON provenance remains authoritative.

## Compute qualification contract

No RunPod shape is qualified yet. The provisional target is one Secure Cloud
NVIDIA H100 SXM with 80 GB VRAM, at least 8 vCPU, and at least 32 GB host RAM.
A read-only query on 2026-07-22 returned $2.99/hour with Low availability in
S3-compatible `US-MO-1`; always re-query immediately before launch.

The repository now has a pinned CUDA image, real-data FP32 probe contract,
TensorBoard event writing, fixed cross-device and interrupted-resume oracles,
resource evidence schemas, a fail-closed BF16 candidate and evidence gate,
explicit CUDA downstream embedding with atomic resume and a CPU-testable parity
gate, and transfer verification. Real-CUDA embedding and BF16 evidence plus the
independent shutdown watchdog remain separate gates. The first paid
benchmark is still a post-implementation human acceptance gate, not evidence
that CUDA is already qualified.

Run the fail-closed FP32 probe inside the pinned image with a unique absent run
identity and machine-local real-data paths:

```bash
just probe-cuda-fp32 \
  mantis-v2/configs/nextleg-parquet-v2-probe.toml \
  mantis-v2/configs/cuda-fp32-qualification.toml \
  cuda-fp32-probe-YYYYMMDDTHHMMSSZ-1234abcd \
  /workspace/artifacts
```

This command performs exactly 32 optimizer updates and one pre-holdout
validation batch, then evaluates and verifies the selected safetensors export.
It cannot resume, overwrite, fall back to scratch weights, or unlock holdout
data. CPU-only execution emits a skip record and does not qualify the path.

BF16 keeps FP32 parameters and optimizer state and applies autocast only to the
forward/loss region. It requires explicit supported CUDA and exact precision
identity across every artifact. After capturing the registered FP32 reference
and BF16 candidate bundles, run:

```bash
just qualify-cuda-bf16 fp32.json bf16.json qualification.json
```

If the single paid candidate attempt fails before producing a bundle, use
`just reject-cuda-bf16 fp32.json '<failure>' qualification.json`. Both paths are
no-overwrite. Any unsupported operation, OOM, overflow, non-finite evidence,
parity mismatch, resume mismatch, export mismatch, or precision-record mismatch
selects FP32 with no retry and no tolerance change. CPU fake evidence tests do
not promote BF16; promotion requires reviewed real-CUDA evidence.

Run the H100 SXM alone first with a 7,200-second workload, 1,800-second startup
allowance, and 120-second shutdown grace. The wall-clock cap is 9,120 seconds.
At the observed $2.99/hour Secure Cloud price, its maximum projected
qualification spend is $7.60. L40S and A40 are explicit sequential fallbacks
only, each requiring a fresh inventory, price, and launch decision after a
failed allocation. Total qualification compute and container-disk overhead must
remain below the $10 allocation. Never launch a fallback automatically.

The matrix covers environment and network-volume I/O; FP32 batch doubling;
CPU/CUDA output, loss, gradient, and update parity; BF16 qualification;
uninterrupted-versus-resumed equivalence; full-upstream versus transformer-only
resource use; CPU/CUDA embedding parity and throughput; CPU RL validation; and
TensorBoard/JSON/checkpoint verification. Keep GPU and host peak memory below
80%, require the existing RL 5,000 steps/second and 100x mmap-latency gates,
and reject any production projection that exceeds the $100 production pool.

For downstream embedding, use only an explicit CUDA config and an ordered
four-timeframe identity. The bounded run must record rows/second, data wait,
peak VRAM and RSS, disk bytes/row, checkpoint-free restart, and the projected
four-timeframe footprint. Apply `just qualify-cuda-embedding ...` only after the
authorized run has produced immutable CPU/CUDA fixtures, exact metadata, shard
receipts, and performance JSON. The command itself is a local evidence check and
does not create a Pod or authorize spend.
