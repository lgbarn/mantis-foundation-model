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
  to `infra/runpod/local.toml` and replace the paths; that filename is ignored.
- Launch Intent, Inventory Snapshot, Spend Ledger, and optional Launch
  Authorization are explicit JSON inputs. The files under `examples/` are
  deterministic synthetic fixtures, not current inventory or authorization.

The rejection dry run below is deliberately historical and synthetic. It is
safe to execute because the Inventory Snapshot is a file, not a live query:

```bash
just runpod-plan \
  infra/runpod/configs/platform-v1.toml \
  infra/runpod/configs/local.example.toml \
  infra/runpod/configs/experiment-cuda-qualification.example.toml \
  infra/runpod/examples/intent-a40-qualification.json \
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

Authorization is exact, expiring, and single-use. A changed config, local
controller mapping, intent, price/inventory observation, ledger/reservation
state, duration, spend ceiling, expiry, or consumed authorization digest rejects
the decision. Issue #31 will own consumption, reservation mutation, provider
inventory collection, and Pod lifecycle; this planner owns none of those side
effects.

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
- A small pinned `runpodctl` adapter owns exact Pod launch and termination until
  the official Terraform provider forwards and verifies the required GPU,
  CPU/RAM, registry, startup-command, and deadline fields.
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

## Reproducible CUDA image

The public image workflow builds Linux amd64 only. The Dockerfile pins the
platform-specific NVIDIA CUDA 13.0.2 cuDNN runtime manifest and the official
`uv` 0.11.30 manifest by SHA-256. It installs Python 3.12 and pinned system
packages, then materializes the committed workspace exclusively with
`uv sync --frozen`. The build refuses a dirty worktree so uncommitted source
cannot enter the declared source identity.

From a clean committed checkout, build and record the two zero-cost contracts:

```bash
just runpod-image-build ghcr.io/lgbarn/mantis-v2-cuda:SOURCE_SHA
just runpod-image-scan ghcr.io/lgbarn/mantis-v2-cuda:SOURCE_SHA reports/image-scan.json
just runpod-image-self-check \
  ghcr.io/lgbarn/mantis-v2-cuda:SOURCE_SHA reports/runtime-inventory.json
```

The scan examines every saved layer plus image history and rejects datasets,
checkpoints, weights, artifacts, private keys, and secret-like assignments. The
self-check runs with GPU access and writes canonical no-overwrite JSON containing
the base, source tree, source revision, lock, and image-contract identities;
Python, `uv`, Git, SSH, Torch, CUDA, driver, architecture, compatibility, and the
complete installed Python package inventory. Missing CUDA, a failed allocation,
a non-CUDA-13 runtime, or an unavailable driver exits nonzero before any market
data or run directory is touched. Rebuilding the same clean commit and lock
preserves the declared source, lock, base, tool, and image-contract identities.

Use a public registry and immutable digest reference by default. If a private
pull is unavoidable, configure a scoped RunPod registry-auth object on the Pod
template. Never pass registry credentials through Pod environment variables.
The image exposes SSH port 22 only. Run TensorBoard inside the Pod on
`127.0.0.1:6006`, then use the existing localhost tunnel contract:

```bash
just tensorboard /network/volume/runs/RUN_ID
ssh -N -L 6006:127.0.0.1:6006 root@POD_HOST -p POD_SSH_PORT
```

Open `http://127.0.0.1:6006` locally. The command rejects `0.0.0.0`, `::`,
`localhost`, wildcard, and public-interface binds; do not add a public HTTP port.

Do not publish this image or create a Pod until the local scan is clean and a
GPU-host self-check records `driver.compatible=true`. A local machine without a
Docker daemon can run the static/unit tests, but it cannot honestly claim build,
layer-scan, or CUDA self-check verification.

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
- one NVIDIA A40 with 48 GiB VRAM
- at least 8 vCPU and 32 GiB host RAM
- 50 GB container disk
- 150 GB network volume
- two-hour automatic termination for the smoke benchmark
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

The clean disaster-recovery input bundle is 1,395,349,697 bytes: three original
DBN.ZST archives, the 64-file repaired corpus, and the pinned official
MantisV2 cache. The 24.675 GiB existing continuation bundle is optional
reference state, not an input to the fresh CUDA run. Do not copy the 54.6 GB
historical artifact tree wholesale.

Use this persistent layout:

```text
/workspace/mantis/
|-- inputs/{raw,corpus,upstream}/<content-identity>/
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
refuses overwrite. Before enabling a concrete S3 client, exercise the exact
injected staging policy against a synthetic HEAD inventory:

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
size semantics. This repository intentionally installs no SDK and ships no
credential-bearing adapter; inject a reviewed local adapter rather than putting
keys in config or command arguments. Never use destructive sync.

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

The following no-I/O dry run validates the synthetic manifest fixture whose
total is the measured 1,395,349,697-byte clean input bundle:

```bash
just transfer-manifest-inspect \
  infra/runpod/examples/measured-input-bundle.fixture.json
```

It reports total size `1395349697` and roots `cache`, `corpus`, and `dbn`.
Fixture entry digests and category sizes are deterministic test oracles, not a
production upload manifest. The production manifest must be rebuilt from the
immutable snapshot containing the three original DBN.ZST archives, 64 repaired
corpus files, and pinned MantisV2 cache. Historical continuation checkpoints
and the 54.6 GB artifact tree are not members and must not be added.

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
subject to the measured A40 memory and throughput gate. Screen seeds 42-44 and
confirm with seeds 42-46. Promotion requires improvement on at least four of
five confirmation seeds, improved median macro 3-minute performance, no
material instrument-family regression, and improved downstream log loss and
Brier score. Keep the sealed 2026 holdout unavailable to selection.

Checkpoint selection uses validation total, candle, and leg loss. TensorBoard
also records learning rate, gradient norm, examples per second, data wait, GPU
memory, host RSS, and checkpoint state. JSON provenance remains authoritative.

## Compute qualification contract

No RunPod shape is qualified yet. The provisional target is one Secure Cloud
NVIDIA A40 with 48 GB VRAM, at least 8 vCPU, and at least 32 GB host RAM. A
read-only query on 2026-07-21 returned $0.44/hour with Low availability; always
re-query immediately before launch.

The repository now has a pinned CUDA image, real-data FP32 probe contract,
TensorBoard event writing, fixed cross-device and interrupted-resume oracles,
resource evidence schemas, and transfer verification. CUDA embedding, BF16,
and the independent shutdown watchdog remain separate gates. The first paid
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

Run the A40 alone first with a two-hour hard deadline. A longer qualification
requires a new explicit approval after reviewing that result. The cumulative
A40 qualification allowance is at most eight hours and $3.52 at the observed
price; this is not permission for one eight-hour Pod. L40, L40S, and A100 PCIe
are explicit sequential fallbacks only. The worst-case approved matrix is $8.53
compute and must remain below the $10 qualification allocation after
container-disk overhead. Never launch a fallback automatically.

The matrix covers environment and network-volume I/O; FP32 batch doubling;
CPU/CUDA output, loss, gradient, and update parity; BF16 qualification;
uninterrupted-versus-resumed equivalence; full-upstream versus transformer-only
resource use; CPU/CUDA embedding parity and throughput; CPU RL validation; and
TensorBoard/JSON/checkpoint verification. Keep GPU and host peak memory below
80%, require the existing RL 5,000 steps/second and 100x mmap-latency gates,
and reject any production projection that exceeds the $100 production pool.
