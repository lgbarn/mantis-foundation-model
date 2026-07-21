# RunPod infrastructure

This directory will contain the reproducible RunPod infrastructure for the
MantisV2 training pipeline. The decision map is GitHub issue #20. The accepted
implementation route is [IMPLEMENTATION_HANDOFF.md](IMPLEMENTATION_HANDOFF.md).

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
  `RUNPOD_API_KEY`; never put it in Terraform variables, state, `.tfvars`,
  container images, committed `.env` files, or Pod environment variables.
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
ssh -L 6006:127.0.0.1:6006 root@POD_HOST -p POD_SSH_PORT
```

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
authoritative provenance and resume records.

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

Do not spend on a benchmark until the repository has a pinned CUDA image, a
real-data CUDA probe, TensorBoard event writing, precision configuration,
cross-device and interrupted-resume parity fixtures, CUDA embedding support,
resource instrumentation, transfer verification, and an independent shutdown
watchdog. The benchmark is the first post-implementation acceptance gate, not
a planning-time claim that CUDA is already qualified.

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
