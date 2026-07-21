# RunPod infrastructure

This directory will contain the reproducible RunPod infrastructure for the
MantisV2 training pipeline. The decision map is GitHub issue #20.

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
- Read the API credential only from `RUNPOD_API_KEY`; never put it in Terraform
  variables, state, `.tfvars`, container images, committed `.env` files, or Pod
  environment variables.
- Use the RunPod account SSH public key for interactive Pod access. Never copy
  a private key into the repository, Terraform state, image, or Pod volume.
- Keep the sealed holdout locked during infrastructure and CUDA qualification.
- Preserve immutable run identities, artifact hashes, atomic checkpoints, and
  fail-closed resume checks on remote workers.

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
- 50 GiB container disk
- 150 GiB network volume
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

## Accuracy-first training contract

The RunPod candidate is a fresh domain-adaptation run from the pinned official
MantisV2 checkpoint, not a resume from the existing local fine-tune and not a
randomly initialized encoder. The verified corpus covers nine instruments
(`ES`, `NQ`, `RTY`, `YM`, `GC`, `SI`, `CL`, `ZB`, `ZN`) at four timeframes
(`1min`, `3min`, `5min`, `15min`). The 3-minute slice is the primary trading
horizon and evaluation slice.

Full-model fine-tuning is the primary accuracy arm. Test LoRA ranks 8 and 16
against MantisV2's `wQKV` and `wO` projections only as non-inferiority arms;
the older FFM Mantis-8M module names are not compatible. Keep the local adapter
and task heads trainable, retain base and adapter hashes separately, and merge
only after native-versus-merged parity passes.

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
