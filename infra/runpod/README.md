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

TensorBoard event files belong under the persistent run directory. Bind the
server to `127.0.0.1:6006` and view it through an SSH tunnel; do not expose a
public TensorBoard HTTP port. JSON manifests and atomic checkpoints remain the
authoritative provenance and resume records.
