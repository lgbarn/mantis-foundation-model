# RunPod V2 Simplification Specification

Status: Accepted

## Problem Statement

The current RunPod execution path implements a custom control plane around a
simple foundation-model training job. Provider REST transport, CLI launch,
inventory snapshots, authorization ledgers, lifecycle reconciliation, S3
staging, Terraform, image construction, monitoring, and artifact replication
are spread across thousands of lines and many operator commands. This makes a
supported build-and-train operation slow to change, hard to understand, and
more likely to delay paid training while a Pod is already provisioned.

RunPod now provides a beta REST API v2 and an official v2 `runpodctl` CLI with
machine-readable Pod lifecycle commands. The repository should consume those
supported interfaces directly and own only the Mantis-specific correctness and
cost controls that RunPod cannot provide.

## Solution

Replace the normal RunPod path with three operator workflows: train, recover,
and TensorBoard. The train workflow performs one API v2 catalog preflight for
GPU price and availability, uses pinned `runpodctl` JSON commands for Pod
lifecycle and SSH discovery, uses OpenSSH for remote execution and tunneling,
and uses rsync for source, checkpoint, data, and artifact synchronization.

Use an official RunPod PyTorch image and install the frozen repository
environment on the Pod. Preload the immutable training corpus onto a reusable
network volume once; subsequent launches attach and validate that volume.
Retain a small durable run receipt, provider-side termination deadline, local
cleanup trap, exact-Pod-ID deletion, checkpoint provenance validation, artifact
hash verification, and final billing query.

This specification supersedes the infrastructure ownership, custom image,
Terraform, S3 transfer, and custom Pod-control decisions in the accepted 2026-07-21
RunPod execution specification. It does not change any foundation-model,
timeframe, label, Trend Magic, Topstep, RL, holdout, checkpoint, or export
semantics from that specification.

## User Stories

1. As an operator, I want one training command, so that provisioning a GPU and
   starting training is a single documented action.
2. As an operator, I want provider operations delegated to official tools, so
   that repository code does not duplicate RunPod's control plane.
3. As a budget owner, I want price and availability checked before creation, so
   that an unsuitable GPU is not intentionally provisioned.
4. As a budget owner, I want every Pod to have a provider-side termination
   deadline, so that loss of the local controller cannot create unlimited spend.
5. As an operator, I want the created Pod ID recorded immediately, so that all
   monitoring, recovery, billing, and deletion target the exact resource.
6. As an operator, I want cleanup to run after success, failure, or interruption,
   so that ordinary workflow exits do not leave paid compute running.
7. As an operator, I want a recovery command that never creates a Pod, so that
   an ambiguous launch cannot accidentally double spend.
8. As a researcher, I want the official PyTorch image and frozen dependencies,
   so that custom image work does not delay training while the Python runtime
   remains reproducible.
9. As a data owner, I want the immutable corpus reused from a network volume, so
   that each training launch does not repeat a multi-gigabyte upload.
10. As a data owner, I want source and artifact transfers to use rsync plus
    content verification, so that large transfers are resumable and corruption
    is detected.
11. As a researcher, I want TensorBoard reachable only through localhost SSH,
    so that progress is visible without exposing a public dashboard.
12. As an auditor, I want Mantis manifests, JSON metrics, and checkpoint hashes
    to remain authoritative, so that simplifying infrastructure does not weaken
    scientific provenance.
13. As a maintainer, I want no Python RunPod SDK or provider adapter hierarchy,
    so that lifecycle schema changes remain owned by the official CLI.
14. As a maintainer, I want obsolete v1, Terraform, S3, and duplicate transfer
    surfaces removed, so that there is only one supported execution path.
15. As an operator, I want concise documentation and configuration, so that GPU,
    image, volume, paths, duration, and price limits change without source edits.

## Implementation Decisions

- Pin one official `runpodctl` v2 release and record its version in each run
  receipt. Consume JSON output and reject missing or malformed required fields.
- Use the beta API v2 GPU catalog only for a read-only price and availability
  preflight. Pin and record the API v2 OpenAPI identity without generating or
  maintaining a repository-owned API client.
- Use `runpodctl` for Pod create, get, list, delete, billing, account, datacenter,
  and SSH discovery. Use noun-verb commands only; deprecated commands are not
  accepted.
- Require a provider-side terminate-after duration on every create. Retain a
  local cleanup trap and exact-ID absence verification as independent controls.
- Use one small orchestration script behind the three `just` workflows. Shell
  owns sequencing and cleanup; existing Python commands own Mantis validation,
  training, checkpoints, evaluation, and artifact acceptance.
- Use OpenSSH for remote commands and localhost-only TensorBoard tunneling. Do
  not introduce a custom SSH implementation.
- Use rsync for resumable source, data, checkpoint, and artifact transfer. A
  reusable network volume holds the immutable corpus and caches; local artifact
  destinations remain authoritative after hash verification.
- Use an official RunPod PyTorch image and run frozen dependency synchronization
  before training. A custom image is out of scope until measured setup cost
  proves it economically necessary.
- Keep one typed, ignored machine-local runtime configuration for GPU SKU,
  maximum hourly price, image, network-volume ID, remote paths, local artifact
  paths, SSH key, and termination duration. Scientific experiment definitions
  remain in committed experiment configuration.
- Write the run receipt before remote training. Recovery accepts only a receipt,
  never an experiment configuration, and is structurally unable to create.
- Validate the returned Pod cost and identity immediately after creation; delete
  and fail if they do not match the approved runtime configuration.
- Preserve existing checkpoint identity, sealed-holdout exclusion, artifact
  provenance, and export-parity gates without reimplementing them.
- Retire the custom REST v1 adapter, hybrid CLI/REST adapter, provider lifecycle
  state machine, ephemeral Terraform workflow, required S3 workflow, custom
  image workflow, duplicate artifact replication, and obsolete operator commands
  only after the new smoke and recovery contracts pass.

## Acceptance Criteria

1. One documented training command validates inputs, checks API v2 price and
   availability, creates one Pod through pinned `runpodctl`, starts the existing
   training recipe, retrieves and validates artifacts, deletes the exact Pod,
   verifies absence, and reports billing.
2. The normal workflow contains no direct REST v1 call and requires no Python
   RunPod SDK, Terraform/OpenTofu, AWS CLI, custom image build, or S3 credential.
3. A strict runtime configuration rejects unknown keys, secrets, missing Pod
   controls, non-positive durations, and prices above the configured ceiling.
4. API v2 catalog responses and `runpodctl` JSON are contract-tested for valid,
   missing, malformed, unavailable, over-price, and changed-schema cases.
5. Every successful create persists an atomic receipt containing exact Pod ID,
   run identity, request digest, CLI version, API schema identity, image, GPU,
   volume, price, creation time, and termination deadline before remote work.
6. Every create includes provider-side terminate-after. Success, training
   failure, transfer failure, validation failure, signal interruption, and
   controller exception all attempt exact-ID deletion and verify absence.
7. Recovery consumes an existing receipt, never invokes create, can download a
   valid checkpoint or artifact set, deletes the exact Pod, verifies absence,
   and queries final billing.
8. An ambiguous create result never triggers an automatic second create. The
   workflow stops with durable evidence and a recovery command.
9. The source and immutable corpus transfer path is resumable and verifies the
   existing content-addressed manifest before the data becomes eligible for
   training.
10. A normal launch reuses a verified network-volume corpus and does not upload
    the complete five-year dataset again.
11. The official PyTorch image passes CUDA, Python, `uv`, dependency-lock, disk,
    volume, and Mantis input preflight checks before the training process starts.
12. TensorBoard binds to `127.0.0.1`, the workflow prints a complete OpenSSH
    tunnel command, and no public TensorBoard endpoint is required.
13. Existing foundation training, checkpoint resume, evaluation, artifact
    validation, holdout exclusion, and export-parity tests remain green.
14. A zero-cost fake-CLI integration test demonstrates the entire orchestration
    including successful cleanup; failure tests demonstrate cleanup and recovery
    without paid provider actions.
15. One authorized smoke Pod demonstrates create, SSH readiness, frozen setup,
    short training, artifact retrieval, TensorBoard tunnel, exact deletion,
    absence verification, and billing within its configured deadline.
16. After smoke acceptance, obsolete REST v1, hybrid adapter, lifecycle command,
    Terraform launch, S3 workload, custom image, and duplicate transfer code and
    documentation are removed, leaving only the three supported operator recipes.

## Owned Surfaces

- RunPod runtime configuration
- RunPod train workflow
- RunPod recovery workflow
- RunPod TensorBoard workflow
- Atomic RunPod run receipt
- API v2 catalog preflight contract
- Pinned `runpodctl` JSON contract
- SSH and rsync execution/transfer contract
- Network-volume corpus verification contract
- RunPod simplification operating guide

## Testing Decisions

- Test orchestration against fake `runpodctl`, curl, SSH, and rsync executables
  whose call logs are independent behavioral oracles.
- Test the money and lifecycle path first: price rejection, exact identity,
  atomic receipt, termination deadline, cleanup on every exit class, and
  recovery's inability to create.
- Test JSON boundaries with recorded official API v2 and CLI fixtures while
  rejecting unknown or missing required fields rather than mirroring all fields.
- Test transfers using local directories, interrupted rsync fixtures, and the
  existing content-addressed manifest verifier.
- Reuse existing Mantis checkpoint, provenance, holdout, evaluation, and export
  tests as regression oracles.
- Keep paid smoke outside CI. It requires explicit runtime authorization and
  uses the same command as production with a smoke experiment configuration.

## Blocked by

None - implementation and zero-cost tests can start immediately. The paid smoke
requires an explicit smoke runtime configuration and uses the user's existing
RunPod account authorization.

## Out of Scope

- A repository-owned general RunPod API client or Python RunPod SDK dependency
- Mirroring the complete beta API v2 schema
- Serverless, Flash, Kubernetes, Ansible, or multi-cloud abstraction
- Terraform/OpenTofu for ephemeral training Pods
- S3 or cloud-sync as a required normal-path transfer mechanism
- A custom Docker image or registry requirement for the normal path
- Public TensorBoard or hosted experiment tracking
- Automatic retry or fallback create after an ambiguous provider response
- Changes to model architecture, four-timeframe training, labels, Trend Magic,
  Topstep rules, RL, Monte Carlo, Optuna, checkpoint, holdout, or export behavior
- Deletion of backward-readable v1 receipts before every v1-created Pod and
  billing record is reconciled

## Research Basis

- **Source:** RunPod v1 responsibility audit in issue #63 and RunPod API
  v2/official CLI capability research in issue #64, 2026-07-23. **Verdict
  adopted:** maximize deletion by delegating provider lifecycle to official
  tools. **Scope chosen:** complete replacement of the normal v1 control plane,
  retaining only Mantis correctness and independent cost/cleanup controls.

| Decision | Backing finding | Confidence |
| --- | --- | --- |
| Use `runpodctl` directly | Official v2 CLI covers Pod lifecycle, billing, account, datacenter, and SSH discovery with JSON output | High |
| Use API v2 only for catalog | API v2 catalog supplies price and opt-in availability missing from documented CLI GPU output | High |
| Use SSH and rsync | Official transfer guidance recommends rsync for large and recurring transfers; CLI send/receive is code-coordinated | High |
| Use official image | `runpodctl` does not build ordinary Pod images; custom build is unnecessary for frozen setup | High |
| Remove custom control plane | Audit found provider mechanics and duplicate orchestration dominate the current implementation | High |
| Retain exact cleanup and provenance | Provider tools do not own experiment identity, checkpoint eligibility, artifact acceptance, or controller-failure cleanup | High |

The audit's semantic safety triggers become criteria 3-9 and 13-14. The CLI and
API capability triggers become criteria 1-2 and 10-12. The deletion trigger
becomes criterion 16. API v2 beta schema drift, CLI create idempotency, and paid
smoke cost remain inherited risks and are bounded by pinned contracts, no
automatic second create, and explicit smoke authorization.

## Further Notes

The API v2 base URL is `https://api.runpod.io/v2`; it is distinct from the
Serverless job API. The accepted 2026-07-21 RunPod specification remains the
source of truth for Mantis scientific behavior but is superseded by this
specification for RunPod infrastructure and operator workflow ownership.
