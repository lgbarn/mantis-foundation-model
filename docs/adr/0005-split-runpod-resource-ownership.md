# ADR 0005: Split RunPod resource ownership and fail closed on paid execution

## Status

Accepted.

## Context

The RunPod Terraform provider can own stable resources, but it does not expose
or verify every GPU, CPU/RAM, startup, registry, and deadline field required by
the MantisV2 qualification contract. Using Terraform alone would hide missing
lifecycle controls; using ad hoc scripts or Ansible for everything would weaken
state ownership, dry-run review, and duplicate-resource prevention.

## Decision

- Terraform owns the non-secret network volume and private Pod template.
- A pinned lifecycle adapter consumes those identities and owns exact Pod
  launch, status, stop, and termination until provider parity is verified.
- A deterministic action plan separates read-only reconciliation from resource
  mutation. The executor requires a Paid Gate for the exact plan.
- Only one live Mantis Pod is permitted. There is no automatic paid retry or
  fallback.
- The local control plane owns secrets, approvals, the spend ledger, and an
  independent termination deadline. Pods never receive cloud control-plane
  credentials.
- Ansible is excluded until a measured long-lived mutable-host requirement
  exists.

## Consequences

Implementation must prove that Terraform and the lifecycle adapter cannot both
create the same Pod, and must provide import/adoption checks for existing
resources. Provider parity can later retire the adapter, but only after the
same fields, dry-run evidence, one-Pod lock, spend controls, and termination
behavior pass contract tests.
