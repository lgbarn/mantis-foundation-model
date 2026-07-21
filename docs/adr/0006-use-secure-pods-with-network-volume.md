# ADR 0006: Use Secure Cloud Pods with a persistent network volume

## Status

Accepted.

## Context

The workflow needs reproducible GPU and CPU-host qualification, SSH-only
TensorBoard, resumable checkpoints, and storage that survives compute
termination. Serverless obscures the required host lifecycle and Community
Cloud does not meet the accepted reliability and security envelope. Local Pod
volume storage disappears when the Pod is terminated.

## Decision

Use one on-demand Secure Cloud Pod at a time with one independently persistent
Standard network volume. Attach the volume when the Pod is created, terminate
the Pod at each deadline, and recreate a Pod against the same verified volume
only after a new Launch Authorization. Never expose TensorBoard publicly or
automatically launch a fallback.

## Consequences

GPU and CPU-host availability must be re-queried before each launch. A stopped
compute lease is not part of the recovery model: recovery is terminate,
preserve the network volume, reconcile cost, and create a newly authorized Pod.
