# ADR 0007: Use content-addressed transfer with independent verified backups

## Status

Accepted.

## Context

The production corpus lives on an external drive that is excluded from Time
Machine, RunPod network volumes are persistent but are not backups, and S3
multipart ETags do not prove file content. Direct directory synchronization
could silently reuse partial or stale artifacts and destructive synchronization
could erase the only good copy.

## Decision

Every cloud input and output moves as a Transfer Bundle with relative paths,
byte sizes, and SHA-256 values. Upload into an incoming directory, verify the
mounted bytes inside a Pod, and atomically promote only a complete bundle.
Download each expensive Completed Artifact to the internal SSD, verify it, then
make and verify a second copy on the external drive. Never use destructive sync
or delete remote state before both copies pass and a human approves retention.

## Consequences

Transfers may repeat work after corruption or interruption, but artifact
eligibility never depends on upload success, ETags, filenames, or an unverified
single copy. RunPod storage can be discarded only after independent recovery
evidence exists.
