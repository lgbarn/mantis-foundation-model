# ADR 0001: Pin and isolate MantisV2 upstream

## Status

Accepted.

## Context

MantisV2 semantics, executable code, and released weights live in separate upstream artifacts. Floating branches and Hub revisions can change independently. Upstream checkpoint helpers save weights only, which is insufficient for reproducible resume. License declarations also conflict between Apache-2.0 and MIT.

## Decision

- Pin semantics to arXiv 2602.17868.
- Pin code to `vfeofanov/mantis` tag `v1.0.0`, commit `0c94f8ceb9f1d1421dd292ed917090df8c31605b`.
- Pin weights to `paris-noah/MantisV2` revision `99fe0f548960e272fbfa4b82fd9b5b5956779dfd` and verify SHA-256 `49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1`.
- Hide construction, loading, channel handling, and embedding-shape checks behind `MantisV2Adapter`.
- Own a native checkpoint envelope containing optimizer, epoch, step, RNG, configuration, dataset, source-content digest, Git state, lock, and upstream identities.
- Do not redistribute upstream source or weights or state one unified upstream license until the conflict is clarified.

## Consequences

Training fails if upstream weights drift or provenance does not match. Export remains separate from resumable training checkpoints. A future upstream upgrade requires a new ADR and explicit parity testing.
