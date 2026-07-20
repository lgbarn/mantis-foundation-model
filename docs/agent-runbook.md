# AI-agent runbook

This runbook tells Claude, Codex, and other AI agents how to operate this
repository from discovery through completion without weakening safety contracts
or damaging active runs.

It supplements [AGENTS.md](../AGENTS.md). `AGENTS.md` governs development. This
document governs repository operation.

## Operating principles

An agent must:

1. Treat the selected TOML as the experiment definition.
2. Treat source, lock, data, upstream pins, and artifact identities as immutable
   inputs to a resumable run.
3. Prefer read-only discovery before any command that downloads, writes, scans
   large data, or allocates an accelerator.
4. Obtain approval before installing packages or enabling paid, metered,
   account-requiring, or token-requiring services.
5. Never expose the holdout during routine development.
6. Report exact commands and distinguish observed results from inference.
7. Never claim trading value from forecast or classifier metrics alone.

## Read in this order

Before operating a workflow, read:

1. `AGENTS.md`
2. `README.md`
3. `docs/architecture.md`
4. `docs/setup-and-hardware.md`
5. `docs/workflow.md`
6. The selected config
7. `mantis-v2/README.md`
8. Tests for the stage being operated

Treat `.scratch` as historical or investigative material, not operating truth.
Treat a research report as evidence, not an implemented feature.

## Establish authority before changing state

The following actions require explicit user authority when not already contained
in the request:

- Install or update dependencies.
- Download official weights when network access or account use is uncertain.
- Start a long training or embedding run.
- Start a run that can consume meaningful storage.
- Change production config, data, or artifact locations.
- Overwrite any run directory.
- Open the sealed holdout.
- Publish, upload, or share weights or results.
- Enable an external service or anything that may cost money.

Read-only status, file inspection, JSON/manifest inspection, Git inspection, and
non-mutating process inspection are normally safe when relevant.

## Detect repository state

Begin with read-only commands:

```bash
git status --short
git rev-parse HEAD
just --list
ps -Ao pid,ppid,etime,%cpu,%mem,command
```

Narrow process output to known commands. Do not signal or kill anything.

Classify the repository into one state:

| State | Evidence | Allowed next action |
| --- | --- | --- |
| Fresh clone | No managed environment or artifacts | Request setup authority |
| Environment ready | Locked environment exists | Run cheap checks when authorized |
| Foundation run active | Train process and changing metrics/checkpoints | Read-only monitoring only |
| Foundation run interrupted | No process, resumable checkpoint exists | Verify provenance, then resume with authority |
| Foundation complete | Train result, best checkpoint, evaluation/export as applicable | Validate artifacts; do not retrain |
| Downstream stage active | Stage process and changing output | Read-only monitoring only |
| Downstream partial | Completed earlier manifest plus absent later manifest | Diagnose and resume at next safe stage |
| Downstream complete | All stage manifests and results exist | Report; preserve holdout state |
| Holdout sealed | Normal config and no one-time artifact | Do not unlock |
| Holdout consumed | Reviewed holdout artifact exists | Never retune from it |

Do not infer state from filenames alone. Confirm manifests, process state, and
run identity.

## Protect active and resumable runs

During an active or resumable run, never:

- Kill, signal, suspend, renice, or restart the process.
- Edit source under `mantis-v2/src`.
- Edit `uv.lock`.
- Edit the active TOML.
- Edit, move, truncate, or replace source data.
- Edit any file in the active artifact directory.
- Run a second command with the same run name.
- Run broad tests or smoke workflows that compete for the same accelerator.
- Delete a pending checkpoint, heartbeat, partial shard, or manifest.
- Load a changing native checkpoint merely to report status.

Use stable JSON such as `metrics.json` and stage manifests for status. Read files
atomically when possible. If a status file changes during inspection, reread it
instead of stopping the writer.

## Choose the next foundation action

Use this state machine:

```text
environment absent
  -> request sync authority
  -> just sync

environment ready, contracts unverified
  -> just gate

contracts green, upstream unverified
  -> just verify-upstream

upstream verified, corpus unverified
  -> just inspect-data <config>

corpus verified, committed-host MPS recipe unverified
  -> just probe-mps

corpus verified, portable-host MPS recipe unverified
  -> copy/review probe config
  -> uv run mantis-v2 probe --config <copied-probe-config>

preflight complete, no run state
  -> just train <config>

valid resumable state, exact provenance unchanged
  -> just train <same-config>

training complete, export absent
  -> just validated-export <config>

validated export complete
  -> verify manifest and stop foundation work
```

Do not run stages that are already complete merely to create fresh console output.
Use their durable artifacts as evidence.

## Foundation stage gates

### Environment gate

Require:

- Use Python 3.12.
- Confirm `uv`, `just`, and Git are present.
- Confirm `uv.lock` is present.
- Obtain authority to install dependencies.

### Code gate

Require exact success from `just gate`. Do not summarize failures as passing.
Do not run the gate during an active resource-sensitive job unless requested.

### Upstream gate

Require:

- Match the exact source commit.
- Match the exact Hub revision.
- Match the exact weight SHA-256.
- Confirm the expected CPU embedding shape.

### Data gate

Require:

- Confirm every configured stream exists.
- Require sorted, unique UTC timestamps.
- Require finite OHLCV values.
- Require legal train and validation anchors above configured minima.
- Record the holdout count without consuming it.

### Device gate

Require a bounded probe for the intended qualified device. The committed
`just probe-mps` recipe uses this machine's absolute paths. Another host must copy
and review the probe config, choose a disposable unique run identity, and run:

```bash
uv run mantis-v2 probe --config <copied-probe-config>
```

Only MPS has an implemented real-data probe contract. Disclose CPU/CUDA
qualification gaps instead of inventing an equivalent command.

### Training gate

Require:

- Use a unique run name and noncolliding artifact directory.
- Keep `allow_overwrite=false` for production.
- Confirm the intended resume policy.
- Confirm enough storage exists.
- Keep exact provenance stable.

### Export gate

Require:

- Use validation-selected `best.pt`.
- Require finite validation metrics.
- Bind evaluation to the exact checkpoint.
- Confirm durable history proves it is the minimum-loss epoch.
- Require an exact safetensors reload.
- Require native/export inference parity.

Older completed artifacts may predate the current mandatory release-gate schema.
Document their original fields and limitations; do not rewrite historical JSON or
claim it passed fields that did not exist.

## Choose the next downstream action

```text
validated foundation export absent
  -> stop; downstream cannot begin

foundation export verified, prepare manifest absent
  -> downstream-prepare

prepare complete, embed manifest absent
  -> downstream-embed

embed complete, walk-forward manifest absent
  -> downstream-walk-forward

walk-forward complete, simulation manifest absent
  -> downstream-simulate

all four manifests complete
  -> report normal-path result and keep holdout sealed
```

Run one stage at a time when diagnosing or recovering. Use `downstream-run` only
for a clean, reviewed run with no partial output.

## Downstream stage gates

### Prepare gate

Require next-open execution, causal configured-strategy state, explicit stop/target/cost
rules, session force-close, and no normal holdout materialization.

### Embedding gate

Require exact foundation manifest and weights digest, expected preprocessing,
complete shards, complete metadata, and a chained manifest.

### Walk-forward gate

Require chronological folds, purged event spans, context embargo, train-only
scaler and model, validation-owned threshold, and unchanged test application.

### Simulation gate

Require correct multipliers, costs, sizing, Maximum Loss Limit, consistency,
session dates, and no future-exit-based selection.

## Holdout policy

The agent must reject routine holdout requests that are part of iterative tuning.

The downstream holdout may run only when the user has authorized a final,
one-time evaluation and all of these are frozen:

- Freeze the foundation export.
- Freeze the embedding contract.
- Freeze the classifier and scaler.
- Freeze the probability threshold.
- Freeze the simulator rules.
- Freeze the acceptance criteria.

The command requires both a reviewed config and the exact unlock. Record that the
holdout has been consumed. Do not modify the model based on the result and rerun
the same holdout.

## Safe diagnosis order

When a stage fails:

1. Preserve the exact error text.
2. Identify the first failed contract.
3. Read the relevant config and manifest.
4. Inspect stable JSON before native checkpoints.
5. Compare current source/config/data/lock/upstream identities with the artifact.
6. Use the narrowest read-only check that can prove the cause.
7. Choose either exact restoration/resume or a new run identity.
8. Never weaken a test, provenance check, split rule, or holdout lock.

See [Troubleshooting](troubleshooting.md) for symptom-specific paths.

## Completion report format

Every completed operation report should include:

```text
RESULT: <completed | not completed | blocked>

Workflow and stage:
Config and run name:
Device:
Source commit and dirty state:
Commands executed:
Input dataset identity and counts:
Checkpoint or stage identity:
Validation metrics and batch count:
Export hashes and parity status:
Downstream fold/simulation result:
Holdout status:
Artifacts:
Limitations or unqualified claims:
```

Never state that tests passed unless they ran. Never state that a model is
profitable because validation loss improved. Never call a validation result a
holdout result.

## Documentation-only work during a run

When the user authorizes documentation changes during an active run:

- Edit Markdown only.
- Avoid `just sync`, `just gate`, tests, smoke, data inspection, evaluation,
  export, and downstream commands.
- Use read-only source and artifact inspection.
- Do not format or rewrite unrelated files.
- Verify the final diff contains documentation only.
- Report that heavy runtime verification was deferred because the run was
  protected.

## Related documentation

- [Repository rules](../AGENTS.md)
- [System architecture](architecture.md)
- [Setup, dependencies, and hardware](setup-and-hardware.md)
- [End-to-end workflow](workflow.md)
- [Troubleshooting](troubleshooting.md)
