# Troubleshooting

This guide provides safe diagnosis and recovery for the implemented MantisV2
workflows. Diagnose the first failed contract. Do not bypass it.

## Safety rules

Before diagnosing:

- Preserve the exact error text.
- Do not kill or signal an active process.
- Do not edit an active config, source data, source code, lock file, checkpoint,
  metric file, shard, or manifest.
- Do not enable overwrite to hide a collision.
- Do not delete partial artifacts before identifying the last durable stage.
- Do not weaken tests, provenance, split rules, parity checks, or holdout locks.

Use stable JSON and manifests before loading native checkpoints.

## Setup problems

| Symptom | Likely cause | Safe diagnosis | Recovery |
| --- | --- | --- | --- |
| Python version rejected | Interpreter is not 3.12 | `python3 --version`; `uv run python --version` | Install/select Python 3.12 with approval, then resync |
| `uv` not found | Tool is not installed or not on PATH | `command -v uv` | Ask before installing; follow official uv installation guidance |
| `just` not found | Task runner is absent | `command -v just` | Ask before installing; commands can be read from `justfile` meanwhile |
| Sync changes unexpectedly | Lock or project files changed | `git diff -- pyproject.toml mantis-v2/pyproject.toml uv.lock` | Restore intended tracked versions or create a reviewed dependency change |
| Import failure after sync | Wrong interpreter or partial environment | `uv run python -c "import mantis_v2"` | Reconcile environment with `uv.lock`; do not hand-install packages |

## Official-weight problems

| Symptom | Likely cause | Safe diagnosis | Recovery |
| --- | --- | --- | --- |
| Hub download fails | Network, cache, or Hub availability | Check configured model/revision and existing cache; preserve HTTP error | Restore connectivity or approved authentication; retry exact revision |
| Weight digest mismatch | Wrong or modified file | Compare configured SHA-256 with observed digest | Remove only the bad cache entry after confirming its exact path and authority, then fetch pinned revision |
| Embedding shape mismatch | Upstream package or adapter drift | Check `mantis-tsfm`, source pin, and config | Restore locked package and pins; do not reshape around the failure |
| Legacy checkpoint fails | V1 checkpoint passed to V2 architecture | Inspect state-key prefixes and documented checkpoint identity | Use verified official V2 weights; do not convert implicitly |

## Data problems

| Symptom | Likely cause | Safe diagnosis | Recovery |
| --- | --- | --- | --- |
| Data root missing | External volume is not mounted or path is local to another host | Check mount and copied config path | Mount the correct volume or update a new config copy |
| Stream missing | Filename does not match symbol/interval contract | Compare directory listing with configured matrix | Produce or rename the intended source file outside an active run |
| Timestamp parse error | Invalid or mixed timestamp format | Inspect the reported rows only | Correct upstream data production and create a new dataset identity |
| Timestamps unsorted or duplicated | Source assembly error | Inspect reported stream around first violation | Rebuild source deterministically; never sort active input in place |
| Non-finite OHLCV | Missing or infinite market value | Inspect exact stream and row | Correct source policy and regenerate data |
| Too few anchors | Context, horizon, leg cap, or data length makes samples illegal | Run `inspect-data` only when resource-safe; inspect split dates | Add valid history or create a reviewed target config; do not weaken split containment |
| Anchor count changed | Source bytes, paths, or config changed | Compare dataset identities and config digest | Restore exact inputs for resume or create a new run |

## Device problems

| Symptom | Likely cause | Safe diagnosis | Recovery |
| --- | --- | --- | --- |
| Explicit MPS unavailable | Non-Apple host or incompatible PyTorch/macOS | Inspect PyTorch MPS availability | Use a new CPU config for bounded work or a qualified host |
| Explicit CUDA unavailable | Driver/build/device mismatch | Inspect `torch.cuda.is_available()` when no active GPU job is protected | Repair host with approval or use another reviewed device config |
| CPU rejected | `require_accelerator=true` | Read selected config | Copy config, set CPU and false only for a new run identity |
| Downstream uses CPU on NVIDIA host | Downstream `auto` never selects CUDA | Read downstream device and implementation capability | Use CPU or MPS; CUDA downstream requires code and test work |
| Out of memory | Batch or resident data exceeds device capacity | Preserve error; inspect batch/config and device memory outside active run | Create a new bounded config with smaller batch; do not mutate resumable config |
| Cross-device metrics differ | Floating-point and kernel differences | Compare same config family on independent run identities | Treat as a new qualification; do not resume across config/device digests |

## Training and checkpoint problems

| Symptom | Likely cause | Safe diagnosis | Recovery |
| --- | --- | --- | --- |
| Existing run collision | Run name or artifact directory already exists | Inspect config and target directory read-only | Choose a new run name; never enable overwrite for production casually |
| Artifacts exist but no resumable checkpoint | Interrupted or partial first write | Inspect directory and durable JSON without deleting | Preserve it; decide on explicit recovery config or new run |
| Provenance mismatch | Config, data, source, lock, or upstream identity changed | Compare checkpoint error with current identities | Restore exact inputs or start a new run identity |
| `metrics.json` not contiguous | Interrupted or edited history | Inspect epoch sequence and pending/latest state | Use built-in pending recovery when valid; do not repair history manually |
| Non-finite loss | Invalid data, unstable targets, or optimization failure | Preserve batch/metric context and inspect finite-data contracts | Fix root data/config/model cause in a new run; do not skip the batch |
| Training loss falls, validation rises | Overfitting | Inspect several epochs and both loss components | Let configured early stopping act; review objective/data in a new experiment |
| One loss improves and one degrades | Objective imbalance | Compare candle and leg trends separately | Review weights/targets as a new config; do not judge only total loss |
| `best.pt` does not update | No strict validation-total improvement | Read metrics and patience | Normal until patience triggers; do not copy latest over best |
| Resume is not exact | Seed/RNG or provenance restoration failed | Compare exact resumed metrics with durable history and test contract | Stop new writes and diagnose checkpoint restoration; preserve artifacts |

## Evaluation and export problems

| Symptom | Likely cause | Safe diagnosis | Recovery |
| --- | --- | --- | --- |
| `best.pt` missing | No valid improvement checkpoint exists | Inspect metrics and checkpoint directory | Diagnose training completion; do not substitute `latest.pt` |
| Best checkpoint changed during evaluation | Concurrent writer touched run | Check active process and checkpoint hashes | Wait for the writer to complete; rerun evaluation only with authority |
| Export says evaluation missing or stale | Direct export lacks exact authorization | Inspect evaluation/checkpoint/config hashes | Use `validated-export` on stable completed state |
| Best epoch not minimum history loss | Checkpoint/history inconsistency | Compare durable metric records | Preserve run and diagnose transactional history; do not edit JSON |
| Safetensors reload mismatch | Serialization or tensor-state error | Preserve exact tensor mismatch | Diagnose export implementation; do not publish output |
| Inference parity fails | Exported model differs numerically from native state | Preserve tolerance and output differences | Stop release; identify architecture/state/preprocessing mismatch |
| Older artifact lacks new gate fields | Artifact predates mandatory gate schema | Inspect dated run record and original JSON | Treat as historical pre-gate evidence; do not rewrite it to look current |

## Downstream problems

| Symptom | Likely cause | Safe diagnosis | Recovery |
| --- | --- | --- | --- |
| Foundation identity mismatch | Manifest path or weights hash is stale | Compare config with export manifest and file SHA-256 | Point a new config to the intended immutable export |
| Partial stage directory | Prior stage stopped before manifest completion | Inspect existing files and last complete upstream manifest | Preserve partial output; use a new run name or reviewed recovery path |
| Alignment test fails | A higher-timeframe bar closes after the decision time | Inspect reported timestamps and timestamp semantics | Fix causal alignment; never use incomplete higher-timeframe data |
| Labels appear in 2026 normal output | Holdout protection failed | Stop downstream development and preserve evidence | Fix the guard before continuing; do not inspect outcomes for tuning |
| Embedding shard incomplete | Interrupted inference or write failure | Compare shard metadata with manifest expectations | Resume only through a supported recovery identity; do not concatenate partial data |
| Classifier metrics look excellent but simulation fails | Predictive metric does not survive execution/cost rules | Compare threshold ownership, turnover, costs, and fills | Treat as a valid negative result; do not optimize on test or holdout |
| Simulation rule mismatch | Wrong profile, multiplier, session, or override | Compare selected TOML and manifest | Create a corrected new run identity; preserve prior result |
| No test trades | Threshold, fold, or candidate filters are too restrictive | Inspect counts at each stage | Report the empty result; change rules only as a new experiment |

## Holdout problems

| Symptom | Meaning | Action |
| --- | --- | --- |
| Foundation evaluator rejects holdout | Expected; foundation holdout path is not implemented | Do not bypass it |
| Downstream holdout rejects config | `allow_holdout` is false or config is not reviewed | Keep sealed until final authorization |
| Downstream holdout rejects unlock | Exact second key is absent | Do not guess or weaken the key |
| Holdout already exists | Final evidence may already be consumed | Stop and review governance record before any action |
| User wants to retune after holdout | Holdout would become validation data | Record result and design a future unseen holdout instead |

## Read-only status checklist

For foundation training, report from `metrics.json` and stable result JSON:

- Epoch index and human epoch number
- Global step
- Learning rate
- Train and validation component losses
- Best validation loss and checkpoint update
- Early-stop patience when available
- Draw count and unique coverage when reconstructed

For a completed export, report:

- Best epoch and step
- Evaluation batch count and exact reproduction
- Component metrics
- Tensor count
- Parity tolerances and status
- Model SHA-256 when recorded or safely computed
- Config, dataset, source, lock, and upstream identities
- Whether the artifact uses current or historical manifest schema

For downstream stages, report the last complete manifest, output counts, next
safe stage, and holdout state.

## When to start a new run

Start a new run identity when any load-bearing input intentionally changes:

- Config values
- Device recipe
- Data bytes or paths
- Source code
- Dependency lock
- Upstream source or weights
- Target definition
- Preprocessing
- Foundation export
- Trading or account rules

Do not reuse a name merely because the previous run failed. A unique identity is
part of the audit trail.

## Related documentation

- [AI-agent runbook](agent-runbook.md)
- [End-to-end workflow](workflow.md)
- [System architecture](architecture.md)
- [Setup, dependencies, and hardware](setup-and-hardware.md)
