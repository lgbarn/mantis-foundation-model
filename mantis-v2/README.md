# MantisV2 Training

This package trains MantisV2 on the FFM NextLeg objective while preserving a strict boundary around the official upstream model.

Repository-wide guides:

- [Why the Mantis family is used](../docs/mantis-family.md)
- [System architecture](../docs/architecture.md)
- [Setup, dependencies, and hardware](../docs/setup-and-hardware.md)
- [End-to-end workflow](../docs/workflow.md)
- [AI-agent runbook](../docs/agent-runbook.md)
- [Troubleshooting](../docs/troubleshooting.md)

## Authoritative inputs

- Semantics: MantisV2 paper, arXiv 2602.17868.
- Code: `vfeofanov/mantis` tag `v1.0.0`, commit `0c94f8ceb9f1d1421dd292ed917090df8c31605b`.
- Weights: `paris-noah/MantisV2`, revision `99fe0f548960e272fbfa4b82fd9b5b5956779dfd`.
- Weight SHA-256: `49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1`.

The adapter verifies the immutable Hub revision and weight digest before using pretrained weights. Upstream license declarations conflict between Apache-2.0 and MIT, so this repository does not redistribute upstream source or weights.

## Data contract

The current production config binds 27 repaired Parquet streams under
`/Volumes/Storage/trading-research/data/MANTIS_NEXTLEG_PARQUET_V1/market`: nine
roots (`ES`, `NQ`, `RTY`, `YM`, `GC`, `SI`, `CL`, `ZB`, `ZN`) at `1min`,
`3min`, and `15min`. Every file must match the corpus manifest, provide sorted,
unique UTC timestamps, and contain finite `open,high,low,close,volume` values.

Verified on 2026-07-20: 40,251,760 rows spanning 2021-07-14 through
2026-07-13. The corpus contains 27 streams and is bound by manifest SHA-256
`2d8cf8b708c7c743c849059410b3654bc44f9fe4f7d795928de82526e120d703`.

Data is split independently per stream:

- Train: earliest pre-2026 90%.
- Validation: final pre-2026 10%.
- Holdout: timestamps on or after 2026-01-01; inaccessible to the normal evaluation command.

NextLeg pivots are detected per stream. A sample is legal only when its complete context, candle horizons, and both future legs remain inside one split. The production reserve is `200 + 2 * 256 = 712` bars.

## Objective

Each batch contains context-normalized OHLCV shaped `[B, 5, L]`, linearly resized to the MantisV2 native length of 512. Each channel is encoded independently and the five embeddings are concatenated. Context-only statistics standardize both context and future values; both are clamped to `[-10, 10]` before the future-minus-current candle target is calculated. This bounds every candle target to `[-20, 20]`, including near-constant price windows.

The model jointly predicts:

- Standardized future-candle moves shaped `[B, 5, 4]` at horizons 5, 10, 20, and 25.
- `log1p` durations for the newborn leg and following counter-leg, shaped `[B, 2]`.

The loss is candle MSE plus leg SmoothL1, with explicit configurable weights.

## Commands

Run commands from the repository root:

```bash
just sync
just gate
just verify-upstream
just inspect-data mantis-v2/configs/nextleg-parquet-v2.toml
just probe-mps
just train mantis-v2/configs/nextleg-parquet-v2.toml
just validated-export mantis-v2/configs/nextleg-parquet-v2.toml
```

`gate` includes deterministic synthetic smoke training, evaluation, checkpoint,
and export parity. `verify-upstream`
downloads or reuses the pinned official weights, checks their digest, constructs
the real upstream MantisV2 on CPU, and verifies its embedding contract.
`inspect-data` validates all configured files and computes legal anchor counts.
The probe is guarded to exactly 32 real-data optimizer updates and one
validation batch, with configured validation coverage across all streams.
`validated-export` is the normal release path: it evaluates the
validation-selected checkpoint and then exports only if the evaluation gate
passes. The qualified production path starts from pinned official MantisV2
weights and runs on Apple MPS.

For diagnosis, `just evaluate <config>` and `just export <config>` expose the individual stages. Direct export still enforces the same evaluation gate.

## Training recipe

This is supervised NextLeg fine-tuning, not a reproduction of MantisV2 contrastive foundation pretraining. It combines the strongest compatible parts of the authoritative sources:

- Upstream MantisV2: official pretrained weights, linear resizing to 512, channel-independent encoding, concatenated 256-dimensional embeddings, transformer fine-tuning with the MPS-unstable token generator and unused projection frozen, AdamW weight decay `0.05`, per-update 10-epoch linear warmup, and cosine decay.
- FFM NextLeg: 120 epochs, 200 sampled training batches per epoch, 20 validation batches, learning rate `0.0001188117389055629`, horizons 5/10/20/25, two pivot legs, cap 256, and validation-loss model selection.
- Local MPS: batch 128 and zero data-loader workers. A local throughput sweep showed batch 192 was fastest, but 128 preserves memory headroom while the 1.4 GiB corpus and optimizer state are resident.

Training stops after eight epochs without validation-total improvement. `metrics.json` records the learning rate plus train and validation total, candle MSE, and leg SmoothL1 for every completed epoch. These regression checkpoints do not report AUC or log loss; those belong to the later 3-minute classifier stage.

## Artifacts and resume

Outputs live under the configured external `artifact_root`:

```text
/Volumes/Storage/trading-research/artifacts/mantis-foundation-model/mantisv2-nextleg-parquet-v2/
|-- checkpoints/latest.pt
|-- checkpoints/best.pt
|-- metrics.json
|-- provenance.json
|-- train-result.json
|-- evaluation.json
`-- export/
    |-- evaluation.json
    |-- model.safetensors
    `-- manifest.json
```

Native checkpoints include model and optimizer state, epoch, global step, Python, NumPy, CPU Torch, CUDA, and MPS RNG state, config digest, full dataset content identity, training-source content digest, Git state, dependency-lock digest, and pinned upstream identities. Resume fails closed if any load-bearing identity changes, including uncommitted source content. Each epoch uses a pending checkpoint that is promoted only after metric history is durable, so an interrupted two-file update falls back to the prior valid epoch.

`run.name` creates the final directory below `run.artifact_root`. A non-resume production run refuses to replace an existing named run unless `run.allow_overwrite` is explicitly enabled. Resume uses `latest.pt`; evaluation and export use validation-selected `best.pt`. Export fails closed unless `evaluation.json` records finite validation metrics and is bound to the exact `best.pt` SHA-256, epoch, global step, config, dataset, source, dependency lock, and upstream model identities. It also confirms that the checkpoint is the minimum-loss epoch in durable metric history before writing safetensors. The exact validated evaluation is copied into the export bundle and its SHA-256 is recorded in the manifest. The smoke and guarded probe configs enable overwrite only for disposable verification artifacts.

## Release storage

Large training artifacts are not committed to the source Git repository. Native checkpoints and complete run state remain on the external drive. Safetensors is the deployment format; pickle-based native checkpoints remain private training state. The upstream license declarations currently conflict, so derived weights must not be uploaded to Hugging Face or otherwise redistributed until the license is resolved. Publishing remains an explicit operator action and is not performed by training or export.

The completed `mantisv2-nextleg-parquet-v2` artifact records source commit
`c98e00ce5a7a7242c4c635ec1de39f5d15d99812`, early-stopped after 32 epochs,
selected human epoch 24, and passed the mandatory checkpoint-bound evaluation
and export gate. See the
[current run record](../docs/runs/2026-07-20-mantisv2-nextleg-parquet-v2.md).
The older target-clamp artifact remains documented as historical evidence.

The earlier `mantisv2-nextleg-mps` run is preserved as a superseded diagnostic artifact. It completed 12 epochs before investigation found that unclamped future normalization could amplify a one-tick move in a constant-price window into a target of 15,625. Because correcting the target changes both source and config provenance, that checkpoint is intentionally not resumable by the corrected production config.

The existing `Futures-Foundation-Model/checkpoints/mantis_ssl_ctr_seq2seq.pt` is a legacy Mantis checkpoint with `tokgen_unit`/`vit_unit` keys. It is not compatible with the MantisV2 architecture and is deliberately not loaded.

## Production notes

- The production config explicitly requires Apple MPS and fails rather than falling back to CPU.
- The internal disk is nearly full, so production checkpoints and metrics are written to `/Volumes/Storage`.
- No experiment-tracking service is enabled. Metrics and provenance are local JSON artifacts.
- The normal foundation evaluator refuses holdout access, and no foundation
  holdout command is implemented. The separate downstream holdout has a reviewed
  two-key workflow.
- Full data hashing validates all 27 Parquet streams and their corpus manifest before training.

## Strategy and Topstep 100K downstream pipeline

The current production config is `configs/trend-magic-topstep-100k.toml`. It
uses `ES`, `NQ`, `RTY`, `YM`, `GC`, `CL`, and `ZB` at 1m, 3m, and 15m. A
candidate is emitted for every eligible closed 3m bar and is directed by the
current close-CCI(20)/SMA-TR(5) Trend Magic state; flips are not required.
Historical Supertrend configs retain their original semantics and artifact
identities. Entry is the next 3m
open, the stop is 0.5 ATR(20), the primary target is 3R, the horizon is 120
bars, same-bar ties stop out, and round-trip cost is 0.03R. Any still-open trade
is force-closed on the final completed bar before 3:10 PM CT; the simulator
never selects trades based on their future realized exit time.

Run one stage at a time for bounded failure recovery and inspection:

```bash
just downstream-prepare mantis-v2/configs/trend-magic-topstep-100k.toml
just downstream-embed mantis-v2/configs/trend-magic-topstep-100k.toml
just downstream-walk-forward mantis-v2/configs/trend-magic-topstep-100k.toml
just downstream-simulate mantis-v2/configs/trend-magic-topstep-100k.toml
```

Or run the normal chain:

```bash
just downstream-run mantis-v2/configs/trend-magic-topstep-100k.toml
```

All locations and parameters live in TOML. A command can record a one-off
scalar override without changing code, for example:

```bash
uv run mantis-v2 downstream-simulate \
  --config mantis-v2/configs/supertrend-topstep-100k.toml \
  --set topstep.contracts=2
```

Prepared candidates and predictions are Parquet. MantisV2 embeddings are
bounded float16 `.npy` shards with paired Parquet metadata. Each stage verifies
the prior stage's hashes and emits a manifest. The foundation export path and
trusted safetensors SHA-256 are both explicit config values. The logistic head stores scaler
and coefficient arrays in `.npz`, never pickle.

Head-only reruns may reuse an existing embedding stage without copying or
rewriting it. Set the manifest path/SHA and producer-config path/SHA fields,
choose a new `run.name`, and run only `downstream-walk-forward`. The producer's
legacy workflow identity and full data, label, and encoder semantics must match
before any shard is read. Every feature and metadata shard is then rehashed,
and the new walk-forward manifest records the exact embed identity and a
separate head configuration digest. The tuned production example is:

```bash
just downstream-walk-forward \
  mantis-v2/configs/supertrend-topstep-100k-head-c0001-v2.toml
```

Logistic solver, inverse regularization strength `regularization_c`, tolerance,
iteration ceiling, class weighting, and convergence policy are TOML settings.
Production uses `convergence_policy = "fail"`; a `ConvergenceWarning` writes a
durable `failure.json` and stops the stage. The completed walk-forward manifest
also compares mean class-balanced test log loss and Brier score with the
constant 0.5 probability baselines (`log(2)` and `0.25`). Simulation and sealed
holdout evaluation reject a run unless both convergence and primary-loss gates
pass. See [ADR 0003](../docs/adr/0003-reuse-embeddings-for-isolated-head-runs.md).

`just downstream-smoke` runs a deterministic CPU-only component smoke and
writes all four stage manifests. It is part of `just gate`. Production stages
fail closed on partial output; resume with a new run name, or explicitly use an
overwrite-enabled recovery config after inspecting the partial directory.

Walk-forward defaults to 24 training months, 3 validation months, 3 test
months, and a 3-month stride. Label event spans are purged and the 1m/3m/15m
context is embargoed at partition boundaries. Standardization and logistic
training use train only. Validation selects and persists the numeric top-50%
probability threshold; test uses that exact value. Class-balanced log loss and
Brier score are primary, while unweighted scores, ROC AUC, and PR AUC are
diagnostic.

The simulator defaults to the current Topstep 100K Combine baseline: $100,000
start, $6,000 target, $3,000 end-of-day trailing Maximum Loss Limit locked at
$100,000, strict 50% consistency, 10-mini maximum, and at least two trading
days. The base profile has no Daily Loss Limit. Use
`configs/supertrend-topstep-100k-dll.toml` only for the optional $2,000 soft
daily-halt variant. Express Funded and Live Funded rules are not mixed into the
Combine simulator.

Normal preparation does not materialize 2026 labels or embeddings. The sealed
`downstream-holdout` command requires a reviewed config
with `evaluation.allow_holdout=true` and the matching explicit `--unlock`
value. Only then does it generate 2026 candidates and embeddings and apply the
final validation-owned head and threshold once.
