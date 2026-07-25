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
just train mantis-v2/configs/nextleg-runpod-cuda-bundled-v1.toml
just validated-export mantis-v2/configs/nextleg-runpod-cuda-bundled-v1.toml
just rl-dry-run mantis-v2/configs/rl-entry-smoke.toml
just rl-build-episodes mantis-v2/configs/rl-entry-smoke.toml 0 training 21
just rl-account-replay fixture.json replay.json mantis-v2/configs/rl-entry-smoke.toml
just rl-validate-environment training.json validation.json environment-validation.json \
  mantis-v2/configs/rl-entry-smoke.toml
just rl-smoke artifacts/rl-entry-smoke-v1 mantis-v2/configs/rl-entry-smoke.toml
just rl-train training.json artifacts/rl-entry-training \
  mantis-v2/configs/rl-entry-topstep-100k.toml shared_ticker_value
just rl-optuna-search training.json validation.json artifacts/rl-optuna \
  mantis-v2-topstep-100k-shared-ticker-value-v1
just rl-freeze-architecture-plan winner.json artifacts/rl-architecture-plan \
  2026-07-22T12:00:00+00:00 \
  --training-manifest training.json --validation-manifest validation.json
just rl-run-architecture-ablation architecture-plan.json artifacts/rl-ablation
just rl-qualify-architecture winner.json ablation-evidence.json \
  artifacts/rl-candidate
just rl-run-seed-campaign candidate-freeze.json artifacts/rl-seed-campaign \
  --training-manifest training.json --validation-manifest validation.json
just rl-decide-continuation candidate-freeze.json budget-evidence.json \
  artifacts/rl-continuation-decision
just rl-run-seed-campaign candidate-freeze.json artifacts/rl-seed-campaign \
  --training-manifest training.json --validation-manifest validation.json \
  --resume --continuation-decision continuation-decision.json
just runpod-official-bootstrap \
  reports/runpod/source-SOURCE_SHA.tar.gz \
  reports/runpod/official-bootstrap-SOURCE_SHA.json
just qualify-cuda-bf16 fp32.json bf16.json qualification.json
```

### Official frozen expected-R screen

The MNQ screen uses accepted `NQ_3min.parquet` only. Preparation runs locally
before a Pod exists; it does not repeat the corpus audit. It writes one immutable
candidate/window/context bundle whose manifest binds every row and byte. Run in
this order with unique external artifact paths:

```bash
just frozen-screen-prepare /path/to/NQ_3min.parquet /external/frozen-screen-v1/input
just frozen-screen-preflight \
  /external/frozen-screen-v1/input/manifest.json \
  /network-volume/frozen-screen-v1/embed \
  /external/frozen-screen-v1/focused-checks.json \
  "uv run mantis-v2 frozen-screen-embed --config /workspace/mantis/mantis-v2/configs/frozen-expected-r-v1.json --input /workspace/input/manifest.json --output /workspace/output/embed" \
  HOURLY_RATE DEADLINE_HOURS /external/frozen-screen-v1/preflight.json
just frozen-screen-run \
  /external/frozen-screen-v1/preflight.json \
  /external/frozen-screen-v1/launch-manifest.json \
  /external/frozen-screen-v1/bound-decision.json \
  /path/to/runpod-local.toml /path/to/runpodctl
just frozen-screen-compare \
  /external/frozen-screen-v1/input/manifest.json \
  /external/frozen-screen-v1/embed/manifest.json \
  /external/frozen-screen-v1/selection.json \
  mantis-v2/configs/frozen-expected-r-v1.json cuda
```

The preflight receipt requires the four focused checks to pass in under 60
seconds, one L40S, a maximum $10 budget, a deadline no longer than six hours or
the rate-derived spend limit, a 30-second health interval, one provenance-safe
resume, and termination after success or a second failure. `frozen-screen-run`
is the only supported paid route: it verifies that the staged bundle contains
the receipted input, the workload command retries the exact embed command only
once, and the existing RunPod supervisor enforces milestones, deadline, budget,
termination, artifact replication, and billing reconciliation. The embedder verifies
the pinned official revision and weights, uses native MantisV2 preprocessing,
transformer layer 2 and combined CLS/mean pooling, then performs fixed-fixture
BF16/FP32 parity. Failed parity uses FP32. Each feature/metadata shard is atomic;
a rerun verifies complete shards and resumes at the first uncommitted row.

Comparison first runs the fixed July 2023-June 2025 training, July-September
2025 validation, and October-December 2025 test gate. Only a passing initial
gate permits the three additional anchored development folds; the initial gate
never counts toward the two required development wins. Every boundary purges
the full 512-bar feature lookback and outcome horizon. The CUDA comparison is
the supported default and fails closed when CUDA is unavailable. CPU requires
an explicit user-approved or CUDA-qualification-failure reason recorded in the
artifact with `--cpu-exception`. CUDA uses a
weighted FP64 Cholesky ridge solve, parallel threshold recurrence, and FP64
bootstrap reductions with progress markers; it never falls back to the CPU
implementations for those stages. Representative CPU/GPU qualification requires
prediction and prediction-derived interval agreement within 0.001 (observed
maximum absolute differences: 0.00080863 and 0.000077325), MSE agreement within
1e-4 relative or 1e-6 absolute, and exact selected-trade, outcome-only interval,
expectancy, gate, status, and final-selection assertions. These numerical solver
tolerances do not relax any trading-quality or promotion gate. Mantis is selected
only with at least two fold wins and a positive pooled
selected-expectancy interval. Otherwise qualifying raw is selected; if neither
qualifies the durable result is `stopped`. TensorBoard, fine-tuning, PPO, and the
sealed 2026 holdout are not part of this stage.

The default remote runtime is RunPod's supported PyTorch 2.8 template
`runpod-torch-v280`, pinned to the image digest recorded in
`infra/runpod/examples/intent-h100-qualification.json`. The source archive and
receipt are staged and hash-verified before the paid Pod is created. Custom
images remain a recovery option, not the production default.

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

The bundled CUDA candidate uses
`configs/nextleg-runpod-cuda-bundled-v1.toml`. It installs zero-delta rank-8,
alpha-16 attention LoRA at construction, warms only the candle and leg heads
for at most 2,000 optimizer updates, then enables only those heads and LoRA
under a fresh AdamW optimizer. The original encoder and unused adapter remain
frozen. Both phases share one hard 10,000-update ceiling; unused warm-start
allowance can be consumed only after the LoRA transition. Checkpoints record
the phase, phase-local and total counts, transition-parent digest, optimizer
identity, and trainable counts. `just train` resumes either phase from those
identities. `validated-export` rejects a warm-start checkpoint and accepts only
a validation-selected LoRA-phase checkpoint with native, adapter-reload, and
merged-model parity.

Foundation precision is a strict experiment identity: `fp32` is the accepted
default and `bf16` is rejected unless explicit CUDA reports BF16 support. BF16
uses autocast for forward/loss only, retains FP32 parameters and optimizer
state, and must pass the registered fixed-fixture, resume, export, and manifest
gate before promotion. CPU policy tests cannot qualify BF16.

For diagnosis, `just evaluate <config>` and `just export <config>` expose the individual stages. Direct export still enforces the same evaluation gate.

`rl-dry-run` validates the locked entry-only RL experiment without training or
reading the sealed holdout. It rehashes the source, dependency lock, corpus,
embeddings, foundation export, and foundation weights before atomically writing
a no-overwrite identity manifest. Use `configs/rl-entry-topstep-100k.toml` for
the production contract; the smoke config reduces budgets only.

`rl-build-episodes` creates an immutable Monte Carlo schedule for one named
walk-forward fold and partition. The command rehashes every owned corpus stream
and every embedding feature/metadata shard before sampling. Each episode keeps
one ticker, one fixed mini/micro profile, complete lookbacks and exit horizons,
and at most 20 chronological trading sessions inside its partition. ZB is
mini-only. The manifest stores shard row spans for memory-mapped observation
lookup plus all upstream identities. Repeating the same command resumes by
accepting an identical completed manifest; changed inputs or parameters fail
without replacing it. Use a new RL run name for a different schedule contract.

`rl-validate-environment` is the dependency-light Stage 0 gate. It consumes
training and validation episode manifests from the same fold, rejects test and
sealed-holdout partitions, and reconstructs complete chronological 3-minute
episodes from the pinned Parquet corpus and memory-mapped embedding shards. The
environment exposes the shared `[skip, enter]` mask used by every baseline,
fills accepted close-formed decisions at the next open, and permits only no-op
while positioned. Invalid entry raises instead of being rewritten.

The validation output records finite and prefix-causality checks, exact replays
for reject-all, take-all, participation-matched seeded random-take, the immutable
rejected logistic fold head, and a fixed HistGradientBoosting contextual
baseline. Supervised fitting uses training only and selects thresholds on
validation only. It also records mmap p95 latency, deterministic environment
steps per second, the 100x/5,000 steps-per-second gates, input hashes, host,
Python, NumPy, and scikit-learn versions.

`rl-smoke` is the Stage 1 CPU qualification. It runs the official Gymnasium and
Stable-Baselines3 environment checks through the production entry adapter, then
trains MaskablePPO for exactly 50,000 steps on a balanced, deterministic,
learnable synthetic schedule. Completion requires finite policy parameters,
zero illegal actions, better reward than reject-all, identical deterministic
actions after native reload, and a reproduced schedule. Every 10,000 steps is
an immutable `checkpoints/step-N/` bundle containing the model, RNG/runtime
state, and a hash manifest. One atomic `state.json` pointer commits a completed
bundle; an interrupted save or pointer swap leaves the prior bundle resumable.
A trained pointer can resume evaluation and publication without retraining.
Resume requires the loaded model step, pointer step, source, config, lock,
policy, dependencies, seed, schedule, Python/NumPy/Torch/Gym RNG state, and
checkpoint hashes to agree. Completed or changed run identities are never
overwritten. This smoke does not read production data or the sealed holdout and
does not qualify MPS for policy training.

`rl-optuna-search` is the validation-owned Stage 2 search. It uses the immutable
`configs/rl-optuna-v1.toml` contract, Optuna 4.9.0 persistent SQLite storage,
fresh deterministically seeded TPE proposals, and at most 30 sequential trials.
Each trial trains exactly three derived seeds to the first complete 14-, 28-, or
56-episode rollout reaching at least 500,000 policy decisions. Any validation
blow is infeasible; feasible COMPLETE trials rank by pass-rate LCB, median pass
days, then trial number. The command accepts only same-fold training and
validation manifests before the sealed holdout. It cannot receive a test or
holdout path and never exports a winner when no feasible trial exists.

`rl-qualify-architecture` is the validation-only Stage 3 transfer-ablation gate.
The plan command accepts repeated, ordered `--training-manifest` and
`--validation-manifest` flags and freezes every same-fold pair. The ablation
runner ledgers every variant/seed/fold tuple; rerun it with `--resume` after an
interruption. Qualification consumes the frozen `winner.json` and aligned
fold/seed/ticker/profile/regime/
calendar-block evidence for all three preregistered architectures. It fails on
missing pairs, non-finite runs, action collapse, any blow, or a negative pooled
or ticker/profile paired one-sided 95% lower bound against independent PPO. A
passing run writes only the shared-actor/ticker-specific-critic candidate under
its content hash, with the 5 x 2M development, 10 x 5M confirmation, 10M maximum
continuation, and serving seed 42 protocol frozen before any test access.
`rl-decide-continuation` consumes only matched 2M/5M validation outcomes and a
candidate freeze. It writes an immutable decision before any 10M update: all ten
seeds continue together only when pooled improvement is positive, its
synchronized two-week adjacent moving-block bootstrap LCB is nonnegative, every
ticker/profile point estimate is
nonnegative, and no safety or estimability gate regresses. The seed campaign API
then records immutable attempts, verifies 2M-to-5M lineage and fresh endpoint
parity, and emits `serving-freeze-v1`; test remains inaccessible throughout.

The approved CPU RL supply chain resolves from hash-locked `uv.lock` entries:

| Package | Resolved version | License | Accepted use |
| --- | --- | --- | --- |
| Gymnasium | 1.3.0 | MIT | Environment protocol and official checks |
| Stable-Baselines3 | 2.9.0 | MIT | CPU policy utilities and persistence |
| SB3-Contrib | 2.9.0 | MIT | MaskablePPO and action-mask utilities |
| Optuna | 4.9.0 | MIT | Later local persistent search; unused by this smoke |

These are free local libraries. No API key, hosted service, telemetry service,
or metered dependency is enabled. Version ranges are constrained in
`mantis-v2/pyproject.toml`; exact wheel/source hashes and transitive versions are
recorded in `uv.lock`. Any future upgrade must update the lock, verify installed
versions and licenses, refresh provenance identities, and rerun this gate.

### Bar-level Topstep account replay

`rl-account-replay` is the deterministic qualification seam for the Topstep
100K account rules. It consumes a schema-versioned JSON marked-equity fixture,
loads the pinned rule and fee identities from the RL config, and atomically
publishes a no-overwrite JSON manifest. It does not read market data,
embeddings, policy outputs, or the sealed holdout.

Each fixture has exactly `schema_version`, `fixture_id`, `ticker`, and `bars`.
`ticker` binds the entire Combine attempt to one underlying; mini and micro
contracts for another underlying fail closed. Bars are strictly increasing and
use these fields:

- `timestamp`: timezone-aware ISO-8601 bar timestamp;
- `action`: `none`, `enter`, or `exit`;
- `contract`, `quantity`, and `side`: required by `enter`; supported profiles
  are one mini or 10 micros, while ZB is mini-only;
- `mark_ticks`: signed P&L ticks for the open position, not a raw price move;
- `realized_pnl`: optional deterministic dollar adjustment for independent
  hand-calculated account fixtures;
- `pending_orders`: non-negative count used to prove cancellation behavior.

Open-position marked equity includes gross tick P&L, one adverse tick per side,
and the pinned product round-turn fee. Balance receives that net amount only
when the fixture exits or the account forces a flatten, so friction is booked
once. MLL touch uses `<=` for both realized and unrealized equity and terminates
as `BLOW`. DLL touch uses `<=`, flattens, cancels, and locks entry until the next
5 PM CT session without failing the Combine. Positions are forced flat at 3:10
PM CT. EOD balance alone ratchets MLL; the floor never falls and locks at
$100,000. Pass requires $6,000 profit, best-day consistency at or below 50%, and
two trading days. An active attempt becomes `TIMEOUT` after 20 session days.

The manifest records the complete account path, terminal state, input SHA-256,
and config/rule/fee digests. Replaying identical input with identical pinned
configuration produces byte-identical output. An existing destination fails
closed instead of being replaced.

## Training recipe

This is supervised NextLeg fine-tuning, not a reproduction of MantisV2 contrastive foundation pretraining. It combines the strongest compatible parts of the authoritative sources:

- Upstream MantisV2: official pretrained weights, linear resizing to 512, channel-independent encoding, concatenated 256-dimensional embeddings, transformer fine-tuning with the MPS-unstable token generator and unused projection frozen, AdamW weight decay `0.05`, per-update 10-epoch linear warmup, and cosine decay.
- FFM NextLeg: 120 epochs, 200 sampled training batches per epoch, 20 validation batches, learning rate `0.0001188117389055629`, horizons 5/10/20/25, two pivot legs, cap 256, and validation-loss model selection.
- Local MPS: batch 128 and zero data-loader workers. A local throughput sweep showed batch 192 was fastest, but 128 preserves memory headroom while the 1.4 GiB corpus and optimizer state are resident.

Training stops after eight epochs without validation-total improvement. `metrics.json` records the learning rate plus train and validation total, candle MSE, and leg SmoothL1 for every completed epoch. These regression checkpoints do not report AUC or log loss; those belong to the later 3-minute classifier stage.

The CUDA accuracy matrix compares a `1min`/`3min`/`15min` ablation with the
full four-timeframe `1min`/`3min`/`5min`/`15min` recipe. The production choice
is the four-timeframe recipe; the three-timeframe cells remain only as a
compute-matched test of the incremental 5-minute signal. Seeds 42-44 screen the
recipe and seeds 42-46 confirm it. The matrix can compare full fine-tuning with
rank-8 and rank-16 LoRA, but LoRA is selected only when its confirmation median
meets the frozen non-inferiority margins against the full-fine-tune median.

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

The loader names this exact recipe `trend_magic_fixed_3r_v1` and rejects any
drift in its direction, timing, risk, target ladder, horizon, cost, or session
settings. This declaration does not change the completed producer TOML or its
pinned SHA-256. The 2R-activated, 0.75R-giveback trail is a separate downstream
execution policy; it does not define the supervised label.

Run one stage at a time for bounded failure recovery and inspection:

```bash
just trend-magic-verify
just trend-magic-prepare
just trend-magic-embed
just trend-magic-head
```

The head-only config pins the completed production embeddings and their producer
config by SHA-256, uses a distinct run identity, and keeps convergence and
proper-score gates fail-closed. The original `C=1.0` fold-0 convergence failure
is retained as evidence. Do not use the all-stage chain for this consumer config.

No all-stage Trend Magic command is currently qualified. Use the separate
producer and head-consumer commands above; this prevents the known-failing head
settings from triggering another expensive embed run.

The completed `head-c0001-v2` walk-forward converged on all eight folds but
failed both primary proper-score gates (weighted log loss 0.695306 and weighted
Brier 0.251044). Do not run `downstream-simulate` or unlock the holdout for this
head. A future candidate must use a new run identity and pass the same gates.

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
`rl-train` is the Stage 2 CPU production seam. It accepts only a provenance-
matching `training` episode manifest and trains every declared development seed.
The candidate is `shared_ticker_value`; `independent_actor` and `shared_critic`
use the identical command and checkpoint contract as preregistered ablations.
Each actor emits only `skip` or `enter`. Trend Magic direction, the deterministic
2R/0.75R exit, the fixed mini/micro profile, and the Topstep risk shield remain
outside actor authority. Training uses episodic PPO-Lagrangian with separate
ticker-owned reward and BLOW-cost critics. PASS reward and BLOW cost are binary
terminal signals with gamma 1.0; minimum MLL cushion remains an observation and
logged path metric, never a dense cost. Reward and cost advantages are
standardized independently per ticker, and the actor uses
`A_reward - lambda*A_cost`. The projected multiplier starts at 1.0, uses a 0.01
cost limit and 0.01 learning rate, and is capped at 100. A run
freezes rollout log probabilities and both values before four configured PPO
epochs. Every minibatch contains exactly equal samples per present ticker,
oversampling shorter streams when legal decision counts differ. The
independent-actor ablation owns separate profile embeddings, trunks, and heads
per ticker. A run checkpoints only after a complete schedule cycle and resumes
optimizer, both critic families, ticker-owned return statistics, multiplier and
raw cost statistics, RNG state, last rollout evidence, and exact source/lock/
config/schedule/collection/artifact identities. Training mode and requested
update/timestep budgets are also run identities, so bounded and production runs
cannot share checkpoints. Immutable bundles are recovered if a crash occurs
before the atomic state pointer or final metrics publication. Old unconstrained
checkpoints and published or mismatched runs are rejected. The sealed holdout
is never a training input.
