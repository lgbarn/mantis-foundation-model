# MantisV2 Training

This package trains MantisV2 on the FFM NextLeg objective while preserving a strict boundary around the official upstream model.

## Authoritative inputs

- Semantics: MantisV2 paper, arXiv 2602.17868.
- Code: `vfeofanov/mantis` tag `v1.0.0`, commit `0c94f8ceb9f1d1421dd292ed917090df8c31605b`.
- Weights: `paris-noah/MantisV2`, revision `99fe0f548960e272fbfa4b82fd9b5b5956779dfd`.
- Weight SHA-256: `49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1`.

The adapter verifies the immutable Hub revision and weight digest before using pretrained weights. Upstream license declarations conflict between Apache-2.0 and MIT, so this repository does not redistribute upstream source or weights.

## Data contract

The production config expects 36 CSV streams under `/Volumes/Storage/trading-research/data/FFM_NEXTLEG`: nine roots (`ES`, `NQ`, `RTY`, `YM`, `GC`, `SI`, `CL`, `ZB`, `ZN`) at four intervals (`1min`, `3min`, `5min`, `15min`). Every file must provide sorted, unique UTC timestamps and finite `open,high,low,close,volume` values.

Verified on 2026-07-18: 25,057,097 rows spanning 2021-07-14 through 2026-07-13, with 1,967,568 legal train anchors, 224,519 validation anchors, and 294,781 holdout anchors. Full inspection took about 91 seconds on the local Apple Silicon host.

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
just inspect-data
just probe-mps
just train mantis-v2/configs/nextleg.toml
just evaluate mantis-v2/configs/nextleg.toml
just export mantis-v2/configs/nextleg.toml
```

`gate` includes deterministic synthetic smoke training, evaluation, checkpoint, and export parity. `verify-upstream` downloads or reuses the pinned official weights, checks their digest, constructs the real upstream MantisV2 on CPU, and verifies its embedding contract. `inspect-data` validates all configured files and computes legal anchor counts. `probe-mps` is guarded to exactly one real-data train batch and one validation batch of 36, so every configured stream is represented once. Production training starts from the pinned official MantisV2 weights and runs on Apple MPS.

## Training recipe

This is supervised NextLeg fine-tuning, not a reproduction of MantisV2 contrastive foundation pretraining. It combines the strongest compatible parts of the authoritative sources:

- Upstream MantisV2: official pretrained weights, linear resizing to 512, channel-independent encoding, concatenated 256-dimensional embeddings, full-model fine-tuning, AdamW weight decay `0.05`, 10-epoch linear warmup, and cosine decay.
- FFM NextLeg: 120 epochs, 200 sampled training batches per epoch, 20 validation batches, learning rate `0.0001188117389055629`, horizons 5/10/20/25, two pivot legs, cap 256, and validation-loss model selection.
- Local MPS: batch 128 and zero data-loader workers. A local throughput sweep showed batch 192 was fastest, but 128 preserves memory headroom while the 1.4 GiB corpus and optimizer state are resident.

Training stops after eight epochs without validation-total improvement. `metrics.json` records the learning rate plus train and validation total, candle MSE, and leg SmoothL1 for every completed epoch. These regression checkpoints do not report AUC or log loss; those belong to the later 3-minute classifier stage.

## Artifacts and resume

Outputs live under the configured external `artifact_root`:

```text
/Volumes/Storage/trading-research/artifacts/mantis-foundation-model/mantisv2-nextleg-mps-target-clamp-v2/
|-- checkpoints/latest.pt
|-- checkpoints/best.pt
|-- metrics.json
|-- provenance.json
|-- train-result.json
|-- evaluation.json
`-- export/
    |-- model.safetensors
    `-- manifest.json
```

Native checkpoints include model and optimizer state, epoch, global step, Python, NumPy, CPU Torch, CUDA, and MPS RNG state, config digest, full dataset content identity, training-source content digest, Git state, dependency-lock digest, and pinned upstream identities. Resume fails closed if any load-bearing identity changes, including uncommitted source content. Each epoch uses a pending checkpoint that is promoted only after metric history is durable, so an interrupted two-file update falls back to the prior valid epoch.

`run.name` creates the final directory below `run.artifact_root`. A non-resume production run refuses to replace an existing named run unless `run.allow_overwrite` is explicitly enabled. Resume uses `latest.pt`; evaluation and export use validation-selected `best.pt`. The smoke and guarded probe configs enable overwrite only for disposable verification artifacts.

The earlier `mantisv2-nextleg-mps` run is preserved as a superseded diagnostic artifact. It completed 12 epochs before investigation found that unclamped future normalization could amplify a one-tick move in a constant-price window into a target of 15,625. Because correcting the target changes both source and config provenance, that checkpoint is intentionally not resumable by the corrected production config.

The existing `Futures-Foundation-Model/checkpoints/mantis_ssl_ctr_seq2seq.pt` is a legacy Mantis checkpoint with `tokgen_unit`/`vit_unit` keys. It is not compatible with the MantisV2 architecture and is deliberately not loaded.

## Production notes

- The production config explicitly requires Apple MPS and fails rather than falling back to CPU.
- The internal disk is nearly full, so production checkpoints and metrics are written to `/Volumes/Storage`.
- No experiment-tracking service is enabled. Metrics and provenance are local JSON artifacts.
- The normal evaluator refuses holdout access. A release-specific config and review are required before a one-time 2026 holdout evaluation.
- Full data hashing reads all 36 CSVs before training. This is intentional and binds checkpoints to exact input content.
