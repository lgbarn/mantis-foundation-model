# MantisV2 repaired-Parquet production run

## Record status

- Foundation run: `mantisv2-nextleg-parquet-v2`
- Status: Complete and validated-exported
- Device: Apple MPS
- Corpus span: July 14, 2021 through July 13, 2026
- Fit boundary: January 1, 2026 holdout start
- Foundation source revision: `c98e00ce5a7a7242c4c635ec1de39f5d15d99812`
- Downstream Trend Magic prepare source revision: `9a03fa3`

The full repaired corpus spans July 2021 through July 2026. Model fitting and
routine downstream preparation stop before January 1, 2026 so January through
July 2026 remains a sealed out-of-sample holdout.

## Corpus identity

| Field | Value |
| --- | --- |
| Corpus | `MANTIS_NEXTLEG_PARQUET_V1` |
| Format | Parquet |
| Rows | 40,251,760 |
| Streams | 27 |
| Symbols | ES, NQ, RTY, YM, GC, SI, CL, ZB, ZN |
| Timeframes | 1m, 3m, 15m |
| Manifest SHA-256 | `2d8cf8b708c7c743c849059410b3654bc44f9fe4f7d795928de82526e120d703` |
| Training dataset digest | `84dd590ea5ae5b9e42acc1368060f88375bb48cc0a3ef3ba2e0c5d7c1739ae2a` |

## Training result

| Field | Value |
| --- | ---: |
| Epochs completed | 32 |
| Final global step | 6,400 |
| Early stopped | Yes |
| Best epoch index | 23 |
| Best human epoch number | 24 |
| Best global step | 4,800 |
| Best validation total | 2.4601407408714295 |
| Best validation candle | 2.1815908193588256 |
| Best validation leg | 0.2785499066114426 |
| Final validation total | 2.461732339859009 |

The primary foundation metric is validation loss. This is neither AUC nor a
trading-performance result.

## Validated export

| Field | Value |
| --- | --- |
| Checkpoint SHA-256 | `94ce00361894c2df818c158bcdc01b2c43463408f97e0dc22b4d4489c1c11b78` |
| Evaluation SHA-256 | `9398bcc37a34e50ed48fb010e22c5fa4537a5a8ce82c1ef4911ed109ff1b4c42` |
| Safetensors SHA-256 | `aab5bac1459a3cf0b08663d1d327e79ee3c1337dd51a30e16def53b8f563c1b9` |
| Absolute tolerance | `0.00001` |
| Relative tolerance | `0.0001` |
| Native/export parity | Verified |
| Validation gate | Verified |

## Trend Magic preparation

The first downstream stage used
`mantis-v2/configs/trend-magic-topstep-100k.toml` and the validated export above.

| Field | Value |
| --- | --- |
| Run | `mantisv2-trend-magic-topstep-100k` |
| Candidate semantics | Every eligible closed 3m Trend Magic state bar |
| Pre-holdout candidate rows | 3,259,736 |
| Holdout locked | Yes |
| Workflow digest | `bd0913b868f0e4b3fee5e1792371f7e5a0fa3cf3151162010fac1a6f72e82a44` |
| Contamination digest | `aba3ff00056b99810226482db554cc104673fe69af5e0d0e9be5299757117e64` |

Candidate rows by symbol:

| Symbol | Rows |
| --- | ---: |
| ES | 483,203 |
| NQ | 474,830 |
| RTY | 438,135 |
| YM | 462,259 |
| GC | 463,125 |
| CL | 447,300 |
| ZB | 490,884 |

## Trend Magic embedding

| Field | Value |
| --- | --- |
| Feature/metadata shard pairs | 400 |
| Rows | 3,259,736 |
| Feature width | 3,840 |
| Storage dtype | float16 |
| Embed manifest SHA-256 | `bffb74988497aec1c1c51821188e9c77b8a028c3016ddab30a843cbd785eea35` |
| Manifest row parity | Verified against prepare |
| File hashes and shapes | Verified for all 800 files |
| Finite samples | Verified at first, middle, and last row of every shard |

## Walk-forward convergence incident

The original run failed closed on fold 0 because its class-balanced LBFGS head
reached 500 of 500 iterations at `C=1.0`. The failure is preserved at
`mantisv2-trend-magic-topstep-100k/walk-forward/failure.json`; no embedding was
modified or discarded.

The operator-recorded exact fold-0 diagnostic used the production pipeline's
deterministic masks and caps: 25,000 train and 25,000 validation rows, each with
a 0.2498 positive rate. There were no zero-variance embedding dimensions.
Changing only `C` to `0.0001` converged in 56 iterations. Fold-0 validation was
still weak: weighted log loss 0.696498, weighted Brier 0.251644, and ROC AUC
0.504507. These observations diagnose the failure; they are not a pipeline
manifest or production qualification. The production walk-forward manifest is
the required durable evidence for all-fold results.

The replacement consumer config is
`mantis-v2/configs/trend-magic-topstep-100k-head-c0001-v2.toml`. It pins the
audited embed manifest and its producer config by SHA-256, preserves the failed
run, and writes head artifacts under a new run identity. Full walk-forward must
still pass convergence and proper-score gates before Topstep simulation can run.

The first consumer launch then exposed a provenance-reader defect before any
head fit: new embed manifests store the producer's current `workflow_digest`,
but the reuse guard accepted only `legacy_workflow_digest`. The two exact values
for this producer were different, so the valid pinned manifest was rejected.
The reader now accepts either explicitly supported producer digest while still
requiring the exact manifest SHA-256, producer-config SHA-256, embedding contract,
foundation weights, feature width, and row-count checks. Regression coverage
exercises both current and legacy manifests. No artifact was modified.

## Walk-forward result

The corrected consumer completed all eight chronological folds. Every LBFGS
head converged without warnings in 47-61 iterations, so the convergence gate
passed. The proper-score quality gate failed:

| Metric | Model mean | Constant baseline | Result |
| --- | ---: | ---: | --- |
| Class-balanced test log loss | 0.695306 | 0.693147 | Fail |
| Class-balanced test Brier | 0.251044 | 0.250000 | Fail |

Test ROC AUC ranged from 0.507711 to 0.518689 across folds. This is diagnostic
near-chance ranking and is consistent with the failed proper-score gate. The
walk-forward manifest SHA-256 is
`f0b73c480e56481827a50af1014e27d938c4ce4352174a724b1059543bd2ec73`.
All 16 recorded head and prediction files matched their manifest sizes and
SHA-256 values during the completion audit.

Topstep simulation was not run. The simulator requires both convergence and
quality gates to pass, so running it on these predictions would violate the
production workflow. The January-July 2026 holdout remains sealed and unused.

## Evidence boundary

This record proves repaired-corpus binding, completed foundation adaptation,
checkpoint-bound validation, safetensors parity, pre-holdout candidate
preparation, embedding completeness, and the completed pre-holdout walk-forward
rejection. It proves that this exact linear-head configuration does not beat the
constant proper-score baselines. It does not prove Topstep success,
sealed-holdout performance, or profitability.
