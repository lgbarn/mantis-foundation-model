# MantisV2 NextLeg MPS production run

## Record status

- Run: `mantisv2-nextleg-mps-target-clamp-v2`
- Status: Complete; historical pre-mandatory-gate artifact
- Device: Apple MPS
- Training outcome: Early stopped
- Artifact source revision: `cc30dcd0a0034140cde0f2ed3b9f35db7f9f3361`
- Artifact source state: Dirty, as recorded in the export manifest

This record documents an existing run. It does not authorize resuming,
overwriting, regenerating, or modifying the artifact.

## Training result

| Field | Value |
| --- | ---: |
| Epochs completed | 102 |
| Last epoch index | 101 |
| Last global step | 20,400 |
| Early stopped | Yes |
| Best epoch index | 93 |
| Best human epoch number | 94 |
| Best global step | 18,800 |
| Best validation total | 2.269178491830826 |
| Best validation candle | 1.9975997865200044 |
| Best validation leg | 0.2715787053108215 |
| Last validation total | 2.2700400352478027 |

## Evaluation result

Evaluation used the validation split and reproduced the best metrics across 20
batches.

| Field | Value |
| --- | ---: |
| Eligible validation anchors | 224,519 |
| Evaluated batches | 20 |
| Total loss | 2.269178491830826 |
| Candle MSE | 1.9975997865200044 |
| Leg SmoothL1 | 0.2715787053108215 |

The 20 batches are the configured fixed validation subset. The result is not a
full-validation-corpus evaluation and is not a holdout or trading result.

## Corpus identity

- Streams: 36
- Symbols: ES, NQ, RTY, YM, GC, SI, CL, ZB, ZN
- Intervals: 1 minute, 3 minutes, 5 minutes, 15 minutes
- Total rows: 25,057,097
- Eligible train anchors: 1,967,568
- Eligible validation anchors: 224,519
- Dataset digest:
  `f26b1f7130f74559238cc3e0af724d66d0ada7a1e6b0a8d3c24e5c1617883463`
- Total bytes recorded across the 36 manifest file identities: 1,515,874,206

## Export result

| Field | Value |
| --- | --- |
| Format | safetensors |
| File size | 21,803,200 bytes |
| Tensor entries | 161 |
| Model SHA-256 | `e375cfdb9df0bfd86281468855150a6f95754f4b2b6ed7db42219ec6d309a675` |
| Parity verified | Yes |
| Absolute tolerance | `0.00001` |
| Relative tolerance | `0.0001` |

## Provenance

| Identity | Value |
| --- | --- |
| Config digest | `e5e49d5446d8d9c4c3030d51d2e5f36ea6e45d7d353e41c6b856d7887d2ce50` |
| Dataset digest | `f26b1f7130f74559238cc3e0af724d66d0ada7a1e6b0a8d3c24e5c1617883463` |
| Source revision | `cc30dcd0a0034140cde0f2ed3b9f35db7f9f3361` |
| Source dirty | `true` |
| Source digest | `b37f7aebc684c746f2c863b29cb0cfc1f294033499f4968ed7cdc394b712536c` |
| Lock digest | `cbd18c6f921482edd4d3a84b55620c67f9bb7f6953d5415a8f02317144c04aed` |
| Upstream source revision | `0c94f8ceb9f1d1421dd292ed917090df8c31605b` |
| Upstream Hub revision | `99fe0f548960e272fbfa4b82fd9b5b5956779dfd` |
| Upstream weight SHA-256 | `49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1` |

The repository's Git revision can move after a completed run. Reproduce or audit
this artifact against its recorded revision and source digest, not current HEAD.

## Historical schema limitation

This artifact was created before the repository added its mandatory
checkpoint-bound release gate.

The historical `evaluation.json` does not contain the later schema, pass,
checkpoint, and complete provenance fields. The historical export manifest does
not contain the later `validation_gate` block or a recorded exported-weight
digest. The safetensors SHA-256 above was computed separately from the immutable
file.

The recorded parity result and matching config/dataset identities remain valid
historical evidence. The artifact must not be rewritten to imitate the newer
schema, and it must not be reopened with later source.

## Artifact paths

- Training result:
  `/Volumes/Storage/trading-research/artifacts/mantis-foundation-model/mantisv2-nextleg-mps-target-clamp-v2/train-result.json`
- Evaluation:
  `/Volumes/Storage/trading-research/artifacts/mantis-foundation-model/mantisv2-nextleg-mps-target-clamp-v2/evaluation.json`
- Safetensors model:
  `/Volumes/Storage/trading-research/artifacts/mantis-foundation-model/mantisv2-nextleg-mps-target-clamp-v2/export/model.safetensors`
- Export manifest:
  `/Volumes/Storage/trading-research/artifacts/mantis-foundation-model/mantisv2-nextleg-mps-target-clamp-v2/export/manifest.json`

## Evidence boundary

This run proves that the historical MPS NextLeg adaptation completed, reproduced
its best validation metrics, and exported with recorded parity. It does not prove:

- Foundation holdout performance
- Downstream classifier performance
- Trading profitability
- CUDA or CPU production parity
- Publication or redistribution rights

The downstream workflow must treat the exact safetensors SHA-256 and historical
manifest as immutable inputs.
