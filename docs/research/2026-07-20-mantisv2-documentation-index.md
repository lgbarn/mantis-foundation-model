# MantisV2 documentation index

Inventory date: 2026-07-20

This is the companion index for the compiled MantisV2 documentation dossier.
It distinguishes official upstream facts, local implementation decisions,
completed-run evidence, and proposed work. Model weights, upstream source code,
and restricted presentation pages are not redistributed.

## Evidence labels

| Label | Meaning |
| --- | --- |
| Upstream | Published by the MantisV2 authors or project maintainers |
| Local implemented | Present in this repository's current code or configuration |
| Run evidence | Recorded by a completed, provenance-bound local run |
| Specification | Accepted intended behavior that may not be implemented yet |
| Research | Recommendation or comparison, not implementation evidence |
| Superseded | Preserved for history but not a current production authority |

## Recommended reading order

1. `README.md`
2. `docs/mantis-family.md`
3. `docs/workflow.md`
4. `mantis-v2/README.md`
5. `docs/architecture.md`
6. `docs/setup-and-hardware.md`
7. `docs/agent-runbook.md`
8. `docs/troubleshooting.md`
9. The current run record and accepted ADRs
10. The official MantisV2 paper, pinned code, and V2 notebooks

## Canonical external sources

### MantisV2

| Source | Status | URL |
| --- | --- | --- |
| MantisV2 technical report, arXiv:2602.17868v1 | Upstream, CC BY 4.0 | https://arxiv.org/abs/2602.17868v1 |
| MantisV2 HTML report | Upstream | https://arxiv.org/html/2602.17868v1 |
| MantisV2 PDF | Upstream | https://arxiv.org/pdf/2602.17868v1 |
| ICLR 2026 TSALM OpenReview entry | Upstream, link only | https://openreview.net/forum?id=W03ApbRV4i |
| Workshop accepted-paper index | Upstream | https://tsalm-workshop.github.io/accepted-papers/ |
| Author publication index | Upstream | https://vfeofanov.github.io/publications.html |
| Institutional publication page | Upstream | https://www.42.com/publications/closing-the-zero-shot-gap-in-time-series-classification-with-synthetic-data-and-test-time-strategies |
| MantisV2 presentation, 52 pages | Upstream, proprietary/restricted, link only | https://vfeofanov.github.io/pdf/slides/2026-tabtalk_mantis_v2.pdf |

No official MantisV2 video, public recording, separate poster, generated API
site, or author-written tutorial blog was verified as of the inventory date.

### Pinned code and package

| Source | Status | URL |
| --- | --- | --- |
| GitHub tag `v1.0.0` | Upstream | https://github.com/vfeofanov/mantis/tree/v1.0.0 |
| Pinned source commit `0c94f8c...` | Upstream | https://github.com/vfeofanov/mantis/commit/0c94f8ceb9f1d1421dd292ed917090df8c31605b |
| Root README | Upstream | https://github.com/vfeofanov/mantis/blob/v1.0.0/README.md |
| MantisV2 architecture source | Upstream | https://github.com/vfeofanov/mantis/blob/v1.0.0/src/mantis/architecture/version2.py |
| Trainer source | Upstream | https://github.com/vfeofanov/mantis/blob/v1.0.0/src/mantis/trainer/trainer.py |
| Package dependencies | Upstream | https://github.com/vfeofanov/mantis/blob/v1.0.0/pyproject.toml |
| Backward compatibility guide | Upstream | https://github.com/vfeofanov/mantis/blob/v1.0.0/BACKWARD_COMPATIBILITY.md |
| Repository license text | Upstream | https://github.com/vfeofanov/mantis/blob/v1.0.0/LICENSE |
| PyPI `mantis-tsfm` 1.0.0 | Upstream | https://pypi.org/project/mantis-tsfm/1.0.0/ |

There is no GitHub v1.0.0 Release page, no v1.0.0 changelog entry, no repository
wiki, and no upstream `docs/` directory. The README, notebooks, source docstrings,
tests, and package metadata are the operative API documentation.

### Official notebooks

| Notebook | Direct MantisV2? | Purpose |
| --- | --- | --- |
| https://github.com/vfeofanov/mantis/blob/v1.0.0/getting_started/intermediate_layers.ipynb | Yes | All six layers and CLS versus combined aggregation |
| https://github.com/vfeofanov/mantis/blob/v1.0.0/getting_started/single_channel_fine_tune.ipynb | Yes | Head-only and full single-channel fine-tuning |
| https://github.com/vfeofanov/mantis/blob/v1.0.0/getting_started/multichannel_fine_tune.ipynb | Yes | Independent-channel and adapter workflows |
| https://github.com/vfeofanov/mantis/blob/v1.0.0/getting_started/single_channel_extract_feats.ipynb | No, legacy V1 | Frozen-feature workflow concept |
| https://github.com/vfeofanov/mantis/blob/v1.0.0/getting_started/multichannel_extract_feats.ipynb | No, legacy V1 | Multichannel feature extraction concept |
| https://github.com/vfeofanov/mantis/blob/v1.0.0/getting_started/multichannel_adapters.ipynb | No, legacy V1 | PCA, channel selection, and learnable adapters |
| https://github.com/vfeofanov/mantis/blob/v1.0.0/getting_started/pretrain.py | No, legacy V1 demo | Family-level contrastive pretraining example, not the V2 production recipe |

### Checkpoint

| Source | Status | URL |
| --- | --- | --- |
| MantisV2 model page | Upstream | https://huggingface.co/paris-noah/MantisV2 |
| Pinned revision `99fe0f5...` | Upstream | https://huggingface.co/paris-noah/MantisV2/tree/99fe0f548960e272fbfa4b82fd9b5b5956779dfd |
| Model card | Upstream, incomplete | https://huggingface.co/paris-noah/MantisV2/blob/99fe0f548960e272fbfa4b82fd9b5b5956779dfd/README.md |
| Serialized configuration | Upstream | https://huggingface.co/paris-noah/MantisV2/blob/99fe0f548960e272fbfa4b82fd9b5b5956779dfd/config.json |
| Weight artifact listing | Upstream, link only | https://huggingface.co/paris-noah/MantisV2/blob/99fe0f548960e272fbfa4b82fd9b5b5956779dfd/model.safetensors |

The pinned weight file is 16,771,648 bytes with SHA-256
`49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1`.
The Hub card is only 455 bytes and says more information is needed.

### CauKer pretraining sources

| Source | Status | URL |
| --- | --- | --- |
| CauKer 2M dataset card | Upstream, Apache-2.0 declaration | https://huggingface.co/datasets/paris-noah/CauKer2M |
| CauKer paper, arXiv:2508.02879v3 | Upstream, CC BY-NC-SA 4.0 | https://arxiv.org/abs/2508.02879v3 |
| CauKer code repository | Upstream, no verified repository license | https://github.com/ShifengXIE/CauKer |
| CauKer tutorial notebook | Upstream | https://github.com/ShifengXIE/CauKer/blob/main/CauKer.ipynb |
| CauKer generator source | Upstream | https://github.com/ShifengXIE/CauKer/blob/main/CauKer.py |

MantisV2 and Mantis+ use 2,000,000 CauKer synthetic sequences. The paper reports
200 pretraining epochs. The exact production optimizer, hardware, seed schedule,
filtering, and checkpoint-selection manifest is not publicly documented.

### Mantis family context

| Source | Status | URL |
| --- | --- | --- |
| Original Mantis paper, arXiv:2502.15637v2 | Upstream, CC BY 4.0 | https://arxiv.org/abs/2502.15637v2 |
| Mantis+ checkpoint | Upstream, comparison only | https://huggingface.co/paris-noah/MantisPlus |

Mantis uses the V1 architecture and an earlier real/public pretraining corpus.
Mantis+ uses the V1 architecture and CauKer 2M. MantisV2 uses the V2 architecture
and CauKer 2M. They are related but are not interchangeable checkpoints.

## Official model contract

- Input to a direct forward pass: `(batch, 1, time)`.
- Time length: positive multiple of 32; official examples resize to 512.
- Multichannel behavior: encode each channel independently, then concatenate.
- Default model: 6 transformer layers, hidden size 256, 8 heads, head dimension
  32, MLP dimension 512, dropout 0.1, 32 patches, convolution kernel 41.
- Default checkpoint output: last-layer CLS token, 256 dimensions per channel.
- Highlighted frozen-feature recipe: layer index 2 with combined CLS and mean
  tokens, 512 dimensions per channel.
- Parameter count: 4,188,672 for the full model; 2,214,144 when returning at
  layer index 2.
- Attention is non-causal. Trading causality must therefore be enforced by
  preventing future values from entering each input window.

## Local documentation

### Core guides

| Path | Authority |
| --- | --- |
| `README.md` | Repository status and navigation |
| `mantis-v2/README.md` | Package command and artifact reference |
| `docs/mantis-family.md` | Novice-oriented model-family explanation |
| `docs/architecture.md` | Component, leakage, provenance, and stage architecture |
| `docs/workflow.md` | Canonical human operating sequence |
| `docs/agent-runbook.md` | Safe deterministic agent protocol |
| `docs/setup-and-hardware.md` | Dependencies and CPU/MPS/CUDA differences |
| `docs/troubleshooting.md` | Fail-closed diagnosis and recovery |
| `mantis-v2/docs/mps-training.md` | Qualified Apple MPS training contract |
| `mantis-v2/docs/corpus-repair.md` | Contract-aware futures corpus construction |

### Accepted architecture decisions

| ADR | Decision |
| --- | --- |
| `docs/adr/0001-pin-and-isolate-mantis-v2.md` | Pin paper, source, checkpoint, and preserve license conflict |
| `docs/adr/0002-nextleg-split-safety.md` | Preserve complete causal NextLeg targets within every split |
| `docs/adr/0003-reuse-embeddings-for-isolated-head-runs.md` | Reuse immutable embeddings for isolated downstream head experiments |
| `docs/adr/0004-rebuild-contract-aware-futures-corpus.md` | Replace stitched continuous CSV with repaired contract-aware Parquet |

### Completed run records

| Path | Status |
| --- | --- |
| `docs/runs/2026-07-19-mantisv2-nextleg-mps-target-clamp-v2.md` | Historical and diagnostic after corpus contamination was found |
| `docs/runs/2026-07-20-mantisv2-nextleg-parquet-v2.md` | Current repaired-corpus foundation evidence |

The current run uses 27 streams and 40,251,760 rows. Training stopped after 32
epochs and selected epoch 24. Evaluation and safetensors parity passed. The
subsequent Trend Magic heads converged but failed constant-baseline log-loss and
Brier gates, so simulation was not run and the holdout remained sealed.

### Research and specifications

| Path | Status |
| --- | --- |
| `docs/research/2026-07-18-time-series-foundation-models-for-trading.md` | Comparative TSFM research |
| `docs/research/2026-07-19-mantisv2-supertrend-topstep-spec.md` | Historical Supertrend downstream specification |
| `docs/research/2026-07-20-trend-magic-spec.md` | Current implemented Trend Magic contract; rejected head |
| `docs/research/2026-07-20-foundation-model-reinforcement-learning-futures.md` | FM plus RL literature review and recommendation |
| `docs/research/2026-07-20-mantisv2-topstep-rl-entry-spec.md` | Accepted RL design; only supporting infrastructure is implemented |
| `docs/mantisv2-official-pipeline-audit.html` | Upstream-to-local pipeline comparison; review status varies with Git state |

## Current trading configuration

The current foundation definition is
`mantis-v2/configs/nextleg-parquet-v2.toml`:

- Apple MPS required.
- Nine symbols: ES, NQ, RTY, YM, GC, SI, CL, ZB, and ZN.
- Intervals: 1-minute, 3-minute, and 15-minute.
- OHLCV channels encoded independently and concatenated.
- Contexts: 64, 100, 150, and 200 bars; resized to 512.
- Training mode: `transformer_finetune`.
- Batch 128, learning rate `0.0001188117389055629`, weight decay 0.05.
- Ten warmup epochs, cosine decay, 200 train and 20 validation batches per
  epoch, patience 8.
- NextLeg horizons: 5, 10, 20, and 25 bars.
- Context-only normalization with clamp 10.
- Sealed holdout begins 2026-01-01.
- Safetensors export must pass numerical parity.

The layer-2 combined-token recipe, multi-length/difference self-ensembling, raw
feature comparisons, and PPO strategy training are proposed isolated
experiments. They are not current production behavior.

## Hardware and implementation boundary

| Workflow | CPU | Apple MPS | CUDA |
| --- | --- | --- | --- |
| Foundation smoke/verification | Implemented | Implemented | Foundation path exists |
| Foundation production | Not qualified | Qualified | Not qualified |
| Downstream embeddings | Supported fallback | Qualified | Not supported |
| Logistic/walk-forward/simulation | Qualified CPU path | Not the primary target | Not supported as a qualified path |
| RL infrastructure | CPU contract | Not qualified | Not qualified |
| PPO policy training | Not implemented | Not implemented | Not implemented |

## Contradictions and limitations

1. Upstream code/package/checkpoint surfaces conflict between Apache-2.0 and
   MIT. Do not claim a unified license or redistribute weights under one.
2. The exact MantisV2 production pretraining run configuration is unpublished.
3. The Hub default is final-layer CLS, while the highlighted zero-shot recipe is
   layer 2 plus combined CLS/mean.
4. Some local prose says full fine-tuning; current config and code use
   transformer-only fine-tuning.
5. The old 36-stream CSV lineage is diagnostic after cross-contract
   contamination. The current lineage is repaired 27-stream Parquet.
6. MantisV2 is a classification/embedding model, not a trading strategy,
   forecaster, or RL policy.
7. The local RL specification is ahead of implementation: config, provenance,
   account replay, and episode scheduling exist; environment, PPO training,
   evaluation, and export do not.
8. Topstep rules and fees are dated experiment inputs, not timeless statements
   of current external policy.
9. CPU and CUDA foundation production are not qualified end to end.
10. No public paper establishes MantisV2 as profitable for intraday futures.

## Redistribution notes

- The MantisV2 and current original-Mantis arXiv reports declare CC BY 4.0.
- CauKer declares CC BY-NC-SA 4.0.
- The presentation explicitly states proprietary/restricted distribution and is
  linked only.
- The OpenReview manuscript is public but has no verified reuse license and is
  linked only.
- Upstream source and model weights are linked only because their license
  declarations conflict.

