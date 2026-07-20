# MantisV2 MPS training contract

## Qualified production path

Apple MPS production training uses:

- config: `mantis-v2/configs/nextleg-parquet-v2.toml`
- model mode: `transformer_finetune`
- corpus: `MANTIS_NEXTLEG_PARQUET_V1`
- corpus span: 2021-07-14 through 2026-07-13
- timeframes: 1 minute, 3 minutes, and 15 minutes
- development holdout start: 2026-01-01 UTC
- batch size: 128
- optimizer steps per epoch: 200
- warmup: 10 epochs, adjusted on every optimizer update
- checkpoint cadence: every completed epoch
- selection metric: validation total NextLeg loss

`transformer_finetune` freezes every upstream encoder component except the
MantisV2 transformer, then trains that transformer plus the local candle and
NextLeg heads. The export still contains the complete encoder, and downstream
embedding loads the adapted transformer weights. This is different from `head_only` and
`adapter_head`, whose frozen encoder would produce the original upstream
embeddings in the current downstream pipeline.

The upstream Mantis package documents CPU and CUDA devices. It does not claim
MPS training support. MPS support in this repository is therefore a locally
qualified execution mode, not an upstream guarantee. See the pinned
[Mantis trainer](https://github.com/vfeofanov/mantis/blob/0c94f8ceb9f1d1421dd292ed917090df8c31605b/mantis/trainer/trainer.py)
and [per-step scheduler](https://github.com/vfeofanov/mantis/blob/0c94f8ceb9f1d1421dd292ed917090df8c31605b/mantis/trainer/trainer_utils/scheduling.py).

## Required commands

Run these commands from the repository root, in order:

```bash
just inspect-data mantis-v2/configs/nextleg-parquet-v2.toml
uv run mantis-v2 probe --config mantis-v2/configs/nextleg-parquet-v2-probe.toml
just train mantis-v2/configs/nextleg-parquet-v2.toml
just validated-export mantis-v2/configs/nextleg-parquet-v2.toml
```

The probe is a 32-update optimizer stress test followed by one real validation
batch. A one-update probe is not sufficient for MPS qualification.

## 2026-07-20 non-finite gradient incident

The failed run is preserved at:

```text
/Volumes/Storage/trading-research/artifacts/mantis-foundation-model/mantisv2-nextleg-parquet-v1
```

It contains corpus/provenance evidence but no metric history or checkpoint. It
must not be resumed, overwritten, or presented as a trained model.

Observed reproductions:

- batch 128 with the former epoch-level warmup: finite step 1, NaN gradient at
  step 5
- batch 128 with the pinned upstream per-step warmup: NaN gradient at step 25
- batch 32 with per-step warmup: NaN gradient at step 11
- batch 128 with per-step warmup and the token generator frozen: all 200 steps
  completed; train total 3.07369110, candle 2.50625273, leg 0.56743838

In every failing reproduction, inputs, targets, forward outputs, and the current
loss were finite. The first non-finite tensor was the gradient of
`tokgen_unit.convs.1.conv.weight`. Replaying the first five batches without
optimizer updates remained finite, proving that the reviewed market windows
were not the direct cause.

### Five Whys

1. Training stopped because the reported training loss became non-finite.
2. The loss became non-finite because an earlier optimizer update corrupted an
   encoder parameter.
3. The optimizer received corrupted input because a token-generator convolution
   gradient became NaN on MPS.
4. The corrupted update was allowed because gradient clipping did not reject a
   non-finite norm before `optimizer.step()`.
5. Production reached this condition because the former probe covered only one
   optimizer update and the local scheduler adjusted warmup once per epoch
   instead of once per update.

The process fixes are permanent:

- learning rate warmup and cosine decay are computed for every optimizer step
- a non-finite gradient norm fails before the optimizer can mutate parameters
- MPS probes cover 32 optimizer updates
- MPS production uses the provenance-bound 32-step probe and the
  `transformer_finetune` policy selected by the 200-step diagnostic replay
- every changed training semantic requires a new config digest and run identity

## Metrics

Foundation training reports total, candle, and leg losses for training and
validation. AUC is not defined for these regression heads. Downstream
walk-forward classification reports weighted log loss and weighted Brier score
as primary metrics, with ROC AUC and PR AUC as diagnostics.
