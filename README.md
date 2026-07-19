# Mantis Foundation Model

This repository builds and trains Mantis-family time-series foundation models. MantisV2 is the first and only active model family; `mantis/` and `mantis-plus/` will be introduced only when their independent contracts are scoped.

## Status

The MantisV2 NextLeg foundation pipeline and the config-driven 1m/3m/15m Supertrend downstream pipeline are implemented. The downstream workflow uses every eligible closed 3-minute state bar, purged walk-forward logistic probes, and a separate Topstep 100K Combine simulator. See `mantis-v2/README.md` for commands, `docs/training-guide.html` for the operator guide, and `AGENTS.md` for repository rules.

## Quick start

```bash
just sync
just gate
just inspect-data
just verify-upstream
just probe-mps
```

Production training is intentionally separate from smoke verification:

```bash
just train mantis-v2/configs/nextleg.toml
```

After the validated NextLeg export exists, run the downstream pipeline:

```bash
just downstream-run mantis-v2/configs/supertrend-topstep-100k.toml
```

`just gate` includes the complete synthetic foundation smoke workflow and downstream unit contracts. `just probe-mps` performs one real-data training batch and one validation batch with the pinned pretrained MantisV2 model. Production uses Apple MPS and writes checkpoints, Parquet tables, embedding shards, predictions, and simulation results to the external drive. It never copies raw market data into this repository.
