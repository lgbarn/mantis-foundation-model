# Mantis Foundation Model

This repository builds and trains Mantis-family time-series foundation models. MantisV2 is the first and only active model family; `mantis/` and `mantis-plus/` will be introduced only when their independent contracts are scoped.

## Status

The MantisV2 NextLeg pipeline is implemented and locally verified. The full 36-stream corpus passes its data contract with 1,967,568 train anchors, 224,519 validation anchors, and 294,781 isolated 2026 holdout anchors. See `mantis-v2/README.md` for the model-specific workflow, `docs/training-guide.html` for the operator guide, and `AGENTS.md` for repository rules.

## Quick start

```bash
just sync
just gate
just inspect-data
just verify-upstream
```

Production training is intentionally separate from smoke verification:

```bash
just train mantis-v2/configs/nextleg.toml
```

`just gate` includes the complete synthetic smoke workflow. The production config reads external data in place and writes ignored local artifacts. It never copies raw market data into this repository.
