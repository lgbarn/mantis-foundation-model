set dotenv-load := false

default:
    @just --list

sync:
    uv sync --all-packages
    uv run tensorboard --version

format:
    uv run ruff format .

format-check:
    uv run ruff format --check .

lint:
    uv run ruff check .

typecheck:
    uv run mypy

test:
    uv run pytest

gate: format-check lint typecheck test smoke downstream-smoke downstream-trend-magic-smoke

inspect-data config:
    uv run mantis-v2 inspect-data --config {{config}}

smoke:
    uv run mantis-v2 smoke --config mantis-v2/configs/smoke.toml

probe-mps:
    uv run mantis-v2 probe --config mantis-v2/configs/nextleg-parquet-v2-probe.toml

verify-upstream:
    uv run mantis-v2 verify-upstream --config mantis-v2/configs/nextleg.toml

repair-corpus config="mantis-v2/configs/corpus-repair-v1.toml":
    uv run mantis-v2 repair-corpus --config {{config}}

validate-corpus config="mantis-v2/configs/corpus-repair-v1.toml":
    uv run mantis-v2 validate-corpus --config {{config}}

runpod-image-build image:
    infra/runpod/scripts/build-image.sh {{quote(image)}}

runpod-image-self-check image output:
    infra/runpod/scripts/self-check-image.sh {{quote(image)}} {{quote(output)}}

runpod-image-scan image output:
    infra/runpod/scripts/scan-image.sh {{quote(image)}} {{quote(output)}}

runpod-plan platform local experiment intent inventory ledger evaluated_at output:
    uv run mantis-v2 runpod-plan --platform {{quote(platform)}} --local {{quote(local)}} --experiment {{quote(experiment)}} --intent {{quote(intent)}} --inventory {{quote(inventory)}} --ledger {{quote(ledger)}} --evaluated-at {{quote(evaluated_at)}} --output {{quote(output)}}

runpod-plan-authorized platform local experiment intent inventory ledger authorization evaluated_at output:
    uv run mantis-v2 runpod-plan --platform {{quote(platform)}} --local {{quote(local)}} --experiment {{quote(experiment)}} --intent {{quote(intent)}} --inventory {{quote(inventory)}} --ledger {{quote(ledger)}} --authorization {{quote(authorization)}} --evaluated-at {{quote(evaluated_at)}} --output {{quote(output)}}

train config:
    uv run mantis-v2 train --config {{config}}

tensorboard run_root port="6006":
    uv run mantis-v2 tensorboard --run-root {{quote(run_root)}} --host 127.0.0.1 --port {{port}}

evaluate config:
    uv run mantis-v2 evaluate --config {{config}}

export config:
    uv run mantis-v2 export --config {{config}}

validated-export config:
    uv run mantis-v2 validated-export --config {{config}}

downstream-prepare config:
    uv run mantis-v2 downstream-prepare --config {{config}}

downstream-verify config:
    uv run mantis-v2 downstream-verify --config {{config}}

downstream-embed config:
    uv run mantis-v2 downstream-embed --config {{config}}

downstream-walk-forward config:
    uv run mantis-v2 downstream-walk-forward --config {{config}}

downstream-simulate config:
    uv run mantis-v2 downstream-simulate --config {{config}}

downstream-run config:
    uv run mantis-v2 downstream-run --config {{config}}

downstream-smoke:
    uv run mantis-v2 downstream-smoke --config mantis-v2/configs/downstream-smoke.toml

downstream-trend-magic-smoke:
    uv run mantis-v2 downstream-smoke --config mantis-v2/configs/downstream-trend-magic-smoke.toml

trend-magic-verify config="mantis-v2/configs/trend-magic-topstep-100k.toml":
    uv run mantis-v2 downstream-verify --config {{config}}

trend-magic-prepare config="mantis-v2/configs/trend-magic-topstep-100k.toml":
    uv run mantis-v2 downstream-prepare --config {{config}}

trend-magic-embed config="mantis-v2/configs/trend-magic-topstep-100k.toml":
    uv run mantis-v2 downstream-embed --config {{config}}

trend-magic-head config="mantis-v2/configs/trend-magic-topstep-100k-head-c0001-v2.toml":
    uv run mantis-v2 downstream-walk-forward --config {{config}}

downstream-holdout config unlock="":
    uv run mantis-v2 downstream-holdout --config {{config}} --unlock {{unlock}}

rl-dry-run config="mantis-v2/configs/rl-entry-smoke.toml":
    uv run mantis-v2 rl-dry-run --config {{config}}
rl-build-episodes config fold partition episodes:
    uv run mantis-v2 rl-build-episodes --config {{config}} --fold {{fold}} --partition {{partition}} --episodes {{episodes}}

rl-account-replay input output config="mantis-v2/configs/rl-entry-smoke.toml":
    uv run mantis-v2 rl-account-replay --config {{config}} --input {{input}} --output {{output}}

rl-validate-environment training_manifest validation_manifest output config="mantis-v2/configs/rl-entry-smoke.toml":
    uv run mantis-v2 rl-validate-environment --config {{config}} --training-manifest {{training_manifest}} --validation-manifest {{validation_manifest}} --output {{output}}

rl-smoke output config="mantis-v2/configs/rl-entry-smoke.toml" resume="":
    uv run mantis-v2 rl-smoke --config {{config}} --output {{output}} {{resume}}
