set dotenv-load := false

default:
    @just --list

sync:
    uv sync --all-packages

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

gate: format-check lint typecheck test smoke downstream-smoke

inspect-data config="mantis-v2/configs/nextleg.toml":
    uv run mantis-v2 inspect-data --config {{config}}

smoke:
    uv run mantis-v2 smoke --config mantis-v2/configs/smoke.toml

probe-mps:
    uv run mantis-v2 probe --config mantis-v2/configs/nextleg-mps-probe.toml

verify-upstream:
    uv run mantis-v2 verify-upstream --config mantis-v2/configs/nextleg.toml

train config="mantis-v2/configs/nextleg.toml":
    uv run mantis-v2 train --config {{config}}

evaluate config="mantis-v2/configs/nextleg.toml":
    uv run mantis-v2 evaluate --config {{config}}

export config="mantis-v2/configs/nextleg.toml":
    uv run mantis-v2 export --config {{config}}

validated-export config="mantis-v2/configs/nextleg.toml":
    uv run mantis-v2 validated-export --config {{config}}

downstream-prepare config="mantis-v2/configs/supertrend-topstep-100k.toml":
    uv run mantis-v2 downstream-prepare --config {{config}}

downstream-embed config="mantis-v2/configs/supertrend-topstep-100k.toml":
    uv run mantis-v2 downstream-embed --config {{config}}

downstream-walk-forward config="mantis-v2/configs/supertrend-topstep-100k.toml":
    uv run mantis-v2 downstream-walk-forward --config {{config}}

downstream-simulate config="mantis-v2/configs/supertrend-topstep-100k.toml":
    uv run mantis-v2 downstream-simulate --config {{config}}

downstream-run config="mantis-v2/configs/supertrend-topstep-100k.toml":
    uv run mantis-v2 downstream-run --config {{config}}

downstream-smoke:
    uv run mantis-v2 downstream-smoke --config mantis-v2/configs/downstream-smoke.toml

downstream-holdout config="mantis-v2/configs/supertrend-topstep-100k.toml" unlock="":
    uv run mantis-v2 downstream-holdout --config {{config}} --unlock {{unlock}}
