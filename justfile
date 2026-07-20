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

train config:
    uv run mantis-v2 train --config {{config}}

evaluate config:
    uv run mantis-v2 evaluate --config {{config}}

export config:
    uv run mantis-v2 export --config {{config}}

validated-export config:
    uv run mantis-v2 validated-export --config {{config}}

downstream-prepare config:
    uv run mantis-v2 downstream-prepare --config {{config}}

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

downstream-holdout config unlock="":
    uv run mantis-v2 downstream-holdout --config {{config}} --unlock {{unlock}}

rl-dry-run config="mantis-v2/configs/rl-entry-smoke.toml":
    uv run mantis-v2 rl-dry-run --config {{config}}
