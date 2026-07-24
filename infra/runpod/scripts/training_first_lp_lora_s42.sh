#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
bundle="4d12c3f334dba95fb045905d89654f16d63e75b68f8423ae8726eac06d70fba6"
source_root="/workspace/mantis/inputs/${bundle}"
working_root="/tmp/mantis/inputs/${bundle}"
staged_marker="${working_root}/.training-first-staged"
direct_config="mantis-v2/configs/nextleg-runpod-cuda-3tf-direct-lora-s42-v1.toml"
warm_config="mantis-v2/configs/nextleg-runpod-cuda-3tf-lp-lora-s42-v1.toml"
direct_run_root="/workspace/mantis/runs/mantisv2-foundation-training-first-3tf-direct-lora-s42-v1"
warm_run_root="/workspace/mantis/runs/mantisv2-foundation-training-first-3tf-lp-lora-s42-v1"
screen_output="/workspace/mantis/runs/mantisv2-foundation-training-first-3tf-lp-lora-s42-screen-v1/screen-decision.json"

test -d "${source_root}"
if test ! -f "${staged_marker}"; then
    if test -e "${working_root}"; then
        echo "incomplete local corpus staging exists: ${working_root}" >&2
        exit 2
    fi
    mkdir -p "${working_root}"
    cp -R "${source_root}/." "${working_root}/"
    touch "${staged_marker}"
fi

cd "${repo_root}"
uv run mantis-v2 train --config "${direct_config}"
uv run mantis-v2 validated-export --config "${direct_config}"
uv run mantis-v2 train --config "${warm_config}"
uv run mantis-v2 validated-export --config "${warm_config}"
uv run mantis-v2 foundation-adaptation-screen \
    --direct-run-root "${direct_run_root}" \
    --warm-run-root "${warm_run_root}" \
    --output "${screen_output}"
