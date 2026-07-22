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

probe-cuda-fp32 config qualification run_id artifact_root:
    uv run mantis-v2 cuda-fp32-probe --config {{quote(config)}} --qualification-config {{quote(qualification)}} --run-id {{quote(run_id)}} --artifact-root {{quote(artifact_root)}}

qualify-cuda-bf16 reference candidate output qualification="mantis-v2/configs/cuda-bf16-qualification.toml":
    uv run mantis-v2 cuda-bf16-qualify --qualification-config {{quote(qualification)}} --reference {{quote(reference)}} --candidate {{quote(candidate)}} --output {{quote(output)}}

reject-cuda-bf16 reference failure output qualification="mantis-v2/configs/cuda-bf16-qualification.toml":
    uv run mantis-v2 cuda-bf16-qualify --qualification-config {{quote(qualification)}} --reference {{quote(reference)}} --failure {{quote(failure)}} --output {{quote(output)}}

qualify-cuda-embedding identity foundation_manifest cpu_features cuda_features cpu_metadata cuda_metadata shard_directory performance output qualification="mantis-v2/configs/cuda-embedding-qualification.toml":
    uv run mantis-v2 cuda-embedding-qualify --qualification-config {{quote(qualification)}} --identity {{quote(identity)}} --foundation-manifest {{quote(foundation_manifest)}} --cpu-features {{quote(cpu_features)}} --cuda-features {{quote(cuda_features)}} --cpu-metadata {{quote(cpu_metadata)}} --cuda-metadata {{quote(cuda_metadata)}} --shard-directory {{quote(shard_directory)}} --performance {{quote(performance)}} --output {{quote(output)}}

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

runpod-launch decision local:
    uv run mantis-v2 runpod-launch --decision {{quote(decision)}} --local {{quote(local)}}

runpod-reconcile-launch decision local:
    uv run mantis-v2 runpod-reconcile-launch --decision {{quote(decision)}} --local {{quote(local)}}

runpod-status pod_id run_name local:
    uv run mantis-v2 runpod-status --pod-id {{quote(pod_id)}} --run-name {{quote(run_name)}} --local {{quote(local)}}

runpod-terminate pod_id run_name local:
    uv run mantis-v2 runpod-terminate --pod-id {{quote(pod_id)}} --run-name {{quote(run_name)}} --local {{quote(local)}}

runpod-reconcile-termination pod_id run_name local:
    uv run mantis-v2 runpod-reconcile-termination --pod-id {{quote(pod_id)}} --run-name {{quote(run_name)}} --local {{quote(local)}}

runpod-reconcile-spend pod_id run_name local:
    uv run mantis-v2 runpod-reconcile-spend --pod-id {{quote(pod_id)}} --run-name {{quote(run_name)}} --local {{quote(local)}}

runpod-enforce-deadline pod_id local:
    uv run mantis-v2 runpod-enforce-deadline --pod-id {{quote(pod_id)}} --local {{quote(local)}}

transfer-bundle config:
    uv run mantis-v2 transfer-bundle --config {{quote(config)}}

transfer-stage-dry-run config remote_inventory:
    uv run mantis-v2 transfer-stage-dry-run --config {{quote(config)}} --remote-inventory {{quote(remote_inventory)}}

transfer-manifest-inspect manifest:
    uv run mantis-v2 transfer-manifest-inspect --manifest {{quote(manifest)}}

transfer-promote config:
    uv run mantis-v2 transfer-promote --config {{quote(config)}}

transfer-backup-verify config completed_artifact_digest:
    uv run mantis-v2 transfer-backup-verify --config {{quote(config)}} --completed-artifact-digest {{quote(completed_artifact_digest)}}

transfer-retention-check config completed_artifact_digest run_state:
    uv run mantis-v2 transfer-retention-check --config {{quote(config)}} --completed-artifact-digest {{quote(completed_artifact_digest)}} --run-state {{quote(run_state)}}

transfer-retention-check-authorized config completed_artifact_digest run_state authorization:
    uv run mantis-v2 transfer-retention-check --config {{quote(config)}} --completed-artifact-digest {{quote(completed_artifact_digest)}} --run-state {{quote(run_state)}} --authorization {{quote(authorization)}}

runpod-terraform-fmt:
    tofu -chdir=infra/runpod/terraform fmt -check

runpod-terraform-validate:
    tofu -chdir=infra/runpod/terraform init -backend=false -lockfile=readonly
    tofu -chdir=infra/runpod/terraform validate

runpod-terraform-adoption desired inventory state output:
    uv run python infra/runpod/scripts/stable_resources.py adoption --desired {{quote(desired)}} --inventory {{quote(inventory)}} --state {{quote(state)}} --output {{quote(output)}}

runpod-terraform-plan vars desired inventory state plan plan_json adoption policy:
    #!/usr/bin/env bash
    set -euo pipefail
    test -n "${RUNPOD_API_KEY:-}" || { echo "RUNPOD_API_KEY is required" >&2; exit 2; }
    plan_path=infra/runpod/terraform/{{quote(plan)}}
    test ! -e "$plan_path" && test ! -e {{quote(plan_json)}} && test ! -e {{quote(adoption)}} && test ! -e {{quote(policy)}}
    uv run python infra/runpod/scripts/stable_resources.py adoption --desired {{quote(desired)}} --inventory {{quote(inventory)}} --state {{quote(state)}} --output {{quote(adoption)}}
    TF_VAR_runpod_api_key="$RUNPOD_API_KEY" tofu -chdir=infra/runpod/terraform plan -input=false -var-file={{quote(vars)}} -out={{quote(plan)}}
    tofu -chdir=infra/runpod/terraform show -json {{quote(plan)}} > {{quote(plan_json)}}
    uv run python infra/runpod/scripts/stable_resources.py policy --desired {{quote(desired)}} --plan {{quote(plan_json)}} --plan-binary "$plan_path" --adoption {{quote(adoption)}} --output {{quote(policy)}}

runpod-terraform-import-human address import_id:
    #!/usr/bin/env bash
    set -euo pipefail
    test -n "${RUNPOD_API_KEY:-}" || { echo "RUNPOD_API_KEY is required" >&2; exit 2; }
    address={{quote(address)}}
    import_id={{quote(import_id)}}
    case "$address" in
      restapi_object.network_volume) [[ "$import_id" == /networkvolumes/* ]] ;;
      restapi_object.pod_template) [[ "$import_id" == /templates/* ]] ;;
      *) echo "only stable RunPod resources may be imported" >&2; exit 2 ;;
    esac
    TF_VAR_runpod_api_key="$RUNPOD_API_KEY" tofu -chdir=infra/runpod/terraform import -input=false "$address" "$import_id"

runpod-terraform-apply-human plan policy authorization evaluated_at decision:
    #!/usr/bin/env bash
    set -euo pipefail
    test -n "${RUNPOD_API_KEY:-}" || { echo "RUNPOD_API_KEY is required" >&2; exit 2; }
    plan_path=infra/runpod/terraform/{{quote(plan)}}
    uv run python infra/runpod/scripts/stable_resources.py apply-authorization --policy {{quote(policy)}} --plan-binary "$plan_path" --authorization {{quote(authorization)}} --evaluated-at {{quote(evaluated_at)}} --output {{quote(decision)}}
    TF_VAR_runpod_api_key="$RUNPOD_API_KEY" tofu -chdir=infra/runpod/terraform apply -input=false {{quote(plan)}}

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

rl-train training_manifest output config="mantis-v2/configs/rl-entry-topstep-100k.toml" variant="shared_ticker_value" resume="" bounded="":
    uv run mantis-v2 rl-train --config {{config}} --training-manifest {{training_manifest}} --output {{output}} --variant {{variant}} {{resume}} {{bounded}}

rl-optuna-search training_manifest validation_manifest output study_name config="mantis-v2/configs/rl-entry-topstep-100k.toml" variant="shared_ticker_value":
    uv run mantis-v2 rl-optuna-search --config {{config}} --training-manifest {{training_manifest}} --validation-manifest {{validation_manifest}} --output {{output}} --study-name {{study_name}} --variant {{variant}}
