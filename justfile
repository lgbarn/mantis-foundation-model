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

frozen-screen-prepare market output config="mantis-v2/configs/frozen-expected-r-v1.json":
    uv run mantis-v2 frozen-screen-prepare --config {{quote(config)}} --market {{quote(market)}} --output {{quote(output)}}

frozen-screen-embed input output config="mantis-v2/configs/frozen-expected-r-v1.json":
    uv run mantis-v2 frozen-screen-embed --config {{quote(config)}} --input {{quote(input)}} --output {{quote(output)}}

frozen-screen-compare input embeddings output config="mantis-v2/configs/frozen-expected-r-v1.json" comparison_device="cuda" cpu_exception="":
    uv run mantis-v2 frozen-screen-compare --config {{quote(config)}} --input {{quote(input)}} --embeddings {{quote(embeddings)}} --output {{quote(output)}} --comparison-device {{quote(comparison_device)}} --cpu-exception {{quote(cpu_exception)}}

frozen-screen-preflight input embedding_output checks exact_command hourly_rate deadline_hours output:
    uv run mantis-v2 frozen-screen-preflight --input {{quote(input)}} --embedding-output {{quote(embedding_output)}} --checks {{quote(checks)}} --exact-command {{quote(exact_command)}} --hourly-rate {{quote(hourly_rate)}} --deadline-hours {{quote(deadline_hours)}} --output {{quote(output)}}

frozen-screen-plan-paid control_config output:
    uv run mantis-v2 frozen-screen-plan-paid --control-config {{quote(control_config)}} --output {{quote(output)}}

frozen-screen-seal-paid control_config planning_root decision manifest_root pod_manifest_path bound_decision evaluated_at:
    uv run mantis-v2 frozen-screen-seal-paid --control-config {{quote(control_config)}} --planning-root {{quote(planning_root)}} --decision {{quote(decision)}} --manifest-root {{quote(manifest_root)}} --pod-manifest-path {{quote(pod_manifest_path)}} --bound-decision {{quote(bound_decision)}} --evaluated-at {{quote(evaluated_at)}}

frozen-screen-run preflight manifest decision local runpodctl_binary aws_binary="aws":
    uv run mantis-v2 runpod-supervise-workload --preflight {{quote(preflight)}} --manifest {{quote(manifest)}} --decision {{quote(decision)}} --local {{quote(local)}} --runpodctl-binary {{quote(runpodctl_binary)}} --aws-binary {{quote(aws_binary)}}

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

runpod-official-bootstrap archive receipt:
    uv run python infra/runpod/scripts/prepare_official_bootstrap.py --archive {{quote(archive)}} --receipt {{quote(receipt)}}

runpod-train experiment runtime:
    infra/runpod/runpod-workflow train {{quote(experiment)}} {{quote(runtime)}}

runpod-recover receipt runtime:
    infra/runpod/runpod-workflow recover {{quote(receipt)}} {{quote(runtime)}}

runpod-tensorboard receipt runtime port="6006":
    infra/runpod/runpod-workflow tensorboard {{quote(receipt)}} {{quote(runtime)}} {{quote(port)}}

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

runpod-workload-execute manifest:
    uv run mantis-v2 workload-execute --manifest {{quote(manifest)}}

runpod-seal-workload spec output_root:
    uv run mantis-v2 runpod-seal-workload --spec {{quote(spec)}} --output-root {{quote(output_root)}}

runpod-bind-workload manifest decision pod_manifest_path evaluated_at output:
    uv run mantis-v2 runpod-bind-workload --manifest {{quote(manifest)}} --decision {{quote(decision)}} --pod-manifest-path {{quote(pod_manifest_path)}} --evaluated-at {{quote(evaluated_at)}} --output {{quote(output)}}

runpod-supervise-workload manifest decision local runpodctl_binary aws_binary="aws":
    uv run mantis-v2 runpod-supervise-workload --manifest {{quote(manifest)}} --decision {{quote(decision)}} --local {{quote(local)}} --runpodctl-binary {{quote(runpodctl_binary)}} --aws-binary {{quote(aws_binary)}}

transfer-bundle config:
    uv run mantis-v2 transfer-bundle --config {{quote(config)}}

transfer-stage-dry-run config remote_inventory:
    uv run mantis-v2 transfer-stage-dry-run --config {{quote(config)}} --remote-inventory {{quote(remote_inventory)}}

transfer-stage-runpod config local decision aws_binary="aws":
    uv run mantis-v2 transfer-stage-runpod --config {{quote(config)}} --local {{quote(local)}} --decision {{quote(decision)}} --aws-binary {{quote(aws_binary)}}

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

foundation-fixture-freeze config output_root:
    uv run mantis-v2 foundation-fixture-freeze --config {{quote(config)}} --output-root {{quote(output_root)}}

foundation-fixture-embed config fixture foundation_manifest output_root:
    uv run mantis-v2 foundation-fixture-embed --config {{quote(config)}} --fixture {{quote(fixture)}} --foundation-manifest {{quote(foundation_manifest)}} --output-root {{quote(output_root)}}

foundation-diagnostic-score fixture candidate reference output:
    uv run mantis-v2 foundation-diagnostic-score --fixture {{quote(fixture)}} --candidate {{quote(candidate)}} --reference {{quote(reference)}} --output {{quote(output)}}

foundation-matrix-plan-initial config output_root:
    uv run mantis-v2 foundation-matrix-plan-initial --config {{quote(config)}} --output-root {{quote(output_root)}}

foundation-matrix-plan-mode config decision output_root:
    uv run mantis-v2 foundation-matrix-plan-mode --config {{quote(config)}} --decision {{quote(decision)}} --output-root {{quote(output_root)}}

foundation-matrix-plan-confirmation config decision output_root:
    uv run mantis-v2 foundation-matrix-plan-confirmation --config {{quote(config)}} --decision {{quote(decision)}} --output-root {{quote(output_root)}}

foundation-matrix-decide-five-minute output *results:
    uv run mantis-v2 foundation-matrix-decide-five-minute --output {{quote(output)}} --result {{results}}

foundation-matrix-decide-mode output *results:
    uv run mantis-v2 foundation-matrix-decide-mode --output {{quote(output)}} --result {{results}}

foundation-matrix-decide-confirmation selection output *results:
    uv run mantis-v2 foundation-matrix-decide-confirmation --selection {{quote(selection)}} --output {{quote(output)}} --result {{results}}

foundation-matrix-cell plan cell_id:
    uv run mantis-v2 foundation-matrix-cell --plan {{quote(plan)}} --cell-id {{quote(cell_id)}}

foundation-matrix-finalize plan cell_id foundation_receipt diagnostic:
    uv run mantis-v2 foundation-matrix-finalize --plan {{quote(plan)}} --cell-id {{quote(cell_id)}} --foundation-receipt {{quote(foundation_receipt)}} --diagnostic {{quote(diagnostic)}}

foundation-matrix-promote decision cell_result output_root:
    uv run mantis-v2 foundation-matrix-promote --decision {{quote(decision)}} --cell-result {{quote(cell_result)}} --output-root {{quote(output_root)}}

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

rl-qualify-architecture winner evidence output config="mantis-v2/configs/rl-entry-topstep-100k.toml":
    uv run mantis-v2 rl-qualify-architecture --config {{config}} --winner {{winner}} --evidence {{evidence}} --output {{output}}

rl-freeze-architecture-plan winner output created_at *manifest_args:
    uv run mantis-v2 rl-freeze-architecture-plan --config mantis-v2/configs/rl-entry-topstep-100k.toml --winner {{quote(winner)}} --output {{quote(output)}} --created-at {{quote(created_at)}} {{manifest_args}}

rl-decide-continuation candidate evidence output config="mantis-v2/configs/rl-entry-topstep-100k.toml":
    uv run mantis-v2 rl-decide-continuation --config {{config}} --candidate {{candidate}} --evidence {{evidence}} --output {{output}}

rl-run-architecture-ablation plan output *args:
    uv run mantis-v2 rl-run-architecture-ablation --config mantis-v2/configs/rl-entry-topstep-100k.toml --plan {{quote(plan)}} --output {{quote(output)}} {{args}}

rl-run-seed-campaign candidate output +args:
    uv run mantis-v2 rl-run-seed-campaign --config mantis-v2/configs/rl-entry-topstep-100k.toml --candidate {{quote(candidate)}} --output {{quote(output)}} {{args}}
