from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from mantis_v2.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)
    runtime = path.parents[1] / "runtime.json"
    if path.name == "runpodctl" and runtime.is_file():
        payload = json.loads(runtime.read_text())
        payload["runpodctl"]["binary_sha256"] = sha256(path.read_bytes()).hexdigest()
        runtime.write_text(json.dumps(payload))


def _runtime(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (source / ".gitignore").write_text(".env\n.venv/\n")
    (source / "uv.lock").write_text("version = 1\n")
    if not (source / ".git").is_dir():
        subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=RunPod Test",
            "-c",
            "user.email=runpod-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    corpus_manifest = tmp_path / "corpus-manifest.json"
    corpus_manifest.write_text('{"dataset":"five-year"}\n')
    ssh_key = tmp_path / "id_ed25519"
    ssh_key.write_text("test-key")
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "api_key_env": "TEST_RUNPOD_API_KEY",
                "api_v2": {
                    "base_url": "https://api.runpod.io/v2",
                    "openapi_sha256": "a" * 64,
                },
                "runpodctl": {
                    "version": "2.7.2",
                    "source_commit": "309512b4926eb7d218bbc8a8f11d380ce54f59c4",
                    "binary_sha256": "0" * 64,
                },
                "gpu": {
                    "id": "NVIDIA A100 80GB PCIe",
                    "count": 1,
                    "cloud": "SECURE",
                    "min_memory_gb": 80,
                    "max_hourly_price_usd": 2.0,
                },
                "pod": {
                    "name": "mantis-smoke",
                    "image": "runpod/pytorch:official",
                    "container_disk_gb": 80,
                    "min_free_disk_gb": 20,
                    "network_volume_id": "volume-1",
                    "volume_mount_path": "/workspace",
                    "ports": ["22/tcp"],
                    "terminate_after_minutes": 120,
                },
                "ssh": {
                    "key_path": str(ssh_key),
                    "user": "root",
                    "ready_timeout_seconds": 30,
                    "poll_seconds": 1,
                },
                "paths": {
                    "source_root": str(source),
                    "remote_repo": "/workspace/mantis/repo",
                    "corpus_manifest": str(corpus_manifest),
                    "remote_corpus_manifest": "/workspace/mantis/data/manifest.json",
                    "remote_artifact_root": "/workspace/mantis/artifacts/mantis-smoke",
                    "local_artifact_root": str(tmp_path / "artifacts"),
                    "receipt_root": str(tmp_path / "receipts"),
                },
            }
        )
    )
    return runtime


def _experiment(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    experiment = source / "experiment.toml"
    experiment.write_text(
        """[run]
name = "mantis-smoke"
seed = 42
artifact_root = "/workspace/mantis/artifacts"
device = "cuda"
require_accelerator = true
allow_overwrite = false

[data]
root = "/workspace/mantis/data/market"
corpus_manifest_path = "/workspace/mantis/data/manifest.json"
file_format = "parquet"
corpus_manifest_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
symbols = ["NQ"]
intervals = ["1min", "3min", "5min", "15min"]
timestamp_column = "datetime"
feature_columns = ["open", "high", "low", "close", "volume"]
holdout_start = "2026-01-01T00:00:00+00:00"
validation_fraction = 0.10
context_lengths = [64, 100, 150, 200]
target_reserve = 712
max_relative_close_jump = 0.05

[model]
source_repository = "vfeofanov/mantis"
source_revision = "0c94f8ceb9f1d1421dd292ed917090df8c31605b"
hub_model = "paris-noah/MantisV2"
hub_revision = "99fe0f548960e272fbfa4b82fd9b5b5956779dfd"
weights_sha256 = "49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1"
input_length = 512
channel_strategy = "independent_concat"
mode = "full_finetune"

[training]
precision = "fp32"
epochs = 1
batch_size = 4
learning_rate = 0.0001
weight_decay = 0.05
num_workers = 0
checkpoint_every = 1
resume = true
max_steps_per_epoch = 1
validation_max_steps = 1
warmup_epochs = 0
early_stopping_patience = 0

[target]
kind = "nextleg"
horizons = [5, 10, 20, 25]
leg_cap = 256
leg_k = 2
normalization_clamp = 10.0
candle_loss_weight = 1.0
leg_loss_weight = 1.0
minimum_train_anchors = 1
minimum_validation_anchors = 1

[evaluation]
allow_holdout = false

[export]
format = "safetensors"
verify_atol = 0.00001
verify_rtol = 0.0001
"""
    )
    return experiment


def _environment(tmp_path: Path, *, curl_body: str) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"
    _write_executable(bin_dir / "curl", f"printf '%s' '{curl_body}'\n")
    _write_executable(
        bin_dir / "runpodctl",
        f"printf 'runpodctl %s\\n' \"$*\" >> {call_log}\nprintf 'runpodctl 2.7.2-309512b\\n'\n",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["TEST_RUNPOD_API_KEY"] = "secret"
    return env, call_log


def _success_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"
    deleted = tmp_path / "deleted"
    created = tmp_path / "created"
    identity = _receipt(tmp_path / "runtime.json")
    provenance = json.dumps(
        {
            "source_revision": identity["source_revision"],
            "source_dirty": False,
            "config_digest": identity["config_digest"],
            "lock_digest": identity["lock_sha256"],
        }
    )
    catalog = json.dumps(
        {
            "gpus": [
                {
                    "id": "NVIDIA A100 80GB PCIe",
                    "memory": 80,
                    "price": {"secure": 1.5, "community": 1.0},
                    "availability": "HIGH",
                }
            ]
        }
    )
    _write_executable(
        bin_dir / "curl",
        f"printf 'curl %s\\n' \"$*\" >> {call_log}\nprintf '%s' '{catalog}'\n",
    )
    _write_executable(
        bin_dir / "runpodctl",
        f"""
printf 'runpodctl %s\\n' "$*" >> {call_log}
case "${{1:-}} ${{2:-}}" in
  "version ") printf 'runpodctl 2.7.2-309512b\\n' ;;
  "pod create")
    touch {created}
    printf '%s\\n' '{{"id":"pod-123"}}'
    ;;
  "ssh info")
    printf '%s%s\\n' '{{"id":"pod-123","ip":"127.0.0.1","port":2222,' \
      '"ssh_command":"ssh root@127.0.0.1 -p 2222"}}'
    ;;
  "pod delete")
    if [[ "${{DELETE_FAIL:-0}}" == 1 ]]; then exit 7; fi
    touch {deleted}; printf '%s\\n' '{{"id":"pod-123","deleted":true}}'
    ;;
  "pod get")
    printf '%s%s\\n' '{{"id":"pod-123","name":"mantis-smoke",' \
      '"imageName":"runpod/pytorch:official","networkVolumeId":"volume-1",' \
      '"desiredStatus":"RUNNING","costPerHr":"1.5"}}'
    ;;
  "pod list")
    if [[ -f {deleted} && "${{STALE_POD_AFTER_DELETE:-0}}" == 1 ]]; then
      printf '[{{"id":"pod-123","name":"mantis-smoke"}}]\\n'
    elif [[ -f {deleted} || ! -f {created} ]]; then
      printf '[]\\n'
    else
      printf '[{{"id":"pod-123","name":"mantis-smoke"}}]\\n'
    fi
    ;;
  "billing pods")
    if [[ "${{BILLING_FAIL:-0}}" == 1 ]]; then exit 7; fi
    printf '%s\\n' '[{{"podId":"pod-123","amount":0.01}}]'
    ;;
  *) printf 'unexpected runpodctl call: %s\\n' "$*" >&2; exit 9 ;;
esac
""",
    )
    _write_executable(
        bin_dir / "ssh",
        f"printf 'ssh %s\\n' \"$*\" >> {call_log}\n",
    )
    _write_executable(
        bin_dir / "rsync",
        f"""
printf 'rsync %s\\n' "$*" >> {call_log}
if [[ "$*" == *"--files-from=-"* ]]; then cat >/dev/null; fi
destination="${{@: -1}}"
if [[ "$destination" != root@* && "$*" == *"root@127.0.0.1:"* ]]; then
  mkdir -p "$destination/export" "$destination/checkpoints"
  printf '{{"device":"cuda","epochs_completed":1}}\\n' > "$destination/train-result.json"
  printf '{{"passed":true}}\\n' > "$destination/evaluation.json"
  cp "$destination/evaluation.json" "$destination/export/evaluation.json"
  printf 'weights' > "$destination/export/model.safetensors"
  printf 'checkpoint' > "$destination/checkpoints/best.pt"
  printf '%s\n' '{provenance}' > "$destination/provenance.json"
  if [[ "${{STALE_PROVENANCE:-0}}" == 1 ]]; then
    jq '.source_revision = "0000000000000000000000000000000000000000"' \
      "$destination/provenance.json" > "$destination/provenance.json.tmp"
    mv "$destination/provenance.json.tmp" "$destination/provenance.json"
  fi
  jq --slurpfile provenance "$destination/provenance.json" \
    '. + {{provenance: $provenance[0]}}' "$destination/evaluation.json" \
    > "$destination/evaluation.json.tmp"
  mv "$destination/evaluation.json.tmp" "$destination/evaluation.json"
  cp "$destination/evaluation.json" "$destination/export/evaluation.json"
  weights_sha="$(shasum -a 256 "$destination/export/model.safetensors" | awk '{{print $1}}')"
  evaluation_sha="$(shasum -a 256 "$destination/export/evaluation.json" | awk '{{print $1}}')"
  checkpoint_sha="$(shasum -a 256 "$destination/checkpoints/best.pt" | awk '{{print $1}}')"
  jq -n --arg weights "$weights_sha" --arg evaluation "$evaluation_sha" \
    --arg checkpoint "$checkpoint_sha" \
    '{{config:{{run:{{name:"mantis-smoke"}}}},weights_sha256:$weights,
      validation_gate:{{verified:true,evaluation_sha256:$evaluation,
      checkpoint_sha256:$checkpoint}},parity:{{verified:true}}}}' \
    > "$destination/export/manifest.json"
  jq --slurpfile provenance "$destination/provenance.json" \
    '. + {{provenance: $provenance[0]}}' "$destination/export/manifest.json" \
    > "$destination/export/manifest.json.tmp"
  mv "$destination/export/manifest.json.tmp" "$destination/export/manifest.json"
  if [[ "${{STALE_EXPORT_PROVENANCE:-0}}" == 1 ]]; then
    jq '.provenance.source_revision = "0000000000000000000000000000000000000000"' \
      "$destination/export/manifest.json" > "$destination/export/manifest.json.tmp"
    mv "$destination/export/manifest.json.tmp" "$destination/export/manifest.json"
  fi
  if [[ "${{CORRUPT_ARTIFACT:-0}}" == 1 ]]; then
    printf 'corrupt' >> "$destination/export/model.safetensors"
  fi
fi
""",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["TEST_RUNPOD_API_KEY"] = "secret"
    return env, call_log


def _receipt(runtime: Path) -> dict[str, object]:
    experiment = runtime.parent / "source" / "experiment.toml"
    if not experiment.is_file():
        experiment = _experiment(runtime.parent)
    source = runtime.parent / "source"
    return {
        "schema_version": 1,
        "pod_id": "pod-123",
        "run_id": "mantis-smoke",
        "request_digest": "b" * 64,
        "runpodctl_version": "2.7.2-309512b",
        "api_v2_openapi_sha256": "a" * 64,
        "image": "runpod/pytorch:official",
        "gpu_id": "NVIDIA A100 80GB PCIe",
        "network_volume_id": "volume-1",
        "hourly_price_usd": 1.5,
        "created_at": "2026-07-23T12:00:00Z",
        "termination_deadline": "2026-07-23T14:00:00Z",
        "experiment_sha256": "c" * 64,
        "config_digest": load_config(experiment).digest,
        "lock_sha256": sha256((source / "uv.lock").read_bytes()).hexdigest(),
        "runtime_sha256": sha256(runtime.read_bytes()).hexdigest(),
        "source_revision": subprocess.run(
            ["git", "-C", str(runtime.parent / "source"), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }


def _wait_for_call(call_log: Path, expected: str, timeout: float = 10) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        calls = call_log.read_text() if call_log.exists() else ""
        if expected in calls:
            return calls
        time.sleep(0.05)
    return call_log.read_text() if call_log.exists() else ""


def test_train_rejects_unknown_runtime_config_keys_before_provider_access(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"schema_version": 1, "unexpected": True}))

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "runtime_config_unknown_keys:unexpected" in completed.stderr


def test_train_rejects_unavailable_gpu_before_create(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _environment(
        tmp_path,
        curl_body=json.dumps(
            {
                "gpus": [
                    {
                        "id": "NVIDIA A100 80GB PCIe",
                        "memory": 80,
                        "price": {"secure": 1.5, "community": 1.0},
                        "availability": "NONE",
                    }
                ]
            }
        ),
    )

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "gpu_unavailable:NVIDIA A100 80GB PCIe" in completed.stderr
    assert "pod create" not in call_log.read_text()


def test_train_runs_and_always_deletes_exact_pod(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((tmp_path / "receipts" / "mantis-smoke.json").read_text())
    assert receipt["pod_id"] == "pod-123"
    assert receipt["run_id"] == "mantis-smoke"
    assert receipt["runpodctl_version"] == "2.7.2-309512b"
    assert receipt["gpu_id"] == "NVIDIA A100 80GB PCIe"
    assert receipt["hourly_price_usd"] == 1.5
    assert receipt["termination_deadline"].endswith("Z")
    calls = call_log.read_text()
    assert "runpodctl pod create" in calls
    assert "count=1" in calls
    assert "cloud=SECURE" in calls
    assert f"--terminate-after={receipt['termination_deadline']}" in calls
    assert "--from0 --files-from=-" in calls
    assert "--delete --checksum --partial" in calls
    assert f"{tmp_path / 'source'}/.git/ root@127.0.0.1:/workspace/mantis/repo/.git/" in calls
    assert "nvidia-smi --query-gpu=memory.total" in calls
    assert "git status --porcelain" in calls
    base_cuda_smoke = "python3 -c 'import torch; assert torch.cuda.is_available()"
    resolved_cuda_smoke = "uv run python -c 'import torch; assert torch.cuda.is_available()"
    assert base_cuda_smoke in calls
    assert resolved_cuda_smoke in calls
    assert "uv run mantis-v2 inspect-data --config /tmp/mantis-smoke-experiment.toml" in calls
    assert "uv run mantis-v2 train --config /tmp/mantis-smoke-experiment.toml" in calls
    assert "uv run mantis-v2 validated-export --config /tmp/mantis-smoke-experiment.toml" in calls
    assert "rsync" in calls
    assert "ssh" in calls
    assert "runpodctl pod delete pod-123" in calls
    assert "runpodctl pod list --all --output=json" in calls
    assert calls.index("runpodctl pod create") < calls.index("runpodctl pod delete pod-123")
    assert calls.index(base_cuda_smoke) < calls.index("rsync")
    assert calls.index(resolved_cuda_smoke) < calls.index(
        "uv run mantis-v2 inspect-data --config /tmp/mantis-smoke-experiment.toml"
    )
    assert (tmp_path / "artifacts" / "train-result.json").is_file()


def test_train_uses_immutable_runtime_snapshot_after_catalog(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    catalog = json.dumps(
        {
            "gpus": [
                {
                    "id": "NVIDIA A100 80GB PCIe",
                    "memory": 80,
                    "price": {"secure": 1.5, "community": 1.0},
                    "availability": "HIGH",
                }
            ]
        }
    )
    _write_executable(
        tmp_path / "bin" / "curl",
        f"""
printf 'curl %s\\n' "$*" >> {call_log}
jq '.gpu.count = 2 | .gpu.max_hourly_price_usd = 99' {runtime} > {runtime}.tmp
mv {runtime}.tmp {runtime}
printf '%s' '{catalog}'
""",
    )

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    calls = call_log.read_text()
    assert "count=1" in calls
    assert "count=2" not in calls


def test_train_rejects_complete_artifacts_from_different_provenance(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    env["STALE_PROVENANCE"] = "1"

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "artifact_provenance_mismatch" in completed.stderr
    assert "runpodctl pod delete pod-123" in call_log.read_text()


def test_train_rejects_export_manifest_mixed_with_current_provenance(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    env["STALE_EXPORT_PROVENANCE"] = "1"

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "export_provenance_mismatch" in completed.stderr
    assert "runpodctl pod delete pod-123" in call_log.read_text()


def test_recover_never_creates_and_deletes_receipted_pod(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    receipt = tmp_path / "run-receipt.json"
    receipt.write_text(json.dumps(_receipt(runtime)))

    completed = subprocess.run(
        ["just", "runpod-recover", str(receipt), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    calls = call_log.read_text()
    assert "pod create" not in calls
    assert "curl " not in calls
    assert "runpodctl pod delete pod-123" in calls
    assert "runpodctl billing pods --pod-id pod-123" in calls
    assert (tmp_path / "artifacts" / "train-result.json").is_file()


def test_recover_is_not_blocked_by_missing_train_inputs_or_dirty_source(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    receipt = tmp_path / "run-receipt.json"
    receipt.write_text(json.dumps(_receipt(runtime)))
    (tmp_path / "corpus-manifest.json").unlink()
    (tmp_path / "source" / "unrelated.txt").write_text("dirty\n")

    completed = subprocess.run(
        ["just", "runpod-recover", str(receipt), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    calls = call_log.read_text()
    assert "pod create" not in calls
    assert "runpodctl pod delete pod-123" in calls


def test_recover_accepts_a_provenance_bound_partial_checkpoint(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    receipt_payload = _receipt(runtime)
    receipt = tmp_path / "run-receipt.json"
    receipt.write_text(json.dumps(receipt_payload))
    remote = tmp_path / "partial-remote"
    (remote / "checkpoints").mkdir(parents=True)
    provenance = {
        "schema_version": 1,
        "precision": "fp32",
        "config_digest": receipt_payload["config_digest"],
        "dataset_digest": "2" * 64,
        "dataset_files": [],
        "source_revision": receipt_payload["source_revision"],
        "source_dirty": False,
        "source_digest": "3" * 64,
        "lock_digest": receipt_payload["lock_sha256"],
        "upstream_source_revision": "5" * 40,
        "upstream_hub_revision": "6" * 40,
        "upstream_weights_sha256": "7" * 64,
        "contamination_digest": "8" * 64,
    }
    numpy_state = np.random.get_state()
    torch.save(
        {
            "schema_version": 1,
            "epoch": 0,
            "global_step": 200,
            "model": {"weight": torch.ones(1)},
            "optimizer": {
                "state": {
                    0: {
                        "step": torch.tensor(1.0),
                        "exp_avg": torch.zeros(1),
                        "exp_avg_sq": torch.zeros(1),
                    }
                },
                "param_groups": [{"params": [0]}],
            },
            "rng": {
                "python": random.getstate(),
                "numpy": {
                    "bit_generator": numpy_state[0],
                    "keys": torch.from_numpy(numpy_state[1].copy()),
                    "position": numpy_state[2],
                    "has_gauss": numpy_state[3],
                    "cached_gaussian": numpy_state[4],
                },
                "torch": torch.get_rng_state(),
                "cuda": [],
                "mps": torch.empty(0, dtype=torch.uint8),
            },
            "provenance": provenance,
        },
        remote / "checkpoints" / "latest.pt",
    )
    (remote / "metrics.json").write_text(json.dumps([{"epoch": 0}]))
    (remote / "provenance.json").write_text(json.dumps(provenance))
    _write_executable(
        tmp_path / "bin" / "rsync",
        f"""
printf 'rsync %s\\n' "$*" >> {call_log}
destination="${{@: -1}}"
if [[ "$destination" != root@* && "$*" == *"root@127.0.0.1:"* ]]; then
  mkdir -p "$destination"
  cp -R {remote}/. "$destination/"
fi
""",
    )

    completed = subprocess.run(
        ["just", "runpod-recover", str(receipt), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "artifacts" / "checkpoints" / "latest.pt").is_file()
    assert not (tmp_path / "artifacts" / "train-result.json").exists()
    assert "runpodctl pod delete pod-123" in call_log.read_text()


def test_recover_rejects_runtime_drift_before_pod_access(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    receipt = tmp_path / "run-receipt.json"
    receipt.write_text(json.dumps(_receipt(runtime)))
    payload = json.loads(runtime.read_text())
    payload["paths"]["local_artifact_root"] = str(tmp_path / "different-artifacts")
    runtime.write_text(json.dumps(payload))
    env, call_log = _success_environment(tmp_path)

    completed = subprocess.run(
        ["just", "runpod-recover", str(receipt), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "run_receipt_runtime_mismatch" in completed.stderr
    calls = call_log.read_text()
    assert "runpodctl version" in calls
    assert "runpodctl pod get" not in calls
    assert "pod create" not in calls


def test_tensorboard_emits_localhost_only_ssh_tunnel(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    receipt = tmp_path / "run-receipt.json"
    receipt.write_text(json.dumps(_receipt(runtime)))

    completed = subprocess.run(
        ["just", "runpod-tensorboard", str(receipt), str(runtime), "6006"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        completed.stdout.strip() == f"ssh -N -L 6006:127.0.0.1:6006 -i {tmp_path / 'id_ed25519'} "
        "-p 2222 root@127.0.0.1"
    )
    calls = call_log.read_text()
    assert "runpodctl ssh info pod-123 --output=json" in calls
    assert "pod create" not in calls
    assert "pod delete" not in calls
    assert "curl " not in calls


def test_recover_deletes_exact_pod_when_status_read_fails(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    _write_executable(
        tmp_path / "bin" / "runpodctl",
        f"""
printf 'runpodctl %s\\n' "$*" >> {call_log}
case "${{1:-}} ${{2:-}}" in
  "version ") printf 'runpodctl 2.7.2-309512b\\n' ;;
  "pod get") exit 1 ;;
  "pod delete") printf '%s\\n' '{{"id":"pod-123","deleted":true}}' ;;
  "pod list") printf '[]\\n' ;;
  "billing pods") printf '%s\\n' '[{{"podId":"pod-123","amount":0.01}}]' ;;
  *) exit 9 ;;
esac
""",
    )
    receipt = tmp_path / "run-receipt.json"
    receipt.write_text(json.dumps(_receipt(runtime)))

    completed = subprocess.run(
        ["just", "runpod-recover", str(receipt), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    calls = call_log.read_text()
    assert "pod create" not in calls
    assert "runpodctl pod delete pod-123" in calls


def test_successful_train_fails_when_delete_fails(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    env["DELETE_FAIL"] = "1"

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "pod_delete_failed:pod-123" in completed.stderr
    assert "runpodctl billing pods --pod-id pod-123" in call_log.read_text()


def test_successful_train_fails_when_billing_query_fails(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    env["BILLING_FAIL"] = "1"

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "billing_query_failed:pod-123" in completed.stderr
    assert "runpodctl pod delete pod-123" in call_log.read_text()


def test_train_rejects_overpriced_gpu_before_create(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    payload = json.loads(runtime.read_text())
    payload["gpu"]["max_hourly_price_usd"] = 1.0
    runtime.write_text(json.dumps(payload))
    env, call_log = _environment(
        tmp_path,
        curl_body=json.dumps(
            {
                "gpus": [
                    {
                        "id": "NVIDIA A100 80GB PCIe",
                        "memory": 80,
                        "price": {"secure": 1.5, "community": 1.0},
                        "availability": "HIGH",
                    }
                ]
            }
        ),
    )

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "gpu_price_exceeds_limit:1.5:1" in completed.stderr
    assert "pod create" not in call_log.read_text()


def test_train_prices_the_complete_multi_gpu_request_before_create(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    payload = json.loads(runtime.read_text())
    payload["gpu"]["count"] = 2
    payload["gpu"]["max_hourly_price_usd"] = 2.0
    runtime.write_text(json.dumps(payload))
    env, call_log = _environment(
        tmp_path,
        curl_body=json.dumps(
            {
                "gpus": [
                    {
                        "id": "NVIDIA A100 80GB PCIe",
                        "memory": 80,
                        "price": {"secure": 1.5, "community": 1.0},
                        "availability": "HIGH",
                    }
                ]
            }
        ),
    )

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "gpu_price_exceeds_limit:3:2" in completed.stderr
    assert "pod create" not in call_log.read_text()


def test_train_rejects_an_existing_atomic_run_lock_before_provider_access(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    lock = tmp_path / "receipts" / "mantis-smoke.lock"
    lock.mkdir(parents=True)
    env, call_log = _environment(tmp_path, curl_body='{"gpus":[]}')

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert f"run_lock_exists:{lock}" in completed.stderr
    assert not call_log.exists()


def test_successful_train_removes_its_atomic_run_lock(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, _ = _success_environment(tmp_path)

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "receipts" / "mantis-smoke.lock").exists()


def test_train_deletes_exact_pod_after_remote_failure(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    _write_executable(
        tmp_path / "bin" / "ssh",
        f"""
printf 'ssh %s\\n' "$*" >> {call_log}
if [[ "$*" == *"uv sync --frozen"* ]]; then exit 7; fi
""",
    )

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    calls = call_log.read_text()
    assert "runpodctl pod delete pod-123" in calls
    assert "runpodctl billing pods --pod-id pod-123" in calls


def test_train_deletes_returned_id_when_create_schema_is_invalid(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    deleted = tmp_path / "deleted"
    created = tmp_path / "created"
    _write_executable(
        tmp_path / "bin" / "runpodctl",
        f"""
printf 'runpodctl %s\\n' "$*" >> {call_log}
case "${{1:-}} ${{2:-}}" in
  "version ") printf 'runpodctl 2.7.2-309512b\\n' ;;
  "pod create") touch {created}; printf '%s\\n' '{{"id":"pod-123"}}' ;;
  "pod delete") touch {deleted}; printf '%s\\n' '{{"id":"pod-123","deleted":true}}' ;;
  "pod get") printf '%s\\n' '{{"id":"pod-123"}}' ;;
  "pod list") [[ -f {deleted} ]] && printf '[]\\n' || printf '[{{"id":"pod-123"}}]\\n' ;;
  "billing pods") printf '%s\\n' '[]' ;;
  *) exit 9 ;;
esac
""",
    )

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "provider_status_schema_or_price_invalid" in completed.stderr
    calls = call_log.read_text()
    assert calls.count("runpodctl pod create") == 1
    assert "runpodctl pod delete pod-123" in calls


def test_train_deletes_id_returned_by_failed_create_and_blocks_retry(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    deleted = tmp_path / "deleted"
    _write_executable(
        tmp_path / "bin" / "runpodctl",
        f"""
printf 'runpodctl %s\\n' "$*" >> {call_log}
case "${{1:-}} ${{2:-}}" in
  "version ") printf 'runpodctl 2.7.2-309512b\\n' ;;
  "pod create") printf '%s\\n' '{{"id":"pod-123"}}'; exit 7 ;;
  "pod delete") touch {deleted}; printf '%s\\n' '{{"id":"pod-123","deleted":true}}' ;;
  "pod list") printf '[]\\n' ;;
  "billing pods") printf '%s\\n' '[{{"podId":"pod-123","amount":0.01}}]' ;;
  *) exit 9 ;;
esac
""",
    )

    first = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    second = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert first.returncode == 2
    assert "provider_create_failed" in first.stderr
    assert second.returncode == 2
    assert "ambiguous_create_evidence_exists" in second.stderr
    evidence = json.loads(
        (tmp_path / "receipts" / "mantis-smoke-ambiguous-create.json").read_text()
    )
    assert evidence["pod_id"] == "pod-123"
    calls = call_log.read_text()
    assert calls.count("runpodctl pod create") == 1
    assert "runpodctl pod delete pod-123" in calls


def test_train_fails_when_exact_pod_absence_cannot_be_verified(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    deleted = tmp_path / "deleted"
    created = tmp_path / "created"
    _write_executable(
        tmp_path / "bin" / "runpodctl",
        f"""
printf 'runpodctl %s\\n' "$*" >> {call_log}
case "${{1:-}} ${{2:-}}" in
  "version ") printf 'runpodctl 2.7.2-309512b\\n' ;;
  "pod create") touch {created}; printf '%s\\n' '{{"id":"pod-123"}}' ;;
  "pod get")
    printf '%s%s\\n' '{{"id":"pod-123","name":"mantis-smoke",' \
      '"gpuTypeId":"NVIDIA A100 80GB PCIe","gpuCount":1,' \
      '"imageName":"runpod/pytorch:official","networkVolumeId":"volume-1",' \
      '"containerDiskInGb":80,' \
      '"volumeMountPath":"/workspace","desiredStatus":"RUNNING","costPerHr":"1.5"}}'
    ;;
  "ssh info")
    printf '%s%s\\n' '{{"id":"pod-123","ip":"127.0.0.1","port":2222,' \
      '"ssh_command":"ssh root@127.0.0.1 -p 2222"}}'
    ;;
  "pod delete") touch {deleted}; printf '%s\\n' '{{"id":"pod-123","deleted":true}}' ;;
  "pod list")
    if [[ ! -f {created} ]]; then printf '[]\\n'; else exit 7; fi
    ;;
  "billing pods") printf '%s\\n' '[{{"podId":"pod-123","amount":0.01}}]' ;;
  *) exit 9 ;;
esac
""",
    )

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "pod_absence_unverifiable:pod-123" in completed.stderr
    calls = call_log.read_text()
    assert "runpodctl pod delete pod-123" in calls
    assert "runpodctl billing pods --pod-id pod-123" in calls


def test_train_rejects_inline_api_key_before_provider_access(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    payload = json.loads(runtime.read_text())
    payload["pod"]["name"] = "secret"
    runtime.write_text(json.dumps(payload))
    env, call_log = _environment(tmp_path, curl_body='{"gpus":[]}')

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "runtime_config_contains_api_key" in completed.stderr
    assert not call_log.exists()


def test_train_rejects_paths_outside_network_volume(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    payload = json.loads(runtime.read_text())
    payload["paths"]["remote_artifact_root"] = "/tmp/artifacts"
    runtime.write_text(json.dumps(payload))
    env, call_log = _environment(tmp_path, curl_body='{"gpus":[]}')

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "runtime_config_invalid" in completed.stderr
    assert not call_log.exists()


def test_train_rejects_changed_catalog_schema_before_create(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _environment(
        tmp_path,
        curl_body=json.dumps(
            {
                "gpus": [
                    {
                        "id": "NVIDIA A100 80GB PCIe",
                        "memory": 80,
                        "price": {"secure": 1.5},
                    }
                ]
            }
        ),
    )

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "catalog_gpu_schema_invalid:NVIDIA A100 80GB PCIe" in completed.stderr
    assert "pod create" not in call_log.read_text()


def test_train_deletes_exact_pod_after_transfer_failure(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    _write_executable(
        tmp_path / "bin" / "rsync",
        f"printf 'rsync %s\\n' \"$*\" >> {call_log}\nexit 8\n",
    )

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    calls = call_log.read_text()
    assert "runpodctl pod delete pod-123" in calls
    assert "runpodctl billing pods --pod-id pod-123" in calls


def test_train_deletes_exact_pod_after_artifact_rejection(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    _write_executable(
        tmp_path / "bin" / "rsync",
        f"""
printf 'rsync %s\\n' "$*" >> {call_log}
destination="${{@: -1}}"
if [[ "$destination" != root@* && "$*" == *"root@127.0.0.1:"* ]]; then
  mkdir -p "$destination"
  printf '{{"status":"complete"}}\\n' > "$destination/train-result.json"
fi
""",
    )

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    calls = call_log.read_text()
    assert "runpodctl pod delete pod-123" in calls
    assert "runpodctl billing pods --pod-id pod-123" in calls


def test_train_deletes_exact_pod_after_signal(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    _write_executable(
        tmp_path / "bin" / "ssh",
        f"""
printf 'ssh %s\\n' "$*" >> {call_log}
if [[ "$*" == *"uv sync --frozen"* ]]; then sleep 30; fi
""",
    )

    process = subprocess.Popen(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    if "uv sync --frozen" not in _wait_for_call(call_log, "uv sync --frozen"):
        process.kill()
        raise AssertionError("remote training command did not start")
    os.killpg(process.pid, signal.SIGINT)
    process.communicate(timeout=10)

    assert process.returncode != 0
    calls = _wait_for_call(call_log, "runpodctl billing pods --pod-id pod-123")
    assert "runpodctl pod delete pod-123" in calls
    assert "runpodctl billing pods --pod-id pod-123" in calls


def test_committed_runpod_experiment_is_cuda_four_timeframe_production() -> None:
    config = load_config(ROOT / "mantis-v2/configs/nextleg-runpod-cuda-v1.toml")

    assert config.run.device == "cuda"
    assert config.run.require_accelerator is True
    assert config.model.mode == "full_finetune"
    assert config.data.intervals == ("1min", "3min", "5min", "15min")
    assert config.training.precision == "fp32"
    assert config.run.artifact_root == Path("/workspace/mantis/runs")


def test_train_rejects_runpodctl_version_drift_before_catalog(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    _write_executable(
        tmp_path / "bin" / "runpodctl",
        f"printf 'runpodctl %s\\n' \"$*\" >> {call_log}\nprintf 'runpodctl 2.7.3-deadbee\\n'\n",
    )

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "runpodctl_version_mismatch:runpodctl 2.7.3-deadbee" in completed.stderr
    calls = call_log.read_text()
    assert "curl " not in calls
    assert "pod create" not in calls


def test_tensorboard_rejects_malformed_ssh_json(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    _write_executable(
        tmp_path / "bin" / "runpodctl",
        f"""
printf 'runpodctl %s\\n' "$*" >> {call_log}
case "${{1:-}} ${{2:-}}" in
  "version ") printf 'runpodctl 2.7.2-309512b\\n' ;;
  "ssh info") printf '%s\\n' '{{"id":"pod-123","ssh_command":"not-ssh"}}' ;;
  *) exit 9 ;;
esac
""",
    )
    receipt = tmp_path / "run-receipt.json"
    receipt.write_text(json.dumps(_receipt(runtime)))

    completed = subprocess.run(
        ["just", "runpod-tensorboard", str(receipt), str(runtime), "6006"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "ssh_connection_schema_invalid:pod-123" in completed.stderr


def test_failed_create_without_id_reconciles_and_deletes_unique_pod(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    created = tmp_path / "created"
    deleted = tmp_path / "deleted"
    _write_executable(
        tmp_path / "bin" / "runpodctl",
        f"""
printf 'runpodctl %s\\n' "$*" >> {call_log}
case "${{1:-}} ${{2:-}}" in
  "version ") printf 'runpodctl 2.7.2-309512b\\n' ;;
  "pod create") touch {created}; printf '{{}}\\n'; exit 7 ;;
  "pod list")
    if [[ -f {deleted} || ! -f {created} ]]; then printf '[]\\n';
    else printf '[{{"id":"pod-123","name":"mantis-smoke"}}]\\n'; fi
    ;;
  "pod delete") touch {deleted}; printf '{{}}\\n' ;;
  "billing pods") printf '[]\\n' ;;
  *) exit 9 ;;
esac
""",
    )

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    evidence = json.loads((tmp_path / "receipts/mantis-smoke-ambiguous-create.json").read_text())
    assert evidence["pod_id"] == "pod-123"
    calls = call_log.read_text()
    assert calls.count("runpodctl pod create") == 1
    assert "runpodctl pod delete pod-123" in calls


def test_recover_ambiguous_create_reconciles_by_unique_name_without_create(
    tmp_path: Path,
) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    (tmp_path / "created").touch()
    receipt = tmp_path / "ambiguous.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "mantis-smoke",
                "pod_id": "",
                "reason": "provider_create_failed",
                "provider_output": "{}",
                "runtime_sha256": sha256(runtime.read_bytes()).hexdigest(),
                "termination_deadline": "2026-07-23T14:00:00Z",
            }
        )
    )

    completed = subprocess.run(
        ["just", "runpod-recover", str(receipt), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    calls = call_log.read_text()
    assert "pod create" not in calls
    assert "runpodctl pod delete pod-123" in calls
    assert not receipt.exists()
    assert (tmp_path / "ambiguous-resolved.json").is_file()

    retried = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert retried.returncode == 0, retried.stderr
    assert call_log.read_text().count("runpodctl pod create") == 1


def test_recover_ambiguous_known_pod_already_absent_is_success(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    receipt = tmp_path / "ambiguous.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "mantis-smoke",
                "pod_id": "pod-123",
                "reason": "provider_create_failed",
                "provider_output": '{"id":"pod-123"}',
                "runtime_sha256": sha256(runtime.read_bytes()).hexdigest(),
                "termination_deadline": "2026-07-23T14:00:00Z",
            }
        )
    )

    completed = subprocess.run(
        ["just", "runpod-recover", str(receipt), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ambiguous_create_resolved_absent" in completed.stdout
    calls = call_log.read_text()
    assert "pod create" not in calls
    assert "runpodctl pod get pod-123" not in calls
    assert "runpodctl pod delete pod-123" not in calls
    assert not receipt.exists()
    assert (tmp_path / "ambiguous-resolved.json").is_file()

    retried = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert retried.returncode == 0, retried.stderr
    assert call_log.read_text().count("runpodctl pod create") == 1


def test_train_rejects_corrupted_download_and_preserves_staging(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    env["CORRUPT_ARTIFACT"] = "1"

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "export_weights_digest_mismatch" in completed.stderr
    assert not (tmp_path / "artifacts").exists()
    assert (tmp_path / ".mantis-smoke.download").is_dir()
    assert "runpodctl pod delete pod-123" in call_log.read_text()


def test_artifact_retry_reuses_resumable_content_checked_staging(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, first_log = _success_environment(tmp_path)
    env["CORRUPT_ARTIFACT"] = "1"

    first = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 2
    staging = tmp_path / ".mantis-smoke.download"
    assert staging.is_dir()

    (tmp_path / "receipts" / "mantis-smoke.json").unlink()
    (tmp_path / "created").unlink(missing_ok=True)
    (tmp_path / "deleted").unlink(missing_ok=True)
    env.pop("CORRUPT_ARTIFACT")
    second = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert second.returncode == 0, second.stderr
    assert (tmp_path / "artifacts" / "train-result.json").is_file()
    assert not staging.exists()
    calls = first_log.read_text()
    assert "--checksum" in calls
    assert "--partial-dir=.rsync-partial" in calls


def test_train_fails_if_deleted_pod_remains_in_inventory(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    env["STALE_POD_AFTER_DELETE"] = "1"

    completed = subprocess.run(
        ["just", "runpod-train", str(experiment), str(runtime)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "pod_absence_not_verified:pod-123" in completed.stderr
    assert call_log.read_text().count("runpodctl pod list --all --output=json") >= 4
