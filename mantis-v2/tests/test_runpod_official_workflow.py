from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def _runtime(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
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
                "runpodctl": {"version": "v2.7.2"},
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
                    "network_volume_id": "volume-1",
                    "volume_mount_path": "/workspace",
                    "ports": ["22/tcp"],
                    "terminate_after": "2h",
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
                    "remote_artifact_root": "/workspace/mantis/artifacts/smoke",
                    "local_artifact_root": str(tmp_path / "artifacts"),
                    "receipt_root": str(tmp_path / "receipts"),
                },
                "commands": {
                    "remote_train": ["just", "smoke"],
                    "local_validate": [
                        "test",
                        "-f",
                        str(tmp_path / "artifacts" / "train-result.json"),
                    ],
                },
            }
        )
    )
    return runtime


def _environment(tmp_path: Path, *, curl_body: str) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"
    _write_executable(bin_dir / "curl", f"printf '%s' '{curl_body}'\n")
    _write_executable(
        bin_dir / "runpodctl",
        f"printf 'runpodctl %s\\n' \"$*\" >> {call_log}\nprintf 'v2.7.2\\n'\n",
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
  "version ") printf 'v2.7.2\\n' ;;
  "pod create")
    printf '%s\\n' '{{"id":"pod-123","name":"mantis-smoke","status":"PROVISIONING","cost":1.5}}'
    ;;
  "ssh connect") printf 'ssh root@127.0.0.1 -p 2222\\n' ;;
  "pod delete") touch {deleted}; printf '%s\\n' '{{"id":"pod-123","deleted":true}}' ;;
  "pod get")
    if [[ -f {deleted} ]]; then exit 1; fi
    printf '%s\\n' '{{"id":"pod-123","name":"mantis-smoke","status":"RUNNING","cost":1.5}}'
    ;;
  "billing pods") printf '%s\\n' '[{{"podId":"pod-123","amount":0.01}}]' ;;
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
destination="${{@: -1}}"
if [[ "$destination" != root@* && "$*" == *"root@127.0.0.1:"* ]]; then
  mkdir -p "$destination"
  printf '{{"status":"complete"}}\\n' > "$destination/train-result.json"
fi
""",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["TEST_RUNPOD_API_KEY"] = "secret"
    return env, call_log


def _receipt(runtime: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "pod_id": "pod-123",
        "run_id": "mantis-smoke",
        "request_digest": "b" * 64,
        "runpodctl_version": "v2.7.2",
        "api_v2_openapi_sha256": "a" * 64,
        "image": "runpod/pytorch:official",
        "gpu_id": "NVIDIA A100 80GB PCIe",
        "network_volume_id": "volume-1",
        "hourly_price_usd": 1.5,
        "created_at": "2026-07-23T12:00:00Z",
        "termination_deadline": "2h",
        "experiment_sha256": "c" * 64,
        "runtime_sha256": sha256(runtime.read_bytes()).hexdigest(),
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
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('[experiment]\nname = "smoke"\n')
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
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('[experiment]\nname = "smoke"\n')
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
    assert not call_log.exists()


def test_train_runs_and_always_deletes_exact_pod(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('[experiment]\nname = "smoke"\n')
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
    assert receipt["runpodctl_version"] == "v2.7.2"
    assert receipt["gpu_id"] == "NVIDIA A100 80GB PCIe"
    assert receipt["hourly_price_usd"] == 1.5
    assert receipt["termination_deadline"] == "2h"
    calls = call_log.read_text()
    assert "runpodctl pod create" in calls
    assert "count=1" in calls
    assert "cloud=SECURE" in calls
    assert "--terminate-after=2h" in calls
    assert "rsync" in calls
    assert "ssh" in calls
    assert "runpodctl pod delete pod-123" in calls
    assert calls.index("runpodctl pod create") < calls.index("runpodctl pod delete pod-123")
    assert (tmp_path / "artifacts" / "train-result.json").is_file()


def test_recover_never_creates_and_deletes_receipted_pod(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    receipt = tmp_path / "run-receipt.json"
    receipt.write_text(json.dumps(_receipt(runtime)))
    env, call_log = _success_environment(tmp_path)

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
    receipt = tmp_path / "run-receipt.json"
    receipt.write_text(json.dumps(_receipt(runtime)))
    env, call_log = _success_environment(tmp_path)

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
    assert "runpodctl ssh connect pod-123" in calls
    assert "pod create" not in calls
    assert "pod delete" not in calls
    assert "curl " not in calls


def test_recover_deletes_exact_pod_when_status_read_fails(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    receipt = tmp_path / "run-receipt.json"
    receipt.write_text(json.dumps(_receipt(runtime)))
    env, call_log = _success_environment(tmp_path)
    _write_executable(
        tmp_path / "bin" / "runpodctl",
        f"""
printf 'runpodctl %s\\n' "$*" >> {call_log}
case "${{1:-}} ${{2:-}}" in
  "version ") printf 'v2.7.2\\n' ;;
  "pod get") exit 1 ;;
  "pod delete") printf '%s\\n' '{{"id":"pod-123","deleted":true}}' ;;
  "billing pods") printf '%s\\n' '[{{"podId":"pod-123","amount":0.01}}]' ;;
  *) exit 9 ;;
esac
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

    assert completed.returncode != 0
    calls = call_log.read_text()
    assert "pod create" not in calls
    assert "runpodctl pod delete pod-123" in calls


def test_train_rejects_overpriced_gpu_before_create(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('[experiment]\nname = "smoke"\n')
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
    assert not call_log.exists()


def test_train_deletes_exact_pod_after_remote_failure(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('[experiment]\nname = "smoke"\n')
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
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('[experiment]\nname = "smoke"\n')
    runtime = _runtime(tmp_path)
    env, call_log = _success_environment(tmp_path)
    deleted = tmp_path / "deleted"
    _write_executable(
        tmp_path / "bin" / "runpodctl",
        f"""
printf 'runpodctl %s\\n' "$*" >> {call_log}
case "${{1:-}} ${{2:-}}" in
  "version ") printf 'v2.7.2\\n' ;;
  "pod create") printf '%s\\n' '{{"id":"pod-123","name":"mantis-smoke","status":"PROVISIONING"}}' ;;
  "pod delete") touch {deleted}; printf '%s\\n' '{{"id":"pod-123","deleted":true}}' ;;
  "pod get") [[ -f {deleted} ]] && exit 1; printf '%s\\n' '{{"id":"pod-123"}}' ;;
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
    assert "provider_create_schema_or_price_invalid" in completed.stderr
    calls = call_log.read_text()
    assert calls.count("runpodctl pod create") == 1
    assert "runpodctl pod delete pod-123" in calls


def test_train_rejects_inline_api_key_before_provider_access(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('[experiment]\nname = "smoke"\n')
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
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('[experiment]\nname = "smoke"\n')
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
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('[experiment]\nname = "smoke"\n')
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
    assert not call_log.exists()


def test_train_deletes_exact_pod_after_transfer_failure(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('[experiment]\nname = "smoke"\n')
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
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('[experiment]\nname = "smoke"\n')
    runtime = _runtime(tmp_path)
    payload = json.loads(runtime.read_text())
    payload["commands"]["local_validate"] = ["false"]
    runtime.write_text(json.dumps(payload))
    env, call_log = _success_environment(tmp_path)

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
    experiment = tmp_path / "experiment.toml"
    experiment.write_text('[experiment]\nname = "smoke"\n')
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
    os.killpg(process.pid, signal.SIGTERM)
    process.communicate(timeout=10)

    assert process.returncode != 0
    calls = _wait_for_call(call_log, "runpodctl billing pods --pod-id pod-123")
    assert "runpodctl pod delete pod-123" in calls
    assert "runpodctl billing pods --pod-id pod-123" in calls
