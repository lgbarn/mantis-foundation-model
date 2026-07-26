from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from mantis_v2.runpod_s3 import AwsCliS3TransferAdapter, RunpodS3Error


class Runner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.timeouts: list[int] = []
        self.responses: list[subprocess.CompletedProcess[str]] = []

    def __call__(
        self, args: list[str], environment: dict[str, str], timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, environment))
        self.timeouts.append(timeout_seconds)
        return self.responses.pop(0)


def _adapter(runner: Runner) -> AwsCliS3TransferAdapter:
    return AwsCliS3TransferAdapter(
        aws_binary="/usr/local/bin/aws",
        datacenter_id="US-GA-2",
        volume_id="volume-123",
        access_key_id="access-secret",
        secret_access_key="secret-secret",
        runner=runner,
    )


def test_s3_adapter_heads_and_uploads_without_secrets_in_argv(tmp_path: Path) -> None:
    runner = Runner()
    runner.responses.extend(
        (
            subprocess.CompletedProcess([], 0, json.dumps({"ContentLength": 4}), ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, json.dumps({"ContentLength": 4}), ""),
        )
    )
    adapter = _adapter(runner)
    source = tmp_path / "input.bin"
    source.write_bytes(b"data")

    assert adapter.head_object("transfer/incoming/object") is not None
    adapter.put_file("transfer/incoming/object", source)

    assert len(runner.calls) == 3
    for args, environment in runner.calls:
        assert "access-secret" not in " ".join(args)
        assert "secret-secret" not in " ".join(args)
        assert environment["AWS_ACCESS_KEY_ID"] == "access-secret"
        assert environment["AWS_SECRET_ACCESS_KEY"] == "secret-secret"
        assert environment["AWS_DEFAULT_REGION"] == "US-GA-2"
        assert environment["AWS_MAX_ATTEMPTS"] == "10"
        assert environment["AWS_RETRY_MODE"] == "standard"
        assert "https://s3api-us-ga-2.runpod.io" in args


def test_s3_adapter_uses_multipart_cli_for_large_files(tmp_path: Path) -> None:
    runner = Runner()
    source = tmp_path / "large.bin"
    source.touch()
    with source.open("r+b") as handle:
        handle.truncate(1_509_478_528)
    runner.responses.extend(
        (
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess(
                [], 0, json.dumps({"ContentLength": source.stat().st_size}), ""
            ),
        )
    )

    _adapter(runner).put_file("transfer/incoming/large.bin", source)

    upload_args = runner.calls[0][0]
    assert upload_args[:3] == ["/usr/local/bin/aws", "s3", "cp"]
    assert upload_args[3] == str(source)
    assert upload_args[4] == "s3://volume-123/transfer/incoming/large.bin"
    assert "--no-progress" in upload_args
    assert upload_args[upload_args.index("--cli-read-timeout") + 1] == "840"
    assert runner.timeouts[0] == 840
    assert runner.calls[1][0][1:3] == ["s3api", "head-object"]


def test_s3_adapter_distinguishes_absent_object_from_provider_failure() -> None:
    runner = Runner()
    runner.responses.append(
        subprocess.CompletedProcess([], 254, "", "An error occurred (404) when calling HeadObject")
    )
    assert _adapter(runner).head_object("missing") is None

    runner.responses.append(
        subprocess.CompletedProcess([], 254, "", "An error occurred (403) when calling HeadObject")
    )
    with pytest.raises(RunpodS3Error, match="head-object failed"):
        _adapter(runner).head_object("forbidden")


def test_s3_adapter_rejects_unsafe_identity_or_key() -> None:
    runner = Runner()
    with pytest.raises(RunpodS3Error, match="datacenter"):
        AwsCliS3TransferAdapter(
            aws_binary="aws",
            datacenter_id="https://attacker.invalid",
            volume_id="volume-123",
            access_key_id="a",
            secret_access_key="b",
            runner=runner,
        )
    adapter = _adapter(runner)
    with pytest.raises(RunpodS3Error, match="object key"):
        adapter.head_object("../escape")


def test_s3_adapter_lists_paginated_prefix_without_accepting_foreign_keys() -> None:
    runner = Runner()
    runner.responses.extend(
        (
            subprocess.CompletedProcess(
                [],
                0,
                json.dumps(
                    {
                        "Contents": [{"Key": "mantis/run/a", "Size": 3, "ETag": "opaque"}],
                        "IsTruncated": True,
                        "NextContinuationToken": "next-token",
                    }
                ),
                "",
            ),
            subprocess.CompletedProcess(
                [],
                0,
                json.dumps(
                    {
                        "Contents": [{"Key": "mantis/run/b", "Size": 4}],
                        "IsTruncated": False,
                    }
                ),
                "",
            ),
        )
    )

    objects = _adapter(runner).list_objects("mantis/run")

    assert {key: value.size for key, value in objects.items()} == {
        "mantis/run/a": 3,
        "mantis/run/b": 4,
    }
    assert "--continuation-token" in runner.calls[1][0]
