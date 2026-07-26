"""Credential-isolated AWS CLI adapter for RunPod network-volume S3 access."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath

from mantis_v2.transfer_bundle import RemoteObject


class RunpodS3Error(RuntimeError):
    """Raised when RunPod S3 staging cannot be proven complete."""


Runner = Callable[[list[str], dict[str, str], int], subprocess.CompletedProcess[str]]
_MULTIPART_THRESHOLD_BYTES = 500 * 1024 * 1024
_MIN_UPLOAD_RATE_BYTES_PER_SECOND = 2 * 1024 * 1024


def _default_runner(
    args: list[str], environment: dict[str, str], timeout_seconds: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=timeout_seconds,
    )


def _safe_key(value: str) -> str:
    parsed = PurePosixPath(value)
    if not value or parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != value:
        raise RunpodS3Error("object key is not canonical and relative")
    return value


class AwsCliS3TransferAdapter:
    """Use a fixed RunPod endpoint while keeping credentials out of argv and files."""

    def __init__(
        self,
        *,
        aws_binary: str | Path,
        datacenter_id: str,
        volume_id: str,
        access_key_id: str,
        secret_access_key: str,
        runner: Runner = _default_runner,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Z]{2}-[A-Z0-9]+-[0-9]+", datacenter_id):
            raise RunpodS3Error("datacenter identity is invalid")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", volume_id):
            raise RunpodS3Error("volume identity is invalid")
        if not access_key_id or not secret_access_key:
            raise RunpodS3Error("RunPod S3 credentials are required")
        self._binary = str(aws_binary)
        self._datacenter_id = datacenter_id
        self._volume_id = volume_id
        self._endpoint = f"https://s3api-{datacenter_id.lower()}.runpod.io"
        self._environment = dict(os.environ if environ is None else environ)
        self._environment.update(
            {
                "AWS_ACCESS_KEY_ID": access_key_id,
                "AWS_SECRET_ACCESS_KEY": secret_access_key,
                "AWS_DEFAULT_REGION": datacenter_id,
                "AWS_EC2_METADATA_DISABLED": "true",
                "AWS_MAX_ATTEMPTS": "10",
                "AWS_RETRY_MODE": "standard",
            }
        )
        self._runner = runner

    def _args(self, operation: str, key: str) -> list[str]:
        return [
            self._binary,
            "s3api",
            operation,
            "--bucket",
            self._volume_id,
            "--key",
            _safe_key(key),
            "--endpoint-url",
            self._endpoint,
            "--region",
            self._datacenter_id,
            "--no-cli-pager",
        ]

    def _run(
        self, args: list[str], *, timeout_seconds: int = 300
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self._runner(args, dict(self._environment), timeout_seconds)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RunpodS3Error("AWS CLI execution failed") from exc

    def head_object(self, key: str) -> RemoteObject | None:
        completed = self._run(self._args("head-object", key))
        if completed.returncode != 0:
            if "(404)" in completed.stderr or "Not Found" in completed.stderr:
                return None
            raise RunpodS3Error("head-object failed")
        try:
            payload = json.loads(completed.stdout)
            size = payload["ContentLength"]
            etag = payload.get("ETag")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RunpodS3Error("head-object returned invalid JSON") from exc
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RunpodS3Error("head-object returned invalid size")
        if etag is not None and not isinstance(etag, str):
            raise RunpodS3Error("head-object returned invalid ETag")
        return RemoteObject(size=size, etag=etag)

    def put_file(self, key: str, source: Path) -> None:
        try:
            expected_size = source.stat().st_size
        except OSError as exc:
            raise RunpodS3Error("upload source is unavailable") from exc
        if expected_size >= _MULTIPART_THRESHOLD_BYTES:
            upload_timeout = min(
                1800,
                max(
                    300,
                    120
                    + (expected_size + _MIN_UPLOAD_RATE_BYTES_PER_SECOND - 1)
                    // _MIN_UPLOAD_RATE_BYTES_PER_SECOND,
                ),
            )
            completed = self._run(
                [
                    self._binary,
                    "s3",
                    "cp",
                    str(source),
                    f"s3://{self._volume_id}/{_safe_key(key)}",
                    "--endpoint-url",
                    self._endpoint,
                    "--region",
                    self._datacenter_id,
                    "--no-cli-pager",
                    "--cli-read-timeout",
                    str(upload_timeout),
                    "--no-progress",
                    "--only-show-errors",
                ],
                timeout_seconds=upload_timeout,
            )
        else:
            completed = self._run([*self._args("put-object", key), "--body", str(source)])
        if completed.returncode != 0:
            raise RunpodS3Error("upload failed")
        remote = self.head_object(key)
        if remote is None or remote.size != expected_size:
            raise RunpodS3Error("uploaded object size verification failed")

    def put_bytes(self, key: str, value: bytes) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix="mantis-runpod-s3-")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            self.put_file(key, Path(temporary))
        finally:
            Path(temporary).unlink(missing_ok=True)

    def get_bytes(self, key: str, *, maximum_size: int = 2 * 1024 * 1024) -> bytes | None:
        remote = self.head_object(key)
        if remote is None:
            return None
        if remote.size > maximum_size:
            raise RunpodS3Error("object exceeds bounded in-memory download size")
        descriptor, temporary = tempfile.mkstemp(prefix="mantis-runpod-s3-read-")
        os.close(descriptor)
        path = Path(temporary)
        try:
            completed = self._run([*self._args("get-object", key), str(path)])
            if completed.returncode != 0:
                raise RunpodS3Error("get-object failed")
            return path.read_bytes()
        except OSError as exc:
            raise RunpodS3Error("downloaded object is unreadable") from exc
        finally:
            path.unlink(missing_ok=True)

    def get_file(self, key: str, destination: Path) -> Path | None:
        remote = self.head_object(key)
        if remote is None:
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(descriptor)
        path = Path(temporary)
        path.unlink()
        try:
            completed = self._run([*self._args("get-object", key), str(path)])
            if completed.returncode != 0:
                raise RunpodS3Error("get-object failed")
            if not path.is_file() or path.stat().st_size != remote.size:
                raise RunpodS3Error("downloaded object size verification failed")
            if destination.exists():
                if (
                    not destination.is_file()
                    or destination.stat().st_size != remote.size
                    or _sha256(destination) != _sha256(path)
                ):
                    raise RunpodS3Error("immutable download destination differs")
            else:
                os.link(path, destination)
            return destination
        finally:
            path.unlink(missing_ok=True)

    def list_objects(self, prefix: str) -> dict[str, RemoteObject]:
        canonical_prefix = _safe_key(prefix.rstrip("/")) + "/"
        objects: dict[str, RemoteObject] = {}
        continuation: str | None = None
        while True:
            args = [
                self._binary,
                "s3api",
                "list-objects-v2",
                "--bucket",
                self._volume_id,
                "--prefix",
                canonical_prefix,
                "--endpoint-url",
                self._endpoint,
                "--region",
                self._datacenter_id,
                "--no-cli-pager",
            ]
            if continuation is not None:
                args.extend(("--continuation-token", continuation))
            completed = self._run(args)
            if completed.returncode != 0:
                raise RunpodS3Error("list-objects-v2 failed")
            try:
                payload = json.loads(completed.stdout)
                contents = payload.get("Contents", [])
            except (json.JSONDecodeError, AttributeError) as exc:
                raise RunpodS3Error("list-objects-v2 returned invalid JSON") from exc
            if not isinstance(contents, list):
                raise RunpodS3Error("list-objects-v2 returned invalid contents")
            for item in contents:
                if not isinstance(item, dict):
                    raise RunpodS3Error("list-objects-v2 returned invalid object")
                key = item.get("Key")
                size = item.get("Size")
                etag = item.get("ETag")
                if (
                    not isinstance(key, str)
                    or not key.startswith(canonical_prefix)
                    or not isinstance(size, int)
                    or isinstance(size, bool)
                    or size < 0
                    or (etag is not None and not isinstance(etag, str))
                    or key in objects
                ):
                    raise RunpodS3Error("list-objects-v2 returned invalid object")
                objects[key] = RemoteObject(size=size, etag=etag)
            truncated = payload.get("IsTruncated", False)
            if truncated is False:
                return objects
            continuation = payload.get("NextContinuationToken")
            if truncated is not True or not isinstance(continuation, str) or not continuation:
                raise RunpodS3Error("list-objects-v2 pagination is invalid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
