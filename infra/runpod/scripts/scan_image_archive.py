from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

FORBIDDEN_SUFFIXES = (".dbn", ".dbn.zst", ".parquet", ".pt", ".pth", ".safetensors")
FORBIDDEN_PARTS = {"artifacts", "checkpoints", ".git"}
SECRET_PATTERN = re.compile(
    rb"(?:RUNPOD_API_KEY|RUNPOD_S3_SECRET_ACCESS_KEY|AWS_SECRET_ACCESS_KEY|REGISTRY_PASSWORD)\s*=\s*\S+"
)


def scan_archive(path: Path, history: bytes) -> tuple[int, list[str]]:
    violations: set[str] = set()
    layer_count = 0
    if SECRET_PATTERN.search(history):
        violations.add("secret-like assignment in image history")
    with tarfile.open(path) as image:
        manifest_member = image.extractfile("manifest.json")
        if manifest_member is None:
            raise ValueError("Docker archive is missing manifest.json")
        manifest = json.load(manifest_member)
        if not isinstance(manifest, list) or len(manifest) != 1:
            raise ValueError("Docker archive must contain exactly one image")
        for layer_name in manifest[0]["Layers"]:
            layer_member = image.extractfile(layer_name)
            if layer_member is None:
                raise ValueError(f"Docker archive is missing layer {layer_name}")
            layer_count += 1
            with tarfile.open(fileobj=io.BytesIO(layer_member.read())) as layer:
                for member in layer.getmembers():
                    normalized = PurePosixPath(member.name.lstrip("./"))
                    lower_name = str(normalized).lower()
                    if lower_name.endswith(FORBIDDEN_SUFFIXES) or FORBIDDEN_PARTS.intersection(
                        normalized.parts
                    ):
                        violations.add(f"forbidden path in layer: {normalized}")
                    if member.isfile() and member.size <= 8 * 1024 * 1024:
                        handle = layer.extractfile(member)
                        content = handle.read() if handle is not None else b""
                        if b"-----BEGIN OPENSSH PRIVATE KEY-----" in content:
                            violations.add(f"private key material in layer: {normalized}")
                        if SECRET_PATTERN.search(content):
                            violations.add(f"secret-like assignment in layer: {normalized}")
    return layer_count, sorted(violations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--inspect", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    layers, violations = scan_archive(args.archive, args.history.read_bytes())
    inspect_sha256 = hashlib.sha256(args.inspect.read_bytes()).hexdigest()
    result = {
        "schema_version": 1,
        "image_reference": args.image,
        "image_inspect_sha256": inspect_sha256,
        "layers_scanned": layers,
        "passed": not violations,
        "violations": violations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    if violations:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
