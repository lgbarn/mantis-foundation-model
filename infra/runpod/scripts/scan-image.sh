#!/bin/sh
set -eu

if [ "$#" -ne 2 ] || [ -z "$1" ] || [ -z "$2" ]; then
    echo "usage: $0 IMAGE_REFERENCE OUTPUT_JSON" >&2
    exit 2
fi
if [ -e "$2" ]; then
    echo "error: output already exists: $2" >&2
    exit 2
fi

image_reference=$1
output=$2
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
docker save --output "$scratch/image.tar" "$image_reference"
docker image inspect "$image_reference" > "$scratch/inspect.json"
docker history --no-trunc "$image_reference" > "$scratch/history.txt"
uv run python infra/runpod/scripts/scan_image_archive.py \
    --archive "$scratch/image.tar" \
    --inspect "$scratch/inspect.json" \
    --history "$scratch/history.txt" \
    --image "$image_reference" \
    --output "$output"
