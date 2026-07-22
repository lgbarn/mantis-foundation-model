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
mkdir -p "$(dirname "$output")"
temporary="${output}.tmp.$$"
trap 'rm -f "$temporary"' EXIT HUP INT TERM
docker run --rm --entrypoint mantis-v2 "$image_reference" \
    runpod-image-static-check > "$temporary"
ln "$temporary" "$output"
rm -f "$temporary"
trap - EXIT HUP INT TERM
