#!/bin/sh
set -eu

if [ "$#" -ne 1 ] || [ -z "$1" ]; then
    echo "usage: $0 IMAGE_REFERENCE" >&2
    exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
    echo "error: image builds require a clean committed worktree" >&2
    exit 2
fi

image_reference=$1
source_revision=$(git rev-parse HEAD)
source_tree=$(git rev-parse 'HEAD^{tree}')
lock_sha256=$(shasum -a 256 uv.lock | cut -d ' ' -f 1)
contract_sha256=$(shasum -a 256 infra/runpod/Dockerfile | cut -d ' ' -f 1)

docker build \
    --platform linux/amd64 \
    --file infra/runpod/Dockerfile \
    --tag "$image_reference" \
    --build-arg "SOURCE_REVISION=$source_revision" \
    --build-arg "SOURCE_TREE=$source_tree" \
    --build-arg "LOCK_SHA256=$lock_sha256" \
    --build-arg "IMAGE_CONTRACT_SHA256=$contract_sha256" \
    .
