# syntax=docker/dockerfile:1.7
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.30@sha256:173b74e36e4bddd0546771a7041cae9f65b3ad28964eb22a2853bca944b2c67a
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04@sha256:4d242f206abc4b9588a6506cce2d88932cc879849395aae3785075179718cc49

FROM ${UV_IMAGE} AS uv
FROM ${CUDA_IMAGE}

ARG CUDA_IMAGE
ARG UV_IMAGE
ARG SOURCE_REVISION
ARG SOURCE_TREE
ARG LOCK_SHA256
ARG IMAGE_CONTRACT_SHA256

LABEL org.opencontainers.image.source="https://github.com/lgbarn/mantis-foundation-model" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.base.name="${CUDA_IMAGE}" \
      io.mantis.uv-image="${UV_IMAGE}" \
      io.mantis.source-tree="${SOURCE_TREE}" \
      io.mantis.lock-sha256="${LOCK_SHA256}" \
      io.mantis.image-contract-sha256="${IMAGE_CONTRACT_SHA256}"

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    MANTIS_BASE_IMAGE="${CUDA_IMAGE}" \
    MANTIS_SOURCE_REVISION="${SOURCE_REVISION}" \
    MANTIS_SOURCE_TREE="${SOURCE_TREE}" \
    MANTIS_LOCK_SHA256="${LOCK_SHA256}" \
    MANTIS_IMAGE_CONTRACT_SHA256="${IMAGE_CONTRACT_SHA256}" \
    MANTIS_UV_VERSION="0.11.30" \
    PATH="/opt/mantis/.venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates=20260601~24.04.1 \
       git=1:2.43.0-1ubuntu7.3 \
       openssh-server=1:9.6p1-3ubuntu13.18 \
       python3.12=3.12.3-1ubuntu0.15 \
       python3.12-venv=3.12.3-1ubuntu0.15 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.12 /usr/local/bin/python \
    && mkdir -p /run/sshd /root/.ssh \
    && chmod 0700 /root/.ssh \
    && printf '%s\n' \
       'PasswordAuthentication no' \
       'PermitRootLogin prohibit-password' \
       'AllowTcpForwarding yes' \
       'GatewayPorts no' \
       'X11Forwarding no' \
       >> /etc/ssh/sshd_config \
    && rm -f /etc/ssh/ssh_host_*

COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /opt/mantis
COPY pyproject.toml uv.lock ./
COPY mantis-v2/pyproject.toml mantis-v2/pyproject.toml
COPY mantis-v2/src mantis-v2/src
RUN --mount=type=cache,target=/root/.cache/uv \
    test "$(sha256sum uv.lock | cut -d ' ' -f 1)" = "${LOCK_SHA256}" \
    && uv sync --frozen --no-dev --all-packages --python /usr/bin/python3.12 \
    && uv --version \
    && python --version \
    && git --version \
    && ssh -V \
    && python -c 'import torch; print(torch.__version__, torch.version.cuda)'

COPY infra/runpod/scripts/container-entrypoint.sh /usr/local/bin/mantis-entrypoint
RUN chmod 0755 /usr/local/bin/mantis-entrypoint

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/mantis-entrypoint"]
CMD ["/usr/sbin/sshd", "-D", "-e"]
