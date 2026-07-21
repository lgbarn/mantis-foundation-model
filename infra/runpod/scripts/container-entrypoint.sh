#!/bin/sh
set -eu

mkdir -p /run/sshd /root/.ssh
chmod 0700 /root/.ssh
ssh-keygen -A >/dev/null
exec "$@"
