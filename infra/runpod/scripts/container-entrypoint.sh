#!/bin/sh
set -eu

mkdir -p /run/sshd /root/.ssh
chmod 0700 /root/.ssh
ssh-keygen -A >/dev/null

if [ "$#" -gt 0 ] && [ "$1" != "/usr/sbin/sshd" ]; then
    /usr/sbin/sshd -D -e &
fi
exec "$@"
