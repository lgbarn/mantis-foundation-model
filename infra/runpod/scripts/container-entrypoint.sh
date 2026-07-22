#!/bin/sh
set -eu

mkdir -p /run/sshd /root/.ssh
chmod 0700 /root/.ssh
rm -f /root/.ssh/authorized_keys
if [ -n "${PUBLIC_KEY:-}" ]; then
    printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 0600 /root/.ssh/authorized_keys
    ssh-keygen -l -f /root/.ssh/authorized_keys >/dev/null
fi
ssh-keygen -A >/dev/null

if [ "$#" -gt 0 ] && [ "$1" != "/usr/sbin/sshd" ]; then
    /usr/sbin/sshd -D -e &
fi
exec "$@"
