#!/usr/bin/env bash
# Sync the source tree to the GPU dev container and build it there.
# The container has no rsync, so this is tar over ssh.
set -euo pipefail

SRC=$(cd "$(dirname "$0")" && pwd)
DEST=${DEST:-hermes@llm-experiments.local.net}
REMOTE_DIR=${REMOTE_DIR:-inference-engine}

tar -C $SRC --exclude build --exclude legacy --exclude .git -cf - . |
  ssh -o BatchMode=yes $DEST "mkdir -p ~/$REMOTE_DIR && tar -C ~/$REMOTE_DIR -xf - && make -C ~/$REMOTE_DIR -j48 ${MAKE_TARGET:-all gpu} 2>&1 | grep -E 'error|warning' || true"
echo "deploy: synced and built on $DEST:~/$REMOTE_DIR"
