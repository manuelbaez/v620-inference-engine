#!/usr/bin/env bash
# Build the qw-engine Docker image on the serving box from this checkout
# (the repo is private: the source goes over ssh as a tarball, no clone).
#   scripts/build-image.sh [ssh-target] [tag]
# Default target: the llm-backend-amd container through main-srv's incus.
set -euo pipefail

SRC=$(cd "$(dirname "$0")/.." && pwd)
TAG=${2:-qw-engine:$(git -C "$SRC" rev-parse --short HEAD)}
TARGET=${1:-incus}

tarball() { tar -C "$SRC" --exclude ./build --exclude ./.git --exclude __pycache__ -cf - .; }
build_cmd="docker build -t $TAG -t qw-engine:latest -f docker/Dockerfile -"

if [ "$TARGET" = incus ]; then
  # docker build reads the context as a tar on stdin
  tarball | ssh main-srv.local.net "incus exec llm-backend-amd --project llms -- $build_cmd"
else
  tarball | ssh "$TARGET" "$build_cmd"
fi
echo "built $TAG (and qw-engine:latest) on $TARGET"
