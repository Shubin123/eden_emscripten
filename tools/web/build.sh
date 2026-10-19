#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
build_dir="$root/build-web"

# Activate emsdk if emcc is not in PATH
if ! command -v emcc >/dev/null 2>&1; then
  if [ -f "$HOME/emsdk/emsdk_env.sh" ]; then
    echo "Activating EMSDK from $HOME/emsdk..."
    source "$HOME/emsdk/emsdk_env.sh"
  elif [ -n "${EMSDK:-}" ] && [ -f "$EMSDK/emsdk_env.sh" ]; then
    echo "Activating EMSDK from $EMSDK..."
    source "$EMSDK/emsdk_env.sh"
  else
    echo "Error: emcc not found and emsdk_env.sh not located." >&2
    exit 1
  fi
fi

echo "Configuring Eden WebAssembly build..."
emcmake cmake -S "$root" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_QT=OFF \
  -DENABLE_WEB_SERVICE=OFF \
  -DENABLE_LIBUSB=OFF \
  -DYUZU_TESTS=OFF \
  -DENABLE_CUBEB=OFF \
  -DUSE_DISCORD_PRESENCE=OFF \
  "$@"

echo "Building Eden WebAssembly targets..."
cmake --build "$build_dir" --parallel "${EDEN_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

echo "Build complete. Artifacts in $build_dir"
