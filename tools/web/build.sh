#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
# EDEN_DEV=1 builds into build-web-dev and links with -O1 instead of -O3. Object code is the same
# -O3 as a release build, so emulation behaves identically; only the whole-program wasm-opt pass
# at link time is lighter (seconds instead of minutes, larger wasm). Never publish a dev build.
dev="${EDEN_DEV:-0}"
build_dir="$root/build-web"
link_flags=""
if [ "$dev" = 1 ]; then
  build_dir="$root/build-web-dev"
  link_flags="-O1"
fi

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

# Memory64 + shared-memory threads must apply to every object. The Emscripten toolchain file reads
# CMAKE_C_FLAGS to pick wasm64, so they have to be passed on the command line.
web_flags="-sMEMORY64=1 -pthread -msimd128 -fwasm-exceptions"

echo "Configuring Eden WebAssembly build..."
emcmake cmake -S "$root" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_C_FLAGS=$web_flags" \
  "-DCMAKE_CXX_FLAGS=$web_flags" \
  "-DCMAKE_EXE_LINKER_FLAGS=$web_flags $link_flags" \
  -DENABLE_QT=OFF \
  -DENABLE_WEB_SERVICE=OFF \
  -DENABLE_LIBUSB=OFF \
  -DYUZU_TESTS=OFF \
  -DENABLE_CUBEB=OFF \
  -DUSE_DISCORD_PRESENCE=OFF \
  "$@"

[ "${EDEN_CONFIGURE_ONLY:-0}" = 1 ] && exit 0

echo "Building Eden WebAssembly targets..."
cmake --build "$build_dir" --target eden_web eden_headless --parallel "${EDEN_BUILD_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

echo "Build complete. Artifacts are in $build_dir/bin; run 'npm run sync' in ../eden_web to install them."
