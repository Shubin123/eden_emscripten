#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Builds eden_archive.{js,wasm}, the libarchive decoder the browser frontend uses to load
# games from 7z, RAR, tar and other compressed archives.
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
work="$root/build-web/archive"
prefix="$work/prefix"
out="$root/build-web/bin"

if ! command -v emcc >/dev/null 2>&1; then
  if [ -f "$HOME/emsdk/emsdk_env.sh" ]; then
    source "$HOME/emsdk/emsdk_env.sh"
  elif [ -n "${EMSDK:-}" ] && [ -f "$EMSDK/emsdk_env.sh" ]; then
    source "$EMSDK/emsdk_env.sh"
  else
    echo "Error: emcc not found and emsdk_env.sh not located." >&2
    exit 1
  fi
fi

jobs="${EDEN_BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"
mkdir -p "$work/src" "$prefix" "$out"

fetch() {
  local name=$1 url=$2 sha=$3
  local archive="$work/src/$(basename "$url")"
  if [ ! -d "$work/src/$name" ]; then
    curl -fsSL -o "$archive" "$url"
    echo "$sha  $archive" | sha256sum -c --quiet
    tar -xf "$archive" -C "$work/src"
  fi
}

fetch xz-5.8.1 \
  https://github.com/tukaani-project/xz/releases/download/v5.8.1/xz-5.8.1.tar.xz \
  0b54f79df85912504de0b14aec7971e3f964491af1812d83447005807513cd9e
fetch zstd-1.5.7 \
  https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz \
  eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
fetch libarchive-3.8.1 \
  https://github.com/libarchive/libarchive/releases/download/v3.8.1/libarchive-3.8.1.tar.xz \
  19f917d42d530f98815ac824d90c7eaf648e9d9a50e4f309c812457ffa5496b5

# zlib and bzip2 come from Emscripten ports.
embuilder build zlib bzip2 >/dev/null
sysroot=$(em-config CACHE)/sysroot

configure() {
  local name=$1 src=$2
  shift 2
  emcmake cmake -S "$src" -B "$work/build/$name" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DBUILD_SHARED_LIBS=OFF \
    "$@" >/dev/null
  cmake --build "$work/build/$name" --parallel "$jobs" >/dev/null
  cmake --install "$work/build/$name" >/dev/null
}

configure xz "$work/src/xz-5.8.1" \
  -DXZ_NLS=OFF -DXZ_THREADS=no -DXZ_SANDBOX=no -DXZ_DOC=OFF -DXZ_TOOL_XZ=OFF \
  -DXZ_TOOL_XZDEC=OFF -DXZ_TOOL_LZMADEC=OFF -DXZ_TOOL_LZMAINFO=OFF -DXZ_TOOL_SCRIPTS=OFF \
  -DXZ_TOOL_SYMLINKS=OFF -DXZ_TOOL_SYMLINKS_LZMA=OFF -DXZ_MICROLZMA_ENCODER=OFF

configure zstd "$work/src/zstd-1.5.7/build/cmake" \
  -DZSTD_BUILD_PROGRAMS=OFF -DZSTD_BUILD_SHARED=OFF -DZSTD_BUILD_STATIC=ON \
  -DZSTD_BUILD_TESTS=OFF -DZSTD_MULTITHREAD_SUPPORT=OFF -DZSTD_LEGACY_SUPPORT=OFF

configure libarchive "$work/src/libarchive-3.8.1" \
  -DCMAKE_INSTALL_LIBDIR=lib -DENABLE_TAR=OFF -DENABLE_CPIO=OFF -DENABLE_CAT=OFF -DENABLE_UNZIP=OFF -DENABLE_TEST=OFF \
  -DENABLE_WERROR=OFF -DENABLE_ACL=OFF -DENABLE_XATTR=OFF -DENABLE_ICONV=OFF \
  -DENABLE_OPENSSL=OFF -DENABLE_MBEDTLS=OFF -DENABLE_NETTLE=OFF -DENABLE_CNG=OFF \
  -DENABLE_LIBB2=OFF -DENABLE_LZ4=OFF -DENABLE_LZO=OFF -DENABLE_LIBXML2=OFF \
  -DENABLE_EXPAT=OFF -DENABLE_PCREPOSIX=OFF -DENABLE_PCRE2POSIX=OFF -DENABLE_LIBGCC=OFF \
  -DENABLE_ZLIB=ON -DENABLE_BZip2=ON -DENABLE_LZMA=ON -DENABLE_ZSTD=ON \
  -DZLIB_INCLUDE_DIR="$sysroot/include" -DZLIB_LIBRARY="$sysroot/lib/wasm32-emscripten/libz.a" \
  -DBZIP2_INCLUDE_DIR="$sysroot/include" \
  -DBZIP2_LIBRARIES="$sysroot/lib/wasm32-emscripten/libbz2.a" \
  -DLIBLZMA_INCLUDE_DIR="$prefix/include" -DLIBLZMA_LIBRARY="$prefix/lib/liblzma.a" \
  -DZSTD_INCLUDE_DIR="$prefix/include" -DZSTD_LIBRARY="$prefix/lib/libzstd.a"

emcc -O2 "$root/src/web_archive/eden_archive.c" \
  -I"$prefix/include" \
  "$prefix/lib/libarchive.a" "$prefix/lib/liblzma.a" "$prefix/lib/libzstd.a" \
  -sUSE_ZLIB=1 -sUSE_BZIP2=1 \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createEdenArchive \
  -sENVIRONMENT=worker,node -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB \
  -sEXPORTED_FUNCTIONS=_malloc,_free -sEXPORTED_RUNTIME_METHODS=HEAPU8,UTF8ToString \
  -o "$out/eden_archive.js"

echo "Built $out/eden_archive.js and eden_archive.wasm."
