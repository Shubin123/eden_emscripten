#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Packages a built web runtime with its license, notice and source reference into a
# release directory and zip.
# Usage: package-release.sh <runtime-dir> <out-dir> [benchmark-report.md]
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <runtime-dir> <out-dir> [benchmark-report.md]" >&2
  exit 1
fi

root=$(cd "$(dirname "$0")/../.." && pwd)
runtime=$1
out=$2
report=${3:-}
repo_url="https://github.com/Shubin123/eden_emscripten"
files=(eden.js eden.wasm eden_archive.js eden_archive.wasm)

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: zip not found." >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha256=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
  sha256=(shasum -a 256)
else
  echo "Error: neither sha256sum nor shasum found." >&2
  exit 1
fi

for f in "${files[@]}"; do
  if [ ! -f "$runtime/$f" ]; then
    echo "Error: missing runtime file $runtime/$f" >&2
    exit 1
  fi
done

if [ -n "$report" ] && [ ! -f "$report" ]; then
  echo "Error: benchmark report $report not found." >&2
  exit 1
fi

# The release must correspond to a committed source revision.
if [ -n "$(git -C "$root" status --porcelain)" ]; then
  if [ "${EDEN_ALLOW_DIRTY:-0}" = 1 ]; then
    echo "Warning: working tree is dirty; SOURCE.txt will not match the packaged build." >&2
  else
    echo "Error: working tree is dirty. Commit or stash changes first." >&2
    exit 1
  fi
fi

if [ -e "$out" ]; then
  echo "Error: $out already exists." >&2
  exit 1
fi

commit=$(git -C "$root" rev-parse HEAD)
mkdir -p "$out"
out=$(cd "$out" && pwd)

for f in "${files[@]}"; do
  cp "$runtime/$f" "$out/"
done
cp "$root/LICENSE.txt" "$out/"
cp -R "$root/LICENSES" "$out/"
cp "$root/dist/fused/NOTICE.md" "$out/"
if [ -n "$report" ]; then
  cp "$report" "$out/BENCHMARK.md"
fi

cat > "$out/SOURCE.txt" <<SRC
Corresponding source for this release:
Repository: $repo_url
Commit: $commit
SRC

(cd "$out" && find . -type f ! -name SHA256SUMS | sed 's|^\./||' | LC_ALL=C sort \
  | while IFS= read -r f; do "${sha256[@]}" "$f"; done > SHA256SUMS)

rm -f "$out.zip"
(cd "$(dirname "$out")" && zip -qr "$(basename "$out").zip" "$(basename "$out")")

echo "Packaged $out and $out.zip (source $commit)"
