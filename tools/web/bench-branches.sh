#!/usr/bin/env bash
# Benchmarks engine branches against a reference with the scripted game suite in ../eden_web.
#
#   tools/web/bench-branches.sh [--repeat N] [--only <scenario>]... <reference> <branch>...
#   tools/web/bench-branches.sh --repeat 3 main upstream-sync citron-ports
#
# Each branch is built once per commit into bench/<branch>-<sha>/bin (reused on later runs), synced
# into eden_web/web/runtime, and run through tests/games.mjs. The comparison table from
# tests/compare-games.mjs is printed at the end and saved as compare.md in the run directory. The
# checkout returns to the branch it started on, with that branch's runtime synced back.
# Needs EDEN_KEYS and EDEN_FIRMWARE (see tests/games.mjs) and a clean working tree.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
web="${EDEN_WEB_DIR:-$root/../eden_web}"
suite_args=()
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --repeat|--only) suite_args+=("$1" "$2"); shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
if [[ $# -lt 2 ]]; then
  echo "usage: $0 [--repeat N] [--only <scenario>]... <reference> <branch>..." >&2
  exit 2
fi
: "${EDEN_KEYS:?set EDEN_KEYS}" "${EDEN_FIRMWARE:?set EDEN_FIRMWARE}"
cd "$root"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "The working tree has changes; commit or stash them first." >&2
  exit 2
fi

start_branch="$(git rev-parse --abbrev-ref HEAD)"
run_dir="$web/artifacts/bench-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"
reports=()

runtime_for() {
  local branch="$1" sha
  sha="$(git rev-parse --short=10 "$branch")"
  local dir="$root/bench/$branch-$sha"
  if [[ ! -f "$dir/bin/eden.wasm" ]]; then
    echo "== Building $branch ($sha)" >&2
    git switch --quiet "$branch"
    tools/web/build.sh >"$run_dir/build-$branch.log" 2>&1 || { echo "Build of $branch failed; see $run_dir/build-$branch.log" >&2; git switch --quiet "$start_branch"; exit 1; }
    mkdir -p "$dir"
    rm -rf "$dir/bin"
    cp -R build-web/bin "$dir/bin"
  fi
  echo "$dir/bin"
}

restore() {
  git switch --quiet "$start_branch"
  (cd "$web" && npm run --silent sync -- "$(runtime_for "$start_branch")" >/dev/null) || true
}
trap restore EXIT

for branch in "$@"; do
  bin="$(runtime_for "$branch")"
  echo "== Running the game suite on $branch"
  (cd "$web" && npm run --silent sync -- "$bin" && node tests/games.mjs "${suite_args[@]}" --out "$run_dir/$branch") || true
  reports+=("$branch=$run_dir/$branch/games.json")
done

echo
(cd "$web" && node tests/compare-games.mjs --out "$run_dir/compare.md" "${reports[@]}")
