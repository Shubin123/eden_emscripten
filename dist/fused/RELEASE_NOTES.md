# Eden WebAssembly fused core

A WebAssembly build of Eden from the `fused` branch: the eden_emscripten web port
with Eden's multithreading and frame-pacing refactor merged and eight fixes
ported from Citron Neo.

## Benchmarks

<!-- benchmark table goes here -->

## What changed

- Multithreading refactor from upstream Eden (`989768aa46`): fixes core, GPU and thread synchronization and frame pacing.
- The timing thread is only woken when a new earliest event is scheduled, cutting wakeups when games poll with very short sleeps.
- GPU memory unmaps and sparse maps clear the big-page-continuous flag, fixing stale mappings.
- `DeleteCacheStorage` returns success instead of crashing games that call it (seen in Tears of the Kingdom).
- The audio control service no longer blocks at startup waiting for the settings service.
- Uninitialized texture descriptors fall back to a default format instead of asserting.
- Semaphore releases write the requested size instead of always 16 bytes, so they no longer overwrite adjacent memory, and are written as soon as earlier work completes.
- Rewritten guest code is no longer recompiled from a stale page snapshot, and instruction cache invalidation now applies to every CPU core.
- Shader attribute 31 is treated as generic on all four components, fixing its misclassification.

Credits, original authors and licensing are in `NOTICE.md`. The exact source
commit is in `SOURCE.txt`.

## How to use

1. Download the release zip and extract it.
2. Either serve the extracted files from a web server that sends
   `Cross-Origin-Opener-Policy: same-origin` and
   `Cross-Origin-Embedder-Policy: require-corp` (needed for threads and
   SharedArrayBuffer), or copy them into the eden_web frontend by running
   `npm run sync -- <dir>` in the eden_web repository, which places them in
   `eden_web/web/runtime/`.
3. Supply your own keys, firmware and games. None are included.
