# Browser widget bridge

The browser frontend lives in the sibling `eden_web` repository. Build artifacts
remain here and are installed there with `npm run sync`.

API version 3 adds SDL Web Audio output to the browser build and these exports:

| Export | Contract |
| --- | --- |
| `eden_get_audio_state()` | 0 before a stream opens, 1 when browser playback needs resuming, 2 when running, -1 when unavailable |
| `eden_resume_audio()` | Resume playback from a user gesture; failures go to the engine log |
| `eden_set_audio_volume(percent, muted)` | Clamp volume to 0–100 and apply volume/mute to the browser output gain |

Headless builds keep the null audio sink and do not advertise the audio operation.
The SDL pointer conversion accepts the Number pointers supplied by current
Emscripten EM_ASM and the BigInts supplied by older toolchains, including addresses
above 4 GB.

Live introspection getters use fixed buffers or immutable cached build information.
They must not allocate on the browser thread: a worker growing the heap can hold
the allocator lock while waiting for that thread to service a synchronous proxy.
Game metadata is copied under a short lock without allocating. Build metadata can
briefly be `{}` before the engine worker initializes it.

Uncaught C++ exceptions include their message and stack trace in browser/worker
logs. The frontend must stop querying a crashed runtime; a dead worker may have
held a lock when it failed.

API version 4 adds the browser ROM reader. `eden_register_browser_rom(path, size)`
registers one read-only file before boot. The filesystem delegates other paths,
including keys, firmware and saves, to the usual runtime filesystem.

| Export | Contract |
| --- | --- |
| `eden_get_rom_io_control()` | Address of the shared control structure |
| `eden_get_rom_io_buffer()` | Address of the fixed 4 MB read buffer |
| `eden_get_rom_io_buffer_size()` | Read buffer capacity in bytes |
| `eden_register_browser_rom(path, size)` | 0 on success, -1 for an invalid path or size; call before boot |

The control structure contains four 32-bit fields (`status`, `length`, `result`,
reserved), followed by a 64-bit unsigned byte offset at byte 16. Native readers
serialize requests, set status to 1 and wake the reader worker. That worker reads
the browser `File`, copies into the buffer, sets the result count and status 2
(or -1 on failure), then notifies the native waiter. Native code copies the result
before resetting status to 0. Requests time out after ten seconds. Browser main
thread reads are rejected; reads must happen on engine threads. The browser keeps
the worker alive until the engine instance is discarded.

Raw ROMs use their original browser `File`. Archives are decoded sequentially to
OPFS before registration. Neither path retains an entire decoded ROM in JS heap.
The frontend can use its legacy block-backed mount with older cores or when OPFS
is unavailable. Headless builds retain regular filesystem input and do not
advertise the `browserRom` operation.

`-DEDEN_WEB_ALLOC_DIAGNOSTICS=ON` enables optional failed-malloc and memory-growth
logging. It is off by default. Web builds also report the active exception from
the terminate handler, or explicitly report when no exception is active. If a
JIT region cannot allocate its temporary code buffer, it falls back to interpreting
that region; this does not promise recovery from general memory exhaustion.

The WebGPU texture cache checks the device's ASTC feature before allocating
compressed textures. When ASTC or the requested block size is unavailable, it
marks the image for the shared CPU decoder and creates an RGBA8 texture. Uploads
then use pixel rows instead of compressed-block rows; views preserve linear/sRGB
interpretation, and memory counters account for the decoded allocation. Web
settings keep ASTC recompression disabled for this RGBA path.

Allocation diagnostics also record texture creation exceptions with their
descriptors, and the start/completion of heap growth. These hooks are diagnostic
only. CMake tracks the native JS bridge and pre-JS files as link dependencies so
changes to them rebuild the runtime modules.

Frame stepping, emulator save-state serialization and browser LDN transport are
not exposed by this API. In-game save persistence is handled separately by the
frontend's IndexedDB save watcher.
