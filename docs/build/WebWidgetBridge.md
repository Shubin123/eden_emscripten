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

Frame stepping, emulator save-state serialization and browser LDN transport are
not exposed by this API. In-game save persistence is handled separately by the
frontend's IndexedDB save watcher.
