# Eden WebAssembly / Emscripten Port

Backend: https://github.com/Shubin123/eden_emscripten  
Frontend: https://github.com/Shubin123/eden_web  

## Overview

This repository hosts the Emscripten/WebAssembly port of the **Eden** Nintendo Switch emulator. It provides the core C++ emulator engine compiled to WebAssembly (Wasm) with POSIX/pthread support for high-performance execution inside modern web browsers.

Frontend presentation, ROM mounting, touch/gamepad input mapping, and browser test harnesses are maintained in the sibling repository [`eden_web`](https://github.com/Shubin123/eden_web).

## Architecture

- **CPU Core**: ARM64 JIT/interpreter targeting WebAssembly with `SharedArrayBuffer` threading support.
- **GPU Subsystem**: WebGPU / WebGL2 graphics pipeline shims translating NVN / Vulkan graphics commands.
- **Audio Engine**: Web Audio PCM streaming sink via low-latency ring buffers.
- **Host Bridge**: Emscripten HTML5 library, MEMFS/IDBFS virtual filesystem for ROM loading, NAND simulation, and save-data persistence.

## Upstream Provenance

- Upstream Source: https://github.com/eden-emulator/mirror (and `https://git.eden-emu.dev/eden-emu/eden.git`)
- Tracking branch: `main`
- Upstream mirror tracks upstream Eden master.

## Build and Setup

### Prerequisites

- **Emscripten SDK**: >= 6.0.9 (available at `~/emsdk`)
- **CMake**: >= 3.25
- **Ninja**: >= 1.10
- **Node.js**: >= 20.x

### Building the WebAssembly Core

```bash
# From eden_emscripten root:
bash tools/web/build.sh
```

The browser frontend loads games from 7z, RAR, tar and other archives with a separate libarchive
decoder. Build it once (it fetches pinned libarchive, xz and zstd sources):

```bash
bash tools/web/build-archive.sh
```

### Syncing to Frontend

Once the WebAssembly artifacts (`.wasm`, `.js`) are generated in `build-web/`:

```bash
cd ../eden_web
npm run sync
npm start
```

Then visit `http://localhost:8080` to launch the browser runner.
