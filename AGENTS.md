# AGENTS

Read and follow the AI policy found in `docs/policies/AI.md`, and the coding policies found in `docs/policies/Coding.md`.

Do not, under any circumstances, ever submit PRs or push commits on behalf of the user.

Keep comments to a reasonable level. Do not overexplain or add/remove comments from unrelated parts of the codebase.

## Repository Boundaries & Sibling eden_web

This repository owns Eden source patches, native C++ shims, Emscripten builds, and WebAssembly engine modules.
The sibling `eden_web` repository owns the browser UI in `web/` and development harness in `scripts/`.
Never edit generated engine files or copy backend native source into `eden_web`.
Synchronize compiled WebAssembly artifacts to `eden_web/web/runtime/` via `eden_web`'s `npm run sync`.
Keep commercial ROMs (NSP, XCI), cryptographic keys, and user save data strictly local and uncommitted.

