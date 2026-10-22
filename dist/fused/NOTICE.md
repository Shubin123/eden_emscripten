# Eden WebAssembly fused core

This package contains a WebAssembly build of the Eden Nintendo Switch emulator
(`eden.js`, `eden.wasm`) and its archive decoder (`eden_archive.js`,
`eden_archive.wasm`), built from the `fused` branch of eden_emscripten. The fused
branch is the eden_emscripten web port with one upstream Eden commit merged and a
set of fixes ported from Citron Neo, listed below.

## License

This software is licensed under the GNU General Public License, version 3 or
(at your option) any later version (GPL-3.0-or-later), the same license as Eden.
The full text is in `LICENSE.txt`.

The corresponding source code is the repository and commit named in
`SOURCE.txt`.

Third-party components are distributed under their own licenses; their texts
are in `LICENSES/`.

## Credits

- Eden Emulator Project
- yuzu Emulator Project
- citron Emulator Project and Citron Neo contributors

### Upstream Eden commit

- `989768aa46` Multithreading refactor (#4254) (#4548), by John

### Changes ported from Citron Neo

| Change | Original author | Citron commit |
| --- | --- | --- |
| core_timing: wake the timing thread only for a new earliest event | borb | `967da6a86acdc941da2713666b269ef4b69b3fc3` |
| video_core: clear big_page_continuous bit on Unmap and MapSparse (#158) | cheezwiz7899 | `d5863c0f6ae4e4955bf8322e0aa7cf577bdd54db` |
| fs: implement DeleteCacheStorage stub | Zephyron | `8cc5aa143f3700eed3e4c0f5b9dc3c16a6caa1d4` |
| audio: look up set:sys on first use instead of blocking in audctl constructor | hopesuffers | `21813e8cb3ea8194a5bcd8acc05b69e885f4d5c2` |
| video: handle invalid texture format in format lookup table | Zephyron | `750af88ed9a9bfbd64ebd0c551a57e007382b69d` |
| puller: honour SEMAPHORED RELEASE_SIZE and write releases at their own fence | borb | `97f7df6f5a`, `0887a06888` |
| dynarmic: fix stale code-page snapshot and single-core icache invalidation | collecting | `e4248a585c` |
| shader: treat all four components of generic 31 as generic | borb | `64aa69d1a8` |

Some changes were adapted to Eden's code; the commit messages on the `fused`
branch describe the adaptations.

## Not included

No games (ROMs), cryptographic keys, firmware or save data are included. You
must supply your own files, legally dumped from hardware and software you own.
