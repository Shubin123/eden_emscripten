// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>

namespace Dynarmic {

/// Guest work one emulated core has done since the process started. Implemented by the interp
/// backend, for frontends that measure emulation speed independently of a game's frame cap.
struct ExecStats {
    std::uint64_t cycles = 0;
    /// Host time spent inside Jit::Run.
    std::uint64_t busy_ns = 0;
};

ExecStats GetExecStats(std::uint32_t core);

}  // namespace Dynarmic
