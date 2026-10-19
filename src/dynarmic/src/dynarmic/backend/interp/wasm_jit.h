// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Compiles interpreter blocks to WebAssembly. Scalar operations are emitted inline; anything else
// calls back into the interpreter for that single operation, so both paths share one frame.

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "dynarmic/backend/interp/interp.h"

namespace Dynarmic::Backend::Interp {

/// Signature of a compiled block: (state, frame, impl, block).
using CompiledBlockFn = void (*)(void*, void*, void*, void*);
/// Signature of the interpreter fallback: (impl, block, op index).
using OpHelperFn = void (*)(void*, void*, u32);

/// Describes the guest state layout and memory configuration a block is compiled against.
struct JitTarget {
    bool a32 = false;

    u32 reg_offset = 0;
    u32 sp_offset = 0;
    u32 pc_offset = 0;
    u32 nzcv_offset = 0;
    u32 check_bit_offset = 0;
    u32 upper_offset = 0;
    u32 q_offset = 0;
    u32 ge_offset = 0;
    u32 ext_offset = 0;
    u32 wrote_pc_offset = 0;  ///< Offset of the A32 BXWritePC flag within the Impl.

    u64 page_table = 0;
    u64 page_table_pointer_mask = 0;
    int pointer_sign_shift = 0;
    std::optional<u8> page_table_marked_bit;
    bool absolute_offset_page_table = false;
    u32 page_table_address_space_bits = 32;
    bool silently_mirror_page_table = false;

    u64 tpidr_el0 = 0;
    u64 tpidrro_el0 = 0;

    /// Whether the cycle budget must be honoured at every block link.
    bool cycle_counting = true;

    /// Addresses of the s64 cycle budget and the u32 halt reason.
    u64 cycles_remaining = 0;
    u64 halt_reason = 0;

    OpHelperFn helper = nullptr;

    /// Floating-point optimizations the guest configuration allows.
    bool inaccurate_nan = false;
    bool unfuse_fma = false;
    bool ignore_standard_fpcr = false;
    u32 fpsr_nzcv_offset = 0;

    /// A32 exclusive accesses with a per-core reservation (Unsafe_IgnoreGlobalMonitor).
    bool local_exclusive_monitor = false;
    u32 exclusive_state_offset = 0;
    u32 exclusive_address_offset = 0;
    u32 exclusive_value_offset = 0;

    /// Returns the address a coprocessor read resolves to, or 0 if it needs a callback.
    std::function<u64(u64)> coproc_read_pointer;
};

/// Builds a module for a region of linked blocks; blocks[0] is the entry point.
bool CompileRegion(const std::vector<CompiledBlock*>& blocks, const JitTarget& target, std::vector<u8>& out);

/// Returns the compiled function for `region` on the calling thread, instantiating its module on
/// first use. Returns nullptr if instantiation failed.
CompiledBlockFn GetThreadFunction(WasmRegion& region);

/// Compile and instantiation counts and times since the last call, for the stats report.
std::string JitStats();

/// Whether compiled execution is enabled (EDEN_INTERP_JIT=0 disables it).
bool WasmJitEnabled();

/// Entries into a block before a region is compiled for it (EDEN_JIT_THRESHOLD, default 8).
u32 RegionThreshold();

}  // namespace Dynarmic::Backend::Interp
