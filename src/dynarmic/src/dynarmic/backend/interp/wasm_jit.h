// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Compiles interpreter blocks to WebAssembly. Scalar operations are emitted inline; anything else
// calls back into the interpreter in consecutive runs, so both paths share one frame.

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "dynarmic/backend/interp/interp.h"

namespace Dynarmic::Backend::Interp {

/// Signature of a compiled region: (state, frame, impl, index of the block to enter at).
using CompiledBlockFn = void (*)(void*, void*, void*, void*);
/// Signature of the interpreter fallback: (impl, block, first op, exclusive end).
using OpHelperFn = void (*)(void*, void*, u32, u32);

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

    /// Whether regions continue into the next region through the dispatch table at
    /// dispatch_table_offset in the guest state, instead of returning to the dispatcher.
    bool chain_regions = false;
    u32 fpcr_offset = 0;
    u32 dispatch_table_offset = 0;

    /// Exclusive accesses with a per-core reservation (Unsafe_IgnoreGlobalMonitor).
    bool local_exclusive_monitor = false;
    u32 exclusive_state_offset = 0;
    u32 exclusive_address_offset = 0;
    u32 exclusive_value_offset = 0;

    /// Returns the address a coprocessor read resolves to, or 0 if it needs a callback.
    std::function<u64(u64)> coproc_read_pointer;
};

/// A guest location and the function and block index of the region that runs it, valid on the
/// thread that owns the table.
struct DispatchEntry {
    u64 location;
    u64 function;
    u64 index;
    u64 padding;
};
static_assert(sizeof(DispatchEntry) == 32);

/// Direct-mapped tables of compiled regions, one per thread that runs a Jit. Region code looks up
/// the next location there and tail-calls its region while there is budget left.
class DispatchTables {
public:
    static constexpr size_t kEntries = 8192;

    static size_t IndexOf(u64 location) {
        return ((location ^ (location >> 32)) >> 2) % kEntries;
    }

    /// The calling thread's table.
    DispatchEntry* ForThisThread();

    /// Clears every thread's table, for when regions are dropped.
    void Clear();

private:
    using Table = std::array<DispatchEntry, kEntries>;
    const u64 id = NextId();
    std::mutex mutex;
    std::vector<std::unique_ptr<Table>> tables;

    static u64 NextId();
};

/// Builds a module for a region of linked blocks; blocks[0] is the entry point.
bool CompileRegion(const std::vector<CompiledBlock*>& blocks, const JitTarget& target, std::vector<u8>& out);

/// Returns the compiled function for `region` on the calling thread, instantiating its module on
/// first use. Returns nullptr if instantiation failed.
CompiledBlockFn GetThreadFunction(WasmRegion& region);

/// Id for a new region, unique across every Jit (A32 and A64): threads key their instances by it.
u64 NextRegionId();

/// Counts a built region towards the live code budget. Returns true when the live code exceeds
/// it (EDEN_JIT_CODE_BUDGET_MB, default 384 MB of module bytes), and the caller should drop
/// regions: the engine cannot free compiled code that regions still reference, and running out
/// of wasm code space aborts the process.
bool TrackRegionCode(WasmRegion& region);

/// Asks every Jit to drop the regions it has not entered since its previous eviction. At most
/// twice a second, so regions in use have time to be entered again between evictions.
void RequestEviction();

/// Counts eviction requests; a Jit evicts when this differs from the value it last saw.
u64 EvictionGeneration();

/// Whether the live code exceeds the budget by more than an eighth even after evictions, so a
/// Jit has to drop everything to stay clear of the engine's code space limit.
bool CodeOverHardLimit();

/// Compile and instantiation counts and times since the last call, for the stats report.
std::string JitStats();

/// Debug: EDEN_JIT_NAMES names each region's function r<entry pc>_<blocks> for profilers.
bool RegionNamesEnabled();

/// Whether compiled execution is enabled (EDEN_INTERP_JIT=0 disables it).
bool WasmJitEnabled();

/// Entries into a block before a region is compiled for it (EDEN_JIT_THRESHOLD, default 64).
u32 RegionThreshold();

/// Whether a region may be compiled at this guest PC. EDEN_JIT_PC_RANGE=lo-hi (hex) limits
/// compilation to entry PCs in [lo, hi), for bisecting miscompiled regions.
bool RegionAllowedAt(u64 pc);

/// Debug: opcodes in EDEN_JIT_HELPER_OPS=lo-hi (decimal) run through interpreter helpers.
bool ForcedHelperOp(IR::Opcode opcode);

}  // namespace Dynarmic::Backend::Interp
