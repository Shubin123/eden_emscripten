// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <cstring>
#include <vector>

#include "common/common_types.h"
#include "dynarmic/ir/cond.h"
#include "dynarmic/ir/location_descriptor.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/terminal.h"

namespace Dynarmic::IR {
class Block;
}

namespace Dynarmic::Backend::Interp {

/// Storage for one IR value. Scalars are zero-extended into lo. Instructions with pseudo-operations
/// also record their side results here so GetCarryFromOp and friends can read them.
struct Slot {
    u64 lo;
    u64 hi;
    u64 upper_lo;
    u64 upper_hi;
    u8 carry;
    u8 overflow;
};

struct Op {
    IR::Opcode opcode;
    /// Bit width of the parent result, for flag pseudo-operations.
    u8 width;
    u32 dst;
    std::array<u32, 4> args;
};

/// Slots in a Jit's frame. The frame never moves, so compiled regions address it directly; blocks
/// that need more slots run interpreted on a separate overflow frame.
constexpr size_t kFrameSlots = 16384;

/// An IR block lowered to a flat instruction array over a slot frame. Immediates occupy the
/// slots after the instruction results and are copied into the frame before execution.
struct CompiledBlock {
    std::vector<Op> ops;
    std::vector<Slot> constants;
    u32 first_constant = 0;
    u32 num_slots = 0;
    IR::Term::Terminal terminal;
    u64 cycle_count = 0;
    IR::LocationDescriptor location{0};
    IR::LocationDescriptor end_location{0};
    /// A32 blocks may be conditional; when the condition fails execution continues at
    /// cond_failed_location instead.
    IR::Cond cond = IR::Cond::AL;
    IR::LocationDescriptor cond_failed_location{0};
    u64 cond_failed_cycle_count = 0;
    u64 start_address = 0;
    u64 end_address = 0;
    /// Cached successors for LinkBlock terminals, invalidated with the cache.
    CompiledBlock* link_then = nullptr;
    CompiledBlock* link_else = nullptr;
    /// Compiled region containing this block, if any, and the block's index within it.
    std::shared_ptr<struct WasmRegion> region;
    u32 region_index = 0;
    /// Number of times the block was entered from the dispatcher.
    u32 entry_count = 0;
};

/// A compiled set of linked blocks. Each thread instantiates the module on first use.
struct WasmRegion {
    /// Lets every thread release its instance of the module once the region is dropped.
    ~WasmRegion();

    u64 id = 0;
    std::vector<u8> wasm;
    /// Blocks compiled into the region; it must be dropped when any of them is invalidated.
    std::vector<const CompiledBlock*> blocks;
    /// Whether `wasm` counts towards the live code budget (see TrackRegionCode).
    bool tracked = false;
    /// Last thread to look up its instance, and that instance's table index, packed as
    /// (thread << 32) | index so both are read together: indices are only valid on their thread.
    std::atomic<u64> last_use{0};
};

/// Accounts executed guest cycles; prints a throughput summary every few seconds when the
/// EDEN_INTERP_STATS environment variable is set.
void RecordExecutedCycles(u64 cycles, u32 core, u64 busy_ns);

/// Counts an operation executed through the interpreter fallback of a compiled region.
void CountHelperOp(IR::Opcode opcode);

/// Whether the IR of the block at `pc` should be printed (EDEN_INTERP_DUMP=hex,hex,...).
bool ShouldDumpBlock(u64 pc);

/// Records the guest PC of an executed block for the hot-spot summary printed with the stats.
/// Returns true when the stats asked for this block's IR to be printed.
bool SamplePC(u64 pc);

/// Lowers an optimized IR block. Values are assigned slots in program order.
void Compile(CompiledBlock& out, const IR::Block& block);

}  // namespace Dynarmic::Backend::Interp
