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
    /// Compiled region entered at this block, if any.
    std::shared_ptr<struct WasmRegion> region;
    /// Number of times the block was entered from the dispatcher.
    u32 entry_count = 0;
};

/// A compiled set of linked blocks. Each thread instantiates the module on first use.
struct WasmRegion {
    u64 id = 0;
    std::vector<u8> wasm;
    std::atomic<u32> last_thread{0};
    std::atomic<void (*)(void*, void*, void*, void*)> last_function{nullptr};
};

/// Accounts executed guest cycles; prints a throughput summary every few seconds when the
/// EDEN_INTERP_STATS environment variable is set.
void RecordExecutedCycles(u64 cycles, u32 core);

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
