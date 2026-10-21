// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "dynarmic/backend/interp/wasm_jit.h"

#include <atomic>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <optional>
#include <cstring>
#include <limits>
#include <new>
#include <unordered_map>

#include <emscripten.h>
#include <emscripten/heap.h>
#include <fmt/format.h>

#include "dynarmic/backend/interp/wasm_emitter.h"
#include "dynarmic/frontend/A32/a32_location_descriptor.h"
#include "dynarmic/frontend/A32/a32_types.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/ir/cond.h"
#include "dynarmic/ir/type.h"
#include "dynarmic/common/fp/info.h"
#include "dynarmic/common/fp/rounding_mode.h"
#include <bit>
#include <cmath>

// Compiles a module and adds its exported function to this thread's function table.
// Returns the table index, or 0 on failure.
EM_JS(uint64_t, dynarmic_wasm_instantiate, (const void* data, size_t size), {
    try {
        const start = Number(data);
        const bytes = HEAPU8.slice(start, start + Number(size));
        const module = new WebAssembly.Module(bytes);
        const instance = new WebAssembly.Instance(module, {e: {m: wasmMemory, t: wasmTable}});
        return BigInt(wasmTable.grow(1n, instance.exports.f));
    } catch (e) {
        err('dynarmic: failed to instantiate compiled block: ' + e);
        if (typeof process != 'undefined' && process.env && process.env.EDEN_JIT_DUMP_DIR) {
            const fs = require('fs');
            const name = process.env.EDEN_JIT_DUMP_DIR + '/bad-' + Date.now() + '-' + Math.floor(Math.random() * 1e6) + '.wasm';
            fs.writeFileSync(name, HEAPU8.slice(Number(data), Number(data) + Number(size)));
        }
        return 0n;
    }
});

namespace Dynarmic::Backend::Interp {

using IR::Opcode;
using namespace Wasm;

namespace {

constexpr u32 kNone = std::numeric_limits<u32>::max();
constexpr u32 kSlotSize = sizeof(Slot);
constexpr u32 kLoOffset = offsetof(Slot, lo);
constexpr u32 kHiOffset = offsetof(Slot, hi);
constexpr u32 kCarryOffset = offsetof(Slot, carry);
constexpr u32 kOverflowOffset = offsetof(Slot, overflow);

// Typed scratch locals shared by all blocks of a region.
constexpr u32 kF32A = 25;
constexpr u32 kF32B = 26;
constexpr u32 kF64A = 27;
constexpr u32 kF64B = 28;
constexpr u32 kV128A = 29;
constexpr u32 kV128B = 30;
constexpr u32 kV128C = 31;

// Function parameters.
constexpr u32 kState = 0;
constexpr u32 kFrame = 1;
constexpr u32 kImpl = 2;
constexpr u32 kBlock = 3;

bool IsFlagPseudoOp(Opcode op) {
    switch (op) {
    case Opcode::GetCarryFromOp:
    case Opcode::GetOverflowFromOp:
    case Opcode::GetNZCVFromOp:
    case Opcode::GetNZFromOp:
        return true;
    default:
        return false;
    }
}

/// Operations whose result is a 128-bit value; those always live in the frame.
bool ProducesVectorInFrame(Opcode op) {
    switch (op) {
    case Opcode::ZeroExtendLongToQuad:
    case Opcode::A64GetS:
    case Opcode::A64GetD:
    case Opcode::A64GetQ:
    case Opcode::A32GetVector:
        return true;
    default:
        return false;
    }
}

bool IsInlineOpcode(Opcode op, bool a32) {
    switch (op) {
    case Opcode::Void:
    case Opcode::Breakpoint:
    case Opcode::PushRSB:
    case Opcode::GetCFlagFromNZCV:
    case Opcode::NZCVFromPackedFlags:
    case Opcode::Pack2x32To1x64:
    case Opcode::LeastSignificantWord:
    case Opcode::LeastSignificantHalf:
    case Opcode::LeastSignificantByte:
    case Opcode::MostSignificantWord:
    case Opcode::MostSignificantBit:
    case Opcode::IsZero32:
    case Opcode::IsZero64:
    case Opcode::TestBit:
    case Opcode::ConditionalSelect32:
    case Opcode::ConditionalSelect64:
    case Opcode::ConditionalSelectNZCV:
    case Opcode::LogicalShiftLeft32:
    case Opcode::LogicalShiftLeft64:
    case Opcode::LogicalShiftRight32:
    case Opcode::LogicalShiftRight64:
    case Opcode::ArithmeticShiftRight32:
    case Opcode::ArithmeticShiftRight64:
    case Opcode::BitRotateRight32:
    case Opcode::BitRotateRight64:
    case Opcode::RotateRightExtended:
    case Opcode::LogicalShiftLeftMasked32:
    case Opcode::LogicalShiftLeftMasked64:
    case Opcode::LogicalShiftRightMasked32:
    case Opcode::LogicalShiftRightMasked64:
    case Opcode::ArithmeticShiftRightMasked32:
    case Opcode::ArithmeticShiftRightMasked64:
    case Opcode::RotateRightMasked32:
    case Opcode::RotateRightMasked64:
    case Opcode::Add32:
    case Opcode::Add64:
    case Opcode::Sub32:
    case Opcode::Sub64:
    case Opcode::Mul32:
    case Opcode::Mul64:
    case Opcode::UnsignedDiv32:
    case Opcode::UnsignedDiv64:
    case Opcode::SignedDiv32:
    case Opcode::SignedDiv64:
    case Opcode::And32:
    case Opcode::And64:
    case Opcode::AndNot32:
    case Opcode::AndNot64:
    case Opcode::Eor32:
    case Opcode::Eor64:
    case Opcode::Or32:
    case Opcode::Or64:
    case Opcode::Not32:
    case Opcode::Not64:
    case Opcode::SignExtendByteToWord:
    case Opcode::SignExtendHalfToWord:
    case Opcode::SignExtendByteToLong:
    case Opcode::SignExtendHalfToLong:
    case Opcode::SignExtendWordToLong:
    case Opcode::ZeroExtendByteToWord:
    case Opcode::ZeroExtendHalfToWord:
    case Opcode::ZeroExtendByteToLong:
    case Opcode::ZeroExtendHalfToLong:
    case Opcode::ZeroExtendWordToLong:
    case Opcode::ZeroExtendLongToQuad:
    case Opcode::CountLeadingZeros32:
    case Opcode::CountLeadingZeros64:
    case Opcode::ExtractRegister32:
    case Opcode::ExtractRegister64:
    case Opcode::ReplicateBit32:
    case Opcode::ReplicateBit64:
    case Opcode::MaxSigned32:
    case Opcode::MaxSigned64:
    case Opcode::MaxUnsigned32:
    case Opcode::MaxUnsigned64:
    case Opcode::MinSigned32:
    case Opcode::MinSigned64:
    case Opcode::MinUnsigned32:
    case Opcode::MinUnsigned64:
        return true;
    case Opcode::A64SetCheckBit:
    case Opcode::A64GetCFlag:
    case Opcode::A64GetNZCVRaw:
    case Opcode::A64SetNZCVRaw:
    case Opcode::A64SetNZCV:
    case Opcode::A64GetW:
    case Opcode::A64GetX:
    case Opcode::A64GetS:
    case Opcode::A64GetD:
    case Opcode::A64GetQ:
    case Opcode::A64GetSP:
    case Opcode::A64SetW:
    case Opcode::A64SetX:
    case Opcode::A64SetS:
    case Opcode::A64SetD:
    case Opcode::A64SetQ:
    case Opcode::A64SetSP:
    case Opcode::A64SetPC:
    case Opcode::A64GetTPIDR:
    case Opcode::A64GetTPIDRRO:
    case Opcode::A64SetTPIDR:
    case Opcode::A64ReadMemory8:
    case Opcode::A64ReadMemory16:
    case Opcode::A64ReadMemory32:
    case Opcode::A64ReadMemory64:
    case Opcode::A64WriteMemory8:
    case Opcode::A64WriteMemory16:
    case Opcode::A64WriteMemory32:
    case Opcode::A64WriteMemory64:
        return !a32;
    case Opcode::A32SetCheckBit:
    case Opcode::A32GetRegister:
    case Opcode::A32GetExtendedRegister32:
    case Opcode::A32GetExtendedRegister64:
    case Opcode::A32GetVector:
    case Opcode::A32SetRegister:
    case Opcode::A32SetExtendedRegister32:
    case Opcode::A32SetExtendedRegister64:
    case Opcode::A32SetVector:
    case Opcode::A32SetCpsrNZCV:
    case Opcode::A32SetCpsrNZCVRaw:
    case Opcode::A32SetCpsrNZCVQ:
    case Opcode::A32SetCpsrNZ:
    case Opcode::A32SetCpsrNZC:
    case Opcode::A32GetCFlag:
    case Opcode::A32OrQFlag:
    case Opcode::A32GetGEFlags:
    case Opcode::A32SetGEFlags:
    case Opcode::A32BXWritePC:
    case Opcode::A32UpdateUpperLocationDescriptor:
    case Opcode::A32ReadMemory8:
    case Opcode::A32ReadMemory16:
    case Opcode::A32ReadMemory32:
    case Opcode::A32ReadMemory64:
    case Opcode::A32WriteMemory8:
    case Opcode::A32WriteMemory16:
    case Opcode::A32WriteMemory32:
    case Opcode::A32WriteMemory64:
        return a32;
    default:
        return false;
    }
}

bool IsMemoryOp(Opcode op) {
    switch (op) {
    case Opcode::A64ReadMemory8:
    case Opcode::A64ReadMemory16:
    case Opcode::A64ReadMemory32:
    case Opcode::A64ReadMemory64:
    case Opcode::A64WriteMemory8:
    case Opcode::A64WriteMemory16:
    case Opcode::A64WriteMemory32:
    case Opcode::A64WriteMemory64:
    case Opcode::A32ReadMemory8:
    case Opcode::A32ReadMemory16:
    case Opcode::A32ReadMemory32:
    case Opcode::A32ReadMemory64:
    case Opcode::A32WriteMemory8:
    case Opcode::A32WriteMemory16:
    case Opcode::A32WriteMemory32:
    case Opcode::A32WriteMemory64:
    case Opcode::A32ExclusiveReadMemory8:
    case Opcode::A32ExclusiveReadMemory16:
    case Opcode::A32ExclusiveReadMemory32:
    case Opcode::A32ExclusiveReadMemory64:
    case Opcode::A32ExclusiveWriteMemory8:
    case Opcode::A32ExclusiveWriteMemory16:
    case Opcode::A32ExclusiveWriteMemory32:
    case Opcode::A32ExclusiveWriteMemory64:
        return true;
    default:
        return false;
    }
}

/// Guest state the region keeps in locals (A32 only). Values are loaded at entry and written
/// back at exits and around fallbacks that access the guest state.
struct RegionLocals {
    std::array<u32, 16> reg;
    u32 nzcv;
    u32 cycles;
    u32 tlb_page;
    u32 tlb_base;
};

/// Fallback operations that read or write guest registers, flags or the cycle budget.
bool HelperTouchesState(Opcode op) {
    switch (op) {
    case Opcode::A32CallSupervisor:
    case Opcode::A32ExceptionRaised:
    case Opcode::A32GetCpsr:
    case Opcode::A32SetCpsr:
    case Opcode::A32InstructionSynchronizationBarrier:
        return true;
    default:
        return false;
    }
}

/// Emits the operations of one block into a region function. Locals are allocated from
/// `local_base`; values do not live across blocks, so regions reuse the same range.
class BlockCompiler {
public:
    BlockCompiler(const CompiledBlock& block_, const JitTarget& target_, Code& code_, u32 local_base, const RegionLocals* cached_)
        : block{block_}, target{target_}, code{code_}, next_local{local_base}, cached{cached_} {}

    /// Writes the cached guest state back to memory.
    void FlushState() {
        if (!cached) {
            return;
        }
        for (u32 r = 0; r < 16; ++r) {
            code.Get(kState);
            code.Get(cached->reg[r]);
            code.Mem(Insn::I64Store32, target.reg_offset + 4 * r);
        }
        code.Get(kState);
        code.Get(cached->nzcv);
        code.Mem(Insn::I64Store32, target.nzcv_offset);
        code.I64Const(target.cycles_remaining);
        code.Get(cached->cycles);
        code.Mem(Insn::I64Store, 0);
    }

    /// Reloads the cached guest state from memory.
    void ReloadState() {
        if (!cached) {
            return;
        }
        for (u32 r = 0; r < 16; ++r) {
            code.Get(kState);
            code.Mem(Insn::I64Load32U, target.reg_offset + 4 * r);
            code.Set(cached->reg[r]);
        }
        code.Get(kState);
        code.Mem(Insn::I64Load32U, target.nzcv_offset);
        code.Set(cached->nzcv);
        code.I64Const(target.cycles_remaining);
        code.Mem(Insn::I64Load, 0);
        code.Set(cached->cycles);
        // The page table may have changed.
        code.I64Const(~u64(0));
        code.Set(cached->tlb_page);
    }

    /// Allocates locals; must be called before anything else is emitted for the block.
    void Prepare() {
        Analyze();
    }

    bool Emit() {
        EmitBody();
        return !failed;
    }

    u32 LocalsEnd() const {
        return next_local;
    }

    /// Pushes the i32 result of evaluating `cond` against the guest NZCV.
    void PushCondition(IR::Cond cond) {
        PushCond(cond);
    }

private:
    // Analysis

    void Analyze() {
        const size_t count = block.ops.size();
        inline_op.assign(count, false);
        value_local.assign(count, kNone);
        carry_local.assign(count, kNone);
        overflow_local.assign(count, kNone);
        spill.assign(count, false);
        spill_flags.assign(count, false);
        wants_carry.assign(count, false);
        wants_overflow.assign(count, false);

        for (size_t i = 0; i < count; ++i) {
            const Op& op = block.ops[i];
            if (IR::IsAPseudoOperation(op.opcode)) {
                const u32 parent = op.args[0];
                inline_op[i] = IsFlagPseudoOp(op.opcode) && parent < count && inline_op[parent];
            } else {
                inline_op[i] = (IsInlineOpcode(op.opcode, target.a32) || IsInlineA32Special(op) || IsInlineFloat(op)) &&
                               !ForcedHelperOp(op.opcode);
            }
        }

        for (size_t i = 0; i < count; ++i) {
            const Op& op = block.ops[i];
            const size_t num_args = IR::GetNumArgsOf(op.opcode);
            if (IR::IsAPseudoOperation(op.opcode)) {
                const u32 parent = op.args[0];
                if (op.opcode == Opcode::GetCarryFromOp || op.opcode == Opcode::GetNZCVFromOp) {
                    wants_carry[parent] = true;
                }
                if (op.opcode == Opcode::GetOverflowFromOp || op.opcode == Opcode::GetNZCVFromOp) {
                    wants_overflow[parent] = true;
                }
                if (!inline_op[i] && parent < count) {
                    spill[parent] = true;
                    spill_flags[parent] = true;
                }
            }
            const bool slow_path = inline_op[i] && IsMemoryOp(op.opcode);
            if (inline_op[i] && !slow_path) {
                continue;
            }
            for (size_t a = 0; a < num_args; ++a) {
                if (op.args[a] < count) {
                    if (!inline_op[i]) {
                        spill[op.args[a]] = true;
                    }
                } else if (op.args[a] != kNone) {
                    helper_constants.push_back(op.args[a]);
                }
            }
        }
        std::sort(helper_constants.begin(), helper_constants.end());
        helper_constants.erase(std::unique(helper_constants.begin(), helper_constants.end()), helper_constants.end());

        for (size_t i = 0; i < count; ++i) {
            const Op& op = block.ops[i];
            const IR::Type type = IR::GetTypeOf(op.opcode);
            if (!inline_op[i] || ProducesVectorInFrame(op.opcode) || type == IR::Type::Void || type == IR::Type::U128) {
                continue;
            }
            value_local[i] = NewLocal();
            if (wants_carry[i]) {
                carry_local[i] = NewLocal();
            }
            if (wants_overflow[i]) {
                overflow_local[i] = NewLocal();
            }
        }
        tmp0 = NewLocal();
        tmp1 = NewLocal();
        tmp2 = NewLocal();
        tmp3 = NewLocal();
        nzcv_tmp = NewLocal();
        wrote_pc_local = NewLocal();
    }

    u32 NewLocal() {
        return next_local++;
    }

    // Value access

    bool IsConstant(u32 slot) const {
        return slot >= block.first_constant;
    }

    u64 ConstantValue(u32 slot) const {
        return block.constants[slot - block.first_constant].lo;
    }

    void PushArg(const Op& op, size_t index) {
        const u32 slot = op.args[index];
        if (IsConstant(slot)) {
            code.I64Const(ConstantValue(slot));
        } else if (value_local[slot] != kNone) {
            code.Get(value_local[slot]);
        } else {
            code.Get(kFrame);
            code.Mem(Insn::I64Load, u64(slot) * kSlotSize + kLoOffset);
        }
    }

    void PushCarry(u32 parent) {
        if (carry_local[parent] != kNone) {
            code.Get(carry_local[parent]);
        } else {
            code.I64Const(0);
        }
    }

    void PushOverflow(u32 parent) {
        if (overflow_local[parent] != kNone) {
            code.Get(overflow_local[parent]);
        } else {
            code.I64Const(0);
        }
    }

    /// Consumes the i64 on the stack as the result of op `index`.
    void Define(size_t index) {
        code.Set(value_local[index]);
        if (spill[index]) {
            code.Get(kFrame);
            code.Get(value_local[index]);
            code.Mem(Insn::I64Store, u64(index) * kSlotSize + kLoOffset);
        }
    }

    void SpillFlags(size_t index) {
        if (!spill_flags[index]) {
            return;
        }
        if (carry_local[index] != kNone) {
            code.Get(kFrame);
            code.Get(carry_local[index]);
            code.Mem(Insn::I64Store8, u64(index) * kSlotSize + kCarryOffset);
        }
        if (overflow_local[index] != kNone) {
            code.Get(kFrame);
            code.Get(overflow_local[index]);
            code.Mem(Insn::I64Store8, u64(index) * kSlotSize + kOverflowOffset);
        }
    }

    void Mask32() {
        code.I64Const(0xFFFFFFFF);
        code.Byte(Insn::I64And);
    }

    void And(u64 mask) {
        code.I64Const(mask);
        code.Byte(Insn::I64And);
    }

    void Shr(u32 amount) {
        code.I64Const(amount);
        code.Byte(Insn::I64ShrU);
    }

    void Shl(u32 amount) {
        code.I64Const(amount);
        code.Byte(Insn::I64Shl);
    }

    /// Pushes the i32 result of evaluating `cond` against the guest NZCV.
    void PushNZCV() {
        if (cached) {
            code.Get(cached->nzcv);
        } else {
            code.Get(kState);
            code.Mem(Insn::I64Load32U, target.nzcv_offset);
        }
    }

    template<typename F>
    void SetNZCV(F&& value) {
        if (cached) {
            value();
            code.Set(cached->nzcv);
        } else {
            code.Get(kState);
            value();
            EmitStateStore32(target.nzcv_offset);
        }
    }

    void PushCond(IR::Cond cond) {
        PushNZCV();
        code.Set(nzcv_tmp);
        const auto bit = [&](u32 shift) {
            code.Get(nzcv_tmp);
            Shr(shift);
            And(1);
        };
        const auto n = [&] { bit(31); };
        const auto z = [&] { bit(30); };
        const auto c = [&] { bit(29); };
        const auto v = [&] { bit(28); };
        switch (cond) {
        case IR::Cond::EQ:
            z();
            break;
        case IR::Cond::NE:
            z();
            code.Byte(Insn::I64Eqz);
            code.Byte(Insn::I64ExtendI32U);
            break;
        case IR::Cond::CS:
            c();
            break;
        case IR::Cond::CC:
            c();
            code.Byte(Insn::I64Eqz);
            code.Byte(Insn::I64ExtendI32U);
            break;
        case IR::Cond::MI:
            n();
            break;
        case IR::Cond::PL:
            n();
            code.Byte(Insn::I64Eqz);
            code.Byte(Insn::I64ExtendI32U);
            break;
        case IR::Cond::VS:
            v();
            break;
        case IR::Cond::VC:
            v();
            code.Byte(Insn::I64Eqz);
            code.Byte(Insn::I64ExtendI32U);
            break;
        case IR::Cond::HI:
            // C && !Z  ==  ((nzcv >> 29) & 3) == 1
            code.Get(nzcv_tmp);
            Shr(29);
            And(3);
            code.I64Const(1);
            code.Byte(Insn::I64Eq);
            code.Byte(Insn::I64ExtendI32U);
            break;
        case IR::Cond::LS:
            code.Get(nzcv_tmp);
            Shr(29);
            And(3);
            code.I64Const(1);
            code.Byte(Insn::I64Ne);
            code.Byte(Insn::I64ExtendI32U);
            break;
        case IR::Cond::GE:
        case IR::Cond::LT:
            n();
            v();
            code.Byte(Insn::I64Xor);
            if (cond == IR::Cond::GE) {
                code.Byte(Insn::I64Eqz);
                code.Byte(Insn::I64ExtendI32U);
            }
            break;
        case IR::Cond::GT:
        case IR::Cond::LE:
            // GT: !Z && N == V
            n();
            v();
            code.Byte(Insn::I64Xor);
            z();
            code.Byte(Insn::I64Or);
            if (cond == IR::Cond::GT) {
                code.Byte(Insn::I64Eqz);
                code.Byte(Insn::I64ExtendI32U);
            }
            break;
        default:
            code.I64Const(1);
            break;
        }
        code.Byte(Insn::I32WrapI64);
    }

    void CallHelper(size_t index) {
        const bool touches_state = cached && HelperTouchesState(block.ops[index].opcode);
        if (touches_state) {
            FlushState();
        }
        CallHelperRaw(index);
        if (touches_state) {
            ReloadState();
        }
    }

    void CallHelperRaw(size_t index) {
        code.Get(kImpl);
        code.I64Const(reinterpret_cast<u64>(&block));
        code.I32Const(u32(index));
        code.I64Const(reinterpret_cast<u64>(target.helper));
        code.Byte(Insn::CallIndirect);
        code.U32(1);  // helper type
        code.U32(0);  // table
    }

    /// Spills a scalar argument that lives in a local into its frame slot.
    void SpillArg(const Op& op, size_t index) {
        const u32 slot = op.args[index];
        if (IsConstant(slot) || value_local[slot] == kNone || spill[slot]) {
            return;
        }
        code.Get(kFrame);
        code.Get(value_local[slot]);
        code.Mem(Insn::I64Store, u64(slot) * kSlotSize + kLoOffset);
    }

    // Memory

    /// Leaves the host address for an access of `bytes` at the address in tmp0 in tmp1, or 0
    /// when the access needs the slow path.
    void EmitHostAddress(u32 bytes) {
        code.I64Const(0);
        code.Set(tmp1);
        if (target.page_table == 0) {
            return;
        }
        code.BlockStart();
        // Page-crossing accesses take the callbacks.
        code.Get(tmp0);
        And(0xFFF);
        code.I64Const(0x1000 - bytes);
        code.Byte(Insn::I64GtU);
        code.Byte(Insn::BrIf);
        code.U32(0);
        const bool use_tlb = cached && target.absolute_offset_page_table;
        if (use_tlb) {
            // One-entry translation cache: same page as the previous access.
            code.Get(tmp0);
            Shr(12);
            code.Get(cached->tlb_page);
            code.Byte(Insn::I64Eq);
            code.If();
            code.Get(cached->tlb_base);
            code.Get(tmp0);
            code.Byte(Insn::I64Add);
            code.Set(tmp1);
            code.Byte(Insn::Br);
            code.U32(1);
            code.End();
        }
        // Page index.
        code.Get(tmp0);
        Shr(12);
        code.Set(tmp2);
        if (!target.a32 && target.page_table_address_space_bits < 64) {
            const u32 index_bits = target.page_table_address_space_bits - 12;
            if (target.silently_mirror_page_table) {
                code.Get(tmp2);
                And((u64(1) << index_bits) - 1);
                code.Set(tmp2);
            } else {
                code.Get(tmp2);
                Shr(index_bits);
                code.I64Const(0);
                code.Byte(Insn::I64Ne);
                code.Byte(Insn::BrIf);
                code.U32(0);
            }
        }
        code.Get(tmp2);
        Shl(3);
        code.I64Const(target.page_table);
        code.Byte(Insn::I64Add);
        code.Mem(Insn::I64Load, 0);
        code.Set(tmp2);
        if (target.page_table_marked_bit) {
            code.Get(tmp2);
            Shr(*target.page_table_marked_bit);
            And(1);
            code.Byte(Insn::I32WrapI64);
            code.Byte(Insn::BrIf);
            code.U32(0);
        }
        if (target.page_table_pointer_mask != 0) {
            code.Get(tmp2);
            And(target.page_table_pointer_mask);
            Shl(u32(target.pointer_sign_shift));
            code.I64Const(u64(target.pointer_sign_shift));
            code.Byte(Insn::I64ShrS);
            code.Set(tmp2);
        }
        code.Get(tmp2);
        code.Byte(Insn::I64Eqz);
        code.Byte(Insn::BrIf);
        code.U32(0);
        if (use_tlb) {
            code.Get(tmp2);
            code.Set(cached->tlb_base);
            code.Get(tmp0);
            Shr(12);
            code.Set(cached->tlb_page);
        }
        code.Get(tmp2);
        code.Get(tmp0);
        if (!target.absolute_offset_page_table) {
            And(0xFFF);
        }
        code.Byte(Insn::I64Add);
        code.Set(tmp1);
        code.End();
    }

    void EmitRead(size_t index, u32 bytes) {
        const Op& op = block.ops[index];
        PushArg(op, 1);
        if (target.a32) {
            Mask32();
        }
        code.Set(tmp0);
        EmitHostAddress(bytes);
        code.Get(tmp1);
        code.Byte(Insn::I64Eqz);
        code.IfI64();
        {
            SpillArg(op, 1);
            CallHelper(index);
            code.Get(kFrame);
            code.Mem(Insn::I64Load, u64(index) * kSlotSize + kLoOffset);
        }
        code.Else();
        {
            code.Get(tmp1);
            switch (bytes) {
            case 1:
                code.Mem(Insn::I64Load8U, 0);
                break;
            case 2:
                code.Mem(Insn::I64Load16U, 0);
                break;
            case 4:
                code.Mem(Insn::I64Load32U, 0);
                break;
            default:
                code.Mem(Insn::I64Load, 0);
                break;
            }
        }
        code.End();
        Define(index);
    }


    bool IsInlineA32Special(const Op& op) const {
        if (!target.a32) {
            return false;
        }
        switch (op.opcode) {
        case Opcode::A32GetFpscrNZCV:
        case Opcode::A32SetFpscrNZCV:
            return true;
        case Opcode::A32ClearExclusive:
        case Opcode::A32ExclusiveReadMemory8:
        case Opcode::A32ExclusiveReadMemory16:
        case Opcode::A32ExclusiveReadMemory32:
        case Opcode::A32ExclusiveReadMemory64:
        case Opcode::A32ExclusiveWriteMemory8:
        case Opcode::A32ExclusiveWriteMemory16:
        case Opcode::A32ExclusiveWriteMemory32:
        case Opcode::A32ExclusiveWriteMemory64:
            return target.local_exclusive_monitor && target.page_table != 0;
        case Opcode::A32CoprocGetOneWord:
            return target.coproc_read_pointer && target.coproc_read_pointer(ConstantValue(op.args[0])) != 0;
        default:
            return false;
        }
    }

    void EmitExclusiveRead(size_t index, u32 bytes) {
        EmitRead(index, bytes);
        code.Get(kState);
        code.I64Const(1);
        code.Mem(Insn::I64Store32, target.exclusive_state_offset);
        code.Get(kState);
        code.Get(tmp0);
        code.Mem(Insn::I64Store32, target.exclusive_address_offset);
        code.Get(kState);
        code.Get(value_local[index]);
        code.Mem(Insn::I64Store, target.exclusive_value_offset);
    }

    /// Stores with compare-and-swap against the reserved value; 0 on success, 1 on failure.
    void EmitExclusiveWrite(size_t index, u32 bytes) {
        const Op& op = block.ops[index];
        PushArg(op, 1);
        Mask32();
        code.Set(tmp0);
        // No reservation for this address: fail.
        code.Get(kState);
        code.Mem(Insn::I64Load32U, target.exclusive_state_offset);
        code.Byte(Insn::I64Eqz);
        code.Get(kState);
        code.Mem(Insn::I64Load32U, target.exclusive_address_offset);
        code.Get(tmp0);
        code.Byte(Insn::I64Ne);
        code.Byte(Insn::I32Or);
        code.IfI64();
        {
            code.Get(kState);
            code.I64Const(0);
            code.Mem(Insn::I64Store32, target.exclusive_state_offset);
            code.I64Const(1);
        }
        code.Else();
        {
            EmitHostAddress(bytes);
            code.Get(tmp1);
            code.Byte(Insn::I64Eqz);
            code.IfI64();
            {
                SpillArg(op, 1);
                SpillArg(op, 2);
                CallHelper(index);
                code.Get(kFrame);
                code.Mem(Insn::I64Load, u64(index) * kSlotSize + kLoOffset);
            }
            code.Else();
            {
                code.Get(kState);
                code.I64Const(0);
                code.Mem(Insn::I64Store32, target.exclusive_state_offset);
                // old = cmpxchg(host, expected, value); result = old != expected
                code.Get(kState);
                code.Mem(Insn::I64Load, target.exclusive_value_offset);
                code.Set(tmp3);
                code.Get(tmp1);
                if (bytes == 8) {
                    code.Get(tmp3);
                    PushArg(op, 2);
                    code.Byte(0xFE);
                    code.U32(0x49);  // i64.atomic.rmw.cmpxchg
                    code.U32(3);
                    code.U64(0);
                    code.Get(tmp3);
                    code.Byte(Insn::I64Ne);
                } else {
                    code.Get(tmp3);
                    code.Byte(Insn::I32WrapI64);
                    PushArg(op, 2);
                    code.Byte(Insn::I32WrapI64);
                    code.Byte(0xFE);
                    code.U32(bytes == 1 ? 0x4A : bytes == 2 ? 0x4B : 0x48);  // i32.atomic.rmw{8,16,}.cmpxchg
                    code.U32(bytes == 1 ? 0 : bytes == 2 ? 1 : 2);
                    code.U64(0);
                    code.Get(tmp3);
                    code.Byte(Insn::I32WrapI64);
                    code.Byte(Insn::I32Ne);
                }
                code.Byte(Insn::I64ExtendI32U);
            }
            code.End();
        }
        code.End();
        Define(index);
    }

    void EmitWrite(size_t index, u32 bytes) {
        const Op& op = block.ops[index];
        PushArg(op, 1);
        if (target.a32) {
            Mask32();
        }
        code.Set(tmp0);
        EmitHostAddress(bytes);
        code.Get(tmp1);
        code.Byte(Insn::I64Eqz);
        code.If();
        {
            SpillArg(op, 1);
            SpillArg(op, 2);
            CallHelper(index);
        }
        code.Else();
        {
            code.Get(tmp1);
            PushArg(op, 2);
            switch (bytes) {
            case 1:
                code.Mem(Insn::I64Store8, 0);
                break;
            case 2:
                code.Mem(Insn::I64Store16, 0);
                break;
            case 4:
                code.Mem(Insn::I64Store32, 0);
                break;
            default:
                code.Mem(Insn::I64Store, 0);
                break;
            }
        }
        code.End();
    }



    // Floating point

    /// Floating-point mode the block was translated for (it is part of the location).
    u32 BlockFPCR() const {
        if (target.a32) {
            return A32::LocationDescriptor{block.location}.FPSCR().Value();
        }
        return A64::LocationDescriptor{block.location}.FPCR().Value();
    }

    bool FlushToZero() const {
        return (BlockFPCR() >> 24) & 1;
    }

    /// Native host arithmetic matches the guest when rounding to nearest and NaN payloads need
    /// not be preserved (Unsafe_InaccurateNaN).
    bool NativeFloatAllowed() const {
        return target.inaccurate_nan && ((BlockFPCR() >> 22) & 3) == 0;
    }

    bool VectorFPCRUsable(const Op& op, size_t fpcr_arg) const {
        // Without the IgnoreStandardFPCRValue optimization, uncontrolled ops use the ASIMD
        // standard value (flush-to-zero, default NaN).
        const bool controlled = ConstantValue(op.args[fpcr_arg]) & 1;
        return controlled || target.ignore_standard_fpcr;
    }

    bool IsInlineFloat(const Op& op) const {
        switch (op.opcode) {
        case Opcode::FPAbs32:
        case Opcode::FPAbs64:
        case Opcode::FPNeg32:
        case Opcode::FPNeg64:
        case Opcode::FPVectorAbs32:
        case Opcode::FPVectorAbs64:
        case Opcode::FPVectorNeg32:
        case Opcode::FPVectorNeg64:
        case Opcode::VectorAdd8:
        case Opcode::VectorAdd16:
        case Opcode::VectorAdd32:
        case Opcode::VectorAdd64:
        case Opcode::VectorSub8:
        case Opcode::VectorSub16:
        case Opcode::VectorSub32:
        case Opcode::VectorSub64:
        case Opcode::VectorMultiply16:
        case Opcode::VectorMultiply32:
        case Opcode::VectorAnd:
        case Opcode::VectorAndNot:
        case Opcode::VectorOr:
        case Opcode::VectorEor:
        case Opcode::VectorNot:
        case Opcode::VectorZeroUpper:
        case Opcode::ZeroVector:
        case Opcode::VectorEqual8:
        case Opcode::VectorEqual16:
        case Opcode::VectorEqual32:
        case Opcode::VectorGetElement32:
        case Opcode::VectorGetElement64:
        case Opcode::VectorSetElement32:
        case Opcode::VectorSetElement64:
        case Opcode::VectorBroadcast32:
        case Opcode::VectorBroadcast64:
            return true;
        default:
            break;
        }
        if (!NativeFloatAllowed()) {
            return false;
        }
        switch (op.opcode) {
        case Opcode::FPAdd32:
        case Opcode::FPAdd64:
        case Opcode::FPSub32:
        case Opcode::FPSub64:
        case Opcode::FPMul32:
        case Opcode::FPMul64:
        case Opcode::FPDiv32:
        case Opcode::FPDiv64:
        case Opcode::FPSqrt32:
        case Opcode::FPSqrt64:
        case Opcode::FPMax32:
        case Opcode::FPMax64:
        case Opcode::FPMin32:
        case Opcode::FPMin64:
        case Opcode::FPCompare32:
        case Opcode::FPCompare64:
            return true;
        case Opcode::FPMulAdd32:
        case Opcode::FPMulAdd64:
        case Opcode::FPMulSub32:
        case Opcode::FPMulSub64:
            return target.unfuse_fma;
        case Opcode::FPSingleToDouble:
        case Opcode::FPDoubleToSingle:
            return ConstantValue(op.args[1]) == u64(FP::RoundingMode::ToNearest_TieEven);
        case Opcode::FPRoundInt32:
        case Opcode::FPRoundInt64:
            return ConstantValue(op.args[1]) <= u64(FP::RoundingMode::TowardsZero);
        case Opcode::FPSingleToFixedS32:
        case Opcode::FPSingleToFixedU32:
        case Opcode::FPSingleToFixedS64:
        case Opcode::FPSingleToFixedU64:
        case Opcode::FPDoubleToFixedS32:
        case Opcode::FPDoubleToFixedU32:
        case Opcode::FPDoubleToFixedS64:
        case Opcode::FPDoubleToFixedU64:
            return ConstantValue(op.args[2]) <= u64(FP::RoundingMode::TowardsZero);
        case Opcode::FPFixedS32ToSingle:
        case Opcode::FPFixedU32ToSingle:
        case Opcode::FPFixedS32ToDouble:
        case Opcode::FPFixedU32ToDouble:
        case Opcode::FPFixedS64ToSingle:
        case Opcode::FPFixedU64ToSingle:
        case Opcode::FPFixedS64ToDouble:
        case Opcode::FPFixedU64ToDouble:
            return ConstantValue(op.args[2]) == u64(FP::RoundingMode::ToNearest_TieEven);
        case Opcode::FPVectorAdd32:
        case Opcode::FPVectorAdd64:
        case Opcode::FPVectorSub32:
        case Opcode::FPVectorSub64:
        case Opcode::FPVectorMul32:
        case Opcode::FPVectorMul64:
        case Opcode::FPVectorDiv32:
        case Opcode::FPVectorDiv64:
        case Opcode::FPVectorMax32:
        case Opcode::FPVectorMax64:
        case Opcode::FPVectorMin32:
        case Opcode::FPVectorMin64:
        case Opcode::FPVectorEqual32:
        case Opcode::FPVectorEqual64:
        case Opcode::FPVectorGreater32:
        case Opcode::FPVectorGreater64:
        case Opcode::FPVectorGreaterEqual32:
        case Opcode::FPVectorGreaterEqual64:
            return VectorFPCRUsable(op, 2);
        case Opcode::FPVectorSqrt32:
        case Opcode::FPVectorSqrt64:
            return VectorFPCRUsable(op, 1);
        case Opcode::FPVectorMulAdd32:
        case Opcode::FPVectorMulAdd64:
            return target.unfuse_fma && VectorFPCRUsable(op, 3);
        default:
            return false;
        }
    }

    void F32Const(u32 bits) {
        code.Byte(0x43);
        for (int i = 0; i < 4; ++i) {
            code.Byte(u8(bits >> (8 * i)));
        }
    }

    void F64Const(u64 bits) {
        code.Byte(0x44);
        for (int i = 0; i < 8; ++i) {
            code.Byte(u8(bits >> (8 * i)));
        }
    }

    /// With guest float bits (i64) on the stack, flushes denormals to signed zero under FZ.
    void FlushBits(bool is64) {
        if (!FlushToZero()) {
            return;
        }
        const u64 sign = is64 ? 0x8000000000000000 : 0x80000000;
        const u64 exponent = is64 ? 0x7FF0000000000000 : 0x7F800000;
        code.Set(tmp2);
        code.Get(tmp2);
        And(sign);
        code.Get(tmp2);
        code.Get(tmp2);
        And(exponent);
        code.Byte(Insn::I64Eqz);
        code.Byte(Insn::Select);
    }

    void PushFloat(const Op& op, size_t index, bool is64) {
        PushArg(op, index);
        FlushBits(is64);
        if (is64) {
            code.Byte(0xBF);  // f64.reinterpret_i64
        } else {
            code.Byte(Insn::I32WrapI64);
            code.Byte(0xBE);  // f32.reinterpret_i32
        }
    }

    /// Consumes a float result: canonicalizes NaN, flushes denormals and defines the value.
    void DefineFloat(size_t index, bool is64) {
        const u32 local = is64 ? kF64A : kF32A;
        code.Tee(local);
        if (is64) {
            code.Set(local);
            F64Const(FP::FPInfo<u64>::DefaultNaN());
            code.Get(local);
            code.Get(local);
            code.Get(local);
            code.Byte(0x62);  // f64.ne
            code.Byte(Insn::Select);
            code.Byte(0xBD);  // i64.reinterpret_f64
        } else {
            code.Set(local);
            F32Const(FP::FPInfo<u32>::DefaultNaN());
            code.Get(local);
            code.Get(local);
            code.Get(local);
            code.Byte(0x5C);  // f32.ne
            code.Byte(Insn::Select);
            code.Byte(0xBC);  // i32.reinterpret_f32
            code.Byte(Insn::I64ExtendI32U);
        }
        FlushBits(is64);
        Define(index);
    }

    void SimdOp(u32 opcode) {
        code.Byte(0xFD);
        code.U32(opcode);
    }

    /// Pushes a 128-bit argument (frame slot or constant).
    void PushV128(const Op& op, size_t index) {
        const u32 slot = op.args[index];
        if (IsConstant(slot)) {
            const Slot& constant = block.constants[slot - block.first_constant];
            SimdOp(12);  // v128.const
            for (int i = 0; i < 8; ++i) {
                code.Byte(u8(constant.lo >> (8 * i)));
            }
            for (int i = 0; i < 8; ++i) {
                code.Byte(u8(constant.hi >> (8 * i)));
            }
            return;
        }
        code.Get(kFrame);
        SimdOp(0);  // v128.load
        code.U32(0);
        code.U64(u64(slot) * kSlotSize + kLoOffset);
    }

    /// Stores the v128 on the stack as the result of op `index`.
    void StoreV128(size_t index) {
        code.Set(kV128A);
        code.Get(kFrame);
        code.Get(kV128A);
        SimdOp(11);  // v128.store
        code.U32(0);
        code.U64(u64(index) * kSlotSize + kLoOffset);
    }

    void V128Splat(u32 bits32) {
        SimdOp(12);
        for (int i = 0; i < 4; ++i) {
            for (int b = 0; b < 4; ++b) {
                code.Byte(u8(bits32 >> (8 * b)));
            }
        }
    }

    void V128Splat64(u64 bits64) {
        SimdOp(12);
        for (int i = 0; i < 2; ++i) {
            for (int b = 0; b < 8; ++b) {
                code.Byte(u8(bits64 >> (8 * b)));
            }
        }
    }

    /// Flushes denormal lanes of the v128 on the stack under FZ.
    void FlushV128(bool is64) {
        if (!FlushToZero()) {
            return;
        }
        code.Set(kV128B);
        // mask = (x & exponent) == 0
        code.Get(kV128B);
        if (is64) {
            V128Splat64(0x8000000000000000);
        } else {
            V128Splat(0x80000000);
        }
        SimdOp(78);  // v128.and -> sign only
        code.Get(kV128B);
        code.Get(kV128B);
        if (is64) {
            V128Splat64(0x7FF0000000000000);
        } else {
            V128Splat(0x7F800000);
        }
        SimdOp(78);
        if (is64) {
            V128Splat64(0);
            SimdOp(214);  // i64x2.eq
        } else {
            V128Splat(0);
            SimdOp(55);  // i32x4.eq
        }
        SimdOp(82);  // v128.bitselect(sign_only, x, mask)
    }

    void PushFloatV128(const Op& op, size_t index, bool is64) {
        PushV128(op, index);
        FlushV128(is64);
    }

    /// Canonicalizes NaN lanes and flushes denormals of the vector result, then stores it.
    void StoreFloatV128(size_t index, bool is64) {
        code.Set(kV128C);
        if (is64) {
            V128Splat64(FP::FPInfo<u64>::DefaultNaN());
        } else {
            V128Splat(FP::FPInfo<u32>::DefaultNaN());
        }
        code.Get(kV128C);
        code.Get(kV128C);
        code.Get(kV128C);
        SimdOp(is64 ? 72 : 66);  // f64x2.ne / f32x4.ne
        SimdOp(82);              // v128.bitselect
        FlushV128(is64);
        StoreV128(index);
    }

    bool EmitFloat(size_t index) {
        const Op& op = block.ops[index];
        const auto unary_bits = [&](u64 mask, bool xor_mask) {
            PushArg(op, 0);
            code.I64Const(mask);
            code.Byte(xor_mask ? Insn::I64Xor : Insn::I64And);
            Define(index);
        };
        const auto binary = [&](bool is64, u8 opcode) {
            PushFloat(op, 0, is64);
            PushFloat(op, 1, is64);
            code.Byte(opcode);
            DefineFloat(index, is64);
        };
        const auto vbinary = [&](bool is64, u32 simd, bool float_result) {
            if (float_result) {
                PushFloatV128(op, 0, is64);
                PushFloatV128(op, 1, is64);
                SimdOp(simd);
                StoreFloatV128(index, is64);
            } else {
                PushFloatV128(op, 0, is64);
                PushFloatV128(op, 1, is64);
                SimdOp(simd);
                StoreV128(index);
            }
        };
        const auto vint = [&](u32 simd) {
            PushV128(op, 0);
            PushV128(op, 1);
            SimdOp(simd);
            StoreV128(index);
        };

        switch (op.opcode) {
        case Opcode::FPAbs32:
            unary_bits(0x7FFFFFFF, false);
            return true;
        case Opcode::FPAbs64:
            unary_bits(0x7FFFFFFFFFFFFFFF, false);
            return true;
        case Opcode::FPNeg32:
            unary_bits(0x80000000, true);
            return true;
        case Opcode::FPNeg64:
            unary_bits(0x8000000000000000, true);
            return true;
        case Opcode::FPAdd32:
            binary(false, 0x92);
            return true;
        case Opcode::FPAdd64:
            binary(true, 0xA0);
            return true;
        case Opcode::FPSub32:
            binary(false, 0x93);
            return true;
        case Opcode::FPSub64:
            binary(true, 0xA1);
            return true;
        case Opcode::FPMul32:
            binary(false, 0x94);
            return true;
        case Opcode::FPMul64:
            binary(true, 0xA2);
            return true;
        case Opcode::FPDiv32:
            binary(false, 0x95);
            return true;
        case Opcode::FPDiv64:
            binary(true, 0xA3);
            return true;
        case Opcode::FPMin32:
            binary(false, 0x96);
            return true;
        case Opcode::FPMin64:
            binary(true, 0xA4);
            return true;
        case Opcode::FPMax32:
            binary(false, 0x97);
            return true;
        case Opcode::FPMax64:
            binary(true, 0xA5);
            return true;
        case Opcode::FPSqrt32:
            PushFloat(op, 0, false);
            code.Byte(0x91);
            DefineFloat(index, false);
            return true;
        case Opcode::FPSqrt64:
            PushFloat(op, 0, true);
            code.Byte(0x9F);
            DefineFloat(index, true);
            return true;
        case Opcode::FPCompare32:
        case Opcode::FPCompare64: {
            const bool is64 = op.opcode == Opcode::FPCompare64;
            const u32 a = is64 ? kF64A : kF32A;
            const u32 b = is64 ? kF64B : kF32B;
            PushFloat(op, 0, is64);
            code.Set(a);
            PushFloat(op, 1, is64);
            code.Set(b);
            code.I64Const(0x60000000);
            code.I64Const(0x80000000);
            code.I64Const(0x20000000);
            code.I64Const(0x30000000);
            code.Get(a);
            code.Get(b);
            code.Byte(is64 ? 0x64 : 0x5E);  // gt
            code.Byte(Insn::Select);
            code.Get(a);
            code.Get(b);
            code.Byte(is64 ? 0x63 : 0x5D);  // lt
            code.Byte(Insn::Select);
            code.Get(a);
            code.Get(b);
            code.Byte(is64 ? 0x61 : 0x5B);  // eq
            code.Byte(Insn::Select);
            Define(index);
            return true;
        }
        case Opcode::FPMulAdd32:
        case Opcode::FPMulAdd64:
        case Opcode::FPMulSub32:
        case Opcode::FPMulSub64: {
            const bool is64 = op.opcode == Opcode::FPMulAdd64 || op.opcode == Opcode::FPMulSub64;
            const bool subtract = op.opcode == Opcode::FPMulSub32 || op.opcode == Opcode::FPMulSub64;
            PushFloat(op, 0, is64);
            PushFloat(op, 1, is64);
            PushFloat(op, 2, is64);
            code.Byte(is64 ? 0xA2 : 0x94);  // mul
            code.Byte(is64 ? (subtract ? 0xA1 : 0xA0) : (subtract ? 0x93 : 0x92));
            DefineFloat(index, is64);
            return true;
        }
        case Opcode::FPSingleToDouble:
            PushFloat(op, 0, false);
            code.Byte(0xBB);  // f64.promote_f32
            DefineFloat(index, true);
            return true;
        case Opcode::FPDoubleToSingle:
            PushFloat(op, 0, true);
            code.Byte(0xB6);  // f32.demote_f64
            DefineFloat(index, false);
            return true;
        case Opcode::FPRoundInt32:
        case Opcode::FPRoundInt64: {
            const bool is64 = op.opcode == Opcode::FPRoundInt64;
            static constexpr u8 f32_ops[] = {0x90, 0x8D, 0x8E, 0x8F};  // nearest, ceil, floor, trunc
            static constexpr u8 f64_ops[] = {0x9E, 0x9B, 0x9C, 0x9D};
            const u64 rounding = ConstantValue(op.args[1]);
            PushFloat(op, 0, is64);
            code.Byte(is64 ? f64_ops[rounding] : f32_ops[rounding]);
            DefineFloat(index, is64);
            return true;
        }
        case Opcode::FPSingleToFixedS32:
        case Opcode::FPSingleToFixedU32:
        case Opcode::FPSingleToFixedS64:
        case Opcode::FPSingleToFixedU64:
        case Opcode::FPDoubleToFixedS32:
        case Opcode::FPDoubleToFixedU32:
        case Opcode::FPDoubleToFixedS64:
        case Opcode::FPDoubleToFixedU64: {
            const bool from64 = op.opcode >= Opcode::FPDoubleToFixedS32 && op.opcode <= Opcode::FPDoubleToFixedU64;
            const bool is_unsigned = op.opcode == Opcode::FPSingleToFixedU32 || op.opcode == Opcode::FPSingleToFixedU64 ||
                                     op.opcode == Opcode::FPDoubleToFixedU32 || op.opcode == Opcode::FPDoubleToFixedU64;
            const bool to64 = op.opcode == Opcode::FPSingleToFixedS64 || op.opcode == Opcode::FPSingleToFixedU64 ||
                              op.opcode == Opcode::FPDoubleToFixedS64 || op.opcode == Opcode::FPDoubleToFixedU64;
            const u64 fbits = ConstantValue(op.args[1]);
            const u64 rounding = ConstantValue(op.args[2]);
            static constexpr u8 f64_round[] = {0x9E, 0x9B, 0x9C, 0x9D};
            PushFloat(op, 0, from64);
            if (!from64) {
                code.Byte(0xBB);  // promote to f64 for exact scaling
            }
            if (fbits != 0) {
                F64Const(std::bit_cast<u64>(std::ldexp(1.0, int(fbits))));
                code.Byte(0xA2);
            }
            code.Byte(f64_round[rounding]);
            code.Byte(0xFC);
            code.U32(to64 ? (is_unsigned ? 7 : 6) : (is_unsigned ? 3 : 2));  // trunc_sat_f64
            if (!to64) {
                code.Byte(Insn::I64ExtendI32U);
            }
            Define(index);
            return true;
        }
        case Opcode::FPFixedS32ToSingle:
        case Opcode::FPFixedU32ToSingle:
        case Opcode::FPFixedS32ToDouble:
        case Opcode::FPFixedU32ToDouble:
        case Opcode::FPFixedS64ToSingle:
        case Opcode::FPFixedU64ToSingle:
        case Opcode::FPFixedS64ToDouble:
        case Opcode::FPFixedU64ToDouble: {
            const bool from64 = op.opcode == Opcode::FPFixedS64ToSingle || op.opcode == Opcode::FPFixedU64ToSingle ||
                                op.opcode == Opcode::FPFixedS64ToDouble || op.opcode == Opcode::FPFixedU64ToDouble;
            const bool is_unsigned = op.opcode == Opcode::FPFixedU32ToSingle || op.opcode == Opcode::FPFixedU32ToDouble ||
                                     op.opcode == Opcode::FPFixedU64ToSingle || op.opcode == Opcode::FPFixedU64ToDouble;
            const bool to64 = op.opcode == Opcode::FPFixedS32ToDouble || op.opcode == Opcode::FPFixedU32ToDouble ||
                              op.opcode == Opcode::FPFixedS64ToDouble || op.opcode == Opcode::FPFixedU64ToDouble;
            const u64 fbits = ConstantValue(op.args[1]);
            PushArg(op, 0);
            if (!from64) {
                code.Byte(Insn::I32WrapI64);
            }
            if (to64) {
                code.Byte(from64 ? (is_unsigned ? 0xBA : 0xB9) : (is_unsigned ? 0xB8 : 0xB7));
                if (fbits != 0) {
                    F64Const(std::bit_cast<u64>(std::ldexp(1.0, -int(fbits))));
                    code.Byte(0xA2);
                }
            } else {
                code.Byte(from64 ? (is_unsigned ? 0xB5 : 0xB4) : (is_unsigned ? 0xB3 : 0xB2));
                if (fbits != 0) {
                    F32Const(std::bit_cast<u32>(std::ldexp(1.0f, -int(fbits))));
                    code.Byte(0x94);
                }
            }
            DefineFloat(index, to64);
            return true;
        }

        // Vectors
        case Opcode::FPVectorAbs32:
            PushV128(op, 0);
            SimdOp(224);
            StoreV128(index);
            return true;
        case Opcode::FPVectorAbs64:
            PushV128(op, 0);
            SimdOp(236);
            StoreV128(index);
            return true;
        case Opcode::FPVectorNeg32:
            PushV128(op, 0);
            SimdOp(225);
            StoreV128(index);
            return true;
        case Opcode::FPVectorNeg64:
            PushV128(op, 0);
            SimdOp(237);
            StoreV128(index);
            return true;
        case Opcode::FPVectorAdd32:
            vbinary(false, 228, true);
            return true;
        case Opcode::FPVectorAdd64:
            vbinary(true, 240, true);
            return true;
        case Opcode::FPVectorSub32:
            vbinary(false, 229, true);
            return true;
        case Opcode::FPVectorSub64:
            vbinary(true, 241, true);
            return true;
        case Opcode::FPVectorMul32:
            vbinary(false, 230, true);
            return true;
        case Opcode::FPVectorMul64:
            vbinary(true, 242, true);
            return true;
        case Opcode::FPVectorDiv32:
            vbinary(false, 231, true);
            return true;
        case Opcode::FPVectorDiv64:
            vbinary(true, 243, true);
            return true;
        case Opcode::FPVectorMin32:
            vbinary(false, 232, true);
            return true;
        case Opcode::FPVectorMin64:
            vbinary(true, 244, true);
            return true;
        case Opcode::FPVectorMax32:
            vbinary(false, 233, true);
            return true;
        case Opcode::FPVectorMax64:
            vbinary(true, 245, true);
            return true;
        case Opcode::FPVectorEqual32:
            vbinary(false, 65, false);
            return true;
        case Opcode::FPVectorEqual64:
            vbinary(true, 71, false);
            return true;
        case Opcode::FPVectorGreater32:
            vbinary(false, 68, false);
            return true;
        case Opcode::FPVectorGreater64:
            vbinary(true, 74, false);
            return true;
        case Opcode::FPVectorGreaterEqual32:
            vbinary(false, 70, false);
            return true;
        case Opcode::FPVectorGreaterEqual64:
            vbinary(true, 76, false);
            return true;
        case Opcode::FPVectorSqrt32:
            PushFloatV128(op, 0, false);
            SimdOp(227);
            StoreFloatV128(index, false);
            return true;
        case Opcode::FPVectorSqrt64:
            PushFloatV128(op, 0, true);
            SimdOp(239);
            StoreFloatV128(index, true);
            return true;
        case Opcode::FPVectorMulAdd32:
        case Opcode::FPVectorMulAdd64: {
            const bool is64 = op.opcode == Opcode::FPVectorMulAdd64;
            PushFloatV128(op, 0, is64);
            PushFloatV128(op, 1, is64);
            PushFloatV128(op, 2, is64);
            SimdOp(is64 ? 242 : 230);
            SimdOp(is64 ? 240 : 228);
            StoreFloatV128(index, is64);
            return true;
        }
        case Opcode::VectorAdd8:
            vint(110);
            return true;
        case Opcode::VectorAdd16:
            vint(142);
            return true;
        case Opcode::VectorAdd32:
            vint(174);
            return true;
        case Opcode::VectorAdd64:
            vint(206);
            return true;
        case Opcode::VectorSub8:
            vint(113);
            return true;
        case Opcode::VectorSub16:
            vint(145);
            return true;
        case Opcode::VectorSub32:
            vint(177);
            return true;
        case Opcode::VectorSub64:
            vint(209);
            return true;
        case Opcode::VectorMultiply16:
            vint(149);
            return true;
        case Opcode::VectorMultiply32:
            vint(181);
            return true;
        case Opcode::VectorAnd:
            vint(78);
            return true;
        case Opcode::VectorAndNot:
            vint(79);
            return true;
        case Opcode::VectorOr:
            vint(80);
            return true;
        case Opcode::VectorEor:
            vint(81);
            return true;
        case Opcode::VectorEqual8:
            vint(35);
            return true;
        case Opcode::VectorEqual16:
            vint(45);
            return true;
        case Opcode::VectorEqual32:
            vint(55);
            return true;
        case Opcode::VectorNot:
            PushV128(op, 0);
            SimdOp(77);
            StoreV128(index);
            return true;
        case Opcode::VectorZeroUpper:
            code.Get(kFrame);
            PushVectorHalf(op, 0, false);
            code.Mem(Insn::I64Store, u64(index) * kSlotSize + kLoOffset);
            code.Get(kFrame);
            code.I64Const(0);
            code.Mem(Insn::I64Store, u64(index) * kSlotSize + kHiOffset);
            return true;
        case Opcode::ZeroVector:
            code.Get(kFrame);
            code.I64Const(0);
            code.Mem(Insn::I64Store, u64(index) * kSlotSize + kLoOffset);
            code.Get(kFrame);
            code.I64Const(0);
            code.Mem(Insn::I64Store, u64(index) * kSlotSize + kHiOffset);
            return true;
        case Opcode::VectorGetElement32:
        case Opcode::VectorGetElement64: {
            const bool is64 = op.opcode == Opcode::VectorGetElement64;
            const u64 element = ConstantValue(op.args[1]);
            const u32 slot = op.args[0];
            if (IsConstant(slot)) {
                const Slot& constant = block.constants[slot - block.first_constant];
                u64 lanes[2] = {constant.lo, constant.hi};
                u64 value;
                std::memcpy(&value, reinterpret_cast<const u8*>(lanes) + element * (is64 ? 8 : 4), is64 ? 8 : 4);
                code.I64Const(is64 ? value : u32(value));
            } else {
                code.Get(kFrame);
                code.Mem(is64 ? Insn::I64Load : Insn::I64Load32U, u64(slot) * kSlotSize + kLoOffset + element * (is64 ? 8 : 4));
            }
            Define(index);
            return true;
        }
        case Opcode::VectorSetElement32:
        case Opcode::VectorSetElement64: {
            const bool is64 = op.opcode == Opcode::VectorSetElement64;
            const u64 element = ConstantValue(op.args[1]);
            PushV128(op, 0);
            StoreV128(index);
            code.Get(kFrame);
            PushArg(op, 2);
            code.Mem(is64 ? Insn::I64Store : Insn::I64Store32, u64(index) * kSlotSize + kLoOffset + element * (is64 ? 8 : 4));
            return true;
        }
        case Opcode::VectorBroadcast32:
            PushArg(op, 0);
            code.Byte(Insn::I32WrapI64);
            SimdOp(17);  // i32x4.splat
            StoreV128(index);
            return true;
        case Opcode::VectorBroadcast64:
            PushArg(op, 0);
            SimdOp(18);  // i64x2.splat
            StoreV128(index);
            return true;
        default:
            return false;
        }
    }


    // Body

    void EmitBody() {
        // Constants read by interpreter fallbacks must be present in the shared frame.
        for (const u32 slot : helper_constants) {
            const Slot& constant = block.constants[slot - block.first_constant];
            code.Get(kFrame);
            code.I64Const(constant.lo);
            code.Mem(Insn::I64Store, u64(slot) * kSlotSize + kLoOffset);
            if (constant.hi != 0) {
                code.Get(kFrame);
                code.I64Const(constant.hi);
                code.Mem(Insn::I64Store, u64(slot) * kSlotSize + kHiOffset);
            }
        }
        code.I64Const(0);
        code.Set(wrote_pc_local);
        for (size_t i = 0; i < block.ops.size() && !failed; ++i) {
            if (inline_op[i]) {
                // Ops without carry or overflow (And, Or, ...) leave these unset, and locals are
                // shared by every block of the region, so clear them to match the interpreter.
                for (const u32 local : {carry_local[i], overflow_local[i]}) {
                    if (local != kNone) {
                        code.I64Const(0);
                        code.Set(local);
                    }
                }
                EmitInline(i);
                SpillFlags(i);
            } else {
                CallHelper(i);
            }
        }
    }

    void EmitStateStore32(u32 offset) {
        code.Mem(Insn::I64Store32, offset);
    }

    void EmitInline(size_t index) {
        const Op& op = block.ops[index];
        const auto arg = [&](size_t i) { PushArg(op, i); };
        const auto define = [&] { Define(index); };
        const auto binary = [&](u8 opcode, bool mask32) {
            arg(0);
            arg(1);
            code.Byte(opcode);
            if (mask32) {
                Mask32();
            }
            define();
        };
        const u64 frame_lo = u64(index) * kSlotSize + kLoOffset;
        const u64 frame_hi = u64(index) * kSlotSize + kHiOffset;

        switch (op.opcode) {
        case Opcode::Void:
        case Opcode::Breakpoint:
        case Opcode::PushRSB:
            break;

        case Opcode::GetCarryFromOp:
            PushCarry(op.args[0]);
            define();
            break;
        case Opcode::GetOverflowFromOp:
            PushOverflow(op.args[0]);
            define();
            break;
        case Opcode::GetNZCVFromOp:
        case Opcode::GetNZFromOp: {
            const u32 parent = op.args[0];
            const u32 width = op.width;
            // N
            code.Get(value_local[parent]);
            Shr(width - 1);
            And(1);
            Shl(31);
            // Z
            code.Get(value_local[parent]);
            if (width < 64) {
                And((u64(1) << width) - 1);
            }
            code.Byte(Insn::I64Eqz);
            code.Byte(Insn::I64ExtendI32U);
            Shl(30);
            code.Byte(Insn::I64Or);
            if (op.opcode == Opcode::GetNZCVFromOp) {
                PushCarry(parent);
                Shl(29);
                code.Byte(Insn::I64Or);
                PushOverflow(parent);
                Shl(28);
                code.Byte(Insn::I64Or);
            }
            define();
            break;
        }
        case Opcode::GetCFlagFromNZCV:
            arg(0);
            Shr(29);
            And(1);
            define();
            break;
        case Opcode::NZCVFromPackedFlags:
            arg(0);
            And(0xF0000000);
            define();
            break;

        case Opcode::Pack2x32To1x64:
            arg(0);
            Mask32();
            arg(1);
            Shl(32);
            code.Byte(Insn::I64Or);
            define();
            break;
        case Opcode::LeastSignificantWord:
            arg(0);
            Mask32();
            define();
            break;
        case Opcode::LeastSignificantHalf:
            arg(0);
            And(0xFFFF);
            define();
            break;
        case Opcode::LeastSignificantByte:
            arg(0);
            And(0xFF);
            define();
            break;
        case Opcode::MostSignificantWord:
            if (carry_local[index] != kNone) {
                arg(0);
                Shr(31);
                And(1);
                code.Set(carry_local[index]);
            }
            arg(0);
            Shr(32);
            define();
            break;
        case Opcode::MostSignificantBit:
            arg(0);
            Shr(31);
            And(1);
            define();
            break;
        case Opcode::IsZero32:
            arg(0);
            Mask32();
            code.Byte(Insn::I64Eqz);
            code.Byte(Insn::I64ExtendI32U);
            define();
            break;
        case Opcode::IsZero64:
            arg(0);
            code.Byte(Insn::I64Eqz);
            code.Byte(Insn::I64ExtendI32U);
            define();
            break;
        case Opcode::TestBit:
            arg(0);
            arg(1);
            And(63);
            code.Byte(Insn::I64ShrU);
            And(1);
            define();
            break;
        case Opcode::ConditionalSelect32:
        case Opcode::ConditionalSelect64:
        case Opcode::ConditionalSelectNZCV:
            arg(1);
            arg(2);
            PushCond(IR::Cond(ConstantValue(op.args[0])));
            code.Byte(Insn::Select);
            define();
            break;

        case Opcode::LogicalShiftLeft32:
        case Opcode::LogicalShiftRight32:
        case Opcode::ArithmeticShiftRight32:
        case Opcode::BitRotateRight32:
            EmitShift32(index);
            break;
        case Opcode::LogicalShiftLeft64:
            // shift < 64 ? value << shift : 0
            arg(0);
            arg(1);
            And(0xFF);
            code.Byte(Insn::I64Shl);
            code.I64Const(0);
            arg(1);
            And(0xFF);
            code.I64Const(64);
            code.Byte(Insn::I64LtU);
            code.Byte(Insn::Select);
            define();
            break;
        case Opcode::LogicalShiftRight64:
            arg(0);
            arg(1);
            And(0xFF);
            code.Byte(Insn::I64ShrU);
            code.I64Const(0);
            arg(1);
            And(0xFF);
            code.I64Const(64);
            code.Byte(Insn::I64LtU);
            code.Byte(Insn::Select);
            define();
            break;
        case Opcode::ArithmeticShiftRight64:
            // value >> min(shift, 63)
            arg(0);
            arg(1);
            And(0xFF);
            code.I64Const(63);
            arg(1);
            And(0xFF);
            code.I64Const(64);
            code.Byte(Insn::I64LtU);
            code.Byte(Insn::Select);
            code.Byte(Insn::I64ShrS);
            define();
            break;
        case Opcode::BitRotateRight64:
        case Opcode::RotateRightMasked64:
            binary(Insn::I64Rotr, false);
            break;
        case Opcode::RotateRightExtended:
            if (carry_local[index] != kNone) {
                arg(0);
                And(1);
                code.Set(carry_local[index]);
            }
            arg(0);
            Mask32();
            Shr(1);
            arg(1);
            And(1);
            Shl(31);
            code.Byte(Insn::I64Or);
            define();
            break;
        case Opcode::LogicalShiftLeftMasked32:
            arg(0);
            arg(1);
            And(31);
            code.Byte(Insn::I64Shl);
            Mask32();
            define();
            break;
        case Opcode::LogicalShiftLeftMasked64:
            binary(Insn::I64Shl, false);
            break;
        case Opcode::LogicalShiftRightMasked32:
            arg(0);
            Mask32();
            arg(1);
            And(31);
            code.Byte(Insn::I64ShrU);
            define();
            break;
        case Opcode::LogicalShiftRightMasked64:
            binary(Insn::I64ShrU, false);
            break;
        case Opcode::ArithmeticShiftRightMasked32:
            arg(0);
            code.Byte(Insn::I64Extend32S);
            arg(1);
            And(31);
            code.Byte(Insn::I64ShrS);
            Mask32();
            define();
            break;
        case Opcode::ArithmeticShiftRightMasked64:
            binary(Insn::I64ShrS, false);
            break;
        case Opcode::RotateRightMasked32:
            arg(0);
            code.Byte(Insn::I32WrapI64);
            arg(1);
            code.Byte(Insn::I32WrapI64);
            code.Byte(0x78);  // i32.rotr
            code.Byte(Insn::I64ExtendI32U);
            define();
            break;

        case Opcode::Add32:
        case Opcode::Sub32:
            EmitAddSub32(index, op.opcode == Opcode::Sub32);
            break;
        case Opcode::Add64:
        case Opcode::Sub64:
            EmitAddSub64(index, op.opcode == Opcode::Sub64);
            break;
        case Opcode::Mul32:
            binary(Insn::I64Mul, true);
            break;
        case Opcode::Mul64:
            binary(Insn::I64Mul, false);
            break;
        case Opcode::UnsignedDiv32:
        case Opcode::UnsignedDiv64:
        case Opcode::SignedDiv32:
        case Opcode::SignedDiv64:
            EmitDivide(index);
            break;
        case Opcode::And32:
        case Opcode::And64:
            binary(Insn::I64And, false);
            break;
        case Opcode::AndNot32:
        case Opcode::AndNot64:
            arg(0);
            arg(1);
            code.I64Const(~u64(0));
            code.Byte(Insn::I64Xor);
            code.Byte(Insn::I64And);
            define();
            break;
        case Opcode::Eor32:
        case Opcode::Eor64:
            binary(Insn::I64Xor, false);
            break;
        case Opcode::Or32:
        case Opcode::Or64:
            binary(Insn::I64Or, false);
            break;
        case Opcode::Not32:
            arg(0);
            code.I64Const(0xFFFFFFFF);
            code.Byte(Insn::I64Xor);
            Mask32();
            define();
            break;
        case Opcode::Not64:
            arg(0);
            code.I64Const(~u64(0));
            code.Byte(Insn::I64Xor);
            define();
            break;
        case Opcode::SignExtendByteToWord:
            arg(0);
            code.Byte(Insn::I64Extend8S);
            Mask32();
            define();
            break;
        case Opcode::SignExtendHalfToWord:
            arg(0);
            code.Byte(Insn::I64Extend16S);
            Mask32();
            define();
            break;
        case Opcode::SignExtendByteToLong:
            arg(0);
            code.Byte(Insn::I64Extend8S);
            define();
            break;
        case Opcode::SignExtendHalfToLong:
            arg(0);
            code.Byte(Insn::I64Extend16S);
            define();
            break;
        case Opcode::SignExtendWordToLong:
            arg(0);
            code.Byte(Insn::I64Extend32S);
            define();
            break;
        case Opcode::ZeroExtendByteToWord:
        case Opcode::ZeroExtendByteToLong:
            arg(0);
            And(0xFF);
            define();
            break;
        case Opcode::ZeroExtendHalfToWord:
        case Opcode::ZeroExtendHalfToLong:
            arg(0);
            And(0xFFFF);
            define();
            break;
        case Opcode::ZeroExtendWordToLong:
            arg(0);
            Mask32();
            define();
            break;
        case Opcode::ZeroExtendLongToQuad:
            code.Get(kFrame);
            arg(0);
            code.Mem(Insn::I64Store, frame_lo);
            code.Get(kFrame);
            code.I64Const(0);
            code.Mem(Insn::I64Store, frame_hi);
            break;
        case Opcode::CountLeadingZeros32:
            arg(0);
            Mask32();
            code.Byte(Insn::I64Clz);
            code.I64Const(32);
            code.Byte(Insn::I64Sub);
            define();
            break;
        case Opcode::CountLeadingZeros64:
            arg(0);
            code.Byte(Insn::I64Clz);
            define();
            break;
        case Opcode::ExtractRegister32:
        case Opcode::ExtractRegister64: {
            const bool is32 = op.opcode == Opcode::ExtractRegister32;
            const u32 lsb = u32(ConstantValue(op.args[2])) & (is32 ? 31 : 63);
            arg(0);
            if (lsb != 0) {
                if (is32) {
                    Mask32();
                }
                Shr(lsb);
                arg(1);
                Shl((is32 ? 32 : 64) - lsb);
                code.Byte(Insn::I64Or);
                if (is32) {
                    Mask32();
                }
            }
            define();
            break;
        }
        case Opcode::ReplicateBit32:
        case Opcode::ReplicateBit64:
            code.I64Const(0);
            arg(0);
            arg(1);
            code.Byte(Insn::I64ShrU);
            And(1);
            code.Byte(Insn::I64Sub);
            if (op.opcode == Opcode::ReplicateBit32) {
                Mask32();
            }
            define();
            break;
        case Opcode::MaxSigned32:
        case Opcode::MinSigned32:
        case Opcode::MaxUnsigned32:
        case Opcode::MinUnsigned32:
        case Opcode::MaxSigned64:
        case Opcode::MinSigned64:
        case Opcode::MaxUnsigned64:
        case Opcode::MinUnsigned64:
            EmitMinMax(index);
            break;

        // A64 state
        case Opcode::A64SetCheckBit:
        case Opcode::A32SetCheckBit:
            code.Get(kState);
            arg(0);
            code.Mem(Insn::I64Store8, target.check_bit_offset);
            break;
        case Opcode::A64GetCFlag:
        case Opcode::A32GetCFlag:
            PushNZCV();
            Shr(29);
            And(1);
            define();
            break;
        case Opcode::A64GetNZCVRaw:
            code.Get(kState);
            code.Mem(Insn::I64Load32U, target.nzcv_offset);
            define();
            break;
        case Opcode::A64SetNZCVRaw:
        case Opcode::A64SetNZCV:
        case Opcode::A32SetCpsrNZCV:
        case Opcode::A32SetCpsrNZCVRaw:
            SetNZCV([&] {
                arg(0);
                And(0xF0000000);
            });
            break;
        case Opcode::A64GetW:
            code.Get(kState);
            code.Mem(Insn::I64Load32U, target.reg_offset + 8 * ConstantValue(op.args[0]));
            define();
            break;
        case Opcode::A64GetX:
            code.Get(kState);
            code.Mem(Insn::I64Load, target.reg_offset + 8 * ConstantValue(op.args[0]));
            define();
            break;
        case Opcode::A64GetS:
        case Opcode::A64GetD:
        case Opcode::A64GetQ: {
            const u64 vec = target.ext_offset + 16 * ConstantValue(op.args[0]);
            code.Get(kFrame);
            code.Get(kState);
            code.Mem(op.opcode == Opcode::A64GetS ? Insn::I64Load32U : Insn::I64Load, vec);
            code.Mem(Insn::I64Store, frame_lo);
            code.Get(kFrame);
            if (op.opcode == Opcode::A64GetQ) {
                code.Get(kState);
                code.Mem(Insn::I64Load, vec + 8);
            } else {
                code.I64Const(0);
            }
            code.Mem(Insn::I64Store, frame_hi);
            break;
        }
        case Opcode::A64GetSP:
            code.Get(kState);
            code.Mem(Insn::I64Load, target.sp_offset);
            define();
            break;
        case Opcode::A64SetW:
            code.Get(kState);
            arg(1);
            Mask32();
            code.Mem(Insn::I64Store, target.reg_offset + 8 * ConstantValue(op.args[0]));
            break;
        case Opcode::A64SetX:
            code.Get(kState);
            arg(1);
            code.Mem(Insn::I64Store, target.reg_offset + 8 * ConstantValue(op.args[0]));
            break;
        case Opcode::A64SetS:
        case Opcode::A64SetD:
        case Opcode::A64SetQ: {
            const u64 vec = target.ext_offset + 16 * ConstantValue(op.args[0]);
            code.Get(kState);
            PushVectorHalf(op, 1, false);
            if (op.opcode == Opcode::A64SetS) {
                Mask32();
            }
            code.Mem(Insn::I64Store, vec);
            code.Get(kState);
            if (op.opcode == Opcode::A64SetQ) {
                PushVectorHalf(op, 1, true);
            } else {
                code.I64Const(0);
            }
            code.Mem(Insn::I64Store, vec + 8);
            break;
        }
        case Opcode::A64SetSP:
            code.Get(kState);
            arg(0);
            code.Mem(Insn::I64Store, target.sp_offset);
            break;
        case Opcode::A64SetPC:
            code.Get(kState);
            arg(0);
            code.Mem(Insn::I64Store, target.pc_offset);
            break;
        case Opcode::A64GetTPIDR:
        case Opcode::A64GetTPIDRRO: {
            const u64 address = op.opcode == Opcode::A64GetTPIDR ? target.tpidr_el0 : target.tpidrro_el0;
            if (address == 0) {
                code.I64Const(0);
            } else {
                code.I64Const(address);
                code.Mem(Insn::I64Load, 0);
            }
            define();
            break;
        }
        case Opcode::A64SetTPIDR:
            if (target.tpidr_el0 != 0) {
                code.I64Const(target.tpidr_el0);
                arg(0);
                code.Mem(Insn::I64Store, 0);
            }
            break;
        case Opcode::A64ReadMemory8:
        case Opcode::A32ReadMemory8:
            EmitRead(index, 1);
            break;
        case Opcode::A64ReadMemory16:
        case Opcode::A32ReadMemory16:
            EmitRead(index, 2);
            break;
        case Opcode::A64ReadMemory32:
        case Opcode::A32ReadMemory32:
            EmitRead(index, 4);
            break;
        case Opcode::A64ReadMemory64:
        case Opcode::A32ReadMemory64:
            EmitRead(index, 8);
            break;
        case Opcode::A64WriteMemory8:
        case Opcode::A32WriteMemory8:
            EmitWrite(index, 1);
            break;
        case Opcode::A64WriteMemory16:
        case Opcode::A32WriteMemory16:
            EmitWrite(index, 2);
            break;
        case Opcode::A64WriteMemory32:
        case Opcode::A32WriteMemory32:
            EmitWrite(index, 4);
            break;
        case Opcode::A64WriteMemory64:
        case Opcode::A32WriteMemory64:
            EmitWrite(index, 8);
            break;

        // A32 state
        case Opcode::A32GetRegister: {
            const size_t r = A32::RegNumber(A32::Reg(ConstantValue(op.args[0])));
            if (cached) {
                code.Get(cached->reg[r]);
            } else {
                code.Get(kState);
                code.Mem(Insn::I64Load32U, target.reg_offset + 4 * u64(r));
            }
            define();
            break;
        }
        case Opcode::A32SetRegister: {
            const size_t r = A32::RegNumber(A32::Reg(ConstantValue(op.args[0])));
            if (cached) {
                arg(1);
                Mask32();
                code.Set(cached->reg[r]);
            } else {
                code.Get(kState);
                arg(1);
                EmitStateStore32(target.reg_offset + 4 * u64(r));
            }
            break;
        }
        case Opcode::A32GetExtendedRegister32:
            code.Get(kState);
            code.Mem(Insn::I64Load32U, target.ext_offset + 4 * (ConstantValue(op.args[0]) - u64(A32::ExtReg::S0)));
            define();
            break;
        case Opcode::A32SetExtendedRegister32:
            code.Get(kState);
            arg(1);
            EmitStateStore32(target.ext_offset + 4 * (ConstantValue(op.args[0]) - u64(A32::ExtReg::S0)));
            break;
        case Opcode::A32GetExtendedRegister64:
            code.Get(kState);
            code.Mem(Insn::I64Load, target.ext_offset + 8 * (ConstantValue(op.args[0]) - u64(A32::ExtReg::D0)));
            define();
            break;
        case Opcode::A32SetExtendedRegister64:
            code.Get(kState);
            arg(1);
            code.Mem(Insn::I64Store, target.ext_offset + 8 * (ConstantValue(op.args[0]) - u64(A32::ExtReg::D0)));
            break;
        case Opcode::A32GetVector: {
            const A32::ExtReg reg = A32::ExtReg(ConstantValue(op.args[0]));
            const bool is_double = A32::IsDoubleExtReg(reg);
            const u64 offset = is_double ? target.ext_offset + 8 * (u64(reg) - u64(A32::ExtReg::D0))
                                         : target.ext_offset + 16 * (u64(reg) - u64(A32::ExtReg::Q0));
            code.Get(kFrame);
            code.Get(kState);
            code.Mem(Insn::I64Load, offset);
            code.Mem(Insn::I64Store, frame_lo);
            code.Get(kFrame);
            if (is_double) {
                code.I64Const(0);
            } else {
                code.Get(kState);
                code.Mem(Insn::I64Load, offset + 8);
            }
            code.Mem(Insn::I64Store, frame_hi);
            break;
        }
        case Opcode::A32SetVector: {
            const A32::ExtReg reg = A32::ExtReg(ConstantValue(op.args[0]));
            const bool is_double = A32::IsDoubleExtReg(reg);
            const u64 offset = is_double ? target.ext_offset + 8 * (u64(reg) - u64(A32::ExtReg::D0))
                                         : target.ext_offset + 16 * (u64(reg) - u64(A32::ExtReg::Q0));
            code.Get(kState);
            PushVectorHalf(op, 1, false);
            code.Mem(Insn::I64Store, offset);
            if (!is_double) {
                code.Get(kState);
                PushVectorHalf(op, 1, true);
                code.Mem(Insn::I64Store, offset + 8);
            }
            break;
        }
        case Opcode::A32SetCpsrNZCVQ:
            SetNZCV([&] {
                arg(0);
                And(0xF0000000);
            });
            code.Get(kState);
            arg(0);
            Shr(27);
            And(1);
            EmitStateStore32(target.q_offset);
            break;
        case Opcode::A32SetCpsrNZ:
            SetNZCV([&] {
                PushNZCV();
                And(0x30000000);
                arg(0);
                And(0xC0000000);
                code.Byte(Insn::I64Or);
            });
            break;
        case Opcode::A32SetCpsrNZC:
            SetNZCV([&] {
                PushNZCV();
                And(0x10000000);
                arg(0);
                And(0xC0000000);
                code.Byte(Insn::I64Or);
                arg(1);
                And(1);
                Shl(29);
                code.Byte(Insn::I64Or);
            });
            break;
        case Opcode::A32OrQFlag:
            code.Get(kState);
            code.Get(kState);
            code.Mem(Insn::I64Load32U, target.q_offset);
            arg(0);
            And(1);
            code.Byte(Insn::I64Or);
            EmitStateStore32(target.q_offset);
            break;
        case Opcode::A32GetGEFlags:
            code.Get(kState);
            code.Mem(Insn::I64Load32U, target.ge_offset);
            define();
            break;
        case Opcode::A32SetGEFlags:
            code.Get(kState);
            arg(0);
            EmitStateStore32(target.ge_offset);
            break;
        case Opcode::A32BXWritePC: {
            const u32 upper_without_t = u32(A32::LocationDescriptor{block.end_location}.SetSingleStepping(false).UniqueHash() >> 32) & 0xFFFFFFFE;
            // pc = value & (value & 1 ? ~1 : ~3)
            if (!cached) {
                code.Get(kState);
            }
            arg(0);
            code.I64Const(0xFFFFFFFE);
            code.I64Const(0xFFFFFFFC);
            arg(0);
            And(1);
            code.Byte(Insn::I32WrapI64);
            code.Byte(Insn::Select);
            code.Byte(Insn::I64And);
            if (cached) {
                code.Set(cached->reg[15]);
            } else {
                EmitStateStore32(target.reg_offset + 4 * 15);
            }
            // upper = upper_without_t | (value & 1)
            code.Get(kState);
            arg(0);
            And(1);
            code.I64Const(upper_without_t);
            code.Byte(Insn::I64Or);
            EmitStateStore32(target.upper_offset);
            code.I64Const(1);
            code.Set(wrote_pc_local);
            break;
        }
        case Opcode::A32GetFpscrNZCV:
            code.Get(kState);
            code.Mem(Insn::I64Load32U, target.fpsr_nzcv_offset);
            define();
            break;
        case Opcode::A32SetFpscrNZCV:
            code.Get(kState);
            arg(0);
            And(0xF0000000);
            code.Mem(Insn::I64Store32, target.fpsr_nzcv_offset);
            break;
        case Opcode::A32ClearExclusive:
            code.Get(kState);
            code.I64Const(0);
            code.Mem(Insn::I64Store32, target.exclusive_state_offset);
            break;
        case Opcode::A32ExclusiveReadMemory8:
            EmitExclusiveRead(index, 1);
            break;
        case Opcode::A32ExclusiveReadMemory16:
            EmitExclusiveRead(index, 2);
            break;
        case Opcode::A32ExclusiveReadMemory32:
            EmitExclusiveRead(index, 4);
            break;
        case Opcode::A32ExclusiveReadMemory64:
            EmitExclusiveRead(index, 8);
            break;
        case Opcode::A32ExclusiveWriteMemory8:
            EmitExclusiveWrite(index, 1);
            break;
        case Opcode::A32ExclusiveWriteMemory16:
            EmitExclusiveWrite(index, 2);
            break;
        case Opcode::A32ExclusiveWriteMemory32:
            EmitExclusiveWrite(index, 4);
            break;
        case Opcode::A32ExclusiveWriteMemory64:
            EmitExclusiveWrite(index, 8);
            break;
        case Opcode::A32CoprocGetOneWord:
            code.I64Const(target.coproc_read_pointer(ConstantValue(op.args[0])));
            code.Mem(Insn::I64Load32U, 0);
            define();
            break;
        case Opcode::A32UpdateUpperLocationDescriptor: {
            const u32 upper = u32(A32::LocationDescriptor{block.end_location}.SetSingleStepping(false).UniqueHash() >> 32);
            code.Get(wrote_pc_local);
            code.Byte(Insn::I64Eqz);
            code.If();
            code.Get(kState);
            code.I64Const(upper);
            EmitStateStore32(target.upper_offset);
            code.End();
            break;
        }

        default:
            if (!EmitFloat(index)) {
                failed = true;
            }
            break;
        }
    }

    /// Pushes the low or high half of a 128-bit argument, which always lives in the frame.
    void PushVectorHalf(const Op& op, size_t index, bool high) {
        const u32 slot = op.args[index];
        if (IsConstant(slot)) {
            const Slot& constant = block.constants[slot - block.first_constant];
            code.I64Const(high ? constant.hi : constant.lo);
            return;
        }
        code.Get(kFrame);
        code.Mem(Insn::I64Load, u64(slot) * kSlotSize + (high ? kHiOffset : kLoOffset));
    }

    void EmitShift32(size_t index) {
        const Op& op = block.ops[index];
        const bool want_carry = carry_local[index] != kNone;
        // tmp0 = value (zero-extended), tmp1 = shift amount (0-255)
        PushArg(op, 0);
        Mask32();
        code.Set(tmp0);
        PushArg(op, 1);
        And(0xFF);
        code.Set(tmp1);

        switch (op.opcode) {
        case Opcode::LogicalShiftLeft32:
            // result = shift < 32 ? (value << shift) & mask : 0
            code.Get(tmp0);
            code.Get(tmp1);
            code.Byte(Insn::I64Shl);
            Mask32();
            code.I64Const(0);
            code.Get(tmp1);
            code.I64Const(32);
            code.Byte(Insn::I64LtU);
            code.Byte(Insn::Select);
            Define(index);
            if (want_carry) {
                // carry = shift == 0 ? carry_in : shift <= 32 ? (value >> (32 - shift)) & 1 : 0
                code.Get(tmp0);
                code.I64Const(32);
                code.Get(tmp1);
                code.Byte(Insn::I64Sub);
                code.Byte(Insn::I64ShrU);
                And(1);
                code.I64Const(0);
                code.Get(tmp1);
                code.I64Const(32);
                code.Byte(Insn::I64LeU);
                code.Byte(Insn::Select);
                EmitCarryIn(op);
            }
            break;
        case Opcode::LogicalShiftRight32:
            code.Get(tmp0);
            code.Get(tmp1);
            code.Byte(Insn::I64ShrU);
            code.I64Const(0);
            code.Get(tmp1);
            code.I64Const(32);
            code.Byte(Insn::I64LtU);
            code.Byte(Insn::Select);
            Define(index);
            if (want_carry) {
                // carry = shift == 0 ? carry_in : shift <= 32 ? (value >> (shift - 1)) & 1 : 0
                code.Get(tmp0);
                code.Get(tmp1);
                code.I64Const(1);
                code.Byte(Insn::I64Sub);
                code.Byte(Insn::I64ShrU);
                And(1);
                code.I64Const(0);
                code.Get(tmp1);
                code.I64Const(32);
                code.Byte(Insn::I64LeU);
                code.Byte(Insn::Select);
                EmitCarryIn(op);
            }
            break;
        case Opcode::ArithmeticShiftRight32:
            // result = (sext(value) >> min(shift, 31)) & mask
            code.Get(tmp0);
            code.Byte(Insn::I64Extend32S);
            code.Set(tmp2);
            code.Get(tmp2);
            code.Get(tmp1);
            code.I64Const(31);
            code.Get(tmp1);
            code.I64Const(32);
            code.Byte(Insn::I64LtU);
            code.Byte(Insn::Select);
            code.Byte(Insn::I64ShrS);
            Mask32();
            Define(index);
            if (want_carry) {
                // carry = shift == 0 ? carry_in : (sext(value) >> (min(shift, 32) - 1)) & 1
                code.Get(tmp2);
                code.Get(tmp1);
                code.I64Const(32);
                code.Get(tmp1);
                code.I64Const(33);
                code.Byte(Insn::I64LtU);
                code.Byte(Insn::Select);
                code.I64Const(1);
                code.Byte(Insn::I64Sub);
                code.Byte(Insn::I64ShrS);
                And(1);
                EmitCarryIn(op);
            }
            break;
        case Opcode::BitRotateRight32:
            // result = shift == 0 ? value : rotr32(value, shift & 31)
            code.Get(tmp0);
            code.Byte(Insn::I32WrapI64);
            code.Get(tmp1);
            code.Byte(Insn::I32WrapI64);
            code.Byte(0x78);  // i32.rotr
            code.Byte(Insn::I64ExtendI32U);
            Define(index);
            if (want_carry) {
                // carry = shift == 0 ? carry_in : result >> 31
                code.Get(value_local[index]);
                Shr(31);
                EmitCarryIn(op);
            }
            break;
        default:
            failed = true;
            break;
        }
    }

    /// With the shifted-out carry on the stack, selects the carry-in when the shift was 0.
    void EmitCarryIn(const Op& op) {
        const size_t index = &op - block.ops.data();
        PushArg(op, 2);
        And(1);
        code.Get(tmp1);
        code.I64Const(0);
        code.Byte(Insn::I64Ne);
        code.Byte(Insn::Select);
        code.Set(carry_local[index]);
    }

    void EmitAddSub32(size_t index, bool subtract) {
        const Op& op = block.ops[index];
        // tmp0 = a, tmp1 = b (inverted for subtraction), tmp2 = wide sum
        PushArg(op, 0);
        Mask32();
        code.Set(tmp0);
        PushArg(op, 1);
        if (subtract) {
            code.I64Const(0xFFFFFFFF);
            code.Byte(Insn::I64Xor);
        }
        Mask32();
        code.Set(tmp1);
        code.Get(tmp0);
        code.Get(tmp1);
        code.Byte(Insn::I64Add);
        PushArg(op, 2);
        And(1);
        code.Byte(Insn::I64Add);
        code.Set(tmp2);
        code.Get(tmp2);
        Mask32();
        Define(index);
        if (carry_local[index] != kNone) {
            code.Get(tmp2);
            Shr(32);
            And(1);
            code.Set(carry_local[index]);
        }
        if (overflow_local[index] != kNone) {
            // ((~(a ^ b) & (a ^ result)) >> 31) & 1
            code.Get(tmp0);
            code.Get(tmp1);
            code.Byte(Insn::I64Xor);
            code.I64Const(~u64(0));
            code.Byte(Insn::I64Xor);
            code.Get(tmp0);
            code.Get(value_local[index]);
            code.Byte(Insn::I64Xor);
            code.Byte(Insn::I64And);
            Shr(31);
            And(1);
            code.Set(overflow_local[index]);
        }
    }

    void EmitAddSub64(size_t index, bool subtract) {
        const Op& op = block.ops[index];
        PushArg(op, 0);
        code.Set(tmp0);
        PushArg(op, 1);
        if (subtract) {
            code.I64Const(~u64(0));
            code.Byte(Insn::I64Xor);
        }
        code.Set(tmp1);
        PushArg(op, 2);
        And(1);
        code.Set(tmp3);
        code.Get(tmp0);
        code.Get(tmp1);
        code.Byte(Insn::I64Add);
        code.Get(tmp3);
        code.Byte(Insn::I64Add);
        Define(index);
        if (carry_local[index] != kNone) {
            // carry = result < a || (carry_in && result == a)
            code.Get(value_local[index]);
            code.Get(tmp0);
            code.Byte(Insn::I64LtU);
            code.Get(value_local[index]);
            code.Get(tmp0);
            code.Byte(Insn::I64Eq);
            code.Get(tmp3);
            code.Byte(Insn::I32WrapI64);
            code.Byte(Insn::I32And);
            code.Byte(Insn::I32Or);
            code.Byte(Insn::I64ExtendI32U);
            code.Set(carry_local[index]);
        }
        if (overflow_local[index] != kNone) {
            code.Get(tmp0);
            code.Get(tmp1);
            code.Byte(Insn::I64Xor);
            code.I64Const(~u64(0));
            code.Byte(Insn::I64Xor);
            code.Get(tmp0);
            code.Get(value_local[index]);
            code.Byte(Insn::I64Xor);
            code.Byte(Insn::I64And);
            Shr(63);
            code.Set(overflow_local[index]);
        }
    }

    void EmitDivide(size_t index) {
        const Op& op = block.ops[index];
        const bool is32 = op.opcode == Opcode::UnsignedDiv32 || op.opcode == Opcode::SignedDiv32;
        const bool is_signed = op.opcode == Opcode::SignedDiv32 || op.opcode == Opcode::SignedDiv64;
        for (size_t i = 0; i < 2; ++i) {
            PushArg(op, i);
            if (is32) {
                if (is_signed) {
                    code.Byte(Insn::I64Extend32S);
                } else {
                    Mask32();
                }
            }
            code.Set(i == 0 ? tmp0 : tmp1);
        }

        code.Get(tmp1);
        code.Byte(Insn::I64Eqz);
        code.IfI64();
        code.I64Const(0);
        code.Else();
        if (is_signed && !is32) {
            // INT64_MIN / -1 traps in Wasm; the guest result is -n.
            code.Get(tmp1);
            code.I64Const(~u64(0));
            code.Byte(Insn::I64Eq);
            code.IfI64();
            code.I64Const(0);
            code.Get(tmp0);
            code.Byte(Insn::I64Sub);
            code.Else();
            code.Get(tmp0);
            code.Get(tmp1);
            code.Byte(Insn::I64DivS);
            code.End();
        } else {
            code.Get(tmp0);
            code.Get(tmp1);
            code.Byte(is_signed ? Insn::I64DivS : Insn::I64DivU);
        }
        code.End();
        if (is32) {
            Mask32();
        }
        Define(index);
    }

    void EmitMinMax(size_t index) {
        const Op& op = block.ops[index];
        const bool is32 = op.opcode == Opcode::MaxSigned32 || op.opcode == Opcode::MinSigned32 ||
                          op.opcode == Opcode::MaxUnsigned32 || op.opcode == Opcode::MinUnsigned32;
        const bool is_signed = op.opcode == Opcode::MaxSigned32 || op.opcode == Opcode::MinSigned32 ||
                               op.opcode == Opcode::MaxSigned64 || op.opcode == Opcode::MinSigned64;
        const bool is_max = op.opcode == Opcode::MaxSigned32 || op.opcode == Opcode::MaxUnsigned32 ||
                            op.opcode == Opcode::MaxSigned64 || op.opcode == Opcode::MaxUnsigned64;
        const auto load = [&](size_t i, u32 local) {
            PushArg(op, i);
            if (is32) {
                if (is_signed) {
                    code.Byte(Insn::I64Extend32S);
                } else {
                    Mask32();
                }
            }
            code.Set(local);
        };
        load(0, tmp0);
        load(1, tmp1);
        code.Get(tmp0);
        code.Get(tmp1);
        code.Get(tmp0);
        code.Get(tmp1);
        code.Byte(is_signed ? (is_max ? Insn::I64GtS : Insn::I64LtS) : (is_max ? Insn::I64GtU : Insn::I64LtU));
        code.Byte(Insn::Select);
        if (is32) {
            Mask32();
        }
        Define(index);
    }

    const CompiledBlock& block;
    const JitTarget& target;
    Code& code;
    bool failed = false;

    u32 next_local;
    const RegionLocals* cached;
    std::vector<u32> helper_constants;
    std::vector<bool> inline_op;
    std::vector<u32> value_local;
    std::vector<u32> carry_local;
    std::vector<u32> overflow_local;
    std::vector<bool> spill;
    std::vector<bool> spill_flags;
    std::vector<bool> wants_carry;
    std::vector<bool> wants_overflow;
    u32 tmp0 = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0;
    u32 nzcv_tmp = 0;
    u32 wrote_pc_local = 0;
};

}  // Anonymous namespace

namespace {

/// Compiles a set of linked blocks into one function. blocks[0] is the entry; control moves
/// between member blocks through a dispatch loop and returns to the host dispatcher otherwise.
class RegionCompiler {
public:
    RegionCompiler(const std::vector<CompiledBlock*>& blocks_, const JitTarget& target_)
        : blocks{blocks_}, target{target_} {
        for (size_t i = 0; i < blocks.size(); ++i) {
            index_of.emplace(blocks[i]->location.Value(), u32(i));
        }
    }

    bool Compile(std::vector<u8>& out) {
        try {
            return CompileImpl(out);
        } catch (const std::bad_alloc&) {
            EM_ASM({
                err('dynarmic: region allocation failure: entry=' + $0.toString(16) +
                    ' blocks=' + $1 + ' active=' + $2 + ' code=' + $3 +
                    ' capacity=' + $4 + ' heap=' + $5 + '; using interpreter');
            }, blocks.front()->location.Value(), blocks.size(), current_index,
               code.bytes.size(), code.bytes.capacity(), emscripten_get_heap_size());
            return false;
        }
    }

    bool CompileImpl(std::vector<u8>& out) {
        const u32 n = u32(blocks.size());
        u32 locals_end = kLocalBase;

        if (target.a32) {
            // Guest registers, flags and the cycle budget live in locals for the whole region.
            cached.emplace();
            for (u32 r = 0; r < 16; ++r) {
                cached->reg[r] = kCachedBase + r;
            }
            cached->nzcv = kCachedBase + 16;
            cached->cycles = kCachedBase + 17;
            cached->tlb_page = kCachedBase + 18;
            cached->tlb_base = kCachedBase + 19;
            BlockCompiler loader{*blocks[0], target, code, kLocalBase, &*cached};
            loader.ReloadState();
        }
        const RegionLocals* cached_ptr = cached ? &*cached : nullptr;

        code.I64Const(0);
        code.Set(kCurrent);
        code.BlockStart();
        exit_nest = code.nest;
        code.LoopStart();
        loop_nest = code.nest;
        for (u32 i = 0; i < n; ++i) {
            code.BlockStart();
        }
        code.Get(kCurrent);
        code.Byte(Insn::I32WrapI64);
        code.Byte(0x0E);  // br_table
        code.U32(n);
        for (u32 i = 0; i < n; ++i) {
            code.U32(i);
        }
        code.U32(n - 1);

        for (u32 i = 0; i < n; ++i) {
            code.End();
            current_index = i;
            const CompiledBlock& block = *blocks[i];
            BlockCompiler compiler{block, target, code, kLocalBase, cached_ptr};
            compiler.Prepare();
            if (block.cond != IR::Cond::AL) {
                // Conditional A32 block: skip the body when the condition fails.
                compiler.PushCondition(block.cond);
                code.Byte(Insn::I32Eqz);
                code.If();
                SubtractCycles(block.cond_failed_cycle_count);
                EmitLink(block.cond_failed_location);
                code.End();
            }
            if (!compiler.Emit()) {
                return false;
            }
            locals_end = std::max(locals_end, compiler.LocalsEnd());
            SubtractCycles(block.cycle_count);
            EmitTerminal(compiler, block.terminal);
            Exit();
        }
        code.End();  // loop
        code.End();  // exit block
        if (cached) {
            BlockCompiler flusher{*blocks[0], target, code, kLocalBase, &*cached};
            flusher.FlushState();
        }

        BuildModule(out, locals_end);
        return true;
    }

private:
    static constexpr u32 kCurrent = 4;
    static constexpr u32 kCachedBase = 5;
    static constexpr u32 kLocalBase = kV128C + 1;
    static_assert(kCachedBase + 20 == kF32A);

    void Exit() {
        code.BrTo(exit_nest);
    }

    void PushCycles() {
        if (cached) {
            code.Get(cached->cycles);
        } else {
            code.I64Const(target.cycles_remaining);
            code.Mem(Insn::I64Load, 0);
        }
    }

    void SubtractCycles(u64 cycles) {
        if (cycles == 0) {
            return;
        }
        if (cached) {
            code.Get(cached->cycles);
            code.I64Const(cycles);
            code.Byte(Insn::I64Sub);
            code.Set(cached->cycles);
            return;
        }
        code.I64Const(target.cycles_remaining);
        code.I64Const(target.cycles_remaining);
        code.Mem(Insn::I64Load, 0);
        code.I64Const(cycles);
        code.Byte(Insn::I64Sub);
        code.Mem(Insn::I64Store, 0);
    }

    void EmitTerminal(BlockCompiler& compiler, const IR::Term::Terminal& terminal) {
        using namespace IR::Term;
        std::visit([&](const auto& term) {
            using T = std::decay_t<decltype(term)>;
            if constexpr (std::is_same_v<T, LeafTerminal>) {
                EmitLeaf(term);
            } else if constexpr (std::is_same_v<T, If>) {
                compiler.PushCondition(term.if_);
                code.If();
                EmitLeaf(term.then_);
                code.Else();
                EmitLeaf(term.else_);
                code.End();
            } else if constexpr (std::is_same_v<T, CheckBit>) {
                code.Get(kState);
                code.Mem(Insn::I64Load8U, target.check_bit_offset);
                code.Byte(Insn::I32WrapI64);
                code.If();
                EmitLeaf(term.then_);
                code.Else();
                EmitLeaf(term.else_);
                code.End();
            } else if constexpr (std::is_same_v<T, CheckHalt>) {
                PushHalted();
                code.If();
                Exit();
                code.End();
                EmitLeaf(term.else_);
            }
        }, terminal);
    }

    void EmitLeaf(const IR::Term::LeafTerminal& terminal) {
        using namespace IR::Term;
        std::visit([&](const auto& term) {
            using T = std::decay_t<decltype(term)>;
            if constexpr (std::is_same_v<T, LinkBlock> || std::is_same_v<T, LinkBlockFast>) {
                EmitLink(term.next);
            } else {
                Exit();
            }
        }, terminal);
    }

    void PushHalted() {
        code.I64Const(target.halt_reason);
        code.Mem(Insn::I64Load32U, 0);
        code.I64Const(0);
        code.Byte(Insn::I64Ne);
    }

    /// Writes the location of `next` to the guest state and continues there if it is part of
    /// this region and there is budget left; otherwise returns to the dispatcher.
    void EmitLink(IR::LocationDescriptor next) {
        IR::LocationDescriptor location = next;
        if (target.a32) {
            const A32::LocationDescriptor a32{next};
            const A32::LocationDescriptor plain = a32.SetSingleStepping(false);
            location = plain;
            if (cached) {
                code.I64Const(plain.PC());
                code.Set(cached->reg[15]);
            } else {
                code.Get(kState);
                code.I64Const(plain.PC());
                code.Mem(Insn::I64Store32, target.reg_offset + 4 * 15);
            }
            code.Get(kState);
            code.I64Const(u32(plain.UniqueHash() >> 32));
            code.Mem(Insn::I64Store32, target.upper_offset);
        } else {
            const A64::LocationDescriptor a64{next};
            location = a64.SetSingleStepping(false);
            code.Get(kState);
            code.I64Const(a64.PC());
            code.Mem(Insn::I64Store, target.pc_offset);
        }

        const auto it = index_of.find(location.Value());
        if (it == index_of.end()) {
            Exit();
            return;
        }
        code.I64Const(it->second);
        code.Set(kCurrent);
        if (it->second > current_index && !target.cycle_counting) {
            // Without cycle counting only loops need to observe halts, and forward edges cannot
            // loop; check halts on backward edges only.
            code.BrTo(loop_nest);
            return;
        }
        // Continue while cycles remain and nothing asked execution to halt.
        PushCycles();
        code.I64Const(0);
        code.Byte(Insn::I64LeS);
        PushHalted();
        code.Byte(Insn::I32Or);
        code.If();
        Exit();
        code.End();
        code.BrTo(loop_nest);
    }

    void BuildModule(std::vector<u8>& out, u32 locals_end) {
        Writer module;
        module.bytes = {0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00};

        // Types: 0 = region (i64 i64 i64 i64) -> (), 1 = helper (i64 i64 i32) -> ().
        Writer types;
        types.U32(2);
        types.Byte(0x60);
        types.U32(4);
        for (int i = 0; i < 4; ++i) {
            types.Byte(u8(ValType::I64));
        }
        types.U32(0);
        types.Byte(0x60);
        types.U32(3);
        types.Byte(u8(ValType::I64));
        types.Byte(u8(ValType::I64));
        types.Byte(u8(ValType::I32));
        types.U32(0);
        Section(module, 1, types);

        // Imports: shared 64-bit memory and the 64-bit function table.
        Writer imports;
        imports.U32(2);
        imports.Name("e");
        imports.Name("m");
        imports.Byte(0x02);
        imports.Byte(0x07);  // has max | shared | 64-bit
        imports.U64(0);
        imports.U64((u64(16) << 30) >> 16);
        imports.Name("e");
        imports.Name("t");
        imports.Byte(0x01);
        imports.Byte(0x70);  // funcref
        imports.Byte(0x04);  // 64-bit, no max
        imports.U64(0);
        Section(module, 2, imports);

        Writer functions;
        functions.U32(1);
        functions.U32(0);
        Section(module, 3, functions);

        Writer exports;
        exports.U32(1);
        exports.Name("f");
        exports.Byte(0x00);
        exports.U32(0);
        Section(module, 7, exports);

        // Local groups: i64 current + cached state, f32, f64, v128 scratch, then i64 values.
        Writer body;
        body.U32(5);
        body.U32(kF32A - 4);
        body.Byte(u8(ValType::I64));
        body.U32(2);
        body.Byte(0x7D);  // f32
        body.U32(2);
        body.Byte(0x7C);  // f64
        body.U32(3);
        body.Byte(0x7B);  // v128
        body.U32(locals_end - kLocalBase);
        body.Byte(u8(ValType::I64));
        body.Append(code);
        body.Byte(Insn::End);

        Writer bodies;
        bodies.U32(1);
        bodies.Sized(body);
        Section(module, 10, bodies);

        out = std::move(module.bytes);
    }

    static void Section(Writer& module, u8 id, const Writer& content) {
        module.Byte(id);
        module.Sized(content);
    }

    const std::vector<CompiledBlock*>& blocks;
    const JitTarget& target;
    std::unordered_map<u64, u32> index_of;
    Code code;
    u32 loop_nest = 0;
    u32 exit_nest = 0;
    u32 current_index = 0;
    std::optional<RegionLocals> cached;
};

}  // Anonymous namespace

namespace {
std::atomic<u64> g_compile_count{0};
std::atomic<u64> g_compile_us{0};
std::atomic<u64> g_instantiate_count{0};
std::atomic<u64> g_instantiate_us{0};

u64 NowUs() {
    return u64(emscripten_get_now() * 1000.0);
}
}  // Anonymous namespace

std::string JitStats() {
    return fmt::format("compile {}/{}ms instantiate {}/{}ms", g_compile_count.exchange(0), g_compile_us.exchange(0) / 1000,
                       g_instantiate_count.exchange(0), g_instantiate_us.exchange(0) / 1000);
}

bool CompileRegion(const std::vector<CompiledBlock*>& blocks, const JitTarget& target, std::vector<u8>& out) {
    const u64 start = NowUs();
    RegionCompiler compiler{blocks, target};
    const bool ok = compiler.Compile(out);
    g_compile_count.fetch_add(1, std::memory_order_relaxed);
    g_compile_us.fetch_add(NowUs() - start, std::memory_order_relaxed);
    return ok;
}

CompiledBlockFn GetThreadFunction(WasmRegion& region) {
    thread_local const u32 thread_tag = [] {
        static std::atomic<u32> next_tag{1};
        return next_tag.fetch_add(1);
    }();
    if (region.last_thread.load(std::memory_order_relaxed) == thread_tag) {
        return region.last_function.load(std::memory_order_relaxed);
    }

    thread_local std::unordered_map<u64, CompiledBlockFn> functions;
    CompiledBlockFn fn;
    if (const auto it = functions.find(region.id); it != functions.end()) {
        fn = it->second;
    } else {
        const u64 start = NowUs();
        const u64 index = dynarmic_wasm_instantiate(region.wasm.data(), region.wasm.size());
        g_instantiate_count.fetch_add(1, std::memory_order_relaxed);
        g_instantiate_us.fetch_add(NowUs() - start, std::memory_order_relaxed);
        fn = index == 0 ? nullptr : reinterpret_cast<CompiledBlockFn>(index);
        functions.emplace(region.id, fn);
    }
    region.last_function.store(fn, std::memory_order_relaxed);
    region.last_thread.store(thread_tag, std::memory_order_relaxed);
    return fn;
}

u32 RegionThreshold() {
    static const u32 threshold = [] {
        const char* value = std::getenv("EDEN_JIT_THRESHOLD");
        return value ? std::max(1, std::atoi(value)) : 8;
    }();
    return u32(threshold);
}

bool ForcedHelperOp(IR::Opcode opcode) {
    static const std::pair<u32, u32> range = [] {
        const char* value = std::getenv("EDEN_JIT_HELPER_OPS");
        if (!value) {
            return std::pair<u32, u32>{0, 0};
        }
        char* end = nullptr;
        const u32 lo = u32(std::strtoul(value, &end, 10));
        const u32 hi = *end == '-' ? u32(std::strtoul(end + 1, nullptr, 10)) : lo + 1;
        return std::pair<u32, u32>{lo, hi};
    }();
    return u32(opcode) >= range.first && u32(opcode) < range.second;
}

bool RegionAllowedAt(u64 pc) {
    static const std::pair<u64, u64> range = [] {
        const char* value = std::getenv("EDEN_JIT_PC_RANGE");
        if (!value) {
            return std::pair<u64, u64>{0, ~u64{0}};
        }
        char* end = nullptr;
        const u64 lo = std::strtoull(value, &end, 16);
        const u64 hi = *end == '-' ? std::strtoull(end + 1, nullptr, 16) : ~u64{0};
        return std::pair<u64, u64>{lo, hi};
    }();
    return pc >= range.first && pc < range.second;
}

bool WasmJitEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_INTERP_JIT");
        return !(value && value[0] == '0');
    }();
    return enabled;
}

}  // namespace Dynarmic::Backend::Interp
