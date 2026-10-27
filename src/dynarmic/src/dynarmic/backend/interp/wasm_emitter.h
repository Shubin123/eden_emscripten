// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Minimal WebAssembly binary writer used by the interpreter's block compiler. Generated modules
// import the host's shared Memory64 linear memory and its 64-bit function table.

#pragma once

#include <cstring>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Dynarmic::Backend::Interp::Wasm {

/// Values of the metadata.code.branch_hint section.
enum class BranchHint : u8 {
    Unlikely = 0,
    Likely = 1,
    None = 2,
};

enum class ValType : u8 {
    I32 = 0x7F,
    I64 = 0x7E,
};

namespace Insn {
constexpr u8 Unreachable = 0x00;
constexpr u8 Block = 0x02;
constexpr u8 If = 0x04;
constexpr u8 Else = 0x05;
constexpr u8 End = 0x0B;
constexpr u8 Br = 0x0C;
constexpr u8 BrIf = 0x0D;
constexpr u8 Return = 0x0F;
constexpr u8 CallIndirect = 0x11;
constexpr u8 Drop = 0x1A;
constexpr u8 Select = 0x1B;
constexpr u8 LocalGet = 0x20;
constexpr u8 LocalSet = 0x21;
constexpr u8 LocalTee = 0x22;
constexpr u8 I32Load = 0x28;
constexpr u8 I64Load = 0x29;
constexpr u8 I32Load8U = 0x2D;
constexpr u8 I64Load8U = 0x31;
constexpr u8 I64Load16U = 0x33;
constexpr u8 I64Load32U = 0x35;
constexpr u8 I32Store = 0x36;
constexpr u8 I64Store = 0x37;
constexpr u8 I32Store8 = 0x3A;
constexpr u8 I64Store8 = 0x3C;
constexpr u8 I64Store16 = 0x3D;
constexpr u8 I64Store32 = 0x3E;
constexpr u8 I32Const = 0x41;
constexpr u8 I64Const = 0x42;
constexpr u8 I32Eqz = 0x45;
constexpr u8 I32Eq = 0x46;
constexpr u8 I32Ne = 0x47;
constexpr u8 I32LtS = 0x48;
constexpr u8 I32LtU = 0x49;
constexpr u8 I32GtS = 0x4A;
constexpr u8 I32GtU = 0x4B;
constexpr u8 I32LeS = 0x4C;
constexpr u8 I32LeU = 0x4D;
constexpr u8 I32GeS = 0x4E;
constexpr u8 I32GeU = 0x4F;
constexpr u8 I64Eqz = 0x50;
constexpr u8 I64Eq = 0x51;
constexpr u8 I64Ne = 0x52;
constexpr u8 I64LtS = 0x53;
constexpr u8 I64LtU = 0x54;
constexpr u8 I64GtS = 0x55;
constexpr u8 I64GtU = 0x56;
constexpr u8 I64LeS = 0x57;
constexpr u8 I64LeU = 0x58;
constexpr u8 I64GeS = 0x59;
constexpr u8 I64GeU = 0x5A;
constexpr u8 I32And = 0x71;
constexpr u8 I32Or = 0x72;
constexpr u8 I32Xor = 0x73;
constexpr u8 I64Clz = 0x79;
constexpr u8 I64Add = 0x7C;
constexpr u8 I64Sub = 0x7D;
constexpr u8 I64Mul = 0x7E;
constexpr u8 I64DivS = 0x7F;
constexpr u8 I64DivU = 0x80;
constexpr u8 I64And = 0x83;
constexpr u8 I64Or = 0x84;
constexpr u8 I64Xor = 0x85;
constexpr u8 I64Shl = 0x86;
constexpr u8 I64ShrS = 0x87;
constexpr u8 I64ShrU = 0x88;
constexpr u8 I64Rotr = 0x8A;
constexpr u8 I32WrapI64 = 0xA7;
constexpr u8 I64ExtendI32S = 0xAC;
constexpr u8 I64ExtendI32U = 0xAD;
constexpr u8 I64Extend8S = 0xC2;
constexpr u8 I64Extend16S = 0xC3;
constexpr u8 I64Extend32S = 0xC4;
}  // namespace Insn

class Writer {
public:
    void Byte(u8 value) {
        bytes.push_back(value);
    }

    void U32(u32 value) {
        do {
            u8 byte = value & 0x7F;
            value >>= 7;
            if (value != 0) {
                byte |= 0x80;
            }
            bytes.push_back(byte);
        } while (value != 0);
    }

    void U64(u64 value) {
        do {
            u8 byte = value & 0x7F;
            value >>= 7;
            if (value != 0) {
                byte |= 0x80;
            }
            bytes.push_back(byte);
        } while (value != 0);
    }

    void S64(s64 value) {
        bool more = true;
        while (more) {
            u8 byte = value & 0x7F;
            value >>= 7;
            if ((value == 0 && !(byte & 0x40)) || (value == -1 && (byte & 0x40))) {
                more = false;
            } else {
                byte |= 0x80;
            }
            bytes.push_back(byte);
        }
    }

    void Name(const char* name) {
        const size_t length = std::strlen(name);
        U32(u32(length));
        bytes.insert(bytes.end(), name, name + length);
    }

    void Append(const Writer& other) {
        bytes.insert(bytes.end(), other.bytes.begin(), other.bytes.end());
    }

    /// Appends `content` prefixed with its byte length.
    void Sized(const Writer& content) {
        U32(u32(content.bytes.size()));
        Append(content);
    }

    std::vector<u8> bytes;
};

/// Function body builder with helpers for the instructions the block compiler emits.
class Code : public Writer {
public:
    void I64Const(u64 value) {
        Byte(Insn::I64Const);
        S64(s64(value));
    }
    void I32Const(u32 value) {
        Byte(Insn::I32Const);
        S64(s64(s32(value)));
    }
    void Get(u32 local) {
        Byte(Insn::LocalGet);
        U32(local);
    }
    void Set(u32 local) {
        Byte(Insn::LocalSet);
        U32(local);
    }
    void Tee(u32 local) {
        Byte(Insn::LocalTee);
        U32(local);
    }
    /// Memory access with an i64 address on the stack; log2 alignment hint is 0 (unaligned ok).
    void Mem(u8 opcode, u64 offset) {
        Byte(opcode);
        U32(0);
        U64(offset);
    }
    /// `unlikely`/`likely` say whether the then-branch (or a taken br_if) is rare or common, which
    /// V8 uses to lay rare code out of line (WebAssembly branch hinting).
    void If(BranchHint hint = BranchHint::None) {
        Hint(hint);
        Byte(Insn::If);
        Byte(0x40);
        ++nest;
    }
    void IfI64(BranchHint hint = BranchHint::None) {
        Hint(hint);
        Byte(Insn::If);
        Byte(u8(ValType::I64));
        ++nest;
    }
    /// Branches `depth` constructs out when the i32 on the stack is non-zero.
    void BrIf(u32 depth, BranchHint hint = BranchHint::None) {
        Hint(hint);
        Byte(Insn::BrIf);
        U32(depth);
    }
    void Else() {
        Byte(Insn::Else);
    }
    void BlockStart() {
        Byte(Insn::Block);
        Byte(0x40);
        ++nest;
    }
    void LoopStart() {
        Byte(0x03);
        Byte(0x40);
        ++nest;
    }
    void End() {
        Byte(Insn::End);
        --nest;
    }
    /// Branches to the construct that was innermost when `target_nest` constructs were open.
    void BrTo(u32 target_nest) {
        Byte(Insn::Br);
        U32(nest - target_nest);
    }

    /// Number of currently open structured control constructs.
    u32 nest = 0;

    /// Branch hints recorded so far: offset of the instruction in this code, and the hint.
    std::vector<std::pair<u32, BranchHint>> hints;

private:
    void Hint(BranchHint hint) {
        if (hint != BranchHint::None) {
            hints.emplace_back(u32(bytes.size()), hint);
        }
    }
};

}  // namespace Dynarmic::Backend::Interp::Wasm
