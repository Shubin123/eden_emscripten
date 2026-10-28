// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/frontend/A64/translate/impl/impl.h"

#include "dynarmic/mcl/bit.hpp"

#include "dynarmic/ir/terminal.h"

namespace Dynarmic::A64 {

bool TranslatorVisitor::UnpredictableInstruction() {
    return RaiseException(Exception::UnpredictableInstruction);
}

bool TranslatorVisitor::DecodeError() {
    UNREACHABLE();
}

bool TranslatorVisitor::ReservedValue() {
    return RaiseException(Exception::ReservedValue);
}

bool TranslatorVisitor::UnallocatedEncoding() {
    return RaiseException(Exception::UnallocatedEncoding);
}

bool TranslatorVisitor::RaiseException(Exception exception) {
    ir.SetPC(ir.Imm64(ir.current_location->PC() + 4));
    ir.ExceptionRaised(exception);
    ir.SetTerm(IR::Term::CheckHalt{IR::Term::ReturnToDispatch{}});
    return false;
}

std::optional<TranslatorVisitor::BitMasks> TranslatorVisitor::DecodeBitMasks(bool immN, Imm<6> imms, Imm<6> immr, bool immediate) {
    const int len = mcl::bit::highest_set_bit((immN ? 1 << 6 : 0) | (imms.ZeroExtend() ^ 0b111111));
    if (len < 1) {
        return std::nullopt;
    }

    const size_t levels = mcl::bit::ones<size_t>(len);
    if (immediate && (imms.ZeroExtend() & levels) == levels) {
        return std::nullopt;
    }

    const s32 S = s32(imms.ZeroExtend() & levels);
    const s32 R = s32(immr.ZeroExtend() & levels);
    const u64 d = u64(S - R) & levels;

    const size_t esize = size_t{1} << len;
    const u64 welem = mcl::bit::ones<u64>(S + 1);
    const u64 telem = mcl::bit::ones<u64>(d + 1);
    const u64 wmask = std::rotr(mcl::bit::replicate_element<u64>(esize, welem), R);
    const u64 tmask = mcl::bit::replicate_element<u64>(esize, telem);

    return BitMasks{wmask, tmask};
}

IR::UAny TranslatorVisitor::I(size_t bitsize, u64 value) {
    switch (bitsize) {
    case 8:
        return ir.Imm8(static_cast<u8>(value));
    case 16:
        return ir.Imm16(static_cast<u16>(value));
    case 32:
        return ir.Imm32(static_cast<u32>(value));
    case 64:
        return ir.Imm64(value);
    }
    UNREACHABLE();
}

IR::UAny TranslatorVisitor::X(size_t bitsize, Reg reg) {
    switch (bitsize) {
    case 8:
        return ir.LeastSignificantByte(ir.GetW(reg));
    case 16:
        return ir.LeastSignificantHalf(ir.GetW(reg));
    case 32:
        return ir.GetW(reg);
    case 64:
        return ir.GetX(reg);
    }
    UNREACHABLE();
}

void TranslatorVisitor::X(size_t bitsize, Reg reg, IR::U32U64 value) {
    switch (bitsize) {
    case 32:
        ir.SetW(reg, value);
        return;
    case 64:
        ir.SetX(reg, value);
        return;
    }
    UNREACHABLE();
}

IR::U32U64 TranslatorVisitor::SP(size_t bitsize) {
    switch (bitsize) {
    case 32:
        return ir.LeastSignificantWord(ir.GetSP());
    case 64:
        return ir.GetSP();
    }
    UNREACHABLE();
}

void TranslatorVisitor::SP(size_t bitsize, IR::U32U64 value) {
    switch (bitsize) {
    case 32:
        ir.SetSP(ir.ZeroExtendWordToLong(value));
        break;
    case 64:
        ir.SetSP(value);
        break;
    default:
        UNREACHABLE();
    }
}

IR::U128 TranslatorVisitor::V(size_t bitsize, Vec vec) {
    switch (bitsize) {
    case 32:
        return ir.GetS(vec);
    case 64:
        return ir.GetD(vec);
    case 128:
        return ir.GetQ(vec);
    }
    UNREACHABLE();
}

void TranslatorVisitor::V(size_t bitsize, Vec vec, IR::U128 value) {
    switch (bitsize) {
    case 32:
        ir.SetS(vec, value);
        return;
    case 64:
        // TODO: Remove VectorZeroUpper when possible.
        ir.SetD(vec, ir.VectorZeroUpper(value));
        return;
    case 128:
        ir.SetQ(vec, value);
        return;
    }
    UNREACHABLE();
}

IR::UAnyU128 TranslatorVisitor::V_scalar(size_t bitsize, Vec vec) {
    if (bitsize == 128) {
        return V(128, vec);
    }
    // TODO: Optimize
    return ir.VectorGetElement(bitsize, ir.GetQ(vec), 0);
}

void TranslatorVisitor::V_scalar(size_t bitsize, Vec vec, IR::UAnyU128 value) {
    if (bitsize == 128) {
        V(128, vec, value);
        return;
    }
    // TODO: Optimize
    ir.SetQ(vec, ir.ZeroExtendToQuad(value));
}

IR::U128 TranslatorVisitor::Vpart(size_t bitsize, Vec vec, size_t part) {
    ASSERT(part == 0 || part == 1);
    ASSERT(bitsize == 64);
    if (part == 0) {
        return V(64, vec);
    }
    return ir.ZeroExtendToQuad(ir.VectorGetElement(bitsize, V(128, vec), part));
}

void TranslatorVisitor::Vpart(size_t bitsize, Vec vec, size_t part, IR::U128 value) {
    ASSERT(part == 0 || part == 1);
    if (part == 0) {
        ASSERT(bitsize == 64);
        V(128, vec, ir.VectorZeroExtend(bitsize, value));
    } else {
        ASSERT(bitsize == 64);
        V(128, vec, ir.VectorInterleaveLower(64, V(128, vec), value));
    }
}

IR::UAny TranslatorVisitor::Vpart_scalar(size_t bitsize, Vec vec, size_t part) {
    ASSERT(part == 0 || part == 1);
    if (part == 0) {
        ASSERT(bitsize == 8 || bitsize == 16 || bitsize == 32 || bitsize == 64);
    } else {
        ASSERT(bitsize == 64);
    }
    return ir.VectorGetElement(bitsize, V(128, vec), part);
}

void TranslatorVisitor::Vpart_scalar(size_t bitsize, Vec vec, size_t part, IR::UAny value) {
    ASSERT(part == 0 || part == 1);
    if (part == 0) {
        ASSERT(bitsize == 8 || bitsize == 16 || bitsize == 32 || bitsize == 64);
        V(128, vec, ir.ZeroExtendToQuad(value));
    } else {
        ASSERT(bitsize == 64);
        V(128, vec, ir.VectorSetElement(64, V(128, vec), 1, value));
    }
}

IR::UAnyU128 TranslatorVisitor::Mem(IR::U64 address, size_t bytesize, IR::AccType acc_type) {
    switch (bytesize) {
    case 1:
        return ir.ReadMemory8(address, acc_type);
    case 2:
        return ir.ReadMemory16(address, acc_type);
    case 4:
        return ir.ReadMemory32(address, acc_type);
    case 8:
        return ir.ReadMemory64(address, acc_type);
    case 16:
        return ir.ReadMemory128(address, acc_type);
    }
    UNREACHABLE();
}

void TranslatorVisitor::Mem(IR::U64 address, size_t bytesize, IR::AccType acc_type, IR::UAnyU128 value) {
    switch (bytesize) {
    case 1:
        ir.WriteMemory8(address, value, acc_type);
        return;
    case 2:
        ir.WriteMemory16(address, value, acc_type);
        return;
    case 4:
        ir.WriteMemory32(address, value, acc_type);
        return;
    case 8:
        ir.WriteMemory64(address, value, acc_type);
        return;
    case 16:
        ir.WriteMemory128(address, value, acc_type);
        return;
    }
    UNREACHABLE();
}

IR::UAnyU128 TranslatorVisitor::ExclusiveMem(IR::U64 address, size_t bytesize, IR::AccType acc_type) {
    switch (bytesize) {
    case 1:
        return ir.ExclusiveReadMemory8(address, acc_type);
    case 2:
        return ir.ExclusiveReadMemory16(address, acc_type);
    case 4:
        return ir.ExclusiveReadMemory32(address, acc_type);
    case 8:
        return ir.ExclusiveReadMemory64(address, acc_type);
    case 16:
        return ir.ExclusiveReadMemory128(address, acc_type);
    }
    UNREACHABLE();
}

IR::U32 TranslatorVisitor::ExclusiveMem(IR::U64 address, size_t bytesize, IR::AccType acc_type, IR::UAnyU128 value) {
    switch (bytesize) {
    case 1:
        return ir.ExclusiveWriteMemory8(address, value, acc_type);
    case 2:
        return ir.ExclusiveWriteMemory16(address, value, acc_type);
    case 4:
        return ir.ExclusiveWriteMemory32(address, value, acc_type);
    case 8:
        return ir.ExclusiveWriteMemory64(address, value, acc_type);
    case 16:
        return ir.ExclusiveWriteMemory128(address, value, acc_type);
    }
    UNREACHABLE();
}

IR::U32U64 TranslatorVisitor::SignExtend(IR::UAny value, size_t to_size) {
    switch (to_size) {
    case 32:
        return ir.SignExtendToWord(value);
    case 64:
        return ir.SignExtendToLong(value);
    }
    UNREACHABLE();
}

namespace {

/// All ones when `value` is nonzero.
IR::U32 HalfMaskNonzero(IR::IREmitter& ir, const IR::U32& value) {
    return IR::U32{ir.ArithmeticShiftRight(ir.Or(value, ir.Sub(ir.Imm32(0), value)), ir.Imm8(31))};
}

/// `a` where `mask` is set, else `b`.
IR::U32 HalfSelect(IR::IREmitter& ir, const IR::U32& mask, const IR::U32& a, const IR::U32& b) {
    return IR::U32{ir.Or(ir.And(a, mask), ir.AndNot(b, mask))};
}

/// Half bits with a denormal replaced by zero of the same sign.
IR::U32 HalfFlush(IR::IREmitter& ir, const IR::U32& bits) {
    const IR::U32 normal = HalfMaskNonzero(ir, IR::U32{ir.And(bits, ir.Imm32(0x7C00))});
    return HalfSelect(ir, normal, bits, IR::U32{ir.And(bits, ir.Imm32(0x8000))});
}

/// All ones when the half bits are a NaN.
IR::U32 HalfNaNMask(IR::IREmitter& ir, const IR::U32& bits) {
    return IR::U32{ir.ArithmeticShiftRight(ir.Sub(ir.Imm32(0x7C00), ir.And(bits, ir.Imm32(0x7FFF))), ir.Imm8(31))};
}

}  // Anonymous namespace

bool TranslatorVisitor::HalfArithmeticSupported() {
    // The result returns to half through a conversion, which alternative half precision changes.
    return !ir.current_location->FPCR().AHP();
}

IR::U32 TranslatorVisitor::HalfOperand(const IR::U16& value) {
    IR::U32 bits = ir.ZeroExtendHalfToWord(value);
    if (ir.current_location->FPCR().FZ16()) {
        bits = HalfFlush(ir, bits);
    }
    return ir.FPHalfToSingle(ir.LeastSignificantHalf(bits), ir.current_location->FPCR().RMode());
}

IR::U16 TranslatorVisitor::HalfResult(const IR::U32& value) {
    const IR::U16 half = ir.FPSingleToHalf(value, ir.current_location->FPCR().RMode());
    if (!ir.current_location->FPCR().FZ16()) {
        return half;
    }
    return ir.LeastSignificantHalf(HalfFlush(ir, ir.ZeroExtendHalfToWord(half)));
}

IR::U16 TranslatorVisitor::HalfNaNs(const IR::U16& a16, const IR::U16& b16, const IR::U16& result16, bool numeric) {
    // Converting to single quiets a signalling NaN, which would lose its priority over a quiet one:
    // pick the NaN from the half operands. FMAXNM and FMINNM (numeric) still treat a single quiet
    // NaN as missing data, which the single-precision operation already did.
    const IR::U32 a = ir.ZeroExtendHalfToWord(a16);
    const IR::U32 b = ir.ZeroExtendHalfToWord(b16);
    const IR::U32 nan_a = HalfNaNMask(ir, a);
    const IR::U32 nan_b = HalfNaNMask(ir, b);
    const IR::U32 snan_a = IR::U32{ir.AndNot(nan_a, HalfMaskNonzero(ir, IR::U32{ir.And(a, ir.Imm32(0x200))}))};
    const IR::U32 snan_b = IR::U32{ir.AndNot(nan_b, HalfMaskNonzero(ir, IR::U32{ir.And(b, ir.Imm32(0x200))}))};
    IR::U32 nan = HalfSelect(ir, snan_a, IR::U32{ir.Or(a, ir.Imm32(0x200))},
                             HalfSelect(ir, snan_b, IR::U32{ir.Or(b, ir.Imm32(0x200))}, HalfSelect(ir, nan_a, a, b)));
    if (ir.current_location->FPCR().DN()) {
        nan = ir.Imm32(0x7E00);
    }
    const IR::U32 use_nan = numeric ? IR::U32{ir.Or(ir.Or(snan_a, snan_b), ir.And(nan_a, nan_b))} : IR::U32{ir.Or(nan_a, nan_b)};
    return ir.LeastSignificantHalf(HalfSelect(ir, use_nan, nan, ir.ZeroExtendHalfToWord(result16)));
}

IR::U32U64 TranslatorVisitor::ZeroExtend(IR::UAny value, size_t to_size) {
    switch (to_size) {
    case 32:
        return ir.ZeroExtendToWord(value);
    case 64:
        return ir.ZeroExtendToLong(value);
    }
    UNREACHABLE();
}

IR::U32U64 TranslatorVisitor::ShiftReg(size_t bitsize, Reg reg, Imm<2> shift, IR::U8 amount) {
    IR::U32U64 result = X(bitsize, reg);
    switch (shift.ZeroExtend()) {
    case 0b00:
        return ir.LogicalShiftLeft(result, amount);
    case 0b01:
        return ir.LogicalShiftRight(result, amount);
    case 0b10:
        return ir.ArithmeticShiftRight(result, amount);
    case 0b11:
        return ir.RotateRight(result, amount);
    }
    UNREACHABLE();
}

IR::U32U64 TranslatorVisitor::ExtendReg(size_t bitsize, Reg reg, Imm<3> option, u8 shift) {
    ASSERT(shift <= 4);
    ASSERT(bitsize == 32 || bitsize == 64);
    IR::UAny val = X(bitsize, reg);
    size_t len;
    IR::U32U64 extended;
    bool signed_extend;

    switch (option.ZeroExtend()) {
    case 0b000: {  // UXTB
        val = ir.LeastSignificantByte(val);
        len = 8;
        signed_extend = false;
        break;
    }
    case 0b001: {  // UXTH
        val = ir.LeastSignificantHalf(val);
        len = 16;
        signed_extend = false;
        break;
    }
    case 0b010: {  // UXTW
        if (bitsize != 32) {
            val = ir.LeastSignificantWord(val);
        }
        len = 32;
        signed_extend = false;
        break;
    }
    case 0b011: {  // UXTX
        len = 64;
        signed_extend = false;
        break;
    }
    case 0b100: {  // SXTB
        val = ir.LeastSignificantByte(val);
        len = 8;
        signed_extend = true;
        break;
    }
    case 0b101: {  // SXTH
        val = ir.LeastSignificantHalf(val);
        len = 16;
        signed_extend = true;
        break;
    }
    case 0b110: {  // SXTW
        if (bitsize != 32) {
            val = ir.LeastSignificantWord(val);
        }
        len = 32;
        signed_extend = true;
        break;
    }
    case 0b111: {  // SXTX
        len = 64;
        signed_extend = true;
        break;
    }
    default:
        UNREACHABLE();
    }

    if (len < bitsize) {
        if (bitsize == 32) {
            extended = signed_extend ? ir.SignExtendToWord(val) : ir.ZeroExtendToWord(val);
        } else {
            extended = signed_extend ? ir.SignExtendToLong(val) : ir.ZeroExtendToLong(val);
        }
    } else {
        extended = val;
    }

    return ir.LogicalShiftLeft(extended, ir.Imm8(shift));
}

}  // namespace Dynarmic::A64
