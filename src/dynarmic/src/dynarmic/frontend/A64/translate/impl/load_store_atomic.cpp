// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Large System Extension atomics (Armv8.1: CAS, CASP, LD<op>, SWP) and LDAPR (RCpc, Armv8.3),
// which code built for Armv8.2 cores such as the Switch 2's Cortex-A78C uses.
//
// Each read-modify-write is an exclusive load and an exclusive store of the result. The block
// ends after the instruction: the next instruction runs when the store succeeded, and the same
// instruction runs again when another access got in between, so the operation takes effect once
// and atomically. Register writes only land when the store succeeded, so a retry reads the
// instruction's operands unchanged.

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

namespace {

enum class LseOp {
    Add,
    Clear,
    Eor,
    Set,
    SignedMax,
    SignedMin,
    UnsignedMax,
    UnsignedMin,
    Swap,
};

IR::U64 LseAddress(TranslatorVisitor& v, Reg Rn) {
    return Rn == Reg::SP ? IR::U64{v.SP(64)} : IR::U64{v.X(64, Rn)};
}

IR::AccType LseAccType(bool acquire, bool release) {
    return acquire || release ? IR::AccType::ORDERED : IR::AccType::ATOMIC;
}

/// The low `datasize` bits of `value`, as the type an exclusive store of that size takes.
IR::UAny LseTruncate(TranslatorVisitor& v, IR::U32U64 value, size_t datasize) {
    switch (datasize) {
    case 8:
        return v.ir.LeastSignificantByte(value);
    case 16:
        return v.ir.LeastSignificantHalf(value);
    case 32:
        return value.GetType() == IR::Type::U64 ? IR::UAny{v.ir.LeastSignificantWord(IR::U64{value})} : IR::UAny{value};
    default:
        return value;
    }
}

/// All ones when `value` is nonzero, zero otherwise.
IR::U32U64 LseNonzeroMask(TranslatorVisitor& v, IR::U32U64 value) {
    const bool is64 = value.GetType() == IR::Type::U64;
    const IR::U32U64 zero = is64 ? IR::U32U64{v.ir.Imm64(0)} : IR::U32U64{v.ir.Imm32(0)};
    const IR::U32U64 top = v.ir.Or(value, v.ir.Sub(zero, value));
    return v.ir.ArithmeticShiftRight(top, v.ir.Imm8(is64 ? 63 : 31));
}

/// `value` when the exclusive store succeeded (status 0), else `current`.
IR::U32U64 LseOnStored(TranslatorVisitor& v, const IR::U32& status, IR::U32U64 value, IR::U32U64 current) {
    const IR::U32U64 failed = value.GetType() == IR::Type::U64 ? IR::U32U64{v.ir.ZeroExtendWordToLong(status)} : IR::U32U64{status};
    const IR::U32U64 mask = LseNonzeroMask(v, failed);
    return v.ir.Or(v.ir.AndNot(value, mask), v.ir.And(current, mask));
}

void LseSetOnStored(TranslatorVisitor& v, const IR::U32& status, size_t regsize, Reg reg, IR::U32U64 value) {
    if (reg != Reg::ZR) {
        v.X(regsize, reg, LseOnStored(v, status, value, v.X(regsize, reg)));
    }
}

/// Ends the block: the next instruction when `status` is 0, else this instruction again.
bool LseRetryUnlessStored(TranslatorVisitor& v, const IR::U32& status) {
    v.ir.SetCheckBit(v.ir.IsZero(status));
    const auto next = IR::Term::LinkBlock{v.ir.current_location->AdvancePC(4)};
    const auto again = IR::Term::LinkBlock{*v.ir.current_location};
    v.ir.SetTerm(IR::Term::CheckBit{next, again});
    return false;
}

bool LseMemoryOperation(TranslatorVisitor& v, Imm<2> sz, bool A, bool R, Reg Rs, LseOp op, Reg Rn, Reg Rt) {
    const size_t datasize = 8 << sz.ZeroExtend<size_t>();
    const size_t regsize = datasize == 64 ? 64 : 32;
    const auto acctype = LseAccType(A, R);
    const IR::U64 address = LseAddress(v, Rn);

    const IR::U32U64 old = v.ZeroExtend(v.ExclusiveMem(address, datasize / 8, acctype), regsize);
    const IR::U32U64 operand = v.X(regsize, Rs);
    const auto sign_extended = [&](IR::U32U64 value) -> IR::U32U64 {
        return datasize < 32 ? v.SignExtend(LseTruncate(v, value, datasize), regsize) : value;
    };
    const auto zero_extended = [&](IR::U32U64 value) -> IR::U32U64 {
        return datasize < 32 ? v.ZeroExtend(LseTruncate(v, value, datasize), regsize) : value;
    };

    IR::U32U64 result;
    switch (op) {
    case LseOp::Add:
        result = v.ir.Add(old, operand);
        break;
    case LseOp::Clear:
        result = v.ir.AndNot(old, operand);
        break;
    case LseOp::Eor:
        result = v.ir.Eor(old, operand);
        break;
    case LseOp::Set:
        result = v.ir.Or(old, operand);
        break;
    case LseOp::SignedMax:
        result = v.ir.MaxSigned(sign_extended(old), sign_extended(operand));
        break;
    case LseOp::SignedMin:
        result = v.ir.MinSigned(sign_extended(old), sign_extended(operand));
        break;
    case LseOp::UnsignedMax:
        result = v.ir.MaxUnsigned(old, zero_extended(operand));
        break;
    case LseOp::UnsignedMin:
        result = v.ir.MinUnsigned(old, zero_extended(operand));
        break;
    case LseOp::Swap:
        result = operand;
        break;
    }

    const IR::U32 status = v.ExclusiveMem(address, datasize / 8, acctype, LseTruncate(v, result, datasize));
    LseSetOnStored(v, status, regsize, Rt, old);
    return LseRetryUnlessStored(v, status);
}

}  // Anonymous namespace

bool TranslatorVisitor::CAS(Imm<2> sz, bool L, Reg Rs, bool o0, Reg Rn, Reg Rt) {
    const size_t datasize = 8 << sz.ZeroExtend<size_t>();
    const size_t regsize = datasize == 64 ? 64 : 32;
    const auto acctype = LseAccType(L, o0);
    const IR::U64 address = LseAddress(*this, Rn);

    const IR::U32U64 old = ZeroExtend(ExclusiveMem(address, datasize / 8, acctype), regsize);
    IR::U32U64 expected = X(regsize, Rs);
    if (datasize < 32) {
        expected = ZeroExtend(LseTruncate(*this, expected, datasize), regsize);
    }
    // A failed comparison stores the value it read, so the operation is still one atomic access.
    const IR::U32U64 differs = LseNonzeroMask(*this, ir.Eor(old, expected));
    const IR::U32U64 stored = ir.Or(ir.AndNot(X(regsize, Rt), differs), ir.And(old, differs));

    const IR::U32 status = ExclusiveMem(address, datasize / 8, acctype, LseTruncate(*this, stored, datasize));
    LseSetOnStored(*this, status, regsize, Rs, old);
    return LseRetryUnlessStored(*this, status);
}

bool TranslatorVisitor::CASP(bool sz, bool L, Reg Rs, bool o0, Reg Rn, Reg Rt) {
    if (RegNumber(Rs) % 2 != 0 || RegNumber(Rt) % 2 != 0) {
        return UnpredictableInstruction();
    }
    const auto acctype = LseAccType(L, o0);
    const IR::U64 address = LseAddress(*this, Rn);

    if (!sz) {
        // A pair of words, as one doubleword with Rs (and Rt) in the lower half.
        const IR::U64 old = IR::U64{ExclusiveMem(address, 8, acctype)};
        const IR::U64 expected = ir.Pack2x32To1x64(X(32, Rs), X(32, Rs + 1));
        const IR::U64 desired = ir.Pack2x32To1x64(X(32, Rt), X(32, Rt + 1));
        const IR::U64 differs = IR::U64{LseNonzeroMask(*this, ir.Eor(old, expected))};
        const IR::U64 stored = IR::U64{ir.Or(ir.AndNot(desired, differs), ir.And(old, differs))};
        const IR::U32 status = ExclusiveMem(address, 8, acctype, stored);
        LseSetOnStored(*this, status, 32, Rs, ir.LeastSignificantWord(old));
        LseSetOnStored(*this, status, 32, Rs + 1, ir.MostSignificantWord(old).result);
        return LseRetryUnlessStored(*this, status);
    }

    const IR::U128 old = IR::U128{ExclusiveMem(address, 16, acctype)};
    const IR::U64 old_lo = IR::U64{ir.VectorGetElement(64, old, 0)};
    const IR::U64 old_hi = IR::U64{ir.VectorGetElement(64, old, 1)};
    const IR::U64 differs = IR::U64{LseNonzeroMask(*this, ir.Or(ir.Eor(old_lo, X(64, Rs)), ir.Eor(old_hi, X(64, Rs + 1))))};
    const IR::U64 stored_lo = IR::U64{ir.Or(ir.AndNot(X(64, Rt), differs), ir.And(old_lo, differs))};
    const IR::U64 stored_hi = IR::U64{ir.Or(ir.AndNot(X(64, Rt + 1), differs), ir.And(old_hi, differs))};
    const IR::U32 status = ExclusiveMem(address, 16, acctype, ir.Pack2x64To1x128(stored_lo, stored_hi));
    LseSetOnStored(*this, status, 64, Rs, old_lo);
    LseSetOnStored(*this, status, 64, Rs + 1, old_hi);
    return LseRetryUnlessStored(*this, status);
}

bool TranslatorVisitor::LDADD(Imm<2> sz, bool A, bool R, Reg Rs, Reg Rn, Reg Rt) {
    return LseMemoryOperation(*this, sz, A, R, Rs, LseOp::Add, Rn, Rt);
}

bool TranslatorVisitor::LDCLR(Imm<2> sz, bool A, bool R, Reg Rs, Reg Rn, Reg Rt) {
    return LseMemoryOperation(*this, sz, A, R, Rs, LseOp::Clear, Rn, Rt);
}

bool TranslatorVisitor::LDEOR(Imm<2> sz, bool A, bool R, Reg Rs, Reg Rn, Reg Rt) {
    return LseMemoryOperation(*this, sz, A, R, Rs, LseOp::Eor, Rn, Rt);
}

bool TranslatorVisitor::LDSET(Imm<2> sz, bool A, bool R, Reg Rs, Reg Rn, Reg Rt) {
    return LseMemoryOperation(*this, sz, A, R, Rs, LseOp::Set, Rn, Rt);
}

bool TranslatorVisitor::LDSMAX(Imm<2> sz, bool A, bool R, Reg Rs, Reg Rn, Reg Rt) {
    return LseMemoryOperation(*this, sz, A, R, Rs, LseOp::SignedMax, Rn, Rt);
}

bool TranslatorVisitor::LDSMIN(Imm<2> sz, bool A, bool R, Reg Rs, Reg Rn, Reg Rt) {
    return LseMemoryOperation(*this, sz, A, R, Rs, LseOp::SignedMin, Rn, Rt);
}

bool TranslatorVisitor::LDUMAX(Imm<2> sz, bool A, bool R, Reg Rs, Reg Rn, Reg Rt) {
    return LseMemoryOperation(*this, sz, A, R, Rs, LseOp::UnsignedMax, Rn, Rt);
}

bool TranslatorVisitor::LDUMIN(Imm<2> sz, bool A, bool R, Reg Rs, Reg Rn, Reg Rt) {
    return LseMemoryOperation(*this, sz, A, R, Rs, LseOp::UnsignedMin, Rn, Rt);
}

bool TranslatorVisitor::SWP(Imm<2> sz, bool A, bool R, Reg Rs, Reg Rn, Reg Rt) {
    return LseMemoryOperation(*this, sz, A, R, Rs, LseOp::Swap, Rn, Rt);
}

bool TranslatorVisitor::LDAPR(Imm<2> sz, Reg Rn, Reg Rt) {
    const size_t datasize = 8 << sz.ZeroExtend<size_t>();
    const size_t regsize = datasize == 64 ? 64 : 32;
    const IR::U64 address = LseAddress(*this, Rn);
    X(regsize, Rt, ZeroExtend(Mem(address, datasize / 8, IR::AccType::ORDERED), regsize));
    return true;
}

}  // namespace Dynarmic::A64
