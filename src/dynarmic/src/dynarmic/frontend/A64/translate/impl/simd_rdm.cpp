// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SQRDMLAH and SQRDMLSH (Armv8.1 RDM): per lane, the accumulator plus (or minus) the rounded high
// half of twice the product, saturated, which sets FPSR.QC.
//
// (d << e) + 2nm + 2^(e-1), shifted right by e, is d + ((2nm + 2^(e-1)) >> e) since d << e is a
// multiple of 2^e, and that is d + ((nm + 2^(e-2)) >> (e-1)), which fits double-width lanes even
// for n = m = -2^(e-1).

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

namespace {

/// The operation on the lanes in the lower 64 bits of n, m and d; the result is in the lower 64 bits.
IR::U128 RoundingDoublingAccumulateLow(TranslatorVisitor& v, size_t esize, const IR::U128& n, const IR::U128& m, const IR::U128& d, bool subtract) {
    const size_t wide = esize * 2;
    IR::U128 product = v.ir.VectorMultiply(wide, v.ir.VectorSignExtend(esize, n), v.ir.VectorSignExtend(esize, m));
    if (subtract) {
        product = v.ir.VectorSub(wide, v.ir.ZeroVector(), product);
    }
    const IR::U128 round = wide == 64 ? v.ir.VectorBroadcast(64, v.ir.Imm64(u64(1) << (esize - 2)))
                                      : v.ir.VectorBroadcast(32, v.ir.Imm32(u32(1) << (esize - 2)));
    const IR::U128 high = v.ir.VectorArithmeticShiftRight(wide, v.ir.VectorAdd(wide, product, round), u8(esize - 1));
    return v.ir.VectorSignedSaturatedNarrowToSigned(wide, v.ir.VectorAdd(wide, v.ir.VectorSignExtend(esize, d), high));
}

IR::U128 RoundingDoublingAccumulate(TranslatorVisitor& v, size_t datasize, size_t esize, const IR::U128& n, const IR::U128& m, const IR::U128& d, bool subtract) {
    const IR::U128 low = RoundingDoublingAccumulateLow(v, esize, n, m, d, subtract);
    if (datasize == 64) {
        return low;
    }
    const auto upper = [&](const IR::U128& value) {
        return v.ir.ZeroExtendToQuad(v.ir.VectorGetElement(64, value, 1));
    };
    const IR::U128 high = RoundingDoublingAccumulateLow(v, esize, upper(n), upper(m), upper(d), subtract);
    return v.ir.VectorInterleaveLower(64, low, high);
}

bool RoundingDoublingAccumulateVector(TranslatorVisitor& v, bool Q, Imm<2> size, Vec Vm, Vec Vn, Vec Vd, bool subtract) {
    if (size == 0b00 || size == 0b11) {
        return v.ReservedValue();
    }
    const size_t esize = 8 << size.ZeroExtend();
    const size_t datasize = Q ? 128 : 64;
    const IR::U128 result = RoundingDoublingAccumulate(v, datasize, esize, v.V(datasize, Vn), v.V(datasize, Vm), v.V(datasize, Vd), subtract);
    v.V(datasize, Vd, result);
    return true;
}

bool RoundingDoublingAccumulateElement(TranslatorVisitor& v, bool Q, Imm<2> size, Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H, Vec Vn, Vec Vd, bool subtract) {
    if (size == 0b00 || size == 0b11) {
        return v.ReservedValue();
    }
    const size_t idxsize = H == 1 ? 128 : 64;
    const size_t esize = 8 << size.ZeroExtend();
    const size_t datasize = Q ? 128 : 64;
    const auto [index, Vm] = CombineVector(size, H, L, M, Vmlo);
    const IR::U128 element = v.ir.VectorBroadcastElement(esize, v.V(idxsize, Vm), index);
    const IR::U128 result = RoundingDoublingAccumulate(v, datasize, esize, v.V(datasize, Vn), element, v.V(datasize, Vd), subtract);
    v.V(datasize, Vd, result);
    return true;
}

/// The scalar forms: only element 0 takes part, so other lanes cannot saturate.
bool RoundingDoublingAccumulateScalar(TranslatorVisitor& v, size_t esize, const IR::U128& m, Vec Vn, Vec Vd, bool subtract) {
    const auto element0 = [&](const IR::U128& value) {
        return v.ir.ZeroExtendToQuad(v.ir.VectorGetElement(esize, value, 0));
    };
    const IR::U128 result = RoundingDoublingAccumulateLow(v, esize, element0(v.V(128, Vn)), element0(m), element0(v.V(128, Vd)), subtract);
    v.V_scalar(esize, Vd, v.ir.VectorGetElement(esize, result, 0));
    return true;
}

}  // Anonymous namespace

bool TranslatorVisitor::SQRDMLAH_vec_2(bool Q, Imm<2> size, Vec Vm, Vec Vn, Vec Vd) {
    return RoundingDoublingAccumulateVector(*this, Q, size, Vm, Vn, Vd, false);
}

bool TranslatorVisitor::SQRDMLSH_vec_2(bool Q, Imm<2> size, Vec Vm, Vec Vn, Vec Vd) {
    return RoundingDoublingAccumulateVector(*this, Q, size, Vm, Vn, Vd, true);
}

bool TranslatorVisitor::SQRDMLAH_elt_2(bool Q, Imm<2> size, Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H, Vec Vn, Vec Vd) {
    return RoundingDoublingAccumulateElement(*this, Q, size, L, M, Vmlo, H, Vn, Vd, false);
}

bool TranslatorVisitor::SQRDMLSH_elt_2(bool Q, Imm<2> size, Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H, Vec Vn, Vec Vd) {
    return RoundingDoublingAccumulateElement(*this, Q, size, L, M, Vmlo, H, Vn, Vd, true);
}

bool TranslatorVisitor::SQRDMLAH_vec_1(Imm<2> size, Vec Vm, Vec Vn, Vec Vd) {
    if (size == 0b00 || size == 0b11) {
        return ReservedValue();
    }
    return RoundingDoublingAccumulateScalar(*this, 8 << size.ZeroExtend(), V(128, Vm), Vn, Vd, false);
}

bool TranslatorVisitor::SQRDMLSH_vec_1(Imm<2> size, Vec Vm, Vec Vn, Vec Vd) {
    if (size == 0b00 || size == 0b11) {
        return ReservedValue();
    }
    return RoundingDoublingAccumulateScalar(*this, 8 << size.ZeroExtend(), V(128, Vm), Vn, Vd, true);
}

bool TranslatorVisitor::SQRDMLAH_elt_1(Imm<2> size, Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H, Vec Vn, Vec Vd) {
    if (size == 0b00 || size == 0b11) {
        return ReservedValue();
    }
    const size_t esize = 8 << size.ZeroExtend();
    const auto [index, Vm] = CombineScalar(size, H, L, M, Vmlo);
    return RoundingDoublingAccumulateScalar(*this, esize, ir.VectorBroadcastElement(esize, V(128, Vm), index), Vn, Vd, false);
}

bool TranslatorVisitor::SQRDMLSH_elt_1(Imm<2> size, Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H, Vec Vn, Vec Vd) {
    if (size == 0b00 || size == 0b11) {
        return ReservedValue();
    }
    const size_t esize = 8 << size.ZeroExtend();
    const auto [index, Vm] = CombineScalar(size, H, L, M, Vmlo);
    return RoundingDoublingAccumulateScalar(*this, esize, ir.VectorBroadcastElement(esize, V(128, Vm), index), Vn, Vd, true);
}

}  // namespace Dynarmic::A64
