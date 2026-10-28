// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Advanced SIMD half-precision (Armv8.2 FP16) arithmetic, compares and conversions. As for the
// scalar forms (TranslatorVisitor::HalfOperand), four lanes at a time are converted to single
// precision, computed there and rounded back, with FZ16 and the NaN rules of the half operands
// applied on the half bits. Scalar forms run the same operations on lane 0.

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

namespace {

IR::U128 HalfSplat(TranslatorVisitor& v, u16 value) {
    return v.ir.VectorBroadcast(16, v.ir.Imm16(value));
}

/// `a` in the lanes where `mask` is set, else `b`.
IR::U128 HalfLaneSelect(TranslatorVisitor& v, const IR::U128& mask, const IR::U128& a, const IR::U128& b) {
    return v.ir.VectorOr(v.ir.VectorAnd(a, mask), v.ir.VectorAndNot(b, mask));
}

/// Under FZ16, half lanes with denormals replaced by zero of the same sign.
IR::U128 HalfLaneFlush(TranslatorVisitor& v, const IR::U128& bits) {
    if (!v.ir.current_location->FPCR().FZ16()) {
        return bits;
    }
    const IR::U128 zero_exponent = v.ir.VectorEqual(16, v.ir.VectorAnd(bits, HalfSplat(v, 0x7C00)), v.ir.ZeroVector());
    return HalfLaneSelect(v, zero_exponent, v.ir.VectorAnd(bits, HalfSplat(v, 0x8000)), bits);
}

/// All ones in the half lanes that hold a NaN.
IR::U128 HalfLaneNaNMask(TranslatorVisitor& v, const IR::U128& bits) {
    return v.ir.VectorGreaterSigned(16, v.ir.VectorAnd(bits, HalfSplat(v, 0x7FFF)), HalfSplat(v, 0x7C00));
}

/// The lower four half lanes as singles.
IR::U128 HalfLanesToSingle(TranslatorVisitor& v, const IR::U128& halves) {
    return v.ir.FPVectorFromHalf(32, HalfLaneFlush(v, halves), v.ir.current_location->FPCR().RMode());
}

/// Four singles rounded to half lanes in the lower 64 bits.
IR::U128 SingleToHalfLanes(TranslatorVisitor& v, const IR::U128& singles) {
    return HalfLaneFlush(v, v.ir.FPVectorToHalf(32, singles, v.ir.current_location->FPCR().RMode()));
}

/// The NaN rules of TranslatorVisitor::HalfNaNs, per lane.
IR::U128 HalfLaneNaNs(TranslatorVisitor& v, const IR::U128& a, const IR::U128& b, const IR::U128& result, bool numeric) {
    const IR::U128 quiet = HalfSplat(v, 0x200);
    const IR::U128 nan_a = HalfLaneNaNMask(v, a);
    const IR::U128 nan_b = HalfLaneNaNMask(v, b);
    const IR::U128 snan_a = v.ir.VectorAndNot(nan_a, v.ir.VectorEqual(16, v.ir.VectorAnd(a, quiet), quiet));
    const IR::U128 snan_b = v.ir.VectorAndNot(nan_b, v.ir.VectorEqual(16, v.ir.VectorAnd(b, quiet), quiet));
    const IR::U128 nan = v.ir.current_location->FPCR().DN()
                           ? HalfSplat(v, 0x7E00)
                           : HalfLaneSelect(v, snan_a, v.ir.VectorOr(a, quiet),
                                            HalfLaneSelect(v, snan_b, v.ir.VectorOr(b, quiet), HalfLaneSelect(v, nan_a, a, b)));
    const IR::U128 use_nan = numeric ? v.ir.VectorOr(v.ir.VectorOr(snan_a, snan_b), v.ir.VectorAnd(nan_a, nan_b))
                                     : v.ir.VectorOr(nan_a, nan_b);
    return HalfLaneSelect(v, use_nan, nan, result);
}

/// `op` on four half lanes at a time (in the lower 64 bits of its operands and result): the lower
/// four lanes, and for 128-bit data also the upper four.
template<typename Op>
IR::U128 HalfQuads(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b, Op op) {
    const IR::U128 low = op(a, b);
    if (datasize == 64) {
        return low;
    }
    const auto upper = [&](const IR::U128& value) {
        return v.ir.ZeroExtendToQuad(v.ir.VectorGetElement(64, value, 1));
    };
    return v.ir.VectorInterleaveLower(64, low, op(upper(a), upper(b)));
}

/// A half-precision result from a single-precision operation.
template<typename Op>
IR::U128 HalfLanesArithmetic(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b, bool numeric, Op op) {
    const IR::U128 result = HalfQuads(v, datasize, a, b, [&](const IR::U128& x, const IR::U128& y) {
        return SingleToHalfLanes(v, op(HalfLanesToSingle(v, x), HalfLanesToSingle(v, y)));
    });
    return HalfLaneNaNs(v, a, b, result, numeric);
}

/// Half lane masks from a single-precision comparison.
template<typename Op>
IR::U128 HalfLanesCompare(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b, Op op) {
    return HalfQuads(v, datasize, a, b, [&](const IR::U128& x, const IR::U128& y) {
        return v.ir.VectorNarrow(32, op(HalfLanesToSingle(v, x), HalfLanesToSingle(v, y)));
    });
}

using HalfLanesFunction = IR::U128 (*)(TranslatorVisitor&, size_t, const IR::U128&, const IR::U128&);

IR::U128 HalfAdd(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfLanesArithmetic(v, datasize, a, b, false, [&](const IR::U128& x, const IR::U128& y) { return v.ir.FPVectorAdd(32, x, y); });
}

IR::U128 HalfSub(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfLanesArithmetic(v, datasize, a, b, false, [&](const IR::U128& x, const IR::U128& y) { return v.ir.FPVectorSub(32, x, y); });
}

IR::U128 HalfMul(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfLanesArithmetic(v, datasize, a, b, false, [&](const IR::U128& x, const IR::U128& y) { return v.ir.FPVectorMul(32, x, y); });
}

IR::U128 HalfMulX(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfLanesArithmetic(v, datasize, a, b, false, [&](const IR::U128& x, const IR::U128& y) { return v.ir.FPVectorMulX(32, x, y); });
}

IR::U128 HalfDiv(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfLanesArithmetic(v, datasize, a, b, false, [&](const IR::U128& x, const IR::U128& y) { return v.ir.FPVectorDiv(32, x, y); });
}

IR::U128 HalfMax(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfLanesArithmetic(v, datasize, a, b, false, [&](const IR::U128& x, const IR::U128& y) { return v.ir.FPVectorMax(32, x, y); });
}

IR::U128 HalfMin(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfLanesArithmetic(v, datasize, a, b, false, [&](const IR::U128& x, const IR::U128& y) { return v.ir.FPVectorMin(32, x, y); });
}

IR::U128 HalfMaxNumeric(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfLanesArithmetic(v, datasize, a, b, true, [&](const IR::U128& x, const IR::U128& y) { return v.ir.FPVectorMaxNumeric(32, x, y); });
}

IR::U128 HalfMinNumeric(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfLanesArithmetic(v, datasize, a, b, true, [&](const IR::U128& x, const IR::U128& y) { return v.ir.FPVectorMinNumeric(32, x, y); });
}

/// The difference is rounded before its sign is cleared, NaNs included.
IR::U128 HalfAbsoluteDifference(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return v.ir.VectorAnd(HalfSub(v, datasize, a, b), HalfSplat(v, 0x7FFF));
}

IR::U128 HalfGreaterEqual(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfLanesCompare(v, datasize, a, b, [&](const IR::U128& x, const IR::U128& y) { return v.ir.FPVectorGreaterEqual(32, x, y); });
}

IR::U128 HalfGreater(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfLanesCompare(v, datasize, a, b, [&](const IR::U128& x, const IR::U128& y) { return v.ir.FPVectorGreater(32, x, y); });
}

IR::U128 HalfAbsoluteGreaterEqual(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfLanesCompare(v, datasize, a, b, [&](const IR::U128& x, const IR::U128& y) {
        return v.ir.FPVectorGreaterEqual(32, v.ir.FPVectorAbs(32, x), v.ir.FPVectorAbs(32, y));
    });
}

IR::U128 HalfAbsoluteGreater(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfLanesCompare(v, datasize, a, b, [&](const IR::U128& x, const IR::U128& y) {
        return v.ir.FPVectorGreater(32, v.ir.FPVectorAbs(32, x), v.ir.FPVectorAbs(32, y));
    });
}

IR::U128 HalfLessEqual(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfGreaterEqual(v, datasize, b, a);
}

IR::U128 HalfLess(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128& b) {
    return HalfGreater(v, datasize, b, a);
}

IR::U128 HalfSqrt(TranslatorVisitor& v, size_t datasize, const IR::U128& a, const IR::U128&) {
    const IR::U128 result = HalfQuads(v, datasize, a, a, [&](const IR::U128& x, const IR::U128&) {
        return SingleToHalfLanes(v, v.ir.FPVectorSqrt(32, HalfLanesToSingle(v, x)));
    });
    return HalfLaneNaNs(v, a, a, result, false);
}

/// Integers of 16 bits convert to single exactly, so rounding to half happens once.
IR::U128 HalfFromInteger(TranslatorVisitor& v, size_t datasize, const IR::U128& a, bool is_signed) {
    const auto rounding = v.ir.current_location->FPCR().RMode();
    return HalfQuads(v, datasize, a, a, [&](const IR::U128& x, const IR::U128&) {
        return SingleToHalfLanes(v, is_signed ? v.ir.FPVectorFromSignedFixed(32, v.ir.VectorSignExtend(16, x), 0, rounding)
                                              : v.ir.FPVectorFromUnsignedFixed(32, v.ir.VectorZeroExtend(16, x), 0, rounding));
    });
}

/// Halves convert to integers of 32 bits without saturating, then saturate to 16 bits (which does
/// not set FPSR.QC).
IR::U128 HalfToInteger(TranslatorVisitor& v, size_t datasize, const IR::U128& a, bool is_signed, FP::RoundingMode rounding) {
    return HalfQuads(v, datasize, a, a, [&](const IR::U128& x, const IR::U128&) {
        const IR::U128 single = HalfLanesToSingle(v, x);
        if (is_signed) {
            const IR::U128 wide = v.ir.FPVectorToSignedFixed(32, single, 0, rounding);
            const IR::U128 high = v.ir.VectorBroadcast(32, v.ir.Imm32(0x7FFF));
            const IR::U128 low = v.ir.VectorBroadcast(32, v.ir.Imm32(0xFFFF8000));
            return v.ir.VectorNarrow(32, v.ir.VectorMaxSigned(32, v.ir.VectorMinSigned(32, wide, high), low));
        }
        const IR::U128 wide = v.ir.FPVectorToUnsignedFixed(32, single, 0, rounding);
        return v.ir.VectorNarrow(32, v.ir.VectorMinUnsigned(32, wide, v.ir.VectorBroadcast(32, v.ir.Imm32(0xFFFF))));
    });
}

/// Lane 0 of a value, alone in a vector, so the other lanes compute nothing.
IR::U128 HalfLane0(TranslatorVisitor& v, const IR::U128& value) {
    return v.ir.ZeroExtendToQuad(v.ir.VectorGetElement(16, value, 0));
}

bool HalfVector(TranslatorVisitor& v, bool Q, Vec Vm, Vec Vn, Vec Vd, HalfLanesFunction op) {
    if (!v.HalfArithmeticSupported()) {
        return v.UnallocatedEncoding();
    }
    const size_t datasize = Q ? 128 : 64;
    v.V(datasize, Vd, op(v, datasize, v.V(datasize, Vn), v.V(datasize, Vm)));
    return true;
}

bool HalfVectorUnary(TranslatorVisitor& v, bool Q, Vec Vn, Vec Vd, HalfLanesFunction op) {
    if (!v.HalfArithmeticSupported()) {
        return v.UnallocatedEncoding();
    }
    const size_t datasize = Q ? 128 : 64;
    v.V(datasize, Vd, op(v, datasize, v.V(datasize, Vn), v.ir.ZeroVector()));
    return true;
}

bool HalfScalar(TranslatorVisitor& v, const IR::U128& n, const IR::U128& m, Vec Vd, HalfLanesFunction op) {
    if (!v.HalfArithmeticSupported()) {
        return v.UnallocatedEncoding();
    }
    const IR::U128 result = op(v, 64, HalfLane0(v, n), HalfLane0(v, m));
    v.V_scalar(16, Vd, v.ir.VectorGetElement(16, result, 0));
    return true;
}

bool HalfVectorToInteger(TranslatorVisitor& v, bool Q, Vec Vn, Vec Vd, bool is_signed, FP::RoundingMode rounding) {
    if (!v.HalfArithmeticSupported()) {
        return v.UnallocatedEncoding();
    }
    const size_t datasize = Q ? 128 : 64;
    v.V(datasize, Vd, HalfToInteger(v, datasize, v.V(datasize, Vn), is_signed, rounding));
    return true;
}

bool HalfScalarToInteger(TranslatorVisitor& v, Vec Vn, Vec Vd, bool is_signed, FP::RoundingMode rounding) {
    if (!v.HalfArithmeticSupported()) {
        return v.UnallocatedEncoding();
    }
    const IR::U128 result = HalfToInteger(v, 64, HalfLane0(v, v.V(128, Vn)), is_signed, rounding);
    v.V_scalar(16, Vd, v.ir.VectorGetElement(16, result, 0));
    return true;
}

bool HalfVectorFromInteger(TranslatorVisitor& v, bool Q, Vec Vn, Vec Vd, bool is_signed) {
    if (!v.HalfArithmeticSupported()) {
        return v.UnallocatedEncoding();
    }
    const size_t datasize = Q ? 128 : 64;
    v.V(datasize, Vd, HalfFromInteger(v, datasize, v.V(datasize, Vn), is_signed));
    return true;
}

bool HalfScalarFromInteger(TranslatorVisitor& v, Vec Vn, Vec Vd, bool is_signed) {
    if (!v.HalfArithmeticSupported()) {
        return v.UnallocatedEncoding();
    }
    const IR::U128 result = HalfFromInteger(v, 64, HalfLane0(v, v.V(128, Vn)), is_signed);
    v.V_scalar(16, Vd, v.ir.VectorGetElement(16, result, 0));
    return true;
}

/// Adjacent pairs of the concatenation of Vn and Vm.
bool HalfPairwise(TranslatorVisitor& v, bool Q, Vec Vm, Vec Vn, Vec Vd, HalfLanesFunction op) {
    if (!v.HalfArithmeticSupported()) {
        return v.UnallocatedEncoding();
    }
    const size_t datasize = Q ? 128 : 64;
    const IR::U128 n = v.V(datasize, Vn);
    const IR::U128 m = v.V(datasize, Vm);
    const IR::U128 even = Q ? v.ir.VectorDeinterleaveEven(16, n, m) : v.ir.VectorDeinterleaveEvenLower(16, n, m);
    const IR::U128 odd = Q ? v.ir.VectorDeinterleaveOdd(16, n, m) : v.ir.VectorDeinterleaveOddLower(16, n, m);
    v.V(datasize, Vd, op(v, datasize, even, odd));
    return true;
}

bool HalfScalarPairwise(TranslatorVisitor& v, Vec Vn, Vec Vd, HalfLanesFunction op) {
    const IR::U128 n = v.V(128, Vn);
    return HalfScalar(v, n, v.ir.ZeroExtendToQuad(v.ir.VectorGetElement(16, n, 1)), Vd, op);
}

/// The architectural reduction order: pairs of lanes, then pairs of those results.
bool HalfAcrossLanes(TranslatorVisitor& v, bool Q, Vec Vn, Vec Vd, HalfLanesFunction op) {
    if (!v.HalfArithmeticSupported()) {
        return v.UnallocatedEncoding();
    }
    IR::U128 lanes = v.V(Q ? 128 : 64, Vn);
    for (size_t count = Q ? 8 : 4; count > 1; count /= 2) {
        lanes = op(v, 64, v.ir.VectorDeinterleaveEven(16, lanes, lanes), v.ir.VectorDeinterleaveOdd(16, lanes, lanes));
    }
    v.V_scalar(16, Vd, v.ir.VectorGetElement(16, lanes, 0));
    return true;
}

IR::U128 HalfIndexedElement(TranslatorVisitor& v, Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H) {
    const size_t index = concatenate(H, L, M).ZeroExtend();
    return v.ir.VectorBroadcastElement(16, v.V(H == 1 ? 128 : 64, Vmlo.ZeroExtend<Vec>()), index);
}

}  // Anonymous namespace

bool TranslatorVisitor::FADD_1(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfAdd);
}

bool TranslatorVisitor::FSUB_1(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfSub);
}

bool TranslatorVisitor::FMUL_vec_1(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfMul);
}

bool TranslatorVisitor::FMULX_vec_3(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfMulX);
}

bool TranslatorVisitor::FDIV_1(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfDiv);
}

bool TranslatorVisitor::FMAX_1(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfMax);
}

bool TranslatorVisitor::FMIN_1(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfMin);
}

bool TranslatorVisitor::FMAXNM_1(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfMaxNumeric);
}

bool TranslatorVisitor::FMINNM_1(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfMinNumeric);
}

bool TranslatorVisitor::FABD_3(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfAbsoluteDifference);
}

bool TranslatorVisitor::FCMGE_reg_3(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfGreaterEqual);
}

bool TranslatorVisitor::FCMGT_reg_3(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfGreater);
}

bool TranslatorVisitor::FACGE_3(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfAbsoluteGreaterEqual);
}

bool TranslatorVisitor::FACGT_3(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfVector(*this, Q, Vm, Vn, Vd, HalfAbsoluteGreater);
}

bool TranslatorVisitor::FADDP_vec_1(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfPairwise(*this, Q, Vm, Vn, Vd, HalfAdd);
}

bool TranslatorVisitor::FMAXP_vec_1(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfPairwise(*this, Q, Vm, Vn, Vd, HalfMax);
}

bool TranslatorVisitor::FMINP_vec_1(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfPairwise(*this, Q, Vm, Vn, Vd, HalfMin);
}

bool TranslatorVisitor::FMAXNMP_vec_1(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfPairwise(*this, Q, Vm, Vn, Vd, HalfMaxNumeric);
}

bool TranslatorVisitor::FMINNMP_vec_1(bool Q, Vec Vm, Vec Vn, Vec Vd) {
    return HalfPairwise(*this, Q, Vm, Vn, Vd, HalfMinNumeric);
}

bool TranslatorVisitor::FSQRT_1(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorUnary(*this, Q, Vn, Vd, HalfSqrt);
}

bool TranslatorVisitor::FCMGT_zero_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorUnary(*this, Q, Vn, Vd, HalfGreater);
}

bool TranslatorVisitor::FCMGE_zero_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorUnary(*this, Q, Vn, Vd, HalfGreaterEqual);
}

bool TranslatorVisitor::FCMLT_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorUnary(*this, Q, Vn, Vd, HalfLess);
}

bool TranslatorVisitor::FCMLE_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorUnary(*this, Q, Vn, Vd, HalfLessEqual);
}

bool TranslatorVisitor::SCVTF_int_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorFromInteger(*this, Q, Vn, Vd, true);
}

bool TranslatorVisitor::UCVTF_int_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorFromInteger(*this, Q, Vn, Vd, false);
}

bool TranslatorVisitor::FCVTNS_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorToInteger(*this, Q, Vn, Vd, true, FP::RoundingMode::ToNearest_TieEven);
}

bool TranslatorVisitor::FCVTMS_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorToInteger(*this, Q, Vn, Vd, true, FP::RoundingMode::TowardsMinusInfinity);
}

bool TranslatorVisitor::FCVTAS_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorToInteger(*this, Q, Vn, Vd, true, FP::RoundingMode::ToNearest_TieAwayFromZero);
}

bool TranslatorVisitor::FCVTPS_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorToInteger(*this, Q, Vn, Vd, true, FP::RoundingMode::TowardsPlusInfinity);
}

bool TranslatorVisitor::FCVTZS_int_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorToInteger(*this, Q, Vn, Vd, true, FP::RoundingMode::TowardsZero);
}

bool TranslatorVisitor::FCVTNU_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorToInteger(*this, Q, Vn, Vd, false, FP::RoundingMode::ToNearest_TieEven);
}

bool TranslatorVisitor::FCVTMU_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorToInteger(*this, Q, Vn, Vd, false, FP::RoundingMode::TowardsMinusInfinity);
}

bool TranslatorVisitor::FCVTAU_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorToInteger(*this, Q, Vn, Vd, false, FP::RoundingMode::ToNearest_TieAwayFromZero);
}

bool TranslatorVisitor::FCVTPU_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorToInteger(*this, Q, Vn, Vd, false, FP::RoundingMode::TowardsPlusInfinity);
}

bool TranslatorVisitor::FCVTZU_int_3(bool Q, Vec Vn, Vec Vd) {
    return HalfVectorToInteger(*this, Q, Vn, Vd, false, FP::RoundingMode::TowardsZero);
}

bool TranslatorVisitor::FMAXV_1(bool Q, Vec Vn, Vec Vd) {
    return HalfAcrossLanes(*this, Q, Vn, Vd, HalfMax);
}

bool TranslatorVisitor::FMINV_1(bool Q, Vec Vn, Vec Vd) {
    return HalfAcrossLanes(*this, Q, Vn, Vd, HalfMin);
}

bool TranslatorVisitor::FMAXNMV_1(bool Q, Vec Vn, Vec Vd) {
    return HalfAcrossLanes(*this, Q, Vn, Vd, HalfMaxNumeric);
}

bool TranslatorVisitor::FMINNMV_1(bool Q, Vec Vn, Vec Vd) {
    return HalfAcrossLanes(*this, Q, Vn, Vd, HalfMinNumeric);
}

bool TranslatorVisitor::FMUL_elt_3(bool Q, Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H, Vec Vn, Vec Vd) {
    if (!HalfArithmeticSupported()) {
        return UnallocatedEncoding();
    }
    const size_t datasize = Q ? 128 : 64;
    V(datasize, Vd, HalfMul(*this, datasize, V(datasize, Vn), HalfIndexedElement(*this, L, M, Vmlo, H)));
    return true;
}

bool TranslatorVisitor::FMULX_elt_3(bool Q, Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H, Vec Vn, Vec Vd) {
    if (!HalfArithmeticSupported()) {
        return UnallocatedEncoding();
    }
    const size_t datasize = Q ? 128 : 64;
    V(datasize, Vd, HalfMulX(*this, datasize, V(datasize, Vn), HalfIndexedElement(*this, L, M, Vmlo, H)));
    return true;
}

bool TranslatorVisitor::FABD_1(Vec Vm, Vec Vn, Vec Vd) {
    return HalfScalar(*this, V(128, Vn), V(128, Vm), Vd, HalfAbsoluteDifference);
}

bool TranslatorVisitor::FMULX_vec_1(Vec Vm, Vec Vn, Vec Vd) {
    return HalfScalar(*this, V(128, Vn), V(128, Vm), Vd, HalfMulX);
}

bool TranslatorVisitor::FCMGE_reg_1(Vec Vm, Vec Vn, Vec Vd) {
    return HalfScalar(*this, V(128, Vn), V(128, Vm), Vd, HalfGreaterEqual);
}

bool TranslatorVisitor::FCMGT_reg_1(Vec Vm, Vec Vn, Vec Vd) {
    return HalfScalar(*this, V(128, Vn), V(128, Vm), Vd, HalfGreater);
}

bool TranslatorVisitor::FACGE_1(Vec Vm, Vec Vn, Vec Vd) {
    return HalfScalar(*this, V(128, Vn), V(128, Vm), Vd, HalfAbsoluteGreaterEqual);
}

bool TranslatorVisitor::FACGT_1(Vec Vm, Vec Vn, Vec Vd) {
    return HalfScalar(*this, V(128, Vn), V(128, Vm), Vd, HalfAbsoluteGreater);
}

bool TranslatorVisitor::FCMGT_zero_1(Vec Vn, Vec Vd) {
    return HalfScalar(*this, V(128, Vn), ir.ZeroVector(), Vd, HalfGreater);
}

bool TranslatorVisitor::FCMGE_zero_1(Vec Vn, Vec Vd) {
    return HalfScalar(*this, V(128, Vn), ir.ZeroVector(), Vd, HalfGreaterEqual);
}

bool TranslatorVisitor::FCMLT_1(Vec Vn, Vec Vd) {
    return HalfScalar(*this, V(128, Vn), ir.ZeroVector(), Vd, HalfLess);
}

bool TranslatorVisitor::FCMLE_1(Vec Vn, Vec Vd) {
    return HalfScalar(*this, V(128, Vn), ir.ZeroVector(), Vd, HalfLessEqual);
}

bool TranslatorVisitor::SCVTF_int_1(Vec Vn, Vec Vd) {
    return HalfScalarFromInteger(*this, Vn, Vd, true);
}

bool TranslatorVisitor::UCVTF_int_1(Vec Vn, Vec Vd) {
    return HalfScalarFromInteger(*this, Vn, Vd, false);
}

bool TranslatorVisitor::FCVTNS_1(Vec Vn, Vec Vd) {
    return HalfScalarToInteger(*this, Vn, Vd, true, FP::RoundingMode::ToNearest_TieEven);
}

bool TranslatorVisitor::FCVTMS_1(Vec Vn, Vec Vd) {
    return HalfScalarToInteger(*this, Vn, Vd, true, FP::RoundingMode::TowardsMinusInfinity);
}

bool TranslatorVisitor::FCVTAS_1(Vec Vn, Vec Vd) {
    return HalfScalarToInteger(*this, Vn, Vd, true, FP::RoundingMode::ToNearest_TieAwayFromZero);
}

bool TranslatorVisitor::FCVTPS_1(Vec Vn, Vec Vd) {
    return HalfScalarToInteger(*this, Vn, Vd, true, FP::RoundingMode::TowardsPlusInfinity);
}

bool TranslatorVisitor::FCVTZS_int_1(Vec Vn, Vec Vd) {
    return HalfScalarToInteger(*this, Vn, Vd, true, FP::RoundingMode::TowardsZero);
}

bool TranslatorVisitor::FCVTNU_1(Vec Vn, Vec Vd) {
    return HalfScalarToInteger(*this, Vn, Vd, false, FP::RoundingMode::ToNearest_TieEven);
}

bool TranslatorVisitor::FCVTMU_1(Vec Vn, Vec Vd) {
    return HalfScalarToInteger(*this, Vn, Vd, false, FP::RoundingMode::TowardsMinusInfinity);
}

bool TranslatorVisitor::FCVTAU_1(Vec Vn, Vec Vd) {
    return HalfScalarToInteger(*this, Vn, Vd, false, FP::RoundingMode::ToNearest_TieAwayFromZero);
}

bool TranslatorVisitor::FCVTPU_1(Vec Vn, Vec Vd) {
    return HalfScalarToInteger(*this, Vn, Vd, false, FP::RoundingMode::TowardsPlusInfinity);
}

bool TranslatorVisitor::FCVTZU_int_1(Vec Vn, Vec Vd) {
    return HalfScalarToInteger(*this, Vn, Vd, false, FP::RoundingMode::TowardsZero);
}

bool TranslatorVisitor::FADDP_pair_1(Vec Vn, Vec Vd) {
    return HalfScalarPairwise(*this, Vn, Vd, HalfAdd);
}

bool TranslatorVisitor::FMAXP_pair_1(Vec Vn, Vec Vd) {
    return HalfScalarPairwise(*this, Vn, Vd, HalfMax);
}

bool TranslatorVisitor::FMINP_pair_1(Vec Vn, Vec Vd) {
    return HalfScalarPairwise(*this, Vn, Vd, HalfMin);
}

bool TranslatorVisitor::FMAXNMP_pair_1(Vec Vn, Vec Vd) {
    return HalfScalarPairwise(*this, Vn, Vd, HalfMaxNumeric);
}

bool TranslatorVisitor::FMINNMP_pair_1(Vec Vn, Vec Vd) {
    return HalfScalarPairwise(*this, Vn, Vd, HalfMinNumeric);
}

bool TranslatorVisitor::FMUL_elt_1(Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H, Vec Vn, Vec Vd) {
    return HalfScalar(*this, V(128, Vn), HalfIndexedElement(*this, L, M, Vmlo, H), Vd, HalfMul);
}

bool TranslatorVisitor::FMULX_elt_1(Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H, Vec Vn, Vec Vd) {
    return HalfScalar(*this, V(128, Vn), HalfIndexedElement(*this, L, M, Vmlo, H), Vd, HalfMulX);
}

}  // namespace Dynarmic::A64
