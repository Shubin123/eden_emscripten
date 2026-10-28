/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

namespace {

/// A half-precision two-operand operation, `op` carried out in single precision.
template<typename Op>
bool HalfFloatBinary(TranslatorVisitor& v, Vec Vm, Vec Vn, Vec Vd, bool numeric, bool negate, Op op) {
    if (!v.HalfArithmeticSupported()) {
        return v.UnallocatedEncoding();
    }
    const IR::U16 operand1 = IR::U16{v.V_scalar(16, Vn)};
    const IR::U16 operand2 = IR::U16{v.V_scalar(16, Vm)};
    const IR::U32 single = IR::U32{op(v.HalfOperand(operand1), v.HalfOperand(operand2))};
    IR::U16 result = v.HalfNaNs(operand1, operand2, v.HalfResult(single), numeric);
    if (negate) {
        result = IR::U16{v.ir.FPNeg(result)};
    }
    v.V_scalar(16, Vd, result);
    return true;
}

}  // Anonymous namespace

bool TranslatorVisitor::FMUL_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        return HalfFloatBinary(*this, Vm, Vn, Vd, false, false, [this](const IR::U32& a, const IR::U32& b) { return ir.FPMul(a, b); });
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPMul(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FDIV_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        return HalfFloatBinary(*this, Vm, Vn, Vd, false, false, [this](const IR::U32& a, const IR::U32& b) { return ir.FPDiv(a, b); });
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPDiv(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FADD_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        return HalfFloatBinary(*this, Vm, Vn, Vd, false, false, [this](const IR::U32& a, const IR::U32& b) { return ir.FPAdd(a, b); });
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPAdd(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FSUB_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        return HalfFloatBinary(*this, Vm, Vn, Vd, false, false, [this](const IR::U32& a, const IR::U32& b) { return ir.FPSub(a, b); });
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPSub(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FMAX_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        return HalfFloatBinary(*this, Vm, Vn, Vd, false, false, [this](const IR::U32& a, const IR::U32& b) { return ir.FPMax(a, b); });
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPMax(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FMIN_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        return HalfFloatBinary(*this, Vm, Vn, Vd, false, false, [this](const IR::U32& a, const IR::U32& b) { return ir.FPMin(a, b); });
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPMin(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FMAXNM_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        return HalfFloatBinary(*this, Vm, Vn, Vd, true, false, [this](const IR::U32& a, const IR::U32& b) { return ir.FPMaxNumeric(a, b); });
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPMaxNumeric(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FMINNM_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        return HalfFloatBinary(*this, Vm, Vn, Vd, true, false, [this](const IR::U32& a, const IR::U32& b) { return ir.FPMinNumeric(a, b); });
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPMinNumeric(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FNMUL_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        return HalfFloatBinary(*this, Vm, Vn, Vd, false, true, [this](const IR::U32& a, const IR::U32& b) { return ir.FPMul(a, b); });
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPNeg(ir.FPMul(operand1, operand2));

    V_scalar(*datasize, Vd, result);
    return true;
}

}  // namespace Dynarmic::A64
