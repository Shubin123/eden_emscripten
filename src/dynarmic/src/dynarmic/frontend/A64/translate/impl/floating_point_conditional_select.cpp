/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

bool TranslatorVisitor::FCSEL_float(Imm<2> type, Vec Vm, Cond cond, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        const IR::U32 operand1 = ir.ZeroExtendHalfToWord(IR::U16{V_scalar(16, Vn)});
        const IR::U32 operand2 = ir.ZeroExtendHalfToWord(IR::U16{V_scalar(16, Vm)});
        V_scalar(16, Vd, ir.LeastSignificantHalf(ir.ConditionalSelect(cond, operand1, operand2)));
        return true;
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);
    const IR::U32U64 result = ir.ConditionalSelect(cond, operand1, operand2);
    V_scalar(*datasize, Vd, result);

    return true;
}

}  // namespace Dynarmic::A64
