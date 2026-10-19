// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::WGSL {
namespace {
void SetZeroFlag(EmitContext& ctx, IR::Inst& inst, std::string_view result) {
    IR::Inst* const zero{inst.GetAssociatedPseudoOperation(IR::Opcode::GetZeroFromOp)};
    if (!zero) {
        return;
    }
    ctx.AddU1("{}={}==0u;", *zero, result);
    zero->Invalidate();
}

void SetSignFlag(EmitContext& ctx, IR::Inst& inst, std::string_view result) {
    IR::Inst* const sign{inst.GetAssociatedPseudoOperation(IR::Opcode::GetSignFromOp)};
    if (!sign) {
        return;
    }
    ctx.AddU1("{}=i32({})<0;", *sign, result);
    sign->Invalidate();
}

void BitwiseLogicalOp(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b,
                      char lop) {
    const auto result{ctx.var_alloc.Define(inst, WgslVarType::U32)};
    ctx.Add("{}=({}){}({});", result, a, lop, b);
    SetZeroFlag(ctx, inst, result);
    SetSignFlag(ctx, inst, result);
}
} // Anonymous namespace

void EmitIAdd32(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    // Compute the overflow CC first as it requires the original operand values,
    // which may be overwritten by the result of the addition
    if (IR::Inst * overflow{inst.GetAssociatedPseudoOperation(IR::Opcode::GetOverflowFromOp)}) {
        constexpr u32 s32_max{static_cast<u32>((std::numeric_limits<s32>::max)())};
        const auto sub_a{fmt::format("{}u-({})", s32_max, a)};
        const auto positive_result{fmt::format("i32({})>i32({})", b, sub_a)};
        const auto negative_result{fmt::format("i32({})<i32({})", b, sub_a)};
        ctx.AddU1("{}=select({},{},i32({})>=0);", *overflow, negative_result, positive_result, a);
        overflow->Invalidate();
    }
    const auto result{ctx.var_alloc.Define(inst, WgslVarType::U32)};
    if (IR::Inst* const carry{inst.GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp)}) {
        ctx.uses_cc_carry = true;
        ctx.Add("carry=({});", a);
        ctx.Add("{}=carry+({});", result, b);
        ctx.AddU1("{}={}<carry;", *carry, result);
        carry->Invalidate();
    } else {
        ctx.Add("{}=({})+({});", result, a, b);
    }
    SetZeroFlag(ctx, inst, result);
    SetSignFlag(ctx, inst, result);
}

void EmitIAdd64(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    ctx.AddU64("{}=vec2<u32>({}.x+{}.x,{}.y+{}.y+select(0u,1u,({}.x+{}.x)<{}.x));", inst, a, b, a,
               b, a, b, a);
}

void EmitISub32(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    ctx.AddU32("{}=({})-({});", inst, a, b);
}

void EmitISub64(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    ctx.AddU64("{}=vec2<u32>({}.x-{}.x,{}.y-{}.y-select(0u,1u,{}.x<{}.x));", inst, a, b, a, b, a,
               b);
}

void EmitIMul32(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    ctx.AddU32("{}=({})*({});", inst, a, b);
}

void EmitSDiv32(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    ctx.AddU32("{}=u32(i32({})/i32({}));", inst, a, b);
}

void EmitUDiv32(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    ctx.AddU32("{}=({})/({});", inst, a, b);
}

void EmitINeg32(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU32("{}=0u-({});", inst, value);
}

void EmitINeg64(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU64("{}=vec2<u32>(0u-{}.x,0u-{}.y-select(0u,1u,{}.x!=0u));", inst, value, value, value);
}

void EmitIAbs32(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU32("{}=u32(abs(i32({})));", inst, value);
}

void EmitIAbs64(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU64("{}={};", inst, value);
}

void EmitShiftLeftLogical32(EmitContext& ctx, IR::Inst& inst, std::string_view base,
                            std::string_view shift) {
    ctx.AddU32("{}=shl32({},{});", inst, base, shift);
}

void EmitShiftLeftLogical64(EmitContext& ctx, IR::Inst& inst, std::string_view base,
                            std::string_view shift) {
    ctx.AddU64("{}=shl64({},{});", inst, base, shift);
}

void EmitShiftRightLogical32(EmitContext& ctx, IR::Inst& inst, std::string_view base,
                             std::string_view shift) {
    ctx.AddU32("{}=shr32({},{});", inst, base, shift);
}

void EmitShiftRightLogical64(EmitContext& ctx, IR::Inst& inst, std::string_view base,
                             std::string_view shift) {
    ctx.AddU64("{}=shr64({},{});", inst, base, shift);
}

void EmitShiftRightArithmetic32(EmitContext& ctx, IR::Inst& inst, std::string_view base,
                                std::string_view shift) {
    ctx.AddU32("{}=sar32({},{});", inst, base, shift);
}

void EmitShiftRightArithmetic64(EmitContext& ctx, IR::Inst& inst, std::string_view base,
                                std::string_view shift) {
    ctx.AddU64("{}=sar64({},{});", inst, base, shift);
}

void EmitBitwiseAnd32(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    BitwiseLogicalOp(ctx, inst, a, b, '&');
}

void EmitBitwiseOr32(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    BitwiseLogicalOp(ctx, inst, a, b, '|');
}

void EmitBitwiseXor32(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    BitwiseLogicalOp(ctx, inst, a, b, '^');
}

void EmitBitFieldInsert(EmitContext& ctx, IR::Inst& inst, std::string_view base,
                        std::string_view insert, std::string_view offset, std::string_view count) {
    ctx.AddU32("{}=insertBits({},{},min({},32u),min({},32u-min({},32u)));", inst, base, insert,
               offset, count, offset);
}

void EmitBitFieldSExtract(EmitContext& ctx, IR::Inst& inst, std::string_view base,
                          std::string_view offset, std::string_view count) {
    const auto result{ctx.var_alloc.Define(inst, WgslVarType::U32)};
    ctx.Add("{}=u32(extractBits(i32({}),min({},32u),min({},32u-min({},32u))));", result, base,
            offset, count, offset);
    SetZeroFlag(ctx, inst, result);
    SetSignFlag(ctx, inst, result);
}

void EmitBitFieldUExtract(EmitContext& ctx, IR::Inst& inst, std::string_view base,
                          std::string_view offset, std::string_view count) {
    const auto result{ctx.var_alloc.Define(inst, WgslVarType::U32)};
    ctx.Add("{}=extractBits({},min({},32u),min({},32u-min({},32u)));", result, base, offset, count,
            offset);
    SetZeroFlag(ctx, inst, result);
    SetSignFlag(ctx, inst, result);
}

void EmitBitReverse32(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU32("{}=reverseBits({});", inst, value);
}

void EmitBitCount32(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU32("{}=countOneBits({});", inst, value);
}

void EmitBitwiseNot32(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU32("{}=~({});", inst, value);
}

void EmitFindSMsb32(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU32("{}=u32(firstLeadingBit(i32({})));", inst, value);
}

void EmitFindUMsb32(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU32("{}=firstLeadingBit({});", inst, value);
}

void EmitSMin32(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    ctx.AddU32("{}=u32(min(i32({}),i32({})));", inst, a, b);
}

void EmitUMin32(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    ctx.AddU32("{}=min({},{});", inst, a, b);
}

void EmitSMax32(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    ctx.AddU32("{}=u32(max(i32({}),i32({})));", inst, a, b);
}

void EmitUMax32(EmitContext& ctx, IR::Inst& inst, std::string_view a, std::string_view b) {
    ctx.AddU32("{}=max({},{});", inst, a, b);
}

void EmitSClamp32(EmitContext& ctx, IR::Inst& inst, std::string_view value, std::string_view min,
                  std::string_view max) {
    const auto result{ctx.var_alloc.Define(inst, WgslVarType::U32)};
    ctx.Add("{}=u32(min(max(i32({}),i32({})),i32({})));", result, value, min, max);
    SetZeroFlag(ctx, inst, result);
    SetSignFlag(ctx, inst, result);
}

void EmitUClamp32(EmitContext& ctx, IR::Inst& inst, std::string_view value, std::string_view min,
                  std::string_view max) {
    const auto result{ctx.var_alloc.Define(inst, WgslVarType::U32)};
    ctx.Add("{}=min(max({},{}),{});", result, value, min, max);
    SetZeroFlag(ctx, inst, result);
    SetSignFlag(ctx, inst, result);
}

void EmitSLessThan(EmitContext& ctx, IR::Inst& inst, std::string_view lhs, std::string_view rhs) {
    ctx.AddU1("{}=i32({})<i32({});", inst, lhs, rhs);
}

void EmitULessThan(EmitContext& ctx, IR::Inst& inst, std::string_view lhs, std::string_view rhs) {
    ctx.AddU1("{}=({})<({});", inst, lhs, rhs);
}

void EmitIEqual(EmitContext& ctx, IR::Inst& inst, std::string_view lhs, std::string_view rhs) {
    ctx.AddU1("{}=({})==({});", inst, lhs, rhs);
}

void EmitSLessThanEqual(EmitContext& ctx, IR::Inst& inst, std::string_view lhs,
                        std::string_view rhs) {
    ctx.AddU1("{}=i32({})<=i32({});", inst, lhs, rhs);
}

void EmitULessThanEqual(EmitContext& ctx, IR::Inst& inst, std::string_view lhs,
                        std::string_view rhs) {
    ctx.AddU1("{}=({})<=({});", inst, lhs, rhs);
}

void EmitSGreaterThan(EmitContext& ctx, IR::Inst& inst, std::string_view lhs,
                      std::string_view rhs) {
    ctx.AddU1("{}=i32({})>i32({});", inst, lhs, rhs);
}

void EmitUGreaterThan(EmitContext& ctx, IR::Inst& inst, std::string_view lhs,
                      std::string_view rhs) {
    ctx.AddU1("{}=({})>({});", inst, lhs, rhs);
}

void EmitINotEqual(EmitContext& ctx, IR::Inst& inst, std::string_view lhs, std::string_view rhs) {
    ctx.AddU1("{}=({})!=({});", inst, lhs, rhs);
}

void EmitSGreaterThanEqual(EmitContext& ctx, IR::Inst& inst, std::string_view lhs,
                           std::string_view rhs) {
    ctx.AddU1("{}=i32({})>=i32({});", inst, lhs, rhs);
}

void EmitUGreaterThanEqual(EmitContext& ctx, IR::Inst& inst, std::string_view lhs,
                           std::string_view rhs) {
    ctx.AddU1("{}=({})>=({});", inst, lhs, rhs);
}
} // namespace Shader::Backend::WGSL
