// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string_view>

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/profile.h"

namespace Shader::Backend::WGSL {
// Subgroup operations are emulated for a single invocation; WebGPU subgroups are not used yet.
namespace {
void SetInBoundsFlag(EmitContext& ctx, IR::Inst& inst) {
    IR::Inst* const in_bounds{inst.GetAssociatedPseudoOperation(IR::Opcode::GetInBoundsFromOp)};
    if (!in_bounds) {
        return;
    }
    ctx.AddU1("{}=true;", *in_bounds);
    in_bounds->Invalidate();
}

void Passthrough(EmitContext& ctx, IR::Inst& inst, std::string_view value) {
    ctx.AddU32("{}={};", inst, value);
    SetInBoundsFlag(ctx, inst);
}
} // Anonymous namespace

void EmitLaneId(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=0u;", inst);
}

void EmitVoteAll(EmitContext& ctx, IR::Inst& inst, std::string_view pred) {
    ctx.AddU1("{}={};", inst, pred);
}

void EmitVoteAny(EmitContext& ctx, IR::Inst& inst, std::string_view pred) {
    ctx.AddU1("{}={};", inst, pred);
}

void EmitVoteEqual(EmitContext& ctx, IR::Inst& inst, std::string_view pred) {
    ctx.AddU1("{}=true;", inst);
}

void EmitSubgroupBallot(EmitContext& ctx, IR::Inst& inst, std::string_view pred) {
    ctx.AddU32("{}=select(0u,1u,{});", inst, pred);
}

void EmitSubgroupEqMask(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=1u;", inst);
}

void EmitSubgroupLtMask(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=0u;", inst);
}

void EmitSubgroupLeMask(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=1u;", inst);
}

void EmitSubgroupGtMask(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=0xfffffffeu;", inst);
}

void EmitSubgroupGeMask(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=0xffffffffu;", inst);
}

void EmitShuffleIndex(EmitContext& ctx, IR::Inst& inst, std::string_view value,
                      std::string_view index, std::string_view clamp, std::string_view seg_mask) {
    Passthrough(ctx, inst, value);
}

void EmitShuffleUp(EmitContext& ctx, IR::Inst& inst, std::string_view value, std::string_view index,
                   std::string_view clamp, std::string_view seg_mask) {
    Passthrough(ctx, inst, value);
}

void EmitShuffleDown(EmitContext& ctx, IR::Inst& inst, std::string_view value,
                     std::string_view index, std::string_view clamp, std::string_view seg_mask) {
    Passthrough(ctx, inst, value);
}

void EmitShuffleButterfly(EmitContext& ctx, IR::Inst& inst, std::string_view value,
                          std::string_view index, std::string_view clamp,
                          std::string_view seg_mask) {
    Passthrough(ctx, inst, value);
}

void EmitQuadBroadcast(EmitContext& ctx, IR::Inst& inst, std::string_view value,
                       std::string_view lane) {
    ctx.AddU32("{}={};", inst, value);
}

void EmitQuadSwap(EmitContext& ctx, IR::Inst& inst, std::string_view value,
                  std::string_view direction) {
    ctx.AddU32("{}={};", inst, value);
}

void EmitFSwizzleAdd(EmitContext& ctx, IR::Inst& inst, std::string_view op_a, std::string_view op_b,
                     std::string_view swizzle) {
    const auto mask{fmt::format("({})&3u", swizzle)};
    ctx.AddF32("{}=({})*FSWZ_A[{}]+({})*FSWZ_B[{}];", inst, op_a, mask, op_b, mask);
}

void EmitDPdxFine(EmitContext& ctx, IR::Inst& inst, std::string_view op_a) {
    ctx.AddF32("{}=dpdxFine({});", inst, op_a);
}

void EmitDPdyFine(EmitContext& ctx, IR::Inst& inst, std::string_view op_a) {
    ctx.AddF32("{}=dpdyFine({});", inst, op_a);
}

void EmitDPdxCoarse(EmitContext& ctx, IR::Inst& inst, std::string_view op_a) {
    ctx.AddF32("{}=dpdxCoarse({});", inst, op_a);
}

void EmitDPdyCoarse(EmitContext& ctx, IR::Inst& inst, std::string_view op_a) {
    ctx.AddF32("{}=dpdyCoarse({});", inst, op_a);
}
} // namespace Shader::Backend::WGSL
