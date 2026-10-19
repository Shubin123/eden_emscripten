// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/profile.h"

namespace Shader::Backend::WGSL {
namespace {
void InitializeOutputVaryings(EmitContext& ctx) {
    if (ctx.stage == Stage::VertexB || ctx.stage == Stage::Geometry) {
        ctx.Add("v_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);");
    }
    for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
        if (ctx.info.stores.Generic(index)) {
            ctx.Add("out_attr{} = vec4<f32>(0.0, 0.0, 0.0, 1.0);", index);
        }
    }
}
} // Anonymous namespace

void EmitPhi(EmitContext& ctx, IR::Inst& phi) {
    const size_t num_args{phi.NumArgs()};
    for (size_t i = 0; i < num_args; ++i) {
        ctx.var_alloc.Consume(phi.Arg(i));
    }
    if (!phi.Definition<Id>().is_valid) {
        // The phi node wasn't forward defined
        ctx.var_alloc.PhiDefine(phi, phi.Type());
    }
}

void EmitVoid(EmitContext&) {}

void EmitReference(EmitContext& ctx, const IR::Value& value) {
    ctx.var_alloc.Consume(value);
}

void EmitPhiMove(EmitContext& ctx, const IR::Value& phi_value, const IR::Value& value) {
    IR::Inst& phi{*phi_value.InstRecursive()};
    const auto phi_type{phi.Type()};
    if (!phi.Definition<Id>().is_valid) {
        // The phi node wasn't forward defined
        ctx.var_alloc.PhiDefine(phi, phi_type);
    }
    const auto phi_reg{ctx.var_alloc.Consume(IR::Value{&phi})};
    const auto val_reg{ctx.var_alloc.Consume(value)};
    if (phi_reg == val_reg) {
        return;
    }
    ctx.Add("{}={};", phi_reg, val_reg);
}

void EmitPrologue(EmitContext& ctx) {
    InitializeOutputVaryings(ctx);
}

void EmitEpilogue(EmitContext&) {}

void EmitEmitVertex(EmitContext& ctx, const IR::Value& stream) {
    // WebGPU has no geometry stage.
    ctx.var_alloc.Consume(stream);
}

void EmitEndPrimitive(EmitContext& ctx, const IR::Value& stream) {
    ctx.var_alloc.Consume(stream);
}

} // namespace Shader::Backend::WGSL
