// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string_view>

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::WGSL {
namespace {
std::string SharedPtr(std::string_view offset) {
    return fmt::format("&smem[({})>>2u]", offset);
}

std::string SsboPtr(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset) {
    return "&" + ctx.SsboElement(binding.U32(),
                                 fmt::format("({})>>2u", ctx.var_alloc.Consume(offset)));
}

/// Atomic read-modify-write through a compare-exchange loop; `function` maps (old, value).
void CasLoop(EmitContext& ctx, std::string_view ret, std::string_view ptr, std::string_view value,
             std::string_view function) {
    ctx.Add("loop {{ let old = atomicLoad({}); let r = atomicCompareExchangeWeak({},old,{}(old,{}));"
            " if (r.exchanged) {{ {}=old; break; }} }}",
            ptr, ptr, function, value, ret);
}

void Native(EmitContext& ctx, IR::Inst& inst, bool available, std::string_view ptr,
            std::string_view op, std::string_view value) {
    if (!available) {
        ctx.AddU32("{}=0u;", inst);
        return;
    }
    ctx.AddU32("{}={}({},{});", inst, op, ptr, value);
}

void Cas(EmitContext& ctx, IR::Inst& inst, bool available, std::string_view ptr,
         std::string_view value, std::string_view function) {
    const auto ret{ctx.var_alloc.Define(inst, WgslVarType::U32)};
    if (!available) {
        ctx.Add("{}=0u;", ret);
        return;
    }
    CasLoop(ctx, ret, ptr, value, function);
}
} // Anonymous namespace

void EmitSharedAtomicIAdd32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                    std::string_view value) {
    Native(ctx, inst, ctx.uses_shared_atomics, SharedPtr(pointer_offset), "atomicAdd", value);
}

void EmitStorageAtomicIAdd32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, ctx.uses_storage_atomics, SsboPtr(ctx, binding, offset), "atomicAdd", value);
}

void EmitSharedAtomicSMin32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                    std::string_view value) {
    Cas(ctx, inst, ctx.uses_shared_atomics, SharedPtr(pointer_offset), value, "CasMinS32");
}

void EmitStorageAtomicSMin32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    Cas(ctx, inst, ctx.uses_storage_atomics, SsboPtr(ctx, binding, offset), value, "CasMinS32");
}

void EmitSharedAtomicUMin32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                    std::string_view value) {
    Native(ctx, inst, ctx.uses_shared_atomics, SharedPtr(pointer_offset), "atomicMin", value);
}

void EmitStorageAtomicUMin32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, ctx.uses_storage_atomics, SsboPtr(ctx, binding, offset), "atomicMin", value);
}

void EmitSharedAtomicSMax32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                    std::string_view value) {
    Cas(ctx, inst, ctx.uses_shared_atomics, SharedPtr(pointer_offset), value, "CasMaxS32");
}

void EmitStorageAtomicSMax32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    Cas(ctx, inst, ctx.uses_storage_atomics, SsboPtr(ctx, binding, offset), value, "CasMaxS32");
}

void EmitSharedAtomicUMax32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                    std::string_view value) {
    Native(ctx, inst, ctx.uses_shared_atomics, SharedPtr(pointer_offset), "atomicMax", value);
}

void EmitStorageAtomicUMax32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, ctx.uses_storage_atomics, SsboPtr(ctx, binding, offset), "atomicMax", value);
}

void EmitSharedAtomicInc32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                    std::string_view value) {
    Cas(ctx, inst, ctx.uses_shared_atomics, SharedPtr(pointer_offset), value, "CasIncrement");
}

void EmitStorageAtomicInc32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    Cas(ctx, inst, ctx.uses_storage_atomics, SsboPtr(ctx, binding, offset), value, "CasIncrement");
}

void EmitSharedAtomicDec32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                    std::string_view value) {
    Cas(ctx, inst, ctx.uses_shared_atomics, SharedPtr(pointer_offset), value, "CasDecrement");
}

void EmitStorageAtomicDec32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    Cas(ctx, inst, ctx.uses_storage_atomics, SsboPtr(ctx, binding, offset), value, "CasDecrement");
}

void EmitSharedAtomicAnd32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                    std::string_view value) {
    Native(ctx, inst, ctx.uses_shared_atomics, SharedPtr(pointer_offset), "atomicAnd", value);
}

void EmitStorageAtomicAnd32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, ctx.uses_storage_atomics, SsboPtr(ctx, binding, offset), "atomicAnd", value);
}

void EmitSharedAtomicOr32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                    std::string_view value) {
    Native(ctx, inst, ctx.uses_shared_atomics, SharedPtr(pointer_offset), "atomicOr", value);
}

void EmitStorageAtomicOr32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, ctx.uses_storage_atomics, SsboPtr(ctx, binding, offset), "atomicOr", value);
}

void EmitSharedAtomicXor32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                    std::string_view value) {
    Native(ctx, inst, ctx.uses_shared_atomics, SharedPtr(pointer_offset), "atomicXor", value);
}

void EmitStorageAtomicXor32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, ctx.uses_storage_atomics, SsboPtr(ctx, binding, offset), "atomicXor", value);
}

void EmitSharedAtomicExchange32(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                    std::string_view value) {
    Native(ctx, inst, ctx.uses_shared_atomics, SharedPtr(pointer_offset), "atomicExchange", value);
}

void EmitStorageAtomicExchange32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    Native(ctx, inst, ctx.uses_storage_atomics, SsboPtr(ctx, binding, offset), "atomicExchange", value);
}

void EmitStorageAtomicIAdd64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    // No 64-bit atomics in WebGPU; reads the old value only.
    ctx.var_alloc.Consume(offset);
    ctx.AddU64("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicSMin64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    // No 64-bit atomics in WebGPU; reads the old value only.
    ctx.var_alloc.Consume(offset);
    ctx.AddU64("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicUMin64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    // No 64-bit atomics in WebGPU; reads the old value only.
    ctx.var_alloc.Consume(offset);
    ctx.AddU64("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicSMax64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    // No 64-bit atomics in WebGPU; reads the old value only.
    ctx.var_alloc.Consume(offset);
    ctx.AddU64("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicUMax64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    // No 64-bit atomics in WebGPU; reads the old value only.
    ctx.var_alloc.Consume(offset);
    ctx.AddU64("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicAnd64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    // No 64-bit atomics in WebGPU; reads the old value only.
    ctx.var_alloc.Consume(offset);
    ctx.AddU64("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicOr64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    // No 64-bit atomics in WebGPU; reads the old value only.
    ctx.var_alloc.Consume(offset);
    ctx.AddU64("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicXor64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    // No 64-bit atomics in WebGPU; reads the old value only.
    ctx.var_alloc.Consume(offset);
    ctx.AddU64("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicExchange64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    // No 64-bit atomics in WebGPU; reads the old value only.
    ctx.var_alloc.Consume(offset);
    ctx.AddU64("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicIAdd32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    ctx.var_alloc.Consume(offset);
    ctx.AddU32x2("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicSMin32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    ctx.var_alloc.Consume(offset);
    ctx.AddU32x2("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicUMin32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    ctx.var_alloc.Consume(offset);
    ctx.AddU32x2("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicSMax32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    ctx.var_alloc.Consume(offset);
    ctx.AddU32x2("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicUMax32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    ctx.var_alloc.Consume(offset);
    ctx.AddU32x2("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicAnd32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    ctx.var_alloc.Consume(offset);
    ctx.AddU32x2("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicOr32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    ctx.var_alloc.Consume(offset);
    ctx.AddU32x2("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicXor32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    ctx.var_alloc.Consume(offset);
    ctx.AddU32x2("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicExchange32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset, std::string_view value) {
    ctx.var_alloc.Consume(offset);
    ctx.AddU32x2("{}=vec2<u32>(0u);", inst);
}

void EmitSharedAtomicExchange64(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                                std::string_view value) {
    ctx.AddU64("{}=vec2<u32>(0u);", inst);
}

void EmitSharedAtomicExchange32x2(EmitContext& ctx, IR::Inst& inst, std::string_view pointer_offset,
                                  std::string_view value) {
    ctx.AddU32x2("{}=vec2<u32>(0u);", inst);
}

void EmitStorageAtomicAddF32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                             const IR::Value& offset, std::string_view value) {
    const auto ptr{SsboPtr(ctx, binding, offset)};
    const auto ret{ctx.var_alloc.Define(inst, WgslVarType::F32)};
    if (!ctx.uses_storage_atomics) {
        ctx.Add("{}=0.0;", ret);
        return;
    }
    ctx.Add("loop {{ let old = atomicLoad({}); let r = atomicCompareExchangeWeak({},old,"
            "bitcast<u32>(bitcast<f32>(old)+({}))); if (r.exchanged) {{ {}=bitcast<f32>(old); break; }} }}",
            ptr, ptr, value, ret);
}

void EmitStorageAtomicAddF16x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicAddF32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicMinF16x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicMinF32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicMaxF16x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitStorageAtomicMaxF32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                               const IR::Value& offset, std::string_view value) {
    NotImplemented();
}

void EmitGlobalAtomicIAdd32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicSMin32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicUMin32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicSMax32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicUMax32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicInc32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicDec32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicAnd32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicOr32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicXor32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicExchange32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicIAdd64(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicSMin64(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicUMin64(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicSMax64(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicUMax64(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicInc64(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicDec64(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicAnd64(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicOr64(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicXor64(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicExchange64(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicIAdd32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicSMin32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicUMin32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicSMax32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicUMax32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicInc32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicDec32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicAnd32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicOr32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicXor32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicExchange32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicAddF32(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicAddF16x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicAddF32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicMinF16x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicMinF32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicMaxF16x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}

void EmitGlobalAtomicMaxF32x2(EmitContext&) {
    throw NotImplementedException("GLSL Instruction");
}
} // namespace Shader::Backend::WGSL
