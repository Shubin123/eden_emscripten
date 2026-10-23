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
namespace {
std::string Ssbo(EmitContext& ctx, const IR::Value& binding, std::string_view word) {
    return ctx.SsboElement(binding.U32(), word);
}

std::string SsboLoad(EmitContext& ctx, const IR::Value& binding, std::string_view word) {
    const auto element{Ssbo(ctx, binding, word)};
    return ctx.uses_storage_atomics ? fmt::format("atomicLoad(&{})", element) : element;
}

void SsboStore(EmitContext& ctx, const IR::Value& binding, std::string_view word,
               std::string_view value) {
    if (ctx.storage_read_only) {
        return;
    }
    const auto element{Ssbo(ctx, binding, word)};
    if (ctx.uses_storage_atomics) {
        ctx.Add("atomicStore(&{},{});", element, value);
    } else {
        ctx.Add("{}={};", element, value);
    }
}

std::string Word(std::string_view offset, u32 add = 0) {
    if (add == 0) {
        return fmt::format("({})>>2u", offset);
    }
    return fmt::format("(({})+{}u)>>2u", offset, add);
}

void SsboWritePartial(EmitContext& ctx, const IR::Value& binding, std::string_view offset,
                      std::string_view value, std::string_view bit_offset, u32 num_bits) {
    if (ctx.storage_read_only) {
        return;
    }
    const auto element{Ssbo(ctx, binding, Word(offset))};
    if (ctx.uses_storage_atomics) {
        ctx.Add("loop {{ let old_value = atomicLoad(&{}); let result = atomicCompareExchangeWeak(&{},"
                "old_value,insertBits(old_value,{},{},{}u)); if (result.exchanged) {{ break; }} }}",
                element, element, value, bit_offset, num_bits);
    } else {
        ctx.Add("{}=insertBits({},{},{},{}u);", element, element, value, bit_offset, num_bits);
    }
}
} // Anonymous namespace

void EmitLoadGlobalU8(EmitContext&) {
    NotImplemented();
}

void EmitLoadGlobalS8(EmitContext&) {
    NotImplemented();
}

void EmitLoadGlobalU16(EmitContext&) {
    NotImplemented();
}

void EmitLoadGlobalS16(EmitContext&) {
    NotImplemented();
}

void EmitLoadGlobal32(EmitContext& ctx, IR::Inst& inst, std::string_view) {
    // Without 64-bit integers, global memory not lowered to storage buffers is unreachable.
    ctx.AddU32("{}=0u;", inst);
}

void EmitLoadGlobal64(EmitContext& ctx, IR::Inst& inst, std::string_view) {
    ctx.AddU32x2("{}=vec2<u32>(0u);", inst);
}

void EmitLoadGlobal128(EmitContext& ctx, IR::Inst& inst, std::string_view) {
    ctx.AddU32x4("{}=vec4<u32>(0u);", inst);
}

void EmitWriteGlobalU8(EmitContext&) {
    NotImplemented();
}

void EmitWriteGlobalS8(EmitContext&) {
    NotImplemented();
}

void EmitWriteGlobalU16(EmitContext&) {
    NotImplemented();
}

void EmitWriteGlobalS16(EmitContext&) {
    NotImplemented();
}

void EmitWriteGlobal32(EmitContext&, std::string_view, std::string_view) {}

void EmitWriteGlobal64(EmitContext&, std::string_view, std::string_view) {}

void EmitWriteGlobal128(EmitContext&, std::string_view, std::string_view) {}

void EmitLoadStorageU8(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                       const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32("{}=extractBits({},(({})%4u)*8u,8u);", inst, SsboLoad(ctx, binding, Word(offset_var)),
               offset_var);
}

void EmitLoadStorageS8(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                       const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32("{}=u32(extractBits(i32({}),(({})%4u)*8u,8u));", inst,
               SsboLoad(ctx, binding, Word(offset_var)), offset_var);
}

void EmitLoadStorageU16(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                        const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32("{}=extractBits({},((({})>>1u)%2u)*16u,16u);", inst,
               SsboLoad(ctx, binding, Word(offset_var)), offset_var);
}

void EmitLoadStorageS16(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                        const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32("{}=u32(extractBits(i32({}),((({})>>1u)%2u)*16u,16u));", inst,
               SsboLoad(ctx, binding, Word(offset_var)), offset_var);
}

void EmitLoadStorage32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                       const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32("{}={};", inst, SsboLoad(ctx, binding, Word(offset_var)));
}

void EmitLoadStorage64(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                       const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32x2("{}=vec2<u32>({},{});", inst, SsboLoad(ctx, binding, Word(offset_var)),
                 SsboLoad(ctx, binding, Word(offset_var, 4)));
}

void EmitLoadStorage128(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                        const IR::Value& offset) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    ctx.AddU32x4("{}=vec4<u32>({},{},{},{});", inst, SsboLoad(ctx, binding, Word(offset_var)),
                 SsboLoad(ctx, binding, Word(offset_var, 4)),
                 SsboLoad(ctx, binding, Word(offset_var, 8)),
                 SsboLoad(ctx, binding, Word(offset_var, 12)));
}

void EmitWriteStorageU8(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                        std::string_view value) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    SsboWritePartial(ctx, binding, offset_var, value, fmt::format("(({})%4u)*8u", offset_var), 8);
}

void EmitWriteStorageS8(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                        std::string_view value) {
    EmitWriteStorageU8(ctx, binding, offset, value);
}

void EmitWriteStorageU16(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                         std::string_view value) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    SsboWritePartial(ctx, binding, offset_var, value,
                     fmt::format("((({})>>1u)%2u)*16u", offset_var), 16);
}

void EmitWriteStorageS16(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                         std::string_view value) {
    EmitWriteStorageU16(ctx, binding, offset, value);
}

void EmitWriteStorage32(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                        std::string_view value) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    SsboStore(ctx, binding, Word(offset_var), value);
}

void EmitWriteStorage64(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                        std::string_view value) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    SsboStore(ctx, binding, Word(offset_var), fmt::format("{}.x", value));
    SsboStore(ctx, binding, Word(offset_var, 4), fmt::format("{}.y", value));
}

void EmitWriteStorage128(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                         std::string_view value) {
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    SsboStore(ctx, binding, Word(offset_var), fmt::format("{}.x", value));
    SsboStore(ctx, binding, Word(offset_var, 4), fmt::format("{}.y", value));
    SsboStore(ctx, binding, Word(offset_var, 8), fmt::format("{}.z", value));
    SsboStore(ctx, binding, Word(offset_var, 12), fmt::format("{}.w", value));
}
} // namespace Shader::Backend::WGSL
