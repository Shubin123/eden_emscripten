// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"

namespace Shader::Backend::WGSL {
namespace {
std::string Smem(EmitContext& ctx, std::string_view offset, u32 add = 0) {
    const auto element{add == 0 ? fmt::format("smem[({})>>2u]", offset)
                                : fmt::format("smem[(({})+{}u)>>2u]", offset, add)};
    return ctx.uses_shared_atomics ? fmt::format("atomicLoad(&{})", element) : element;
}

void SmemStore(EmitContext& ctx, std::string_view offset, u32 add, std::string_view value) {
    const auto element{add == 0 ? fmt::format("smem[({})>>2u]", offset)
                                : fmt::format("smem[(({})+{}u)>>2u]", offset, add)};
    if (ctx.uses_shared_atomics) {
        ctx.Add("atomicStore(&{},{});", element, value);
    } else {
        ctx.Add("{}={};", element, value);
    }
}

void SharedWritePartial(EmitContext& ctx, std::string_view offset, std::string_view value,
                        std::string_view bit_offset, u32 num_bits) {
    const auto element{fmt::format("smem[({})>>2u]", offset)};
    if (ctx.uses_shared_atomics) {
        ctx.Add("loop {{ let old_value = atomicLoad(&{}); let result = atomicCompareExchangeWeak(&{},"
                "old_value,insertBits(old_value,{},{},{}u)); if (result.exchanged) {{ break; }} }}",
                element, element, value, bit_offset, num_bits);
    } else {
        ctx.Add("{}=insertBits({},{},{},{}u);", element, element, value, bit_offset, num_bits);
    }
}
} // Anonymous namespace

void EmitLoadSharedU8(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32("{}=extractBits({},(({})%4u)*8u,8u);", inst, Smem(ctx, offset), offset);
}

void EmitLoadSharedS8(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32("{}=u32(extractBits(i32({}),(({})%4u)*8u,8u));", inst, Smem(ctx, offset), offset);
}

void EmitLoadSharedU16(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32("{}=extractBits({},((({})>>1u)%2u)*16u,16u);", inst, Smem(ctx, offset), offset);
}

void EmitLoadSharedS16(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32("{}=u32(extractBits(i32({}),((({})>>1u)%2u)*16u,16u));", inst, Smem(ctx, offset),
               offset);
}

void EmitLoadSharedU32(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32("{}={};", inst, Smem(ctx, offset));
}

void EmitLoadSharedU64(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32x2("{}=vec2<u32>({},{});", inst, Smem(ctx, offset), Smem(ctx, offset, 4));
}

void EmitLoadSharedU128(EmitContext& ctx, IR::Inst& inst, std::string_view offset) {
    ctx.AddU32x4("{}=vec4<u32>({},{},{},{});", inst, Smem(ctx, offset), Smem(ctx, offset, 4),
                 Smem(ctx, offset, 8), Smem(ctx, offset, 12));
}

void EmitWriteSharedU8(EmitContext& ctx, std::string_view offset, std::string_view value) {
    SharedWritePartial(ctx, offset, value, fmt::format("(({})%4u)*8u", offset), 8);
}

void EmitWriteSharedU16(EmitContext& ctx, std::string_view offset, std::string_view value) {
    SharedWritePartial(ctx, offset, value, fmt::format("((({})>>1u)%2u)*16u", offset), 16);
}

void EmitWriteSharedU32(EmitContext& ctx, std::string_view offset, std::string_view value) {
    SmemStore(ctx, offset, 0, value);
}

void EmitWriteSharedU64(EmitContext& ctx, std::string_view offset, std::string_view value) {
    SmemStore(ctx, offset, 0, fmt::format("{}.x", value));
    SmemStore(ctx, offset, 4, fmt::format("{}.y", value));
}

void EmitWriteSharedU128(EmitContext& ctx, std::string_view offset, std::string_view value) {
    SmemStore(ctx, offset, 0, fmt::format("{}.x", value));
    SmemStore(ctx, offset, 4, fmt::format("{}.y", value));
    SmemStore(ctx, offset, 8, fmt::format("{}.z", value));
    SmemStore(ctx, offset, 12, fmt::format("{}.w", value));
}
} // namespace Shader::Backend::WGSL
