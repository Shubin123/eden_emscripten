// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <string_view>

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Backend::WGSL {
namespace {
constexpr char SWIZZLE[]{"xyzw"};

/// Formats accesses to one constant buffer, consuming a dynamic binding exactly once.
class CbufAccess {
public:
    CbufAccess(EmitContext& ctx, const IR::Value& binding) {
        if (binding.IsImmediate()) {
            prefix = fmt::format("{}_cbuf{}[", ctx.stage_name, binding.U32());
            suffix = "]";
        } else {
            prefix = fmt::format("GetCbufIndirect({},", ctx.var_alloc.Consume(binding));
            suffix = ")";
        }
    }

    std::string operator()(std::string_view index) const {
        return prefix + std::string{index} + suffix;
    }

private:
    std::string prefix;
    std::string suffix;
};

std::string ChooseCbuf(EmitContext& ctx, const IR::Value& binding, std::string_view index) {
    return CbufAccess{ctx, binding}(index);
}

bool CbufInBounds(EmitContext& ctx, const IR::Value& binding, u32 offset) {
    if (!binding.IsImmediate()) {
        return true;
    }
    const u32 size{ctx.info.constant_buffer_used_sizes[binding.U32()]};
    return offset < std::max(size, 16U);
}

/// Returns the u32 word of a constant buffer at a byte offset.
std::string CbufWord(EmitContext& ctx, const IR::Value& binding, const IR::Value& offset,
                     u32 extra_bytes = 0) {
    if (offset.IsImmediate()) {
        const u32 byte_offset{offset.U32() + extra_bytes};
        if (!CbufInBounds(ctx, binding, byte_offset)) {
            return "0u";
        }
        return fmt::format("{}.{}", ChooseCbuf(ctx, binding, fmt::format("{}u", byte_offset / 16)),
                           SWIZZLE[(byte_offset / 4) % 4]);
    }
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    const auto byte_offset{extra_bytes ? fmt::format("(({})+{}u)", offset_var, extra_bytes)
                                       : fmt::format("({})", offset_var)};
    return fmt::format("{}[({}>>2u)%4u]", ChooseCbuf(ctx, binding, fmt::format("{}>>4u", byte_offset)),
                       byte_offset);
}

std::string BitOffset(const IR::Value& offset, std::string_view offset_var, u32 bits) {
    if (offset.IsImmediate()) {
        return fmt::format("{}u", bits == 8 ? (offset.U32() % 4) * 8 : ((offset.U32() / 2) % 2) * 16);
    }
    return bits == 8 ? fmt::format("(({})%4u)*8u", offset_var)
                     : fmt::format("((({})>>1u)%2u)*16u", offset_var);
}

void GetCbufSub(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                const IR::Value& offset, u32 bits, bool is_signed) {
    // The offset is consumed by CbufWord; read its name first for the bit offset.
    const std::string offset_name{offset.IsImmediate() ? "" : ctx.var_alloc.Consume(offset)};
    std::string word;
    if (offset.IsImmediate()) {
        word = CbufWord(ctx, binding, offset);
    } else {
        word = fmt::format("{}[(({})>>2u)%4u]",
                           ChooseCbuf(ctx, binding, fmt::format("({})>>4u", offset_name)), offset_name);
    }
    const auto bit_offset{BitOffset(offset, offset_name, bits)};
    if (is_signed) {
        ctx.AddU32("{}=u32(extractBits(i32({}),{},{}u));", inst, word, bit_offset, bits);
    } else {
        ctx.AddU32("{}=extractBits({},{},{}u);", inst, word, bit_offset, bits);
    }
}
} // Anonymous namespace

void EmitGetCbufU8(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                   const IR::Value& offset) {
    GetCbufSub(ctx, inst, binding, offset, 8, false);
}

void EmitGetCbufS8(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                   const IR::Value& offset) {
    GetCbufSub(ctx, inst, binding, offset, 8, true);
}

void EmitGetCbufU16(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset) {
    GetCbufSub(ctx, inst, binding, offset, 16, false);
}

void EmitGetCbufS16(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset) {
    GetCbufSub(ctx, inst, binding, offset, 16, true);
}

void EmitGetCbufU32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset) {
    ctx.AddU32("{}={};", inst, CbufWord(ctx, binding, offset));
}

void EmitGetCbufF32(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                    const IR::Value& offset) {
    ctx.AddF32("{}=bitcast<f32>({});", inst, CbufWord(ctx, binding, offset));
}

void EmitGetCbufU32x2(EmitContext& ctx, IR::Inst& inst, const IR::Value& binding,
                      const IR::Value& offset) {
    if (offset.IsImmediate()) {
        ctx.AddU32x2("{}=vec2<u32>({},{});", inst, CbufWord(ctx, binding, offset),
                     CbufWord(ctx, binding, offset, 4));
        return;
    }
    const auto offset_var{ctx.var_alloc.Consume(offset)};
    const CbufAccess cbuf{ctx, binding};
    const auto lo{fmt::format("{}[(({})>>2u)%4u]", cbuf(fmt::format("({})>>4u", offset_var)),
                              offset_var)};
    const auto hi{fmt::format("{}[((({})+4u)>>2u)%4u]", cbuf(fmt::format("(({})+4u)>>4u", offset_var)),
                              offset_var)};
    ctx.AddU32x2("{}=vec2<u32>({},{});", inst, lo, hi);
}

void EmitGetAttribute(EmitContext& ctx, IR::Inst& inst, IR::Attribute attr,
                      std::string_view vertex) {
    const u32 element{static_cast<u32>(attr) % 4};
    const char swizzle{SWIZZLE[element]};
    if (IR::IsGeneric(attr)) {
        const u32 index{IR::GenericAttributeIndex(attr)};
        if (ctx.stage == Stage::Fragment &&
            !ctx.runtime_info.previous_stage_stores.Generic(index, element)) {
            ctx.AddF32("{}={};", inst, element == 3 ? "1.0" : "0.0");
            return;
        }
        ctx.AddF32("{}=in_attr{}.{};", inst, index, swizzle);
        return;
    }
    switch (attr) {
    case IR::Attribute::PositionX:
    case IR::Attribute::PositionY:
    case IR::Attribute::PositionZ:
    case IR::Attribute::PositionW:
        if (ctx.stage == Stage::Fragment) {
            ctx.AddF32("{}=v_frag_coord.{};", inst, swizzle);
        } else {
            ctx.AddF32("{}=v_position.{};", inst, swizzle);
        }
        break;
    case IR::Attribute::InstanceId:
        ctx.AddF32("{}=bitcast<f32>(v_instance_id);", inst);
        break;
    case IR::Attribute::VertexId:
        ctx.AddF32("{}=bitcast<f32>(v_vertex_id);", inst);
        break;
    case IR::Attribute::FrontFace:
        ctx.AddF32("{}=bitcast<f32>(select(0u,0xffffffffu,v_front_facing));", inst);
        break;
    case IR::Attribute::BaseInstance:
        ctx.AddF32("{}=bitcast<f32>(draw_info.base_instance);", inst);
        break;
    case IR::Attribute::BaseVertex:
        ctx.AddF32("{}=bitcast<f32>(draw_info.base_vertex);", inst);
        break;
    case IR::Attribute::PrimitiveId:
    case IR::Attribute::Layer:
    case IR::Attribute::DrawID:
        ctx.AddF32("{}=0.0;", inst);
        break;
    case IR::Attribute::PointSpriteS:
    case IR::Attribute::PointSpriteT:
        // Points are always one pixel in WebGPU.
        ctx.AddF32("{}=0.5;", inst);
        break;
    default:
        throw NotImplementedException("Get attribute {}", attr);
    }
}

void EmitGetAttributeU32(EmitContext& ctx, IR::Inst& inst, IR::Attribute attr, std::string_view) {
    switch (attr) {
    case IR::Attribute::InstanceId:
        ctx.AddU32("{}=v_instance_id;", inst);
        break;
    case IR::Attribute::VertexId:
        ctx.AddU32("{}=v_vertex_id;", inst);
        break;
    case IR::Attribute::BaseInstance:
        ctx.AddU32("{}=draw_info.base_instance;", inst);
        break;
    case IR::Attribute::BaseVertex:
        ctx.AddU32("{}=draw_info.base_vertex;", inst);
        break;
    case IR::Attribute::PrimitiveId:
    case IR::Attribute::DrawID:
        ctx.AddU32("{}=0u;", inst);
        break;
    default:
        throw NotImplementedException("Get U32 attribute {}", attr);
    }
}

void EmitSetAttribute(EmitContext& ctx, IR::Attribute attr, std::string_view value,
                      [[maybe_unused]] std::string_view vertex) {
    if (IR::IsGeneric(attr)) {
        const u32 index{IR::GenericAttributeIndex(attr)};
        const u32 element{IR::GenericAttributeElement(attr)};
        ctx.Add("out_attr{}.{}={};", index, SWIZZLE[element], value);
        return;
    }
    const u32 element{static_cast<u32>(attr) % 4};
    switch (attr) {
    case IR::Attribute::PositionX:
    case IR::Attribute::PositionY:
    case IR::Attribute::PositionZ:
    case IR::Attribute::PositionW:
        ctx.Add("v_position.{}={};", SWIZZLE[element], value);
        break;
    case IR::Attribute::PointSize:
        ctx.Add("v_point_size={};", value);
        break;
    case IR::Attribute::Layer:
    case IR::Attribute::ViewportIndex:
    case IR::Attribute::ViewportMask:
    case IR::Attribute::ClipDistance0:
    case IR::Attribute::ClipDistance1:
    case IR::Attribute::ClipDistance2:
    case IR::Attribute::ClipDistance3:
    case IR::Attribute::ClipDistance4:
    case IR::Attribute::ClipDistance5:
    case IR::Attribute::ClipDistance6:
    case IR::Attribute::ClipDistance7:
        // Not expressible in WebGPU's vertex stage yet.
        break;
    default:
        throw NotImplementedException("Set attribute {}", attr);
    }
}

void EmitGetAttributeIndexed(EmitContext& ctx, IR::Inst& inst, std::string_view offset,
                             std::string_view vertex) {
    // Indexed attribute loads cover generics only.
    std::string expr{"0.0"};
    for (u32 index = 0; index < IR::NUM_GENERICS; ++index) {
        if (!ctx.info.loads.Generic(index)) {
            continue;
        }
        const u32 base{(static_cast<u32>(IR::Attribute::Generic0X) >> 2) + index};
        expr = fmt::format("select({},in_attr{}[(({})>>2u)&3u],(({})>>4u)=={}u)", expr, index, offset,
                           offset, base);
    }
    ctx.AddF32("{}={};", inst, expr);
}

void EmitSetAttributeIndexed([[maybe_unused]] EmitContext& ctx,
                             [[maybe_unused]] std::string_view offset,
                             [[maybe_unused]] std::string_view value,
                             [[maybe_unused]] std::string_view vertex) {
    NotImplemented();
}

void EmitGetPatch(EmitContext& ctx, IR::Inst& inst, IR::Patch patch) {
    NotImplemented();
}

void EmitSetPatch(EmitContext& ctx, IR::Patch patch, std::string_view value) {
    NotImplemented();
}

void EmitSetFragColor(EmitContext& ctx, u32 index, u32 component, std::string_view value) {
    if (ctx.info.stores_frag_color[index]) {
        ctx.Add("frag_color{}.{}={};", index, SWIZZLE[component], value);
    }
}

void EmitSetSampleMask(EmitContext& ctx, std::string_view value) {
    ctx.Add("v_sample_mask={};", value);
}

void EmitSetFragDepth(EmitContext& ctx, std::string_view value) {
    ctx.Add("v_frag_depth={};", value);
}

void EmitLocalInvocationId(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32x3("{}=v_local_id;", inst);
}

void EmitWorkgroupId(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32x3("{}=v_workgroup_id;", inst);
}

void EmitInvocationId(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=0u;", inst);
}

void EmitInvocationInfo(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=0x00ff0000u;", inst);
}

void EmitSampleId(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=v_sample_id;", inst);
}

void EmitIsHelperInvocation(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU1("{}=false;", inst);
}

void EmitSR_WScaleFactorXY(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=0u;", inst);
}

void EmitSR_WScaleFactorZ(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddU32("{}=0u;", inst);
}

void EmitYDirection(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddF32("{}=1.0;", inst);
}

void EmitResolutionDownFactor(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddF32("{}=1.0;", inst);
}

void EmitRenderArea(EmitContext& ctx, IR::Inst& inst) {
    ctx.AddF32x4("{}=draw_info.render_area;", inst);
}

void EmitLoadLocal(EmitContext& ctx, IR::Inst& inst, std::string_view word_offset) {
    ctx.AddU32("{}=lmem[{}];", inst, word_offset);
}

void EmitWriteLocal(EmitContext& ctx, std::string_view word_offset, std::string_view value) {
    ctx.Add("lmem[{}]={};", word_offset, value);
}

} // namespace Shader::Backend::WGSL
