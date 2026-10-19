// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <string_view>

#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/modifiers.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/profile.h"

namespace Shader::Backend::WGSL {
namespace {
struct TextureNames {
    std::string texture;
    std::string sampler;
    const TextureDefinition* def;
};

TextureNames Texture(EmitContext& ctx, const IR::TextureInstInfo& info, const IR::Value& index) {
    const TextureDefinition& def{ctx.textures.at(info.descriptor_index)};
    u32 element{0};
    if (def.count > 1) {
        if (index.IsImmediate()) {
            element = std::min(index.U32(), def.count - 1);
        } else {
            // Bindings of texture arrays are separate variables; dynamic indexing uses element 0.
            ctx.var_alloc.Consume(index);
        }
    }
    const u32 binding{def.binding + element * 2};
    return {fmt::format("tex{}", binding), fmt::format("samp{}", binding), &def};
}

bool IsDepthTexture(const TextureNames& names) {
    return names.def->is_depth || names.def->kind == TextureSampleKind::Depth;
}

bool IsIntegerTexture(const TextureNames& names) {
    return !names.def->is_depth &&
           (names.def->kind == TextureSampleKind::Sint || names.def->kind == TextureSampleKind::Uint);
}

/// Splits Maxwell coordinates into WGSL coordinates and an optional array index.
struct Coords {
    std::string coords;
    std::string array; ///< Empty, or ",i32(layer)"
};

Coords SampleCoords(TextureType type, std::string_view value) {
    switch (type) {
    case TextureType::Color1D:
        return {fmt::format("vec2<f32>({},0.5)", value), ""};
    case TextureType::ColorArray1D:
        return {fmt::format("vec2<f32>(({}).x,0.5)", value), fmt::format(",i32(({}).y)", value)};
    case TextureType::Color2D:
    case TextureType::Color2DRect:
    case TextureType::Color3D:
    case TextureType::ColorCube:
        return {std::string{value}, ""};
    case TextureType::ColorArray2D:
        return {fmt::format("({}).xy", value), fmt::format(",i32(({}).z)", value)};
    case TextureType::ColorArrayCube:
        return {fmt::format("({}).xyz", value), fmt::format(",i32(({}).w)", value)};
    case TextureType::Buffer:
        break;
    }
    return {std::string{value}, ""};
}

Coords FetchCoords(TextureType type, std::string_view value) {
    switch (type) {
    case TextureType::Color1D:
        return {fmt::format("vec2<i32>(i32({}),0)", value), ""};
    case TextureType::ColorArray1D:
        return {fmt::format("vec2<i32>(i32(({}).x),0)", value), fmt::format(",i32(({}).y)", value)};
    case TextureType::Color2D:
    case TextureType::Color2DRect:
        return {fmt::format("vec2<i32>({})", value), ""};
    case TextureType::ColorArray2D:
        return {fmt::format("vec2<i32>(({}).xy)", value), fmt::format(",i32(({}).z)", value)};
    case TextureType::Color3D:
    case TextureType::ColorCube:
        return {fmt::format("vec3<i32>({})", value), ""};
    case TextureType::ColorArrayCube:
        return {fmt::format("vec3<i32>(({}).xyz)", value), fmt::format(",i32(({}).w)", value)};
    case TextureType::Buffer:
        break;
    }
    return {std::string{value}, ""};
}

bool SupportsOffset(TextureType type) {
    return type != TextureType::ColorCube && type != TextureType::ColorArrayCube;
}

/// Constant texel offset, or empty when there is none (WGSL requires constant offsets).
std::string OffsetVec(EmitContext& ctx, TextureType type, const IR::Value& offset) {
    if (offset.IsEmpty() || !SupportsOffset(type)) {
        if (!offset.IsEmpty() && !offset.IsImmediate()) {
            ctx.var_alloc.Consume(offset);
        }
        return "";
    }
    const auto clamp{[](u32 value) { return std::clamp(static_cast<s32>(value), -8, 7); }};
    const bool is_3d{type == TextureType::Color3D};
    if (offset.IsImmediate()) {
        const s32 value{clamp(offset.U32())};
        return is_3d ? fmt::format(",vec3<i32>({},0,0)", value)
                     : fmt::format(",vec2<i32>({},0)", value);
    }
    IR::Inst* const inst{offset.InstRecursive()};
    if (inst->AreAllArgsImmediates()) {
        switch (inst->GetOpcode()) {
        case IR::Opcode::CompositeConstructU32x2:
            return is_3d ? fmt::format(",vec3<i32>({},{},0)", clamp(inst->Arg(0).U32()),
                                       clamp(inst->Arg(1).U32()))
                         : fmt::format(",vec2<i32>({},{})", clamp(inst->Arg(0).U32()),
                                       clamp(inst->Arg(1).U32()));
        case IR::Opcode::CompositeConstructU32x3:
            return fmt::format(",vec3<i32>({},{},{})", clamp(inst->Arg(0).U32()),
                               clamp(inst->Arg(1).U32()), clamp(inst->Arg(2).U32()));
        default:
            break;
        }
    }
    LOG_WARNING(Shader, "WGSL: dynamic texture offsets are not supported");
    ctx.var_alloc.Consume(offset);
    return "";
}

/// Offset applied to integer fetch coordinates (dynamic offsets allowed here).
std::string FetchOffset(EmitContext& ctx, TextureType type, const IR::Value& offset) {
    if (offset.IsEmpty()) {
        return "";
    }
    const auto value{ctx.var_alloc.Consume(offset)};
    switch (type) {
    case TextureType::Color1D:
    case TextureType::ColorArray1D:
        return fmt::format("+vec2<i32>(i32({}),0)", offset.Type() == IR::Type::U32 ? value : value + ".x");
    case TextureType::Color3D:
        return fmt::format("+vec3<i32>({})", value);
    default:
        return fmt::format("+vec2<i32>({})", value);
    }
}

void ResolveSparse(EmitContext& ctx, IR::Inst& inst) {
    if (IR::Inst* const sparse{inst.GetAssociatedPseudoOperation(IR::Opcode::GetSparseFromOp)}) {
        ctx.AddU1("{}=true;", *sparse);
        sparse->Invalidate();
    }
}

/// Converts a sampling result to the vec4<f32> the IR expects.
std::string ToTexel(const TextureNames& names, std::string_view sample) {
    if (IsDepthTexture(names)) {
        return fmt::format("vec4<f32>({},0.0,0.0,1.0)", sample);
    }
    if (IsIntegerTexture(names)) {
        return fmt::format("bitcast<vec4<f32>>({})", sample);
    }
    return std::string{sample};
}

/// Integer textures cannot be sampled; read the nearest texel of the base level instead.
std::string IntegerSample(const TextureNames& names, TextureType type, std::string_view coords) {
    const Coords c{SampleCoords(type, coords)};
    return fmt::format("bitcast<vec4<f32>>(textureLoad({},vec2<i32>(({})*vec2<f32>(textureDimensions({})))"
                       "{},0))",
                       names.texture, c.coords, names.texture, c.array);
}
} // Anonymous namespace

void EmitImageSampleImplicitLod(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                                std::string_view coords, std::string_view bias_lc,
                                const IR::Value& offset) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    const auto names{Texture(ctx, info, index)};
    const auto offset_str{OffsetVec(ctx, info.type, offset)};
    const auto texel{ctx.var_alloc.Define(inst, WgslVarType::F32x4)};
    ResolveSparse(ctx, inst);
    if (IsIntegerTexture(names)) {
        ctx.Add("{}={};", texel, IntegerSample(names, info.type, coords));
        return;
    }
    const Coords c{SampleCoords(info.type, coords)};
    std::string sample;
    if (ctx.stage != Stage::Fragment) {
        const auto level{IsDepthTexture(names) ? "0" : "0.0"};
        sample = fmt::format("textureSampleLevel({},{},{}{},{}{})", names.texture, names.sampler,
                             c.coords, c.array, level, offset_str);
    } else if (info.has_bias && !IsDepthTexture(names)) {
        sample = fmt::format("textureSampleBias({},{},{}{},{}{})", names.texture, names.sampler,
                             c.coords, c.array, bias_lc, offset_str);
    } else {
        sample = fmt::format("textureSample({},{},{}{}{})", names.texture, names.sampler, c.coords,
                             c.array, offset_str);
    }
    ctx.Add("{}={};", texel, ToTexel(names, sample));
}

void EmitImageSampleExplicitLod(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                                std::string_view coords, std::string_view lod_lc,
                                const IR::Value& offset) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    const auto names{Texture(ctx, info, index)};
    const auto offset_str{OffsetVec(ctx, info.type, offset)};
    const auto texel{ctx.var_alloc.Define(inst, WgslVarType::F32x4)};
    ResolveSparse(ctx, inst);
    if (IsIntegerTexture(names)) {
        ctx.Add("{}={};", texel, IntegerSample(names, info.type, coords));
        return;
    }
    const Coords c{SampleCoords(info.type, coords)};
    const auto level{IsDepthTexture(names) ? fmt::format("i32({})", lod_lc) : std::string{lod_lc}};
    const auto sample{fmt::format("textureSampleLevel({},{},{}{},{}{})", names.texture,
                                  names.sampler, c.coords, c.array, level, offset_str)};
    ctx.Add("{}={};", texel, ToTexel(names, sample));
}

void EmitImageSampleDrefImplicitLod(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                                    std::string_view coords, std::string_view dref,
                                    std::string_view bias_lc, const IR::Value& offset) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    const auto names{Texture(ctx, info, index)};
    const auto offset_str{OffsetVec(ctx, info.type, offset)};
    ResolveSparse(ctx, inst);
    const Coords c{SampleCoords(info.type, coords)};
    if (!names.def->is_depth) {
        ctx.AddF32("{}=1.0;", inst);
        return;
    }
    const auto function{ctx.stage == Stage::Fragment ? "textureSampleCompare"
                                                     : "textureSampleCompareLevel"};
    ctx.AddF32("{}={}({},{},{}{},{}{});", inst, function, names.texture, names.sampler, c.coords,
               c.array, dref, offset_str);
}

void EmitImageSampleDrefExplicitLod(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                                    std::string_view coords, std::string_view dref,
                                    std::string_view lod_lc, const IR::Value& offset) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    const auto names{Texture(ctx, info, index)};
    const auto offset_str{OffsetVec(ctx, info.type, offset)};
    ResolveSparse(ctx, inst);
    const Coords c{SampleCoords(info.type, coords)};
    if (!names.def->is_depth) {
        ctx.AddF32("{}=1.0;", inst);
        return;
    }
    // WGSL only compares against level zero.
    ctx.AddF32("{}=textureSampleCompareLevel({},{},{}{},{}{});", inst, names.texture,
               names.sampler, c.coords, c.array, dref, offset_str);
}

void EmitImageGather(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                     std::string_view coords, const IR::Value& offset, const IR::Value& offset2) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    const auto names{Texture(ctx, info, index)};
    const auto offset_str{OffsetVec(ctx, info.type, offset)};
    if (!offset2.IsEmpty() && !offset2.IsImmediate()) {
        ctx.var_alloc.Consume(offset2);
    }
    const auto texel{ctx.var_alloc.Define(inst, WgslVarType::F32x4)};
    ResolveSparse(ctx, inst);
    const Coords c{SampleCoords(info.type, coords)};
    if (IsIntegerTexture(names)) {
        ctx.Add("{}={};", texel, IntegerSample(names, info.type, coords));
        return;
    }
    if (IsDepthTexture(names)) {
        ctx.Add("{}=textureGather({},{},{}{}{});", texel, names.texture, names.sampler, c.coords,
                c.array, offset_str);
        return;
    }
    ctx.Add("{}=textureGather({},{},{},{}{}{});", texel, info.gather_component.Value(),
            names.texture, names.sampler, c.coords, c.array, offset_str);
}

void EmitImageGatherDref(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                         std::string_view coords, const IR::Value& offset, const IR::Value& offset2,
                         std::string_view dref) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    const auto names{Texture(ctx, info, index)};
    const auto offset_str{OffsetVec(ctx, info.type, offset)};
    if (!offset2.IsEmpty() && !offset2.IsImmediate()) {
        ctx.var_alloc.Consume(offset2);
    }
    ResolveSparse(ctx, inst);
    const Coords c{SampleCoords(info.type, coords)};
    if (!names.def->is_depth) {
        ctx.AddF32x4("{}=vec4<f32>(1.0);", inst);
        return;
    }
    ctx.AddF32x4("{}=textureGatherCompare({},{},{}{},{}{});", inst, names.texture, names.sampler,
                 c.coords, c.array, dref, offset_str);
}

void EmitImageFetch(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                    std::string_view coords, const IR::Value& offset, std::string_view lod,
                    std::string_view ms) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    if (info.type == TextureType::Buffer) {
        if (!offset.IsEmpty() && !offset.IsImmediate()) {
            ctx.var_alloc.Consume(offset);
        }
        ctx.AddF32x4("{}=vec4<f32>(0.0);", inst);
        ResolveSparse(ctx, inst);
        return;
    }
    const auto names{Texture(ctx, info, index)};
    const auto fetch_offset{FetchOffset(ctx, info.type, offset)};
    const auto texel{ctx.var_alloc.Define(inst, WgslVarType::F32x4)};
    ResolveSparse(ctx, inst);
    const Coords c{FetchCoords(info.type, coords)};
    const auto load{fmt::format("textureLoad({},{}{}{},i32({}))", names.texture, c.coords,
                                fetch_offset, c.array, lod.empty() ? "0u" : lod)};
    ctx.Add("{}={};", texel, ToTexel(names, load));
}

void EmitImageQueryDimensions(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                              std::string_view lod, const IR::Value& skip_mips_val) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    if (info.type == TextureType::Buffer) {
        ctx.AddU32x4("{}=vec4<u32>(0u);", inst);
        return;
    }
    const auto names{Texture(ctx, info, index)};
    const bool skip_mips{skip_mips_val.U1()};
    const auto mips{skip_mips ? "0u" : fmt::format("textureNumLevels({})", names.texture)};
    const auto size{fmt::format("textureDimensions({},u32({}))", names.texture, lod)};
    switch (info.type) {
    case TextureType::Color1D:
        return ctx.AddU32x4("{}=vec4<u32>({}.x,0u,0u,{});", inst, size, mips);
    case TextureType::Color2D:
    case TextureType::ColorCube:
    case TextureType::Color2DRect:
        return ctx.AddU32x4("{}=vec4<u32>({},0u,{});", inst, size, mips);
    case TextureType::ColorArray1D:
        return ctx.AddU32x4("{}=vec4<u32>({}.x,textureNumLayers({}),0u,{});", inst, size,
                            names.texture, mips);
    case TextureType::ColorArray2D:
    case TextureType::ColorArrayCube:
        return ctx.AddU32x4("{}=vec4<u32>({},textureNumLayers({}),{});", inst, size, names.texture,
                            mips);
    case TextureType::Color3D:
        return ctx.AddU32x4("{}=vec4<u32>({},{});", inst, size, mips);
    case TextureType::Buffer:
        break;
    }
    throw LogicError("Unspecified image type {}", info.type.Value());
}

void EmitImageQueryLod(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                       std::string_view coords) {
    // WGSL cannot query the level of detail.
    ctx.AddF32x4("{}=vec4<f32>(0.0);", inst);
}

void EmitImageGradient(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                       std::string_view coords, const IR::Value& derivatives,
                       const IR::Value& offset, [[maybe_unused]] const IR::Value& lod_clamp) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    const auto names{Texture(ctx, info, index)};
    const auto offset_str{info.num_derivatives >= 3 ? std::string{} : OffsetVec(ctx, info.type, offset)};
    const auto texel{ctx.var_alloc.Define(inst, WgslVarType::F32x4)};
    ResolveSparse(ctx, inst);
    const auto d{ctx.var_alloc.Consume(derivatives)};
    if (info.num_derivatives >= 3 && !offset.IsEmpty()) {
        const auto z{ctx.var_alloc.Consume(offset)};
        const Coords c{SampleCoords(info.type, coords)};
        ctx.Add("{}=textureSampleGrad({},{},{}{},vec3<f32>({}.xz,{}.x),vec3<f32>({}.yw,{}.y));",
                texel, names.texture, names.sampler, c.coords, c.array, d, z, d, z);
        return;
    }
    const Coords c{SampleCoords(info.type, coords)};
    if (IsDepthTexture(names) || IsIntegerTexture(names)) {
        ctx.Add("{}=vec4<f32>(0.0);", texel);
        return;
    }
    if (info.num_derivatives > 1) {
        ctx.Add("{}=textureSampleGrad({},{},{}{},{}.xz,{}.yw{});", texel, names.texture,
                names.sampler, c.coords, c.array, d, d, offset_str);
    } else {
        ctx.Add("{}=textureSampleGrad({},{},{}{},vec2<f32>({}.x,0.0),vec2<f32>({}.y,0.0){});", texel,
                names.texture, names.sampler, c.coords, c.array, d, d, offset_str);
    }
}

void EmitImageRead(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                   std::string_view coords) {
    // Storage images are not bound yet.
    ResolveSparse(ctx, inst);
    ctx.AddU32x4("{}=vec4<u32>(0u);", inst);
}

void EmitImageWrite(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,
                    std::string_view coords, std::string_view color) {}

#define IMAGE_ATOMIC(name)                                                                         \
    void EmitImageAtomic##name(EmitContext& ctx, IR::Inst& inst, const IR::Value& index,           \
                               std::string_view coords, std::string_view value) {                 \
        ctx.AddU32("{}=0u;", inst);                                                                \
    }
IMAGE_ATOMIC(IAdd32)
IMAGE_ATOMIC(SMin32)
IMAGE_ATOMIC(UMin32)
IMAGE_ATOMIC(SMax32)
IMAGE_ATOMIC(UMax32)
IMAGE_ATOMIC(Inc32)
IMAGE_ATOMIC(Dec32)
IMAGE_ATOMIC(And32)
IMAGE_ATOMIC(Or32)
IMAGE_ATOMIC(Xor32)
IMAGE_ATOMIC(Exchange32)
#undef IMAGE_ATOMIC

void EmitIsTextureScaled(EmitContext& ctx, IR::Inst& inst, const IR::Value& index) {
    ctx.AddU1("{}=false;", inst);
}

void EmitIsImageScaled(EmitContext& ctx, IR::Inst& inst, const IR::Value& index) {
    ctx.AddU1("{}=false;", inst);
}

void EmitBindlessImageSampleImplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageSampleExplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageSampleDrefImplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageSampleDrefExplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageGather(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageGatherDref(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageFetch(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageQueryDimensions(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageQueryLod(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageGradient(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageRead(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageWrite(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageSampleImplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageSampleExplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageSampleDrefImplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageSampleDrefExplicitLod(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageGather(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageGatherDref(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageFetch(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageQueryDimensions(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageQueryLod(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageGradient(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageRead(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageWrite(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicIAdd32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicSMin32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicUMin32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicSMax32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicUMax32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicInc32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicDec32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicAnd32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicOr32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicXor32(EmitContext&) {
    NotImplemented();
}

void EmitBindlessImageAtomicExchange32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicIAdd32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicSMin32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicUMin32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicSMax32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicUMax32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicInc32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicDec32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicAnd32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicOr32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicXor32(EmitContext&) {
    NotImplemented();
}

void EmitBoundImageAtomicExchange32(EmitContext&) {
    NotImplemented();
}

} // namespace Shader::Backend::WGSL
