// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "common/div_ceil.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Backend::WGSL {
namespace {

std::string_view InterpolationAttribute(Interpolation interp) {
    switch (interp) {
    case Interpolation::Smooth:
        return "@interpolate(perspective)";
    case Interpolation::Flat:
        return "@interpolate(flat)";
    case Interpolation::NoPerspective:
        return "@interpolate(linear)";
    }
    return "";
}

std::string_view TextureDimension(TextureType type) {
    switch (type) {
    case TextureType::Color1D:
    case TextureType::Color2D:
    case TextureType::Color2DRect:
        return "2d";
    case TextureType::ColorArray1D:
    case TextureType::ColorArray2D:
        return "2d_array";
    case TextureType::Color3D:
        return "3d";
    case TextureType::ColorCube:
        return "cube";
    case TextureType::ColorArrayCube:
        return "cube_array";
    case TextureType::Buffer:
        break;
    }
    return "2d";
}

bool UsesSharedAtomics(const IR::Program& program) {
    for (const IR::Block* const block : program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            const auto name{IR::NameOf(inst.GetOpcode())};
            if (name.starts_with("SharedAtomic")) {
                return true;
            }
        }
    }
    return false;
}

bool UsesStorageAtomics(const IR::Program& program) {
    for (const IR::Block* const block : program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            switch (inst.GetOpcode()) {
#define CASE(name) case IR::Opcode::name:
                CASE(StorageAtomicIAdd32) CASE(StorageAtomicSMin32) CASE(StorageAtomicUMin32)
                CASE(StorageAtomicSMax32) CASE(StorageAtomicUMax32) CASE(StorageAtomicInc32)
                CASE(StorageAtomicDec32) CASE(StorageAtomicAnd32) CASE(StorageAtomicOr32)
                CASE(StorageAtomicXor32) CASE(StorageAtomicExchange32) CASE(StorageAtomicIAdd64)
                CASE(StorageAtomicSMin64) CASE(StorageAtomicUMin64) CASE(StorageAtomicSMax64)
                CASE(StorageAtomicUMax64) CASE(StorageAtomicAnd64) CASE(StorageAtomicOr64)
                CASE(StorageAtomicXor64) CASE(StorageAtomicExchange64) CASE(StorageAtomicAddF32)
                CASE(StorageAtomicAddF16x2) CASE(StorageAtomicAddF32x2) CASE(StorageAtomicMinF16x2)
                CASE(StorageAtomicMinF32x2) CASE(StorageAtomicMaxF16x2) CASE(StorageAtomicMaxF32x2)
#undef CASE
                return true;
            default:
                break;
            }
        }
    }
    return false;
}

std::string_view AttributeInputType(AttributeType type) {
    switch (type) {
    case AttributeType::SignedInt:
    case AttributeType::SignedScaled:
        return "vec4<i32>";
    case AttributeType::UnsignedInt:
    case AttributeType::UnsignedScaled:
        return "vec4<u32>";
    default:
        return "vec4<f32>";
    }
}

std::string AttributeConversion(AttributeType type, std::string_view value) {
    switch (type) {
    case AttributeType::SignedInt:
    case AttributeType::UnsignedInt:
        return fmt::format("bitcast<vec4<f32>>({})", value);
    case AttributeType::SignedScaled:
    case AttributeType::UnsignedScaled:
        return fmt::format("vec4<f32>({})", value);
    default:
        return std::string{value};
    }
}

std::string_view ColorOutputType(AttributeType type) {
    switch (type) {
    case AttributeType::SignedInt:
        return "vec4<i32>";
    case AttributeType::UnsignedInt:
        return "vec4<u32>";
    default:
        return "vec4<f32>";
    }
}

} // Anonymous namespace

EmitContext::EmitContext(IR::Program& program, Bindings& bindings, const Profile& profile_,
                         const RuntimeInfo& runtime_info_, const Options& options_)
    : info{program.info}, profile{profile_}, runtime_info{runtime_info_}, options{options_},
      stage{program.stage} {
    switch (program.stage) {
    case Stage::VertexA:
    case Stage::VertexB:
        stage_name = "vs";
        storage_read_only = true;
        break;
    case Stage::TessellationControl:
        stage_name = "tcs";
        break;
    case Stage::TessellationEval:
        stage_name = "tes";
        break;
    case Stage::Geometry:
        stage_name = "gs";
        break;
    case Stage::Fragment:
        stage_name = "fs";
        break;
    case Stage::Compute:
        stage_name = "cs";
        break;
    }
    uses_storage_atomics = !storage_read_only && UsesStorageAtomics(program);
    uses_shared_atomics = UsesSharedAtomics(program);
    header += "diagnostic(off, derivative_uniformity);\n";
    header += "struct DrawInfo { viewport: vec4<f32>, render_area: vec4<f32>, base_vertex: u32, "
              "base_instance: u32, depth_flip: u32, padding: u32 }\n";
    if (stage != Stage::Compute) {
        header += fmt::format("@group({}) @binding(0) var<uniform> draw_info: DrawInfo;\n",
                              options.draw_info_group);
    }
    DefineConstantBuffers(bindings);
    DefineConstantBufferIndirect();
    DefineStorageBuffers(bindings);
    DefineTextures(bindings);
    DefineInterface();
    DefineHelperFunctions();
    if (program.shared_memory_size > 0) {
        header += fmt::format("var<workgroup> smem: array<{}, {}>;\n",
                              uses_shared_atomics ? "atomic<u32>" : "u32",
                              Common::DivCeil(program.shared_memory_size, 4U));
    }
    if (program.local_memory_size > 0) {
        header += fmt::format("var<private> lmem: array<u32, {}>;\n",
                              Common::DivCeil(program.local_memory_size, 4U));
    }
}

void EmitContext::DefineConstantBuffers(Bindings& bindings) {
    for (const auto& desc : info.constant_buffer_descriptors) {
        const u32 used_size{std::max(info.constant_buffer_used_sizes[desc.index], 16U)};
        const u32 num_vectors{std::min(Common::DivCeil(used_size, 16U), 4096U)};
        header += fmt::format("@group(0) @binding({}) var<uniform> {}_cbuf{}: array<vec4<u32>, {}>;\n",
                              bindings.unified, stage_name, desc.index, num_vectors);
        bindings.unified += desc.count;
    }
}

void EmitContext::DefineConstantBufferIndirect() {
    if (!info.uses_cbuf_indirect) {
        return;
    }
    header += "fn GetCbufIndirect(binding: u32, offset: u32) -> vec4<u32> {\nswitch (binding) {\n";
    for (const auto& desc : info.constant_buffer_descriptors) {
        const u32 used_size{std::max(info.constant_buffer_used_sizes[desc.index], 16U)};
        const u32 num_vectors{std::min(Common::DivCeil(used_size, 16U), 4096U)};
        header += fmt::format("case {}u: {{ return {}_cbuf{}[min(offset, {}u)]; }}\n", desc.index,
                              stage_name, desc.index, num_vectors - 1);
    }
    header += "default: { return vec4<u32>(0u); }\n}\n}\n";
}

void EmitContext::DefineStorageBuffers(Bindings& bindings) {
    u32 index{};
    for (const auto& desc : info.storage_buffers_descriptors) {
        const auto access{storage_read_only ? "read" : "read_write"};
        const auto element{uses_storage_atomics ? "atomic<u32>" : "u32"};
        header += fmt::format("@group(0) @binding({}) var<storage, {}> {}_ssbo{}: array<{}>;\n",
                              bindings.unified, access, stage_name, index, element);
        bindings.unified += desc.count;
        index += desc.count;
    }
}

void EmitContext::DefineTextures(Bindings& bindings) {
    // Texture buffers and images are not bound yet; their reads return zero.
    size_t kind_index{};
    for (const auto& desc : info.texture_descriptors) {
        const TextureSampleKind kind{kind_index < options.texture_kinds.size()
                                         ? options.texture_kinds[kind_index]
                                         : TextureSampleKind::Float};
        kind_index += desc.count;
        textures.push_back({bindings.unified, desc.count, desc.type, desc.is_depth, kind});
        for (u32 element = 0; element < desc.count; ++element) {
            const u32 binding{bindings.unified + element * 2};
            const auto dimension{TextureDimension(desc.type)};
            std::string type;
            if (desc.is_depth || kind == TextureSampleKind::Depth) {
                type = fmt::format("texture_depth_{}", dimension);
            } else if (kind == TextureSampleKind::Sint) {
                type = fmt::format("texture_{}<i32>", dimension);
            } else if (kind == TextureSampleKind::Uint) {
                type = fmt::format("texture_{}<u32>", dimension);
            } else {
                type = fmt::format("texture_{}<f32>", dimension);
            }
            const auto sampler{desc.is_depth ? "sampler_comparison" : "sampler"};
            header += fmt::format("@group(0) @binding({}) var tex{}: {};\n", binding, binding, type);
            header += fmt::format("@group(0) @binding({}) var samp{}: {};\n", binding + 1, binding,
                                  sampler);
        }
        bindings.unified += desc.count * 2;
    }
}

void EmitContext::DefineInterface() {
    switch (stage) {
    case Stage::VertexA:
    case Stage::VertexB:
        header += "var<private> v_position: vec4<f32>;\n"
                  "var<private> v_vertex_id: u32;\n"
                  "var<private> v_instance_id: u32;\n"
                  "var<private> v_point_size: f32;\n";
        break;
    case Stage::Fragment:
        header += "var<private> v_frag_coord: vec4<f32>;\n"
                  "var<private> v_front_facing: bool;\n"
                  "var<private> v_sample_id: u32;\n"
                  "var<private> v_frag_depth: f32;\n"
                  "var<private> v_sample_mask: u32;\n";
        for (size_t index = 0; index < info.stores_frag_color.size(); ++index) {
            if (info.stores_frag_color[index]) {
                header += fmt::format("var<private> frag_color{}: vec4<f32>;\n", index);
            }
        }
        break;
    case Stage::Compute:
        header += "var<private> v_local_id: vec3<u32>;\n"
                  "var<private> v_workgroup_id: vec3<u32>;\n";
        break;
    default:
        break;
    }
    for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
        if (info.loads.Generic(index)) {
            header += fmt::format("var<private> in_attr{}: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 1.0);\n",
                                  index);
        }
        if (info.stores.Generic(index)) {
            header += fmt::format("var<private> out_attr{}: vec4<f32>;\n", index);
        }
    }
}

void EmitContext::DefineHelperFunctions() {
    header += "fn fimm(v: u32) -> f32 { return bitcast<f32>(v); }\n"
              "fn ftou(v: f32) -> u32 { return bitcast<u32>(v); }\n"
              "fn utof(v: u32) -> f32 { return bitcast<f32>(v); }\n"
              "fn ftoi(v: f32) -> i32 { return bitcast<i32>(v); }\n"
              "fn itof(v: i32) -> f32 { return bitcast<f32>(v); }\n"
              "fn isnan_f32(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }\n"
              // Maxwell shifts of 32 or more produce zero (or the sign); WGSL masks the amount.
              "fn shl32(v: u32, s: u32) -> u32 { if (s >= 32u) { return 0u; } return v << s; }\n"
              "fn shr32(v: u32, s: u32) -> u32 { if (s >= 32u) { return 0u; } return v >> s; }\n"
              "fn sar32(v: u32, s: u32) -> u32 { return u32(i32(v) >> min(s, 31u)); }\n"
              "fn shl64(v: vec2<u32>, s: u32) -> vec2<u32> { if (s >= 64u) { return vec2<u32>(0u); }"
              " if (s >= 32u) { return vec2<u32>(0u, v.x << (s - 32u)); } if (s == 0u) { return v; }"
              " return vec2<u32>(v.x << s, (v.y << s) | (v.x >> (32u - s))); }\n"
              "fn shr64(v: vec2<u32>, s: u32) -> vec2<u32> { if (s >= 64u) { return vec2<u32>(0u); }"
              " if (s >= 32u) { return vec2<u32>(v.y >> (s - 32u), 0u); } if (s == 0u) { return v; }"
              " return vec2<u32>((v.x >> s) | (v.y << (32u - s)), v.y >> s); }\n"
              "fn sar64(v: vec2<u32>, s: u32) -> vec2<u32> { let hi = i32(v.y);"
              " if (s >= 32u) { return vec2<u32>(u32(hi >> min(s - 32u, 31u)), u32(hi >> 31u)); }"
              " if (s == 0u) { return v; }"
              " return vec2<u32>((v.x >> s) | (v.y << (32u - s)), u32(hi >> s)); }\n";
    header += "fn CasIncrement(op_a: u32, op_b: u32) -> u32 { return select(op_a + 1u, 0u, op_a >= op_b); }\n"
              "fn CasDecrement(op_a: u32, op_b: u32) -> u32 { "
              "return select(op_a - 1u, op_b, op_a == 0u || op_a > op_b); }\n"
              "fn CasMinS32(op_a: u32, op_b: u32) -> u32 { return u32(min(i32(op_a), i32(op_b))); }\n"
              "fn CasMaxS32(op_a: u32, op_b: u32) -> u32 { return u32(max(i32(op_a), i32(op_b))); }\n";
    if (info.uses_fswzadd) {
        header += "var<private> FSWZ_A: array<f32, 4> = array<f32, 4>(-1.0, 1.0, -1.0, 0.0);\n"
                  "var<private> FSWZ_B: array<f32, 4> = array<f32, 4>(-1.0, -1.0, 1.0, -1.0);\n";
    }
}

std::string EmitContext::Finish(const IR::Program& program) {
    std::string entry;
    switch (stage) {
    case Stage::VertexA:
    case Stage::VertexB: {
        entry += "struct VSInput {\n@builtin(vertex_index) vertex_index: u32,\n"
                 "@builtin(instance_index) instance_index: u32,\n";
        for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
            const AttributeType type{runtime_info.generic_input_types[index]};
            if (info.loads.Generic(index) && type != AttributeType::Disabled &&
                options.attribute_components[index] != 0) {
                entry += fmt::format("@location({}) attr{}: {},\n", index, index,
                                     AttributeInputType(type));
            }
        }
        entry += "}\nstruct VSOutput {\n@builtin(position) position: vec4<f32>,\n";
        for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
            if (info.stores.Generic(index)) {
                entry += fmt::format("@location({}) {} out{}: vec4<f32>,\n", index,
                                     InterpolationAttribute(options.output_interpolation[index]),
                                     index);
            }
        }
        entry += "}\n@vertex fn main(input: VSInput) -> VSOutput {\n"
                 "v_vertex_id = input.vertex_index;\nv_instance_id = input.instance_index;\n";
        for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
            const AttributeType type{runtime_info.generic_input_types[index]};
            if (!info.loads.Generic(index) || type == AttributeType::Disabled ||
                options.attribute_components[index] == 0) {
                continue;
            }
            entry += fmt::format("in_attr{} = {};\n", index,
                                 AttributeConversion(type, fmt::format("input.attr{}", index)));
            if (options.attribute_components[index] == 3) {
                entry += fmt::format("in_attr{}.w = 1.0;\n", index);
            }
        }
        entry += "main_body();\nvar output: VSOutput;\nvar pos = v_position;\n";
        if (runtime_info.convert_depth_mode) {
            entry += "pos.z = (pos.z + pos.w) * 0.5;\n";
        }
        entry += "if (draw_info.depth_flip != 0u) { pos.z = pos.w - pos.z; }\n"
                 "pos.x = pos.x * draw_info.viewport.x + pos.w * draw_info.viewport.y;\n"
                 "pos.y = pos.y * draw_info.viewport.z + pos.w * draw_info.viewport.w;\n"
                 "output.position = pos;\n";
        for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
            if (info.stores.Generic(index)) {
                entry += fmt::format("output.out{} = out_attr{};\n", index, index);
            }
        }
        entry += "return output;\n}\n";
        break;
    }
    case Stage::Fragment: {
        entry += "struct FSInput {\n@builtin(position) frag_coord: vec4<f32>,\n"
                 "@builtin(front_facing) front_facing: bool,\n";
        if (info.uses_sample_id) {
            entry += "@builtin(sample_index) sample_index: u32,\n";
        }
        for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
            if (info.loads.Generic(index) && runtime_info.previous_stage_stores.Generic(index)) {
                entry += fmt::format("@location({}) {} in{}: vec4<f32>,\n", index,
                                     InterpolationAttribute(info.interpolation[index]), index);
            }
        }
        entry += "}\nstruct FSOutput {\n";
        bool has_outputs{false};
        for (size_t index = 0; index < info.stores_frag_color.size(); ++index) {
            if (info.stores_frag_color[index]) {
                entry += fmt::format("@location({}) color{}: {},\n", index, index,
                                     ColorOutputType(runtime_info.color_output_types[index]));
                has_outputs = true;
            }
        }
        const bool writes_depth{info.stores_frag_depth && options.has_depth_target};
        if (writes_depth) {
            entry += "@builtin(frag_depth) frag_depth: f32,\n";
            has_outputs = true;
        }
        if (info.stores_sample_mask) {
            entry += "@builtin(sample_mask) sample_mask: u32,\n";
            has_outputs = true;
        }
        if (!has_outputs) {
            // Structures cannot be empty.
            entry += "@location(0) unused: vec4<f32>,\n";
        }
        entry += "}\n@fragment fn main(input: FSInput) -> FSOutput {\n"
                 "v_frag_coord = input.frag_coord;\nv_front_facing = input.front_facing;\n";
        if (info.uses_sample_id) {
            entry += "v_sample_id = input.sample_index;\n";
        }
        for (size_t index = 0; index < IR::NUM_GENERICS; ++index) {
            if (info.loads.Generic(index) && runtime_info.previous_stage_stores.Generic(index)) {
                entry += fmt::format("in_attr{} = input.in{};\n", index, index);
            }
        }
        entry += "main_body();\nvar output: FSOutput;\n";
        for (size_t index = 0; index < info.stores_frag_color.size(); ++index) {
            if (!info.stores_frag_color[index]) {
                continue;
            }
            switch (runtime_info.color_output_types[index]) {
            case AttributeType::SignedInt:
                entry += fmt::format("output.color{} = bitcast<vec4<i32>>(frag_color{});\n", index, index);
                break;
            case AttributeType::UnsignedInt:
                entry += fmt::format("output.color{} = bitcast<vec4<u32>>(frag_color{});\n", index, index);
                break;
            default:
                entry += fmt::format("output.color{} = frag_color{};\n", index, index);
                break;
            }
        }
        if (writes_depth) {
            entry += "output.frag_depth = v_frag_depth;\n";
        }
        if (info.stores_sample_mask) {
            entry += "output.sample_mask = v_sample_mask;\n";
        }
        entry += "return output;\n}\n";
        break;
    }
    case Stage::Compute: {
        const u32 x{std::max(program.workgroup_size[0], 1u)};
        const u32 y{std::max(program.workgroup_size[1], 1u)};
        const u32 z{std::max(program.workgroup_size[2], 1u)};
        entry += fmt::format("@compute @workgroup_size({}, {}, {}) fn main("
                             "@builtin(local_invocation_id) local_id: vec3<u32>, "
                             "@builtin(workgroup_id) workgroup_id: vec3<u32>) {{\n"
                             "v_local_id = local_id;\nv_workgroup_id = workgroup_id;\n"
                             "main_body();\n}}\n",
                             x, y, z);
        break;
    }
    default:
        break;
    }
    return header + code + entry;
}

} // namespace Shader::Backend::WGSL
