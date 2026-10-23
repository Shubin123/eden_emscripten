// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <string>
#include <vector>

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Backend::WGSL {

/// How a texture binding has to be declared; WebGPU bind group layouts are typed.
enum class TextureSampleKind : u8 {
    Float,
    UnfilterableFloat,
    Depth,
    Sint,
    Uint,
};

/// Pipeline state the WGSL backend needs beyond Shader::RuntimeInfo.
/// How a vertex attribute is read when its buffer is fetched by the shader instead of the vertex
/// stage (WebGPU has fewer vertex buffer slots than the guest).
struct PulledAttribute {
    enum class Kind : u8 { Unorm, Snorm, Uint, Sint, Float, Unorm1010102 };
    bool enabled{};
    u8 buffer{};     ///< Index of the pulled buffer
    u8 bits{};       ///< Bits per component: 8, 16 or 32
    u8 components{}; ///< Components read, 1 to 4
    Kind kind{};
    u32 offset{};    ///< Byte offset within each element
};

struct PulledBuffer {
    u32 stride{};
    bool per_instance{};
};

/// Bind group with the pulled vertex buffers: read-only storage at bindings 0 to
/// MAX_PULLED_BUFFERS - 1, and at MAX_PULLED_BUFFERS a uniform array<vec4<u32>, 2> with each
/// buffer's byte offset from its binding's start.
constexpr u32 PULLED_VERTEX_GROUP = 2;
constexpr u32 MAX_PULLED_BUFFERS = 8;

struct Options {
    /// Kind of each texture descriptor element, in descriptor order.
    std::vector<TextureSampleKind> texture_kinds;
    /// Interpolation the next stage declares for each generic (vertex outputs must match).
    std::array<Interpolation, 32> output_interpolation{};
    /// Components the vertex format provides per attribute; zero when the pipeline provides no
    /// data for it (the shader then reads the default (0, 0, 0, 1)).
    std::array<u8, 32> attribute_components{};
    /// Attributes in signed A2B10G10R10, which WebGPU cannot fetch: they are read as one u32 and
    /// unpacked in the shader.
    std::array<bool, 32> attribute_snorm_1010102{};
    /// Whether a depth attachment exists; fragment depth writes are dropped otherwise.
    bool has_depth_target = true;
    /// Bind group holding the per-draw info uniform.
    u32 draw_info_group = 1;
    /// Storage buffer bindings the stage may use; zero for no limit. Past it, the storage buffers
    /// from index max_storage_buffers - 1 on share one binding (see PackedSsboFirst).
    u32 max_storage_buffers = 0;
    /// Samplers the stage may use; zero for no limit. Past it, the textures from element
    /// max_samplers - 1 on share that element's sampler (see SharedSamplerFirst).
    u32 max_samplers = 0;
    /// Emit barriers inside control flow. WGSL accepts them only where control flow is provably
    /// uniform, so callers retry without them when the shader fails to compile.
    bool nested_barriers = false;
    /// Vertex attributes read from pulled buffers, and those buffers.
    std::array<PulledAttribute, 32> pulled_attributes{};
    std::vector<PulledBuffer> pulled_buffers;
};

/// Index of the first texture element whose sampler is shared, or `count` when every texture
/// element has its own. The shared sampler is the first shared element's (non-filtering, or
/// comparison for depth compare reads); the other shared elements' sampler bindings are unused.
[[nodiscard]] constexpr u32 SharedSamplerFirst(u32 count, u32 max_samplers) {
    if (max_samplers < 2 || count <= max_samplers) {
        return count;
    }
    return max_samplers - 1;
}

/// Maximum number of storage buffers that can share the packed binding.
constexpr u32 MAX_PACKED_SSBOS = 16;

/// Index of the first storage buffer read through the packed binding, or `count` when all storage
/// buffers get their own binding. The packed binding takes the first packed buffer's binding
/// number and is followed by a uniform array<vec4<u32>, 4> holding each packed buffer's base, in
/// 32-bit words; the bindings of the remaining packed buffers are left unused.
[[nodiscard]] constexpr u32 PackedSsboFirst(u32 count, u32 max_storage_buffers) {
    if (max_storage_buffers < 2 || count <= max_storage_buffers) {
        return count;
    }
    return max_storage_buffers - 1;
}

/// Layout of the per-draw uniform (group `draw_info_group`, binding 0).
struct DrawInfo {
    /// Clip space remap from the guest viewport to the full render target: x*[0]+w*[1], y*[2]+w*[3]
    std::array<f32, 4> viewport;
    std::array<f32, 4> render_area;
    u32 base_vertex;
    u32 base_instance;
    u32 depth_flip; ///< 1 maps z to w - z (reversed guest depth ranges)
    u32 padding;
};

[[nodiscard]] std::string EmitWGSL(const Profile& profile, const RuntimeInfo& runtime_info,
                                   IR::Program& program, Bindings& bindings,
                                   const Options& options);

} // namespace Shader::Backend::WGSL
