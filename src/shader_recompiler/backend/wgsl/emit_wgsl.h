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
struct Options {
    /// Kind of each texture descriptor element, in descriptor order.
    std::vector<TextureSampleKind> texture_kinds;
    /// Interpolation the next stage declares for each generic (vertex outputs must match).
    std::array<Interpolation, 32> output_interpolation{};
    /// Components the vertex format provides per attribute; zero when the pipeline provides no
    /// data for it (the shader then reads the default (0, 0, 0, 1)).
    std::array<u8, 32> attribute_components{};
    /// Whether a depth attachment exists; fragment depth writes are dropped otherwise.
    bool has_depth_target = true;
    /// Bind group holding the per-draw info uniform.
    u32 draw_info_group = 1;
};

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
