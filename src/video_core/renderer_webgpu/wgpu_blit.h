// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <map>
#include <tuple>

#include <webgpu/webgpu_cpp.h>

#include "common/common_types.h"
#include "video_core/renderer_webgpu/wgpu_scheduler.h"
#include "video_core/renderer_webgpu/wgpu_uniform_ring.h"
#include "video_core/texture_cache/types.h"

namespace WebGPU {

enum class TextureKind : u8;

/// Copies image regions with a draw, converting formats and scaling as needed.
class Blitter {
public:
    explicit Blitter(Scheduler& scheduler, UniformRing& uniforms);

    void Blit(const wgpu::TextureView& dst, wgpu::TextureFormat dst_format, u32 dst_width,
              u32 dst_height, const VideoCommon::Region2D& dst_region,
              const wgpu::TextureView& src, TextureKind src_kind, u32 src_width, u32 src_height,
              const VideoCommon::Region2D& src_region, bool linear);

private:
    const wgpu::RenderPipeline& Pipeline(wgpu::TextureFormat dst_format, TextureKind src_kind);

    Scheduler& scheduler;
    UniformRing& uniforms;
    wgpu::Sampler linear_sampler;
    wgpu::Sampler nearest_sampler;
    std::map<std::pair<wgpu::TextureFormat, TextureKind>, wgpu::RenderPipeline> pipelines;
    std::map<TextureKind, wgpu::BindGroupLayout> layouts;
};

} // namespace WebGPU
