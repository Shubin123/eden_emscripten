// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include <webgpu/webgpu_cpp.h>

#include "common/common_types.h"
#include "common/math_util.h"
#include "video_core/framebuffer_config.h"
#include "video_core/host1x/gpu_device_memory_manager.h"

namespace WebGPU {

/// Draws guest display layers into the canvas surface.
class Presenter {
public:
    explicit Presenter(Tegra::MaxwellDeviceMemoryManager& device_memory);

    /// Starts a frame; returns false if the surface has no texture to draw into.
    bool BeginFrame(wgpu::CommandEncoder& encoder);

    /// Draws `view` (sampled with `crop`, in normalized texture coordinates) letterboxed into
    /// the frame. The first draw of a frame clears the surface.
    void Draw(wgpu::CommandEncoder& encoder, const wgpu::TextureView& view, u32 width, u32 height,
              const Common::Rectangle<f32>& crop);

    /// Loads a layer straight from guest memory, for frames not rendered by the GPU.
    wgpu::TextureView LoadFromMemory(const Tegra::FramebufferConfig& framebuffer);

    void EndFrame();

private:
    void CreatePipeline();

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    wgpu::RenderPipeline pipeline;
    wgpu::BindGroupLayout bind_group_layout;
    wgpu::Sampler sampler;
    wgpu::Buffer uniforms;
    wgpu::SurfaceTexture surface_texture;
    wgpu::TextureView surface_view;
    bool cleared = false;

    wgpu::Texture memory_texture;
    u32 memory_width = 0;
    u32 memory_height = 0;
    std::vector<u8> unswizzled;
};

} // namespace WebGPU
