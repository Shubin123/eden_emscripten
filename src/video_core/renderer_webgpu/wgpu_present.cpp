// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>

#include "common/logging.h"
#include "video_core/renderer_webgpu/wgpu_present.h"
#include "video_core/renderer_webgpu/wgpu_thread.h"
#include "video_core/textures/decoders.h"

namespace WebGPU {

namespace {
constexpr u32 MAX_DRAWS_PER_FRAME = 8;
constexpr u32 UNIFORM_STRIDE = 256;

constexpr char PRESENT_SHADER[] = R"(
struct Params { rect: vec4<f32>, crop: vec4<f32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> VSOut {
    let corner = vec2<f32>(f32(i & 1u), f32(i >> 1u));
    var out: VSOut;
    let xy = params.rect.xy + corner * params.rect.zw;
    out.pos = vec4<f32>(xy.x * 2.0 - 1.0, 1.0 - xy.y * 2.0, 0.0, 1.0);
    out.uv = params.crop.xy + corner * params.crop.zw;
    return out;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(tex, samp, in.uv).rgb, 1.0);
}
)";

u32 draw_index = 0;
} // Anonymous namespace

Presenter::Presenter(Tegra::MaxwellDeviceMemoryManager& device_memory_)
    : device_memory{device_memory_} {
    CreatePipeline();
}

void Presenter::CreatePipeline() {
    auto& gpu = Thread::Get();
    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = PRESENT_SHADER;
    wgpu::ShaderModuleDescriptor module_desc{};
    module_desc.nextInChain = &wgsl;
    const wgpu::ShaderModule module = gpu.device.CreateShaderModule(&module_desc);

    std::array<wgpu::BindGroupLayoutEntry, 3> entries{};
    entries[0].binding = 0;
    entries[0].visibility = wgpu::ShaderStage::Vertex;
    entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    entries[0].buffer.hasDynamicOffset = true;
    entries[0].buffer.minBindingSize = 32;
    entries[1].binding = 1;
    entries[1].visibility = wgpu::ShaderStage::Fragment;
    entries[1].texture.sampleType = wgpu::TextureSampleType::Float;
    entries[1].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    entries[2].binding = 2;
    entries[2].visibility = wgpu::ShaderStage::Fragment;
    entries[2].sampler.type = wgpu::SamplerBindingType::Filtering;
    wgpu::BindGroupLayoutDescriptor layout_desc{};
    layout_desc.entryCount = entries.size();
    layout_desc.entries = entries.data();
    bind_group_layout = gpu.device.CreateBindGroupLayout(&layout_desc);

    wgpu::PipelineLayoutDescriptor pipeline_layout_desc{};
    pipeline_layout_desc.bindGroupLayoutCount = 1;
    pipeline_layout_desc.bindGroupLayouts = &bind_group_layout;
    const wgpu::PipelineLayout pipeline_layout =
        gpu.device.CreatePipelineLayout(&pipeline_layout_desc);

    wgpu::ColorTargetState target{};
    target.format = gpu.surface_format;
    wgpu::FragmentState fragment{};
    fragment.module = module;
    fragment.entryPoint = "fs";
    fragment.targetCount = 1;
    fragment.targets = &target;
    wgpu::RenderPipelineDescriptor desc{};
    desc.layout = pipeline_layout;
    desc.vertex.module = module;
    desc.vertex.entryPoint = "vs";
    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleStrip;
    desc.fragment = &fragment;
    pipeline = gpu.device.CreateRenderPipeline(&desc);

    wgpu::SamplerDescriptor sampler_desc{};
    sampler_desc.magFilter = wgpu::FilterMode::Linear;
    sampler_desc.minFilter = wgpu::FilterMode::Linear;
    sampler = gpu.device.CreateSampler(&sampler_desc);

    wgpu::BufferDescriptor buffer_desc{};
    buffer_desc.size = UNIFORM_STRIDE * MAX_DRAWS_PER_FRAME;
    buffer_desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniforms = gpu.device.CreateBuffer(&buffer_desc);
}

bool Presenter::BeginFrame(wgpu::CommandEncoder&) {
    auto& gpu = Thread::Get();
    gpu.surface.GetCurrentTexture(&surface_texture);
    if (surface_texture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
        surface_texture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
        static u32 failures = 0;
        if (failures++ < 5) {
            LOG_ERROR(Render, "WebGPU surface texture unavailable: status {}",
                      static_cast<u32>(surface_texture.status));
        }
        return false;
    }
    surface_view = surface_texture.texture.CreateView();
    cleared = false;
    draw_index = 0;
    return true;
}

void Presenter::Draw(wgpu::CommandEncoder& encoder, const wgpu::TextureView& view, u32 width,
                     u32 height, const Common::Rectangle<f32>& crop) {
    static u32 draw_logs = 0;
    if (draw_logs++ < 30) {
        LOG_INFO(Render, "Presenter::Draw #{}: surf={} w={} h={} crop=[{},{},{},{}]",
                 draw_logs, surface_view != nullptr, width, height, crop.left, crop.top, crop.right, crop.bottom);
    }
    if (!surface_view || draw_index >= MAX_DRAWS_PER_FRAME || width == 0 || height == 0) {
        return;
    }
    auto& gpu = Thread::Get();
    const f32 crop_width = crop.right - crop.left;
    const f32 crop_height = crop.bottom - crop.top;
    const f32 src_aspect = (static_cast<f32>(width) * std::abs(crop_width)) /
                           (static_cast<f32>(height) * std::abs(crop_height));
    const f32 dst_aspect =
        static_cast<f32>(gpu.surface_width) / static_cast<f32>(gpu.surface_height);
    f32 w = 1.0f, h = 1.0f;
    if (src_aspect > dst_aspect) {
        h = dst_aspect / src_aspect;
    } else {
        w = src_aspect / dst_aspect;
    }
    const std::array<f32, 8> params{(1.0f - w) * 0.5f, (1.0f - h) * 0.5f, w, h,
                                    crop.left, crop.top, crop_width, crop_height};
    const u32 offset = draw_index * UNIFORM_STRIDE;
    gpu.queue.WriteBuffer(uniforms, offset, params.data(), sizeof(params));

    std::array<wgpu::BindGroupEntry, 3> entries{};
    entries[0].binding = 0;
    entries[0].buffer = uniforms;
    entries[0].size = sizeof(params);
    entries[1].binding = 1;
    entries[1].textureView = view;
    entries[2].binding = 2;
    entries[2].sampler = sampler;
    wgpu::BindGroupDescriptor bind_desc{};
    bind_desc.layout = bind_group_layout;
    bind_desc.entryCount = entries.size();
    bind_desc.entries = entries.data();
    const wgpu::BindGroup bind_group = gpu.device.CreateBindGroup(&bind_desc);

    wgpu::RenderPassColorAttachment attachment{};
    attachment.view = surface_view;
    attachment.loadOp = cleared ? wgpu::LoadOp::Load : wgpu::LoadOp::Clear;
    attachment.storeOp = wgpu::StoreOp::Store;
    attachment.clearValue = {0.02, 0.02, 0.08, 1.0};
    wgpu::RenderPassDescriptor pass_desc{};
    pass_desc.colorAttachmentCount = 1;
    pass_desc.colorAttachments = &attachment;
    const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&pass_desc);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group, 1, &offset);
    pass.Draw(4);
    pass.End();
    cleared = true;
    ++draw_index;
}

wgpu::TextureView Presenter::LoadFromMemory(const Tegra::FramebufferConfig& framebuffer) {
    auto& gpu = Thread::Get();
    const u32 width = framebuffer.width;
    const u32 height = framebuffer.height;
    if (width == 0 || height == 0) {
        return {};
    }
    if (!memory_texture || memory_width != width || memory_height != height) {
        wgpu::TextureDescriptor desc{};
        desc.size = {width, height, 1};
        desc.format = wgpu::TextureFormat::RGBA8Unorm;
        desc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
        memory_texture = gpu.device.CreateTexture(&desc);
        memory_width = width;
        memory_height = height;
    }
    using Service::android::PixelFormat;
    const bool is_565 = framebuffer.pixel_format == PixelFormat::Rgb565;
    const u32 bytes_per_pixel = is_565 ? 2 : 4;
    constexpr u32 block_height_log2 = 4;
    const size_t size = Tegra::Texture::CalculateSize(true, bytes_per_pixel, framebuffer.stride,
                                                      height, 1, block_height_log2, 0);
    const u8* const host_ptr = device_memory.GetPointer<u8>(framebuffer.address + framebuffer.offset);
    static u32 mem_logs = 0;
    if (mem_logs++ < 10) {
        size_t nz = 0;
        if (host_ptr) {
            for (size_t i = 0; i < std::min(size, (size_t)10000); ++i) if (host_ptr[i] != 0) ++nz;
        }
        LOG_INFO(Render, "Presenter::LoadFromMemory #{}: addr={:#x} size={} host_ptr={} sample_nonzero={}",
                 mem_logs, framebuffer.address + framebuffer.offset, size, host_ptr != nullptr, nz);
    }
    if (!host_ptr) {
        return memory_texture.CreateView();
    }
    unswizzled.resize(static_cast<size_t>(framebuffer.stride) * height * bytes_per_pixel);
    Tegra::Texture::UnswizzleTexture(unswizzled, std::span<const u8>(host_ptr, size),
                                     bytes_per_pixel, framebuffer.stride, height, 1,
                                     block_height_log2, 0);
    std::vector<u8> rgba(static_cast<size_t>(width) * height * 4);
    for (u32 y = 0; y < height; ++y) {
        const u8* src = unswizzled.data() + static_cast<size_t>(y) * framebuffer.stride * bytes_per_pixel;
        u8* dst = rgba.data() + static_cast<size_t>(y) * width * 4;
        for (u32 x = 0; x < width; ++x, dst += 4) {
            if (is_565) {
                const u16 p = static_cast<u16>(src[x * 2] | (src[x * 2 + 1] << 8));
                dst[0] = static_cast<u8>(((p >> 11) & 0x1F) * 255 / 31);
                dst[1] = static_cast<u8>(((p >> 5) & 0x3F) * 255 / 63);
                dst[2] = static_cast<u8>((p & 0x1F) * 255 / 31);
            } else if (framebuffer.pixel_format == PixelFormat::Bgra8888) {
                dst[0] = src[x * 4 + 2];
                dst[1] = src[x * 4 + 1];
                dst[2] = src[x * 4 + 0];
            } else {
                dst[0] = src[x * 4 + 0];
                dst[1] = src[x * 4 + 1];
                dst[2] = src[x * 4 + 2];
            }
            dst[3] = 255;
        }
    }
    wgpu::TexelCopyTextureInfo dst{};
    dst.texture = memory_texture;
    wgpu::TexelCopyBufferLayout layout{};
    layout.bytesPerRow = width * 4;
    layout.rowsPerImage = height;
    const wgpu::Extent3D extent{width, height, 1};
    gpu.queue.WriteTexture(&dst, rgba.data(), rgba.size(), &layout, &extent);
    return memory_texture.CreateView();
}

void Presenter::EndFrame() {
    surface_view = nullptr;
    surface_texture = {};
}

} // namespace WebGPU
