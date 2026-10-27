// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <string>

#include "common/logging.h"
#include "video_core/renderer_webgpu/wgpu_blit.h"
#include "video_core/renderer_webgpu/wgpu_formats.h"
#include "video_core/renderer_webgpu/wgpu_texture_cache.h"
#include "video_core/renderer_webgpu/wgpu_thread.h"

namespace WebGPU {

namespace {

constexpr char VERTEX[] = R"(
struct Params { uv: vec4<f32> }
@group(0) @binding(0) var<uniform> params: Params;
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> VSOut {
    let corner = vec2<f32>(f32(i & 1u), f32(i >> 1u));
    var out: VSOut;
    out.pos = vec4<f32>(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
    out.uv = params.uv.xy + corner * params.uv.zw;
    return out;
}
)";

bool IsDepthFormat(wgpu::TextureFormat format) {
    switch (format) {
    case wgpu::TextureFormat::Depth16Unorm:
    case wgpu::TextureFormat::Depth24Plus:
    case wgpu::TextureFormat::Depth24PlusStencil8:
    case wgpu::TextureFormat::Depth32Float:
    case wgpu::TextureFormat::Depth32FloatStencil8:
    case wgpu::TextureFormat::Stencil8:
        return true;
    default:
        return false;
    }
}

std::string FragmentShader(wgpu::TextureFormat dst_format, TextureKind kind) {
    const bool depth_dst = IsDepthFormat(dst_format);
    std::string code = VERTEX;
    switch (kind) {
    case TextureKind::Depth:
        code += "@group(0) @binding(1) var src: texture_depth_2d;\n";
        break;
    case TextureKind::Sint:
        code += "@group(0) @binding(1) var src: texture_2d<i32>;\n";
        break;
    case TextureKind::Uint:
        code += "@group(0) @binding(1) var src: texture_2d<u32>;\n";
        break;
    default:
        code += "@group(0) @binding(1) var src: texture_2d<f32>;\n";
        break;
    }
    code += "@group(0) @binding(2) var samp: sampler;\n";
    const bool loads = kind != TextureKind::Float;
    std::string fetch;
    if (loads) {
        code += "fn texel(uv: vec2<f32>) -> vec2<i32> { let size = vec2<f32>(textureDimensions(src)); "
                "return clamp(vec2<i32>(floor(uv * size)), vec2<i32>(0), vec2<i32>(size) - 1); }\n";
        fetch = "textureLoad(src, texel(in.uv), 0)";
    } else {
        fetch = "textureSampleLevel(src, samp, in.uv, 0.0)";
    }
    if (depth_dst) {
        const std::string value = kind == TextureKind::Depth ? fetch : fetch + ".r";
        code += "@fragment fn fs(in: VSOut) -> @builtin(frag_depth) f32 { return " + value + "; }\n";
    } else if (kind == TextureKind::Depth) {
        code += "@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> { return vec4<f32>(" + fetch +
                "); }\n";
    } else if (kind == TextureKind::Sint || kind == TextureKind::Uint) {
        const bool sint_dst = dst_format == wgpu::TextureFormat::R8Sint ||
                              dst_format == wgpu::TextureFormat::R16Sint ||
                              dst_format == wgpu::TextureFormat::R32Sint ||
                              dst_format == wgpu::TextureFormat::RG8Sint ||
                              dst_format == wgpu::TextureFormat::RG16Sint ||
                              dst_format == wgpu::TextureFormat::RG32Sint ||
                              dst_format == wgpu::TextureFormat::RGBA8Sint ||
                              dst_format == wgpu::TextureFormat::RGBA16Sint ||
                              dst_format == wgpu::TextureFormat::RGBA32Sint;
        const std::string type = sint_dst ? "vec4<i32>" : "vec4<u32>";
        code += "@fragment fn fs(in: VSOut) -> @location(0) " + type + " { return " + type + "(" +
                fetch + "); }\n";
    } else {
        code += "@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> { return " + fetch + "; }\n";
    }
    return code;
}

} // Anonymous namespace

Blitter::Blitter(Scheduler& scheduler_, UniformRing& uniforms_)
    : scheduler{scheduler_}, uniforms{uniforms_} {
    auto& device = Thread::Get().device;
    wgpu::SamplerDescriptor desc{};
    desc.magFilter = wgpu::FilterMode::Linear;
    desc.minFilter = wgpu::FilterMode::Linear;
    linear_sampler = device.CreateSampler(&desc);
    desc.magFilter = wgpu::FilterMode::Nearest;
    desc.minFilter = wgpu::FilterMode::Nearest;
    nearest_sampler = device.CreateSampler(&desc);
}

const wgpu::RenderPipeline& Blitter::Pipeline(wgpu::TextureFormat dst_format, TextureKind kind) {
    const auto key = std::make_pair(dst_format, kind);
    if (const auto it = pipelines.find(key); it != pipelines.end()) {
        return it->second;
    }
    auto& device = Thread::Get().device;
    auto& layout = layouts[kind];
    if (!layout) {
        std::array<wgpu::BindGroupLayoutEntry, 3> entries{};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Vertex;
        entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[0].buffer.hasDynamicOffset = true;
        entries[0].buffer.minBindingSize = 16;
        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Fragment;
        switch (kind) {
        case TextureKind::Depth:
            entries[1].texture.sampleType = wgpu::TextureSampleType::Depth;
            break;
        case TextureKind::Sint:
            entries[1].texture.sampleType = wgpu::TextureSampleType::Sint;
            break;
        case TextureKind::Uint:
            entries[1].texture.sampleType = wgpu::TextureSampleType::Uint;
            break;
        case TextureKind::UnfilterableFloat:
            entries[1].texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;
            break;
        default:
            entries[1].texture.sampleType = wgpu::TextureSampleType::Float;
            break;
        }
        entries[1].texture.viewDimension = wgpu::TextureViewDimension::e2D;
        entries[2].binding = 2;
        entries[2].visibility = wgpu::ShaderStage::Fragment;
        entries[2].sampler.type = kind == TextureKind::Float ? wgpu::SamplerBindingType::Filtering
                                                             : wgpu::SamplerBindingType::NonFiltering;
        wgpu::BindGroupLayoutDescriptor layout_desc{};
        layout_desc.entryCount = entries.size();
        layout_desc.entries = entries.data();
        layout = device.CreateBindGroupLayout(&layout_desc);
    }
    const std::string code = FragmentShader(dst_format, kind);
    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = code.c_str();
    wgpu::ShaderModuleDescriptor module_desc{};
    module_desc.nextInChain = &wgsl;
    const wgpu::ShaderModule module = device.CreateShaderModule(&module_desc);

    wgpu::PipelineLayoutDescriptor pipeline_layout_desc{};
    pipeline_layout_desc.bindGroupLayoutCount = 1;
    pipeline_layout_desc.bindGroupLayouts = &layout;
    const wgpu::PipelineLayout pipeline_layout = device.CreatePipelineLayout(&pipeline_layout_desc);

    wgpu::RenderPipelineDescriptor desc{};
    desc.layout = pipeline_layout;
    desc.vertex.module = module;
    desc.vertex.entryPoint = "vs";
    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleStrip;
    wgpu::ColorTargetState target{};
    wgpu::FragmentState fragment{};
    fragment.module = module;
    fragment.entryPoint = "fs";
    wgpu::DepthStencilState depth{};
    if (IsDepthFormat(dst_format)) {
        depth.format = dst_format;
        depth.depthWriteEnabled = wgpu::OptionalBool::True;
        depth.depthCompare = wgpu::CompareFunction::Always;
        desc.depthStencil = &depth;
    } else {
        target.format = dst_format;
        fragment.targetCount = 1;
        fragment.targets = &target;
    }
    desc.fragment = &fragment;
    return pipelines.emplace(key, device.CreateRenderPipeline(&desc)).first->second;
}

void Blitter::Blit(const wgpu::TextureView& dst, wgpu::TextureFormat dst_format, u32 dst_width,
                   u32 dst_height, const VideoCommon::Region2D& dst_region,
                   const wgpu::TextureView& src, TextureKind src_kind, u32 src_width,
                   u32 src_height, const VideoCommon::Region2D& src_region, bool linear) {
    if (!dst || !src || dst_width == 0 || dst_height == 0 || src_width == 0 || src_height == 0) {
        return;
    }
    // Flipped destination regions are expressed by flipping the source coordinates.
    s32 dx0 = dst_region.start.x, dx1 = dst_region.end.x;
    s32 dy0 = dst_region.start.y, dy1 = dst_region.end.y;
    f32 sx0 = static_cast<f32>(src_region.start.x), sx1 = static_cast<f32>(src_region.end.x);
    f32 sy0 = static_cast<f32>(src_region.start.y), sy1 = static_cast<f32>(src_region.end.y);
    if (dx1 < dx0) {
        std::swap(dx0, dx1);
        std::swap(sx0, sx1);
    }
    if (dy1 < dy0) {
        std::swap(dy0, dy1);
        std::swap(sy0, sy1);
    }
    const s32 cx0 = std::clamp(dx0, 0, static_cast<s32>(dst_width));
    const s32 cy0 = std::clamp(dy0, 0, static_cast<s32>(dst_height));
    const s32 cx1 = std::clamp(dx1, 0, static_cast<s32>(dst_width));
    const s32 cy1 = std::clamp(dy1, 0, static_cast<s32>(dst_height));
    if (cx1 <= cx0 || cy1 <= cy0) {
        return;
    }
    // Viewports must lie inside the attachment: clip, then map the clipped corners back to source.
    const auto lerp = [](f32 a, f32 b, f32 t) { return a + (b - a) * t; };
    const f32 width = static_cast<f32>(dx1 - dx0);
    const f32 height = static_cast<f32>(dy1 - dy0);
    const f32 u0 = lerp(sx0, sx1, static_cast<f32>(cx0 - dx0) / width);
    const f32 u1 = lerp(sx0, sx1, static_cast<f32>(cx1 - dx0) / width);
    const f32 v0 = lerp(sy0, sy1, static_cast<f32>(cy0 - dy0) / height);
    const f32 v1 = lerp(sy0, sy1, static_cast<f32>(cy1 - dy0) / height);
    const std::array<f32, 4> params{u0 / static_cast<f32>(src_width), v0 / static_cast<f32>(src_height),
                                    (u1 - u0) / static_cast<f32>(src_width),
                                    (v1 - v0) / static_cast<f32>(src_height)};
    const u32 offset =
        uniforms.Push(std::span<const u8>(reinterpret_cast<const u8*>(params.data()), sizeof(params)));
    const wgpu::RenderPipeline& pipeline = Pipeline(dst_format, src_kind);

    std::array<wgpu::BindGroupEntry, 3> entries{};
    entries[0].binding = 0;
    entries[0].buffer = uniforms.Buffer();
    entries[0].size = 16;
    entries[1].binding = 1;
    entries[1].textureView = src;
    entries[2].binding = 2;
    entries[2].sampler = linear && src_kind == TextureKind::Float ? linear_sampler : nearest_sampler;
    wgpu::BindGroupDescriptor bind_desc{};
    bind_desc.layout = layouts[src_kind];
    bind_desc.entryCount = entries.size();
    bind_desc.entries = entries.data();
    const wgpu::BindGroup bind_group = Thread::Get().device.CreateBindGroup(&bind_desc);

    RenderTargets targets{};
    if (IsDepthFormat(dst_format)) {
        targets.depth_stencil = dst;
        targets.depth_format = dst_format;
        targets.has_stencil = HasStencilAspect(dst_format);
    } else {
        targets.colors[0] = dst;
        targets.color_formats[0] = dst_format;
    }
    targets.width = dst_width;
    targets.height = dst_height;
    // A dedicated pass keeps the draw state of the rasterizer's pass untouched.
    scheduler.EndRenderPass();
    wgpu::RenderPassEncoder& pass = scheduler.RenderPass(targets);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group, 1, &offset);
    pass.SetViewport(static_cast<f32>(cx0), static_cast<f32>(cy0), static_cast<f32>(cx1 - cx0),
                     static_cast<f32>(cy1 - cy0), 0.0f, 1.0f);
    pass.Draw(4);
    scheduler.EndRenderPass();
}

} // namespace WebGPU
