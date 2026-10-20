// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/alignment.h"
#include "common/logging.h"
#include "video_core/control/channel_state.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_webgpu/wgpu_rasterizer.h"
#include "video_core/renderer_webgpu/maxwell_to_wgpu.h"
#include "video_core/renderer_webgpu/wgpu_thread.h"
#include "video_core/surface.h"
#include "video_core/gpu.h"
#include "shader_recompiler/backend/wgsl/emit_wgsl.h"

#include "video_core/texture_cache/types.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace WebGPU {

using VideoCommon::Offset2D;
using VideoCommon::Region2D;

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_) : buffer_cache{buffer_cache_} {}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMACopy(src_address, dest_address, amount);
}

bool AccelerateDMA::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::ImageOperand& src,
                                  const Tegra::DMA::BufferOperand& dst) {
    static u32 logs = 0; // TEMP-DEBUG
    if (logs++ < 10) {
        LOG_INFO(Render, "TEMP-DEBUG DMA image->buffer {}x{} src {:#x} dst {:#x} pitch {}",
                 copy_info.length_x, copy_info.length_y, src.address, dst.address, dst.pitch);
    }
    return false;
}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMAClear(src_address, amount, value);
}

RasterizerWebGPU::RasterizerWebGPU(Tegra::GPU& gpu, Tegra::MaxwellDeviceMemoryManager& device_memory)
    : m_gpu{gpu}, m_device_memory{device_memory}, uniform_ring{scheduler},
      blitter{scheduler, uniform_ring}, texture_cache_runtime{scheduler, blitter},
      texture_cache{texture_cache_runtime, device_memory},
      buffer_cache_runtime{device_memory, scheduler}, buffer_cache{device_memory, buffer_cache_runtime},
      pipeline_cache{device_memory, scheduler, uniform_ring, buffer_cache, texture_cache},
      m_accelerate_dma{buffer_cache} {
    BindDrawInfoGroup();
}

void RasterizerWebGPU::BindDrawInfoGroup() {
    auto& device = Thread::Get().device;
    wgpu::BindGroupLayoutEntry entry{};
    entry.binding = 0;
    entry.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    entry.buffer.type = wgpu::BufferBindingType::Uniform;
    entry.buffer.hasDynamicOffset = true;
    entry.buffer.minBindingSize = sizeof(Shader::Backend::WGSL::DrawInfo);
    wgpu::BindGroupLayoutDescriptor layout_desc{};
    layout_desc.entryCount = 1;
    layout_desc.entries = &entry;
    const wgpu::BindGroupLayout layout = device.CreateBindGroupLayout(&layout_desc);
    wgpu::BindGroupEntry group_entry{};
    group_entry.binding = 0;
    group_entry.buffer = uniform_ring.Buffer();
    group_entry.size = sizeof(Shader::Backend::WGSL::DrawInfo);
    wgpu::BindGroupDescriptor desc{};
    desc.layout = layout;
    desc.entryCount = 1;
    desc.entries = &group_entry;
    draw_info_group = device.CreateBindGroup(&desc);
}
RasterizerWebGPU::~RasterizerWebGPU() = default;

void RasterizerWebGPU::Draw(bool is_indexed, u32 instance_count) {
    ++stats.draws;
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    if (!pipeline) {
        return;
    }
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    if (!pipeline->Configure(is_indexed)) {
        return;
    }
    const auto& regs = maxwell3d->regs;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const RenderTargets& targets = *pipeline->targets;
    if (targets.width == 0 || targets.height == 0) {
        return;
    }

    // Draw parameters, after topologies WebGPU lacks were rewritten into index lists.
    auto& runtime = buffer_cache_runtime;
    bool indexed = is_indexed;
    u32 count = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count;
    u32 first = is_indexed ? draw_state.index_buffer.first : draw_state.vertex_buffer.first;
    s32 base_vertex = is_indexed ? static_cast<s32>(draw_state.base_index) : 0;
    const bool needs_conversion = MaxwellToWGPU::NeedsIndexConversion(draw_state.topology);
    if (!is_indexed && needs_conversion) {
        if (draw_state.topology != Maxwell::PrimitiveTopology::Quads &&
            draw_state.topology != Maxwell::PrimitiveTopology::QuadStrip) {
            runtime.BindQuadIndexBuffer(draw_state.topology, draw_state.vertex_buffer.first, count);
        }
        indexed = true;
        count = runtime.index.count;
        first = 0;
    } else if (is_indexed && runtime.index.converted) {
        count = runtime.index.count;
        first = 0;
    }
    if (count == 0 || instance_count == 0) {
        return;
    }

    // The guest viewport is applied in the vertex shader so it may extend past the target.
    const auto& transform = regs.viewport_transform[0];
    f32 vx = transform.translate_x - transform.scale_x;
    f32 vy = transform.translate_y - transform.scale_y;
    const f32 vw = transform.scale_x * 2.0f;
    f32 vh = transform.scale_y * 2.0f;
    if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
        vy += static_cast<f32>(regs.surface_clip.height);
        vh = -vh;
    }
    if (transform.swizzle.y == Maxwell::ViewportSwizzle::NegativeY) {
        vy += vh;
        vh = -vh;
    }
    const f32 width = static_cast<f32>(targets.width);
    const f32 height = static_cast<f32>(targets.height);
    Shader::Backend::WGSL::DrawInfo info{};
    info.viewport = {vw / width, (2.0f * vx + vw) / width - 1.0f, -vh / height,
                     1.0f - (2.0f * vy + vh) / height};
    info.render_area = {width, height, 0.0f, 0.0f};
    info.base_vertex = static_cast<u32>(base_vertex);
    info.base_instance = draw_state.base_instance;
    const f32 reduce_z = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
    f32 min_depth = std::clamp(transform.translate_z - transform.scale_z * reduce_z, 0.0f, 1.0f);
    f32 max_depth = std::clamp(transform.translate_z + transform.scale_z, 0.0f, 1.0f);
    if (min_depth > max_depth) {
        std::swap(min_depth, max_depth);
        info.depth_flip = 1;
    }
    const u32 draw_offset = uniform_ring.Push(
        std::span<const u8>(reinterpret_cast<const u8*>(&info), sizeof(info)));

    // Scissor, bounded by the render target.
    u32 sx = 0, sy = 0, sw = targets.width, sh = targets.height;
    const auto& scissor = regs.scissor_test[0];
    if (scissor.enable) {
        const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
        const s32 clip_height = static_cast<s32>(regs.surface_clip.height);
        s32 min_y = lower_left ? clip_height - static_cast<s32>(scissor.max_y) : static_cast<s32>(scissor.min_y);
        s32 max_y = lower_left ? clip_height - static_cast<s32>(scissor.min_y) : static_cast<s32>(scissor.max_y);
        const u32 x0 = std::min<u32>(scissor.min_x, targets.width);
        const u32 x1 = std::min<u32>(scissor.max_x, targets.width);
        const u32 y0 = static_cast<u32>(std::clamp<s32>(min_y, 0, static_cast<s32>(targets.height)));
        const u32 y1 = static_cast<u32>(std::clamp<s32>(max_y, 0, static_cast<s32>(targets.height)));
        if (x1 <= x0 || y1 <= y0) {
            return;
        }
        sx = x0, sy = y0, sw = x1 - x0, sh = y1 - y0;
    }

    wgpu::RenderPassEncoder& pass = scheduler.RenderPass(targets);
    if (targets.colors[0] && targets.width >= 640) {
        debug_last_target = targets.colors[0];
        debug_last_width = targets.width;
        debug_last_height = targets.height;
        ++debug_last_serial;
    }
    pass.SetPipeline(pipeline->render_pipeline);
    pass.SetBindGroup(0, pipeline->bind_group);
    pass.SetBindGroup(1, draw_info_group, 1, &draw_offset);
    // WebGPU rejects draws reading past a vertex binding (and the whole command buffer with
    // them), while the guest just reads zeros; widen bindings to the buffer end or skip the draw.
    std::array<u64, Maxwell::NumVertexArrays> vertex_sizes{};
    for (u32 index = 0; index < Maxwell::NumVertexArrays; ++index) {
        const auto& use = pipeline->vertex_buffer_uses[index];
        if (!use.used) {
            continue;
        }
        const auto& binding = runtime.vertex_bindings[index];
        if (index >= runtime.num_vertex_bindings || !binding.buffer ||
            binding.offset >= binding.buffer.GetSize()) {
            return;
        }
        const u64 available = binding.buffer.GetSize() - (binding.offset & ~3ULL);
        u64 needed = 0;
        if (use.per_instance) {
            needed = static_cast<u64>(draw_state.base_instance + instance_count) * use.stride;
        } else if (!indexed) {
            needed = static_cast<u64>(first + count) * use.stride;
        }
        if (needed > available) {
            return;
        }
        vertex_sizes[index] = available;
    }
    for (u32 index = 0; index < Maxwell::NumVertexArrays; ++index) {
        if (vertex_sizes[index] != 0) {
            const auto& binding = runtime.vertex_bindings[index];
            pass.SetVertexBuffer(index, binding.buffer, binding.offset & ~3ULL, vertex_sizes[index]);
        }
    }
    pass.SetViewport(0.0f, 0.0f, width, height, min_depth, max_depth);
    pass.SetScissorRect(sx, sy, sw, sh);
    const wgpu::Color blend_color{regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                                  regs.blend_color.a};
    pass.SetBlendConstant(&blend_color);
    pass.SetStencilReference(regs.stencil_front_ref & 0xFF);
    if (indexed) {
        const auto& index = runtime.index;
        if (!index.buffer) {
            return;
        }
        const u64 index_offset = index.offset & ~3ULL;
        if (index_offset >= index.buffer.GetSize()) {
            return;
        }
        const u64 available = index.buffer.GetSize() - index_offset;
        const u64 needed = (static_cast<u64>(first) + count) *
                           (index.format == wgpu::IndexFormat::Uint16 ? 2 : 4);
        if (needed > available) {
            static u32 logged = 0;
            if (logged++ < 8) {
                LOG_WARNING(Render, "WebGPU: index range {}+{} (needed {}) exceeds available {} (buffer {} offset {} "
                            "converted {})", first, count, needed, available, index.buffer.GetSize(),
                            index.offset, index.converted);
            }
            return;
        }
        const u64 index_size = std::min(available, std::max(needed, static_cast<u64>(index.size)));
        pass.SetIndexBuffer(index.buffer, index.format, index_offset, index_size);
        pass.DrawIndexed(count, instance_count, first, index.converted && !is_indexed ? 0 : base_vertex,
                         draw_state.base_instance);
        ++stats.actual_draws;
    } else {
        pass.Draw(count, instance_count, first, draw_state.base_instance);
        ++stats.actual_draws;
    }
}
void RasterizerWebGPU::DrawTexture() {
    static u32 draw_tex_logs = 0;
    if (draw_tex_logs++ < 10) {
        LOG_INFO(Render, "RasterizerWebGPU::DrawTexture called");
    }
    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.SynchronizeDescriptors(false);
    texture_cache.UpdateRenderTargets(false);
    const auto& draw_texture_state = maxwell3d->draw_manager.draw_texture_state;
    const auto& texture = texture_cache.GetImageView(draw_texture_state.src_texture);
    const auto* framebuffer = texture_cache.GetFramebuffer();
    if (!framebuffer) {
        return;
    }
    const auto& targets = framebuffer->Targets();
    if (!targets.colors[0]) {
        return;
    }
    Region2D dst_region = {Offset2D{.x = static_cast<s32>(draw_texture_state.dst_x0),
                                    .y = static_cast<s32>(draw_texture_state.dst_y0)},
                           Offset2D{.x = static_cast<s32>(draw_texture_state.dst_x1),
                                    .y = static_cast<s32>(draw_texture_state.dst_y1)}};
    Region2D src_region = {Offset2D{.x = static_cast<s32>(draw_texture_state.src_x0),
                                    .y = static_cast<s32>(draw_texture_state.src_y0)},
                           Offset2D{.x = static_cast<s32>(draw_texture_state.src_x1),
                                    .y = static_cast<s32>(draw_texture_state.src_y1)}};
    blitter.Blit(targets.colors[0], targets.color_formats[0], targets.width, targets.height, dst_region,
                 texture.Handle(Shader::TextureType::Color2D), texture.Kind(), texture.Width(), texture.Height(),
                 src_region, true);
}
void RasterizerWebGPU::Clear(u32 layer_count) {
    auto& regs = maxwell3d->regs;
    const bool use_color = regs.clear_surface.R || regs.clear_surface.G || regs.clear_surface.B ||
                           regs.clear_surface.A;
    const bool use_depth = regs.clear_surface.Z;
    const bool use_stencil = regs.clear_surface.S;
    if (!use_color && !use_depth && !use_stencil) {
        return;
    }
    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UpdateRenderTargets(true);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    if (!framebuffer) {
        return;
    }
    const RenderTargets& targets = framebuffer->Targets();
    ClearRequest request{};
    const u32 rt = regs.clear_surface.RT;
    if (use_color && rt < targets.colors.size() && targets.colors[rt]) {
        const auto format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[rt].format);
        wgpu::Color color{regs.clear_color[0], regs.clear_color[1], regs.clear_color[2],
                          regs.clear_color[3]};
        if (VideoCore::Surface::IsPixelFormatInteger(format)) {
            const f64 scale = static_cast<f64>(
                u64(VideoCore::Surface::PixelComponentSizeBitsInteger(format)) << 1);
            color = {color.r * scale, color.g * scale, color.b * scale, color.a * scale};
        }
        request.color_mask = 1u << rt;
        request.colors[rt] = color;
    }
    if (targets.depth_stencil) {
        request.depth = use_depth;
        request.depth_value = regs.clear_depth;
        request.stencil = use_stencil && targets.has_stencil;
        request.stencil_value = regs.clear_stencil;
    }
    if (request.color_mask == 0 && !request.depth && !request.stencil) {
        return;
    }
    ++stats.clears;
    // TODO: scissored and channel-masked clears need a draw; clear the whole attachment.
    scheduler.Clear(targets, request);
}
void RasterizerWebGPU::DispatchCompute() {
    static u32 logs = 0; // TEMP-DEBUG
    if (logs++ % 500 == 0) {
        LOG_INFO(Render, "TEMP-DEBUG compute dispatches {}", logs);
    }
}
void RasterizerWebGPU::ResetCounter(VideoCommon::QueryType type) {}
void RasterizerWebGPU::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                           VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    if (!gpu_memory) {
        return;
    }
    if (True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout)) {
        u64 ticks = m_gpu.GetTicks();
        gpu_memory->Write<u64>(gpu_addr + 8, ticks);
        gpu_memory->Write<u64>(gpu_addr, static_cast<u64>(payload));
    } else {
        gpu_memory->Write<u32>(gpu_addr, payload);
    }
}
void RasterizerWebGPU::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                 u32 size) {
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}
void RasterizerWebGPU::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}
void RasterizerWebGPU::FlushAll() {}
void RasterizerWebGPU::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType) {
    // GPU-to-guest readback is not implemented yet.
}
bool RasterizerWebGPU::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType) {
    return false;
}
void RasterizerWebGPU::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::ShaderCache)) {
        pipeline_cache.InvalidateRegion(addr, size);
    }
}
bool RasterizerWebGPU::OnCPUWrite(PAddr addr, u64 size) {
    {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.OnCPUWrite(addr, size)) {
            return true;
        }
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return false;
}
void RasterizerWebGPU::OnCacheInvalidation(PAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
}
VideoCore::RasterizerDownloadArea RasterizerWebGPU::GetFlushArea(PAddr addr, u64 size) {
    VideoCore::RasterizerDownloadArea new_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
    return new_area;
}
void RasterizerWebGPU::InvalidateGPUCache() {
    m_gpu.InvalidateGPUCache();
}
void RasterizerWebGPU::UnmapMemory(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.OnCacheInvalidation(addr, size);
}
void RasterizerWebGPU::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapGPUMemory(as_id, addr, size);
    }
    buffer_cache.UnmapGPUMemory(as_id, addr, size);
}
void RasterizerWebGPU::SignalFence(std::function<void()>&& func) {
    func();
}
void RasterizerWebGPU::SyncOperation(std::function<void()>&& func) {
    func();
}
void RasterizerWebGPU::SignalSyncPoint(u32 value) {
    auto& syncpoint_manager = m_gpu.Host1x().GetSyncpointManager();
    syncpoint_manager.IncrementGuest(value);
    syncpoint_manager.IncrementHost(value);
}
void RasterizerWebGPU::SignalReference() {}
void RasterizerWebGPU::ReleaseFences(bool) {}
void RasterizerWebGPU::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                                VideoCommon::CacheType which) {
    InvalidateRegion(addr, size, which);
}
void RasterizerWebGPU::WaitForIdle() {}
void RasterizerWebGPU::FragmentBarrier() {}
void RasterizerWebGPU::TiledCacheBarrier() {}
void RasterizerWebGPU::FlushCommands() {
    scheduler.Flush();
}
void RasterizerWebGPU::TickFrame() {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.TickFrame();
    }
}
Tegra::Engines::AccelerateDMAInterface& RasterizerWebGPU::AccessAccelerateDMA() {
    return m_accelerate_dma;
}
bool RasterizerWebGPU::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                           const Tegra::Engines::Fermi2D::Surface& dst,
                                           const Tegra::Engines::Fermi2D::Config& copy_config) {
    std::scoped_lock lock{texture_cache.mutex};
    Tegra::Engines::Fermi2D::Config copy = copy_config;
    copy.must_accelerate = true;
    const bool result = texture_cache.BlitImage(dst, src, copy);
    static u32 logs = 0; // TEMP-DEBUG
    if (logs++ < 20) {
        auto src_cpu = gpu_memory ? gpu_memory->GpuToCpuAddress(src.Address()) : std::nullopt;
        auto dst_cpu = gpu_memory ? gpu_memory->GpuToCpuAddress(dst.Address()) : std::nullopt;
        LOG_INFO(Render, "TEMP-DEBUG blit {:#x} (cpu {:#x}) {}x{} fmt {} -> {:#x} (cpu {:#x}) {}x{} fmt {} result {} accel {}",
                 src.Address(), src_cpu.value_or(0), src.width, src.height, static_cast<u32>(src.format),
                 dst.Address(), dst_cpu.value_or(0), dst.width, dst.height, static_cast<u32>(dst.format),
                 result, copy_config.must_accelerate);
    }
    return result;
}
void RasterizerWebGPU::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                              std::span<const u8> memory) {
    if (!gpu_memory) {
        return;
    }
    auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (!cpu_addr) [[unlikely]] {
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    {
        std::unique_lock<std::recursive_mutex> lock{buffer_cache.mutex};
        if (!buffer_cache.InlineMemory(*cpu_addr, copy_size, memory)) {
            buffer_cache.WriteMemory(*cpu_addr, copy_size);
        }
    }
    {
        std::scoped_lock lock_texture{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
}
void RasterizerWebGPU::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                       const VideoCore::DiskResourceLoadCallback& callback) {}
void RasterizerWebGPU::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.CreateChannel(channel);
        buffer_cache.CreateChannel(channel);
    }
    pipeline_cache.CreateChannel(channel);
    // Without these tables Maxwell3D register writes never mark render targets, shaders, blending
    // or vertex state dirty, and the caches and pipeline keys keep the state they first saw.
    state_tracker.SetupTables(channel);
}
void RasterizerWebGPU::BindChannel(Tegra::Control::ChannelState& channel) {
    BindToChannel(channel.bind_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.BindToChannel(channel.bind_id);
        buffer_cache.BindToChannel(channel.bind_id);
    }
    pipeline_cache.BindToChannel(channel.bind_id);
    state_tracker.ChangeChannel(channel);
    state_tracker.InvalidateState();
}
void RasterizerWebGPU::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.EraseChannel(channel_id);
        buffer_cache.EraseChannel(channel_id);
    }
    pipeline_cache.EraseChannel(channel_id);
}

wgpu::TextureView RasterizerWebGPU::AccelerateDisplay(const Tegra::FramebufferConfig& config,
                                                      u32& width, u32& height) {
    const DAddr framebuffer_addr = config.address + config.offset;
    if (!framebuffer_addr) {
        return {};
    }
    std::scoped_lock lock{texture_cache.mutex};

    const auto [image_view, scaled] = texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    static u32 debug_logs = 0;
    if (debug_logs < 30) {
        ++debug_logs;
        LOG_INFO(Render, "AccelerateDisplay #{}: fb={:#x} fmt {} {}x{} hit {} size {}x{} (blit_tgt={} draw_tgt={})",
                 debug_logs, framebuffer_addr, static_cast<u32>(config.pixel_format),
                 config.width, config.height, image_view != nullptr,
                 image_view ? image_view->size.width : 0, image_view ? image_view->size.height : 0,
                 texture_cache_runtime.last_blit_target != nullptr, debug_last_target != nullptr);
    }
    if (image_view) {
        width = image_view->size.width;
        height = image_view->size.height;
        ++stats.display_hits;
        return image_view->Handle(Shader::TextureType::Color2D);
    }

    if (texture_cache_runtime.last_blit_target) {
        static u32 blit_disp_logs = 0;
        if (blit_disp_logs++ < 10) {
            LOG_INFO(Render, "AccelerateDisplay: returning last_blit_target {}x{} (serial {})",
                     texture_cache_runtime.last_blit_width, texture_cache_runtime.last_blit_height,
                     texture_cache_runtime.last_blit_serial);
        }
        width = texture_cache_runtime.last_blit_width;
        height = texture_cache_runtime.last_blit_height;
        ++stats.display_hits;
        return texture_cache_runtime.last_blit_target;
    }
    if (debug_last_target) {
        width = debug_last_width;
        height = debug_last_height;
        ++stats.display_hits;
        return debug_last_target;
    }

    ++stats.display_misses;
    return {};
}

} // namespace WebGPU
