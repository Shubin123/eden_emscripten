// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/rasterizer_interface.h"

#include <webgpu/webgpu_cpp.h>

#include "video_core/framebuffer_config.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_webgpu/wgpu_blit.h"
#include "video_core/renderer_webgpu/wgpu_buffer_cache.h"
#include "video_core/renderer_webgpu/wgpu_pipeline_cache.h"
#include "video_core/renderer_webgpu/wgpu_scheduler.h"
#include "video_core/renderer_webgpu/wgpu_texture_cache.h"
#include "video_core/renderer_webgpu/wgpu_uniform_ring.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"

namespace Core {
class System;
}

namespace WebGPU {

class RasterizerWebGPU;

class AccelerateDMA : public Tegra::Engines::AccelerateDMAInterface {
public:
    explicit AccelerateDMA(BufferCache& buffer_cache);
    bool BufferCopy(GPUVAddr start_address, GPUVAddr end_address, u64 amount) override;
    bool BufferClear(GPUVAddr src_address, u64 amount, u32 value) override;
    bool ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::ImageOperand& src,
                       const Tegra::DMA::BufferOperand& dst) override;
    bool BufferToImage(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& src,
                       const Tegra::DMA::ImageOperand& dst) override {
        return false;
    }

private:
    BufferCache& buffer_cache;
};

class RasterizerWebGPU final : public VideoCore::RasterizerInterface,
                             protected VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo> {
public:
    explicit RasterizerWebGPU(Tegra::GPU& gpu, Tegra::MaxwellDeviceMemoryManager& device_memory);
    ~RasterizerWebGPU() override;

    void Draw(bool is_indexed, u32 instance_count) override;
    void DrawIndirect() override;
    void DrawTexture() override;
    void Clear(u32 layer_count) override;
    void DispatchCompute() override;
    void ResetCounter(VideoCommon::QueryType type) override;
    void Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
               VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) override;
    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size) override;
    void DisableGraphicsUniformBuffer(size_t stage, u32 index) override;
    void FlushAll() override;
    void FlushRegion(DAddr addr, u64 size,
                     VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    bool MustFlushRegion(DAddr addr, u64 size,
                         VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void InvalidateRegion(DAddr addr, u64 size,
                          VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void OnCacheInvalidation(DAddr addr, u64 size) override;
    bool OnCPUWrite(DAddr addr, u64 size) override;
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override;
    void InvalidateGPUCache() override;
    void UnmapMemory(DAddr addr, u64 size) override;
    void ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) override;
    void SignalFence(std::function<void()>&& func) override;
    void SyncOperation(std::function<void()>&& func) override;
    void SignalSyncPoint(u32 value) override;
    void SignalReference() override;
    void ReleaseFences(bool force) override;
    void FlushAndInvalidateRegion(
        DAddr addr, u64 size, VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void WaitForIdle() override;
    void FragmentBarrier() override;
    void TiledCacheBarrier() override;
    void FlushCommands() override;
    void TickFrame() override;
    bool AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                               const Tegra::Engines::Fermi2D::Surface& dst,
                               const Tegra::Engines::Fermi2D::Config& copy_config) override;
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override;
    void AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                  std::span<const u8> memory) override;
    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;
    void InitializeChannel(Tegra::Control::ChannelState& channel) override;
    void BindChannel(Tegra::Control::ChannelState& channel) override;
    void ReleaseChannel(s32 channel_id) override;

    /// Returns the GPU-rendered image backing a display layer, if any, and its size.
    wgpu::TextureView AccelerateDisplay(const Tegra::FramebufferConfig& config, u32& width,
                                        u32& height);

private:
    Tegra::GPU& m_gpu;
    Tegra::MaxwellDeviceMemoryManager& m_device_memory;
public:
    struct Stats {
        u64 draws = 0;
        u64 actual_draws = 0;
        u64 clears = 0;
        u64 display_hits = 0;
        u64 display_misses = 0;
    } stats;

    /// Last color target drawn to.
    wgpu::TextureView debug_last_target;
    u32 debug_last_width = 0;
    u32 debug_last_height = 0;
    u64 debug_last_serial = 0;

private:
    Vulkan::StateTracker state_tracker;
    Scheduler scheduler;
    UniformRing uniform_ring;
    Blitter blitter;
    TextureCacheRuntime texture_cache_runtime;
    TextureCache texture_cache;
    BufferCacheRuntime buffer_cache_runtime;
    BufferCache buffer_cache;
    PipelineCache pipeline_cache;
    wgpu::BindGroup draw_info_group;
    /// Copies of storage ranges a draw writes while reading the same host buffer.
    wgpu::Buffer shadow_buffer;
    AccelerateDMA m_accelerate_dma;
    /// Samples counted for the guest's occlusion queries. Host results are not read back, so
    /// every draw made while counting is enabled passes a fixed number of samples.
    u64 samples_passed = 0;

    void BindDrawInfoGroup();
    /// Draws with the parameters in the draw state, or with those of the indirect state in GPU
    /// memory when `indirect`.
    void DrawCommon(bool is_indexed, u32 instance_count, bool indirect);
};

} // namespace WebGPU
