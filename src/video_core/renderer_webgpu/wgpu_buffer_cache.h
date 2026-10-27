// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <span>
#include <vector>

#include <webgpu/webgpu_cpp.h>

#include "common/common_types.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_webgpu/wgpu_scheduler.h"
#include "video_core/renderer_webgpu/wgpu_texture_cache.h"
#include "video_core/renderer_webgpu/wgpu_uniform_ring.h"

namespace WebGPU {

class BufferCacheRuntime;

class Buffer : public VideoCommon::BufferBase {
public:
    explicit Buffer(BufferCacheRuntime& runtime, DAddr cpu_addr, u64 size_bytes,
                    bool sparse_compatible = false);
    explicit Buffer(BufferCacheRuntime& runtime, VideoCommon::NullBufferParams);

    void ImmediateUpload(size_t offset, std::span<const u8> data) noexcept;
    /// GPU readback is not implemented; keep guest memory as it is.
    void ImmediateDownload(size_t offset, std::span<u8> data) noexcept;
    void MarkUsage(u64, u64) noexcept {}

    [[nodiscard]] bool IsSparseCompatible() const noexcept {
        return false;
    }

    [[nodiscard]] const wgpu::Buffer& Handle() const noexcept {
        return buffer;
    }

    operator const wgpu::Buffer&() const noexcept {
        return buffer;
    }

private:
    BufferCacheRuntime* runtime{};
    wgpu::Buffer buffer;
    u64 host_size = 0;
};

/// A resource bound for the next draw, in shader binding order.
struct BufferDescriptor {
    bool is_uniform{};
    wgpu::Buffer buffer;   ///< Storage buffers
    u64 offset{};
    u64 size{};
    std::vector<u8> data;  ///< Uniform contents, copied into the uniform ring per draw
};

/// Index data the next indexed draw uses, possibly converted to a format WebGPU supports.
struct IndexBinding {
    wgpu::Buffer buffer;
    u64 offset{};
    u64 size{};
    wgpu::IndexFormat format = wgpu::IndexFormat::Uint16;
    bool converted{};      ///< Indices were rewritten; draw from index zero
    u32 count{};
};

class BufferCacheRuntime {
    friend Buffer;

    using PrimitiveTopology = Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology;
    using IndexFormat = Tegra::Engines::Maxwell3D::Regs::IndexFormat;

public:
    static constexpr u8 INVALID_BINDING = (std::numeric_limits<u8>::max)();

    explicit BufferCacheRuntime(Tegra::MaxwellDeviceMemoryManager& device_memory,
                                Scheduler& scheduler);

    void TickFrame(Common::SlotVector<Buffer>&) noexcept {}
    u64 CurrentTick() const {
        return scheduler.SubmitCount();
    }
    bool IsFree(u64) const {
        return true;
    }
    void Wait(u64) {}
    void Finish() {
        scheduler.Flush();
    }

    u64 GetDeviceLocalMemory() const {
        return 2ULL << 30;
    }
    u64 GetDeviceMemoryUsage() const {
        return memory_usage;
    }
    bool CanReportMemoryUsage() const {
        return true;
    }
    u32 GetUniformBufferAlignment() const {
        return 256;
    }
    u32 GetStorageBufferAlignment() const {
        return 256;
    }

    StagingBufferMap UploadStagingBuffer(size_t size);
    StagingBufferMap DownloadStagingBuffer(size_t size, bool deferred = false);
    void FreeDeferredStagingBuffer(StagingBufferMap&) {}

    bool CanReorderUpload(const Buffer&, std::span<const VideoCommon::BufferCopy>) {
        return false;
    }

    void PreCopyBarrier() {}
    void PostCopyBarrier() {}

    void CopyBuffer(const wgpu::Buffer& dst, const wgpu::Buffer& src,
                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                    bool can_reorder_upload = false);

    void ClearBuffer(const wgpu::Buffer& dst, u32 offset, size_t size, u32 value);

    void BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format, u32 first,
                         u32 count, const wgpu::Buffer& buffer, u32 offset, u32 size);
    void BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format, u32 first,
                         u32 count, Buffer& buffer, u32 offset, u32 size) {
        BindIndexBuffer(topology, index_format, first, count, buffer.Handle(), offset, size,
                        buffer.CpuAddr());
    }
    void BindQuadIndexBuffer(PrimitiveTopology topology, u32 first, u32 count);

    void BindVertexBuffer(u32 index, const wgpu::Buffer& buffer, u32 offset, u32 size, u32 stride);
    void BindVertexBuffers(VideoCommon::HostBindings<Buffer>& bindings);

    void BindTransformFeedbackBuffer(u32, const wgpu::Buffer&, u32, u32) {}
    void BindTransformFeedbackBuffers(VideoCommon::HostBindings<Buffer>&) {}

    std::span<u8> BindMappedUniformBuffer(size_t stage, u32 binding_index, u32 size);

    void BindUniformBuffer(const Buffer& buffer, u32 offset, u32 size);
    void BindStorageBuffer(const Buffer& buffer, u32 offset, u32 size, bool is_written);
    void BindTextureBuffer(Buffer&, u32, u32, VideoCore::Surface::PixelFormat) {}
    void BindImageBuffer(Buffer&, u32, u32, VideoCore::Surface::PixelFormat) {}

    bool PrefersSparseSources() const noexcept {
        return false;
    }
    void ResetMultiRange() noexcept {}
    void PushMultiRangeSource(const Buffer&, u32, u32) {}
    bool BindMultiRangeStorageBuffer(u64, bool) {
        return false;
    }
    void InvalidateMultiRange(u64) {}
    void OnBufferDeleted(const Buffer&) {}
    bool ShouldLimitDynamicStorageBuffers() const {
        return false;
    }
    u32 GetMaxDynamicStorageBuffers() const {
        return (std::numeric_limits<u32>::max)();
    }

    /// Descriptors pushed since the last call, in binding order.
    std::vector<BufferDescriptor>& Descriptors() {
        return descriptors;
    }
    void ClearDescriptors() {
        descriptors.clear();
    }

    struct VertexBinding {
        wgpu::Buffer buffer;
        u64 offset{};
        u64 size{};
    };
    std::array<VertexBinding, 32> vertex_bindings{};
    u32 num_vertex_bindings{};
    IndexBinding index;

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    Scheduler& scheduler;
    UniformRing index_ring;
    u64 memory_usage{};

private:
    void BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format, u32 first,
                         u32 count, const wgpu::Buffer& buffer, u32 offset, u32 size,
                         DAddr cpu_addr);

    std::vector<BufferDescriptor> descriptors;
    std::vector<u32> index_scratch;
    wgpu::Buffer copy_scratch;
};

struct BufferCacheParams {
    using Runtime = WebGPU::BufferCacheRuntime;
    using Buffer = WebGPU::Buffer;
    using Async_Buffer = WebGPU::StagingBufferMap;
    using MemoryTracker = VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager>;

    static constexpr bool IS_OPENGL = false;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS = false;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT = false;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = false;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = false;
    static constexpr bool USE_MEMORY_MAPS = false;
    static constexpr bool SEPARATE_IMAGE_BUFFER_BINDINGS = false;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = false;
};

using BufferCache = VideoCommon::BufferCache<BufferCacheParams>;

} // namespace WebGPU
