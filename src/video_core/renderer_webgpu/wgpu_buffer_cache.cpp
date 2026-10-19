// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>

#include "common/alignment.h"
#include "common/logging.h"
#include "video_core/renderer_webgpu/wgpu_buffer_cache.h"
#include "video_core/renderer_webgpu/wgpu_thread.h"

namespace WebGPU {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

namespace {
constexpr wgpu::BufferUsage BUFFER_USAGE = wgpu::BufferUsage::Vertex | wgpu::BufferUsage::Index |
                                           wgpu::BufferUsage::Uniform | wgpu::BufferUsage::Storage |
                                           wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst |
                                           wgpu::BufferUsage::Indirect;

u32 IndexSize(Maxwell::IndexFormat format) {
    switch (format) {
    case Maxwell::IndexFormat::UnsignedByte:
        return 1;
    case Maxwell::IndexFormat::UnsignedShort:
        return 2;
    case Maxwell::IndexFormat::UnsignedInt:
        return 4;
    }
    return 4;
}

bool NeedsConversion(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Polygon:
    case Maxwell::PrimitiveTopology::LineLoop:
        return true;
    default:
        return false;
    }
}

/// Rewrites `in` (vertex indices of `topology`) as a list WebGPU can draw.
void ConvertIndices(Maxwell::PrimitiveTopology topology, std::span<const u32> in,
                    std::vector<u32>& out) {
    out.clear();
    const size_t count = in.size();
    switch (topology) {
    case Maxwell::PrimitiveTopology::Quads:
        for (size_t i = 0; i + 3 < count; i += 4) {
            out.insert(out.end(), {in[i], in[i + 1], in[i + 2], in[i], in[i + 2], in[i + 3]});
        }
        break;
    case Maxwell::PrimitiveTopology::QuadStrip:
        for (size_t i = 0; i + 3 < count; i += 2) {
            out.insert(out.end(), {in[i], in[i + 1], in[i + 3], in[i], in[i + 3], in[i + 2]});
        }
        break;
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Polygon:
        for (size_t i = 1; i + 1 < count; ++i) {
            out.insert(out.end(), {in[0], in[i], in[i + 1]});
        }
        break;
    case Maxwell::PrimitiveTopology::LineLoop:
        for (size_t i = 0; i + 1 < count; ++i) {
            out.insert(out.end(), {in[i], in[i + 1]});
        }
        if (count > 1) {
            out.insert(out.end(), {in[count - 1], in[0]});
        }
        break;
    default:
        out.assign(in.begin(), in.end());
        break;
    }
}
} // Anonymous namespace

Buffer::Buffer(BufferCacheRuntime& runtime_, DAddr cpu_addr_, u64 size_bytes_, bool)
    : VideoCommon::BufferBase(cpu_addr_, size_bytes_), runtime{&runtime_} {
    host_size = Common::AlignUp(std::max<u64>(size_bytes_, 4), 4);
    wgpu::BufferDescriptor desc{};
    desc.size = host_size;
    desc.usage = BUFFER_USAGE;
    buffer = Thread::Get().device.CreateBuffer(&desc);
    runtime->memory_usage += host_size;
}

Buffer::Buffer(BufferCacheRuntime& runtime_, VideoCommon::NullBufferParams params)
    : VideoCommon::BufferBase(params), runtime{&runtime_} {
    host_size = 256;
    wgpu::BufferDescriptor desc{};
    desc.size = host_size;
    desc.usage = BUFFER_USAGE;
    buffer = Thread::Get().device.CreateBuffer(&desc);
}

void Buffer::ImmediateUpload(size_t offset, std::span<const u8> data) noexcept {
    if (data.empty() || !buffer) {
        return;
    }
    runtime->scheduler.BeforeQueueWrite();
    // Queue writes need four byte aligned offsets and sizes; pad with the guest bytes around.
    const size_t start = Common::AlignDown(offset, 4);
    const size_t end = std::min<size_t>(Common::AlignUp(offset + data.size(), 4), host_size);
    if (start == offset && end == offset + data.size()) {
        Thread::Get().queue.WriteBuffer(buffer, offset, data.data(), data.size());
        return;
    }
    std::vector<u8> padded(end - start);
    const size_t guest_end = std::min<size_t>(end, SizeBytes());
    if (guest_end > start) {
        runtime->device_memory.ReadBlockUnsafe(CpuAddr() + start, padded.data(), guest_end - start);
    }
    std::memcpy(padded.data() + (offset - start), data.data(),
                std::min(data.size(), padded.size() - (offset - start)));
    Thread::Get().queue.WriteBuffer(buffer, start, padded.data(), padded.size());
}

void Buffer::ImmediateDownload(size_t offset, std::span<u8> data) noexcept {
    if (CpuAddr() == 0) {
        std::memset(data.data(), 0, data.size());
        return;
    }
    runtime->device_memory.ReadBlockUnsafe(CpuAddr() + offset, data.data(), data.size());
}

BufferCacheRuntime::BufferCacheRuntime(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                       Scheduler& scheduler_)
    : device_memory{device_memory_}, scheduler{scheduler_},
      index_ring{scheduler_, 8ULL << 20, wgpu::BufferUsage::Index} {}

StagingBufferMap BufferCacheRuntime::UploadStagingBuffer(size_t size) {
    StagingBufferMap map;
    map.storage.resize(size);
    map.mapped_span = map.storage;
    return map;
}

StagingBufferMap BufferCacheRuntime::DownloadStagingBuffer(size_t size, bool) {
    return UploadStagingBuffer(size);
}

void BufferCacheRuntime::CopyBuffer(const wgpu::Buffer& dst, const wgpu::Buffer& src,
                                    std::span<const VideoCommon::BufferCopy> copies, bool, bool) {
    if (!dst || !src) {
        return;
    }
    auto& encoder = scheduler.Encoder();
    for (const VideoCommon::BufferCopy& copy : copies) {
        // Buffer copies must be four byte aligned.
        const u64 src_offset = Common::AlignDown(copy.src_offset, 4);
        const u64 dst_offset = Common::AlignDown(copy.dst_offset, 4);
        const u64 size = Common::AlignUp(copy.size + (copy.src_offset - src_offset), 4);
        const u64 limit = std::min(src.GetSize() - src_offset, dst.GetSize() - dst_offset);
        const u64 clamped = std::min(size, Common::AlignDown(limit, 4));
        if (clamped > 0) {
            encoder.CopyBufferToBuffer(src, src_offset, dst, dst_offset, clamped);
        }
    }
}

void BufferCacheRuntime::ClearBuffer(const wgpu::Buffer& dst, u32 offset, size_t size, u32 value) {
    if (!dst || size == 0) {
        return;
    }
    if (value == 0) {
        scheduler.Encoder().ClearBuffer(dst, Common::AlignDown(offset, 4),
                                        Common::AlignUp(static_cast<u64>(size), 4));
        return;
    }
    scheduler.BeforeQueueWrite();
    std::vector<u32> words(Common::DivCeil(size, sizeof(u32)), value);
    Thread::Get().queue.WriteBuffer(dst, Common::AlignDown(offset, 4), words.data(),
                                    words.size() * sizeof(u32));
}

void BufferCacheRuntime::BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format,
                                         u32 first, u32 count, const wgpu::Buffer& buffer,
                                         u32 offset, u32 size) {
    BindIndexBuffer(topology, index_format, first, count, buffer, offset, size, 0);
}

void BufferCacheRuntime::BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format,
                                         u32 first, u32 count, const wgpu::Buffer& buffer,
                                         u32 offset, u32 size, DAddr cpu_addr) {
    const u32 index_size = IndexSize(index_format);
    const bool convert = NeedsConversion(topology) || index_size == 1;
    if (!convert || cpu_addr == 0) {
        index.buffer = buffer;
        index.offset = offset;
        index.size = Common::AlignUp(size, 4);
        index.format = index_size == 2 ? wgpu::IndexFormat::Uint16 : wgpu::IndexFormat::Uint32;
        index.converted = false;
        index.count = count;
        return;
    }
    // Read the guest indices and rewrite them as 32-bit indices of a list topology.
    std::vector<u8> raw(static_cast<size_t>(count) * index_size);
    device_memory.ReadBlockUnsafe(cpu_addr + offset + static_cast<u64>(first) * index_size, raw.data(),
                                  raw.size());
    std::vector<u32> indices(count);
    for (u32 i = 0; i < count; ++i) {
        switch (index_size) {
        case 1:
            indices[i] = raw[i];
            break;
        case 2:
            indices[i] = static_cast<u32>(raw[i * 2] | (raw[i * 2 + 1] << 8));
            break;
        default:
            std::memcpy(&indices[i], raw.data() + i * 4, 4);
            break;
        }
    }
    ConvertIndices(topology, indices, index_scratch);
    if (index_scratch.empty()) {
        index_scratch.push_back(0);
    }
    const auto bytes = std::as_bytes(std::span(index_scratch));
    index.offset = index_ring.Push(
        std::span<const u8>(reinterpret_cast<const u8*>(bytes.data()), bytes.size()));
    index.buffer = index_ring.Buffer();
    index.size = bytes.size();
    index.format = wgpu::IndexFormat::Uint32;
    index.converted = true;
    index.count = static_cast<u32>(index_scratch.size());
}

void BufferCacheRuntime::BindQuadIndexBuffer(PrimitiveTopology topology, u32 first, u32 count) {
    std::vector<u32> vertices(count);
    for (u32 i = 0; i < count; ++i) {
        vertices[i] = first + i;
    }
    ConvertIndices(topology, vertices, index_scratch);
    if (index_scratch.empty()) {
        index_scratch.push_back(first);
    }
    const auto bytes = std::as_bytes(std::span(index_scratch));
    index.offset = index_ring.Push(
        std::span<const u8>(reinterpret_cast<const u8*>(bytes.data()), bytes.size()));
    index.buffer = index_ring.Buffer();
    index.size = bytes.size();
    index.format = wgpu::IndexFormat::Uint32;
    index.converted = true;
    index.count = static_cast<u32>(index_scratch.size());
}

void BufferCacheRuntime::BindVertexBuffer(u32 binding, const wgpu::Buffer& buffer, u32 offset,
                                          u32 size, u32) {
    if (binding >= vertex_bindings.size()) {
        return;
    }
    vertex_bindings[binding] = {buffer, offset, size};
    num_vertex_bindings = std::max(num_vertex_bindings, binding + 1);
}

void BufferCacheRuntime::BindVertexBuffers(VideoCommon::HostBindings<Buffer>& bindings) {
    for (u32 i = 0; i < bindings.buffers.size(); ++i) {
        const u32 binding = bindings.min_index + i;
        if (binding >= vertex_bindings.size()) {
            break;
        }
        vertex_bindings[binding] = {bindings.buffers[i]->Handle(), bindings.offsets[i],
                                    bindings.sizes[i]};
        num_vertex_bindings = std::max(num_vertex_bindings, binding + 1);
    }
}

std::span<u8> BufferCacheRuntime::BindMappedUniformBuffer(size_t, u32, u32 size) {
    BufferDescriptor& desc = descriptors.emplace_back();
    desc.is_uniform = true;
    desc.data.resize(size);
    desc.size = size;
    return desc.data;
}

void BufferCacheRuntime::BindUniformBuffer(const Buffer& buffer, u32 offset, u32 size) {
    // Uniform contents are copied per draw so bindings can always be padded to what the shader
    // declares; the guest memory backs the cached buffer.
    BufferDescriptor& desc = descriptors.emplace_back();
    desc.is_uniform = true;
    desc.size = size;
    desc.data.resize(size);
    if (buffer.CpuAddr() != 0) {
        device_memory.ReadBlockUnsafe(buffer.CpuAddr() + offset, desc.data.data(), size);
    }
    static u32 bind_u_logs = 0;
    if (bind_u_logs++ < 20) {
        size_t nonzero = 0;
        for (u8 b : desc.data) if (b != 0) ++nonzero;
        LOG_INFO(Render, "BindUniformBuffer #{}: cpu_addr={:#x} off={} size={} nonzero={}",
                 bind_u_logs, buffer.CpuAddr(), offset, size, nonzero);
    }
}

void BufferCacheRuntime::BindStorageBuffer(const Buffer& buffer, u32 offset, u32 size, bool) {
    BufferDescriptor& desc = descriptors.emplace_back();
    desc.is_uniform = false;
    desc.buffer = buffer.Handle();
    desc.offset = offset;
    desc.size = Common::AlignUp(std::max<u64>(size, 4), 4);
}

} // namespace WebGPU
