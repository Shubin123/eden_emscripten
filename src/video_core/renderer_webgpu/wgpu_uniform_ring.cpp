// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/alignment.h"
#include "video_core/renderer_webgpu/wgpu_thread.h"
#include "video_core/renderer_webgpu/wgpu_uniform_ring.h"

namespace WebGPU {

UniformRing::UniformRing(Scheduler& scheduler_, u64 size_, wgpu::BufferUsage usage)
    : scheduler{scheduler_}, size{size_} {
    wgpu::BufferDescriptor desc{};
    desc.size = size;
    desc.usage = usage | wgpu::BufferUsage::CopyDst;
    buffer = Thread::Get().device.CreateBuffer(&desc);
}

u32 UniformRing::Push(std::span<const u8> data) {
    const u64 length = Common::AlignUp(std::max<u64>(data.size(), 16), 16);
    if (cursor + length > size) {
        // Data written earlier in this ring may still be read by unsubmitted commands.
        scheduler.Flush();
        cursor = 0;
    }
    const u64 offset = cursor;
    Thread::Get().queue.WriteBuffer(buffer, offset, data.data(), data.size());
    cursor = Common::AlignUp(offset + length, ALIGNMENT);
    return static_cast<u32>(offset);
}

} // namespace WebGPU
