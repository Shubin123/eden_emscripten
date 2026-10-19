// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>

#include <webgpu/webgpu_cpp.h>

#include "common/common_types.h"
#include "video_core/renderer_webgpu/wgpu_scheduler.h"

namespace WebGPU {

/// Append-only buffer for small per-draw data. Queue writes to fresh offsets cannot race with
/// commands still being recorded; on wrap-around the recorded commands are submitted first.
class UniformRing {
public:
    static constexpr u32 ALIGNMENT = 256;

    explicit UniformRing(Scheduler& scheduler, u64 size = 4ULL << 20,
                         wgpu::BufferUsage usage = wgpu::BufferUsage::Uniform |
                                                   wgpu::BufferUsage::Storage);

    /// Copies `data` into the ring and returns its offset.
    u32 Push(std::span<const u8> data);

    const wgpu::Buffer& Buffer() const noexcept {
        return buffer;
    }

private:
    Scheduler& scheduler;
    wgpu::Buffer buffer;
    u64 size;
    u64 cursor = 0;
};

} // namespace WebGPU
