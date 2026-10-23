// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <span>
#include <vector>

#include <webgpu/webgpu_cpp.h>

#include "common/common_types.h"

namespace WebGPU {

/// Attachments of a render pass, with the formats pipelines must be built against.
struct RenderTargets {
    std::array<wgpu::TextureView, 8> colors{};
    /// Textures behind the attachments, to detect attachments that are also sampled.
    std::array<WGPUTexture, 9> textures{};
    std::array<wgpu::TextureFormat, 8> color_formats{};
    wgpu::TextureView depth_stencil{};
    wgpu::TextureFormat depth_format = wgpu::TextureFormat::Undefined;
    bool has_stencil = false;
    u32 width = 0;
    u32 height = 0;
    u32 samples = 1;

    bool operator==(const RenderTargets& rhs) const noexcept {
        for (size_t i = 0; i < colors.size(); ++i) {
            if (colors[i].Get() != rhs.colors[i].Get()) {
                return false;
            }
        }
        return depth_stencil.Get() == rhs.depth_stencil.Get() && width == rhs.width &&
               height == rhs.height;
    }
};

struct ClearRequest {
    u32 color_mask = 0; ///< Attachments to clear
    std::array<wgpu::Color, 8> colors{};
    bool depth = false;
    f32 depth_value = 0.0f;
    bool stencil = false;
    u32 stencil_value = 0;
};

/// Owns the command encoder and the open render pass. WebGPU queue writes run before commands
/// still being recorded, so callers flush via BeforeQueueWrite first.
class Scheduler {
public:
    /// Returns the encoder for copies, ending any render pass.
    wgpu::CommandEncoder& Encoder();

    /// Makes `targets` the open render pass, beginning a new pass if they changed or a clear is
    /// pending. Returns the pass encoder.
    wgpu::RenderPassEncoder& RenderPass(const RenderTargets& targets);

    /// Clears attachments of `targets` by beginning a pass with clear load operations.
    void Clear(const RenderTargets& targets, const ClearRequest& request);

    bool InRenderPass() const {
        return static_cast<bool>(pass);
    }

    const RenderTargets& CurrentTargets() const {
        return current_targets;
    }

    /// Incremented whenever a new render pass begins (pass state must be bound again).
    u64 PassSerial() const {
        return pass_serial;
    }

    void EndRenderPass();

    /// WebGPU rejects a render pass that writes a buffer as storage and uses it any other way.
    /// Returns whether a draw with these buffer usages would break that in the open pass.
    bool PassConflicts(std::span<const WGPUBuffer> written, std::span<const WGPUBuffer> read) const;
    void TrackPassUsage(std::span<const WGPUBuffer> written, std::span<const WGPUBuffer> read);

    /// Submits recorded commands.
    void Flush();

    void BeforeQueueWrite() {
        if (has_commands) {
            Flush();
        }
    }

    u64 SubmitCount() const {
        return submit_count;
    }

private:
    void EnsureEncoder();
    void BeginPass(const RenderTargets& targets, const ClearRequest* clear);

    wgpu::CommandEncoder encoder;
    wgpu::RenderPassEncoder pass;
    std::vector<WGPUBuffer> pass_written;
    std::vector<WGPUBuffer> pass_read;
    RenderTargets current_targets;
    bool has_commands = false;
    u64 pass_serial = 0;
    u64 submit_count = 0;
};

} // namespace WebGPU
