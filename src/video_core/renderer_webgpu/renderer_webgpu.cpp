// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <stdexcept>

#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "common/logging.h"
#include "video_core/capture.h"
#include "video_core/gpu.h"
#include "video_core/renderer_webgpu/renderer_webgpu.h"
#include "video_core/renderer_webgpu/wgpu_stats.h"
#include "video_core/renderer_webgpu/wgpu_thread.h"

namespace WebGPU {

RendererWebGPU::RendererWebGPU(Core::Frontend::EmuWindow& emu_window,
                               Tegra::MaxwellDeviceMemoryManager& device_memory_,
                               Tegra::GPU& gpu_,
                               std::unique_ptr<Core::Frontend::GraphicsContext> context_)
    : RendererBase(emu_window, std::move(context_)), device_memory{device_memory_}, gpu{gpu_} {
    auto& thread = Thread::Get();
    if (!thread.Start()) {
        throw std::runtime_error("WebGPU is not available");
    }
    // WebGPU objects must be created on the thread that uses them.
    thread.RunSync([&] {
        rasterizer.emplace(gpu, device_memory);
        presenter.emplace(device_memory);
    });
}

RendererWebGPU::~RendererWebGPU() {
    Thread::Get().RunSync([&] {
        presenter.reset();
        rasterizer.reset();
    });
}

void RendererWebGPU::Composite(std::span<const Tegra::FramebufferConfig> layers) {
    if (layers.empty()) {
        return;
    }
    auto& thread = Thread::Get();
    rasterizer->FlushCommands();
    const bool present = m_current_frame % std::max(present_interval.load(std::memory_order_relaxed), 1u) == 0;
    wgpu::CommandEncoder encoder = thread.device.CreateCommandEncoder();
    if (present && presenter->BeginFrame(encoder)) {
        for (const auto& layer : layers) {
            u32 width = layer.width;
            u32 height = layer.height;
            wgpu::TextureView view = rasterizer->AccelerateDisplay(layer, width, height);

            if (!view) {
                view = presenter->LoadFromMemory(layer);
            }
            if (view) {
                presenter->Draw(encoder, view, width, height,
                                Tegra::NormalizeCrop(layer, width, height));
            }
        }
        const wgpu::CommandBuffer commands = encoder.Finish();
        thread.queue.Submit(1, &commands);
        presenter->EndFrame();
    }
    presented = true;
    if (++m_current_frame % 120 == 0) {
        const auto& stats = rasterizer->stats;
        LOG_INFO(Render, "WebGPU frame {}: draws {} (actual {}) clears {} display hits {} misses {} layers {} {}x{}",
                 m_current_frame, stats.draws, stats.actual_draws, stats.clears, stats.display_hits,
                 stats.display_misses, layers.size(), layers[0].width, layers[0].height);
    }
    rasterizer->TickFrame();
    gpu.RendererFrameEndNotify();
    render_window.OnFrameDisplayed();
}

std::vector<u8> RendererWebGPU::GetAppletCaptureBuffer() {
    return std::vector<u8>(VideoCore::Capture::TiledSize);
}

} // namespace WebGPU
