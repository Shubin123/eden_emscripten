// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_webgpu/wgpu_present.h"
#include "video_core/renderer_webgpu/wgpu_rasterizer.h"

namespace WebGPU {

class RendererWebGPU final : public VideoCore::RendererBase {
public:
    explicit RendererWebGPU(Core::Frontend::EmuWindow& emu_window,
                            Tegra::MaxwellDeviceMemoryManager& device_memory, Tegra::GPU& gpu,
                            std::unique_ptr<Core::Frontend::GraphicsContext> context);
    ~RendererWebGPU() override;

    void Composite(std::span<const Tegra::FramebufferConfig> layers) override;

    std::vector<u8> GetAppletCaptureBuffer() override;

    VideoCore::RasterizerInterface* ReadRasterizer() override {
        return &*rasterizer;
    }

    [[nodiscard]] std::string GetDeviceVendor() const override {
        return "WebGPU";
    }

    /// Returns whether a frame was presented since the last call (used to yield to the browser).
    bool TakePresented() {
        return std::exchange(presented, false);
    }

private:
    Tegra::MaxwellDeviceMemoryManager& device_memory;
    Tegra::GPU& gpu;
    std::optional<RasterizerWebGPU> rasterizer;
    std::optional<Presenter> presenter;
    bool presented = false;
};

} // namespace WebGPU
