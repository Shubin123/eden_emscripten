// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <vector>

#include <webgpu/webgpu_cpp.h>

#include "common/common_types.h"
#include "video_core/surface.h"

namespace WebGPU {

/// CPU-side repacking needed before uploading a guest format WebGPU lacks.
enum class Conversion : u8 {
    None,
    // 16-bit packed formats decoded to RGBA8; names give the guest bit layout, MSB first.
    R5G6B5,
    B5G6R5,
    A1R5G5B5,
    A1B5G5R5,
    A5B5G5R1,
    A4B4G4R4,
    G4R4,
    A2R10G10B10, ///< To RGB10A2 (red and blue swapped)
    RGB32F,      ///< To RGBA32F with alpha one
};

struct FormatInfo {
    wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
    Conversion conversion = Conversion::None;
    bool renderable = false;
    bool depth = false;
    bool stencil = false;
};

const FormatInfo& GetFormatInfo(VideoCore::Surface::PixelFormat format);

/// Bytes per block of the host format after conversion.
u32 HostBytesPerBlock(VideoCore::Surface::PixelFormat format);

/// Repacks `width` x `rows` texels of `input` (tightly packed guest data) into host layout.
void ConvertTexels(Conversion conversion, std::span<const u8> input, std::vector<u8>& output,
                   size_t texels);

/// Whether a view can use `view` format on a texture created with `base` format.
bool IsViewCompatible(wgpu::TextureFormat base, wgpu::TextureFormat view);

/// sRGB/linear twin of a format, or Undefined.
wgpu::TextureFormat SrgbTwin(wgpu::TextureFormat format);

/// Whether a host format has a stencil aspect (render passes then need stencil load/store ops).
bool HasStencilAspect(wgpu::TextureFormat format);

} // namespace WebGPU
