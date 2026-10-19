// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/renderer_webgpu/wgpu_texture_cache.h"
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCommon {
template class VideoCommon::TextureCache<WebGPU::TextureCacheParams>;
}
