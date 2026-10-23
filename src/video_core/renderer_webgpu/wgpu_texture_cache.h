// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <span>
#include <unordered_map>
#include <memory>
#include <vector>

#include <webgpu/webgpu_cpp.h>

#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_webgpu/wgpu_formats.h"
#include "video_core/renderer_webgpu/wgpu_scheduler.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/texture_cache_base.h"

namespace WebGPU {

using Common::SlotVector;
using VideoCommon::ImageId;
using VideoCommon::NUM_RT;
using VideoCommon::Region2D;
using VideoCore::Surface::PixelFormat;

class Blitter;
class Framebuffer;
class Image;
class ImageView;
class Sampler;

/// How a texture binding must be declared for a given view.
enum class TextureKind : u8 {
    Float,
    UnfilterableFloat,
    Depth,
    Sint,
    Uint,
};

/// CPU memory standing in for a staging buffer: uploads go through queue.WriteTexture.
struct StagingBufferMap {
    /// Left uninitialized: callers overwrite the mapped span before it is read.
    std::shared_ptr<u8[]> storage;
    std::span<u8> mapped_span;
    size_t offset = 0;
    wgpu::Buffer buffer;
};

class TextureCacheRuntime {
public:
    explicit TextureCacheRuntime(Scheduler& scheduler, Blitter& blitter);

    void Finish();

    StagingBufferMap UploadStagingBuffer(size_t size, bool deferred = false);
    StagingBufferMap DownloadStagingBuffer(size_t size, bool deferred = false);
    void FreeDeferredStagingBuffer(StagingBufferMap&) {}

    void TickFrame() {}
    void FlushDeferredClear() {}

    u64 GetDeviceLocalMemory() const {
        return 2ULL << 30;
    }
    u64 GetDeviceMemoryUsage() const {
        return memory_usage;
    }
    bool CanReportMemoryUsage() const {
        return true;
    }
    bool CanDownloadMsaa(const VideoCommon::ImageInfo&) const {
        return false;
    }

    void BlitImage(Framebuffer* dst_framebuffer, ImageView& dst, ImageView& src,
                   const Region2D& dst_region, const Region2D& src_region,
                   Tegra::Engines::Fermi2D::Filter filter,
                   Tegra::Engines::Fermi2D::Operation operation);

    void CopyImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);
    void CopyImageMSAA(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies) {
        CopyImage(dst, src, copies);
    }
    bool ShouldReinterpret(Image& dst, Image& src);
    void ReinterpretImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);
    void ConvertImage(Framebuffer* dst, ImageView& dst_view, ImageView& src_view);
    bool CanImageBeCopied(const Image& dst, const Image& src);
    void EmulateCopyImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    bool CanAccelerateImageUpload(Image&) const noexcept {
        return false;
    }
    bool CanUploadMSAA() const noexcept {
        return true;
    }
    void AccelerateImageUpload(Image&, const StagingBufferMap&,
                               std::span<const VideoCommon::SwizzleParameters>, u32, u32) {}
    void InsertUploadMemoryBarrier() {}
    void TransitionImageLayout(Image&) {}
    bool HasBrokenTextureViewFormats() const noexcept {
        return false;
    }
    bool HasNativeBgr() const noexcept {
        return true;
    }
    void BarrierFeedbackLoop() {}

    static bool CoversLevel(const Image& image, s32 level, const VideoCommon::Offset3D& offset,
                            const VideoCommon::Extent3D& extent);
    void BlitDepthCopy(Image& dst, Image& src, const VideoCommon::ImageCopy& copy);
    void CopyImageThroughBuffer(Image& dst, Image& src,
                                std::span<const VideoCommon::ImageCopy> copies);

    Scheduler& scheduler;
    Blitter& blitter;
    u64 memory_usage = 0;
    wgpu::Buffer copy_buffer;
    wgpu::TextureView last_blit_target;
    u32 last_blit_width = 0;
    u32 last_blit_height = 0;
    u64 last_blit_serial = 0;
};

class Image : public VideoCommon::ImageBase {
public:
    explicit Image(TextureCacheRuntime&, const VideoCommon::ImageInfo& info, GPUVAddr gpu_addr,
                   VAddr cpu_addr);
    explicit Image(const VideoCommon::NullImageParams&);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&&) = default;
    Image& operator=(Image&&) = default;

    void UploadMemory(const StagingBufferMap& map,
                      std::span<const VideoCommon::BufferImageCopy> copies);
    void DownloadMemory(const StagingBufferMap& map,
                        std::span<const VideoCommon::BufferImageCopy> copies);
    void DownloadMemory(std::span<wgpu::Buffer>, std::span<size_t>,
                        std::span<const VideoCommon::BufferImageCopy>) {}
    void DownloadMemory(const wgpu::Buffer&, size_t, std::span<const VideoCommon::BufferImageCopy>) {}

    bool IsRescaled() const noexcept {
        return false;
    }
    bool ScaleUp(bool = false) {
        return false;
    }
    bool ScaleDown(bool = false) {
        return false;
    }

    const wgpu::Texture& Texture() const noexcept {
        return texture;
    }
    wgpu::TextureFormat Format() const noexcept {
        return host_format;
    }

    u64 allocation_tick{};

private:
    TextureCacheRuntime* runtime = nullptr;
    wgpu::Texture texture;
    wgpu::TextureFormat host_format = wgpu::TextureFormat::Undefined;
    Conversion conversion = Conversion::None;
    std::vector<u8> conversion_buffer;
};

class ImageView : public VideoCommon::ImageViewBase {
public:
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageViewInfo&, ImageId, Image&);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageViewInfo&, ImageId, Image&,
                       const SlotVector<Image>&);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageInfo&,
                       const VideoCommon::ImageViewInfo&, GPUVAddr);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::NullImageViewParams&);
    ~ImageView();

    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;
    ImageView(ImageView&&) = default;
    ImageView& operator=(ImageView&&) = default;

    /// Sampled view for the shader texture type, or a null view.
    const wgpu::TextureView& Handle(Shader::TextureType texture_type) const noexcept {
        return views[static_cast<size_t>(texture_type)];
    }

    const wgpu::TextureView& RenderTarget() const noexcept {
        return render_target;
    }

    wgpu::TextureFormat HostFormat() const noexcept {
        return host_format;
    }

    TextureKind Kind() const noexcept {
        return kind;
    }

    const wgpu::Texture& BaseTexture() const noexcept {
        return texture;
    }

    bool IsRescaled() const noexcept {
        return false;
    }

    bool HasStencil() const noexcept {
        return has_stencil;
    }

    GPUVAddr GpuAddr() const noexcept {
        return gpu_addr;
    }

    u32 BufferSize() const noexcept {
        return buffer_size;
    }

    /// Width and height of the viewed level.
    u32 Width() const noexcept {
        return view_width;
    }
    u32 Height() const noexcept {
        return view_height;
    }

private:
    std::array<wgpu::TextureView, Shader::NUM_TEXTURE_TYPES> views{};
    wgpu::TextureView render_target;
    wgpu::Texture texture;
    wgpu::TextureFormat host_format = wgpu::TextureFormat::Undefined;
    TextureKind kind = TextureKind::Float;
    bool has_stencil = false;
    u32 buffer_size = 0;
    u32 view_width = 0;
    u32 view_height = 0;
};

class ImageAlloc : public VideoCommon::ImageAllocBase {};

class Sampler {
public:
    explicit Sampler(TextureCacheRuntime&, const Tegra::Texture::TSCEntry&);

    /// Sampler for a texture of `kind`; comparison samplers for depth compare reads.
    const wgpu::Sampler& Handle(TextureKind kind, bool compare);

private:
    wgpu::SamplerDescriptor desc{};
    wgpu::CompareFunction compare_function = wgpu::CompareFunction::Undefined;
    wgpu::Sampler filtering;
    wgpu::Sampler non_filtering;
    wgpu::Sampler comparison;
};

class Framebuffer {
public:
    explicit Framebuffer(TextureCacheRuntime&, std::span<ImageView*, NUM_RT> color_buffers,
                         ImageView* depth_buffer, const VideoCommon::RenderTargets& key);
    explicit Framebuffer(TextureCacheRuntime&, ImageView* color_buffer, ImageView* depth_buffer,
                         u32 width, u32 height);

    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;
    Framebuffer(Framebuffer&&) = default;
    Framebuffer& operator=(Framebuffer&&) = default;

    const RenderTargets& Targets() const noexcept {
        return targets;
    }

private:
    RenderTargets targets;
};

struct TextureCacheParams {
    static constexpr bool ENABLE_VALIDATION = true;
    static constexpr bool FRAMEBUFFER_BLITS = false;
    static constexpr bool HAS_EMULATED_COPIES = false;
    static constexpr bool HAS_DEVICE_MEMORY_INFO = true;
    static constexpr bool IMPLEMENTS_ASYNC_DOWNLOADS = false;
    static constexpr bool HAS_MSAA_DOWNLOADS = false;

    using Runtime = WebGPU::TextureCacheRuntime;
    using Image = WebGPU::Image;
    using ImageAlloc = WebGPU::ImageAlloc;
    using ImageView = WebGPU::ImageView;
    using Sampler = WebGPU::Sampler;
    using Framebuffer = WebGPU::Framebuffer;
    using AsyncBuffer = WebGPU::StagingBufferMap;
    using BufferType = wgpu::Buffer;
};

using TextureCache = VideoCommon::TextureCache<TextureCacheParams>;

} // namespace WebGPU
