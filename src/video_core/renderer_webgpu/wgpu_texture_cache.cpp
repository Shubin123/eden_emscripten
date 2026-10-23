// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <cmath>

#include "common/alignment.h"
#include "common/logging.h"
#include "video_core/renderer_webgpu/wgpu_blit.h"
#include "video_core/renderer_webgpu/wgpu_texture_cache.h"
#include "video_core/renderer_webgpu/wgpu_stats.h"
#include "video_core/renderer_webgpu/wgpu_thread.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/formatter.h"
#include "video_core/texture_cache/samples_helper.h"
#include "video_core/texture_cache/util.h"
#include "video_core/textures/texture.h"

namespace WebGPU {

using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCommon::ImageViewType;
using VideoCore::Surface::BytesPerBlock;
using VideoCore::Surface::DefaultBlockHeight;
using VideoCore::Surface::DefaultBlockWidth;
using Tegra::Texture::SwizzleSource;

namespace {

wgpu::Device& Device() {
    return Thread::Get().device;
}

bool IsIntegerFormat(PixelFormat format) {
    return VideoCore::Surface::IsPixelFormatInteger(format);
}

TextureKind KindOf(PixelFormat format) {
    const FormatInfo& info = GetFormatInfo(format);
    if (info.depth || info.stencil) {
        return info.depth ? TextureKind::Depth : TextureKind::Uint;
    }
    if (IsIntegerFormat(format)) {
        return VideoCore::Surface::IsPixelFormatSignedInteger(format) ? TextureKind::Sint
                                                                       : TextureKind::Uint;
    }
    return TextureKind::Float;
}

/// Formats that sample as float but cannot be filtered on this device, so they must be bound as
/// unfilterable-float with a non-filtering sampler.
bool IsUnfilterable(wgpu::TextureFormat format) {
    switch (format) {
    case wgpu::TextureFormat::R16Unorm:
    case wgpu::TextureFormat::RG16Unorm:
    case wgpu::TextureFormat::RGBA16Unorm:
    case wgpu::TextureFormat::R16Snorm:
    case wgpu::TextureFormat::RG16Snorm:
    case wgpu::TextureFormat::RGBA16Snorm:
        return true;
    case wgpu::TextureFormat::R32Float:
    case wgpu::TextureFormat::RG32Float:
    case wgpu::TextureFormat::RGBA32Float:
        return !Device().HasFeature(wgpu::FeatureName::Float32Filterable);
    default:
        return false;
    }
}

wgpu::ComponentSwizzle ToComponent(SwizzleSource source) {
    switch (source) {
    case SwizzleSource::Zero:
        return wgpu::ComponentSwizzle::Zero;
    case SwizzleSource::R:
        return wgpu::ComponentSwizzle::R;
    case SwizzleSource::G:
        return wgpu::ComponentSwizzle::G;
    case SwizzleSource::B:
        return wgpu::ComponentSwizzle::B;
    case SwizzleSource::A:
        return wgpu::ComponentSwizzle::A;
    case SwizzleSource::OneInt:
    case SwizzleSource::OneFloat:
        return wgpu::ComponentSwizzle::One;
    }
    return wgpu::ComponentSwizzle::Zero;
}

/// Size of the texture backing `info`; 1D images become 2D and compressed sizes are padded to
/// whole blocks as WebGPU requires.
wgpu::Extent3D TextureExtent(const ImageInfo& info) {
    const u32 block_w = DefaultBlockWidth(info.format);
    const u32 block_h = DefaultBlockHeight(info.format);
    wgpu::Extent3D extent{Common::AlignUp(std::max(info.size.width, 1u), block_w),
                          Common::AlignUp(std::max(info.size.height, 1u), block_h), 1};
    if (info.type == ImageType::e3D) {
        extent.depthOrArrayLayers = std::max(info.size.depth, 1u);
    } else {
        extent.depthOrArrayLayers = std::max(info.resources.layers, 1);
    }
    return extent;
}

wgpu::TextureViewDimension ViewDimension(Shader::TextureType type) {
    switch (type) {
    case Shader::TextureType::Color1D:
    case Shader::TextureType::Color2D:
    case Shader::TextureType::Color2DRect:
        return wgpu::TextureViewDimension::e2D;
    case Shader::TextureType::ColorArray1D:
    case Shader::TextureType::ColorArray2D:
        return wgpu::TextureViewDimension::e2DArray;
    case Shader::TextureType::Color3D:
        return wgpu::TextureViewDimension::e3D;
    case Shader::TextureType::ColorCube:
        return wgpu::TextureViewDimension::Cube;
    case Shader::TextureType::ColorArrayCube:
        return wgpu::TextureViewDimension::CubeArray;
    case Shader::TextureType::Buffer:
        break;
    }
    return wgpu::TextureViewDimension::Undefined;
}

wgpu::AddressMode AddressMode(Tegra::Texture::WrapMode mode) {
    using Tegra::Texture::WrapMode;
    switch (mode) {
    case WrapMode::Wrap:
        return wgpu::AddressMode::Repeat;
    case WrapMode::Mirror:
        return wgpu::AddressMode::MirrorRepeat;
    default:
        // WebGPU has no border colors or mirror-once modes.
        return wgpu::AddressMode::ClampToEdge;
    }
}

wgpu::CompareFunction CompareFunction(Tegra::Texture::DepthCompareFunc func) {
    using Tegra::Texture::DepthCompareFunc;
    switch (func) {
    case DepthCompareFunc::Never:
        return wgpu::CompareFunction::Never;
    case DepthCompareFunc::Less:
        return wgpu::CompareFunction::Less;
    case DepthCompareFunc::Equal:
        return wgpu::CompareFunction::Equal;
    case DepthCompareFunc::LessEqual:
        return wgpu::CompareFunction::LessEqual;
    case DepthCompareFunc::Greater:
        return wgpu::CompareFunction::Greater;
    case DepthCompareFunc::NotEqual:
        return wgpu::CompareFunction::NotEqual;
    case DepthCompareFunc::GreaterEqual:
        return wgpu::CompareFunction::GreaterEqual;
    case DepthCompareFunc::Always:
        return wgpu::CompareFunction::Always;
    }
    return wgpu::CompareFunction::Always;
}

void LogOnce(bool& logged, const char* what) {
    if (!logged) {
        logged = true;
        LOG_WARNING(Render, "WebGPU: {} is not implemented", what);
    }
}

} // Anonymous namespace

TextureCacheRuntime::TextureCacheRuntime(Scheduler& scheduler_, Blitter& blitter_)
    : scheduler{scheduler_}, blitter{blitter_} {}

void TextureCacheRuntime::Finish() {
    scheduler.Flush();
}

StagingBufferMap TextureCacheRuntime::UploadStagingBuffer(size_t size, bool) {
    StagingBufferMap map;
    map.storage = std::make_shared_for_overwrite<u8[]>(size);
    map.mapped_span = std::span<u8>(map.storage.get(), size);
    return map;
}

StagingBufferMap TextureCacheRuntime::DownloadStagingBuffer(size_t size, bool) {
    // Downloads are not read back from the GPU yet, so hand out zeros rather than garbage.
    StagingBufferMap map = UploadStagingBuffer(size);
    std::fill(map.mapped_span.begin(), map.mapped_span.end(), u8{0});
    return map;
}

void TextureCacheRuntime::BlitImage(Framebuffer*, ImageView& dst, ImageView& src,
                                    const Region2D& dst_region, const Region2D& src_region,
                                    Tegra::Engines::Fermi2D::Filter filter,
                                    Tegra::Engines::Fermi2D::Operation) {
    static u32 blit_logs = 0;
    if (blit_logs++ < 20) {
        LOG_INFO(Render, "TextureCacheRuntime::BlitImage #{}: dst_rt={} src_handle={} dst={}x{} src={}x{}",
                 blit_logs, dst.RenderTarget() != nullptr, src.Handle(Shader::TextureType::Color2D) != nullptr,
                 dst.Width(), dst.Height(), src.Width(), src.Height());
    }
    if (!dst.RenderTarget() || !src.Handle(Shader::TextureType::Color2D)) {
        return;
    }
    blitter.Blit(dst.RenderTarget(), dst.HostFormat(), dst.Width(), dst.Height(), dst_region,
                 src.Handle(Shader::TextureType::Color2D), src.Kind(), src.Width(), src.Height(),
                 src_region, filter == Tegra::Engines::Fermi2D::Filter::Bilinear);
    if (dst.Width() >= 640 && dst.Height() >= 360) {
        last_blit_target = dst.Handle(Shader::TextureType::Color2D);
        last_blit_width = dst.Width();
        last_blit_height = dst.Height();
        ++last_blit_serial;
    }
}

void TextureCacheRuntime::CopyImage(Image& dst, Image& src,
                                    std::span<const VideoCommon::ImageCopy> copies) {
    if (!dst.Texture() || !src.Texture()) {
        return;
    }
    const bool compatible = dst.Format() == src.Format() || SrgbTwin(dst.Format()) == src.Format();
    if (!compatible) {
        CopyImageThroughBuffer(dst, src, copies);
        return;
    }
    const FormatInfo& format_info = GetFormatInfo(dst.info.format);
    const bool depth_stencil = format_info.depth || format_info.stencil;
    const bool is_3d_src = src.info.type == ImageType::e3D;
    const bool is_3d_dst = dst.info.type == ImageType::e3D;
    for (const VideoCommon::ImageCopy& copy : copies) {
        // WebGPU only copies whole depth/stencil subresources; draw partial copies instead.
        if (depth_stencil && (!CoversLevel(src, copy.src_subresource.base_level, copy.src_offset,
                                           copy.extent) ||
                              !CoversLevel(dst, copy.dst_subresource.base_level, copy.dst_offset,
                                           copy.extent))) {
            BlitDepthCopy(dst, src, copy);
            continue;
        }
        auto& encoder = scheduler.Encoder();
        wgpu::TexelCopyTextureInfo src_info{};
        src_info.texture = src.Texture();
        src_info.mipLevel = static_cast<u32>(copy.src_subresource.base_level);
        src_info.origin = {static_cast<u32>(copy.src_offset.x), static_cast<u32>(copy.src_offset.y),
                           is_3d_src ? static_cast<u32>(copy.src_offset.z)
                                     : static_cast<u32>(copy.src_subresource.base_layer)};
        wgpu::TexelCopyTextureInfo dst_info{};
        dst_info.texture = dst.Texture();
        dst_info.mipLevel = static_cast<u32>(copy.dst_subresource.base_level);
        dst_info.origin = {static_cast<u32>(copy.dst_offset.x), static_cast<u32>(copy.dst_offset.y),
                           is_3d_dst ? static_cast<u32>(copy.dst_offset.z)
                                     : static_cast<u32>(copy.dst_subresource.base_layer)};
        const u32 layers = is_3d_src || is_3d_dst ? copy.extent.depth
                                                  : static_cast<u32>(copy.src_subresource.num_layers);
        const wgpu::Extent3D extent{copy.extent.width, copy.extent.height, std::max(layers, 1u)};
        encoder.CopyTextureToTexture(&src_info, &dst_info, &extent);
    }
}

void TextureCacheRuntime::CopyImageThroughBuffer(Image& dst, Image& src,
                                                 std::span<const VideoCommon::ImageCopy> copies) {
    // Formats with the same texel size are copied bit for bit through a buffer, as the guest
    // reinterprets them; WebGPU only copies textures of the same format directly.
    const FormatInfo& src_info = GetFormatInfo(src.info.format);
    const FormatInfo& dst_info = GetFormatInfo(dst.info.format);
    const u32 bytes = HostBytesPerBlock(src.info.format);
    const bool plain = [](const Image& image) {
        return !True(image.flags & VideoCommon::ImageFlagBits::Converted) &&
               DefaultBlockWidth(image.info.format) == 1 && DefaultBlockHeight(image.info.format) == 1;
    }(src) && [](const Image& image) {
        return !True(image.flags & VideoCommon::ImageFlagBits::Converted) &&
               DefaultBlockWidth(image.info.format) == 1 && DefaultBlockHeight(image.info.format) == 1;
    }(dst);
    if (!plain || bytes == 0 || bytes != HostBytesPerBlock(dst.info.format) || src_info.depth ||
        src_info.stencil || dst_info.depth || dst_info.stencil ||
        src_info.conversion != Conversion::None || dst_info.conversion != Conversion::None) {
        DebugCount(fmt::format("copy {} -> {} unsupported", src.info.format, dst.info.format));
        return;
    }
    auto& encoder = scheduler.Encoder();
    for (const VideoCommon::ImageCopy& copy : copies) {
        const u32 layers = std::max(src.info.type == ImageType::e3D
                                        ? copy.extent.depth
                                        : static_cast<u32>(copy.src_subresource.num_layers),
                                    1u);
        const u32 row_bytes = Common::AlignUp(copy.extent.width * bytes, 256u);
        const u64 size = static_cast<u64>(row_bytes) * copy.extent.height * layers;
        if (!copy_buffer || copy_buffer.GetSize() < size) {
            wgpu::BufferDescriptor desc{};
            desc.size = std::bit_ceil(std::max<u64>(size, 1ULL << 20));
            desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
            copy_buffer = Device().CreateBuffer(&desc);
        }
        wgpu::TexelCopyTextureInfo src_texture{};
        src_texture.texture = src.Texture();
        src_texture.mipLevel = static_cast<u32>(copy.src_subresource.base_level);
        src_texture.origin = {static_cast<u32>(copy.src_offset.x),
                              static_cast<u32>(copy.src_offset.y),
                              src.info.type == ImageType::e3D
                                  ? static_cast<u32>(copy.src_offset.z)
                                  : static_cast<u32>(copy.src_subresource.base_layer)};
        wgpu::TexelCopyTextureInfo dst_texture{};
        dst_texture.texture = dst.Texture();
        dst_texture.mipLevel = static_cast<u32>(copy.dst_subresource.base_level);
        dst_texture.origin = {static_cast<u32>(copy.dst_offset.x),
                              static_cast<u32>(copy.dst_offset.y),
                              dst.info.type == ImageType::e3D
                                  ? static_cast<u32>(copy.dst_offset.z)
                                  : static_cast<u32>(copy.dst_subresource.base_layer)};
        wgpu::TexelCopyBufferInfo buffer{};
        buffer.buffer = copy_buffer;
        buffer.layout.bytesPerRow = row_bytes;
        buffer.layout.rowsPerImage = copy.extent.height;
        const wgpu::Extent3D extent{copy.extent.width, copy.extent.height, layers};
        encoder.CopyTextureToBuffer(&src_texture, &buffer, &extent);
        encoder.CopyBufferToTexture(&buffer, &dst_texture, &extent);
    }
}

bool TextureCacheRuntime::CoversLevel(const Image& image, s32 level,
                                      const VideoCommon::Offset3D& offset,
                                      const VideoCommon::Extent3D& extent) {
    const u32 width = std::max(image.info.size.width >> level, 1u);
    const u32 height = std::max(image.info.size.height >> level, 1u);
    return offset.x == 0 && offset.y == 0 && extent.width == width && extent.height == height;
}

void TextureCacheRuntime::BlitDepthCopy(Image& dst, Image& src, const VideoCommon::ImageCopy& copy) {
    const FormatInfo& format_info = GetFormatInfo(dst.info.format);
    if (!format_info.depth || !format_info.renderable) {
        static bool logged = false;
        LogOnce(logged, "partial stencil copies");
        return;
    }
    const u32 src_level = static_cast<u32>(copy.src_subresource.base_level);
    const u32 dst_level = static_cast<u32>(copy.dst_subresource.base_level);
    const u32 layers = static_cast<u32>(std::max(copy.src_subresource.num_layers, 1));
    for (u32 layer = 0; layer < layers; ++layer) {
        wgpu::TextureViewDescriptor src_desc{};
        src_desc.dimension = wgpu::TextureViewDimension::e2D;
        src_desc.baseMipLevel = src_level;
        src_desc.mipLevelCount = 1;
        src_desc.baseArrayLayer = static_cast<u32>(copy.src_subresource.base_layer) + layer;
        src_desc.arrayLayerCount = 1;
        src_desc.aspect = wgpu::TextureAspect::DepthOnly;
        wgpu::TextureViewDescriptor dst_desc{};
        dst_desc.dimension = wgpu::TextureViewDimension::e2D;
        dst_desc.baseMipLevel = dst_level;
        dst_desc.mipLevelCount = 1;
        dst_desc.baseArrayLayer = static_cast<u32>(copy.dst_subresource.base_layer) + layer;
        dst_desc.arrayLayerCount = 1;
        const wgpu::TextureView src_view = src.Texture().CreateView(&src_desc);
        const wgpu::TextureView dst_view = dst.Texture().CreateView(&dst_desc);
        const s32 width = static_cast<s32>(copy.extent.width);
        const s32 height = static_cast<s32>(copy.extent.height);
        const Region2D src_region{{copy.src_offset.x, copy.src_offset.y},
                                  {copy.src_offset.x + width, copy.src_offset.y + height}};
        const Region2D dst_region{{copy.dst_offset.x, copy.dst_offset.y},
                                  {copy.dst_offset.x + width, copy.dst_offset.y + height}};
        blitter.Blit(dst_view, dst.Format(), std::max(dst.info.size.width >> dst_level, 1u),
                     std::max(dst.info.size.height >> dst_level, 1u), dst_region, src_view,
                     TextureKind::Depth, std::max(src.info.size.width >> src_level, 1u),
                     std::max(src.info.size.height >> src_level, 1u), src_region, false);
    }
}

bool TextureCacheRuntime::ShouldReinterpret(Image& dst, Image& src) {
    return false;
}

void TextureCacheRuntime::ReinterpretImage(Image& dst, Image& src,
                                           std::span<const VideoCommon::ImageCopy> copies) {
    static bool logged = false;
    LogOnce(logged, "image reinterpretation");
    DebugCount(fmt::format("reinterpret {} -> {}", src.info.format, dst.info.format));
}

void TextureCacheRuntime::ConvertImage(Framebuffer*, ImageView& dst_view, ImageView& src_view) {
    blitter.Blit(dst_view.RenderTarget(), dst_view.HostFormat(), dst_view.Width(),
                 dst_view.Height(), Region2D{{0, 0}, {static_cast<s32>(dst_view.Width()), static_cast<s32>(dst_view.Height())}},
                 src_view.Handle(Shader::TextureType::Color2D), src_view.Kind(), src_view.Width(),
                 src_view.Height(), Region2D{{0, 0}, {static_cast<s32>(src_view.Width()), static_cast<s32>(src_view.Height())}},
                 false);
}

bool TextureCacheRuntime::CanImageBeCopied(const Image& dst, const Image& src) {
    return dst.Format() == src.Format();
}

void TextureCacheRuntime::EmulateCopyImage(Image& dst, Image& src,
                                           std::span<const VideoCommon::ImageCopy> copies) {
    CopyImage(dst, src, copies);
}

Image::Image(TextureCacheRuntime& runtime_, const ImageInfo& info_, GPUVAddr gpu_addr_,
             VAddr cpu_addr_)
    : VideoCommon::ImageBase(info_, gpu_addr_, cpu_addr_), runtime{&runtime_} {
    const FormatInfo& format_info = GetFormatInfo(info.format);
    host_format = format_info.format;
    conversion = format_info.conversion;
    const bool decode_astc =
        VideoCore::Surface::IsPixelFormatASTC(info.format) &&
        (host_format == wgpu::TextureFormat::Undefined ||
         !Device().HasFeature(wgpu::FeatureName::TextureCompressionASTC) ||
         (info.type == ImageType::e3D && info.size.depth > 1 &&
          !Device().HasFeature(wgpu::FeatureName::TextureCompressionASTCSliced3D)));
    if (decode_astc) {
        flags |= VideoCommon::ImageFlagBits::Converted | VideoCommon::ImageFlagBits::CostlyLoad;
        host_format = VideoCore::Surface::IsPixelFormatSRGB(info.format)
                          ? wgpu::TextureFormat::RGBA8UnormSrgb
                          : wgpu::TextureFormat::RGBA8Unorm;
    }
    if (host_format == wgpu::TextureFormat::Undefined || info.type == ImageType::Buffer) {
        LOG_WARNING(Render, "WebGPU: unsupported image format {}", info.format);
        return;
    }
    if (info.type == ImageType::e3D && info.size.depth > 1 &&
        VideoCore::Surface::IsPixelFormatBCn(info.format) &&
        !Device().HasFeature(wgpu::FeatureName::TextureCompressionBCSliced3D)) {
        LOG_WARNING(Render, "WebGPU: 3D {} images need texture-compression-bc-sliced-3d",
                    info.format);
        return;
    }
    wgpu::TextureDescriptor desc{};
    desc.size = TextureExtent(info);
    if (decode_astc) {
        desc.size.width = std::max(info.size.width, 1u);
        desc.size.height = std::max(info.size.height, 1u);
    }
    // Single-slice 3D images are kept 2D: guests view them as 2D, which WebGPU forbids on 3D textures.
    desc.dimension = info.type == ImageType::e3D && info.size.depth > 1
                         ? wgpu::TextureDimension::e3D
                         : wgpu::TextureDimension::e2D;
    desc.format = host_format;
    desc.mipLevelCount = static_cast<u32>(std::max(info.resources.levels, 1));
    desc.sampleCount = 1;
    desc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc |
                 wgpu::TextureUsage::CopyDst;
    if (format_info.renderable && desc.dimension != wgpu::TextureDimension::e3D) {
        desc.usage |= wgpu::TextureUsage::RenderAttachment;
    }
    const wgpu::TextureFormat twin = SrgbTwin(host_format);
    if (twin != wgpu::TextureFormat::Undefined) {
        desc.viewFormatCount = 1;
        desc.viewFormats = &twin;
    }
    texture = Device().CreateTexture(&desc);
    runtime->memory_usage += decode_astc ? converted_size_bytes : guest_size_bytes;
}

Image::Image(const VideoCommon::NullImageParams& params) : VideoCommon::ImageBase{params} {}

Image::~Image() {
    if (runtime && texture) {
        const u64 host_size = True(flags & VideoCommon::ImageFlagBits::Converted)
                                  ? converted_size_bytes
                                  : guest_size_bytes;
        runtime->memory_usage -= std::min(runtime->memory_usage, host_size);
        texture.Destroy();
    }
}

void Image::UploadMemory(const StagingBufferMap& map,
                         std::span<const VideoCommon::BufferImageCopy> copies) {
    if (!texture) {
        return;
    }
    // WebGPU only accepts uploads into depth16unorm and stencil8 among depth formats.
    if (host_format == wgpu::TextureFormat::Depth32Float ||
        host_format == wgpu::TextureFormat::Depth24Plus ||
        host_format == wgpu::TextureFormat::Depth24PlusStencil8 ||
        host_format == wgpu::TextureFormat::Depth32FloatStencil8) {
        return;
    }
    static u32 upload_logs = 0;
    if (upload_logs++ < 30) {
        LOG_INFO(Render, "UploadMemory #{}: {}x{} fmt {} host_fmt {} copies {}",
                 upload_logs, info.size.width, info.size.height, static_cast<u32>(info.format),
                 static_cast<u32>(host_format), copies.size());
    }
    runtime->scheduler.BeforeQueueWrite();
    auto& queue = Thread::Get().queue;
    const bool decoded = True(flags & VideoCommon::ImageFlagBits::Converted);
    const u32 block_w = decoded ? 1 : DefaultBlockWidth(info.format);
    const u32 block_h = decoded ? 1 : DefaultBlockHeight(info.format);
    const u32 guest_bpb = decoded ? 4 : BytesPerBlock(info.format);
    const u32 host_bpb = decoded ? 4 : HostBytesPerBlock(info.format);
    const bool is_3d = info.type == ImageType::e3D;
    for (const VideoCommon::BufferImageCopy& copy : copies) {
        const u32 row_blocks = Common::DivCeil(copy.buffer_row_length, block_w);
        const u32 image_rows = Common::DivCeil(copy.buffer_image_height, block_h);
        const u32 layers = is_3d ? copy.image_extent.depth
                                 : static_cast<u32>(copy.image_subresource.num_layers);
        const size_t num_blocks = static_cast<size_t>(row_blocks) * image_rows * layers;
        std::span<const u8> data = map.mapped_span.subspan(map.offset + copy.buffer_offset,
                                                           num_blocks * guest_bpb);
        if (conversion != Conversion::None) {
            ConvertTexels(conversion, data, conversion_buffer, num_blocks);
            data = conversion_buffer;
        }
        wgpu::TexelCopyTextureInfo dst{};
        dst.texture = texture;
        dst.mipLevel = static_cast<u32>(copy.image_subresource.base_level);
        dst.origin = {static_cast<u32>(copy.image_offset.x), static_cast<u32>(copy.image_offset.y),
                      is_3d ? static_cast<u32>(copy.image_offset.z)
                            : static_cast<u32>(copy.image_subresource.base_layer)};
        wgpu::TexelCopyBufferLayout layout{};
        layout.bytesPerRow = row_blocks * host_bpb;
        layout.rowsPerImage = image_rows;
        // Copies of compressed formats cover whole blocks.
        const wgpu::Extent3D extent{Common::AlignUp(copy.image_extent.width, block_w),
                                    Common::AlignUp(copy.image_extent.height, block_h),
                                    std::max(layers, 1u)};
        queue.WriteTexture(&dst, data.data(), data.size(), &layout, &extent);
    }
}

void Image::DownloadMemory(const StagingBufferMap&, std::span<const VideoCommon::BufferImageCopy>) {
    // Readbacks need an asynchronous map; GPU-to-guest flushes are disabled for now.
}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::ImageViewInfo& info,
                     ImageId image_id_, Image& image)
    : VideoCommon::ImageViewBase{info, image.info, image_id_, image.gpu_addr} {
    if (!image.Texture()) {
        return;
    }
    texture = image.Texture();
    const FormatInfo& format_info = GetFormatInfo(format);
    host_format = format_info.format;
    if (True(image.flags & VideoCommon::ImageFlagBits::Converted) &&
        VideoCore::Surface::IsPixelFormatASTC(format)) {
        host_format = VideoCore::Surface::IsPixelFormatSRGB(format)
                          ? wgpu::TextureFormat::RGBA8UnormSrgb
                          : wgpu::TextureFormat::RGBA8Unorm;
    }
    if (!IsViewCompatible(image.Format(), host_format)) {
        // WebGPU views cannot reinterpret formats; view the image as what it is.
        host_format = image.Format();
    }
    kind = KindOf(format);
    if (kind == TextureKind::Float && IsUnfilterable(host_format)) {
        kind = TextureKind::UnfilterableFloat;
    }
    has_stencil = format_info.stencil;
    view_width = std::max(1u, image.info.size.width >> info.range.base.level);
    view_height = std::max(1u, image.info.size.height >> info.range.base.level);

    std::array<SwizzleSource, 4> swizzle{SwizzleSource::R, SwizzleSource::G, SwizzleSource::B,
                                         SwizzleSource::A};
    if (!info.IsRenderTarget()) {
        swizzle = info.Swizzle();
    }
    const bool identity = swizzle[0] == SwizzleSource::R && swizzle[1] == SwizzleSource::G &&
                          swizzle[2] == SwizzleSource::B && swizzle[3] == SwizzleSource::A;
    wgpu::TextureComponentSwizzleDescriptor swizzle_desc{};
    swizzle_desc.swizzle.r = ToComponent(swizzle[0]);
    swizzle_desc.swizzle.g = ToComponent(swizzle[1]);
    swizzle_desc.swizzle.b = ToComponent(swizzle[2]);
    swizzle_desc.swizzle.a = ToComponent(swizzle[3]);
    const bool can_swizzle =
        !identity && Thread::Get().device.HasFeature(wgpu::FeatureName::TextureComponentSwizzle) &&
        !format_info.depth && !format_info.stencil;

    const u32 base_layer = static_cast<u32>(info.range.base.layer);
    const u32 num_layers = static_cast<u32>(std::max(info.range.extent.layers, 1));
    const u32 base_level = static_cast<u32>(info.range.base.level);
    const u32 num_levels = static_cast<u32>(std::max(info.range.extent.levels, 1));
    const bool image_is_3d = image.Texture().GetDimension() == wgpu::TextureDimension::e3D;

    const auto make = [&](wgpu::TextureViewDimension dimension, u32 layers, u32 levels,
                          bool sampled) -> wgpu::TextureView {
        wgpu::TextureViewDescriptor desc{};
        desc.format = host_format;
        desc.dimension = dimension;
        desc.baseMipLevel = base_level;
        desc.mipLevelCount = levels;
        desc.baseArrayLayer = image_is_3d ? 0 : base_layer;
        desc.arrayLayerCount = image_is_3d ? 1 : layers;
        if (format_info.depth && sampled) {
            desc.aspect = wgpu::TextureAspect::DepthOnly;
        } else if (format_info.stencil && !format_info.depth && sampled) {
            desc.aspect = wgpu::TextureAspect::StencilOnly;
        }
        if (desc.aspect != wgpu::TextureAspect::All) {
            // A single-aspect view of a combined format must use that aspect's format
            // (Depth24Plus for Depth24PlusStencil8); Undefined resolves to it.
            desc.format = wgpu::TextureFormat::Undefined;
        }
        if (sampled && can_swizzle) {
            desc.nextInChain = &swizzle_desc;
        }
        return image.Texture().CreateView(&desc);
    };

    const auto set = [&](Shader::TextureType type, u32 layers) {
        const auto dimension = ViewDimension(type);
        // View dimensions must match the texture's: 3D textures only get 3D views and vice versa.
        if (image_is_3d != (dimension == wgpu::TextureViewDimension::e3D)) {
            return;
        }
        if (dimension == wgpu::TextureViewDimension::Cube && layers < 6) {
            return;
        }
        if (dimension == wgpu::TextureViewDimension::CubeArray && (layers < 6 || layers % 6 != 0)) {
            return;
        }
        views[static_cast<size_t>(type)] = make(dimension, layers, num_levels, true);
    };
    switch (info.type) {
    case ImageViewType::e1D:
    case ImageViewType::e1DArray:
        set(Shader::TextureType::Color1D, 1);
        set(Shader::TextureType::ColorArray1D, num_layers);
        break;
    case ImageViewType::e2D:
    case ImageViewType::e2DArray:
    case ImageViewType::Rect:
        set(Shader::TextureType::Color2D, 1);
        set(Shader::TextureType::Color2DRect, 1);
        set(Shader::TextureType::ColorArray2D, num_layers);
        break;
    case ImageViewType::e3D:
        set(Shader::TextureType::Color3D, 1);
        break;
    case ImageViewType::Cube:
    case ImageViewType::CubeArray:
        set(Shader::TextureType::ColorCube, 6);
        set(Shader::TextureType::ColorArrayCube, num_layers);
        set(Shader::TextureType::Color2D, 1);
        set(Shader::TextureType::ColorArray2D, num_layers);
        break;
    case ImageViewType::Buffer:
        break;
    }
    if (format_info.renderable && !image_is_3d) {
        render_target = make(wgpu::TextureViewDimension::e2D, 1, 1, false);
    }
}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::ImageViewInfo& info,
                     ImageId image_id_, Image& image, const SlotVector<Image>&)
    : ImageView{runtime, info, image_id_, image} {}

ImageView::ImageView(TextureCacheRuntime&, const VideoCommon::ImageInfo& info,
                     const VideoCommon::ImageViewInfo& view_info, GPUVAddr gpu_addr_)
    : VideoCommon::ImageViewBase{info, view_info, gpu_addr_},
      buffer_size{VideoCommon::CalculateGuestSizeInBytes(info)} {}

ImageView::ImageView(TextureCacheRuntime&, const VideoCommon::NullImageViewParams& params)
    : VideoCommon::ImageViewBase{params} {}

ImageView::~ImageView() = default;

Sampler::Sampler(TextureCacheRuntime&, const Tegra::Texture::TSCEntry& tsc) {
    using Tegra::Texture::TextureFilter;
    using Tegra::Texture::TextureMipmapFilter;
    desc.addressModeU = AddressMode(tsc.wrap_u);
    desc.addressModeV = AddressMode(tsc.wrap_v);
    desc.addressModeW = AddressMode(tsc.wrap_p);
    desc.magFilter = tsc.mag_filter == TextureFilter::Linear ? wgpu::FilterMode::Linear
                                                             : wgpu::FilterMode::Nearest;
    desc.minFilter = tsc.min_filter == TextureFilter::Linear ? wgpu::FilterMode::Linear
                                                             : wgpu::FilterMode::Nearest;
    desc.mipmapFilter = tsc.mipmap_filter == TextureMipmapFilter::Linear
                            ? wgpu::MipmapFilterMode::Linear
                            : wgpu::MipmapFilterMode::Nearest;
    desc.lodMinClamp = tsc.MinLod();
    desc.lodMaxClamp = std::max(tsc.MaxLod(), desc.lodMinClamp);
    if (tsc.mipmap_filter == TextureMipmapFilter::None) {
        desc.lodMaxClamp = std::min(desc.lodMaxClamp, 0.25f);
    }
    const f32 anisotropy = tsc.MaxAnisotropy();
    if (anisotropy > 1.0f && desc.magFilter == wgpu::FilterMode::Linear &&
        desc.minFilter == wgpu::FilterMode::Linear &&
        desc.mipmapFilter == wgpu::MipmapFilterMode::Linear) {
        desc.maxAnisotropy = static_cast<u16>(std::clamp(anisotropy, 1.0f, 16.0f));
    }
    if (tsc.depth_compare_enabled) {
        compare_function = CompareFunction(tsc.depth_compare_func);
    }
}

const wgpu::Sampler& Sampler::Handle(TextureKind kind, bool compare) {
    if (compare) {
        if (!comparison) {
            wgpu::SamplerDescriptor compare_desc = desc;
            compare_desc.maxAnisotropy = 1;
            compare_desc.compare = compare_function == wgpu::CompareFunction::Undefined
                                       ? wgpu::CompareFunction::LessEqual
                                       : compare_function;
            comparison = Device().CreateSampler(&compare_desc);
        }
        return comparison;
    }
    if (kind == TextureKind::Float) {
        if (!filtering) {
            filtering = Device().CreateSampler(&desc);
        }
        return filtering;
    }
    if (!non_filtering) {
        wgpu::SamplerDescriptor nearest = desc;
        nearest.magFilter = wgpu::FilterMode::Nearest;
        nearest.minFilter = wgpu::FilterMode::Nearest;
        nearest.mipmapFilter = wgpu::MipmapFilterMode::Nearest;
        nearest.maxAnisotropy = 1;
        non_filtering = Device().CreateSampler(&nearest);
    }
    return non_filtering;
}

Framebuffer::Framebuffer(TextureCacheRuntime&, std::span<ImageView*, NUM_RT> color_buffers,
                         ImageView* depth_buffer, const VideoCommon::RenderTargets& key) {
    // WebGPU needs every attachment of a pass to be the same size, where the guest renders to the
    // common area. Prefer the size matching the render area, then the depth buffer's; attachments
    // of another size are left out.
    u32 width = 0;
    u32 height = 0;
    std::array<const ImageView*, NUM_RT + 1> candidates{};
    candidates[0] = depth_buffer && depth_buffer->RenderTarget() ? depth_buffer : nullptr;
    for (size_t index = 0; index < NUM_RT; ++index) {
        const ImageView* const view = color_buffers[index];
        candidates[index + 1] = view && view->RenderTarget() ? view : nullptr;
    }
    for (const ImageView* view : candidates) {
        if (view && view->Width() == key.size.width && view->Height() == key.size.height) {
            width = view->Width();
            height = view->Height();
            break;
        }
    }
    const auto fits = [&](const ImageView* view) {
        if (width == 0) {
            width = view->Width();
            height = view->Height();
        }
        return view->Width() == width && view->Height() == height;
    };
    if (width == 0 && candidates[0]) {
        fits(candidates[0]);
    }
    for (size_t index = 0; index < NUM_RT; ++index) {
        const ImageView* const view = color_buffers[index];
        if (!view || !view->RenderTarget() || !fits(view)) {
            continue;
        }
        targets.colors[index] = view->RenderTarget();
        targets.color_formats[index] = view->HostFormat();
        targets.textures[index] = view->BaseTexture().Get();
    }
    if (depth_buffer && depth_buffer->RenderTarget() && fits(depth_buffer)) {
        targets.textures[8] = depth_buffer->BaseTexture().Get();
        targets.depth_stencil = depth_buffer->RenderTarget();
        targets.depth_format = depth_buffer->HostFormat();
        targets.has_stencil = depth_buffer->HasStencil();
    }
    targets.width = width ? std::min(key.size.width, width) : key.size.width;
    targets.height = height ? std::min(key.size.height, height) : key.size.height;
    static u32 logs = 0; // TEMP-DEBUG
    if (logs++ < 40 && color_buffers[0]) {
        LOG_INFO(Render, "TEMP-DEBUG framebuffer rt0 {:#x} fmt {} {}x{} depth {}", color_buffers[0]->gpu_addr,
                 color_buffers[0]->format, key.size.width, key.size.height, depth_buffer != nullptr);
    }
}

Framebuffer::Framebuffer(TextureCacheRuntime&, ImageView* color_buffer, ImageView* depth_buffer,
                         u32 width, u32 height) {
    if (color_buffer && color_buffer->RenderTarget()) {
        targets.colors[0] = color_buffer->RenderTarget();
        targets.color_formats[0] = color_buffer->HostFormat();
    }
    if (depth_buffer && depth_buffer->RenderTarget()) {
        targets.depth_stencil = depth_buffer->RenderTarget();
        targets.depth_format = depth_buffer->HostFormat();
        targets.has_stencil = depth_buffer->HasStencil();
    }
    targets.width = width;
    targets.height = height;
}

} // namespace WebGPU
