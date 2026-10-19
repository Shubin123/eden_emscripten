// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>

#include "common/assert.h"
#include "video_core/renderer_webgpu/wgpu_formats.h"

namespace WebGPU {

using VideoCore::Surface::PixelFormat;
using TF = wgpu::TextureFormat;

namespace {

struct Entry {
    PixelFormat pixel_format;
    TF format;
    Conversion conversion;
    bool renderable;
};

constexpr std::array ENTRIES{
    Entry{PixelFormat::A8B8G8R8_UNORM, TF::RGBA8Unorm, Conversion::None, true},
    Entry{PixelFormat::A8B8G8R8_SNORM, TF::RGBA8Snorm, Conversion::None, false},
    Entry{PixelFormat::A8B8G8R8_SINT, TF::RGBA8Sint, Conversion::None, true},
    Entry{PixelFormat::A8B8G8R8_UINT, TF::RGBA8Uint, Conversion::None, true},
    Entry{PixelFormat::R5G6B5_UNORM, TF::RGBA8Unorm, Conversion::R5G6B5, true},
    Entry{PixelFormat::B5G6R5_UNORM, TF::RGBA8Unorm, Conversion::B5G6R5, true},
    Entry{PixelFormat::A1R5G5B5_UNORM, TF::RGBA8Unorm, Conversion::A1R5G5B5, true},
    Entry{PixelFormat::A2B10G10R10_UNORM, TF::RGB10A2Unorm, Conversion::None, true},
    Entry{PixelFormat::A2B10G10R10_UINT, TF::RGB10A2Uint, Conversion::None, true},
    Entry{PixelFormat::A2R10G10B10_UNORM, TF::RGB10A2Unorm, Conversion::A2R10G10B10, true},
    Entry{PixelFormat::A1B5G5R5_UNORM, TF::RGBA8Unorm, Conversion::A1B5G5R5, true},
    Entry{PixelFormat::A5B5G5R1_UNORM, TF::RGBA8Unorm, Conversion::A5B5G5R1, true},
    Entry{PixelFormat::R8_UNORM, TF::R8Unorm, Conversion::None, true},
    Entry{PixelFormat::R8_SNORM, TF::R8Snorm, Conversion::None, false},
    Entry{PixelFormat::R8_SINT, TF::R8Sint, Conversion::None, true},
    Entry{PixelFormat::R8_UINT, TF::R8Uint, Conversion::None, true},
    Entry{PixelFormat::R16G16B16A16_FLOAT, TF::RGBA16Float, Conversion::None, true},
    Entry{PixelFormat::R16G16B16A16_UNORM, TF::RGBA16Unorm, Conversion::None, true},
    Entry{PixelFormat::R16G16B16A16_SNORM, TF::RGBA16Snorm, Conversion::None, false},
    Entry{PixelFormat::R16G16B16A16_SINT, TF::RGBA16Sint, Conversion::None, true},
    Entry{PixelFormat::R16G16B16A16_UINT, TF::RGBA16Uint, Conversion::None, true},
    Entry{PixelFormat::B10G11R11_FLOAT, TF::RG11B10Ufloat, Conversion::None, true},
    Entry{PixelFormat::R32G32B32A32_UINT, TF::RGBA32Uint, Conversion::None, true},
    Entry{PixelFormat::BC1_RGBA_UNORM, TF::BC1RGBAUnorm, Conversion::None, false},
    Entry{PixelFormat::BC2_UNORM, TF::BC2RGBAUnorm, Conversion::None, false},
    Entry{PixelFormat::BC3_UNORM, TF::BC3RGBAUnorm, Conversion::None, false},
    Entry{PixelFormat::BC4_UNORM, TF::BC4RUnorm, Conversion::None, false},
    Entry{PixelFormat::BC4_SNORM, TF::BC4RSnorm, Conversion::None, false},
    Entry{PixelFormat::BC5_UNORM, TF::BC5RGUnorm, Conversion::None, false},
    Entry{PixelFormat::BC5_SNORM, TF::BC5RGSnorm, Conversion::None, false},
    Entry{PixelFormat::BC7_UNORM, TF::BC7RGBAUnorm, Conversion::None, false},
    Entry{PixelFormat::BC6H_UFLOAT, TF::BC6HRGBUfloat, Conversion::None, false},
    Entry{PixelFormat::BC6H_SFLOAT, TF::BC6HRGBFloat, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_4X4_UNORM, TF::ASTC4x4Unorm, Conversion::None, false},
    Entry{PixelFormat::B8G8R8A8_UNORM, TF::BGRA8Unorm, Conversion::None, true},
    Entry{PixelFormat::R32G32B32A32_FLOAT, TF::RGBA32Float, Conversion::None, true},
    Entry{PixelFormat::R32G32B32A32_SINT, TF::RGBA32Sint, Conversion::None, true},
    Entry{PixelFormat::R32G32_FLOAT, TF::RG32Float, Conversion::None, true},
    Entry{PixelFormat::R32G32_SINT, TF::RG32Sint, Conversion::None, true},
    Entry{PixelFormat::R32_FLOAT, TF::R32Float, Conversion::None, true},
    Entry{PixelFormat::R16_FLOAT, TF::R16Float, Conversion::None, true},
    Entry{PixelFormat::R16_UNORM, TF::R16Unorm, Conversion::None, true},
    Entry{PixelFormat::R16_SNORM, TF::R16Snorm, Conversion::None, false},
    Entry{PixelFormat::R16_UINT, TF::R16Uint, Conversion::None, true},
    Entry{PixelFormat::R16_SINT, TF::R16Sint, Conversion::None, true},
    Entry{PixelFormat::R16G16_UNORM, TF::RG16Unorm, Conversion::None, true},
    Entry{PixelFormat::R16G16_FLOAT, TF::RG16Float, Conversion::None, true},
    Entry{PixelFormat::R16G16_UINT, TF::RG16Uint, Conversion::None, true},
    Entry{PixelFormat::R16G16_SINT, TF::RG16Sint, Conversion::None, true},
    Entry{PixelFormat::R16G16_SNORM, TF::RG16Snorm, Conversion::None, false},
    Entry{PixelFormat::R32G32B32_FLOAT, TF::RGBA32Float, Conversion::RGB32F, false},
    Entry{PixelFormat::A8B8G8R8_SRGB, TF::RGBA8UnormSrgb, Conversion::None, true},
    Entry{PixelFormat::R8G8_UNORM, TF::RG8Unorm, Conversion::None, true},
    Entry{PixelFormat::R8G8_SNORM, TF::RG8Snorm, Conversion::None, false},
    Entry{PixelFormat::R8G8_SINT, TF::RG8Sint, Conversion::None, true},
    Entry{PixelFormat::R8G8_UINT, TF::RG8Uint, Conversion::None, true},
    Entry{PixelFormat::R32G32_UINT, TF::RG32Uint, Conversion::None, true},
    Entry{PixelFormat::R16G16B16X16_FLOAT, TF::RGBA16Float, Conversion::None, true},
    Entry{PixelFormat::R32_UINT, TF::R32Uint, Conversion::None, true},
    Entry{PixelFormat::R32_SINT, TF::R32Sint, Conversion::None, true},
    Entry{PixelFormat::ASTC_2D_8X8_UNORM, TF::ASTC8x8Unorm, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_8X5_UNORM, TF::ASTC8x5Unorm, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_5X4_UNORM, TF::ASTC5x4Unorm, Conversion::None, false},
    Entry{PixelFormat::B8G8R8A8_SRGB, TF::BGRA8UnormSrgb, Conversion::None, true},
    Entry{PixelFormat::BC1_RGBA_SRGB, TF::BC1RGBAUnormSrgb, Conversion::None, false},
    Entry{PixelFormat::BC2_SRGB, TF::BC2RGBAUnormSrgb, Conversion::None, false},
    Entry{PixelFormat::BC3_SRGB, TF::BC3RGBAUnormSrgb, Conversion::None, false},
    Entry{PixelFormat::BC7_SRGB, TF::BC7RGBAUnormSrgb, Conversion::None, false},
    Entry{PixelFormat::A4B4G4R4_UNORM, TF::RGBA8Unorm, Conversion::A4B4G4R4, true},
    Entry{PixelFormat::G4R4_UNORM, TF::RGBA8Unorm, Conversion::G4R4, true},
    Entry{PixelFormat::ASTC_2D_4X4_SRGB, TF::ASTC4x4UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_8X8_SRGB, TF::ASTC8x8UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_8X5_SRGB, TF::ASTC8x5UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_5X4_SRGB, TF::ASTC5x4UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_5X5_UNORM, TF::ASTC5x5Unorm, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_5X5_SRGB, TF::ASTC5x5UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_10X8_UNORM, TF::ASTC10x8Unorm, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_10X8_SRGB, TF::ASTC10x8UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_6X6_UNORM, TF::ASTC6x6Unorm, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_6X6_SRGB, TF::ASTC6x6UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_10X6_UNORM, TF::ASTC10x6Unorm, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_10X6_SRGB, TF::ASTC10x6UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_10X5_UNORM, TF::ASTC10x5Unorm, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_10X5_SRGB, TF::ASTC10x5UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_10X10_UNORM, TF::ASTC10x10Unorm, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_10X10_SRGB, TF::ASTC10x10UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_12X10_UNORM, TF::ASTC12x10Unorm, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_12X10_SRGB, TF::ASTC12x10UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_12X12_UNORM, TF::ASTC12x12Unorm, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_12X12_SRGB, TF::ASTC12x12UnormSrgb, Conversion::None, false},
    // No 8x6 and 6x5 ASTC in WebGPU; these would need decoding.
    Entry{PixelFormat::ASTC_2D_8X6_UNORM, TF::Undefined, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_8X6_SRGB, TF::Undefined, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_6X5_UNORM, TF::Undefined, Conversion::None, false},
    Entry{PixelFormat::ASTC_2D_6X5_SRGB, TF::Undefined, Conversion::None, false},
    Entry{PixelFormat::E5B9G9R9_FLOAT, TF::RGB9E5Ufloat, Conversion::None, false},
    Entry{PixelFormat::ETC2_RGB_UNORM, TF::ETC2RGB8Unorm, Conversion::None, false},
    Entry{PixelFormat::ETC2_RGBA_UNORM, TF::ETC2RGBA8Unorm, Conversion::None, false},
    Entry{PixelFormat::ETC2_RGB_PTA_UNORM, TF::ETC2RGB8A1Unorm, Conversion::None, false},
    Entry{PixelFormat::ETC2_RGB_SRGB, TF::ETC2RGB8UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::ETC2_RGBA_SRGB, TF::ETC2RGBA8UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::ETC2_RGB_PTA_SRGB, TF::ETC2RGB8A1UnormSrgb, Conversion::None, false},
    Entry{PixelFormat::EAC_R11_UNORM, TF::EACR11Unorm, Conversion::None, false},
    Entry{PixelFormat::EAC_R11_SNORM, TF::EACR11Snorm, Conversion::None, false},
    Entry{PixelFormat::EAC_R11G11_UNORM, TF::EACRG11Unorm, Conversion::None, false},
    Entry{PixelFormat::EAC_R11G11_SNORM, TF::EACRG11Snorm, Conversion::None, false},
    Entry{PixelFormat::D32_FLOAT, TF::Depth32Float, Conversion::None, true},
    Entry{PixelFormat::D16_UNORM, TF::Depth16Unorm, Conversion::None, true},
    Entry{PixelFormat::X8_D24_UNORM, TF::Depth24Plus, Conversion::None, true},
    Entry{PixelFormat::S8_UINT, TF::Stencil8, Conversion::None, true},
    Entry{PixelFormat::D24_UNORM_S8_UINT, TF::Depth24PlusStencil8, Conversion::None, true},
    Entry{PixelFormat::S8_UINT_D24_UNORM, TF::Depth24PlusStencil8, Conversion::None, true},
    Entry{PixelFormat::D32_FLOAT_S8_UINT, TF::Depth32FloatStencil8, Conversion::None, true},
};

std::array<FormatInfo, VideoCore::Surface::MaxPixelFormat> BuildTable() {
    std::array<FormatInfo, VideoCore::Surface::MaxPixelFormat> table{};
    for (const Entry& entry : ENTRIES) {
        FormatInfo& info = table[static_cast<size_t>(entry.pixel_format)];
        info.format = entry.format;
        info.conversion = entry.conversion;
        info.renderable = entry.renderable;
        const auto type = VideoCore::Surface::GetFormatType(entry.pixel_format);
        info.depth = type == VideoCore::Surface::SurfaceType::Depth ||
                     type == VideoCore::Surface::SurfaceType::DepthStencil;
        info.stencil = type == VideoCore::Surface::SurfaceType::Stencil ||
                       type == VideoCore::Surface::SurfaceType::DepthStencil;
    }
    return table;
}

u8 Expand(u32 value, u32 bits) {
    return static_cast<u8>(value * 255 / ((1u << bits) - 1));
}

u32 Field(u32 value, u32 shift, u32 bits) {
    return (value >> shift) & ((1u << bits) - 1);
}

} // Anonymous namespace

const FormatInfo& GetFormatInfo(PixelFormat format) {
    static const auto table = BuildTable();
    return table[static_cast<size_t>(format)];
}

u32 HostBytesPerBlock(PixelFormat format) {
    switch (GetFormatInfo(format).conversion) {
    case Conversion::None:
    case Conversion::A2R10G10B10:
        return VideoCore::Surface::BytesPerBlock(format);
    case Conversion::RGB32F:
        return 16;
    default:
        return 4;
    }
}

void ConvertTexels(Conversion conversion, std::span<const u8> input, std::vector<u8>& output,
                   size_t texels) {
    if (conversion == Conversion::RGB32F) {
        output.resize(texels * 16);
        constexpr f32 one = 1.0f;
        for (size_t i = 0; i < texels; ++i) {
            std::memcpy(output.data() + i * 16, input.data() + i * 12, 12);
            std::memcpy(output.data() + i * 16 + 12, &one, 4);
        }
        return;
    }
    if (conversion == Conversion::A2R10G10B10) {
        output.resize(texels * 4);
        for (size_t i = 0; i < texels; ++i) {
            u32 v;
            std::memcpy(&v, input.data() + i * 4, 4);
            const u32 out = (v & 0xC00FFC00u) | ((v >> 20) & 0x3FF) | ((v & 0x3FF) << 20);
            std::memcpy(output.data() + i * 4, &out, 4);
        }
        return;
    }
    output.resize(texels * 4);
    for (size_t i = 0; i < texels; ++i) {
        u32 v;
        if (conversion == Conversion::G4R4) {
            v = input[i];
        } else {
            v = static_cast<u32>(input[i * 2] | (input[i * 2 + 1] << 8));
        }
        u8 r = 0, g = 0, b = 0, a = 255;
        switch (conversion) {
        case Conversion::R5G6B5:
            r = Expand(Field(v, 11, 5), 5), g = Expand(Field(v, 5, 6), 6), b = Expand(Field(v, 0, 5), 5);
            break;
        case Conversion::B5G6R5:
            b = Expand(Field(v, 11, 5), 5), g = Expand(Field(v, 5, 6), 6), r = Expand(Field(v, 0, 5), 5);
            break;
        case Conversion::A1R5G5B5:
            a = Field(v, 15, 1) ? 255 : 0;
            r = Expand(Field(v, 10, 5), 5), g = Expand(Field(v, 5, 5), 5), b = Expand(Field(v, 0, 5), 5);
            break;
        case Conversion::A1B5G5R5:
            a = Field(v, 15, 1) ? 255 : 0;
            b = Expand(Field(v, 10, 5), 5), g = Expand(Field(v, 5, 5), 5), r = Expand(Field(v, 0, 5), 5);
            break;
        case Conversion::A5B5G5R1:
            a = Expand(Field(v, 11, 5), 5), b = Expand(Field(v, 6, 5), 5), g = Expand(Field(v, 1, 5), 5);
            r = Field(v, 0, 1) ? 255 : 0;
            break;
        case Conversion::A4B4G4R4:
            a = Expand(Field(v, 12, 4), 4), b = Expand(Field(v, 8, 4), 4), g = Expand(Field(v, 4, 4), 4);
            r = Expand(Field(v, 0, 4), 4);
            break;
        case Conversion::G4R4:
            g = Expand(Field(v, 4, 4), 4), r = Expand(Field(v, 0, 4), 4), b = 0;
            break;
        default:
            break;
        }
        u8* const out = output.data() + i * 4;
        out[0] = r, out[1] = g, out[2] = b, out[3] = a;
    }
}

wgpu::TextureFormat SrgbTwin(wgpu::TextureFormat format) {
    static constexpr std::array<std::pair<TF, TF>, 24> pairs{{
        {TF::RGBA8Unorm, TF::RGBA8UnormSrgb}, {TF::BGRA8Unorm, TF::BGRA8UnormSrgb},
        {TF::BC1RGBAUnorm, TF::BC1RGBAUnormSrgb}, {TF::BC2RGBAUnorm, TF::BC2RGBAUnormSrgb},
        {TF::BC3RGBAUnorm, TF::BC3RGBAUnormSrgb}, {TF::BC7RGBAUnorm, TF::BC7RGBAUnormSrgb},
        {TF::ASTC4x4Unorm, TF::ASTC4x4UnormSrgb}, {TF::ASTC5x4Unorm, TF::ASTC5x4UnormSrgb},
        {TF::ASTC5x5Unorm, TF::ASTC5x5UnormSrgb}, {TF::ASTC6x5Unorm, TF::ASTC6x5UnormSrgb},
        {TF::ASTC6x6Unorm, TF::ASTC6x6UnormSrgb}, {TF::ASTC8x5Unorm, TF::ASTC8x5UnormSrgb},
        {TF::ASTC8x6Unorm, TF::ASTC8x6UnormSrgb}, {TF::ASTC8x8Unorm, TF::ASTC8x8UnormSrgb},
        {TF::ASTC10x5Unorm, TF::ASTC10x5UnormSrgb}, {TF::ASTC10x6Unorm, TF::ASTC10x6UnormSrgb},
        {TF::ASTC10x8Unorm, TF::ASTC10x8UnormSrgb}, {TF::ASTC10x10Unorm, TF::ASTC10x10UnormSrgb},
        {TF::ASTC12x10Unorm, TF::ASTC12x10UnormSrgb}, {TF::ASTC12x12Unorm, TF::ASTC12x12UnormSrgb},
        {TF::ETC2RGB8Unorm, TF::ETC2RGB8UnormSrgb}, {TF::ETC2RGB8A1Unorm, TF::ETC2RGB8A1UnormSrgb},
        {TF::ETC2RGBA8Unorm, TF::ETC2RGBA8UnormSrgb}, {TF::Undefined, TF::Undefined},
    }};
    for (const auto& [linear, srgb] : pairs) {
        if (format == linear) {
            return srgb;
        }
        if (format == srgb) {
            return linear;
        }
    }
    return TF::Undefined;
}

bool IsViewCompatible(wgpu::TextureFormat base, wgpu::TextureFormat view) {
    return base == view || (view != TF::Undefined && SrgbTwin(base) == view);
}

} // namespace WebGPU
