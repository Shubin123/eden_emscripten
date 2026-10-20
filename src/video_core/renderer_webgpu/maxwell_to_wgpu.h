// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <webgpu/webgpu_cpp.h>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"

namespace WebGPU::MaxwellToWGPU {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

/// Marks vertex attributes WebGPU cannot read.
constexpr wgpu::VertexFormat INVALID_VERTEX_FORMAT = static_cast<wgpu::VertexFormat>(0);

/// Topology after index conversion (quads, fans, polygons and loops become lists).
inline wgpu::PrimitiveTopology Topology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return wgpu::PrimitiveTopology::PointList;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LinesAdjacency:
        return wgpu::PrimitiveTopology::LineList;
    case Maxwell::PrimitiveTopology::LineStrip:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        return wgpu::PrimitiveTopology::LineStrip;
    case Maxwell::PrimitiveTopology::TriangleStrip:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        return wgpu::PrimitiveTopology::TriangleStrip;
    default:
        return wgpu::PrimitiveTopology::TriangleList;
    }
}

inline bool NeedsIndexConversion(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Polygon:
    case Maxwell::PrimitiveTopology::LineLoop:
        return true;
    default:
        return false;
    }
}

inline wgpu::CompareFunction Compare(Maxwell::ComparisonOp op) {
    switch (op) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return wgpu::CompareFunction::Never;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return wgpu::CompareFunction::Less;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return wgpu::CompareFunction::Equal;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return wgpu::CompareFunction::LessEqual;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return wgpu::CompareFunction::Greater;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return wgpu::CompareFunction::NotEqual;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return wgpu::CompareFunction::GreaterEqual;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return wgpu::CompareFunction::Always;
    }
    return wgpu::CompareFunction::Always;
}

inline wgpu::StencilOperation StencilOp(Maxwell::StencilOp::Op op) {
    switch (op) {
    case Maxwell::StencilOp::Op::Keep_D3D:
    case Maxwell::StencilOp::Op::Keep_GL:
        return wgpu::StencilOperation::Keep;
    case Maxwell::StencilOp::Op::Zero_D3D:
    case Maxwell::StencilOp::Op::Zero_GL:
        return wgpu::StencilOperation::Zero;
    case Maxwell::StencilOp::Op::Replace_D3D:
    case Maxwell::StencilOp::Op::Replace_GL:
        return wgpu::StencilOperation::Replace;
    case Maxwell::StencilOp::Op::IncrSaturate_D3D:
    case Maxwell::StencilOp::Op::IncrSaturate_GL:
        return wgpu::StencilOperation::IncrementClamp;
    case Maxwell::StencilOp::Op::DecrSaturate_D3D:
    case Maxwell::StencilOp::Op::DecrSaturate_GL:
        return wgpu::StencilOperation::DecrementClamp;
    case Maxwell::StencilOp::Op::Invert_D3D:
    case Maxwell::StencilOp::Op::Invert_GL:
        return wgpu::StencilOperation::Invert;
    case Maxwell::StencilOp::Op::Incr_D3D:
    case Maxwell::StencilOp::Op::Incr_GL:
        return wgpu::StencilOperation::IncrementWrap;
    case Maxwell::StencilOp::Op::Decr_D3D:
    case Maxwell::StencilOp::Op::Decr_GL:
        return wgpu::StencilOperation::DecrementWrap;
    }
    return wgpu::StencilOperation::Keep;
}

inline wgpu::BlendOperation BlendEquation(Maxwell::Blend::Equation equation) {
    switch (equation) {
    case Maxwell::Blend::Equation::Add_D3D:
    case Maxwell::Blend::Equation::Add_GL:
        return wgpu::BlendOperation::Add;
    case Maxwell::Blend::Equation::Subtract_D3D:
    case Maxwell::Blend::Equation::Subtract_GL:
        return wgpu::BlendOperation::Subtract;
    case Maxwell::Blend::Equation::ReverseSubtract_D3D:
    case Maxwell::Blend::Equation::ReverseSubtract_GL:
        return wgpu::BlendOperation::ReverseSubtract;
    case Maxwell::Blend::Equation::Min_D3D:
    case Maxwell::Blend::Equation::Min_GL:
        return wgpu::BlendOperation::Min;
    case Maxwell::Blend::Equation::Max_D3D:
    case Maxwell::Blend::Equation::Max_GL:
        return wgpu::BlendOperation::Max;
    }
    return wgpu::BlendOperation::Add;
}

inline wgpu::BlendFactor BlendFactor(Maxwell::Blend::Factor factor, bool dual_source) {
    using F = Maxwell::Blend::Factor;
    switch (factor) {
    case F::Zero_D3D:
    case F::Zero_GL:
        return wgpu::BlendFactor::Zero;
    case F::One_D3D:
    case F::One_GL:
        return wgpu::BlendFactor::One;
    case F::SourceColor_D3D:
    case F::SourceColor_GL:
        return wgpu::BlendFactor::Src;
    case F::OneMinusSourceColor_D3D:
    case F::OneMinusSourceColor_GL:
        return wgpu::BlendFactor::OneMinusSrc;
    case F::SourceAlpha_D3D:
    case F::SourceAlpha_GL:
        return wgpu::BlendFactor::SrcAlpha;
    case F::OneMinusSourceAlpha_D3D:
    case F::OneMinusSourceAlpha_GL:
        return wgpu::BlendFactor::OneMinusSrcAlpha;
    case F::DestAlpha_D3D:
    case F::DestAlpha_GL:
        return wgpu::BlendFactor::DstAlpha;
    case F::OneMinusDestAlpha_D3D:
    case F::OneMinusDestAlpha_GL:
        return wgpu::BlendFactor::OneMinusDstAlpha;
    case F::DestColor_D3D:
    case F::DestColor_GL:
        return wgpu::BlendFactor::Dst;
    case F::OneMinusDestColor_D3D:
    case F::OneMinusDestColor_GL:
        return wgpu::BlendFactor::OneMinusDst;
    case F::SourceAlphaSaturate_D3D:
    case F::SourceAlphaSaturate_GL:
        return wgpu::BlendFactor::SrcAlphaSaturated;
    case F::BlendFactor_D3D:
    case F::ConstantColor_GL:
    case F::ConstantAlpha_GL:
        return wgpu::BlendFactor::Constant;
    case F::OneMinusBlendFactor_D3D:
    case F::OneMinusConstantColor_GL:
    case F::OneMinusConstantAlpha_GL:
        return wgpu::BlendFactor::OneMinusConstant;
    case F::Source1Color_D3D:
    case F::Source1Color_GL:
        return dual_source ? wgpu::BlendFactor::Src1 : wgpu::BlendFactor::Src;
    case F::OneMinusSource1Color_D3D:
    case F::OneMinusSource1Color_GL:
        return dual_source ? wgpu::BlendFactor::OneMinusSrc1 : wgpu::BlendFactor::OneMinusSrc;
    case F::Source1Alpha_D3D:
    case F::Source1Alpha_GL:
        return dual_source ? wgpu::BlendFactor::Src1Alpha : wgpu::BlendFactor::SrcAlpha;
    case F::OneMinusSource1Alpha_D3D:
    case F::OneMinusSource1Alpha_GL:
        return dual_source ? wgpu::BlendFactor::OneMinusSrc1Alpha
                           : wgpu::BlendFactor::OneMinusSrcAlpha;
    case F::BothSourceAlpha_D3D:
        return wgpu::BlendFactor::SrcAlpha;
    case F::OneMinusBothSourceAlpha_D3D:
        return wgpu::BlendFactor::OneMinusSrcAlpha;
    }
    return wgpu::BlendFactor::One;
}

/// Vertex format for an attribute and the components it provides to the shader. Formats WebGPU
/// lacks are widened (three components read as four); returns INVALID_VERTEX_FORMAT when unsupported.
inline wgpu::VertexFormat VertexFormat(Maxwell::VertexAttribute::Type type,
                                       Maxwell::VertexAttribute::Size size, u8& components) {
    using T = Maxwell::VertexAttribute::Type;
    using S = Maxwell::VertexAttribute::Size;
    using VF = wgpu::VertexFormat;
    components = 4;
    const bool norm = type == T::UNorm || type == T::SNorm;
    const bool is_signed = type == T::SNorm || type == T::SInt || type == T::SScaled;
    const auto pick8 = [&](VF unorm, VF snorm, VF uint, VF sint) {
        if (norm) {
            return is_signed ? snorm : unorm;
        }
        return is_signed ? sint : uint;
    };
    switch (size) {
    case S::Size_R32_G32_B32_A32:
        return type == T::Float ? VF::Float32x4 : (is_signed ? VF::Sint32x4 : VF::Uint32x4);
    case S::Size_R32_G32_B32:
        components = 3;
        return type == T::Float ? VF::Float32x3 : (is_signed ? VF::Sint32x3 : VF::Uint32x3);
    case S::Size_R32_G32:
        components = 2;
        return type == T::Float ? VF::Float32x2 : (is_signed ? VF::Sint32x2 : VF::Uint32x2);
    case S::Size_R32:
        components = 1;
        return type == T::Float ? VF::Float32 : (is_signed ? VF::Sint32 : VF::Uint32);
    case S::Size_R16_G16_B16_A16:
        if (type == T::Float) {
            return VF::Float16x4;
        }
        return pick8(VF::Unorm16x4, VF::Snorm16x4, VF::Uint16x4, VF::Sint16x4);
    case S::Size_R16_G16_B16:
        components = 3;
        if (type == T::Float) {
            return VF::Float16x4;
        }
        return pick8(VF::Unorm16x4, VF::Snorm16x4, VF::Uint16x4, VF::Sint16x4);
    case S::Size_R16_G16:
        components = 2;
        if (type == T::Float) {
            return VF::Float16x2;
        }
        return pick8(VF::Unorm16x2, VF::Snorm16x2, VF::Uint16x2, VF::Sint16x2);
    case S::Size_R16:
        components = 1;
        if (type == T::Float) {
            return VF::Float16;
        }
        return pick8(VF::Unorm16, VF::Snorm16, VF::Uint16, VF::Sint16);
    case S::Size_R8_G8_B8_A8:
    case S::Size_X8_B8_G8_R8:
        return pick8(VF::Unorm8x4, VF::Snorm8x4, VF::Uint8x4, VF::Sint8x4);
    case S::Size_R8_G8_B8:
        components = 3;
        return pick8(VF::Unorm8x4, VF::Snorm8x4, VF::Uint8x4, VF::Sint8x4);
    case S::Size_R8_G8:
    case S::Size_G8_R8:
        components = 2;
        return pick8(VF::Unorm8x2, VF::Snorm8x2, VF::Uint8x2, VF::Sint8x2);
    case S::Size_R8:
    case S::Size_A8:
        components = 1;
        return pick8(VF::Unorm8, VF::Snorm8, VF::Uint8, VF::Sint8);
    case S::Size_A2_B10_G10_R10:
        if (type == T::SNorm) {
            // No signed 10:10:10:2 vertex format; the shader unpacks the raw word.
            return VF::Uint32;
        }
        return type == T::UNorm ? VF::Unorm10_10_10_2 : INVALID_VERTEX_FORMAT;
    default:
        return INVALID_VERTEX_FORMAT;
    }
}

/// Bytes a vertex format reads.
inline u32 VertexFormatSize(wgpu::VertexFormat format) {
    using VF = wgpu::VertexFormat;
    switch (format) {
    case VF::Uint8:
    case VF::Sint8:
    case VF::Unorm8:
    case VF::Snorm8:
        return 1;
    case VF::Uint8x2:
    case VF::Sint8x2:
    case VF::Unorm8x2:
    case VF::Snorm8x2:
    case VF::Uint16:
    case VF::Sint16:
    case VF::Unorm16:
    case VF::Snorm16:
    case VF::Float16:
        return 2;
    case VF::Uint8x4:
    case VF::Sint8x4:
    case VF::Unorm8x4:
    case VF::Snorm8x4:
    case VF::Uint16x2:
    case VF::Sint16x2:
    case VF::Unorm16x2:
    case VF::Snorm16x2:
    case VF::Float16x2:
    case VF::Float32:
    case VF::Uint32:
    case VF::Sint32:
    case VF::Unorm10_10_10_2:
        return 4;
    case VF::Uint16x4:
    case VF::Sint16x4:
    case VF::Unorm16x4:
    case VF::Snorm16x4:
    case VF::Float16x4:
    case VF::Float32x2:
    case VF::Uint32x2:
    case VF::Sint32x2:
        return 8;
    case VF::Float32x3:
    case VF::Uint32x3:
    case VF::Sint32x3:
        return 12;
    case VF::Float32x4:
    case VF::Uint32x4:
    case VF::Sint32x4:
        return 16;
    default:
        return 0;
    }
}

} // namespace WebGPU::MaxwellToWGPU
