// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <mutex>
#include <bit>
#include <chrono>
#include <cstring>

#include <boost/container/small_vector.hpp>

#include "common/alignment.h"
#include "common/cityhash.h"
#include "common/div_ceil.h"
#include "common/logging.h"
#include "common/settings.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/program_header.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_webgpu/maxwell_to_wgpu.h"
#include "video_core/renderer_webgpu/wgpu_pipeline_cache.h"
#include "video_core/renderer_webgpu/wgpu_stats.h"
#include "video_core/renderer_webgpu/wgpu_thread.h"
#include "video_core/shader_environment.h"
#include "video_core/surface.h"
#include "video_core/textures/texture.h"

namespace WebGPU {

namespace {
using Shader::Backend::WGSL::DrawInfo;
using Shader::Backend::WGSL::EmitWGSL;
using Shader::Backend::WGSL::Options;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;

constexpr size_t NUM_STAGES = 5;

Shader::CompareFunction MaxwellToCompareFunction(Maxwell::ComparisonOp comparison) {
    switch (comparison) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return Shader::CompareFunction::Never;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return Shader::CompareFunction::Less;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return Shader::CompareFunction::Always;
    }
    return Shader::CompareFunction::Always;
}

Shader::AttributeType CastAttributeType(const Vulkan::FixedPipelineState::VertexAttribute& attr) {
    if (attr.enabled == 0) {
        return Shader::AttributeType::Disabled;
    }
    switch (attr.Type()) {
    case Maxwell::VertexAttribute::Type::SNorm:
    case Maxwell::VertexAttribute::Type::UNorm:
    case Maxwell::VertexAttribute::Type::Float:
        return Shader::AttributeType::Float;
    case Maxwell::VertexAttribute::Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Maxwell::VertexAttribute::Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Maxwell::VertexAttribute::Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Maxwell::VertexAttribute::Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    default:
        return Shader::AttributeType::Disabled;
    }
}

Shader::RuntimeInfo MakeRuntimeInfo(const GraphicsPipelineKey& key, const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous_program) {
    Shader::RuntimeInfo info;
    if (previous_program) {
        info.previous_stage_stores = previous_program->info.stores;
        info.previous_stage_legacy_stores_mapping = previous_program->info.legacy_stores_mapping;
    } else {
        info.previous_stage_stores.mask.set();
    }
    const bool gl_ndc{key.state.ndc_minus_one_to_one != 0};
    switch (program.stage) {
    case Shader::Stage::VertexB:
        info.convert_depth_mode = gl_ndc;
        std::ranges::transform(key.state.attributes, info.generic_input_types.begin(),
                               &CastAttributeType);
        break;
    case Shader::Stage::Fragment:
        info.alpha_test_func = MaxwellToCompareFunction(
            key.state.UnpackComparisonOp(key.state.alpha_test_func.Value()));
        info.alpha_test_reference = std::bit_cast<float>(key.state.alpha_test_ref);
        info.dual_source_blend = key.state.attachment0_dual_source_blend != 0;
        for (size_t i = 0; i < 8; ++i) {
            const auto format = static_cast<Tegra::RenderTargetFormat>(key.state.color_formats[i]);
            const auto pixel_format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
            if (format != Tegra::RenderTargetFormat::NONE &&
                VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
                info.color_output_types[i] = VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)
                                                 ? Shader::AttributeType::SignedInt
                                                 : Shader::AttributeType::UnsignedInt;
            } else {
                info.color_output_types[i] = Shader::AttributeType::Float;
            }
        }
        break;
    default:
        break;
    }
    switch (key.state.topology) {
    case Maxwell::PrimitiveTopology::Points:
        info.input_topology = Shader::InputTopology::Points;
        break;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        info.input_topology = Shader::InputTopology::Lines;
        break;
    default:
        info.input_topology = Shader::InputTopology::Triangles;
        break;
    }
    info.force_early_z = key.state.early_z != 0;
    info.y_negate = key.state.y_negate != 0;
    return info;
}

wgpu::TextureViewDimension ViewDimension(Shader::TextureType type) {
    switch (type) {
    case Shader::TextureType::ColorArray1D:
    case Shader::TextureType::ColorArray2D:
        return wgpu::TextureViewDimension::e2DArray;
    case Shader::TextureType::Color3D:
        return wgpu::TextureViewDimension::e3D;
    case Shader::TextureType::ColorCube:
        return wgpu::TextureViewDimension::Cube;
    case Shader::TextureType::ColorArrayCube:
        return wgpu::TextureViewDimension::CubeArray;
    default:
        return wgpu::TextureViewDimension::e2D;
    }
}

TextureSampleKind ToSampleKind(TextureKind kind) {
    switch (kind) {
    case TextureKind::Float:
        return TextureSampleKind::Float;
    case TextureKind::UnfilterableFloat:
        return TextureSampleKind::UnfilterableFloat;
    case TextureKind::Depth:
        return TextureSampleKind::Depth;
    case TextureKind::Sint:
        return TextureSampleKind::Sint;
    case TextureKind::Uint:
        return TextureSampleKind::Uint;
    }
    return TextureSampleKind::Float;
}

TextureKind FromSampleKind(TextureSampleKind kind) {
    switch (kind) {
    case TextureSampleKind::Float:
        return TextureKind::Float;
    case TextureSampleKind::UnfilterableFloat:
        return TextureKind::UnfilterableFloat;
    case TextureSampleKind::Depth:
        return TextureKind::Depth;
    case TextureSampleKind::Sint:
        return TextureKind::Sint;
    case TextureSampleKind::Uint:
        return TextureKind::Uint;
    }
    return TextureKind::Float;
}

wgpu::ShaderStage StageVisibility(size_t stage) {
    return stage == 4 ? wgpu::ShaderStage::Fragment : wgpu::ShaderStage::Vertex;
}

/// Components the pipeline feeds to an attribute, or zero when WebGPU cannot read it.
u8 ProvidedComponents(const Vulkan::FixedPipelineState& state, size_t index) {
    const auto& attribute = state.attributes[index];
    if (!attribute.enabled) {
        return 0;
    }
    u8 components{};
    const wgpu::VertexFormat format =
        MaxwellToWGPU::VertexFormat(attribute.Type(), attribute.Size(), components);
    if (format == MaxwellToWGPU::INVALID_VERTEX_FORMAT) {
        return 0;
    }
    const u32 stride = state.vertex_strides[attribute.buffer] & ~3u;
    if (stride != 0 && attribute.offset + MaxwellToWGPU::VertexFormatSize(format) > stride) {
        return 0;
    }
    if (attribute.offset % 4 != 0 && MaxwellToWGPU::VertexFormatSize(format) >= 4) {
        return 0;
    }
    return components;
}

/// How the shader reads `format` from a pulled vertex buffer.
Shader::Backend::WGSL::PulledAttribute PulledFormat(wgpu::VertexFormat format) {
    using VF = wgpu::VertexFormat;
    using Kind = Shader::Backend::WGSL::PulledAttribute::Kind;
    const auto make = [](u8 bits, u8 components, Kind kind) {
        Shader::Backend::WGSL::PulledAttribute attr{};
        attr.enabled = true;
        attr.bits = bits;
        attr.components = components;
        attr.kind = kind;
        return attr;
    };
    switch (format) {
    case VF::Unorm8: return make(8, 1, Kind::Unorm);
    case VF::Unorm8x2: return make(8, 2, Kind::Unorm);
    case VF::Unorm8x4: return make(8, 4, Kind::Unorm);
    case VF::Snorm8: return make(8, 1, Kind::Snorm);
    case VF::Snorm8x2: return make(8, 2, Kind::Snorm);
    case VF::Snorm8x4: return make(8, 4, Kind::Snorm);
    case VF::Uint8: return make(8, 1, Kind::Uint);
    case VF::Uint8x2: return make(8, 2, Kind::Uint);
    case VF::Uint8x4: return make(8, 4, Kind::Uint);
    case VF::Sint8: return make(8, 1, Kind::Sint);
    case VF::Sint8x2: return make(8, 2, Kind::Sint);
    case VF::Sint8x4: return make(8, 4, Kind::Sint);
    case VF::Unorm16: return make(16, 1, Kind::Unorm);
    case VF::Unorm16x2: return make(16, 2, Kind::Unorm);
    case VF::Unorm16x4: return make(16, 4, Kind::Unorm);
    case VF::Snorm16: return make(16, 1, Kind::Snorm);
    case VF::Snorm16x2: return make(16, 2, Kind::Snorm);
    case VF::Snorm16x4: return make(16, 4, Kind::Snorm);
    case VF::Uint16: return make(16, 1, Kind::Uint);
    case VF::Uint16x2: return make(16, 2, Kind::Uint);
    case VF::Uint16x4: return make(16, 4, Kind::Uint);
    case VF::Sint16: return make(16, 1, Kind::Sint);
    case VF::Sint16x2: return make(16, 2, Kind::Sint);
    case VF::Sint16x4: return make(16, 4, Kind::Sint);
    case VF::Float16: return make(16, 1, Kind::Float);
    case VF::Float16x2: return make(16, 2, Kind::Float);
    case VF::Float16x4: return make(16, 4, Kind::Float);
    case VF::Float32: return make(32, 1, Kind::Float);
    case VF::Float32x2: return make(32, 2, Kind::Float);
    case VF::Float32x3: return make(32, 3, Kind::Float);
    case VF::Float32x4: return make(32, 4, Kind::Float);
    case VF::Uint32: return make(32, 1, Kind::Uint);
    case VF::Uint32x2: return make(32, 2, Kind::Uint);
    case VF::Uint32x3: return make(32, 3, Kind::Uint);
    case VF::Uint32x4: return make(32, 4, Kind::Uint);
    case VF::Sint32: return make(32, 1, Kind::Sint);
    case VF::Sint32x2: return make(32, 2, Kind::Sint);
    case VF::Sint32x3: return make(32, 3, Kind::Sint);
    case VF::Sint32x4: return make(32, 4, Kind::Sint);
    case VF::Unorm10_10_10_2: return make(32, 4, Kind::Unorm1010102);
    default: return {};
    }
}

u32 UniformBindingSize(const Shader::Info& info, u32 index) {
    const u32 used_size{std::max(info.constant_buffer_used_sizes[index], 16U)};
    return std::min(Common::DivCeil(used_size, 16U), 4096U) * 16;
}

wgpu::ShaderModule MakeModule(const std::string& code, u64 hash) {
    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = code.c_str();
    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &wgsl;
    wgpu::ShaderModule module = Thread::Get().device.CreateShaderModule(&desc);
    static u32 dumped_modules = 0; // TEMP-DEBUG
    if (dumped_modules++ < 6 || Settings::values.dump_guest_shaders.GetValue()) {
        LOG_INFO(Render, "WGSL {:016x}:\n{}", hash, code);
    }
    module.GetCompilationInfo(
        wgpu::CallbackMode::AllowSpontaneous,
        [hash, code](wgpu::CompilationInfoRequestStatus, const wgpu::CompilationInfo* info) {
            if (!info) {
                return;
            }
            for (size_t i = 0; i < info->messageCount; ++i) {
                const auto& message = info->messages[i];
                if (message.type != wgpu::CompilationMessageType::Error) {
                    continue;
                }
                LOG_ERROR(Render, "WGSL {:016x} line {}: {}", hash, message.lineNum,
                          std::string_view{message.message.data, message.message.length});
                static u32 dumped = 0;
                if (dumped++ < 4) {
                    LOG_ERROR(Render, "WGSL source:\n{}", code);
                }
            }
        });
    return module;
}

} // Anonymous namespace

void DebugCount(const std::string& what) {
    static std::mutex mutex;
    static std::map<std::string, u64> counts;
    static auto last = std::chrono::steady_clock::now();
    std::scoped_lock lock{mutex};
    ++counts[what];
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(10)) {
        return;
    }
    last = now;
    std::string text;
    for (const auto& [key, value] : counts) {
        text += fmt::format("\n  {:8} {}", value, key);
    }
    LOG_INFO(Render, "TEMP-DEBUG counts (10 s):{}", text);
    counts.clear();
}

size_t GraphicsPipelineKey::Hash() const noexcept {
    return static_cast<size_t>(Common::CityHash64(reinterpret_cast<const char*>(this), Size()));
}

bool GraphicsPipelineKey::operator==(const GraphicsPipelineKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, Size()) == 0;
}

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                             Scheduler& scheduler_, UniformRing& uniform_ring_,
                             BufferCache& buffer_cache_, TextureCache& texture_cache_)
    : VideoCommon::ShaderCache{device_memory_}, scheduler{scheduler_}, uniform_ring{uniform_ring_},
      buffer_cache{buffer_cache_}, texture_cache{texture_cache_} {
    profile = Shader::Profile{
        .supported_spirv = 0x00010000,
        .unified_descriptor_binding = true,
        .support_vertex_instance_id = false,
        .support_float_controls = false,
        .support_derivative_control = true,
        .support_native_ndc = false,
        .support_scaled_attributes = true,
        .warp_size_potentially_larger_than_guest = false,
        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .min_ssbo_alignment = 256,
        .max_user_clip_distances = 0,
    };
    host_info = Shader::HostTranslateInfo{
        .min_ssbo_alignment = 256,
        .support_float64 = false,
        .support_float16 = false,
        .support_int64 = false,
        .needs_demote_reorder = false,
        .support_snorm_render_buffer = false,
        .support_viewport_index_layer = false,
        .support_viewport_mask = false,
        .support_geometry_shader_passthrough = false,
        // TODO: the WGSL backend can keep barriers after conditional returns uniform, but TotK's
        // tiled lighting renders darker with them until its tile depth input is correct.
        .support_conditional_barrier = false,
        .single_lane_geometry_subgroups = true,
    };
    host_info.ApplyDescriptorLimitPolicy();
    wgpu::Limits limits{};
    if (Thread::Get().device.GetLimits(&limits) == wgpu::Status::Success) {
        max_storage_buffers = limits.maxStorageBuffersPerShaderStage;
        max_vertex_buffers = limits.maxVertexBuffers;
        max_samplers = limits.maxSamplersPerShaderStage;
    }
}

PipelineCache::~PipelineCache() = default;

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {
    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        return nullptr;
    }
    graphics_key.state.Refresh(*maxwell3d, dynamic_features);
    auto [it, is_new] = graphics_cache.try_emplace(graphics_key);
    pipeline_stats.lookups.fetch_add(1, std::memory_order_relaxed);
    if (is_new) {
        pipeline_stats.misses.fetch_add(1, std::memory_order_relaxed);
        GraphicsEnvironments environments;
        GetGraphicsEnvironments(environments, graphics_key.unique_hashes);
        it->second = std::make_unique<GraphicsPipeline>(*this, graphics_key, std::move(environments));
    }
    current_pipeline = it->second.get();
    return current_pipeline->IsValid() ? current_pipeline : nullptr;
}

const wgpu::TextureView& PipelineCache::DummyTexture(Shader::TextureType type, TextureKind kind) {
    const auto key = std::make_pair(type, kind);
    if (const auto it = dummy_views.find(key); it != dummy_views.end()) {
        return it->second;
    }
    auto& device = Thread::Get().device;
    wgpu::TextureDescriptor desc{};
    const auto dimension = ViewDimension(type);
    const bool cube = dimension == wgpu::TextureViewDimension::Cube ||
                      dimension == wgpu::TextureViewDimension::CubeArray;
    desc.size = {1, 1, cube ? 6u : 1u};
    desc.dimension = dimension == wgpu::TextureViewDimension::e3D ? wgpu::TextureDimension::e3D
                                                                  : wgpu::TextureDimension::e2D;
    switch (kind) {
    case TextureKind::Depth:
        desc.format = wgpu::TextureFormat::Depth32Float;
        break;
    case TextureKind::Sint:
        desc.format = wgpu::TextureFormat::RGBA8Sint;
        break;
    case TextureKind::Uint:
        desc.format = wgpu::TextureFormat::RGBA8Uint;
        break;
    default:
        desc.format = wgpu::TextureFormat::RGBA8Unorm;
        break;
    }
    desc.usage = wgpu::TextureUsage::TextureBinding;
    wgpu::Texture texture = device.CreateTexture(&desc);
    wgpu::TextureViewDescriptor view_desc{};
    view_desc.dimension = dimension;
    view_desc.arrayLayerCount = cube ? 6 : 1;
    dummy_textures.push_back(texture);
    return dummy_views.emplace(key, texture.CreateView(&view_desc)).first->second;
}

wgpu::TextureView PipelineCache::FeedbackCopy(const ImageView& view, Shader::TextureType type) {
    const wgpu::Texture& source = view.BaseTexture();
    FeedbackCopyEntry& entry = feedback_copies[source.Get()];
    if (!entry.copy || entry.copy.GetWidth() != source.GetWidth() ||
        entry.copy.GetHeight() != source.GetHeight() ||
        entry.copy.GetFormat() != source.GetFormat() ||
        entry.copy.GetMipLevelCount() != source.GetMipLevelCount() ||
        entry.copy.GetDepthOrArrayLayers() != source.GetDepthOrArrayLayers()) {
        wgpu::TextureDescriptor desc{};
        desc.size = {source.GetWidth(), source.GetHeight(), source.GetDepthOrArrayLayers()};
        desc.dimension = source.GetDimension();
        desc.format = source.GetFormat();
        desc.mipLevelCount = source.GetMipLevelCount();
        desc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
        const wgpu::TextureFormat twin = SrgbTwin(desc.format);
        if (twin != wgpu::TextureFormat::Undefined) {
            desc.viewFormatCount = 1;
            desc.viewFormats = &twin;
        }
        entry.copy = Thread::Get().device.CreateTexture(&desc);
        entry.pass_serial = ~0ULL;
    }
    // Copy once per render pass; the copy ends the pass, so the serial advances afterwards.
    if (entry.pass_serial != scheduler.PassSerial()) {
        auto& encoder = scheduler.Encoder();
        for (u32 level = 0; level < source.GetMipLevelCount(); ++level) {
            wgpu::TexelCopyTextureInfo src{};
            src.texture = source;
            src.mipLevel = level;
            wgpu::TexelCopyTextureInfo dst{};
            dst.texture = entry.copy;
            dst.mipLevel = level;
            const wgpu::Extent3D extent{std::max(1u, source.GetWidth() >> level),
                                        std::max(1u, source.GetHeight() >> level),
                                        source.GetDimension() == wgpu::TextureDimension::e3D
                                            ? std::max(1u, source.GetDepthOrArrayLayers() >> level)
                                            : source.GetDepthOrArrayLayers()};
            encoder.CopyTextureToTexture(&src, &dst, &extent);
        }
        entry.pass_serial = scheduler.PassSerial() + 1;
    }
    wgpu::TextureViewDescriptor desc{};
    desc.format = view.HostFormat();
    desc.dimension = ViewDimension(type);
    const bool layered = desc.dimension == wgpu::TextureViewDimension::e2DArray ||
                         desc.dimension == wgpu::TextureViewDimension::CubeArray;
    desc.baseMipLevel = static_cast<u32>(view.range.base.level);
    desc.mipLevelCount = static_cast<u32>(std::max(view.range.extent.levels, 1));
    desc.baseArrayLayer = static_cast<u32>(view.range.base.layer);
    desc.arrayLayerCount = desc.dimension == wgpu::TextureViewDimension::Cube ? 6
                           : layered ? static_cast<u32>(std::max(view.range.extent.layers, 1))
                                     : 1;
    if (view.Kind() == TextureKind::Depth) {
        desc.aspect = wgpu::TextureAspect::DepthOnly;
        // Undefined resolves to the depth aspect's format, which combined formats require.
        desc.format = wgpu::TextureFormat::Undefined;
    }
    return entry.copy.CreateView(&desc);
}

GraphicsPipeline::GraphicsPipeline(PipelineCache& cache_, const GraphicsPipelineKey& key_,
                                   VideoCommon::ShaderCache::GraphicsEnvironments&& environments_)
    : cache{cache_}, key{key_}, environments{std::move(environments_)} {
    // The environment pointers refer into the moved array.
    size_t env_index{};
    environments.env_ptrs.fill(nullptr);
    for (size_t index = 0; index < key.unique_hashes.size(); ++index) {
        if (key.unique_hashes[index] != 0) {
            environments.env_ptrs[env_index++] = &environments.envs[index];
        }
    }
    // WebGPU has no tessellation or geometry stages.
    const bool unsupported = key.unique_hashes[2] != 0 || key.unique_hashes[3] != 0 ||
                             key.unique_hashes[4] != 0;
    if (unsupported) {
        LOG_WARNING(Render, "WebGPU: skipping pipeline {:016x} with tessellation/geometry stages",
                    key.Hash());
        return;
    }
    valid = true;
}

bool GraphicsPipeline::Translate(const VariantKey& variant_key, Variant& variant, bool infos_only) {
    auto& pools = cache.pools;
    pools.ReleaseContents();
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    const bool uses_vertex_a{key.unique_hashes[0] != 0};
    size_t env_index{};
    try {
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            Shader::Environment& env{*environments.env_ptrs[env_index++]};
            const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
            if (!uses_vertex_a || index != 1) {
                programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, cache.host_info);
            } else {
                auto program_vb{TranslateProgram(pools.inst, pools.block, env, cfg, cache.host_info)};
                programs[index] = MergeDualVertexPrograms(programs[0], program_vb, env);
            }
        }
        if (!infos_ready) {
            for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
                if (key.unique_hashes[index] == 0) {
                    continue;
                }
                const size_t stage = index - 1;
                stage_infos[stage] = programs[index].info;
                if (!programs[index].info.image_descriptors.empty() ||
                    !programs[index].info.texture_buffer_descriptors.empty() ||
                    programs[index].info.storage_buffers_descriptors.size() > 8) {
                    LOG_INFO(Render, "TEMP-DEBUG graphics {:016x} stage {}: ssbos {} images {} "
                             "texbufs {}", key.Hash(), stage,
                             programs[index].info.storage_buffers_descriptors.size(),
                             programs[index].info.image_descriptors.size(),
                             programs[index].info.texture_buffer_descriptors.size());
                }
                enabled_stages[stage] = true;
                enabled_uniform_buffer_masks[stage] = stage_infos[stage].constant_buffer_mask;
                std::ranges::copy(stage_infos[stage].constant_buffer_used_sizes,
                                  uniform_buffer_sizes[stage].begin());
                num_textures += Shader::NumDescriptors(stage_infos[stage].texture_descriptors);
            }
            infos_ready = true;
        }
        if (infos_only) {
            return true;
        }
        if (variant_key.texture_kinds.size() != num_textures) {
            return false;
        }

        // Guest vertex buffers the pipeline reads take WebGPU's vertex buffer slots in order; the
        // shader fetches the ones past the last slot from storage buffers.
        std::array<bool, Maxwell::NumVertexArrays> buffer_read{};
        for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
            const auto& attribute = key.state.attributes[index];
            if (attribute.enabled && stage_infos[0].loads.Generic(index) &&
                ProvidedComponents(key.state, index) != 0) {
                buffer_read[attribute.buffer] = true;
            }
        }
        pulled_buffers.clear();
        u32 num_read{};
        for (u32 buffer = 0; buffer < Maxwell::NumVertexArrays; ++buffer) {
            if (buffer_read[buffer] && num_read++ >= cache.max_vertex_buffers) {
                pulled_buffers.push_back(buffer);
            }
        }
        if (pulled_buffers.size() > Shader::Backend::WGSL::MAX_PULLED_BUFFERS) {
            LOG_WARNING(Render, "WebGPU: pipeline {:016x} reads too many vertex buffers", key.Hash());
            return false;
        }
        const auto pulled_index = [&](u32 buffer) -> std::optional<u8> {
            const auto it = std::ranges::find(pulled_buffers, buffer);
            if (it == pulled_buffers.end()) {
                return std::nullopt;
            }
            return static_cast<u8>(it - pulled_buffers.begin());
        };

        std::array<std::string, NUM_STAGES> codes;
        Shader::Backend::Bindings binding;
        const Shader::IR::Program* previous_stage{};
        size_t kind_index{};
        for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            Shader::IR::Program& program{programs[index]};
            const size_t stage = index - 1;
            const auto runtime_info{MakeRuntimeInfo(key, program, previous_stage)};
            Shader::Maxwell::ConvertLegacyToGeneric(program, runtime_info);

            Options options;
            const size_t stage_textures{Shader::NumDescriptors(program.info.texture_descriptors)};
            options.texture_kinds.assign(variant_key.texture_kinds.begin() + kind_index,
                                         variant_key.texture_kinds.begin() + kind_index + stage_textures);
            kind_index += stage_textures;
            options.has_depth_target = variant_key.depth_format != wgpu::TextureFormat::Undefined;
            options.max_samplers = cache.max_samplers;
            if (program.stage == Shader::Stage::VertexB && key.unique_hashes[5] != 0) {
                options.output_interpolation = programs[5].info.interpolation;
            }
            if (program.stage == Shader::Stage::VertexB) {
                for (size_t attr = 0; attr < Maxwell::NumVertexAttributes; ++attr) {
                    options.attribute_components[attr] = ProvidedComponents(key.state, attr);
                    const auto& attribute = key.state.attributes[attr];
                    options.attribute_snorm_1010102[attr] =
                        attribute.enabled &&
                        attribute.Size() == Maxwell::VertexAttribute::Size::Size_A2_B10_G10_R10 &&
                        attribute.Type() == Maxwell::VertexAttribute::Type::SNorm;
                    const auto pulled = pulled_index(attribute.buffer);
                    if (attribute.enabled && pulled && options.attribute_components[attr] != 0) {
                        u8 components{};
                        auto& pulled_attr = options.pulled_attributes[attr];
                        pulled_attr = PulledFormat(MaxwellToWGPU::VertexFormat(
                            attribute.Type(), attribute.Size(), components));
                        pulled_attr.buffer = *pulled;
                        pulled_attr.offset = attribute.offset;
                    }
                }
                for (const u32 buffer : pulled_buffers) {
                    options.pulled_buffers.push_back({key.state.vertex_strides[buffer],
                                                      key.state.binding_divisors[buffer] != 0});
                }
            }
            codes[stage] = EmitWGSL(cache.profile, runtime_info, program, binding, options);
            previous_stage = &program;
        }

        auto& device = Thread::Get().device;
        // Bind group layout, in the order descriptors are pushed by Configure.
        std::vector<wgpu::BindGroupLayoutEntry> entries;
        u32 binding_index{};
        kind_index = 0;
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            if (!enabled_stages[stage]) {
                continue;
            }
            const Shader::Info& info{stage_infos[stage]};
            const auto visibility = StageVisibility(stage);
            for (const auto& desc : info.constant_buffer_descriptors) {
                for (u32 i = 0; i < desc.count; ++i) {
                    auto& entry = entries.emplace_back();
                    entry.binding = binding_index++;
                    entry.visibility = visibility;
                    entry.buffer.type = wgpu::BufferBindingType::Uniform;
                    entry.buffer.minBindingSize = UniformBindingSize(info, desc.index);
                }
            }
            for (const auto& desc : info.storage_buffers_descriptors) {
                const bool writable =
                    codes[stage].find(fmt::format("@binding({}) var<storage, read_write>",
                                                  binding_index)) != std::string::npos;
                for (u32 i = 0; i < desc.count; ++i) {
                    auto& entry = entries.emplace_back();
                    entry.binding = binding_index++;
                    entry.visibility = visibility;
                    entry.buffer.type = writable ? wgpu::BufferBindingType::Storage
                                                 : wgpu::BufferBindingType::ReadOnlyStorage;
                    variant.storage_writable.push_back(writable);
                }
            }
            u32 ordinal{};
            const u32 shared_first{Shader::Backend::WGSL::SharedSamplerFirst(
                Shader::NumDescriptors(info.texture_descriptors), cache.max_samplers)};
            for (const auto& desc : info.texture_descriptors) {
                for (u32 i = 0; i < desc.count; ++i) {
                    const TextureSampleKind kind{variant_key.texture_kinds[kind_index++]};
                    auto& texture = entries.emplace_back();
                    texture.binding = binding_index++;
                    texture.visibility = visibility;
                    texture.texture.viewDimension = ViewDimension(desc.type);
                    if (desc.is_depth || kind == TextureSampleKind::Depth) {
                        texture.texture.sampleType = wgpu::TextureSampleType::Depth;
                    } else if (kind == TextureSampleKind::Sint) {
                        texture.texture.sampleType = wgpu::TextureSampleType::Sint;
                    } else if (kind == TextureSampleKind::Uint) {
                        texture.texture.sampleType = wgpu::TextureSampleType::Uint;
                    } else if (kind == TextureSampleKind::UnfilterableFloat) {
                        texture.texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;
                    } else {
                        texture.texture.sampleType = wgpu::TextureSampleType::Float;
                    }
                    const u32 element = ordinal++;
                    if (element > shared_first) {
                        ++binding_index;
                        continue;
                    }
                    auto& sampler = entries.emplace_back();
                    sampler.binding = binding_index++;
                    sampler.visibility = visibility;
                    if (desc.is_depth) {
                        sampler.sampler.type = wgpu::SamplerBindingType::Comparison;
                    } else if (kind == TextureSampleKind::Float && element != shared_first) {
                        sampler.sampler.type = wgpu::SamplerBindingType::Filtering;
                    } else {
                        sampler.sampler.type = wgpu::SamplerBindingType::NonFiltering;
                    }
                }
            }
        }
        wgpu::BindGroupLayoutDescriptor layout_desc{};
        layout_desc.entryCount = entries.size();
        layout_desc.entries = entries.data();
        variant.layout = device.CreateBindGroupLayout(&layout_desc);

        wgpu::BindGroupLayoutEntry draw_entry{};
        draw_entry.binding = 0;
        draw_entry.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
        draw_entry.buffer.type = wgpu::BufferBindingType::Uniform;
        draw_entry.buffer.hasDynamicOffset = true;
        draw_entry.buffer.minBindingSize = sizeof(DrawInfo);
        wgpu::BindGroupLayoutDescriptor draw_layout_desc{};
        draw_layout_desc.entryCount = 1;
        draw_layout_desc.entries = &draw_entry;
        const wgpu::BindGroupLayout draw_layout = device.CreateBindGroupLayout(&draw_layout_desc);
        if (!pulled_buffers.empty()) {
            std::vector<wgpu::BindGroupLayoutEntry> pulled_entries(pulled_buffers.size() + 1);
            for (u32 i = 0; i < pulled_buffers.size(); ++i) {
                pulled_entries[i].binding = i;
                pulled_entries[i].visibility = wgpu::ShaderStage::Vertex;
                pulled_entries[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
            }
            auto& bases = pulled_entries.back();
            bases.binding = Shader::Backend::WGSL::MAX_PULLED_BUFFERS;
            bases.visibility = wgpu::ShaderStage::Vertex;
            bases.buffer.type = wgpu::BufferBindingType::Uniform;
            bases.buffer.minBindingSize = Shader::Backend::WGSL::MAX_PULLED_BUFFERS * 4;
            wgpu::BindGroupLayoutDescriptor pulled_desc{};
            pulled_desc.entryCount = pulled_entries.size();
            pulled_desc.entries = pulled_entries.data();
            variant.pulled_layout = device.CreateBindGroupLayout(&pulled_desc);
        }
        const std::array<wgpu::BindGroupLayout, 3> layouts{variant.layout, draw_layout,
                                                           variant.pulled_layout};
        wgpu::PipelineLayoutDescriptor pipeline_layout_desc{};
        pipeline_layout_desc.bindGroupLayoutCount = variant.pulled_layout ? 3 : 2;
        pipeline_layout_desc.bindGroupLayouts = layouts.data();
        const wgpu::PipelineLayout pipeline_layout = device.CreatePipelineLayout(&pipeline_layout_desc);

        const wgpu::ShaderModule vs_module = MakeModule(codes[0], key.unique_hashes[1]);
        wgpu::ShaderModule fs_module;
        if (enabled_stages[4]) {
            fs_module = MakeModule(codes[4], key.unique_hashes[5]);
        }

        const auto& state = key.state;
        // Vertex input
        std::array<std::vector<wgpu::VertexAttribute>, Maxwell::NumVertexArrays> attributes;
        u32 num_buffers{};
        for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
            const auto& attribute = state.attributes[index];
            if (!attribute.enabled || !stage_infos[0].loads.Generic(index)) {
                continue;
            }
            if (ProvidedComponents(state, index) == 0 || pulled_index(attribute.buffer)) {
                continue;
            }
            u8 components{};
            const wgpu::VertexFormat format =
                MaxwellToWGPU::VertexFormat(attribute.Type(), attribute.Size(), components);
            const u32 buffer = attribute.buffer;
            attributes[buffer].push_back({.format = format,
                                          .offset = attribute.offset,
                                          .shaderLocation = static_cast<u32>(index)});
            num_buffers = std::max(num_buffers, buffer + 1);
        }
        // WebGPU has fewer vertex buffer slots than the guest has buffers, so the buffers the
        // pipeline reads take consecutive slots.
        std::array<wgpu::VertexBufferLayout, Maxwell::NumVertexArrays> buffer_layouts{};
        vertex_buffer_uses = {};
        u32 num_slots{};
        for (u32 index = 0; index < num_buffers; ++index) {
            if (attributes[index].empty()) {
                continue;
            }
            const u32 slot = num_slots++;
            auto& layout = buffer_layouts[slot];
            vertex_buffer_uses[index] = {state.vertex_strides[index] & ~3u,
                                         state.binding_divisors[index] != 0, true, slot};
            layout.arrayStride = state.vertex_strides[index] & ~3u;
            layout.stepMode = state.binding_divisors[index] != 0 ? wgpu::VertexStepMode::Instance
                                                                 : wgpu::VertexStepMode::Vertex;
            layout.attributeCount = attributes[index].size();
            layout.attributes = attributes[index].data();
        }

        wgpu::RenderPipelineDescriptor desc{};
        desc.layout = pipeline_layout;
        desc.vertex.module = vs_module;
        desc.vertex.entryPoint = "main";
        desc.vertex.bufferCount = num_slots;
        desc.vertex.buffers = buffer_layouts.data();

        const auto& dynamic = state.dynamic_state;
        desc.primitive.topology = MaxwellToWGPU::Topology(state.topology);
        if (desc.primitive.topology == wgpu::PrimitiveTopology::TriangleStrip ||
            desc.primitive.topology == wgpu::PrimitiveTopology::LineStrip) {
            desc.primitive.stripIndexFormat = variant_key.strip_index_format;
        }
        desc.primitive.frontFace = dynamic.FrontFace() == Maxwell::FrontFace::ClockWise
                                       ? wgpu::FrontFace::CW
                                       : wgpu::FrontFace::CCW;
        if (dynamic.cull_enable) {
            switch (dynamic.CullFace()) {
            case Maxwell::CullFace::Front:
                desc.primitive.cullMode = wgpu::CullMode::Front;
                break;
            case Maxwell::CullFace::Back:
                desc.primitive.cullMode = wgpu::CullMode::Back;
                break;
            case Maxwell::CullFace::FrontAndBack:
                // Nothing is drawn; WebGPU has no mode for it.
                desc.primitive.cullMode = wgpu::CullMode::None;
                break;
            }
        }
        if (dynamic.depth_clamp_disabled == 0 &&
            device.HasFeature(wgpu::FeatureName::DepthClipControl)) {
            desc.primitive.unclippedDepth = true;
        }

        wgpu::DepthStencilState depth_stencil{};
        if (variant_key.depth_format != wgpu::TextureFormat::Undefined) {
            depth_stencil.format = variant_key.depth_format;
            depth_stencil.depthWriteEnabled = dynamic.depth_test_enable && dynamic.depth_write_enable
                                                  ? wgpu::OptionalBool::True
                                                  : wgpu::OptionalBool::False;
            depth_stencil.depthCompare = dynamic.depth_test_enable
                                             ? MaxwellToWGPU::Compare(dynamic.DepthTestFunc())
                                             : wgpu::CompareFunction::Always;
            const bool has_stencil = variant_key.depth_format == wgpu::TextureFormat::Depth24PlusStencil8 ||
                                     variant_key.depth_format == wgpu::TextureFormat::Depth32FloatStencil8 ||
                                     variant_key.depth_format == wgpu::TextureFormat::Stencil8;
            if (has_stencil && dynamic.stencil_enable) {
                const auto face = [](const auto& f) {
                    wgpu::StencilFaceState s{};
                    s.compare = MaxwellToWGPU::Compare(f.TestFunc());
                    s.failOp = MaxwellToWGPU::StencilOp(f.ActionStencilFail());
                    s.depthFailOp = MaxwellToWGPU::StencilOp(f.ActionDepthFail());
                    s.passOp = MaxwellToWGPU::StencilOp(f.ActionDepthPass());
                    return s;
                };
                depth_stencil.stencilFront = face(dynamic.front);
                depth_stencil.stencilBack = face(dynamic.back);
                depth_stencil.stencilReadMask = variant_key.stencil_masks & 0xFF;
                depth_stencil.stencilWriteMask = (variant_key.stencil_masks >> 8) & 0xFF;
            }
            if (dynamic.depth_bias_enable) {
                depth_stencil.depthBias = static_cast<s32>(std::bit_cast<f32>(variant_key.depth_bias[0]));
                depth_stencil.depthBiasSlopeScale = std::bit_cast<f32>(variant_key.depth_bias[1]);
                depth_stencil.depthBiasClamp = std::bit_cast<f32>(variant_key.depth_bias[2]);
            }
            desc.depthStencil = &depth_stencil;
        }

        std::array<wgpu::ColorTargetState, 8> targets{};
        std::array<wgpu::BlendState, 8> blends{};
        size_t num_targets{};
        const bool dual_source = state.attachment0_dual_source_blend != 0;
        for (size_t index = 0; index < targets.size(); ++index) {
            if (variant_key.color_formats[index] == wgpu::TextureFormat::Undefined) {
                continue;
            }
            num_targets = index + 1;
            auto& target = targets[index];
            target.format = variant_key.color_formats[index];
            const auto& attachment = state.attachments[index];
            const auto mask = attachment.Mask();
            const bool written = stage_infos[4].stores_frag_color[index];
            target.writeMask = wgpu::ColorWriteMask::None;
            if (written) {
                target.writeMask = (mask[0] ? wgpu::ColorWriteMask::Red : wgpu::ColorWriteMask::None) |
                                   (mask[1] ? wgpu::ColorWriteMask::Green : wgpu::ColorWriteMask::None) |
                                   (mask[2] ? wgpu::ColorWriteMask::Blue : wgpu::ColorWriteMask::None) |
                                   (mask[3] ? wgpu::ColorWriteMask::Alpha : wgpu::ColorWriteMask::None);
            }
            const bool integer = stage_infos[4].stores_frag_color[index] &&
                                 (target.format == wgpu::TextureFormat::RGBA8Uint ||
                                  target.format == wgpu::TextureFormat::RGBA8Sint);
            if (attachment.enable && written && !integer) {
                auto& blend = blends[index];
                blend.color.operation = MaxwellToWGPU::BlendEquation(attachment.EquationRGB());
                blend.color.srcFactor = MaxwellToWGPU::BlendFactor(attachment.SourceRGBFactor(), dual_source);
                blend.color.dstFactor = MaxwellToWGPU::BlendFactor(attachment.DestRGBFactor(), dual_source);
                blend.alpha.operation = MaxwellToWGPU::BlendEquation(attachment.EquationAlpha());
                blend.alpha.srcFactor = MaxwellToWGPU::BlendFactor(attachment.SourceAlphaFactor(), dual_source);
                blend.alpha.dstFactor = MaxwellToWGPU::BlendFactor(attachment.DestAlphaFactor(), dual_source);
                // Min and max ignore the factors; WebGPU requires them to be one.
                for (auto* component : {&blend.color, &blend.alpha}) {
                    if (component->operation == wgpu::BlendOperation::Min ||
                        component->operation == wgpu::BlendOperation::Max) {
                        component->srcFactor = wgpu::BlendFactor::One;
                        component->dstFactor = wgpu::BlendFactor::One;
                    }
                }
                target.blend = &blend;
            }
        }
        wgpu::FragmentState fragment{};
        if (fs_module) {
            fragment.module = fs_module;
            fragment.entryPoint = "main";
            fragment.targetCount = num_targets;
            fragment.targets = targets.data();
            desc.fragment = &fragment;
        }
        // Asynchronous creation reports failures before the pipeline is ever used, so an invalid
        // pipeline cannot invalidate the command buffer it would be recorded into.
        device.CreateRenderPipelineAsync(
            &desc, wgpu::CallbackMode::AllowSpontaneous,
            [slot = variant.pipeline, hash = key.Hash(),
             requested = std::chrono::steady_clock::now()](wgpu::CreatePipelineAsyncStatus status,
                                                           wgpu::RenderPipeline pipeline,
                                                           wgpu::StringView message) {
                const auto elapsed = std::chrono::steady_clock::now() - requested;
                pipeline_stats.creation_us.fetch_add(
                    std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count(),
                    std::memory_order_relaxed);
                if (status == wgpu::CreatePipelineAsyncStatus::Success) {
                    pipeline_stats.creations.fetch_add(1, std::memory_order_relaxed);
                    *slot = std::move(pipeline);
                    return;
                }
                pipeline_stats.creation_failures.fetch_add(1, std::memory_order_relaxed);
                LOG_ERROR(Render, "WebGPU pipeline {:016x}: {}", hash,
                          std::string_view{message.data, message.length});
            });
        return true;
    } catch (const std::exception& exception) {
        LOG_ERROR(Render, "WebGPU shader translation failed: {}", exception.what());
        return false;
    }
}

GraphicsPipeline::Variant* GraphicsPipeline::FindVariant(const VariantKey& variant_key) {
    if (last_variant && variant_key == last_variant_key) {
        return last_variant;
    }
    auto [it, is_new] = variants.try_emplace(variant_key);
    if (is_new) {
        const auto start = std::chrono::steady_clock::now();
        if (!Translate(variant_key, it->second, false)) {
            it->second.failed = true;
            pipeline_stats.build_failures.fetch_add(1, std::memory_order_relaxed);
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        pipeline_stats.builds.fetch_add(1, std::memory_order_relaxed);
        pipeline_stats.build_us.fetch_add(
            std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count(),
            std::memory_order_relaxed);
    }
    last_variant = &it->second;
    last_variant_key = variant_key;
    return last_variant;
}

bool GraphicsPipeline::Configure(bool is_indexed) {
    auto& texture_cache = cache.texture_cache;
    auto& buffer_cache = cache.buffer_cache;
    if (!infos_ready) {
        Variant scratch;
        Translate(VariantKey{}, scratch, true);
        if (!infos_ready) {
            valid = false;
            { DebugCount("configure false line 863"); return false; }
        }
    }
    auto* const maxwell3d = cache.maxwell3d;
    auto* const gpu_memory = cache.gpu_memory;
    const auto& regs{maxwell3d->regs};

    boost::container::small_vector<VideoCommon::ImageViewInOut, 32> views;
    boost::container::small_vector<VideoCommon::SamplerId, 32> samplers;

    texture_cache.SynchronizeDescriptors(false);
    buffer_cache.SetUniformBuffersState(enabled_uniform_buffer_masks, &uniform_buffer_sizes);

    const bool via_header_index{regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding};
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!enabled_stages[stage]) {
            continue;
        }
        const Shader::Info& info{stage_infos[stage]};
        buffer_cache.UnbindGraphicsStorageBuffers(stage);
        size_t ssbo_index{};
        for (const auto& desc : info.storage_buffers_descriptors) {
            buffer_cache.BindGraphicsStorageBuffer(stage, ssbo_index, desc.cbuf_index,
                                                   desc.cbuf_offset, desc.is_written);
            ++ssbo_index;
        }
        const auto& cbufs{maxwell3d->state.shader_stages[stage].const_buffers};
        for (const auto& desc : info.texture_descriptors) {
            for (u32 index = 0; index < desc.count; ++index) {
                const u32 index_offset{index << desc.size_shift};
                const u32 offset{desc.cbuf_offset + index_offset};
                const GPUVAddr addr{cbufs[desc.cbuf_index].address + offset};
                u32 raw{gpu_memory->Read<u32>(addr) << desc.shift_left};
                if (desc.has_secondary) {
                    const u32 second_offset{desc.secondary_cbuf_offset + index_offset};
                    const GPUVAddr separate_addr{cbufs[desc.secondary_cbuf_index].address + second_offset};
                    raw |= gpu_memory->Read<u32>(separate_addr) << desc.secondary_shift_left;
                }
                const auto handle{Tegra::Texture::TexturePair(raw, via_header_index)};
                views.push_back({handle.first});
                samplers.push_back(texture_cache.GetSamplerId(handle.second, false));
            }
        }
    }
    texture_cache.FillImageViews(std::span(views.data(), views.size()), false, false);

    buffer_cache.UpdateGraphicsBuffers(is_indexed);
    buffer_cache.BindHostGeometryBuffers(is_indexed);

    auto& runtime = buffer_cache.runtime;
    runtime.ClearDescriptors();
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (enabled_stages[stage]) {
            buffer_cache.BindHostStageBuffers(stage);
        }
    }
    buffer_cache.any_buffer_uploaded = false;
    texture_cache.UpdateRenderTargets(false);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    if (!framebuffer) {
        { DebugCount("configure false line 923"); return false; }
    }
    targets = &framebuffer->Targets();

    VariantKey variant_key;
    variant_key.texture_kinds.reserve(views.size());
    {
        size_t texture_index{};
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            if (!enabled_stages[stage]) {
                continue;
            }
            for (const auto& desc : stage_infos[stage].texture_descriptors) {
                for (u32 i = 0; i < desc.count; ++i) {
                    const ImageView& view{texture_cache.GetImageView(views[texture_index++].id)};
                    TextureKind kind = view.Handle(desc.type) ? view.Kind() : TextureKind::Float;
                    if (desc.is_depth) {
                        kind = TextureKind::Depth;
                    }
                    variant_key.texture_kinds.push_back(ToSampleKind(kind));
                }
            }
        }
    }
    variant_key.color_formats = targets->color_formats;
    variant_key.depth_format = targets->depth_stencil ? targets->depth_format
                                                      : wgpu::TextureFormat::Undefined;
    for (size_t i = 0; i < 8; ++i) {
        if (!targets->colors[i]) {
            variant_key.color_formats[i] = wgpu::TextureFormat::Undefined;
        }
    }
    variant_key.stencil_masks = (regs.stencil_front_func_mask & 0xFF) |
                                ((regs.stencil_front_mask & 0xFF) << 8);
    if (key.state.dynamic_state.depth_bias_enable) {
        variant_key.depth_bias = {std::bit_cast<u32>(regs.depth_bias),
                                  std::bit_cast<u32>(regs.slope_scale_depth_bias),
                                  std::bit_cast<u32>(regs.depth_bias_clamp)};
    }

    {
        const auto topology = MaxwellToWGPU::Topology(key.state.topology);
        if (is_indexed && (topology == wgpu::PrimitiveTopology::TriangleStrip ||
                           topology == wgpu::PrimitiveTopology::LineStrip)) {
            variant_key.strip_index_format = buffer_cache.runtime.index.format;
        }
    }
    Variant* const variant = FindVariant(variant_key);
    if (!variant || variant->failed || !*variant->pipeline) {
        // Still compiling, or failed.
        { DebugCount("configure false line 973"); return false; }
    }
    render_pipeline = *variant->pipeline;
    pulled_layout = variant->pulled_layout;

    // Resource bindings, in layout order.
    auto& descriptors = runtime.Descriptors();
    std::vector<wgpu::BindGroupEntry> entries;
    written_buffers.clear();
    read_buffers.clear();
    written_entries.clear();
    size_t storage_index{};
    size_t descriptor_index{};
    size_t texture_index{};
    u32 binding_index{};
    auto& device = Thread::Get().device;
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!enabled_stages[stage]) {
            continue;
        }
        const Shader::Info& info{stage_infos[stage]};
        for (const auto& desc : info.constant_buffer_descriptors) {
            for (u32 i = 0; i < desc.count; ++i) {
                if (descriptor_index >= descriptors.size()) {
                    { DebugCount("configure false line 995"); return false; }
                }
                BufferDescriptor& bound = descriptors[descriptor_index++];
                const u32 size = UniformBindingSize(info, desc.index);
                static u32 cbuf_logs = 0;
                if (cbuf_logs++ < 30) {
                    size_t nonzero = 0;
                    for (u8 b : bound.data) if (b != 0) ++nonzero;
                    LOG_INFO(Render, "Configure cbuf: stage {} idx {} orig_size {} size {} nonzero {}",
                             stage, desc.index, bound.data.size(), size, nonzero);
                }
                bound.data.resize(size, 0);
                auto& entry = entries.emplace_back();
                entry.binding = binding_index++;
                entry.buffer = cache.uniform_ring.Buffer();
                entry.offset = cache.uniform_ring.Push(bound.data);
                entry.size = size;
            }
        }
        for (const auto& desc : info.storage_buffers_descriptors) {
            for (u32 i = 0; i < desc.count; ++i) {
                if (descriptor_index >= descriptors.size()) {
                    { DebugCount("configure false line 1017"); return false; }
                }
                const BufferDescriptor& bound = descriptors[descriptor_index++];
                auto& entry = entries.emplace_back();
                entry.binding = binding_index++;
                entry.buffer = bound.buffer;
                entry.offset = bound.offset;
                entry.size = std::min<u64>(bound.size, bound.buffer.GetSize() - bound.offset);
                const bool writable = storage_index < variant->storage_writable.size() &&
                                      variant->storage_writable[storage_index];
                ++storage_index;
                if (writable) {
                    written_entries.push_back(entries.size() - 1);
                }
                (writable ? written_buffers : read_buffers).push_back(bound.buffer.Get());
            }
        }
        u32 ordinal{};
        const u32 shared_first{Shader::Backend::WGSL::SharedSamplerFirst(
            Shader::NumDescriptors(info.texture_descriptors), cache.max_samplers)};
        for (const auto& desc : info.texture_descriptors) {
            for (u32 i = 0; i < desc.count; ++i) {
                const size_t slot = texture_index++;
                ImageView& view{texture_cache.GetImageView(views[slot].id)};
                const TextureKind kind{FromSampleKind(variant_key.texture_kinds[slot])};
                wgpu::TextureView handle = view.Handle(desc.type);
                const bool is_dummy = (!handle || view.Kind() != kind);
                if (is_dummy) {
                    DebugCount(fmt::format("dummy texture: view {} fmt {} viewtype {} desctype {} "
                                           "kind {}/{}", views[slot].id.index != 0 ? "real" : "null",
                                           view.format, static_cast<u32>(view.type),
                                           static_cast<u32>(desc.type),
                                           static_cast<u32>(view.Kind()), static_cast<u32>(kind)));
                    handle = cache.DummyTexture(desc.type, kind);
                } else if (std::ranges::find(targets->textures, view.BaseTexture().Get()) !=
                           targets->textures.end()) {
                    handle = cache.FeedbackCopy(view, desc.type);
                }
                static u32 tex_logs = 0;
                if (tex_logs++ < 30) {
                    LOG_INFO(Render, "Configure tex: stage {} slot {} dummy {} view_h={} view_kind={} kind={} size={}x{}",
                             stage, slot, is_dummy, view.Handle(desc.type) != nullptr,
                             static_cast<u32>(view.Kind()), static_cast<u32>(kind),
                             view.size.width, view.size.height);
                }
                auto& texture_entry = entries.emplace_back();
                texture_entry.binding = binding_index++;
                texture_entry.textureView = handle;
                const u32 element = ordinal++;
                if (element > shared_first) {
                    ++binding_index;
                    continue;
                }
                Sampler& sampler{texture_cache.GetSampler(samplers[slot])};
                auto& sampler_entry = entries.emplace_back();
                sampler_entry.binding = binding_index++;
                sampler_entry.sampler = sampler.Handle(
                    element == shared_first && kind == TextureKind::Float ? TextureKind::UnfilterableFloat
                                                                         : kind,
                    desc.is_depth);
            }
        }
    }
    wgpu::BindGroupDescriptor bind_desc{};
    bind_desc.layout = variant->layout;
    bind_desc.entryCount = entries.size();
    bind_desc.entries = entries.data();
    bind_group = device.CreateBindGroup(&bind_desc);
    bind_layout = variant->layout;
    bind_entries = std::move(entries);
    return true;
}

ComputePipeline* PipelineCache::CurrentComputePipeline() {
    const VideoCommon::ShaderInfo* const shader{ComputeShader()};
    if (!shader) {
        return nullptr;
    }
    const auto& qmd{kepler_compute->launch_description};
    const ComputePipelineKey key{
        .unique_hash = shader->unique_hash,
        .shared_memory_size = qmd.shared_alloc,
        .workgroup_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
    };
    auto [it, is_new] = compute_cache.try_emplace(key);
    if (is_new) {
        const GPUVAddr program_base{kepler_compute->regs.code_loc.Address()};
        VideoCommon::ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base,
                                            qmd.program_start};
        env.SetCachedSize(shader->size_bytes);
        it->second = std::make_unique<ComputePipeline>(*this, key, std::move(env));
    }
    return it->second.get();
}

ComputePipeline::ComputePipeline(PipelineCache& cache_, const ComputePipelineKey& key_,
                                 VideoCommon::ComputeEnvironment&& env_)
    : cache{cache_}, key{key_}, env{std::move(env_)} {
    valid = Translate({}, nullptr);
}

bool ComputePipeline::Translate(const std::vector<TextureSampleKind>& texture_kinds,
                                Variant* variant) {
    auto& pools = cache.pools;
    pools.ReleaseContents();
    try {
        Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};
        Shader::IR::Program program{
            TranslateProgram(pools.inst, pools.block, env, cfg, cache.host_info)};
        if (!variant) {
            info = program.info;
            LOG_INFO(Render, "TEMP-DEBUG compute {:016x}: cbufs {} ssbos {} textures {} images {} "
                     "texbufs {} imgbufs {} smem {} wg {}x{}x{}", key.unique_hash,
                     info.constant_buffer_descriptors.size(), info.storage_buffers_descriptors.size(),
                     info.texture_descriptors.size(), info.image_descriptors.size(),
                     info.texture_buffer_descriptors.size(), info.image_buffer_descriptors.size(),
                     program.shared_memory_size, program.workgroup_size[0],
                     program.workgroup_size[1], program.workgroup_size[2]);
            std::copy_n(info.constant_buffer_used_sizes.begin(), uniform_buffer_sizes.size(),
                        uniform_buffer_sizes.begin());
            return true;
        }
        Options options;
        options.texture_kinds = texture_kinds;
        options.max_storage_buffers = cache.max_storage_buffers;
        options.max_samplers = cache.max_samplers;
        Shader::Backend::Bindings binding;
        const std::string code{
            EmitWGSL(cache.profile, Shader::RuntimeInfo{}, program, binding, options)};

        auto& device = Thread::Get().device;
        const auto visibility = wgpu::ShaderStage::Compute;
        std::vector<wgpu::BindGroupLayoutEntry> entries;
        u32 binding_index{};
        for (const auto& desc : info.constant_buffer_descriptors) {
            for (u32 i = 0; i < desc.count; ++i) {
                auto& entry = entries.emplace_back();
                entry.binding = binding_index++;
                entry.visibility = visibility;
                entry.buffer.type = wgpu::BufferBindingType::Uniform;
                entry.buffer.minBindingSize = UniformBindingSize(info, desc.index);
            }
        }
        const u32 num_ssbos{Shader::NumDescriptors(info.storage_buffers_descriptors)};
        const u32 packed_first{
            Shader::Backend::WGSL::PackedSsboFirst(num_ssbos, cache.max_storage_buffers)};
        if (num_ssbos - packed_first > Shader::Backend::WGSL::MAX_PACKED_SSBOS) {
            LOG_WARNING(Render, "WebGPU: compute shader {:016x} uses too many storage buffers",
                        key.unique_hash);
            return false;
        }
        u32 ssbo_index{};
        for (const auto& desc : info.storage_buffers_descriptors) {
            const bool writable =
                code.find(fmt::format("@binding({}) var<storage, read_write>", binding_index)) !=
                std::string::npos;
            for (u32 i = 0; i < desc.count; ++i, ++ssbo_index) {
                const u32 binding_number = binding_index++;
                variant->storage_writable.push_back(writable);
                if (ssbo_index > packed_first) {
                    continue;
                }
                auto& entry = entries.emplace_back();
                entry.binding = binding_number;
                entry.visibility = visibility;
                entry.buffer.type = writable ? wgpu::BufferBindingType::Storage
                                             : wgpu::BufferBindingType::ReadOnlyStorage;
                if (ssbo_index == packed_first) {
                    auto& bases = entries.emplace_back();
                    bases.binding = binding_number + 1;
                    bases.visibility = visibility;
                    bases.buffer.type = wgpu::BufferBindingType::Uniform;
                    bases.buffer.minBindingSize = Shader::Backend::WGSL::MAX_PACKED_SSBOS * 4;
                }
            }
        }
        u32 ordinal{};
        const u32 shared_first{Shader::Backend::WGSL::SharedSamplerFirst(
            Shader::NumDescriptors(info.texture_descriptors), cache.max_samplers)};
        size_t kind_index{};
        for (const auto& desc : info.texture_descriptors) {
            for (u32 i = 0; i < desc.count; ++i) {
                const TextureSampleKind kind{texture_kinds[kind_index++]};
                auto& texture = entries.emplace_back();
                texture.binding = binding_index++;
                texture.visibility = visibility;
                texture.texture.viewDimension = ViewDimension(desc.type);
                if (desc.is_depth || kind == TextureSampleKind::Depth) {
                    texture.texture.sampleType = wgpu::TextureSampleType::Depth;
                } else if (kind == TextureSampleKind::Sint) {
                    texture.texture.sampleType = wgpu::TextureSampleType::Sint;
                } else if (kind == TextureSampleKind::Uint) {
                    texture.texture.sampleType = wgpu::TextureSampleType::Uint;
                } else if (kind == TextureSampleKind::UnfilterableFloat) {
                    texture.texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;
                } else {
                    texture.texture.sampleType = wgpu::TextureSampleType::Float;
                }
                const u32 element = ordinal++;
                if (element > shared_first) {
                    ++binding_index;
                    continue;
                }
                auto& sampler = entries.emplace_back();
                sampler.binding = binding_index++;
                sampler.visibility = visibility;
                if (desc.is_depth) {
                    sampler.sampler.type = wgpu::SamplerBindingType::Comparison;
                } else if (kind == TextureSampleKind::Float && element != shared_first) {
                    sampler.sampler.type = wgpu::SamplerBindingType::Filtering;
                } else {
                    sampler.sampler.type = wgpu::SamplerBindingType::NonFiltering;
                }
            }
        }
        wgpu::BindGroupLayoutDescriptor layout_desc{};
        layout_desc.entryCount = entries.size();
        layout_desc.entries = entries.data();
        variant->layout = device.CreateBindGroupLayout(&layout_desc);
        wgpu::PipelineLayoutDescriptor pipeline_layout_desc{};
        pipeline_layout_desc.bindGroupLayoutCount = 1;
        pipeline_layout_desc.bindGroupLayouts = &variant->layout;

        wgpu::ComputePipelineDescriptor desc{};
        desc.layout = device.CreatePipelineLayout(&pipeline_layout_desc);
        desc.compute.module = MakeModule(code, key.unique_hash);
        desc.compute.entryPoint = "main";
        device.CreateComputePipelineAsync(
            &desc, wgpu::CallbackMode::AllowSpontaneous,
            [slot = variant->pipeline, hash = key.unique_hash](
                wgpu::CreatePipelineAsyncStatus status, wgpu::ComputePipeline pipeline,
                wgpu::StringView message) {
                if (status == wgpu::CreatePipelineAsyncStatus::Success) {
                    *slot = std::move(pipeline);
                    return;
                }
                LOG_ERROR(Render, "WebGPU compute pipeline {:016x}: {}", hash,
                          std::string_view{message.data, message.length});
            });
        return true;
    } catch (const std::exception& exception) {
        LOG_ERROR(Render, "WebGPU compute shader translation failed: {}", exception.what());
        return false;
    }
}

bool ComputePipeline::Configure() {
    if (!valid) {
        return false;
    }
    auto& texture_cache = cache.texture_cache;
    auto& buffer_cache = cache.buffer_cache;
    auto& gpu_memory = *cache.gpu_memory;
    const auto& qmd{cache.kepler_compute->launch_description};

    buffer_cache.SetComputeUniformBufferState(info.constant_buffer_mask, &uniform_buffer_sizes);
    buffer_cache.UnbindComputeStorageBuffers();
    size_t ssbo_index{};
    for (const auto& desc : info.storage_buffers_descriptors) {
        buffer_cache.BindComputeStorageBuffer(ssbo_index, desc.cbuf_index, desc.cbuf_offset,
                                              desc.is_written);
        ++ssbo_index;
    }

    texture_cache.SynchronizeDescriptors(true);
    boost::container::small_vector<VideoCommon::ImageViewInOut, 32> views;
    boost::container::small_vector<VideoCommon::SamplerId, 32> samplers;
    const auto& cbufs{qmd.const_buffer_config};
    const bool via_header_index{qmd.linked_tsc != 0};
    for (const auto& desc : info.texture_descriptors) {
        for (u32 index = 0; index < desc.count; ++index) {
            const u32 index_offset{index << desc.size_shift};
            const GPUVAddr addr{cbufs[desc.cbuf_index].Address() + desc.cbuf_offset + index_offset};
            u32 raw{gpu_memory.Read<u32>(addr) << desc.shift_left};
            if (desc.has_secondary) {
                const GPUVAddr separate_addr{cbufs[desc.secondary_cbuf_index].Address() +
                                             desc.secondary_cbuf_offset + index_offset};
                raw |= gpu_memory.Read<u32>(separate_addr) << desc.secondary_shift_left;
            }
            const auto handle{Tegra::Texture::TexturePair(raw, via_header_index)};
            views.push_back({handle.first});
            samplers.push_back(texture_cache.GetSamplerId(handle.second, true));
        }
    }
    texture_cache.FillImageViews(std::span(views.data(), views.size()), true);

    buffer_cache.UnbindComputeTextureBuffers();
    buffer_cache.UpdateComputeBuffers();
    auto& runtime = buffer_cache.runtime;
    runtime.ClearDescriptors();
    buffer_cache.BindHostComputeBuffers();
    buffer_cache.any_buffer_uploaded = false;

    std::vector<TextureSampleKind> texture_kinds;
    texture_kinds.reserve(views.size());
    {
        size_t texture_index{};
        for (const auto& desc : info.texture_descriptors) {
            for (u32 i = 0; i < desc.count; ++i) {
                const ImageView& view{texture_cache.GetImageView(views[texture_index++].id)};
                TextureKind kind = view.Handle(desc.type) ? view.Kind() : TextureKind::Float;
                if (desc.is_depth) {
                    kind = TextureKind::Depth;
                }
                texture_kinds.push_back(ToSampleKind(kind));
            }
        }
    }
    auto [it, is_new] = variants.try_emplace(texture_kinds);
    Variant& variant = it->second;
    if (is_new && !Translate(texture_kinds, &variant)) {
        variant.failed = true;
    }
    if (variant.failed || !*variant.pipeline) {
        return false;
    }
    pipeline = *variant.pipeline;

    auto& descriptors = runtime.Descriptors();
    std::vector<wgpu::BindGroupEntry> entries;
    written_buffers.clear();
    read_buffers.clear();
    size_t descriptor_index{};
    u32 binding_index{};
    for (const auto& desc : info.constant_buffer_descriptors) {
        for (u32 i = 0; i < desc.count; ++i) {
            if (descriptor_index >= descriptors.size()) {
                return false;
            }
            BufferDescriptor& bound = descriptors[descriptor_index++];
            const u32 size = UniformBindingSize(info, desc.index);
            bound.data.resize(size, 0);
            auto& entry = entries.emplace_back();
            entry.binding = binding_index++;
            entry.buffer = cache.uniform_ring.Buffer();
            entry.offset = cache.uniform_ring.Push(bound.data);
            entry.size = size;
        }
    }
    const u32 num_ssbos{Shader::NumDescriptors(info.storage_buffers_descriptors)};
    const u32 packed_first{
        Shader::Backend::WGSL::PackedSsboFirst(num_ssbos, cache.max_storage_buffers)};
    packed_copies.clear();
    shadow_copies.clear();
    written_entries.clear();
    std::array<u32, Shader::Backend::WGSL::MAX_PACKED_SSBOS> pack_bases{};
    u64 pack_size{};
    u32 pack_binding{};
    size_t storage_index{};
    for (const auto& desc : info.storage_buffers_descriptors) {
        for (u32 i = 0; i < desc.count; ++i) {
            if (descriptor_index >= descriptors.size()) {
                return false;
            }
            const BufferDescriptor& bound = descriptors[descriptor_index++];
            if (!bound.buffer || bound.offset + 4 > bound.buffer.GetSize() || bound.size < 4) {
                return false;
            }
            if (storage_index >= packed_first) {
                // Copies need 4-byte aligned ranges.
                if (bound.offset % 4 != 0) {
                    return false;
                }
                if (storage_index == packed_first) {
                    pack_binding = binding_index;
                }
                const u64 size = std::min<u64>(Common::AlignUp(bound.size, 4),
                                               bound.buffer.GetSize() - bound.offset) & ~3ULL;
                pack_bases[storage_index - packed_first] = static_cast<u32>(pack_size / 4);
                packed_copies.push_back({bound.buffer, bound.offset, size, pack_size,
                                         desc.is_written});
                pack_size = Common::AlignUp(pack_size + size, 256);
                ++binding_index;
                ++storage_index;
                continue;
            }
            auto& entry = entries.emplace_back();
            entry.binding = binding_index++;
            entry.buffer = bound.buffer;
            entry.offset = bound.offset;
            entry.size = std::min<u64>(bound.size, bound.buffer.GetSize() - bound.offset);
            const bool writable = storage_index < variant.storage_writable.size() &&
                                  variant.storage_writable[storage_index];
            ++storage_index;
            if (writable) {
                written_entries.push_back(entries.size() - 1);
            }
            (writable ? written_buffers : read_buffers).push_back(bound.buffer.Get());
        }
    }
    if (!packed_copies.empty()) {
        if (!cache.ssbo_pack || cache.ssbo_pack.GetSize() < pack_size) {
            wgpu::BufferDescriptor pack_desc{};
            pack_desc.size = std::bit_ceil(std::max<u64>(pack_size, 1ULL << 20));
            pack_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                              wgpu::BufferUsage::CopyDst;
            cache.ssbo_pack = Thread::Get().device.CreateBuffer(&pack_desc);
        }
        pack_buffer = cache.ssbo_pack;
        auto& pack_entry = entries.emplace_back();
        pack_entry.binding = pack_binding;
        pack_entry.buffer = pack_buffer;
        pack_entry.size = pack_size;
        auto& bases_entry = entries.emplace_back();
        bases_entry.binding = pack_binding + 1;
        bases_entry.buffer = cache.uniform_ring.Buffer();
        bases_entry.offset = cache.uniform_ring.Push(std::span<const u8>(
            reinterpret_cast<const u8*>(pack_bases.data()), sizeof(pack_bases)));
        bases_entry.size = sizeof(pack_bases);
    }
    u32 ordinal{};
    const u32 shared_first{Shader::Backend::WGSL::SharedSamplerFirst(
        Shader::NumDescriptors(info.texture_descriptors), cache.max_samplers)};
    size_t texture_index{};
    for (const auto& desc : info.texture_descriptors) {
        for (u32 i = 0; i < desc.count; ++i) {
            const size_t slot = texture_index++;
            ImageView& view{texture_cache.GetImageView(views[slot].id)};
            const TextureKind kind{FromSampleKind(texture_kinds[slot])};
            wgpu::TextureView handle = view.Handle(desc.type);
            if (!handle || view.Kind() != kind) {
                DebugCount(fmt::format("compute {:016x} dummy texture: view {} fmt {} size {}x{} "
                                       "viewtype {} desctype {} kind {}/{}", key.unique_hash,
                                       views[slot].id.index != 0 ? "real" : "null", view.format,
                                       view.size.width, view.size.height,
                                       static_cast<u32>(view.type), static_cast<u32>(desc.type),
                                       static_cast<u32>(view.Kind()), static_cast<u32>(kind)));
                handle = cache.DummyTexture(desc.type, kind);
            } else {
                DebugCount(fmt::format("compute {:016x} texture: fmt {} size {}x{}", key.unique_hash,
                                       view.format, view.size.width, view.size.height));
            }
            auto& texture_entry = entries.emplace_back();
            texture_entry.binding = binding_index++;
            texture_entry.textureView = handle;
            const u32 element = ordinal++;
            if (element > shared_first) {
                ++binding_index;
                continue;
            }
            Sampler& sampler{texture_cache.GetSampler(samplers[slot])};
            auto& sampler_entry = entries.emplace_back();
            sampler_entry.binding = binding_index++;
            sampler_entry.sampler = sampler.Handle(
                element == shared_first && kind == TextureKind::Float ? TextureKind::UnfilterableFloat
                                                                     : kind,
                desc.is_depth);
        }
    }
    // Written ranges of host buffers the dispatch also reads go through a scratch copy: one
    // dispatch cannot use a buffer both ways.
    u64 shadow_size{};
    for (const size_t entry_index : written_entries) {
        auto& entry = entries[entry_index];
        if (std::ranges::find(read_buffers, entry.buffer.Get()) == read_buffers.end()) {
            continue;
        }
        const u64 size = Common::AlignUp(entry.size, 4);
        if (entry.offset % 4 != 0 || entry.offset + size > entry.buffer.GetSize()) {
            return false;
        }
        shadow_copies.push_back({entry.buffer, entry.offset, size, shadow_size, true});
        shadow_size = Common::AlignUp(shadow_size + size, 256);
    }
    if (!shadow_copies.empty()) {
        if (!cache.ssbo_shadow || cache.ssbo_shadow.GetSize() < shadow_size) {
            wgpu::BufferDescriptor shadow_desc{};
            shadow_desc.size = std::bit_ceil(std::max<u64>(shadow_size, 1ULL << 20));
            shadow_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                                wgpu::BufferUsage::CopyDst;
            cache.ssbo_shadow = Thread::Get().device.CreateBuffer(&shadow_desc);
        }
        shadow_buffer = cache.ssbo_shadow;
        size_t copy_index{};
        for (const size_t entry_index : written_entries) {
            auto& entry = entries[entry_index];
            if (copy_index < shadow_copies.size() &&
                shadow_copies[copy_index].buffer.Get() == entry.buffer.Get() &&
                shadow_copies[copy_index].offset == entry.offset) {
                std::ranges::replace(written_buffers, entry.buffer.Get(), shadow_buffer.Get());
                entry.buffer = shadow_buffer;
                entry.offset = shadow_copies[copy_index].pack_offset;
                ++copy_index;
            }
        }
    }
    wgpu::BindGroupDescriptor bind_desc{};
    bind_desc.layout = variant.layout;
    bind_desc.entryCount = entries.size();
    bind_desc.entries = entries.data();
    bind_group = Thread::Get().device.CreateBindGroup(&bind_desc);
    return true;
}

} // namespace WebGPU
