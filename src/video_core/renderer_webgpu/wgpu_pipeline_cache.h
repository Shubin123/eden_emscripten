// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <webgpu/webgpu_cpp.h>

#include "common/common_types.h"
#include "shader_recompiler/backend/wgsl/emit_wgsl.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_webgpu/wgpu_buffer_cache.h"
#include "video_core/renderer_webgpu/wgpu_texture_cache.h"
#include "video_core/renderer_webgpu/wgpu_uniform_ring.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_environment.h"

namespace Shader::IR {
struct Program;
}

namespace WebGPU {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using Shader::Backend::WGSL::TextureSampleKind;

struct GraphicsPipelineKey {
    std::array<u64, 6> unique_hashes;
    Vulkan::FixedPipelineState state;

    size_t Hash() const noexcept;
    bool operator==(const GraphicsPipelineKey& rhs) const noexcept;

    size_t Size() const noexcept {
        return sizeof(unique_hashes) + state.Size();
    }
};

struct GraphicsPipelineKeyHash {
    size_t operator()(const GraphicsPipelineKey& key) const noexcept {
        return key.Hash();
    }
};

/// Everything a pipeline variant depends on besides the guest pipeline key.
struct VariantKey {
    std::vector<TextureSampleKind> texture_kinds;
    std::array<wgpu::TextureFormat, 8> color_formats{};
    wgpu::TextureFormat depth_format{};
    u32 stencil_masks{}; ///< front read/write and back read/write masks
    std::array<u32, 3> depth_bias{};
    wgpu::IndexFormat strip_index_format{};

    auto operator<=>(const VariantKey&) const = default;
};

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

class PipelineCache;

class GraphicsPipeline {
public:
    explicit GraphicsPipeline(PipelineCache& cache, const GraphicsPipelineKey& key,
                              VideoCommon::ShaderCache::GraphicsEnvironments&& environments);

    /// Binds this pipeline's resources for the next draw. Returns false if the draw has to be
    /// skipped.
    bool Configure(bool is_indexed);

    /// Resources resolved by the last Configure call.
    wgpu::RenderPipeline render_pipeline;
    /// Layout of the pulled vertex buffer group, null when every buffer has a vertex slot.
    wgpu::BindGroupLayout pulled_layout;
    /// Guest vertex buffers the shader fetches itself, in pulled-buffer order.
    std::vector<u32> pulled_buffers;
    wgpu::BindGroup bind_group;
    u32 draw_info_offset{};
    const RenderTargets* targets{};
    /// Host buffers the last Configure bound as writable and as read-only storage.
    std::vector<WGPUBuffer> written_buffers;
    std::vector<WGPUBuffer> read_buffers;
    /// Entries of `bind_group`, and which of them are writable storage, so a draw can rebind
    /// writable buffers to copies.
    std::vector<wgpu::BindGroupEntry> bind_entries;
    std::vector<size_t> written_entries;
    wgpu::BindGroupLayout bind_layout;

    struct VertexBufferUse {
        u32 stride{};
        bool per_instance{};
        bool used{};
        u32 slot{}; ///< WebGPU vertex buffer slot; used guest buffers are packed from zero
    };
    /// Vertex buffers the pipeline reads, to keep draws within the bound ranges.
    std::array<VertexBufferUse, Maxwell::NumVertexArrays> vertex_buffer_uses{};

    bool IsValid() const noexcept {
        return valid;
    }

private:
    struct Variant {
        /// Set once asynchronous creation succeeded.
        std::shared_ptr<wgpu::RenderPipeline> pipeline = std::make_shared<wgpu::RenderPipeline>();
        wgpu::BindGroupLayout layout;
        wgpu::BindGroupLayout pulled_layout;
        std::vector<bool> storage_writable; ///< Per storage binding, in layout order
        bool failed{};
    };

    /// Translates the guest shaders; builds `variant` unless only descriptor info is wanted.
    bool Translate(const VariantKey& variant_key, Variant& variant, bool infos_only);
    /// Whether the guest program at `index` is translated (a skipped geometry shader is not).
    bool UsesStage(size_t index) const {
        return key.unique_hashes[index] != 0 && !(index == 4 && skip_geometry);
    }
    Variant* FindVariant(const VariantKey& variant_key);

    PipelineCache& cache;
    GraphicsPipelineKey key;
    VideoCommon::ShaderCache::GraphicsEnvironments environments;

    std::array<Shader::Info, 5> stage_infos{};
    std::array<bool, 5> enabled_stages{};
    std::array<u32, 5> enabled_uniform_buffer_masks{};
    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};
    size_t num_textures{};
    bool valid{};
    bool infos_ready{};
    bool skip_geometry{};

    std::map<VariantKey, Variant> variants;
    Variant* last_variant{};
    VariantKey last_variant_key;
};

struct ComputePipelineKey {
    u64 unique_hash{};
    u32 shared_memory_size{};
    std::array<u32, 3> workgroup_size{};

    auto operator<=>(const ComputePipelineKey&) const = default;
};

class ComputePipeline {
public:
    explicit ComputePipeline(PipelineCache& cache, const ComputePipelineKey& key,
                             VideoCommon::ComputeEnvironment&& env);

    /// Binds this pipeline's resources for the next dispatch. Returns false if the dispatch has
    /// to be skipped.
    bool Configure();

    /// Resources resolved by the last Configure call.
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroup bind_group;
    std::vector<WGPUBuffer> written_buffers;
    std::vector<WGPUBuffer> read_buffers;

    /// Storage buffers read through the packed binding: copied into `pack_buffer` before the
    /// dispatch and, when written, back afterwards.
    struct PackedCopy {
        wgpu::Buffer buffer;
        u64 offset{};
        u64 size{};
        u64 pack_offset{};
        bool written{};
    };
    std::vector<PackedCopy> packed_copies;
    wgpu::Buffer pack_buffer;
    /// Written storage ranges of host buffers the dispatch also reads, swapped for copies in
    /// `shadow_buffer` (pack_offset is the offset there).
    std::vector<PackedCopy> shadow_copies;
    wgpu::Buffer shadow_buffer;

private:
    struct Variant {
        /// Set once asynchronous creation succeeded.
        std::shared_ptr<wgpu::ComputePipeline> pipeline =
            std::make_shared<wgpu::ComputePipeline>();
        /// Set when asynchronous creation failed.
        std::shared_ptr<std::atomic<bool>> creation_failed =
            std::make_shared<std::atomic<bool>>(false);
        wgpu::BindGroupLayout layout;
        std::vector<bool> storage_writable; ///< Per storage binding, in layout order
        bool nested_barriers{};
        bool failed{};
    };

    /// Translates the guest shader; builds `variant` unless only descriptor info is wanted.
    bool Translate(const std::vector<TextureSampleKind>& texture_kinds, Variant* variant,
                   bool pack_all = false, bool nested_barriers = false);
    /// The variant for these texture kinds, with every storage buffer packed or not.
    Variant& FindVariant(const std::vector<TextureSampleKind>& texture_kinds, bool pack_all);

    PipelineCache& cache;
    ComputePipelineKey key;
    VideoCommon::ComputeEnvironment env;

    Shader::Info info;
    VideoCommon::ComputeUniformBufferSizes uniform_buffer_sizes{};
    std::vector<size_t> written_entries;
    bool valid{};

    std::map<std::pair<std::vector<TextureSampleKind>, bool>, Variant> variants;
};

class PipelineCache : public VideoCommon::ShaderCache {
    friend GraphicsPipeline;
    friend ComputePipeline;

public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory, Scheduler& scheduler,
                           UniformRing& uniform_ring, BufferCache& buffer_cache,
                           TextureCache& texture_cache);
    ~PipelineCache();

    GraphicsPipeline* CurrentGraphicsPipeline();
    ComputePipeline* CurrentComputePipeline();

    const wgpu::TextureView& DummyTexture(Shader::TextureType type, TextureKind kind);

    /// A copy of a render target sampled by the same draw (WebGPU forbids feedback loops).
    wgpu::TextureView FeedbackCopy(const ImageView& view, Shader::TextureType type);

private:
    Scheduler& scheduler;
    UniformRing& uniform_ring;
    BufferCache& buffer_cache;
    TextureCache& texture_cache;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;
    Vulkan::DynamicFeatures dynamic_features{};
    ShaderPools pools;

    GraphicsPipelineKey graphics_key{};
    std::unordered_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>,
                       GraphicsPipelineKeyHash>
        graphics_cache;
    GraphicsPipeline* current_pipeline{};
    std::map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_cache;
    /// Scratch storage for packed storage buffers, grown on demand.
    wgpu::Buffer ssbo_pack;
    wgpu::Buffer ssbo_shadow;
    u32 max_storage_buffers{};
    u32 max_vertex_buffers{8};
    u32 max_samplers{16};

    std::map<std::pair<Shader::TextureType, TextureKind>, wgpu::TextureView> dummy_views;

    struct FeedbackCopyEntry {
        wgpu::Texture copy;
        u64 pass_serial = ~0ULL;
    };
    std::unordered_map<WGPUTexture, FeedbackCopyEntry> feedback_copies;
    std::vector<wgpu::Texture> dummy_textures;
};

} // namespace WebGPU
