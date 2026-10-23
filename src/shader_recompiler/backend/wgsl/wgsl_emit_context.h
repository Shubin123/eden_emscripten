// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/ranges.h>

#include "shader_recompiler/backend/wgsl/emit_wgsl.h"
#include "shader_recompiler/backend/wgsl/var_alloc.h"
#include "shader_recompiler/stage.h"

namespace Shader {
struct Info;
struct Profile;
struct RuntimeInfo;
} // namespace Shader

namespace Shader::Backend {
struct Bindings;
}

namespace Shader::IR {
class Inst;
struct Program;
} // namespace Shader::IR

namespace Shader::Backend::WGSL {

struct TextureDefinition {
    u32 binding;   ///< Binding of element 0's texture; its sampler follows it
    u32 count;
    TextureType type;
    bool is_depth; ///< Depth compare reads
    TextureSampleKind kind;
};

class EmitContext {
public:
    explicit EmitContext(IR::Program& program, Bindings& bindings, const Profile& profile_,
                         const RuntimeInfo& runtime_info_, const Options& options_);

    template <WgslVarType type, typename... Args>
    void Add(const char* format_str, IR::Inst& inst, Args&&... args) {
        const auto var_def{var_alloc.AddDefine(inst, type)};
        if (var_def.empty()) {
            // Unused result: discard the value with a phony assignment.
            code += "_";
            code += fmt::format(fmt::runtime(format_str + 2), std::forward<Args>(args)...);
        } else {
            code += fmt::format(fmt::runtime(format_str), var_def, std::forward<Args>(args)...);
        }
        code += '\n';
    }

    template <typename... Args>
    void AddU1(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::U1>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddF16x2(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::F16x2>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddU32(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::U32>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddF32(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::F32>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddU64(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::U64>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddF64(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::F64>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddU32x2(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::U32x2>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddF32x2(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::F32x2>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddU32x3(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::U32x3>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddF32x3(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::F32x3>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddU32x4(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::U32x4>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddF32x4(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::F32x4>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddPrecF32(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::PrecF32>(format_str, inst, args...);
    }

    template <typename... Args>
    void AddPrecF64(const char* format_str, IR::Inst& inst, Args&&... args) {
        Add<WgslVarType::PrecF64>(format_str, inst, args...);
    }

    template <typename... Args>
    void Add(const char* format_str, Args&&... args) {
        code += fmt::format(fmt::runtime(format_str), std::forward<Args>(args)...);
        code += '\n';
    }

    /// Wraps the shader body into the module and entry point.
    std::string Finish(const IR::Program& program);

    std::string header;
    std::string code;
    VarAlloc var_alloc;
    const Info& info;
    const Profile& profile;
    const RuntimeInfo& runtime_info;
    const Options& options;

    Stage stage{};
    std::string_view stage_name = "invalid";

    std::vector<TextureDefinition> textures;

    u32 num_safety_loop_vars{};

    bool uses_cc_carry{};
    bool uses_storage_atomics{};
    bool uses_shared_atomics{};
    /// Vertex stages cannot write storage buffers in WebGPU.
    bool storage_read_only{};
    /// Storage buffers from this index on are read through the packed binding.
    u32 packed_ssbo_first{~0u};

    /// Expression for word `word` of storage buffer `index`.
    [[nodiscard]] std::string SsboElement(u32 index, std::string_view word) const;

private:
    void DefineConstantBuffers(Bindings& bindings);
    void DefineConstantBufferIndirect();
    void DefineStorageBuffers(Bindings& bindings);
    void DefineTextures(Bindings& bindings);
    void DefineInterface();
    void DefineHelperFunctions();
};

} // namespace Shader::Backend::WGSL
