// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <string>
#include <tuple>
#include <type_traits>

#include "common/div_ceil.h"
#include "common/settings.h"
#include "shader_recompiler/backend/wgsl/emit_wgsl.h"
#include "shader_recompiler/backend/wgsl/emit_wgsl_instructions.h"
#include "shader_recompiler/backend/wgsl/wgsl_emit_context.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"

namespace Shader::Backend::WGSL {
namespace {
template <class Func>
struct FuncTraits {};

template <class ReturnType_, class... Args>
struct FuncTraits<ReturnType_ (*)(Args...)> {
    using ReturnType = ReturnType_;

    static constexpr size_t NUM_ARGS = sizeof...(Args);

    template <size_t I>
    using ArgType = std::tuple_element_t<I, std::tuple<Args...>>;
};

template <typename ArgType>
auto Arg(EmitContext& ctx, const IR::Value& arg) {
    if constexpr (std::is_same_v<ArgType, std::string_view>) {
        return ctx.var_alloc.Consume(arg);
    } else if constexpr (std::is_same_v<ArgType, const IR::Value&>) {
        return arg;
    } else if constexpr (std::is_same_v<ArgType, u32>) {
        return arg.U32();
    } else if constexpr (std::is_same_v<ArgType, IR::Attribute>) {
        return arg.Attribute();
    } else if constexpr (std::is_same_v<ArgType, IR::Patch>) {
        return arg.Patch();
    } else if constexpr (std::is_same_v<ArgType, IR::Reg>) {
        return arg.Reg();
    }
}

template <auto func, bool is_first_arg_inst, size_t... I>
void Invoke(EmitContext& ctx, IR::Inst* inst, std::index_sequence<I...>) {
    using Traits = FuncTraits<decltype(func)>;
    if constexpr (is_first_arg_inst) {
        func(ctx, *inst, Arg<typename Traits::template ArgType<I + 2>>(ctx, inst->Arg(I))...);
    } else {
        func(ctx, Arg<typename Traits::template ArgType<I + 1>>(ctx, inst->Arg(I))...);
    }
}

template <auto func>
void Invoke(EmitContext& ctx, IR::Inst* inst) {
    using Traits = FuncTraits<decltype(func)>;
    static_assert(Traits::NUM_ARGS >= 1, "Insufficient arguments");
    if constexpr (Traits::NUM_ARGS == 1) {
        Invoke<func, false>(ctx, inst, std::make_index_sequence<0>{});
    } else {
        using FirstArgType = typename Traits::template ArgType<1>;
        static constexpr bool is_first_arg_inst = std::is_same_v<FirstArgType, IR::Inst&>;
        using Indices = std::make_index_sequence<Traits::NUM_ARGS - (is_first_arg_inst ? 2 : 1)>;
        Invoke<func, is_first_arg_inst>(ctx, inst, Indices{});
    }
}

void EmitInst(EmitContext& ctx, IR::Inst* inst) {
    switch (inst->GetOpcode()) {
#define OPCODE(name, result_type, ...)                                                             \
    case IR::Opcode::name:                                                                         \
        return Invoke<&Emit##name>(ctx, inst);
#include "shader_recompiler/frontend/ir/opcodes.inc"
#undef OPCODE
    }
    throw LogicError("Invalid opcode {}", inst->GetOpcode());
}

bool IsReference(IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::Reference;
}

void PrecolorInst(IR::Inst& phi) {
    // Insert phi moves before references to avoid overwriting other phis
    const size_t num_args{phi.NumArgs()};
    for (size_t i = 0; i < num_args; ++i) {
        IR::Block& phi_block{*phi.PhiBlock(i)};
        auto it{std::find_if_not(phi_block.rbegin(), phi_block.rend(), IsReference).base()};
        IR::IREmitter ir{phi_block, it};
        const IR::Value arg{phi.Arg(i)};
        if (arg.IsImmediate()) {
            ir.PhiMove(phi, arg);
        } else {
            ir.PhiMove(phi, IR::Value{arg.InstRecursive()});
        }
    }
    for (size_t i = 0; i < num_args; ++i) {
        IR::IREmitter{*phi.PhiBlock(i)}.Reference(IR::Value{&phi});
    }
}

void Precolor(const IR::Program& program) {
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& phi : block->Instructions()) {
            if (!IR::IsPhi(phi)) {
                break;
            }
            PrecolorInst(phi);
        }
    }
}

void EmitCode(EmitContext& ctx, const IR::Program& program) {
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        switch (node.type) {
        case IR::AbstractSyntaxNode::Type::Block:
            for (IR::Inst& inst : node.data.block->Instructions()) {
                EmitInst(ctx, &inst);
            }
            break;
        case IR::AbstractSyntaxNode::Type::If:
            ctx.Add("if ({}) {{", ctx.var_alloc.Consume(node.data.if_node.cond));
            break;
        case IR::AbstractSyntaxNode::Type::EndIf:
            ctx.Add("}}");
            break;
        case IR::AbstractSyntaxNode::Type::Break:
            if (node.data.break_node.cond.IsImmediate()) {
                if (node.data.break_node.cond.U1()) {
                    ctx.Add("break;");
                }
            } else {
                ctx.Add("if ({}) {{ break; }}", ctx.var_alloc.Consume(node.data.break_node.cond));
            }
            break;
        case IR::AbstractSyntaxNode::Type::Return:
        case IR::AbstractSyntaxNode::Type::Unreachable:
            ctx.Add("return;");
            break;
        case IR::AbstractSyntaxNode::Type::Loop:
            ctx.Add("loop {{");
            break;
        case IR::AbstractSyntaxNode::Type::Repeat:
            if (Settings::values.disable_shader_loop_safety_checks) {
                ctx.Add("if (!({})) {{ break; }}\n}}", ctx.var_alloc.Consume(node.data.repeat.cond));
            } else {
                const u32 loop_var{ctx.num_safety_loop_vars++};
                ctx.Add("loop{} -= 1;\nif (loop{} < 0 || !({})) {{ break; }}\n}}", loop_var,
                        loop_var, ctx.var_alloc.Consume(node.data.repeat.cond));
            }
            break;
        default:
            throw NotImplementedException("AbstractSyntaxNode Type {}", node.type);
        }
    }
}

void DefineVariables(const EmitContext& ctx, std::string& header) {
    for (u32 i = 0; i < static_cast<u32>(WgslVarType::Void); ++i) {
        const auto type{static_cast<WgslVarType>(i)};
        const auto& tracker{ctx.var_alloc.GetUseTracker(type)};
        const auto type_name{ctx.var_alloc.GetWgslType(type)};
        // Temps/return types that are never used are stored at index 0
        if (tracker.uses_temp) {
            header += fmt::format("var t{}: {};\n", ctx.var_alloc.Representation(0, type), type_name);
        }
        for (u32 index = 0; index < tracker.num_used; ++index) {
            header += fmt::format("var {}: {};\n", ctx.var_alloc.Representation(index, type),
                                  type_name);
        }
    }
    for (u32 i = 0; i < ctx.num_safety_loop_vars; ++i) {
        header += fmt::format("var loop{}: i32 = 0x2000;\n", i);
    }
}
} // Anonymous namespace

std::string EmitWGSL(const Profile& profile, const RuntimeInfo& runtime_info, IR::Program& program,
                     Bindings& bindings, const Options& options) {
    EmitContext ctx{program, bindings, profile, runtime_info, options};
    Precolor(program);
    EmitCode(ctx, program);
    std::string body{"fn main_body() {\n"};
    DefineVariables(ctx, body);
    if (ctx.uses_cc_carry) {
        body += "var carry: u32;\n";
    }
    if (program.info.uses_subgroup_shuffles) {
        body += "var shfl_in_bounds: bool;\nvar shfl_result: u32;\n";
    }
    ctx.code.insert(0, body);
    ctx.code += "}\n";
    return ctx.Finish(program);
}

} // namespace Shader::Backend::WGSL
