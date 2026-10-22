// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "dynarmic/interface/exec_stats.h"

#include "common/assert.h"
#include "dynarmic/backend/interp/interp.h"
#include "dynarmic/backend/interp/wasm_jit.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/cond.h"
#include "dynarmic/ir/acc_type.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/value.h"

namespace Dynarmic::Backend::Interp {

namespace {

constexpr size_t kPCSamples = 4096;
std::array<std::atomic<u64>, kPCSamples> g_pc_samples{};
std::atomic<u64> g_pc_sample_index{0};

std::atomic<u64> g_dump_request{0};
std::array<std::atomic<u64>, 1024> g_helper_counts{};

std::string HelperHotSpots() {
    std::vector<std::pair<u64, size_t>> sorted;
    for (size_t i = 0; i < g_helper_counts.size(); ++i) {
        if (const u64 count = g_helper_counts[i].exchange(0, std::memory_order_relaxed)) {
            sorted.emplace_back(count, i);
        }
    }
    std::sort(sorted.rbegin(), sorted.rend());
    std::string out;
    for (size_t i = 0; i < std::min<size_t>(sorted.size(), 8); ++i) {
        out += fmt::format(" {}:{:.1f}M", IR::GetNameOf(IR::Opcode(sorted[i].second)), double(sorted[i].first) / 1e6);
    }
    return out;
}
std::atomic<u32> g_dumps_left{6};

std::string HotSpots() {
    std::unordered_map<u64, u32> counts;
    for (auto& sample : g_pc_samples) {
        const u64 pc = sample.exchange(0, std::memory_order_relaxed);
        if (pc != 0) {
            ++counts[pc];
        }
    }
    std::vector<std::pair<u32, u64>> sorted;
    for (const auto& [pc, count] : counts) {
        sorted.emplace_back(count, pc);
    }
    std::sort(sorted.rbegin(), sorted.rend());
    static u32 prints = 0;
    const char* dump_hot = std::getenv("EDEN_INTERP_DUMP_HOT");
    if (!sorted.empty() && g_dumps_left.load() > 0 && dump_hot && ++prints > u32(std::atoi(dump_hot))) {
        g_dumps_left.fetch_sub(1);
        g_dump_request.store(sorted[std::min<size_t>(sorted.size() - 1, 5 - g_dumps_left.load() % 6)].second);
    }
    std::string out;
    for (size_t i = 0; i < std::min<size_t>(sorted.size(), 6); ++i) {
        out += fmt::format(" {:x}x{}", sorted[i].second, sorted[i].first);
    }
    return out;
}

}  // Anonymous namespace

void CountHelperOp(IR::Opcode opcode) {
    static const bool enabled = std::getenv("EDEN_INTERP_STATS") != nullptr;
    if (enabled && size_t(opcode) < g_helper_counts.size()) {
        g_helper_counts[size_t(opcode)].fetch_add(1, std::memory_order_relaxed);
    }
}

bool ShouldDumpBlock(u64 pc) {
    static const std::vector<u64> pcs = [] {
        std::vector<u64> result;
        if (const char* list = std::getenv("EDEN_INTERP_DUMP")) {
            std::string text{list};
            size_t start = 0;
            while (start < text.size()) {
                const size_t end = std::min(text.find(',', start), text.size());
                result.push_back(std::strtoull(text.substr(start, end - start).c_str(), nullptr, 16));
                start = end + 1;
            }
        }
        return result;
    }();
    return std::find(pcs.begin(), pcs.end(), pc) != pcs.end();
}

bool SamplePC(u64 pc) {
    static const bool enabled = std::getenv("EDEN_INTERP_STATS") != nullptr;
    if (!enabled) {
        return false;
    }
    u64 expected = pc;
    if (g_dump_request.load(std::memory_order_relaxed) == pc && g_dump_request.compare_exchange_strong(expected, 0)) {
        return true;
    }
    thread_local u32 counter = 0;
    if ((++counter & 0xFF) != 0) {
        return false;
    }
    g_pc_samples[g_pc_sample_index.fetch_add(1, std::memory_order_relaxed) % kPCSamples].store(pc, std::memory_order_relaxed);
    return false;
}

namespace {
std::array<std::atomic<u64>, 8> g_exec_cycles{};
std::array<std::atomic<u64>, 8> g_exec_busy_ns{};
}  // Anonymous namespace

void RecordExecutedCycles(u64 cycles, u32 core, u64 busy_ns) {
    g_exec_cycles[core & 7].fetch_add(cycles, std::memory_order_relaxed);
    g_exec_busy_ns[core & 7].fetch_add(busy_ns, std::memory_order_relaxed);
    static const bool enabled = std::getenv("EDEN_INTERP_STATS") != nullptr;
    if (!enabled) {
        return;
    }
    static std::array<std::atomic<u64>, 8> per_core{};
    per_core[core & 7].fetch_add(cycles, std::memory_order_relaxed);
    static std::atomic<u64> total{0};
    static std::atomic<s64> last_report{0};
    const u64 sum = total.fetch_add(cycles, std::memory_order_relaxed) + cycles;
    const s64 now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    static std::atomic<u64> last_sum{0};
    s64 previous = last_report.load(std::memory_order_relaxed);
    if (now - previous >= 2000 && last_report.compare_exchange_strong(previous, now)) {
        const u64 delta = sum - last_sum.exchange(sum);
        if (previous != 0) {
            std::string cores;
            for (auto& count : per_core) {
                cores += fmt::format(" {:.0f}", double(count.exchange(0)) / double(now - previous) / 1000.0);
            }
            std::fprintf(stderr, "[interp] %.2f M guest cycles/s, cores:%s, jit: %s, hot:%s helpers:%s\n", double(delta) / double(now - previous) / 1000.0, cores.c_str(), JitStats().c_str(), HotSpots().c_str(), HelperHotSpots().c_str());
        }
    }
}

static u8 WidthOf(IR::Type type) {
    switch (type) {
    case IR::Type::U1:
        return 1;
    case IR::Type::U8:
        return 8;
    case IR::Type::U16:
        return 16;
    case IR::Type::U32:
        return 32;
    case IR::Type::U64:
        return 64;
    default:
        return 128;
    }
}

static Slot ImmediateSlot(const IR::Value& value) {
    Slot slot{};
    switch (value.GetType()) {
    case IR::Type::U1:
    case IR::Type::U8:
    case IR::Type::U16:
    case IR::Type::U32:
    case IR::Type::U64:
        slot.lo = value.GetImmediateAsU64();
        break;
    case IR::Type::A32Reg:
        slot.lo = u64(value.GetA32RegRef());
        break;
    case IR::Type::A32ExtReg:
        slot.lo = u64(value.GetA32ExtRegRef());
        break;
    case IR::Type::A64Reg:
        slot.lo = u64(value.GetA64RegRef());
        break;
    case IR::Type::A64Vec:
        slot.lo = u64(value.GetA64VecRef());
        break;
    case IR::Type::Cond:
        slot.lo = u64(value.GetCond());
        break;
    case IR::Type::AccType:
        slot.lo = u64(value.GetAccType());
        break;
    case IR::Type::CoprocInfo: {
        const auto info = value.GetCoprocInfo();
        std::memcpy(&slot.lo, info.data(), sizeof(info));
        break;
    }
    case IR::Type::NZCVFlags:
        // Only the empty marker can be an NZCV immediate.
        slot.lo = 0;
        break;
    default:
        UNREACHABLE_MSG("Unexpected immediate type {}", u32(value.GetType()));
    }
    return slot;
}

void Compile(CompiledBlock& out, const IR::Block& block) {
    out.ops.clear();
    out.constants.clear();
    out.terminal = block.GetTerminal();
    out.cycle_count = block.CycleCount();
    out.location = block.Location();
    out.end_location = block.EndLocation();
    out.cond = block.GetCondition();
    if (block.HasConditionFailedLocation()) {
        out.cond_failed_location = block.ConditionFailedLocation();
        out.cond_failed_cycle_count = block.ConditionFailedCycleCount();
    }
    out.link_then = nullptr;
    out.link_else = nullptr;

    std::unordered_map<const IR::Inst*, u32> slot_of;
    slot_of.reserve(64);

    u32 next_slot = 0;
    for (const auto& inst : block.instructions) {
        const IR::Opcode opcode = inst.GetOpcode();
        if (opcode == IR::Opcode::Void) {
            continue;
        }
        slot_of.emplace(&inst, next_slot++);
    }

    out.first_constant = next_slot;
    std::vector<Slot> constants;
    const auto constant = [&](const Slot& slot) {
        constants.push_back(slot);
        return u32(out.first_constant + constants.size() - 1);
    };
    const u32 zero_slot = constant(Slot{});

    for (const auto& inst : block.instructions) {
        const IR::Opcode opcode = inst.GetOpcode();
        if (opcode == IR::Opcode::Void) {
            continue;
        }
        Op op{};
        op.opcode = opcode;
        op.dst = slot_of.at(&inst);
        op.width = WidthOf(inst.GetType());
        const size_t num_args = inst.NumArgs();
        ASSERT(num_args <= op.args.size());
        for (size_t i = 0; i < num_args; ++i) {
            const IR::Value arg = inst.GetArg(i);
            if (arg.IsEmpty()) {
                op.args[i] = zero_slot;
            } else if (arg.IsImmediate()) {
                op.args[i] = constant(ImmediateSlot(arg));
            } else {
                const IR::Inst* source = arg.GetInstRecursive();
                const auto it = slot_of.find(source);
                ASSERT_MSG(it != slot_of.end(), "IR value used before definition");
                op.args[i] = it->second;
            }
            if (i == 0 && IR::IsAPseudoOperation(opcode) && !arg.IsEmpty()) {
                // Flag pseudo-operations may refer to a folded immediate as well as an instruction.
                op.width = WidthOf(arg.IsImmediate() ? arg.GetType() : arg.GetInstRecursive()->GetType());
            }
        }
        if (opcode == IR::Opcode::VectorTable) {
            // The lookup needs to know how many of the four table registers are present.
            op.width = 0;
            for (size_t i = 0; i < num_args; ++i) {
                op.width += inst.GetArg(i).IsEmpty() ? 0 : 1;
            }
        }
        out.ops.push_back(op);
    }

    out.constants = std::move(constants);
    out.num_slots = u32(out.first_constant + out.constants.size());
}

}  // namespace Dynarmic::Backend::Interp

namespace Dynarmic {

ExecStats GetExecStats(std::uint32_t core) {
    using namespace Backend::Interp;
    return {g_exec_cycles[core & 7].load(std::memory_order_relaxed), g_exec_busy_ns[core & 7].load(std::memory_order_relaxed)};
}

}  // namespace Dynarmic
