// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Portable A32 backend that interprets optimized IR blocks. See a64_interface.cpp.

#include <algorithm>
#include <chrono>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include <boost/icl/interval_set.hpp>
#include "common/assert.h"
#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/backend/interp/interp.h"
#include "dynarmic/backend/interp/interp_exec.h"
#include "dynarmic/backend/interp/interp_fp.h"
#include "dynarmic/backend/interp/interp_vector.h"
#include "dynarmic/backend/interp/sm4_sbox.h"
#include "dynarmic/backend/interp/wasm_jit.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/common/crypto/crc32.h"
#include "dynarmic/frontend/A32/a32_location_descriptor.h"
#include "dynarmic/frontend/A32/a32_types.h"
#include "dynarmic/frontend/A32/translate/a32_translate.h"
#include "dynarmic/interface/A32/a32.h"
#include "dynarmic/interface/A32/coprocessor.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/interface/optimization_flags.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opt_passes.h"

namespace Dynarmic::A32 {

using namespace Backend::Interp;
using IR::Opcode;


namespace {

/// The upper half of the location descriptor holds T (bit 0), E (bit 1), IT (bits 8-15) and the
/// FPSCR mode bits, matching A32::LocationDescriptor::UniqueHash.
struct A32State {
    static constexpr u32 FPSCR_MODE_MASK = A32::LocationDescriptor::FPSCR_MODE_MASK;

    std::array<u32, 16> reg{};
    u32 upper = 0;
    u32 nzcv = 0;
    u32 q = 0;
    u32 ge = 0;
    u32 jaifm = 0;
    u32 fpsr_exc = 0;
    u32 fpsr_qc = 0;
    u32 fpsr_nzcv = 0;
    alignas(16) std::array<u32, 64> ext{};
    volatile u32 halt_reason = 0;
    u32 exclusive_state = 0;
    u32 exclusive_address = 0;
    u64 exclusive_value = 0;
    bool check_bit = false;

    IR::LocationDescriptor Location() const {
        return IR::LocationDescriptor{(u64(upper) << 32) | reg[15]};
    }

    static u32 ExpandGE(u32 bits) {
        u32 ge = 0;
        ge |= (bits & 8) ? 0xFF000000 : 0;
        ge |= (bits & 4) ? 0x00FF0000 : 0;
        ge |= (bits & 2) ? 0x0000FF00 : 0;
        ge |= (bits & 1) ? 0x000000FF : 0;
        return ge;
    }

    u32 Cpsr() const {
        u32 cpsr = nzcv;
        cpsr |= q ? 1 << 27 : 0;
        cpsr |= (ge >> 31) & 1 ? 1 << 19 : 0;
        cpsr |= (ge >> 23) & 1 ? 1 << 18 : 0;
        cpsr |= (ge >> 15) & 1 ? 1 << 17 : 0;
        cpsr |= (ge >> 7) & 1 ? 1 << 16 : 0;
        cpsr |= (upper >> 1) & 1 ? 1 << 9 : 0;
        cpsr |= upper & 1 ? 1 << 5 : 0;
        cpsr |= upper & 0b11111100'00000000;
        cpsr |= (upper & 0b00000011'00000000) << 17;
        cpsr |= jaifm;
        return cpsr;
    }

    void SetCpsr(u32 cpsr) {
        nzcv = cpsr & 0xF0000000;
        q = (cpsr >> 27) & 1;
        ge = ExpandGE(cpsr >> 16);
        upper &= 0xFFFF0000;
        upper |= (cpsr >> 9) & 1 ? 2 : 0;
        upper |= (cpsr >> 5) & 1 ? 1 : 0;
        upper |= cpsr & 0b11111100'00000000;
        upper |= (cpsr >> 17) & 0b00000011'00000000;
        jaifm = cpsr & 0x010001DF;
    }

    u32 Fpscr() const {
        return (upper & FPSCR_MODE_MASK) | fpsr_nzcv | fpsr_exc | (fpsr_qc ? 1u << 27 : 0);
    }

    void SetFpscr(u32 fpscr) {
        upper = (upper & 0x0000FFFF) | (fpscr & FPSCR_MODE_MASK);
        fpsr_nzcv = fpscr & 0xF0000000;
        fpsr_qc = (fpscr >> 27) & 1;
        fpsr_exc = fpscr & 0x9F;
    }
};

}  // Anonymous namespace

struct Jit::Impl final {
    Impl(Jit* jit, UserConfig conf_)
        : conf(std::move(conf_))
        , jit_interface(jit) {
        if (conf.page_table_pointer_mask != 0) {
            pointer_sign_shift = std::countl_zero(conf.page_table_pointer_mask);
        }
        frame.resize(kFrameSlots);
        jit_target = MakeJitTarget();
    }

    HaltReason Run() {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(HaltReason(Atomic::Load(&state.halt_reason)));
        jit_interface->is_executing = true;

        const auto started = std::chrono::steady_clock::now();
        cycles_to_run = conf.enable_cycle_counting ? s64(conf.callbacks->GetTicksRemaining()) : std::numeric_limits<s64>::max();
        cycles_remaining = cycles_to_run;

        CompiledBlock* block = nullptr;
        while (Atomic::Load(&state.halt_reason) == 0 && cycles_remaining > 0) {
            if (!block) {
                block = GetBlock(state.Location());
            }
            if (RunRegion(*block)) {
                block = nullptr;
                continue;
            }
            block = Execute(*block, false);
        }

        RecordExecutedCycles(u64(cycles_to_run - cycles_remaining), conf.processor_id,
                             u64(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()));
        if (conf.enable_cycle_counting) {
            conf.callbacks->AddTicks(u64(cycles_to_run - cycles_remaining));
        }
        const HaltReason hr = HaltReason(Atomic::Exchange(&state.halt_reason, 0));
        PerformRequestedCacheInvalidation(hr);
        jit_interface->is_executing = false;
        return hr;
    }

    HaltReason Step() {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(HaltReason(Atomic::Load(&state.halt_reason)));
        jit_interface->is_executing = true;

        cycles_to_run = 1;
        cycles_remaining = 1;
        if (Atomic::Load(&state.halt_reason) == 0) {
            Atomic::Or(&state.halt_reason, u32(HaltReason::Step));
            CompiledBlock* block = GetBlock(A32::LocationDescriptor{state.Location()}.SetSingleStepping(true));
            Execute(*block, true);
        }

        RecordExecutedCycles(u64(cycles_to_run - cycles_remaining), conf.processor_id, 0);
        if (conf.enable_cycle_counting) {
            conf.callbacks->AddTicks(u64(cycles_to_run - cycles_remaining));
        }
        const HaltReason hr = HaltReason(Atomic::Exchange(&state.halt_reason, 0));
        PerformRequestedCacheInvalidation(hr);
        jit_interface->is_executing = false;
        return hr;
    }

    void ClearCache() {
        std::unique_lock lock{invalidation_mutex};
        invalidate_entire_cache = true;
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void InvalidateCacheRange(u32 start_address, size_t length) {
        std::unique_lock lock{invalidation_mutex};
        invalid_cache_ranges.add(boost::icl::discrete_interval<u32>::closed(start_address, u32(start_address + length - 1)));
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void Reset() {
        ASSERT(!jit_interface->is_executing);
        state = {};
    }

    void HaltExecution(HaltReason hr) {
        Atomic::Or(&state.halt_reason, u32(hr));
    }

    void ClearHalt(HaltReason hr) {
        Atomic::And(&state.halt_reason, ~u32(hr));
    }

    void ClearExclusiveState() {
        state.exclusive_state = 0;
    }

    std::array<u32, 16>& Regs() {
        return state.reg;
    }
    const std::array<u32, 16>& Regs() const {
        return state.reg;
    }
    std::array<u32, 64>& ExtRegs() {
        return state.ext;
    }
    const std::array<u32, 64>& ExtRegs() const {
        return state.ext;
    }
    u32 Cpsr() const {
        return state.Cpsr();
    }
    void SetCpsr(u32 value) {
        state.SetCpsr(value);
    }
    u32 Fpscr() const {
        return state.Fpscr();
    }
    void SetFpscr(u32 value) {
        state.SetFpscr(value);
    }

private:
    CompiledBlock* GetBlock(IR::LocationDescriptor descriptor) {
        // Most lookups hit a small direct-mapped cache in front of the block map.
        const u64 key = descriptor.Value();
        FastDispatchEntry& entry = fast_dispatch[((key ^ (key >> 32)) >> 1) % fast_dispatch.size()];
        if (entry.block && entry.location == key) {
            return entry.block;
        }
        if (const auto it = block_cache.find(key); it != block_cache.end()) {
            entry = {key, it->second.get()};
            return it->second.get();
        }

        const auto arch_descriptor = A32::LocationDescriptor{descriptor};
        ir_block.Reset(arch_descriptor);
        A32::Translate(ir_block, arch_descriptor, conf.callbacks, {conf.arch_version, conf.define_unpredictable_behaviour, conf.hook_hint_instructions});
        Optimization::Optimize(ir_block, conf, polyfill_options);
        if (ShouldDumpBlock(arch_descriptor.PC())) {
            std::fprintf(stderr, "%s\n", IR::DumpBlock(ir_block).c_str());
        }

        auto block = std::make_unique<CompiledBlock>();
        Compile(*block, ir_block);
        block->start_address = arch_descriptor.PC();
        block->end_address = A32::LocationDescriptor{ir_block.EndLocation()}.PC();
        if (block->end_address > block->start_address) {
            block_ranges.AddRange(boost::icl::discrete_interval<u32>::closed(u32(block->start_address), u32(block->end_address - 1)), descriptor);
        }
        if (block->num_slots > frame.size() && overflow_frame.size() < block->num_slots) {
            overflow_frame.resize(block->num_slots);
        }

        CompiledBlock* result = block.get();
        block_cache.emplace(descriptor.Value(), std::move(block));
        entry = {key, result};
        return result;
    }

    void DumpBlock(IR::LocationDescriptor descriptor) {
        const auto arch_descriptor = A32::LocationDescriptor{descriptor};
        auto dump = std::make_unique<IR::Block>(arch_descriptor);
        IR::Block& dump_block = *dump;
        A32::Translate(dump_block, arch_descriptor, conf.callbacks, {conf.arch_version, conf.define_unpredictable_behaviour, conf.hook_hint_instructions});
        Optimization::Optimize(dump_block, conf, polyfill_options);
        std::fprintf(stderr, "[interp] hot block:\n%s\n", IR::DumpBlock(dump_block).c_str());
    }

    void PerformRequestedCacheInvalidation(HaltReason hr) {
        if (!Has(hr, HaltReason::CacheInvalidation)) {
            return;
        }
        std::unique_lock lock{invalidation_mutex};
        ClearHalt(HaltReason::CacheInvalidation);
        if (!invalidate_entire_cache && invalid_cache_ranges.empty()) {
            return;
        }
        fast_dispatch.fill({});
        if (invalidate_entire_cache) {
            block_cache.clear();
            block_ranges.ClearCache();
        } else {
            std::vector<u64> erased_keys;
            std::unordered_set<const CompiledBlock*> erased;
            for (const auto& location : block_ranges.InvalidateRanges(invalid_cache_ranges)) {
                if (const auto it = block_cache.find(location.Value()); it != block_cache.end()) {
                    erased_keys.push_back(it->first);
                    erased.insert(it->second.get());
                }
            }
            // Keep the links and regions that only reference surviving blocks, so an unrelated
            // invalidation does not force the whole cache to be recompiled.
            for (auto& [_, block] : block_cache) {
                if (erased.contains(block->link_then)) {
                    block->link_then = nullptr;
                }
                if (erased.contains(block->link_else)) {
                    block->link_else = nullptr;
                }
                if (block->region && std::ranges::any_of(block->region->blocks, [&](const CompiledBlock* b) { return erased.contains(b); })) {
                    block->region.reset();
                    block->entry_count = 0;
                }
            }
            for (const u64 key : erased_keys) {
                block_cache.erase(key);
            }
        }
        invalid_cache_ranges.clear();
        invalidate_entire_cache = false;
    }

    // Memory access

    template<size_t bytes>
    u8* HostPointer(u32 vaddr) const {
        if (!conf.page_table) {
            return nullptr;
        }
        if ((vaddr & 0xFFF) + bytes > 0x1000) {
            return nullptr;
        }
        u64 entry;
        __atomic_load(reinterpret_cast<u64*>(conf.page_table->data()) + (vaddr >> 12), &entry, __ATOMIC_RELAXED);
        if (conf.page_table_marked_bit && ((entry >> *conf.page_table_marked_bit) & 1)) {
            return nullptr;
        }
        if (conf.page_table_pointer_mask != 0) {
            entry &= conf.page_table_pointer_mask;
            entry = u64(s64(entry << pointer_sign_shift) >> pointer_sign_shift);
        }
        if (entry == 0) {
            return nullptr;
        }
        if (conf.absolute_offset_page_table) {
            return reinterpret_cast<u8*>(entry + vaddr);
        }
        return reinterpret_cast<u8*>(entry + (vaddr & 0xFFF));
    }

    template<typename T>
    T Read(u32 vaddr) {
        if (u8* ptr = HostPointer<sizeof(T)>(vaddr)) {
            T value;
            std::memcpy(&value, ptr, sizeof(T));
            return value;
        }
        return T(conf.callbacks->MemoryRead(vaddr, sizeof(T)));
    }

    template<typename T>
    void Write(u32 vaddr, T value) {
        if (u8* ptr = HostPointer<sizeof(T)>(vaddr)) {
            std::memcpy(ptr, &value, sizeof(T));
            return;
        }
        conf.callbacks->MemoryWrite(vaddr, u64(value), sizeof(T));
    }

    /// With Unsafe_IgnoreGlobalMonitor each core keeps its own reservation and the store is a
    /// compare-and-swap against the value that was loaded, as in the native backends.
    bool LocalMonitor() const {
        return conf.HasOptimization(OptimizationFlag::Unsafe_IgnoreGlobalMonitor);
    }

    template<typename T>
    T ExclusiveRead(u32 vaddr) {
        state.exclusive_state = 1;
        if (LocalMonitor()) {
            const T value = Read<T>(vaddr);
            state.exclusive_address = vaddr;
            state.exclusive_value = u64(value);
            return value;
        }
        return conf.global_monitor->ReadAndMark<T>(conf.processor_id, vaddr, [&]() -> T {
            return T(conf.callbacks->MemoryRead(vaddr, sizeof(T)));
        });
    }

    template<typename T>
    bool WriteExclusiveCallback(u32 vaddr, T value, T expected) {
        if constexpr (sizeof(T) == 1) {
            return conf.callbacks->MemoryWriteExclusive8(vaddr, value, expected);
        } else if constexpr (sizeof(T) == 2) {
            return conf.callbacks->MemoryWriteExclusive16(vaddr, value, expected);
        } else if constexpr (sizeof(T) == 4) {
            return conf.callbacks->MemoryWriteExclusive32(vaddr, value, expected);
        } else {
            return conf.callbacks->MemoryWriteExclusive64(vaddr, value, expected);
        }
    }

    template<typename T>
    u32 ExclusiveWrite(u32 vaddr, T value) {
        if (!state.exclusive_state) {
            return 1;
        }
        state.exclusive_state = 0;
        if (LocalMonitor()) {
            if (state.exclusive_address != vaddr) {
                return 1;
            }
            T expected = T(state.exclusive_value);
            if (u8* ptr = HostPointer<sizeof(T)>(vaddr)) {
                return __atomic_compare_exchange_n(reinterpret_cast<T*>(ptr), &expected, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) ? 0 : 1;
            }
            return WriteExclusiveCallback<T>(vaddr, value, expected) ? 0 : 1;
        }
        const bool ok = conf.global_monitor->DoExclusiveOperation<T>(conf.processor_id, vaddr, [&](T expected) -> bool {
            return WriteExclusiveCallback<T>(vaddr, value, expected);
        });
        return ok ? 0 : 1;
    }

    void Coprocessor(const Op& op, Slot* s) {
        std::array<u8, 8> info;
        std::memcpy(info.data(), &ARG(0).lo, sizeof(info));
        const size_t coproc_num = info[0];
        const bool two = info[1] != 0;
        const std::shared_ptr<A32::Coprocessor> coproc = conf.coprocessors[coproc_num];
        ASSERT_MSG(coproc, "Access to unimplemented coprocessor {}", coproc_num);

        const auto call = [](const A32::Coprocessor::Callback& cb, u32 arg0, u32 arg1) {
            return cb.function(cb.user_arg.value_or(nullptr), arg0, arg1);
        };

        switch (op.opcode) {
        case Opcode::A32CoprocInternalOperation: {
            const auto action = coproc->CompileInternalOperation(two, info[2], A32::CoprocReg(info[3]), A32::CoprocReg(info[4]), A32::CoprocReg(info[5]), info[6]);
            ASSERT_MSG(action, "Unimplemented coprocessor operation");
            call(*action, 0, 0);
            break;
        }
        case Opcode::A32CoprocSendOneWord: {
            const auto action = coproc->CompileSendOneWord(two, info[2], A32::CoprocReg(info[3]), A32::CoprocReg(info[4]), info[5]);
            const u32 word = u32(ARG(1).lo);
            if (const auto cb = std::get_if<A32::Coprocessor::Callback>(&action)) {
                call(*cb, word, 0);
            } else if (const auto ptr = std::get_if<u32*>(&action)) {
                **ptr = word;
            } else {
                UNREACHABLE_MSG("Unimplemented coprocessor write");
            }
            break;
        }
        case Opcode::A32CoprocSendTwoWords: {
            const auto action = coproc->CompileSendTwoWords(two, info[2], A32::CoprocReg(info[3]));
            const u32 word1 = u32(ARG(1).lo), word2 = u32(ARG(2).lo);
            if (const auto cb = std::get_if<A32::Coprocessor::Callback>(&action)) {
                call(*cb, word1, word2);
            } else if (const auto ptrs = std::get_if<std::array<u32*, 2>>(&action)) {
                *(*ptrs)[0] = word1;
                *(*ptrs)[1] = word2;
            } else {
                UNREACHABLE_MSG("Unimplemented coprocessor write");
            }
            break;
        }
        case Opcode::A32CoprocGetOneWord: {
            const auto action = coproc->CompileGetOneWord(two, info[2], A32::CoprocReg(info[3]), A32::CoprocReg(info[4]), info[5]);
            if (const auto cb = std::get_if<A32::Coprocessor::Callback>(&action)) {
                DST.lo = u32(call(*cb, 0, 0));
            } else if (const auto ptr = std::get_if<u32*>(&action)) {
                DST.lo = **ptr;
            } else {
                UNREACHABLE_MSG("Unimplemented coprocessor read");
            }
            break;
        }
        case Opcode::A32CoprocGetTwoWords: {
            const auto action = coproc->CompileGetTwoWords(two, info[2], A32::CoprocReg(info[3]));
            if (const auto cb = std::get_if<A32::Coprocessor::Callback>(&action)) {
                DST.lo = call(*cb, 0, 0);
            } else if (const auto ptrs = std::get_if<std::array<u32*, 2>>(&action)) {
                DST.lo = u64(*(*ptrs)[0]) | (u64(*(*ptrs)[1]) << 32);
            } else {
                UNREACHABLE_MSG("Unimplemented coprocessor read");
            }
            break;
        }
        case Opcode::A32CoprocLoadWords:
        case Opcode::A32CoprocStoreWords: {
            const bool load = op.opcode == Opcode::A32CoprocLoadWords;
            const bool long_transfer = info[2] != 0;
            const std::optional<u8> option = info[4] ? std::optional<u8>{info[5]} : std::nullopt;
            const auto action = load ? coproc->CompileLoadWords(two, long_transfer, A32::CoprocReg(info[3]), option)
                                     : coproc->CompileStoreWords(two, long_transfer, A32::CoprocReg(info[3]), option);
            ASSERT_MSG(action, "Unimplemented coprocessor transfer");
            call(*action, u32(ARG(1).lo), 0);
            break;
        }
        default:
            UNREACHABLE();
        }
    }

    void UpdateTicks() {
        if (conf.enable_cycle_counting) {
            conf.callbacks->AddTicks(u64(cycles_to_run - cycles_remaining));
        }
    }

    void ResetTicks() {
        if (conf.enable_cycle_counting) {
            cycles_to_run = s64(conf.callbacks->GetTicksRemaining());
            cycles_remaining = cycles_to_run;
        }
    }

    void SetQC(bool saturated) {
        if (saturated) {
            state.fpsr_qc = 1;
        }
    }

    u32 NZCV() const {
        return state.nzcv;
    }

    void CommitFPSR(const FPEnv& env) {
        state.fpsr_exc |= env.fpsr.Value() & 0x9Fu;
    }

    FPEnv MakeFPEnv(bool fpcr_controlled) const {
        FP::FPCR fpcr{state.upper & A32State::FPSCR_MODE_MASK};
        if (!fpcr_controlled) {
            // ASIMD standard FPSCR value.
            fpcr.DN(true);
            fpcr.FZ(true);
            fpcr.RMode(FP::RoundingMode::ToNearest_TieEven);
        }
        return FPEnv{fpcr, FP::FPSR{state.fpsr_exc}, conf.HasOptimization(OptimizationFlag::Unsafe_InaccurateNaN), conf.HasOptimization(OptimizationFlag::Unsafe_UnfuseFMA)};
    }

    /// Runs the compiled region entered at `block`, building it once the block is hot.
    /// Returns false if the block must be interpreted.
    bool RunRegion(CompiledBlock& block) {
        if (!block.region) {
            if (!WasmJitEnabled() || ++block.entry_count != RegionThreshold() || !BuildRegion(block)) {
                return false;
            }
        }
        const CompiledBlockFn fn = GetThreadFunction(*block.region);
        if (!fn) {
            return false;
        }
        fn(&state, frame.data(), this, reinterpret_cast<void*>(uintptr_t{block.region_index}));
        if (SamplePC(state.reg[15])) {
            DumpBlock(state.Location());
        }
        return true;
    }

    bool BuildRegion(CompiledBlock& entry) {
        if (entry.num_slots > frame.size()) {
            return false;
        }
        std::vector<CompiledBlock*> blocks{&entry};
        std::unordered_set<u64> seen{entry.location.Value()};
        for (size_t i = 0; i < blocks.size() && blocks.size() < kMaxRegionBlocks; ++i) {
            const CompiledBlock& block = *blocks[i];
            const auto add = [&](IR::LocationDescriptor next) {
                const IR::LocationDescriptor location = A32::LocationDescriptor{next}.SetSingleStepping(false);
                if (blocks.size() < kMaxRegionBlocks && seen.insert(location.Value()).second) {
                    // Blocks already compiled elsewhere are reached through their own region.
                    if (CompiledBlock* successor = GetBlock(location); !successor->region && successor->num_slots <= frame.size()) {
                        blocks.push_back(successor);
                    }
                }
            };
            std::visit([&](const auto& term) { AddSuccessors(term, add); }, block.terminal);
        if (block.cond != IR::Cond::AL) {
            add(block.cond_failed_location);
        }
        }
        auto region = std::make_shared<WasmRegion>();
        region->id = NextRegionId();
        if (!CompileRegion(blocks, jit_target, region->wasm)) {
            return false;
        }
        region->blocks.assign(blocks.begin(), blocks.end());
        if (TrackRegionCode(*region)) {
            ClearCache();
        }
        // Blocks without a region of their own enter this one rather than compiling another
        // copy of the same code.
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (!blocks[i]->region) {
                blocks[i]->region = region;
                blocks[i]->region_index = u32(i);
            }
        }
        return true;
    }

    template<typename Add>
    static void AddLeafSuccessors(const IR::Term::LeafTerminal& leaf, const Add& add) {
        std::visit([&](const auto& term) {
            using T = std::decay_t<decltype(term)>;
            if constexpr (std::is_same_v<T, IR::Term::LinkBlock> || std::is_same_v<T, IR::Term::LinkBlockFast>) {
                add(term.next);
            }
        }, leaf);
    }

    template<typename Term, typename Add>
    static void AddSuccessors(const Term& term, const Add& add) {
        using T = std::decay_t<Term>;
        if constexpr (std::is_same_v<T, IR::Term::LeafTerminal>) {
            AddLeafSuccessors(term, add);
        } else if constexpr (std::is_same_v<T, IR::Term::If> || std::is_same_v<T, IR::Term::CheckBit>) {
            AddLeafSuccessors(term.then_, add);
            AddLeafSuccessors(term.else_, add);
        } else if constexpr (std::is_same_v<T, IR::Term::CheckHalt>) {
            AddLeafSuccessors(term.else_, add);
        }
    }

    static constexpr size_t kMaxRegionBlocks = 96;

    void ExecuteOp(const Op& op, Slot* s, CompiledBlock& block) {
        switch (op.opcode) {
#include "dynarmic/backend/interp/ops_common.inc"
#include "dynarmic/backend/interp/ops_packed.inc"
#include "dynarmic/backend/interp/ops_a32.inc"
#include "dynarmic/backend/interp/ops_vector.inc"
#include "dynarmic/backend/interp/ops_fp.inc"
        default:
            UNREACHABLE_MSG("Interpreter: unimplemented IR opcode {}", IR::GetNameOf(op.opcode));
        }
    }

    /// Fallback used by compiled blocks for operations that are not emitted inline.
    static void ExecuteOpHelper(void* impl, void* block, u32 index, u32 end) {
        auto& self = *static_cast<Impl*>(impl);
        auto& compiled = *static_cast<CompiledBlock*>(block);
        for (; index < end; ++index) {
            CountHelperOp(compiled.ops[index].opcode);
            self.ExecuteOp(compiled.ops[index], self.frame.data(), compiled);
        }
    }

    JitTarget MakeJitTarget() const {
        JitTarget target;
        target.state_address = reinterpret_cast<u64>(&state);
        target.frame_address = reinterpret_cast<u64>(frame.data());
        target.a32 = true;
        target.reg_offset = offsetof(A32State, reg);
        target.nzcv_offset = offsetof(A32State, nzcv);
        target.check_bit_offset = offsetof(A32State, check_bit);
        target.upper_offset = offsetof(A32State, upper);
        target.q_offset = offsetof(A32State, q);
        target.ge_offset = offsetof(A32State, ge);
        target.ext_offset = offsetof(A32State, ext);
        if (conf.page_table) {
            target.page_table = reinterpret_cast<u64>(conf.page_table->data());
        }
        target.page_table_pointer_mask = conf.page_table_pointer_mask;
        target.pointer_sign_shift = pointer_sign_shift;
        target.page_table_marked_bit = conf.page_table_marked_bit;
        target.absolute_offset_page_table = conf.absolute_offset_page_table;
        target.helper = &ExecuteOpHelper;
        target.inaccurate_nan = conf.HasOptimization(OptimizationFlag::Unsafe_InaccurateNaN);
        target.unfuse_fma = conf.HasOptimization(OptimizationFlag::Unsafe_UnfuseFMA);
        target.ignore_standard_fpcr = conf.HasOptimization(OptimizationFlag::Unsafe_IgnoreStandardFPCRValue);
        target.local_exclusive_monitor = LocalMonitor();
        target.fpsr_nzcv_offset = offsetof(A32State, fpsr_nzcv);
        target.exclusive_state_offset = offsetof(A32State, exclusive_state);
        target.exclusive_address_offset = offsetof(A32State, exclusive_address);
        target.exclusive_value_offset = offsetof(A32State, exclusive_value);
        target.coproc_read_pointer = [this](u64 info_bits) -> u64 {
            std::array<u8, 8> info;
            std::memcpy(info.data(), &info_bits, sizeof(info));
            const auto& coproc = conf.coprocessors[info[0]];
            if (!coproc) {
                return 0;
            }
            const auto action = coproc->CompileGetOneWord(info[1] != 0, info[2], A32::CoprocReg(info[3]), A32::CoprocReg(info[4]), info[5]);
            if (const auto ptr = std::get_if<u32*>(&action)) {
                return reinterpret_cast<u64>(*ptr);
            }
            return 0;
        };
        target.cycle_counting = conf.enable_cycle_counting;
        target.cycles_remaining = reinterpret_cast<u64>(&cycles_remaining);
        target.halt_reason = reinterpret_cast<u64>(&state.halt_reason);
        return target;
    }

    CompiledBlock* Execute(CompiledBlock& block, bool single_step) {
        if (block.cond != IR::Cond::AL && !CondHolds(block.cond, state.nzcv)) {
            cycles_remaining -= s64(block.cond_failed_cycle_count);
            return Link(block, block.cond_failed_location, single_step, false);
        }

        Slot* const s = block.num_slots <= frame.size() ? frame.data() : overflow_frame.data();
        if (!block.constants.empty()) {
            std::memcpy(s + block.first_constant, block.constants.data(), block.constants.size() * sizeof(Slot));
        }

        wrote_pc = false;
        for (const Op& op : block.ops) {
            ExecuteOp(op, s, block);
        }

        cycles_remaining -= s64(block.cycle_count);
        if (SamplePC(block.start_address)) {
            DumpBlock(block.location);
        }
        return Terminal(block, block.terminal, single_step);
    }

    CompiledBlock* Terminal(CompiledBlock& block, const IR::Term::Terminal& terminal, bool single_step) {
        using namespace IR::Term;
        return std::visit([&](const auto& term) -> CompiledBlock* {
            using T = std::decay_t<decltype(term)>;
            if constexpr (std::is_same_v<T, LeafTerminal>) {
                return LeafTerminalImpl(block, term, single_step, true);
            } else if constexpr (std::is_same_v<T, If>) {
                const bool taken = CondHolds(term.if_, state.nzcv);
                return LeafTerminalImpl(block, taken ? term.then_ : term.else_, single_step, taken);
            } else if constexpr (std::is_same_v<T, CheckBit>) {
                const bool taken = state.check_bit;
                return LeafTerminalImpl(block, taken ? term.then_ : term.else_, single_step, taken);
            } else if constexpr (std::is_same_v<T, CheckHalt>) {
                if (Atomic::Load(&state.halt_reason) != 0) {
                    return nullptr;
                }
                return LeafTerminalImpl(block, term.else_, single_step, false);
            } else {
                return nullptr;
            }
        }, terminal);
    }

    CompiledBlock* LeafTerminalImpl(CompiledBlock& block, const IR::Term::LeafTerminal& terminal, bool single_step, bool then_branch) {
        using namespace IR::Term;
        return std::visit([&](const auto& term) -> CompiledBlock* {
            using T = std::decay_t<decltype(term)>;
            if constexpr (std::is_same_v<T, LinkBlock> || std::is_same_v<T, LinkBlockFast>) {
                return Link(block, term.next, single_step, then_branch);
            } else {
                return nullptr;
            }
        }, terminal);
    }

    CompiledBlock* Link(CompiledBlock& block, IR::LocationDescriptor next, bool single_step, bool then_branch) {
        const A32::LocationDescriptor target = A32::LocationDescriptor{next}.SetSingleStepping(false);
        state.reg[15] = target.PC();
        state.upper = u32(target.UniqueHash() >> 32);
        if (single_step) {
            return nullptr;
        }
        const IR::LocationDescriptor location = target;
        CompiledBlock*& link = then_branch ? block.link_then : block.link_else;
        if (!link || link->location != location) {
            link = GetBlock(location);
        }
        return link;
    }

    const UserConfig conf;
    Jit* jit_interface;
    A32State state;
    IR::Block ir_block = {LocationDescriptor(0, PSR(0), FPSCR(0), false)};
    Optimization::PolyfillOptions polyfill_options{.sha256 = true, .vector_multiply_widen = true};
    std::unordered_map<u64, std::unique_ptr<CompiledBlock>> block_cache;
    struct FastDispatchEntry {
        u64 location = 0;
        CompiledBlock* block = nullptr;
    };
    std::array<FastDispatchEntry, 4096> fast_dispatch{};
    Backend::BlockRangeInformation<u32> block_ranges;
    std::vector<Slot> frame;
    std::vector<Slot> overflow_frame;
    s64 cycles_to_run = 0;
    s64 cycles_remaining = 0;
    int pointer_sign_shift = 0;
    JitTarget jit_target;
    bool wrote_pc = false;
    bool invalidate_entire_cache = false;
    boost::icl::interval_set<u32> invalid_cache_ranges;
    std::mutex invalidation_mutex;
};

Jit::Jit(UserConfig conf)
        : impl(std::make_unique<Impl>(this, std::move(conf))) {}

Jit::~Jit() = default;

HaltReason Jit::Run() {
    return impl->Run();
}

HaltReason Jit::Step() {
    return impl->Step();
}

void Jit::ClearCache() {
    impl->ClearCache();
}

void Jit::InvalidateCacheRange(std::uint32_t start_address, std::size_t length) {
    impl->InvalidateCacheRange(start_address, length);
}

void Jit::Reset() {
    impl->Reset();
}

void Jit::HaltExecution(HaltReason hr) {
    impl->HaltExecution(hr);
}

void Jit::ClearHalt(HaltReason hr) {
    impl->ClearHalt(hr);
}

std::array<std::uint32_t, 16>& Jit::Regs() {
    return impl->Regs();
}

const std::array<std::uint32_t, 16>& Jit::Regs() const {
    return impl->Regs();
}

std::array<std::uint32_t, 64>& Jit::ExtRegs() {
    return impl->ExtRegs();
}

const std::array<std::uint32_t, 64>& Jit::ExtRegs() const {
    return impl->ExtRegs();
}

std::uint32_t Jit::Cpsr() const {
    return impl->Cpsr();
}

void Jit::SetCpsr(std::uint32_t value) {
    impl->SetCpsr(value);
}

std::uint32_t Jit::Fpscr() const {
    return impl->Fpscr();
}

void Jit::SetFpscr(std::uint32_t value) {
    impl->SetFpscr(value);
}

void Jit::ClearExclusiveState() {
    impl->ClearExclusiveState();
}

std::string Jit::Disassemble() const {
    return {};
}

}  // namespace Dynarmic::A32
