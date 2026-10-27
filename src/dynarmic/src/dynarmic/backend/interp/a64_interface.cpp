// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Portable A64 backend that interprets optimized IR blocks. Used on hosts without a code
// generator, such as WebAssembly.

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
#include "common/logging.h"
#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/backend/interp/interp.h"
#include "dynarmic/backend/interp/interp_exec.h"
#include "dynarmic/backend/interp/interp_fp.h"
#include "dynarmic/backend/interp/interp_vector.h"
#include "dynarmic/backend/interp/sm4_sbox.h"
#include "dynarmic/backend/interp/wasm_jit.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/common/crypto/crc32.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/interface/optimization_flags.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opt_passes.h"

namespace Dynarmic::A64 {

using namespace Backend::Interp;
using IR::Opcode;


namespace {

struct A64State {
    static constexpr u32 FPCR_MASK = 0x07C89F00;
    static constexpr u32 FPSR_MASK = 0x0800009F;

    std::array<u64, 31> reg{};
    u64 sp = 0;
    u64 pc = 0;
    alignas(16) std::array<u64, 64> vec{};
    u32 nzcv = 0;
    u32 fpcr = 0;
    u32 fpsr = 0;
    volatile u32 halt_reason = 0;
    u32 exclusive_state = 0;
    bool check_bit = false;
    /// Reservation of the local exclusive monitor (Unsafe_IgnoreGlobalMonitor).
    u64 exclusive_address = 0;
    u64 exclusive_value = 0;
    /// Dispatch table of the thread running the Jit, read by compiled regions.
    DispatchEntry* dispatch_table = nullptr;

    IR::LocationDescriptor Location() const {
        return A64::LocationDescriptor{pc, FP::FPCR{fpcr}, false};
    }
};

}  // Anonymous namespace

struct Jit::Impl final {
    Impl(Jit*, UserConfig conf_)
        : conf(conf_) {
        ASSERT(conf.page_table_address_space_bits >= 12 && conf.page_table_address_space_bits <= 64);
        if (conf.page_table_pointer_mask != 0) {
            pointer_sign_shift = std::countl_zero(conf.page_table_pointer_mask);
        }
        frame.resize(kFrameSlots);
        jit_target = MakeJitTarget();
    }

    HaltReason Run() {
        ASSERT(!is_executing);
        PerformRequestedCacheInvalidation(HaltReason(Atomic::Load(&state.halt_reason)));
        is_executing = true;
        state.dispatch_table = dispatch_tables.ForThisThread();

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
        is_executing = false;
        return hr;
    }

    HaltReason Step() {
        ASSERT(!is_executing);
        PerformRequestedCacheInvalidation(HaltReason(Atomic::Load(&state.halt_reason)));
        is_executing = true;

        cycles_to_run = 1;
        cycles_remaining = 1;
        if (Atomic::Load(&state.halt_reason) == 0) {
            Atomic::Or(&state.halt_reason, u32(HaltReason::Step));
            CompiledBlock* block = GetBlock(A64::LocationDescriptor{state.Location()}.SetSingleStepping(true));
            Execute(*block, true);
        }

        RecordExecutedCycles(u64(cycles_to_run - cycles_remaining), conf.processor_id, 0);
        if (conf.enable_cycle_counting) {
            conf.callbacks->AddTicks(u64(cycles_to_run - cycles_remaining));
        }
        const HaltReason hr = HaltReason(Atomic::Exchange(&state.halt_reason, 0));
        PerformRequestedCacheInvalidation(hr);
        is_executing = false;
        return hr;
    }

    void ClearCache() {
        std::unique_lock lock{invalidation_mutex};
        invalidate_entire_cache = true;
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void InvalidateCacheRange(u64 start_address, size_t length) {
        std::unique_lock lock{invalidation_mutex};
        const auto end_address = u64(start_address + length - 1);
        invalid_cache_ranges.add(boost::icl::discrete_interval<u64>::closed(start_address, end_address));
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void Reset() {
        ASSERT(!is_executing);
        state = {};
    }

    void HaltExecution(HaltReason hr) {
        Atomic::Or(&state.halt_reason, u32(hr));
    }

    void ClearHalt(HaltReason hr) {
        Atomic::And(&state.halt_reason, ~u32(hr));
    }

    u64 GetSP() const {
        return state.sp;
    }
    void SetSP(u64 value) {
        state.sp = value;
    }
    u64 GetPC() const {
        return state.pc;
    }
    void SetPC(u64 value) {
        state.pc = value;
    }
    u64 GetRegister(size_t index) const {
        return index == 31 ? GetSP() : state.reg.at(index);
    }
    void SetRegister(size_t index, u64 value) {
        if (index == 31) {
            return SetSP(value);
        }
        state.reg.at(index) = value;
    }
    std::array<u64, 31> GetRegisters() const {
        return state.reg;
    }
    void SetRegisters(const std::array<u64, 31>& value) {
        state.reg = value;
    }
    Vector GetVector(size_t index) const {
        return {state.vec.at(index * 2), state.vec.at(index * 2 + 1)};
    }
    void SetVector(size_t index, Vector value) {
        state.vec.at(index * 2) = value[0];
        state.vec.at(index * 2 + 1) = value[1];
    }
    std::array<Vector, 32> GetVectors() const {
        std::array<Vector, 32> ret;
        std::memcpy(ret.data(), state.vec.data(), sizeof(state.vec));
        return ret;
    }
    void SetVectors(const std::array<Vector, 32>& value) {
        std::memcpy(state.vec.data(), value.data(), sizeof(state.vec));
    }
    u32 GetFpcr() const {
        return state.fpcr;
    }
    void SetFpcr(u32 value) {
        state.fpcr = value & A64State::FPCR_MASK;
    }
    u32 GetFpsr() const {
        return state.fpsr;
    }
    void SetFpsr(u32 value) {
        state.fpsr = value & A64State::FPSR_MASK;
    }
    u32 GetPstate() const {
        return state.nzcv;
    }
    void SetPstate(u32 value) {
        state.nzcv = value & 0xF0000000;
    }
    void ClearExclusiveState() {
        state.exclusive_state = 0;
    }
    bool IsExecuting() const {
        return is_executing;
    }

private:
    CompiledBlock* GetBlock(IR::LocationDescriptor descriptor) {
        // Most lookups hit a small direct-mapped cache in front of the block map.
        const u64 key = descriptor.Value();
        FastDispatchEntry& entry = fast_dispatch[((key ^ (key >> 32)) >> 2) % fast_dispatch.size()];
        if (entry.block && entry.location == key) {
            return entry.block;
        }
        if (const auto it = block_cache.find(key); it != block_cache.end()) {
            entry = {key, it->second.get()};
            return it->second.get();
        }

        const auto get_code = [this](u64 vaddr) { return conf.callbacks->MemoryReadCode(vaddr); };
        const auto arch_descriptor = A64::LocationDescriptor{descriptor};
        ir_block.Reset(arch_descriptor);
        A64::Translate(ir_block, arch_descriptor, get_code, {conf.define_unpredictable_behaviour, conf.wall_clock_cntpct});
        Optimization::Optimize(ir_block, conf, polyfill_options);
        if (ShouldDumpBlock(arch_descriptor.PC())) {
            std::fprintf(stderr, "%s\n", IR::DumpBlock(ir_block).c_str());
        }

        auto block = std::make_unique<CompiledBlock>();
        Compile(*block, ir_block);
        block->start_address = arch_descriptor.PC();
        block->end_address = A64::LocationDescriptor{ir_block.EndLocation()}.PC();
        if (block->end_address > block->start_address) {
            block_ranges.AddRange(boost::icl::discrete_interval<u64>::closed(block->start_address, block->end_address - 1), descriptor);
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
        const auto arch_descriptor = A64::LocationDescriptor{descriptor};
        auto dump = std::make_unique<IR::Block>(arch_descriptor);
        IR::Block& dump_block = *dump;
        A64::Translate(dump_block, arch_descriptor, [this](u64 vaddr) { return conf.callbacks->MemoryReadCode(vaddr); }, {conf.define_unpredictable_behaviour, conf.wall_clock_cntpct});
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
        dispatch_tables.Clear();
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

    /// Returns a host pointer for an access that does not cross a page, or nullptr when the
    /// access must go through the callbacks (unmapped, rasterizer-cached or page-crossing).
    template<size_t bytes>
    u8* HostPointer(u64 vaddr) const {
        if (!conf.page_table) {
            return nullptr;
        }
        if ((vaddr & 0xFFF) + bytes > 0x1000) {
            return nullptr;
        }
        const u64 index = vaddr >> 12;
        if (conf.page_table_address_space_bits < 64 && (index >> (conf.page_table_address_space_bits - 12)) != 0) {
            if (!conf.silently_mirror_page_table) {
                return nullptr;
            }
        }
        const u64 masked_index = conf.page_table_address_space_bits < 64 ? index & ((u64(1) << (conf.page_table_address_space_bits - 12)) - 1) : index;
        u64 entry;
        __atomic_load(reinterpret_cast<u64*>(conf.page_table) + (masked_index << (conf.page_table_log2_stride - 3)), &entry, __ATOMIC_RELAXED);
        if (conf.page_table_marked_bit && ((entry >> *conf.page_table_marked_bit) & 1)) {
            return nullptr;
        }
        if (conf.page_table_pointer_mask != 0) {
            entry &= conf.page_table_pointer_mask;
            // Packed entries may encode a negative offset; the mask's top bit is its sign.
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
    T Read(u64 vaddr) {
        if (u8* ptr = HostPointer<sizeof(T)>(vaddr)) {
            T value;
            std::memcpy(&value, ptr, sizeof(T));
            return value;
        }
        const T value = T(conf.callbacks->MemoryRead(vaddr, sizeof(T)));
        CheckHaltAfterMemoryCallback();
        return value;
    }

    Vector Read128(u64 vaddr) {
        if (u8* ptr = HostPointer<16>(vaddr)) {
            Vector value;
            std::memcpy(value.data(), ptr, 16);
            return value;
        }
        const Vector value = conf.callbacks->MemoryRead128(vaddr);
        CheckHaltAfterMemoryCallback();
        return value;
    }

    template<typename T>
    void Write(u64 vaddr, T value) {
        if (u8* ptr = HostPointer<sizeof(T)>(vaddr)) {
            std::memcpy(ptr, &value, sizeof(T));
            return;
        }
        conf.callbacks->MemoryWrite(vaddr, u64(value), sizeof(T));
        CheckHaltAfterMemoryCallback();
    }

    void Write128(u64 vaddr, Vector value) {
        if (u8* ptr = HostPointer<16>(vaddr)) {
            std::memcpy(ptr, value.data(), 16);
            return;
        }
        conf.callbacks->MemoryWrite128(vaddr, value);
        CheckHaltAfterMemoryCallback();
    }

    /// With Unsafe_IgnoreGlobalMonitor each core keeps its own reservation and the store is a
    /// compare-and-swap against the value that was loaded, as in the native backends.
    bool LocalMonitor() const {
        return conf.HasOptimization(OptimizationFlag::Unsafe_IgnoreGlobalMonitor);
    }

    template<typename T>
    T ExclusiveRead(u64 vaddr) {
        state.exclusive_state = 1;
        if constexpr (!std::is_same_v<T, Vector>) {
            if (LocalMonitor()) {
                const T value = Read<T>(vaddr);
                state.exclusive_address = vaddr;
                state.exclusive_value = u64(value);
                return value;
            }
        }
        return conf.global_monitor->ReadAndMark<T>(conf.processor_id, vaddr, [&]() -> T {
            if constexpr (std::is_same_v<T, Vector>) {
                return conf.callbacks->MemoryRead128(vaddr);
            } else {
                return T(conf.callbacks->MemoryRead(vaddr, sizeof(T)));
            }
        });
    }

    template<typename T>
    u32 ExclusiveWrite(u64 vaddr, T value) {
        if (!state.exclusive_state) {
            return 1;
        }
        state.exclusive_state = 0;
        if constexpr (!std::is_same_v<T, Vector>) {
            if (LocalMonitor()) {
                if (state.exclusive_address != vaddr) {
                    return 1;
                }
                T expected = T(state.exclusive_value);
                if (u8* ptr = HostPointer<sizeof(T)>(vaddr)) {
                    return __atomic_compare_exchange_n(reinterpret_cast<T*>(ptr), &expected, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) ? 0 : 1;
                }
                if constexpr (sizeof(T) == 1) {
                    return conf.callbacks->MemoryWriteExclusive8(vaddr, value, expected) ? 0 : 1;
                } else if constexpr (sizeof(T) == 2) {
                    return conf.callbacks->MemoryWriteExclusive16(vaddr, value, expected) ? 0 : 1;
                } else if constexpr (sizeof(T) == 4) {
                    return conf.callbacks->MemoryWriteExclusive32(vaddr, value, expected) ? 0 : 1;
                } else {
                    return conf.callbacks->MemoryWriteExclusive64(vaddr, value, expected) ? 0 : 1;
                }
            }
        }
        const bool ok = conf.global_monitor->DoExclusiveOperation<T>(conf.processor_id, vaddr, [&](T expected) -> bool {
            if constexpr (std::is_same_v<T, Vector>) {
                return conf.callbacks->MemoryWriteExclusive128(vaddr, value, expected);
            } else if constexpr (sizeof(T) == 1) {
                return conf.callbacks->MemoryWriteExclusive8(vaddr, value, expected);
            } else if constexpr (sizeof(T) == 2) {
                return conf.callbacks->MemoryWriteExclusive16(vaddr, value, expected);
            } else if constexpr (sizeof(T) == 4) {
                return conf.callbacks->MemoryWriteExclusive32(vaddr, value, expected);
            } else {
                return conf.callbacks->MemoryWriteExclusive64(vaddr, value, expected);
            }
        });
        return ok ? 0 : 1;
    }

    void CheckHaltAfterMemoryCallback() {
        if (conf.check_halt_on_memory_access) {
            CheckHaltAfterCallback();
        }
    }

    void CheckHaltAfterCallback() {
        // Remaining operations still run; the dispatcher stops before the next block.
    }

    void UpdateTicks() {
        if (!conf.enable_cycle_counting) {
            return;
        }
        conf.callbacks->AddTicks(u64(cycles_to_run - cycles_remaining));
        cycles_to_run = s64(conf.callbacks->GetTicksRemaining());
        cycles_remaining = cycles_to_run;
    }

    void SetQC(bool saturated) {
        if (saturated) {
            state.fpsr |= 1u << 27;
        }
    }

    u32 NZCV() const {
        return state.nzcv;
    }

    void CommitFPSR(const FPEnv& env) {
        state.fpsr = (state.fpsr & ~0x9Fu) | (env.fpsr.Value() & 0x9Fu);
    }

    FPEnv MakeFPEnv(bool fpcr_controlled) const {
        FP::FPCR fpcr{state.fpcr};
        if (!fpcr_controlled) {
            // ASIMD standard FPSCR value.
            fpcr.DN(true);
            fpcr.FZ(true);
            fpcr.RMode(FP::RoundingMode::ToNearest_TieEven);
        }
        return FPEnv{fpcr, FP::FPSR{state.fpsr}, conf.HasOptimization(OptimizationFlag::Unsafe_InaccurateNaN), conf.HasOptimization(OptimizationFlag::Unsafe_UnfuseFMA)};
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
        state.dispatch_table[DispatchTables::IndexOf(block.location.Value())] = {block.location.Value(), reinterpret_cast<uintptr_t>(fn), block.region_index, 0};
        fn(&state, frame.data(), this, reinterpret_cast<void*>(uintptr_t{block.region_index}));
        if (SamplePC(state.pc)) {
            DumpBlock(state.Location());
        }
        return true;
    }

    bool BuildRegion(CompiledBlock& entry) {
        if (!RegionAllowedAt(A64::LocationDescriptor{entry.location}.PC())) {
            return false;
        }
        if (entry.num_slots > frame.size()) {
            return false;
        }
        std::vector<CompiledBlock*> blocks{&entry};
        std::unordered_set<u64> seen{entry.location.Value()};
        for (size_t i = 0; i < blocks.size() && blocks.size() < kMaxRegionBlocks; ++i) {
            const CompiledBlock& block = *blocks[i];
            const auto add = [&](IR::LocationDescriptor next) {
                const IR::LocationDescriptor location = A64::LocationDescriptor{next}.SetSingleStepping(false);
                if (blocks.size() < kMaxRegionBlocks && seen.insert(location.Value()).second) {
                    // Blocks already compiled elsewhere are reached through their own region.
                    if (CompiledBlock* successor = GetBlock(location); !successor->region && successor->num_slots <= frame.size()) {
                        blocks.push_back(successor);
                    }
                }
            };
            std::visit([&](const auto& term) { AddSuccessors(term, add); }, block.terminal);
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
#include "dynarmic/backend/interp/ops_a64.inc"
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
        target.a32 = false;
        target.reg_offset = offsetof(A64State, reg);
        target.sp_offset = offsetof(A64State, sp);
        target.pc_offset = offsetof(A64State, pc);
        target.nzcv_offset = offsetof(A64State, nzcv);
        target.check_bit_offset = offsetof(A64State, check_bit);
        target.ext_offset = offsetof(A64State, vec);
        target.page_table = reinterpret_cast<u64>(conf.page_table);
        target.page_table_pointer_mask = conf.page_table_pointer_mask;
        target.pointer_sign_shift = pointer_sign_shift;
        target.page_table_marked_bit = conf.page_table_marked_bit;
        target.absolute_offset_page_table = conf.absolute_offset_page_table;
        target.page_table_address_space_bits = conf.page_table_address_space_bits;
        target.silently_mirror_page_table = conf.silently_mirror_page_table;
        target.tpidr_el0 = reinterpret_cast<u64>(conf.tpidr_el0);
        target.tpidrro_el0 = reinterpret_cast<u64>(conf.tpidrro_el0);
        target.helper = &ExecuteOpHelper;
        target.inaccurate_nan = conf.HasOptimization(OptimizationFlag::Unsafe_InaccurateNaN);
        target.unfuse_fma = conf.HasOptimization(OptimizationFlag::Unsafe_UnfuseFMA);
        target.ignore_standard_fpcr = conf.HasOptimization(OptimizationFlag::Unsafe_IgnoreStandardFPCRValue);
        target.cycle_counting = conf.enable_cycle_counting;
        target.chain_regions = true;
        target.fpcr_offset = offsetof(A64State, fpcr);
        target.dispatch_table_offset = offsetof(A64State, dispatch_table);
        target.local_exclusive_monitor = LocalMonitor();
        target.exclusive_state_offset = offsetof(A64State, exclusive_state);
        target.exclusive_address_offset = offsetof(A64State, exclusive_address);
        target.exclusive_value_offset = offsetof(A64State, exclusive_value);
        target.cycles_remaining = reinterpret_cast<u64>(&cycles_remaining);
        target.halt_reason = reinterpret_cast<u64>(&state.halt_reason);
        if (conf.page_table_log2_stride != 3 || conf.check_halt_on_memory_access) {
            // The compiled memory path assumes a dense u64 table and no halt checks.
            target.page_table = 0;
        }
        return target;
    }

    /// Runs a block and returns the next block to run when it can be chained directly.
    CompiledBlock* Execute(CompiledBlock& block, bool single_step) {
        Slot* const s = block.num_slots <= frame.size() ? frame.data() : overflow_frame.data();
        if (!block.constants.empty()) {
            std::memcpy(s + block.first_constant, block.constants.data(), block.constants.size() * sizeof(Slot));
        }

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
                state.pc = A64::LocationDescriptor{term.next}.PC();
                if (single_step) {
                    return nullptr;
                }
                CompiledBlock*& link = then_branch ? block.link_then : block.link_else;
                if (!link || link->location != term.next) {
                    link = GetBlock(term.next);
                }
                return link;
            } else {
                // ReturnToDispatch, PopRSBHint and FastDispatchHint: PC is already up to date.
                return nullptr;
            }
        }, terminal);
    }

    const UserConfig conf;
    A64State state;
    IR::Block ir_block = {LocationDescriptor(0, FP::FPCR(0), false)};
    Optimization::PolyfillOptions polyfill_options{.sha256 = true, .vector_multiply_widen = true};
    std::unordered_map<u64, std::unique_ptr<CompiledBlock>> block_cache;
    struct FastDispatchEntry {
        u64 location = 0;
        CompiledBlock* block = nullptr;
    };
    std::array<FastDispatchEntry, 4096> fast_dispatch{};
    DispatchTables dispatch_tables;
    Backend::BlockRangeInformation<u64> block_ranges;
    std::vector<Slot> frame;
    std::vector<Slot> overflow_frame;
    s64 cycles_to_run = 0;
    s64 cycles_remaining = 0;
    int pointer_sign_shift = 0;
    JitTarget jit_target;
    bool is_executing = false;
    bool invalidate_entire_cache = false;
    boost::icl::interval_set<u64> invalid_cache_ranges;
    std::mutex invalidation_mutex;
};

Jit::Jit(UserConfig conf)
        : impl(std::make_unique<Jit::Impl>(this, conf)) {}

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

void Jit::InvalidateCacheRange(u64 start_address, size_t length) {
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

u64 Jit::GetSP() const {
    return impl->GetSP();
}

void Jit::SetSP(u64 value) {
    impl->SetSP(value);
}

u64 Jit::GetPC() const {
    return impl->GetPC();
}

void Jit::SetPC(u64 value) {
    impl->SetPC(value);
}

u64 Jit::GetRegister(size_t index) const {
    return impl->GetRegister(index);
}

void Jit::SetRegister(size_t index, u64 value) {
    impl->SetRegister(index, value);
}

std::array<u64, 31> Jit::GetRegisters() const {
    return impl->GetRegisters();
}

void Jit::SetRegisters(const std::array<u64, 31>& value) {
    impl->SetRegisters(value);
}

Vector Jit::GetVector(size_t index) const {
    return impl->GetVector(index);
}

void Jit::SetVector(size_t index, Vector value) {
    impl->SetVector(index, value);
}

std::array<Vector, 32> Jit::GetVectors() const {
    return impl->GetVectors();
}

void Jit::SetVectors(const std::array<Vector, 32>& value) {
    impl->SetVectors(value);
}

u32 Jit::GetFpcr() const {
    return impl->GetFpcr();
}

void Jit::SetFpcr(u32 value) {
    impl->SetFpcr(value);
}

u32 Jit::GetFpsr() const {
    return impl->GetFpsr();
}

void Jit::SetFpsr(u32 value) {
    impl->SetFpsr(value);
}

u32 Jit::GetPstate() const {
    return impl->GetPstate();
}

void Jit::SetPstate(u32 value) {
    impl->SetPstate(value);
}

void Jit::ClearExclusiveState() {
    impl->ClearExclusiveState();
}

bool Jit::IsExecuting() const {
    return impl->IsExecuting();
}

std::string Jit::Disassemble() const {
    return {};
}

}  // namespace Dynarmic::A64
