#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dynarmic/backend/interp/wasm_jit.h"

using namespace Dynarmic;
using namespace Dynarmic::Backend::Interp;

namespace {

struct HelperCalls {
    std::vector<std::array<u32, 2>> ranges;
};

void RecordHelperRange(void* impl, void*, u32 first, u32 end) {
    static_cast<HelperCalls*>(impl)->ranges.push_back({first, end});
}

} // Anonymous namespace

TEST_CASE("Wasm JIT batches consecutive fallbacks but not inline operations", "[interp][wasm]") {
    CompiledBlock block;
    block.ops = {
        {IR::Opcode::FPAdd32, 32, 0, {5, 5}},
        {IR::Opcode::FPMul32, 32, 1, {0, 5}},
        {IR::Opcode::Void, 0, 2, {}},
        {IR::Opcode::FPSub32, 32, 3, {1, 5}},
        {IR::Opcode::FPAdd32, 32, 4, {3, 5}},
    };
    block.first_constant = 5;
    block.num_slots = 6;
    block.constants.resize(1);
    block.terminal = IR::Term::LeafTerminal{IR::Term::ReturnToDispatch{}};

    JitTarget target;
    target.helper = &RecordHelperRange;
    WasmRegion region;
    region.id = 0xffffffffffffffffULL;
    REQUIRE(CompileRegion({&block}, target, region.wasm));
    const CompiledBlockFn function = GetThreadFunction(region);
    REQUIRE(function != nullptr);

    HelperCalls calls;
    std::array<Slot, 6> frame{};
    function(nullptr, frame.data(), &calls, nullptr);

    const std::vector<std::array<u32, 2>> expected{{0, 2}, {3, 5}};
    REQUIRE(calls.ranges == expected);
}

TEST_CASE("Wasm JIT emits zero-test pseudo operations inline", "[interp][wasm]") {
    u64 region_id = 0xfffffffffffffffeULL;
    for (const u64 value : {u64{0}, u64{1}, u64{1} << 32}) {
        CompiledBlock block;
        block.ops = {
            {IR::Opcode::Or64, 64, 0, {4, 5}},
            {IR::Opcode::IsZero32, 1, 1, {0}},
            {IR::Opcode::IsZero64, 1, 2, {0}},
            {IR::Opcode::FPAdd32, 32, 3, {1, 2}},
        };
        block.first_constant = 4;
        block.num_slots = 6;
        block.constants.resize(2);
        block.constants[0].lo = value;
        block.terminal = IR::Term::LeafTerminal{IR::Term::ReturnToDispatch{}};

        JitTarget target;
        target.helper = &RecordHelperRange;
        WasmRegion region;
        region.id = region_id--;
        REQUIRE(CompileRegion({&block}, target, region.wasm));
        const CompiledBlockFn function = GetThreadFunction(region);
        REQUIRE(function != nullptr);

        HelperCalls calls;
        std::array<Slot, 6> frame{};
        function(nullptr, frame.data(), &calls, nullptr);

        REQUIRE(frame[1].lo == u64(u32(value) == 0));
        REQUIRE(frame[2].lo == u64(value == 0));
        const std::vector<std::array<u32, 2>> expected{{3, 4}};
        REQUIRE(calls.ranges == expected);
    }
}
