// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// AArch32 packed (SIMD within a register) operations. GE flags are produced in the expanded
// per-lane mask form and read back through GetGEFromOp.

case Opcode::GetGEFromOp:
    DST.lo = ARG(0).upper_lo;
    break;

#define PACKED_ADDSUB(name, T, Wide, op_expr, ge_expr)                         \
    case Opcode::name: {                                                       \
        constexpr size_t lanes = 4 / sizeof(T);                                \
        constexpr u32 lane_mask = sizeof(T) == 1 ? 0xFF : 0xFFFF;              \
        const u32 a = u32(ARG(0).lo), b = u32(ARG(1).lo);                      \
        u32 result = 0, ge = 0;                                                \
        for (size_t i = 0; i < lanes; ++i) {                                   \
            const size_t shift = i * sizeof(T) * 8;                            \
            const Wide x = Wide(T(a >> shift)), y = Wide(T(b >> shift));       \
            const Wide r = op_expr;                                            \
            result |= (u32(r) & lane_mask) << shift;                           \
            ge |= (ge_expr) ? lane_mask << shift : 0;                          \
        }                                                                      \
        DST.lo = result;                                                       \
        DST.upper_lo = ge;                                                     \
        break;                                                                 \
    }
PACKED_ADDSUB(PackedAddU8, u8, s32, x + y, r >= 0x100)
PACKED_ADDSUB(PackedAddS8, s8, s32, x + y, r >= 0)
PACKED_ADDSUB(PackedSubU8, u8, s32, x - y, r >= 0)
PACKED_ADDSUB(PackedSubS8, s8, s32, x - y, r >= 0)
PACKED_ADDSUB(PackedAddU16, u16, s32, x + y, r >= 0x10000)
PACKED_ADDSUB(PackedAddS16, s16, s32, x + y, r >= 0)
PACKED_ADDSUB(PackedSubU16, u16, s32, x - y, r >= 0)
PACKED_ADDSUB(PackedSubS16, s16, s32, x - y, r >= 0)
#undef PACKED_ADDSUB

// ASX: upper = a.hi + b.lo, lower = a.lo - b.hi. SAX: upper = a.hi - b.lo, lower = a.lo + b.hi.
#define PACKED_EXCHANGE(name, T, add_upper, unsigned_ge)                                  \
    case Opcode::name: {                                                                  \
        const u32 a = u32(ARG(0).lo), b = u32(ARG(1).lo);                                 \
        const s32 a_lo = T(a), a_hi = T(a >> 16), b_lo = T(b), b_hi = T(b >> 16);         \
        const s32 upper = add_upper ? a_hi + b_lo : a_hi - b_lo;                          \
        const s32 lower = add_upper ? a_lo - b_hi : a_lo + b_hi;                          \
        const bool ge_upper = (add_upper && unsigned_ge) ? upper >= 0x10000 : upper >= 0; \
        const bool ge_lower = (!add_upper && unsigned_ge) ? lower >= 0x10000 : lower >= 0; \
        DST.lo = (u32(lower) & 0xFFFF) | (u32(upper) << 16);                              \
        DST.upper_lo = (ge_lower ? 0x0000FFFFu : 0) | (ge_upper ? 0xFFFF0000u : 0);       \
        break;                                                                            \
    }
PACKED_EXCHANGE(PackedAddSubU16, u16, true, true)
PACKED_EXCHANGE(PackedAddSubS16, s16, true, false)
PACKED_EXCHANGE(PackedSubAddU16, u16, false, true)
PACKED_EXCHANGE(PackedSubAddS16, s16, false, false)
#undef PACKED_EXCHANGE

#define PACKED_LANES(name, T, expr)                                            \
    case Opcode::name: {                                                       \
        constexpr size_t lanes = 4 / sizeof(T);                                \
        constexpr u32 lane_mask = sizeof(T) == 1 ? 0xFF : 0xFFFF;              \
        const u32 a = u32(ARG(0).lo), b = u32(ARG(1).lo);                      \
        u32 result = 0;                                                        \
        for (size_t i = 0; i < lanes; ++i) {                                   \
            const size_t shift = i * sizeof(T) * 8;                            \
            const s32 x = s32(T(a >> shift)), y = s32(T(b >> shift));          \
            result |= (u32(expr) & lane_mask) << shift;                        \
        }                                                                      \
        DST.lo = result;                                                       \
        break;                                                                 \
    }
PACKED_LANES(PackedHalvingAddU8, u8, (x + y) >> 1)
PACKED_LANES(PackedHalvingAddS8, s8, (x + y) >> 1)
PACKED_LANES(PackedHalvingSubU8, u8, (x - y) >> 1)
PACKED_LANES(PackedHalvingSubS8, s8, (x - y) >> 1)
PACKED_LANES(PackedHalvingAddU16, u16, (x + y) >> 1)
PACKED_LANES(PackedHalvingAddS16, s16, (x + y) >> 1)
PACKED_LANES(PackedHalvingSubU16, u16, (x - y) >> 1)
PACKED_LANES(PackedHalvingSubS16, s16, (x - y) >> 1)
PACKED_LANES(PackedSaturatedAddU8, u8, std::clamp(x + y, 0, 0xFF))
PACKED_LANES(PackedSaturatedAddS8, s8, std::clamp(x + y, -0x80, 0x7F))
PACKED_LANES(PackedSaturatedSubU8, u8, std::clamp(x - y, 0, 0xFF))
PACKED_LANES(PackedSaturatedSubS8, s8, std::clamp(x - y, -0x80, 0x7F))
PACKED_LANES(PackedSaturatedAddU16, u16, std::clamp(x + y, 0, 0xFFFF))
PACKED_LANES(PackedSaturatedAddS16, s16, std::clamp(x + y, -0x8000, 0x7FFF))
PACKED_LANES(PackedSaturatedSubU16, u16, std::clamp(x - y, 0, 0xFFFF))
PACKED_LANES(PackedSaturatedSubS16, s16, std::clamp(x - y, -0x8000, 0x7FFF))
#undef PACKED_LANES

#define PACKED_HALVING_EXCHANGE(name, T, add_upper)                                       \
    case Opcode::name: {                                                                  \
        const u32 a = u32(ARG(0).lo), b = u32(ARG(1).lo);                                 \
        const s32 a_lo = T(a), a_hi = T(a >> 16), b_lo = T(b), b_hi = T(b >> 16);         \
        const s32 upper = (add_upper ? a_hi + b_lo : a_hi - b_lo) >> 1;                   \
        const s32 lower = (add_upper ? a_lo - b_hi : a_lo + b_hi) >> 1;                   \
        DST.lo = (u32(lower) & 0xFFFF) | (u32(upper) << 16);                              \
        break;                                                                            \
    }
PACKED_HALVING_EXCHANGE(PackedHalvingAddSubU16, u16, true)
PACKED_HALVING_EXCHANGE(PackedHalvingAddSubS16, s16, true)
PACKED_HALVING_EXCHANGE(PackedHalvingSubAddU16, u16, false)
PACKED_HALVING_EXCHANGE(PackedHalvingSubAddS16, s16, false)
#undef PACKED_HALVING_EXCHANGE

case Opcode::PackedAbsDiffSumU8: {
    const u32 a = u32(ARG(0).lo), b = u32(ARG(1).lo);
    u32 sum = 0;
    for (size_t i = 0; i < 4; ++i) {
        const s32 x = u8(a >> (i * 8)), y = u8(b >> (i * 8));
        sum += u32(x > y ? x - y : y - x);
    }
    DST.lo = sum;
    break;
}
case Opcode::PackedSelect: {
    const u32 ge = u32(ARG(0).lo);
    DST.lo = (u32(ARG(2).lo) & ge) | (u32(ARG(1).lo) & ~ge);
    break;
}
