// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Integer SIMD operations. Included inside the interpreter dispatch switch.

#define V1(name, T, expr)                                       \
    case Opcode::name:                                          \
        Map1<T>(DST, ARG(0), [](T a) { return expr; });         \
        break;
#define V2(name, T, expr)                                             \
    case Opcode::name:                                                \
        Map2<T>(DST, ARG(0), ARG(1), [](T a, T b) { return expr; });  \
        break;
#define V2SAT(name, T, fn)                                                                  \
    case Opcode::name:                                                                      \
        SetQC(MapSat2<T>(DST, ARG(0), ARG(1), [](T a, T b, bool& sat) { return fn<T>(a, b, sat); })); \
        break;
#define V_ALL_SIZES(MACRO, name, ...) \
    MACRO(name##8, u8, __VA_ARGS__)   \
    MACRO(name##16, u16, __VA_ARGS__) \
    MACRO(name##32, u32, __VA_ARGS__) \
    MACRO(name##64, u64, __VA_ARGS__)
#define V_ALL_SIGNED(MACRO, name, ...) \
    MACRO(name##8, s8, __VA_ARGS__)    \
    MACRO(name##16, s16, __VA_ARGS__)  \
    MACRO(name##32, s32, __VA_ARGS__)  \
    MACRO(name##64, s64, __VA_ARGS__)

V2SAT(VectorSignedSaturatedAdd8, s8, SatAddS)
V2SAT(VectorSignedSaturatedAdd16, s16, SatAddS)
V2SAT(VectorSignedSaturatedAdd32, s32, SatAddS)
V2SAT(VectorSignedSaturatedAdd64, s64, SatAddS)
V2SAT(VectorSignedSaturatedSub8, s8, SatSubS)
V2SAT(VectorSignedSaturatedSub16, s16, SatSubS)
V2SAT(VectorSignedSaturatedSub32, s32, SatSubS)
V2SAT(VectorSignedSaturatedSub64, s64, SatSubS)
V2SAT(VectorUnsignedSaturatedAdd8, u8, SatAddU)
V2SAT(VectorUnsignedSaturatedAdd16, u16, SatAddU)
V2SAT(VectorUnsignedSaturatedAdd32, u32, SatAddU)
V2SAT(VectorUnsignedSaturatedAdd64, u64, SatAddU)
V2SAT(VectorUnsignedSaturatedSub8, u8, SatSubU)
V2SAT(VectorUnsignedSaturatedSub16, u16, SatSubU)
V2SAT(VectorUnsignedSaturatedSub32, u32, SatSubU)
V2SAT(VectorUnsignedSaturatedSub64, u64, SatSubU)
V2SAT(VectorSignedSaturatedShiftLeft8, s8, SignedSaturatedShiftLeft)
V2SAT(VectorSignedSaturatedShiftLeft16, s16, SignedSaturatedShiftLeft)
V2SAT(VectorSignedSaturatedShiftLeft32, s32, SignedSaturatedShiftLeft)
V2SAT(VectorSignedSaturatedShiftLeft64, s64, SignedSaturatedShiftLeft)
V2SAT(VectorUnsignedSaturatedShiftLeft8, u8, UnsignedSaturatedShiftLeft)
V2SAT(VectorUnsignedSaturatedShiftLeft16, u16, UnsignedSaturatedShiftLeft)
V2SAT(VectorUnsignedSaturatedShiftLeft32, u32, UnsignedSaturatedShiftLeft)
V2SAT(VectorUnsignedSaturatedShiftLeft64, u64, UnsignedSaturatedShiftLeft)

#define V_GET_ELEMENT(name, T, unused)            \
    case Opcode::name:                            \
        DST.lo = Get<T>(ARG(0))[ARG(1).lo];       \
        break;
V_ALL_SIZES(V_GET_ELEMENT, VectorGetElement, 0)
#undef V_GET_ELEMENT

#define V_SET_ELEMENT(name, T, unused)       \
    case Opcode::name: {                     \
        auto lanes = Get<T>(ARG(0));         \
        lanes[ARG(1).lo] = T(ARG(2).lo);     \
        Put(DST, lanes);                     \
        break;                               \
    }
V_ALL_SIZES(V_SET_ELEMENT, VectorSetElement, 0)
#undef V_SET_ELEMENT

#define V_ABS(name, T, unused) V1(name, T, a < 0 ? T(0 - std::make_unsigned_t<T>(a)) : a)
V_ALL_SIGNED(V_ABS, VectorAbs, 0)
#undef V_ABS

V2(VectorAdd8, u8, a + b)
V2(VectorAdd16, u16, a + b)
V2(VectorAdd32, u32, a + b)
V2(VectorAdd64, u64, a + b)
V2(VectorSub8, u8, a - b)
V2(VectorSub16, u16, a - b)
V2(VectorSub32, u32, a - b)
V2(VectorSub64, u64, a - b)
V2(VectorAnd, u64, a & b)
V2(VectorAndNot, u64, a & ~b)
V2(VectorEor, u64, a ^ b)
V2(VectorOr, u64, a | b)
V1(VectorNot, u64, ~a)

#define V_SHIFT_IMM(name, TYPE, expr)                                                \
    case Opcode::name: {                                                             \
        using T = TYPE;                                                              \
        const u32 shift = u32(ARG(1).lo);                                            \
        Map1<T>(DST, ARG(0), [shift](T a) { return expr; });                         \
        break;                                                                       \
    }
#define ESIZE (sizeof(a) * 8)
V_ALL_SIGNED(V_SHIFT_IMM, VectorArithmeticShiftRight, T(a >> std::min<u32>(shift, ESIZE - 1)))
V_ALL_SIZES(V_SHIFT_IMM, VectorLogicalShiftLeft, shift >= ESIZE ? T(0) : T(a << shift))
V_ALL_SIZES(V_SHIFT_IMM, VectorLogicalShiftRight, shift >= ESIZE ? T(0) : T(a >> shift))
#undef ESIZE
#undef V_SHIFT_IMM

V2(VectorArithmeticVShift8, s8, VShift<s8>(a, b))
V2(VectorArithmeticVShift16, s16, VShift<s16>(a, b))
V2(VectorArithmeticVShift32, s32, VShift<s32>(a, b))
V2(VectorArithmeticVShift64, s64, VShift<s64>(a, b))
V2(VectorLogicalVShift8, u8, VShift<u8>(a, b))
V2(VectorLogicalVShift16, u16, VShift<u16>(a, b))
V2(VectorLogicalVShift32, u32, VShift<u32>(a, b))
V2(VectorLogicalVShift64, u64, VShift<u64>(a, b))
V2(VectorRoundingShiftLeftS8, s8, RoundingShiftLeft<s8>(a, b))
V2(VectorRoundingShiftLeftS16, s16, RoundingShiftLeft<s16>(a, b))
V2(VectorRoundingShiftLeftS32, s32, RoundingShiftLeft<s32>(a, b))
V2(VectorRoundingShiftLeftS64, s64, RoundingShiftLeft<s64>(a, b))
V2(VectorRoundingShiftLeftU8, u8, RoundingShiftLeft<u8>(a, b))
V2(VectorRoundingShiftLeftU16, u16, RoundingShiftLeft<u16>(a, b))
V2(VectorRoundingShiftLeftU32, u32, RoundingShiftLeft<u32>(a, b))
V2(VectorRoundingShiftLeftU64, u64, RoundingShiftLeft<u64>(a, b))

#define V_BROADCAST(name, T, lower)                     \
    case Opcode::name: {                                \
        Lanes<T> lanes;                                 \
        lanes.fill(T(ARG(0).lo));                       \
        Put(DST, lanes);                                \
        if (lower) {                                    \
            DST.hi = 0;                                 \
        }                                               \
        break;                                          \
    }
V_BROADCAST(VectorBroadcastLower8, u8, true)
V_BROADCAST(VectorBroadcastLower16, u16, true)
V_BROADCAST(VectorBroadcastLower32, u32, true)
V_BROADCAST(VectorBroadcast8, u8, false)
V_BROADCAST(VectorBroadcast16, u16, false)
V_BROADCAST(VectorBroadcast32, u32, false)
V_BROADCAST(VectorBroadcast64, u64, false)
#undef V_BROADCAST

#define V_BROADCAST_ELEMENT(name, T, lower)             \
    case Opcode::name: {                                \
        Lanes<T> lanes;                                 \
        lanes.fill(Get<T>(ARG(0))[ARG(1).lo]);          \
        Put(DST, lanes);                                \
        if (lower) {                                    \
            DST.hi = 0;                                 \
        }                                               \
        break;                                          \
    }
V_BROADCAST_ELEMENT(VectorBroadcastElementLower8, u8, true)
V_BROADCAST_ELEMENT(VectorBroadcastElementLower16, u16, true)
V_BROADCAST_ELEMENT(VectorBroadcastElementLower32, u32, true)
V_BROADCAST_ELEMENT(VectorBroadcastElement8, u8, false)
V_BROADCAST_ELEMENT(VectorBroadcastElement16, u16, false)
V_BROADCAST_ELEMENT(VectorBroadcastElement32, u32, false)
V_BROADCAST_ELEMENT(VectorBroadcastElement64, u64, false)
#undef V_BROADCAST_ELEMENT

V1(VectorCountLeadingZeros8, u8, std::countl_zero(a))
V1(VectorCountLeadingZeros16, u16, std::countl_zero(a))
V1(VectorCountLeadingZeros32, u32, std::countl_zero(a))

#define V_DEINTERLEAVE(name, T, odd, lower)                                 \
    case Opcode::name: {                                                    \
        auto x = Get<T>(ARG(0));                                            \
        auto y = Get<T>(ARG(1));                                            \
        Lanes<T> r{};                                                       \
        constexpr size_t n = r.size();                                      \
        if (lower) {                                                        \
            for (size_t i = 0; i < n / 4; ++i) {                            \
                r[i] = x[2 * i + (odd)];                                    \
                r[n / 4 + i] = y[2 * i + (odd)];                            \
            }                                                               \
        } else {                                                            \
            for (size_t i = 0; i < n / 2; ++i) {                            \
                r[i] = x[2 * i + (odd)];                                    \
                r[n / 2 + i] = y[2 * i + (odd)];                            \
            }                                                               \
        }                                                                   \
        Put(DST, r);                                                        \
        break;                                                              \
    }
V_DEINTERLEAVE(VectorDeinterleaveEven8, u8, 0, false)
V_DEINTERLEAVE(VectorDeinterleaveEven16, u16, 0, false)
V_DEINTERLEAVE(VectorDeinterleaveEven32, u32, 0, false)
V_DEINTERLEAVE(VectorDeinterleaveEven64, u64, 0, false)
V_DEINTERLEAVE(VectorDeinterleaveEvenLower8, u8, 0, true)
V_DEINTERLEAVE(VectorDeinterleaveEvenLower16, u16, 0, true)
V_DEINTERLEAVE(VectorDeinterleaveEvenLower32, u32, 0, true)
V_DEINTERLEAVE(VectorDeinterleaveOdd8, u8, 1, false)
V_DEINTERLEAVE(VectorDeinterleaveOdd16, u16, 1, false)
V_DEINTERLEAVE(VectorDeinterleaveOdd32, u32, 1, false)
V_DEINTERLEAVE(VectorDeinterleaveOdd64, u64, 1, false)
V_DEINTERLEAVE(VectorDeinterleaveOddLower8, u8, 1, true)
V_DEINTERLEAVE(VectorDeinterleaveOddLower16, u16, 1, true)
V_DEINTERLEAVE(VectorDeinterleaveOddLower32, u32, 1, true)
#undef V_DEINTERLEAVE

V2(VectorEqual8, u8, Ones<u8>(a == b))
V2(VectorEqual16, u16, Ones<u16>(a == b))
V2(VectorEqual32, u32, Ones<u32>(a == b))
V2(VectorEqual64, u64, Ones<u64>(a == b))
case Opcode::VectorEqual128: {
    const bool equal = ARG(0).lo == ARG(1).lo && ARG(0).hi == ARG(1).hi;
    DST.lo = DST.hi = Ones<u64>(equal);
    break;
}
case Opcode::VectorExtract: {
    const auto x = Get<u8>(ARG(0));
    const auto y = Get<u8>(ARG(1));
    const size_t bytes = ARG(2).lo / 8;
    Lanes<u8> r;
    for (size_t i = 0; i < 16; ++i) {
        const size_t j = i + bytes;
        r[i] = j < 16 ? x[j] : y[j - 16];
    }
    Put(DST, r);
    break;
}
case Opcode::VectorExtractLower: {
    const auto x = Get<u8>(ARG(0));
    const auto y = Get<u8>(ARG(1));
    const size_t bytes = ARG(2).lo / 8;
    Lanes<u8> r{};
    for (size_t i = 0; i < 8; ++i) {
        const size_t j = i + bytes;
        r[i] = j < 8 ? x[j] : y[j - 8];
    }
    Put(DST, r);
    break;
}

V2(VectorGreaterS8, s8, Ones<s8>(a > b))
V2(VectorGreaterS16, s16, Ones<s16>(a > b))
V2(VectorGreaterS32, s32, Ones<s32>(a > b))
V2(VectorGreaterS64, s64, Ones<s64>(a > b))

V2(VectorHalvingAddS8, s8, (s16(a) + s16(b)) >> 1)
V2(VectorHalvingAddS16, s16, (s32(a) + s32(b)) >> 1)
V2(VectorHalvingAddS32, s32, (s64(a) + s64(b)) >> 1)
V2(VectorHalvingAddU8, u8, (u16(a) + u16(b)) >> 1)
V2(VectorHalvingAddU16, u16, (u32(a) + u32(b)) >> 1)
V2(VectorHalvingAddU32, u32, (u64(a) + u64(b)) >> 1)
V2(VectorHalvingSubS8, s8, (s16(a) - s16(b)) >> 1)
V2(VectorHalvingSubS16, s16, (s32(a) - s32(b)) >> 1)
V2(VectorHalvingSubS32, s32, (s64(a) - s64(b)) >> 1)
V2(VectorHalvingSubU8, u8, (u16(a) - u16(b)) >> 1)
V2(VectorHalvingSubU16, u16, (u32(a) - u32(b)) >> 1)
V2(VectorHalvingSubU32, u32, (u64(a) - u64(b)) >> 1)
V2(VectorRoundingHalvingAddS8, s8, (s16(a) + s16(b) + 1) >> 1)
V2(VectorRoundingHalvingAddS16, s16, (s32(a) + s32(b) + 1) >> 1)
V2(VectorRoundingHalvingAddS32, s32, (s64(a) + s64(b) + 1) >> 1)
V2(VectorRoundingHalvingAddU8, u8, (u16(a) + u16(b) + 1) >> 1)
V2(VectorRoundingHalvingAddU16, u16, (u32(a) + u32(b) + 1) >> 1)
V2(VectorRoundingHalvingAddU32, u32, (u64(a) + u64(b) + 1) >> 1)

#define V_INTERLEAVE(name, T, upper)                               \
    case Opcode::name: {                                           \
        const auto x = Get<T>(ARG(0));                             \
        const auto y = Get<T>(ARG(1));                             \
        Lanes<T> r;                                                \
        constexpr size_t half = r.size() / 2;                      \
        for (size_t i = 0; i < half; ++i) {                        \
            r[2 * i] = x[i + (upper ? half : 0)];                  \
            r[2 * i + 1] = y[i + (upper ? half : 0)];              \
        }                                                          \
        Put(DST, r);                                               \
        break;                                                     \
    }
V_INTERLEAVE(VectorInterleaveLower8, u8, false)
V_INTERLEAVE(VectorInterleaveLower16, u16, false)
V_INTERLEAVE(VectorInterleaveLower32, u32, false)
V_INTERLEAVE(VectorInterleaveLower64, u64, false)
V_INTERLEAVE(VectorInterleaveUpper8, u8, true)
V_INTERLEAVE(VectorInterleaveUpper16, u16, true)
V_INTERLEAVE(VectorInterleaveUpper32, u32, true)
V_INTERLEAVE(VectorInterleaveUpper64, u64, true)
#undef V_INTERLEAVE

V2(VectorMaxS8, s8, std::max(a, b))
V2(VectorMaxS16, s16, std::max(a, b))
V2(VectorMaxS32, s32, std::max(a, b))
V2(VectorMaxS64, s64, std::max(a, b))
V2(VectorMaxU8, u8, std::max(a, b))
V2(VectorMaxU16, u16, std::max(a, b))
V2(VectorMaxU32, u32, std::max(a, b))
V2(VectorMaxU64, u64, std::max(a, b))
V2(VectorMinS8, s8, std::min(a, b))
V2(VectorMinS16, s16, std::min(a, b))
V2(VectorMinS32, s32, std::min(a, b))
V2(VectorMinS64, s64, std::min(a, b))
V2(VectorMinU8, u8, std::min(a, b))
V2(VectorMinU16, u16, std::min(a, b))
V2(VectorMinU32, u32, std::min(a, b))
V2(VectorMinU64, u64, std::min(a, b))
V2(VectorMultiply8, u8, a * b)
V2(VectorMultiply16, u16, u32(a) * u32(b))
V2(VectorMultiply32, u32, a * b)
V2(VectorMultiply64, u64, a * b)

#define V_MULTIPLY_WIDEN(name, Narrow, Wide)                       \
    case Opcode::name: {                                           \
        const auto x = Get<Narrow>(ARG(0));                        \
        const auto y = Get<Narrow>(ARG(1));                        \
        Lanes<Wide> r;                                             \
        for (size_t i = 0; i < r.size(); ++i) {                    \
            r[i] = Wide(Wide(x[i]) * Wide(y[i]));                  \
        }                                                          \
        Put(DST, r);                                               \
        break;                                                     \
    }
V_MULTIPLY_WIDEN(VectorMultiplySignedWiden8, s8, s16)
V_MULTIPLY_WIDEN(VectorMultiplySignedWiden16, s16, s32)
V_MULTIPLY_WIDEN(VectorMultiplySignedWiden32, s32, s64)
V_MULTIPLY_WIDEN(VectorMultiplyUnsignedWiden8, u8, u16)
V_MULTIPLY_WIDEN(VectorMultiplyUnsignedWiden16, u16, u32)
V_MULTIPLY_WIDEN(VectorMultiplyUnsignedWiden32, u32, u64)
#undef V_MULTIPLY_WIDEN

#define V_NARROW(name, Wide, Narrow)                               \
    case Opcode::name: {                                           \
        const auto x = Get<Wide>(ARG(0));                          \
        Lanes<Narrow> r{};                                         \
        for (size_t i = 0; i < x.size(); ++i) {                    \
            r[i] = Narrow(x[i]);                                   \
        }                                                          \
        Put(DST, r);                                               \
        break;                                                     \
    }
V_NARROW(VectorNarrow16, u16, u8)
V_NARROW(VectorNarrow32, u32, u16)
V_NARROW(VectorNarrow64, u64, u32)
#undef V_NARROW

#define V_SAT_NARROW(name, Wide, Narrow)                           \
    case Opcode::name: {                                           \
        const auto x = Get<Wide>(ARG(0));                          \
        Lanes<Narrow> r{};                                         \
        bool sat = false;                                          \
        for (size_t i = 0; i < x.size(); ++i) {                    \
            r[i] = SaturateTo<Narrow>(x[i], sat);                  \
        }                                                          \
        Put(DST, r);                                               \
        SetQC(sat);                                                \
        break;                                                     \
    }
V_SAT_NARROW(VectorSignedSaturatedNarrowToSigned16, s16, s8)
V_SAT_NARROW(VectorSignedSaturatedNarrowToSigned32, s32, s16)
V_SAT_NARROW(VectorSignedSaturatedNarrowToSigned64, s64, s32)
V_SAT_NARROW(VectorSignedSaturatedNarrowToUnsigned16, s16, u8)
V_SAT_NARROW(VectorSignedSaturatedNarrowToUnsigned32, s32, u16)
V_SAT_NARROW(VectorSignedSaturatedNarrowToUnsigned64, s64, u32)
V_SAT_NARROW(VectorUnsignedSaturatedNarrow16, u16, u8)
V_SAT_NARROW(VectorUnsignedSaturatedNarrow32, u32, u16)
V_SAT_NARROW(VectorUnsignedSaturatedNarrow64, u64, u32)
#undef V_SAT_NARROW

// Paired operations work over the concatenation a:b (or a_lo:b_lo for the Lower variants).
#define V_PAIRED(name, T, lower, fn)                                        \
    case Opcode::name: {                                                    \
        const auto x = Get<T>(ARG(0));                                      \
        const auto y = Get<T>(ARG(1));                                      \
        constexpr size_t n = 16 / sizeof(T);                                \
        std::array<T, 2 * n> concat{};                                      \
        const size_t used = lower ? n / 2 : n;                              \
        for (size_t i = 0; i < used; ++i) {                                 \
            concat[i] = x[i];                                               \
            concat[used + i] = y[i];                                        \
        }                                                                   \
        Lanes<T> r{};                                                       \
        for (size_t i = 0; i < used; ++i) {                                 \
            const T a = concat[2 * i], b = concat[2 * i + 1];               \
            r[i] = T(fn);                                                   \
        }                                                                   \
        Put(DST, r);                                                        \
        break;                                                              \
    }
V_PAIRED(VectorPairedAddLower8, u8, true, a + b)
V_PAIRED(VectorPairedAddLower16, u16, true, a + b)
V_PAIRED(VectorPairedAddLower32, u32, true, a + b)
V_PAIRED(VectorPairedAdd8, u8, false, a + b)
V_PAIRED(VectorPairedAdd16, u16, false, a + b)
V_PAIRED(VectorPairedAdd32, u32, false, a + b)
V_PAIRED(VectorPairedAdd64, u64, false, a + b)
V_PAIRED(VectorPairedMaxS8, s8, false, std::max(a, b))
V_PAIRED(VectorPairedMaxS16, s16, false, std::max(a, b))
V_PAIRED(VectorPairedMaxS32, s32, false, std::max(a, b))
V_PAIRED(VectorPairedMaxU8, u8, false, std::max(a, b))
V_PAIRED(VectorPairedMaxU16, u16, false, std::max(a, b))
V_PAIRED(VectorPairedMaxU32, u32, false, std::max(a, b))
V_PAIRED(VectorPairedMinS8, s8, false, std::min(a, b))
V_PAIRED(VectorPairedMinS16, s16, false, std::min(a, b))
V_PAIRED(VectorPairedMinS32, s32, false, std::min(a, b))
V_PAIRED(VectorPairedMinU8, u8, false, std::min(a, b))
V_PAIRED(VectorPairedMinU16, u16, false, std::min(a, b))
V_PAIRED(VectorPairedMinU32, u32, false, std::min(a, b))
V_PAIRED(VectorPairedMaxLowerS8, s8, true, std::max(a, b))
V_PAIRED(VectorPairedMaxLowerS16, s16, true, std::max(a, b))
V_PAIRED(VectorPairedMaxLowerS32, s32, true, std::max(a, b))
V_PAIRED(VectorPairedMaxLowerU8, u8, true, std::max(a, b))
V_PAIRED(VectorPairedMaxLowerU16, u16, true, std::max(a, b))
V_PAIRED(VectorPairedMaxLowerU32, u32, true, std::max(a, b))
V_PAIRED(VectorPairedMinLowerS8, s8, true, std::min(a, b))
V_PAIRED(VectorPairedMinLowerS16, s16, true, std::min(a, b))
V_PAIRED(VectorPairedMinLowerS32, s32, true, std::min(a, b))
V_PAIRED(VectorPairedMinLowerU8, u8, true, std::min(a, b))
V_PAIRED(VectorPairedMinLowerU16, u16, true, std::min(a, b))
V_PAIRED(VectorPairedMinLowerU32, u32, true, std::min(a, b))
#undef V_PAIRED

#define V_PAIRED_WIDEN(name, Narrow, Wide)                         \
    case Opcode::name: {                                           \
        const auto x = Get<Narrow>(ARG(0));                        \
        Lanes<Wide> r;                                             \
        for (size_t i = 0; i < r.size(); ++i) {                    \
            r[i] = Wide(Wide(x[2 * i]) + Wide(x[2 * i + 1]));      \
        }                                                          \
        Put(DST, r);                                               \
        break;                                                     \
    }
V_PAIRED_WIDEN(VectorPairedAddSignedWiden8, s8, s16)
V_PAIRED_WIDEN(VectorPairedAddSignedWiden16, s16, s32)
V_PAIRED_WIDEN(VectorPairedAddSignedWiden32, s32, s64)
V_PAIRED_WIDEN(VectorPairedAddUnsignedWiden8, u8, u16)
V_PAIRED_WIDEN(VectorPairedAddUnsignedWiden16, u16, u32)
V_PAIRED_WIDEN(VectorPairedAddUnsignedWiden32, u32, u64)
#undef V_PAIRED_WIDEN

V2(VectorPolynomialMultiply8, u8, PolynomialMultiply<u8>(a, b))
case Opcode::VectorPolynomialMultiplyLong8: {
    const auto x = Get<u8>(ARG(0));
    const auto y = Get<u8>(ARG(1));
    Lanes<u16> r;
    for (size_t i = 0; i < r.size(); ++i) {
        r[i] = PolynomialMultiply<u16>(x[i], y[i]);
    }
    Put(DST, r);
    break;
}
case Opcode::VectorPolynomialMultiplyLong64: {
    const u64 a = ARG(0).lo, b = ARG(1).lo;
    u64 lo = 0, hi = 0;
    for (size_t i = 0; i < 64; ++i) {
        if ((a >> i) & 1) {
            lo ^= b << i;
            if (i != 0) {
                hi ^= b >> (64 - i);
            }
        }
    }
    DST.lo = lo;
    DST.hi = hi;
    break;
}
V1(VectorPopulationCount, u8, std::popcount(a))
V1(VectorReverseBits, u8, u8((a * 0x0202020202ull & 0x010884422010ull) % 1023))

#define V_REVERSE_GROUPS(name, T, group_bytes)                     \
    case Opcode::name: {                                           \
        const auto x = Get<T>(ARG(0));                             \
        Lanes<T> r;                                                \
        constexpr size_t per_group = (group_bytes) / sizeof(T);    \
        for (size_t i = 0; i < r.size(); ++i) {                    \
            const size_t base = i - i % per_group;                 \
            r[i] = x[base + (per_group - 1 - i % per_group)];      \
        }                                                          \
        Put(DST, r);                                               \
        break;                                                     \
    }
V_REVERSE_GROUPS(VectorReverseElementsInHalfGroups8, u8, 2)
V_REVERSE_GROUPS(VectorReverseElementsInWordGroups8, u8, 4)
V_REVERSE_GROUPS(VectorReverseElementsInWordGroups16, u16, 4)
V_REVERSE_GROUPS(VectorReverseElementsInLongGroups8, u8, 8)
V_REVERSE_GROUPS(VectorReverseElementsInLongGroups16, u16, 8)
V_REVERSE_GROUPS(VectorReverseElementsInLongGroups32, u32, 8)
#undef V_REVERSE_GROUPS

#define V_REDUCE_ADD(name, T)                                      \
    case Opcode::name: {                                           \
        const auto x = Get<T>(ARG(0));                             \
        T sum = 0;                                                 \
        for (const T v : x) {                                      \
            sum = T(sum + v);                                      \
        }                                                          \
        Lanes<T> r{};                                              \
        r[0] = sum;                                                \
        Put(DST, r);                                               \
        break;                                                     \
    }
V_REDUCE_ADD(VectorReduceAdd8, u8)
V_REDUCE_ADD(VectorReduceAdd16, u16)
V_REDUCE_ADD(VectorReduceAdd32, u32)
V_REDUCE_ADD(VectorReduceAdd64, u64)
#undef V_REDUCE_ADD

case Opcode::VectorRotateWholeVectorRight: {
    const unsigned __int128 v = (unsigned __int128)(ARG(0).hi) << 64 | ARG(0).lo;
    const u32 amount = u32(ARG(1).lo) & 127;
    const unsigned __int128 r = amount == 0 ? v : (v >> amount) | (v << (128 - amount));
    DST.lo = u64(r);
    DST.hi = u64(r >> 64);
    break;
}

#define V_EXTEND(name, Narrow, Wide)                               \
    case Opcode::name: {                                           \
        const auto x = Get<Narrow>(ARG(0));                        \
        Lanes<Wide> r;                                             \
        for (size_t i = 0; i < r.size(); ++i) {                    \
            r[i] = Wide(x[i]);                                     \
        }                                                          \
        Put(DST, r);                                               \
        break;                                                     \
    }
V_EXTEND(VectorSignExtend8, s8, s16)
V_EXTEND(VectorSignExtend16, s16, s32)
V_EXTEND(VectorSignExtend32, s32, s64)
V_EXTEND(VectorZeroExtend8, u8, u16)
V_EXTEND(VectorZeroExtend16, u16, u32)
V_EXTEND(VectorZeroExtend32, u32, u64)
#undef V_EXTEND
case Opcode::VectorSignExtend64:
    DST.lo = ARG(0).lo;
    DST.hi = u64(s64(ARG(0).lo) >> 63);
    break;
case Opcode::VectorZeroExtend64:
case Opcode::VectorZeroUpper:
    DST.lo = ARG(0).lo;
    DST.hi = 0;
    break;

V2(VectorSignedAbsoluteDifference8, s8, a > b ? u8(a) - u8(b) : u8(b) - u8(a))
V2(VectorSignedAbsoluteDifference16, s16, a > b ? u16(a) - u16(b) : u16(b) - u16(a))
V2(VectorSignedAbsoluteDifference32, s32, a > b ? u32(a) - u32(b) : u32(b) - u32(a))
V2(VectorUnsignedAbsoluteDifference8, u8, a > b ? a - b : b - a)
V2(VectorUnsignedAbsoluteDifference16, u16, a > b ? a - b : b - a)
V2(VectorUnsignedAbsoluteDifference32, u32, a > b ? a - b : b - a)

// The product halves are read back through GetLowerFromOp and GetUpperFromOp.
#define V_MULTIPLY_HALVES(name, T, Wide)                           \
    case Opcode::name: {                                           \
        const auto x = Get<T>(ARG(0));                             \
        const auto y = Get<T>(ARG(1));                             \
        Lanes<T> lower, upper;                                     \
        for (size_t i = 0; i < lower.size(); ++i) {                \
            const Wide product = Wide(x[i]) * Wide(y[i]);          \
            lower[i] = T(product);                                 \
            upper[i] = T(product >> (sizeof(T) * 8));              \
        }                                                          \
        Put(DST, lower);                                           \
        std::memcpy(&DST.upper_lo, upper.data(), 16);              \
        break;                                                     \
    }
V_MULTIPLY_HALVES(VectorSignedMultiply16, s16, s32)
V_MULTIPLY_HALVES(VectorSignedMultiply32, s32, s64)
V_MULTIPLY_HALVES(VectorUnsignedMultiply16, u16, u32)
V_MULTIPLY_HALVES(VectorUnsignedMultiply32, u32, u64)
#undef V_MULTIPLY_HALVES

#define V_SAT_ABS_NEG(name, T, neg)                                            \
    case Opcode::name:                                                         \
        SetQC(MapSat1<T>(DST, ARG(0), [](T a, bool& sat) -> T {                \
            if (a == std::numeric_limits<T>::min()) {                          \
                sat = true;                                                    \
                return std::numeric_limits<T>::max();                          \
            }                                                                  \
            return (neg) ? T(-a) : (a < 0 ? T(-a) : a);                        \
        }));                                                                   \
        break;
V_SAT_ABS_NEG(VectorSignedSaturatedAbs8, s8, false)
V_SAT_ABS_NEG(VectorSignedSaturatedAbs16, s16, false)
V_SAT_ABS_NEG(VectorSignedSaturatedAbs32, s32, false)
V_SAT_ABS_NEG(VectorSignedSaturatedAbs64, s64, false)
V_SAT_ABS_NEG(VectorSignedSaturatedNeg8, s8, true)
V_SAT_ABS_NEG(VectorSignedSaturatedNeg16, s16, true)
V_SAT_ABS_NEG(VectorSignedSaturatedNeg32, s32, true)
V_SAT_ABS_NEG(VectorSignedSaturatedNeg64, s64, true)
#undef V_SAT_ABS_NEG

// SUQADD: signed accumulator (arg 1) plus unsigned addend (arg 0).
#define V_SUQADD(name, S)                                                      \
    case Opcode::name: {                                                       \
        using U = std::make_unsigned_t<S>;                                     \
        const auto addend = Get<U>(ARG(0));                                    \
        const auto acc = Get<S>(ARG(1));                                       \
        Lanes<S> r;                                                            \
        bool sat = false;                                                      \
        for (size_t i = 0; i < r.size(); ++i) {                                \
            if constexpr (sizeof(S) < 8) {                                     \
                r[i] = SaturateTo<S>(s64(acc[i]) + s64(addend[i]), sat);       \
            } else {                                                           \
                const __int128 sum = __int128(acc[i]) + __int128(addend[i]);   \
                r[i] = SaturateTo<S>(sum, sat);                                \
            }                                                                  \
        }                                                                      \
        Put(DST, r);                                                           \
        SetQC(sat);                                                            \
        break;                                                                 \
    }
V_SUQADD(VectorSignedSaturatedAccumulateUnsigned8, s8)
V_SUQADD(VectorSignedSaturatedAccumulateUnsigned16, s16)
V_SUQADD(VectorSignedSaturatedAccumulateUnsigned32, s32)
V_SUQADD(VectorSignedSaturatedAccumulateUnsigned64, s64)
#undef V_SUQADD

// USQADD: unsigned accumulator (arg 1) plus signed addend (arg 0).
#define V_USQADD(name, S)                                                      \
    case Opcode::name: {                                                       \
        using U = std::make_unsigned_t<S>;                                     \
        const auto addend = Get<S>(ARG(0));                                    \
        const auto acc = Get<U>(ARG(1));                                       \
        Lanes<U> r;                                                            \
        bool sat = false;                                                      \
        for (size_t i = 0; i < r.size(); ++i) {                                \
            const __int128 sum = __int128(acc[i]) + __int128(addend[i]);       \
            r[i] = SaturateTo<U>(sum, sat);                                    \
        }                                                                      \
        Put(DST, r);                                                           \
        SetQC(sat);                                                            \
        break;                                                                 \
    }
V_USQADD(VectorUnsignedSaturatedAccumulateSigned8, s8)
V_USQADD(VectorUnsignedSaturatedAccumulateSigned16, s16)
V_USQADD(VectorUnsignedSaturatedAccumulateSigned32, s32)
V_USQADD(VectorUnsignedSaturatedAccumulateSigned64, s64)
#undef V_USQADD

#define V_SQDMULH(name, T, round)                                                          \
    case Opcode::name:                                                                     \
        SetQC(MapSat2<T>(DST, ARG(0), ARG(1), [](T a, T b, bool& sat) {                    \
            return SatDoublingMulHigh<T>(a, b, round, sat);                                \
        }));                                                                               \
        break;
V_SQDMULH(VectorSignedSaturatedDoublingMultiplyHigh16, s16, false)
V_SQDMULH(VectorSignedSaturatedDoublingMultiplyHigh32, s32, false)
V_SQDMULH(VectorSignedSaturatedDoublingMultiplyHighRounding16, s16, true)
V_SQDMULH(VectorSignedSaturatedDoublingMultiplyHighRounding32, s32, true)
#undef V_SQDMULH

#define V_SQDMULL(name, T, Wide)                                                   \
    case Opcode::name: {                                                           \
        const auto x = Get<T>(ARG(0));                                             \
        const auto y = Get<T>(ARG(1));                                             \
        Lanes<Wide> r;                                                             \
        bool sat = false;                                                          \
        for (size_t i = 0; i < r.size(); ++i) {                                    \
            if (x[i] == std::numeric_limits<T>::min() && y[i] == std::numeric_limits<T>::min()) { \
                r[i] = std::numeric_limits<Wide>::max();                           \
                sat = true;                                                        \
            } else {                                                               \
                r[i] = Wide(Wide(x[i]) * Wide(y[i]) * 2);                          \
            }                                                                      \
        }                                                                          \
        Put(DST, r);                                                               \
        SetQC(sat);                                                                \
        break;                                                                     \
    }
V_SQDMULL(VectorSignedSaturatedDoublingMultiplyLong16, s16, s32)
V_SQDMULL(VectorSignedSaturatedDoublingMultiplyLong32, s32, s64)
#undef V_SQDMULL

#define V_SQSHLU(name, T)                                                          \
    case Opcode::name: {                                                           \
        using U = std::make_unsigned_t<T>;                                         \
        const auto x = Get<T>(ARG(0));                                             \
        const u32 shift = u32(ARG(1).lo);                                          \
        Lanes<U> r;                                                                \
        bool sat = false;                                                          \
        for (size_t i = 0; i < r.size(); ++i) {                                    \
            const T element = x[i];                                                \
            const U shifted = U(U(element) << shift);                              \
            const bool overflow = element > 0 && U(shifted >> shift) != U(element);\
            sat |= element < 0 || overflow;                                        \
            r[i] = element <= 0 ? U(0) : (overflow ? std::numeric_limits<U>::max() : shifted); \
        }                                                                          \
        Put(DST, r);                                                               \
        SetQC(sat);                                                                \
        break;                                                                     \
    }
V_SQSHLU(VectorSignedSaturatedShiftLeftUnsigned8, s8)
V_SQSHLU(VectorSignedSaturatedShiftLeftUnsigned16, s16)
V_SQSHLU(VectorSignedSaturatedShiftLeftUnsigned32, s32)
V_SQSHLU(VectorSignedSaturatedShiftLeftUnsigned64, s64)
#undef V_SQSHLU

case Opcode::VectorTable:
    // Remember the table registers; the lookup reads them from the frame.
    DST.lo = u64(op.args[0]) | (u64(op.args[1]) << 32);
    DST.hi = u64(op.args[2]) | (u64(op.args[3]) << 32);
    DST.carry = op.width;
    break;
case Opcode::VectorTableLookup64:
case Opcode::VectorTableLookup128: {
    const bool is128 = op.opcode == Opcode::VectorTableLookup128;
    const Slot& table = ARG(1);
    const u32 table_slots[4] = {u32(table.lo), u32(table.lo >> 32), u32(table.hi), u32(table.hi >> 32)};
    const size_t entry_bytes = is128 ? 16 : 8;
    std::array<u8, 64> bytes{};
    for (size_t t = 0; t < table.carry; ++t) {
        std::memcpy(bytes.data() + t * entry_bytes, &s[table_slots[t]].lo, entry_bytes);
    }
    const size_t table_size = table.carry * entry_bytes;
    const auto defaults = Get<u8>(ARG(0));
    const auto indices = Get<u8>(ARG(2));
    Lanes<u8> r{};
    for (size_t i = 0; i < entry_bytes; ++i) {
        r[i] = indices[i] < table_size ? bytes[indices[i]] : defaults[i];
    }
    Put(DST, r);
    break;
}

#define V_TRANSPOSE(name, T)                                       \
    case Opcode::name: {                                           \
        const auto x = Get<T>(ARG(0));                             \
        const auto y = Get<T>(ARG(1));                             \
        const size_t part = ARG(2).lo & 1;                         \
        Lanes<T> r;                                                \
        for (size_t i = 0; i < r.size(); i += 2) {                 \
            r[i] = x[i + part];                                    \
            r[i + 1] = y[i + part];                                \
        }                                                          \
        Put(DST, r);                                               \
        break;                                                     \
    }
V_TRANSPOSE(VectorTranspose8, u8)
V_TRANSPOSE(VectorTranspose16, u16)
V_TRANSPOSE(VectorTranspose32, u32)
V_TRANSPOSE(VectorTranspose64, u64)
#undef V_TRANSPOSE

V1(VectorUnsignedRecipEstimate, u32, (a & 0x80000000) == 0 ? 0xFFFFFFFFu : (0b100000000u | Common::RecipEstimate((a >> 23) & 0x1FF)) << 23)
V1(VectorUnsignedRecipSqrtEstimate, u32, (a & 0xC0000000) == 0 ? 0xFFFFFFFFu : (0b100000000u | Common::RecipSqrtEstimate((a >> 23) & 0x1FF)) << 23)

#undef V_ALL_SIZES
#undef V_ALL_SIGNED
#undef V2SAT
#undef V2
#undef V1
