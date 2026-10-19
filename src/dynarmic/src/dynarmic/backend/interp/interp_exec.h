// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#include "common/common_types.h"
#include "dynarmic/backend/interp/interp.h"
#include "dynarmic/common/crypto/aes.h"
#include "dynarmic/ir/cond.h"

// Helpers shared by the interpreter operation includes. ARG/DST address the slot frame of the
// operation being executed.
#define ARG(i) s[op.args[i]]
#define DST s[op.dst]

namespace Dynarmic::Backend::Interp {

/// Flags are kept in the AArch64 NZCV register layout (N = bit 31).
inline u32 NZFromResult(u64 value, u8 width) {
    const u64 masked = width >= 64 ? value : value & ((u64(1) << width) - 1);
    const u32 n = width >= 64 ? u32(value >> 63) : u32((value >> (width - 1)) & 1);
    return (n << 31) | (u32(masked == 0) << 30);
}

inline bool CondHolds(IR::Cond cond, u32 nzcv) {
    const bool n = (nzcv >> 31) & 1;
    const bool z = (nzcv >> 30) & 1;
    const bool c = (nzcv >> 29) & 1;
    const bool v = (nzcv >> 28) & 1;
    switch (cond) {
    case IR::Cond::EQ:
        return z;
    case IR::Cond::NE:
        return !z;
    case IR::Cond::CS:
        return c;
    case IR::Cond::CC:
        return !c;
    case IR::Cond::MI:
        return n;
    case IR::Cond::PL:
        return !n;
    case IR::Cond::VS:
        return v;
    case IR::Cond::VC:
        return !v;
    case IR::Cond::HI:
        return c && !z;
    case IR::Cond::LS:
        return !c || z;
    case IR::Cond::GE:
        return n == v;
    case IR::Cond::LT:
        return n != v;
    case IR::Cond::GT:
        return !z && n == v;
    case IR::Cond::LE:
        return z || n != v;
    default:
        return true;
    }
}

template<typename T>
inline T SatAddS(T a, T b, bool& sat) {
    T result;
    if (__builtin_add_overflow(a, b, &result)) {
        sat = true;
        return b < 0 ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
    }
    return result;
}

template<typename T>
inline T SatSubS(T a, T b, bool& sat) {
    T result;
    if (__builtin_sub_overflow(a, b, &result)) {
        sat = true;
        return b > 0 ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
    }
    return result;
}

template<typename T>
inline T SatAddU(T a, T b, bool& sat) {
    T result;
    if (__builtin_add_overflow(a, b, &result)) {
        sat = true;
        return std::numeric_limits<T>::max();
    }
    return result;
}

template<typename T>
inline T SatSubU(T a, T b, bool& sat) {
    T result;
    if (__builtin_sub_overflow(a, b, &result)) {
        sat = true;
        return 0;
    }
    return result;
}

/// SQDMULH / SQRDMULH: high half of 2*a*b (optionally rounded), saturating.
template<typename T>
inline T SatDoublingMulHigh(T a, T b, bool round, bool& sat) {
    using Wide = std::conditional_t<sizeof(T) == 2, s32, s64>;
    constexpr int bits = sizeof(T) * 8;
    if (a == std::numeric_limits<T>::min() && b == std::numeric_limits<T>::min()) {
        sat = true;
        return std::numeric_limits<T>::max();
    }
    Wide product = Wide(a) * Wide(b);
    if (round) {
        product += Wide(1) << (bits - 2);
    }
    return T(product >> (bits - 1));
}

template<typename Fn>
inline void AesOp(Slot& dst, const Slot& src, Fn fn) {
    Common::Crypto::AES::State in, out;
    std::memcpy(in.data(), &src.lo, 8);
    std::memcpy(in.data() + 8, &src.hi, 8);
    fn(out, in);
    std::memcpy(&dst.lo, out.data(), 8);
    std::memcpy(&dst.hi, out.data() + 8, 8);
}

}  // namespace Dynarmic::Backend::Interp
