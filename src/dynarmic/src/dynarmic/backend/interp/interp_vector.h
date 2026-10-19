// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <type_traits>

#include "dynarmic/backend/interp/interp_exec.h"
#include "dynarmic/common/math_util.h"

namespace Dynarmic::Backend::Interp {

template<typename T>
using Lanes = std::array<T, 16 / sizeof(T)>;

static_assert(offsetof(Slot, hi) == offsetof(Slot, lo) + 8, "Vector lanes span lo and hi");

template<typename T>
inline Lanes<T> Get(const Slot& slot) {
    Lanes<T> lanes;
    std::memcpy(lanes.data(), &slot.lo, 16);
    return lanes;
}

template<typename T>
inline void Put(Slot& slot, const Lanes<T>& lanes) {
    std::memcpy(&slot.lo, lanes.data(), 16);
}

template<typename T, typename F>
inline void Map1(Slot& dst, const Slot& a, F f) {
    const auto x = Get<T>(a);
    Lanes<T> r;
    for (size_t i = 0; i < r.size(); ++i) {
        r[i] = T(f(x[i]));
    }
    Put(dst, r);
}

template<typename T, typename F>
inline void Map2(Slot& dst, const Slot& a, const Slot& b, F f) {
    const auto x = Get<T>(a);
    const auto y = Get<T>(b);
    Lanes<T> r;
    for (size_t i = 0; i < r.size(); ++i) {
        r[i] = T(f(x[i], y[i]));
    }
    Put(dst, r);
}

/// Lane-wise operation that reports saturation through the return value.
template<typename T, typename F>
inline bool MapSat2(Slot& dst, const Slot& a, const Slot& b, F f) {
    const auto x = Get<T>(a);
    const auto y = Get<T>(b);
    Lanes<T> r;
    bool sat = false;
    for (size_t i = 0; i < r.size(); ++i) {
        r[i] = T(f(x[i], y[i], sat));
    }
    Put(dst, r);
    return sat;
}

template<typename T, typename F>
inline bool MapSat1(Slot& dst, const Slot& a, F f) {
    const auto x = Get<T>(a);
    Lanes<T> r;
    bool sat = false;
    for (size_t i = 0; i < r.size(); ++i) {
        r[i] = T(f(x[i], sat));
    }
    Put(dst, r);
    return sat;
}

template<typename T>
inline T Ones(bool condition) {
    return condition ? T(~T(0)) : T(0);
}

/// Register-specified shift: the low byte of each element is a signed shift amount.
template<typename T>
inline T VShift(T x, T y) {
    const s8 shift = s8(u8(y));
    constexpr s64 bits = s64(sizeof(T) * 8);
    if constexpr (std::is_signed_v<T>) {
        if (shift >= bits) {
            return 0;
        }
        if (shift <= -bits) {
            return T(x >> (bits - 1));
        }
    } else if (shift <= -bits || shift >= bits) {
        return 0;
    }
    if (shift < 0) {
        return T(x >> -shift);
    }
    using U = std::make_unsigned_t<T>;
    return T(U(x) << shift);
}

template<typename T>
inline T RoundingShiftLeft(T x, T y) {
    using S = std::make_signed_t<T>;
    using U = std::make_unsigned_t<T>;
    constexpr s64 bits = s64(sizeof(T) * 8);
    const s64 shift = s64(s8(u8(y)));
    if (shift >= 0) {
        return shift >= bits ? T(0) : T(U(x) << shift);
    }
    if ((std::is_unsigned_v<T> && shift < -bits) || (std::is_signed_v<T> && shift <= -bits)) {
        return 0;
    }
    const s64 shift_value = -shift - 1;
    const T shifted = T((x & T(S(1) << shift_value)) >> shift_value);
    if (shift == -bits) {
        return shifted;
    }
    return T((x >> -shift) + shifted);
}

template<typename T>
inline T SignedSaturatedShiftLeft(T element, T shift_value, bool& sat) {
    using U = std::make_unsigned_t<T>;
    constexpr size_t bits_minus_one = sizeof(T) * 8 - 1;
    const auto saturate = [](T value) {
        return T((U(value) >> bits_minus_one) + (U(1) << bits_minus_one) - 1);
    };
    const T shift = std::clamp<T>(T(s8(u8(shift_value))), -T(bits_minus_one), std::numeric_limits<T>::max());
    if (element == 0) {
        return 0;
    }
    if (shift < 0) {
        return T(element >> -shift);
    }
    if (U(shift) > bits_minus_one) {
        sat = true;
        return saturate(element);
    }
    const T shifted = T(U(element) << shift);
    if ((shifted >> shift) != element) {
        sat = true;
        return saturate(element);
    }
    return shifted;
}

template<typename T>
inline T UnsignedSaturatedShiftLeft(T element, T shift_value, bool& sat) {
    using S = std::make_signed_t<T>;
    constexpr S bits = S(sizeof(T) * 8);
    const S shift = std::clamp<S>(S(s8(u8(shift_value))), S(-bits), std::numeric_limits<S>::max());
    if (element == 0 || shift <= -bits) {
        return 0;
    }
    if (shift < 0) {
        return T(element >> -shift);
    }
    if (shift >= bits) {
        sat = true;
        return std::numeric_limits<T>::max();
    }
    const T shifted = T(element << shift);
    if (T(shifted >> shift) != element) {
        sat = true;
        return std::numeric_limits<T>::max();
    }
    return shifted;
}

template<typename T>
inline T PolynomialMultiply(T a, T b) {
    T result = 0;
    for (size_t i = 0; i < sizeof(T) * 8; ++i) {
        if ((a >> i) & 1) {
            result ^= T(b << i);
        }
    }
    return result;
}

/// Saturates a wide value into the range of Narrow.
template<typename Narrow, typename Wide>
inline Narrow SaturateTo(Wide value, bool& sat) {
    constexpr Wide min = Wide(std::numeric_limits<Narrow>::min());
    constexpr Wide max = Wide(std::numeric_limits<Narrow>::max());
    if (value < min) {
        sat = true;
        return Narrow(min);
    }
    if (value > max) {
        sat = true;
        return Narrow(max);
    }
    return Narrow(value);
}

}  // namespace Dynarmic::Backend::Interp
