// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Floating-point helpers for the interpreter. Arithmetic runs on the host FPU (IEEE round to
// nearest); ARM-specific behaviour (NaN propagation, default NaN, flush-to-zero) is applied
// around it. Half precision and the estimate instructions use the soft-float library.

#pragma once

#include <bit>
#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>

#include "dynarmic/backend/interp/interp_exec.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/fp/fpsr.h"
#include "dynarmic/common/fp/info.h"
#include "dynarmic/common/fp/op.h"
#include "dynarmic/common/fp/process_nan.h"
#include "dynarmic/common/fp/rounding_mode.h"
#include "dynarmic/common/fp/unpacked.h"

namespace Dynarmic::Backend::Interp {

template<typename FPT>
using HostFloat = std::conditional_t<sizeof(FPT) == 4, float, double>;

template<typename FPT>
inline HostFloat<FPT> ToHost(FPT bits) {
    return std::bit_cast<HostFloat<FPT>>(bits);
}

template<typename FPT>
inline FPT FromHost(HostFloat<FPT> value) {
    return std::bit_cast<FPT>(value);
}

template<typename FPT>
inline bool IsNaN(FPT bits) {
    using Info = FP::FPInfo<FPT>;
    return (bits & Info::exponent_mask) == Info::exponent_mask && (bits & Info::mantissa_mask) != 0;
}

template<typename FPT>
inline bool IsSNaN(FPT bits) {
    using Info = FP::FPInfo<FPT>;
    return IsNaN(bits) && (bits & Info::mantissa_msb) == 0;
}

template<typename FPT>
inline bool IsDenormal(FPT bits) {
    using Info = FP::FPInfo<FPT>;
    return (bits & Info::exponent_mask) == 0 && (bits & Info::mantissa_mask) != 0;
}

template<typename FPT>
inline FP::FPType TypeOf(FPT bits) {
    using Info = FP::FPInfo<FPT>;
    if ((bits & Info::exponent_mask) == Info::exponent_mask) {
        if ((bits & Info::mantissa_mask) == 0) {
            return FP::FPType::Infinity;
        }
        return (bits & Info::mantissa_msb) ? FP::FPType::QNaN : FP::FPType::SNaN;
    }
    return (bits & ~FPT(Info::sign_mask)) == 0 ? FP::FPType::Zero : FP::FPType::Nonzero;
}

/// Floating-point environment for one operation.
struct FPEnv {
    FP::FPCR fpcr;
    FP::FPSR fpsr;
    bool inaccurate_nan;
    bool unfuse_fma;
};

template<typename FPT>
inline FPT FlushInput(FPT bits, FPEnv& env) {
    if (env.fpcr.FZ() && IsDenormal(bits)) {
        env.fpsr.IDC(true);
        return FPT(bits & FP::FPInfo<FPT>::sign_mask);
    }
    return bits;
}

template<typename FPT>
inline FPT FixResult(FPT bits, FPEnv& env) {
    using Info = FP::FPInfo<FPT>;
    if (IsNaN(bits)) {
        // The host only produces NaN here for invalid operations.
        env.fpsr.IOC(true);
        return Info::DefaultNaN();
    }
    if (env.fpcr.FZ() && IsDenormal(bits)) {
        env.fpsr.UFC(true);
        return FPT(bits & Info::sign_mask);
    }
    return bits;
}

template<typename FPT>
inline std::optional<FPT> ProcessNaNs(FPT a, FPT b, FPEnv& env) {
    if (!IsNaN(a) && !IsNaN(b)) {
        return std::nullopt;
    }
    if (env.inaccurate_nan) {
        return FP::FPInfo<FPT>::DefaultNaN();
    }
    return FP::FPProcessNaNs(TypeOf(a), TypeOf(b), a, b, env.fpcr, env.fpsr);
}

enum class HostOp { Add, Sub, Mul, Div, Other };

/// Moves a round-to-nearest result to the guest rounding mode. `error` has the sign of
/// (exact result - rounded result) and is zero when the result is exact.
template<typename F>
inline F ApplyRounding(F rounded, F error, FP::RoundingMode mode, bool inputs_finite) {
    constexpr F max = std::numeric_limits<F>::max();
    constexpr F inf = std::numeric_limits<F>::infinity();
    if (std::isinf(rounded) && !inputs_finite) {
        // Exact infinity (infinite operand or division by zero).
        return rounded;
    }
    if (std::isinf(rounded)) {
        // Overflow: directed modes saturate to the largest finite value on one side.
        if (mode == FP::RoundingMode::TowardsZero) {
            return std::copysign(max, rounded);
        }
        if (mode == FP::RoundingMode::TowardsPlusInfinity && rounded < 0) {
            return -max;
        }
        if (mode == FP::RoundingMode::TowardsMinusInfinity && rounded > 0) {
            return max;
        }
        return rounded;
    }
    if (error == 0 || std::isnan(error)) {
        return rounded;
    }
    switch (mode) {
    case FP::RoundingMode::TowardsPlusInfinity:
        return error > 0 ? std::nextafter(rounded, inf) : rounded;
    case FP::RoundingMode::TowardsMinusInfinity:
        return error < 0 ? std::nextafter(rounded, -inf) : rounded;
    case FP::RoundingMode::TowardsZero:
        if (rounded > 0 && error < 0) {
            return std::nextafter(rounded, F(0));
        }
        if (rounded < 0 && error > 0) {
            return std::nextafter(rounded, F(0));
        }
        return rounded;
    default:
        return rounded;
    }
}

/// Basic arithmetic honouring the FPCR rounding mode. The host rounds to nearest; for the other
/// modes the exact error of that result decides whether to step one ulp.
template<typename F>
inline F RoundedArith(HostOp op, F a, F b, FP::RoundingMode mode) {
    F q;
    switch (op) {
    case HostOp::Add:
        q = a + b;
        break;
    case HostOp::Sub:
        q = a - b;
        break;
    case HostOp::Mul:
        q = a * b;
        break;
    case HostOp::Div:
        q = a / b;
        break;
    default:
        return F(0);
    }
    if (mode == FP::RoundingMode::ToNearest_TieEven) {
        return q;
    }
    F error = 0;
    switch (op) {
    case HostOp::Add:
    case HostOp::Sub: {
        const F y = op == HostOp::Sub ? -b : b;
        if (q == 0 && mode == FP::RoundingMode::TowardsMinusInfinity) {
            // An exact zero sum is -0 when rounding down, unless both operands are +0.
            return (a == 0 && y == 0 && !std::signbit(a) && !std::signbit(y)) ? q : -F(0);
        }
        // TwoSum: exact error of a floating-point addition.
        const auto two_sum_error = [](auto x, auto z, auto sum) {
            const auto bb = sum - x;
            return (x - (sum - bb)) + (z - bb);
        };
        if constexpr (std::is_same_v<F, float>) {
            // Done in double so intermediates near FLT_MAX cannot overflow.
            const double sum = double(a) + double(y);
            const double diff = sum - double(q);
            const double err = diff != 0 ? diff : two_sum_error(double(a), double(y), sum);
            error = err == 0 ? F(0) : (err > 0 ? F(1) : F(-1));
        } else {
            error = two_sum_error(a, y, q);
            if (std::isfinite(q) && !std::isfinite(error)) {
                // Intermediates overflowed; both operands are large, so halving them is exact.
                error = two_sum_error(a * F(0.5), y * F(0.5), q * F(0.5));
            }
        }
        break;
    }
    case HostOp::Mul:
        if constexpr (std::is_same_v<F, float>) {
            // The single-precision product is exact in double precision.
            error = F(double(a) * double(b) - double(q));
            if (error == 0 && double(a) * double(b) != double(q)) {
                error = double(a) * double(b) > double(q) ? F(1) : F(-1);
            }
        } else {
            error = std::fma(a, b, -q);
            if (std::abs(q) < std::numeric_limits<F>::min() && a != 0 && b != 0 && std::isfinite(a) && std::isfinite(b)) {
                // Subnormal result: the error underflows, so compare against the normalized
                // product a*b = ma*mb*2^e instead (scaling q up is exact).
                int ea, eb;
                const F ma = std::frexp(a, &ea);
                const F mb = std::frexp(b, &eb);
                const int e = ea + eb;
                if (e < -1100) {
                    // |a*b| < 2^-1100: below half the smallest subnormal.
                    error = q != 0 ? -q : ma * mb;
                } else {
                    error = std::fma(ma, mb, -std::ldexp(q, -e));
                }
            }
        }
        break;
    case HostOp::Div: {
        if constexpr (std::is_same_v<F, float>) {
            const double r = std::fma(-double(q), double(b), double(a));
            error = (r == 0) ? F(0) : ((r > 0) == (b > 0) ? F(1) : F(-1));
        } else {
            F r = std::fma(-q, b, a);
            if (std::abs(q) < std::numeric_limits<F>::min() && a != 0 && std::isfinite(a) && std::isfinite(b) && b != 0) {
                // Subnormal quotient: |a| < 4 here, so scaling a and q keeps the remainder exact.
                constexpr F scale = 0x1p200;
                r = std::fma(-q * scale, b, a * scale);
            }
            error = (r == 0) ? F(0) : ((r > 0) == (b > 0) ? F(1) : F(-1));
        }
        break;
    }
    default:
        break;
    }
    const bool finite = std::isfinite(a) && std::isfinite(b) && !(op == HostOp::Div && b == 0);
    return ApplyRounding(q, error, mode, finite);
}

template<typename FPT>
inline FPT Binary(FPT a, FPT b, FPEnv& env, HostOp kind) {
    a = FlushInput(a, env);
    b = FlushInput(b, env);
    if (auto nan = ProcessNaNs(a, b, env)) {
        return *nan;
    }
    return FixResult<FPT>(FromHost<FPT>(RoundedArith(kind, ToHost(a), ToHost(b), env.fpcr.RMode())), env);
}

template<typename FPT>
inline FPT Sqrt(FPT a, FPEnv& env) {
    a = FlushInput(a, env);
    if (IsNaN(a)) {
        return env.inaccurate_nan ? FP::FPInfo<FPT>::DefaultNaN() : FP::FPProcessNaN(TypeOf(a), a, env.fpcr, env.fpsr);
    }
    const auto x = ToHost(a);
    auto q = std::sqrt(x);
    const auto mode = env.fpcr.RMode();
    if (mode != FP::RoundingMode::ToNearest_TieEven && std::isfinite(q) && q > 0) {
        // The residual x - q*q is exact unless it underflows; avoid that with wider or scaled math.
        decltype(q) residual;
        if constexpr (std::is_same_v<decltype(q), float>) {
            const double r = std::fma(-double(q), double(q), double(x));
            residual = r == 0 ? 0.0f : (r > 0 ? 1.0f : -1.0f);
        } else if (x < 0x1p-800) {
            residual = std::fma(-q * 0x1p500, q * 0x1p500, x * 0x1p1000);
        } else {
            residual = std::fma(-q, q, x);
        }
        q = ApplyRounding(q, residual, mode, true);
    }
    return FixResult<FPT>(FromHost<FPT>(q), env);
}

template<typename FPT>
inline FPT MulAdd(FPT addend, FPT a, FPT b, FPEnv& env) {
    if constexpr (sizeof(FPT) == 2) {
        return FP::FPMulAdd<FPT>(addend, a, b, env.fpcr, env.fpsr);
    } else {
        addend = FlushInput(addend, env);
        a = FlushInput(a, env);
        b = FlushInput(b, env);
        if (IsNaN(addend) || IsNaN(a) || IsNaN(b)) {
            if (env.inaccurate_nan) {
                return FP::FPInfo<FPT>::DefaultNaN();
            }
            return FP::FPMulAdd<FPT>(addend, a, b, env.fpcr, env.fpsr);
        }
        if (env.fpcr.RMode() != FP::RoundingMode::ToNearest_TieEven) {
            return FP::FPMulAdd<FPT>(addend, a, b, env.fpcr, env.fpsr);
        }
        const auto x = ToHost(a), y = ToHost(b), z = ToHost(addend);
        const auto result = env.unfuse_fma ? x * y + z : std::fma(x, y, z);
        return FixResult<FPT>(FromHost<FPT>(result), env);
    }
}

template<typename FPT>
inline FPT Neg(FPT a) {
    return FPT(a ^ FP::FPInfo<FPT>::sign_mask);
}

template<typename FPT>
inline FPT Abs(FPT a) {
    return FPT(a & ~FPT(FP::FPInfo<FPT>::sign_mask));
}

template<typename FPT>
inline FPT Max(FPT a, FPT b, FPEnv& env) {
    a = FlushInput(a, env);
    b = FlushInput(b, env);
    if (auto nan = ProcessNaNs(a, b, env)) {
        return *nan;
    }
    const auto x = ToHost(a), y = ToHost(b);
    if (x == y) {
        // max(+0, -0) is +0.
        return FPT(a & b);
    }
    return x > y ? a : b;
}

template<typename FPT>
inline FPT Min(FPT a, FPT b, FPEnv& env) {
    a = FlushInput(a, env);
    b = FlushInput(b, env);
    if (auto nan = ProcessNaNs(a, b, env)) {
        return *nan;
    }
    const auto x = ToHost(a), y = ToHost(b);
    if (x == y) {
        // min(+0, -0) is -0.
        return FPT(a | b);
    }
    return x < y ? a : b;
}

/// FMAXNM/FMINNM: a single quiet NaN operand is treated as missing data.
template<typename FPT>
inline FPT MaxNumeric(FPT a, FPT b, FPEnv& env) {
    using Info = FP::FPInfo<FPT>;
    const FP::FPType ta = TypeOf(a), tb = TypeOf(b);
    if (ta == FP::FPType::QNaN && tb != FP::FPType::QNaN && tb != FP::FPType::SNaN) {
        a = Info::Infinity(true);
    } else if (tb == FP::FPType::QNaN && ta != FP::FPType::QNaN && ta != FP::FPType::SNaN) {
        b = Info::Infinity(true);
    }
    return Max(a, b, env);
}

template<typename FPT>
inline FPT MinNumeric(FPT a, FPT b, FPEnv& env) {
    using Info = FP::FPInfo<FPT>;
    const FP::FPType ta = TypeOf(a), tb = TypeOf(b);
    if (ta == FP::FPType::QNaN && tb != FP::FPType::QNaN && tb != FP::FPType::SNaN) {
        a = Info::Infinity(false);
    } else if (tb == FP::FPType::QNaN && ta != FP::FPType::QNaN && ta != FP::FPType::SNaN) {
        b = Info::Infinity(false);
    }
    return Min(a, b, env);
}

/// FMULX: like FMUL, except that infinity times zero is 2.0 with the product's sign.
template<typename FPT>
inline FPT MulX(FPT a, FPT b, FPEnv& env) {
    using Info = FP::FPInfo<FPT>;
    a = FlushInput(a, env);
    b = FlushInput(b, env);
    if (auto nan = ProcessNaNs(a, b, env)) {
        return *nan;
    }
    const FP::FPType ta = TypeOf(a), tb = TypeOf(b);
    if ((ta == FP::FPType::Infinity && tb == FP::FPType::Zero) || (ta == FP::FPType::Zero && tb == FP::FPType::Infinity)) {
        const bool sign = ((a ^ b) & Info::sign_mask) != 0;
        return FPT(FromHost<FPT>(HostFloat<FPT>(2.0)) | (sign ? Info::sign_mask : 0));
    }
    return FixResult<FPT>(FromHost<FPT>(ToHost(a) * ToHost(b)), env);
}

template<typename FPT>
inline u32 Compare(FPT a, FPT b, bool exc_on_qnan, FPEnv& env) {
    a = FlushInput(a, env);
    b = FlushInput(b, env);
    if (IsNaN(a) || IsNaN(b)) {
        if (exc_on_qnan || IsSNaN(a) || IsSNaN(b)) {
            env.fpsr.IOC(true);
        }
        return 0x30000000;
    }
    const auto x = ToHost(a), y = ToHost(b);
    if (x == y) {
        return 0x60000000;
    }
    return x < y ? 0x80000000 : 0x20000000;
}

template<typename FPT>
inline FPT RoundInt(FPT a, FP::RoundingMode rounding, bool exact, FPEnv& env) {
    if constexpr (sizeof(FPT) == 2) {
        return FPT(FP::FPRoundInt<FPT>(a, env.fpcr, rounding, exact, env.fpsr));
    } else {
        a = FlushInput(a, env);
        if (IsNaN(a)) {
            return env.inaccurate_nan ? FP::FPInfo<FPT>::DefaultNaN() : FP::FPProcessNaN(TypeOf(a), a, env.fpcr, env.fpsr);
        }
        const auto x = ToHost(a);
        HostFloat<FPT> r;
        switch (rounding) {
        case FP::RoundingMode::ToNearest_TieEven:
            r = std::nearbyint(x);
            break;
        case FP::RoundingMode::TowardsPlusInfinity:
            r = std::ceil(x);
            break;
        case FP::RoundingMode::TowardsMinusInfinity:
            r = std::floor(x);
            break;
        case FP::RoundingMode::TowardsZero:
            r = std::trunc(x);
            break;
        case FP::RoundingMode::ToNearest_TieAwayFromZero:
            r = std::round(x);
            break;
        default:
            return FPT(FP::FPRoundInt<FPT>(a, env.fpcr, rounding, exact, env.fpsr));
        }
        if (exact && r != x) {
            env.fpsr.IXC(true);
        }
        return FromHost<FPT>(std::copysign(r, x));
    }
}

/// Float to fixed-point. Saturates, NaN converts to zero.
template<typename FPT>
inline u64 ToFixed(size_t ibits, FPT a, size_t fbits, bool is_unsigned, FP::RoundingMode rounding, FPEnv& env) {
    if constexpr (sizeof(FPT) == 2) {
        return FP::FPToFixed<FPT>(ibits, a, fbits, is_unsigned, env.fpcr, rounding, env.fpsr);
    } else {
        if (rounding == FP::RoundingMode::ToOdd) {
            return FP::FPToFixed<FPT>(ibits, a, fbits, is_unsigned, env.fpcr, rounding, env.fpsr);
        }
        a = FlushInput(a, env);
        if (IsNaN(a)) {
            env.fpsr.IOC(true);
            return 0;
        }
        const double scaled = std::ldexp(double(ToHost(a)), int(fbits));
        double r;
        switch (rounding) {
        case FP::RoundingMode::ToNearest_TieEven:
            r = std::nearbyint(scaled);
            break;
        case FP::RoundingMode::TowardsPlusInfinity:
            r = std::ceil(scaled);
            break;
        case FP::RoundingMode::TowardsMinusInfinity:
            r = std::floor(scaled);
            break;
        case FP::RoundingMode::ToNearest_TieAwayFromZero:
            r = std::round(scaled);
            break;
        default:
            r = std::trunc(scaled);
            break;
        }
        if (is_unsigned) {
            const double max = std::ldexp(1.0, int(ibits));
            if (r < 0) {
                env.fpsr.IOC(true);
                return 0;
            }
            if (r >= max) {
                env.fpsr.IOC(true);
                return ibits == 64 ? ~u64(0) : (u64(1) << ibits) - 1;
            }
            return u64(r);
        }
        const double max = std::ldexp(1.0, int(ibits - 1));
        const u64 mask = ibits == 64 ? ~u64(0) : (u64(1) << ibits) - 1;
        if (r >= max) {
            env.fpsr.IOC(true);
            return u64(std::numeric_limits<s64>::max()) >> (64 - ibits);
        }
        if (r < -max) {
            env.fpsr.IOC(true);
            return u64(std::numeric_limits<s64>::min() >> (64 - ibits)) & mask;
        }
        return u64(s64(r)) & mask;
    }
}

/// Fixed-point (or integer when fbits == 0) to float.
template<typename FPT, typename Int>
inline FPT FromFixed(Int value, size_t fbits, FP::RoundingMode rounding, FPEnv& env) {
    if (rounding != FP::RoundingMode::ToNearest_TieEven && value != 0) {
        // Directed rounding: go through the soft-float rounding with an exact unpacked value.
        const bool sign = std::is_signed_v<Int> && value < 0;
        const u64 magnitude = sign ? u64(0) - u64(value) : u64(value);
        const FP::FPUnpacked unpacked = FP::ToNormalized(sign, -int(fbits), magnitude);
        return FP::FPRound<FPT>(unpacked, env.fpcr, rounding, env.fpsr);
    }
    const HostFloat<FPT> converted = HostFloat<FPT>(value);
    return FixResult<FPT>(FromHost<FPT>(fbits ? std::ldexp(converted, -int(fbits)) : converted), env);
}

template<typename To, typename From>
inline To Convert(From a, FP::RoundingMode rounding, FPEnv& env) {
    if constexpr (sizeof(To) != 2 && sizeof(From) != 2) {
        if (rounding == FP::RoundingMode::ToNearest_TieEven && !IsNaN(a)) {
            a = FlushInput(a, env);
            return FixResult<To>(FromHost<To>(HostFloat<To>(ToHost(a))), env);
        }
    }
    return FP::FPConvert<To, From>(a, env.fpcr, rounding, env.fpsr);
}

}  // namespace Dynarmic::Backend::Interp
