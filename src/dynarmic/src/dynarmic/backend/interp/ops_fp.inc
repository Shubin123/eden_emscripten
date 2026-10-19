// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Scalar and vector floating-point operations. Included inside the interpreter dispatch switch.
// FP_BEGIN/FP_END bracket an operation with its FPCR and the cumulative FPSR flags.

#define FP_BEGIN(controlled) FPEnv env = MakeFPEnv(controlled);
#define FP_END() CommitFPSR(env);

#define FP_UNARY_BITS(name, FPT, fn) \
    case Opcode::name:               \
        DST.lo = fn<FPT>(FPT(ARG(0).lo)); \
        break;
FP_UNARY_BITS(FPAbs16, u16, Abs)
FP_UNARY_BITS(FPAbs32, u32, Abs)
FP_UNARY_BITS(FPAbs64, u64, Abs)
FP_UNARY_BITS(FPNeg16, u16, Neg)
FP_UNARY_BITS(FPNeg32, u32, Neg)
FP_UNARY_BITS(FPNeg64, u64, Neg)
#undef FP_UNARY_BITS

#define FP_BINARY(name, FPT, kind)                                                         \
    case Opcode::name: {                                                                   \
        FP_BEGIN(true)                                                                     \
        DST.lo = Binary<FPT>(FPT(ARG(0).lo), FPT(ARG(1).lo), env, HostOp::kind);           \
        FP_END()                                                                           \
        break;                                                                             \
    }
FP_BINARY(FPAdd32, u32, Add)
FP_BINARY(FPAdd64, u64, Add)
FP_BINARY(FPSub32, u32, Sub)
FP_BINARY(FPSub64, u64, Sub)
FP_BINARY(FPMul32, u32, Mul)
FP_BINARY(FPMul64, u64, Mul)
FP_BINARY(FPDiv32, u32, Div)
FP_BINARY(FPDiv64, u64, Div)
#undef FP_BINARY

#define FP_ENV_BINARY(name, FPT, fn)                                   \
    case Opcode::name: {                                               \
        FP_BEGIN(true)                                                 \
        DST.lo = fn<FPT>(FPT(ARG(0).lo), FPT(ARG(1).lo), env);         \
        FP_END()                                                       \
        break;                                                         \
    }
FP_ENV_BINARY(FPMax32, u32, Max)
FP_ENV_BINARY(FPMax64, u64, Max)
FP_ENV_BINARY(FPMin32, u32, Min)
FP_ENV_BINARY(FPMin64, u64, Min)
FP_ENV_BINARY(FPMaxNumeric32, u32, MaxNumeric)
FP_ENV_BINARY(FPMaxNumeric64, u64, MaxNumeric)
FP_ENV_BINARY(FPMinNumeric32, u32, MinNumeric)
FP_ENV_BINARY(FPMinNumeric64, u64, MinNumeric)
FP_ENV_BINARY(FPMulX32, u32, MulX)
FP_ENV_BINARY(FPMulX64, u64, MulX)
#undef FP_ENV_BINARY

#define FP_SOFT_BINARY(name, FPT, fn)                                                  \
    case Opcode::name: {                                                               \
        FP_BEGIN(true)                                                                 \
        DST.lo = FP::fn<FPT>(FPT(ARG(0).lo), FPT(ARG(1).lo), env.fpcr, env.fpsr);      \
        FP_END()                                                                       \
        break;                                                                         \
    }
FP_SOFT_BINARY(FPRecipStepFused16, u16, FPRecipStepFused)
FP_SOFT_BINARY(FPRecipStepFused32, u32, FPRecipStepFused)
FP_SOFT_BINARY(FPRecipStepFused64, u64, FPRecipStepFused)
FP_SOFT_BINARY(FPRSqrtStepFused16, u16, FPRSqrtStepFused)
FP_SOFT_BINARY(FPRSqrtStepFused32, u32, FPRSqrtStepFused)
FP_SOFT_BINARY(FPRSqrtStepFused64, u64, FPRSqrtStepFused)
#undef FP_SOFT_BINARY

#define FP_SOFT_UNARY(name, FPT, fn)                                   \
    case Opcode::name: {                                               \
        FP_BEGIN(true)                                                 \
        DST.lo = FP::fn<FPT>(FPT(ARG(0).lo), env.fpcr, env.fpsr);      \
        FP_END()                                                       \
        break;                                                         \
    }
FP_SOFT_UNARY(FPRecipEstimate16, u16, FPRecipEstimate)
FP_SOFT_UNARY(FPRecipEstimate32, u32, FPRecipEstimate)
FP_SOFT_UNARY(FPRecipEstimate64, u64, FPRecipEstimate)
FP_SOFT_UNARY(FPRecipExponent16, u16, FPRecipExponent)
FP_SOFT_UNARY(FPRecipExponent32, u32, FPRecipExponent)
FP_SOFT_UNARY(FPRecipExponent64, u64, FPRecipExponent)
FP_SOFT_UNARY(FPRSqrtEstimate16, u16, FPRSqrtEstimate)
FP_SOFT_UNARY(FPRSqrtEstimate32, u32, FPRSqrtEstimate)
FP_SOFT_UNARY(FPRSqrtEstimate64, u64, FPRSqrtEstimate)
#undef FP_SOFT_UNARY

case Opcode::FPCompare32: {
    FP_BEGIN(true)
    DST.lo = Compare<u32>(u32(ARG(0).lo), u32(ARG(1).lo), ARG(2).lo & 1, env);
    FP_END()
    break;
}
case Opcode::FPCompare64: {
    FP_BEGIN(true)
    DST.lo = Compare<u64>(ARG(0).lo, ARG(1).lo, ARG(2).lo & 1, env);
    FP_END()
    break;
}

#define FP_MULADD(name, FPT, negate)                                                           \
    case Opcode::name: {                                                                       \
        FP_BEGIN(true)                                                                         \
        const FPT a = FPT(ARG(1).lo);                                                          \
        DST.lo = MulAdd<FPT>(FPT(ARG(0).lo), negate ? Neg<FPT>(a) : a, FPT(ARG(2).lo), env);   \
        FP_END()                                                                               \
        break;                                                                                 \
    }
FP_MULADD(FPMulAdd16, u16, false)
FP_MULADD(FPMulAdd32, u32, false)
FP_MULADD(FPMulAdd64, u64, false)
FP_MULADD(FPMulSub16, u16, true)
FP_MULADD(FPMulSub32, u32, true)
FP_MULADD(FPMulSub64, u64, true)
#undef FP_MULADD

case Opcode::FPRoundInt16:
case Opcode::FPRoundInt32:
case Opcode::FPRoundInt64: {
    FP_BEGIN(true)
    const auto rounding = FP::RoundingMode(ARG(1).lo);
    const bool exact = ARG(2).lo & 1;
    if (op.opcode == Opcode::FPRoundInt16) {
        DST.lo = RoundInt<u16>(u16(ARG(0).lo), rounding, exact, env);
    } else if (op.opcode == Opcode::FPRoundInt32) {
        DST.lo = RoundInt<u32>(u32(ARG(0).lo), rounding, exact, env);
    } else {
        DST.lo = RoundInt<u64>(ARG(0).lo, rounding, exact, env);
    }
    FP_END()
    break;
}
case Opcode::FPSqrt32: {
    FP_BEGIN(true)
    DST.lo = Sqrt<u32>(u32(ARG(0).lo), env);
    FP_END()
    break;
}
case Opcode::FPSqrt64: {
    FP_BEGIN(true)
    DST.lo = Sqrt<u64>(ARG(0).lo, env);
    FP_END()
    break;
}

#define FP_CONVERT(name, To, From)                                                     \
    case Opcode::name: {                                                               \
        FP_BEGIN(true)                                                                 \
        DST.lo = Convert<To, From>(From(ARG(0).lo), FP::RoundingMode(ARG(1).lo), env); \
        FP_END()                                                                       \
        break;                                                                         \
    }
FP_CONVERT(FPHalfToDouble, u64, u16)
FP_CONVERT(FPHalfToSingle, u32, u16)
FP_CONVERT(FPSingleToDouble, u64, u32)
FP_CONVERT(FPSingleToHalf, u16, u32)
FP_CONVERT(FPDoubleToHalf, u16, u64)
FP_CONVERT(FPDoubleToSingle, u32, u64)
#undef FP_CONVERT

#define FP_TO_FIXED(name, From, ibits, is_unsigned)                                                           \
    case Opcode::name: {                                                                                      \
        FP_BEGIN(true)                                                                                        \
        DST.lo = ToFixed<From>(ibits, From(ARG(0).lo), size_t(ARG(1).lo), is_unsigned, FP::RoundingMode(ARG(2).lo), env); \
        FP_END()                                                                                              \
        break;                                                                                                \
    }
FP_TO_FIXED(FPDoubleToFixedS16, u64, 16, false)
FP_TO_FIXED(FPDoubleToFixedS32, u64, 32, false)
FP_TO_FIXED(FPDoubleToFixedS64, u64, 64, false)
FP_TO_FIXED(FPDoubleToFixedU16, u64, 16, true)
FP_TO_FIXED(FPDoubleToFixedU32, u64, 32, true)
FP_TO_FIXED(FPDoubleToFixedU64, u64, 64, true)
FP_TO_FIXED(FPHalfToFixedS16, u16, 16, false)
FP_TO_FIXED(FPHalfToFixedS32, u16, 32, false)
FP_TO_FIXED(FPHalfToFixedS64, u16, 64, false)
FP_TO_FIXED(FPHalfToFixedU16, u16, 16, true)
FP_TO_FIXED(FPHalfToFixedU32, u16, 32, true)
FP_TO_FIXED(FPHalfToFixedU64, u16, 64, true)
FP_TO_FIXED(FPSingleToFixedS16, u32, 16, false)
FP_TO_FIXED(FPSingleToFixedS32, u32, 32, false)
FP_TO_FIXED(FPSingleToFixedS64, u32, 64, false)
FP_TO_FIXED(FPSingleToFixedU16, u32, 16, true)
FP_TO_FIXED(FPSingleToFixedU32, u32, 32, true)
FP_TO_FIXED(FPSingleToFixedU64, u32, 64, true)
#undef FP_TO_FIXED

#define FP_FROM_FIXED(name, To, Int)                                                                   \
    case Opcode::name: {                                                                               \
        FP_BEGIN(true)                                                                                 \
        DST.lo = FromFixed<To, Int>(Int(ARG(0).lo), size_t(ARG(1).lo), FP::RoundingMode(ARG(2).lo), env); \
        FP_END()                                                                                       \
        break;                                                                                         \
    }
FP_FROM_FIXED(FPFixedU16ToSingle, u32, u16)
FP_FROM_FIXED(FPFixedS16ToSingle, u32, s16)
FP_FROM_FIXED(FPFixedU16ToDouble, u64, u16)
FP_FROM_FIXED(FPFixedS16ToDouble, u64, s16)
FP_FROM_FIXED(FPFixedU32ToSingle, u32, u32)
FP_FROM_FIXED(FPFixedS32ToSingle, u32, s32)
FP_FROM_FIXED(FPFixedU32ToDouble, u64, u32)
FP_FROM_FIXED(FPFixedS32ToDouble, u64, s32)
FP_FROM_FIXED(FPFixedU64ToDouble, u64, u64)
FP_FROM_FIXED(FPFixedU64ToSingle, u32, u64)
FP_FROM_FIXED(FPFixedS64ToDouble, u64, s64)
FP_FROM_FIXED(FPFixedS64ToSingle, u32, s64)
#undef FP_FROM_FIXED

// Vector forms. The last argument selects FPCR (true) or the ASIMD standard value.

#define FPV_UNARY_BITS(name, FPT, fn) \
    case Opcode::name:                \
        Map1<FPT>(DST, ARG(0), [](FPT a) { return fn<FPT>(a); }); \
        break;
FPV_UNARY_BITS(FPVectorAbs16, u16, Abs)
FPV_UNARY_BITS(FPVectorAbs32, u32, Abs)
FPV_UNARY_BITS(FPVectorAbs64, u64, Abs)
FPV_UNARY_BITS(FPVectorNeg16, u16, Neg)
FPV_UNARY_BITS(FPVectorNeg32, u32, Neg)
FPV_UNARY_BITS(FPVectorNeg64, u64, Neg)
#undef FPV_UNARY_BITS

#define FPV_BINARY(name, FPT, kind)                                                            \
    case Opcode::name: {                                                                       \
        FP_BEGIN(ARG(2).lo & 1)                                                                \
        Map2<FPT>(DST, ARG(0), ARG(1), [&env](FPT x, FPT y) {                                  \
            return Binary<FPT>(x, y, env, HostOp::kind);                                       \
        });                                                                                    \
        FP_END()                                                                               \
        break;                                                                                 \
    }
FPV_BINARY(FPVectorAdd32, u32, Add)
FPV_BINARY(FPVectorAdd64, u64, Add)
FPV_BINARY(FPVectorSub32, u32, Sub)
FPV_BINARY(FPVectorSub64, u64, Sub)
FPV_BINARY(FPVectorMul32, u32, Mul)
FPV_BINARY(FPVectorMul64, u64, Mul)
FPV_BINARY(FPVectorDiv32, u32, Div)
FPV_BINARY(FPVectorDiv64, u64, Div)
#undef FPV_BINARY

#define FPV_ENV_BINARY(name, FPT, fn)                                                  \
    case Opcode::name: {                                                               \
        FP_BEGIN(ARG(2).lo & 1)                                                        \
        Map2<FPT>(DST, ARG(0), ARG(1), [&env](FPT x, FPT y) { return fn<FPT>(x, y, env); }); \
        FP_END()                                                                       \
        break;                                                                         \
    }
FPV_ENV_BINARY(FPVectorMax32, u32, Max)
FPV_ENV_BINARY(FPVectorMax64, u64, Max)
FPV_ENV_BINARY(FPVectorMin32, u32, Min)
FPV_ENV_BINARY(FPVectorMin64, u64, Min)
FPV_ENV_BINARY(FPVectorMaxNumeric32, u32, MaxNumeric)
FPV_ENV_BINARY(FPVectorMaxNumeric64, u64, MaxNumeric)
FPV_ENV_BINARY(FPVectorMinNumeric32, u32, MinNumeric)
FPV_ENV_BINARY(FPVectorMinNumeric64, u64, MinNumeric)
FPV_ENV_BINARY(FPVectorMulX32, u32, MulX)
FPV_ENV_BINARY(FPVectorMulX64, u64, MulX)
#undef FPV_ENV_BINARY

#define FPV_SOFT_BINARY(name, FPT, fn)                                                 \
    case Opcode::name: {                                                               \
        FP_BEGIN(ARG(2).lo & 1)                                                        \
        Map2<FPT>(DST, ARG(0), ARG(1), [&env](FPT x, FPT y) { return FP::fn<FPT>(x, y, env.fpcr, env.fpsr); }); \
        FP_END()                                                                       \
        break;                                                                         \
    }
FPV_SOFT_BINARY(FPVectorRecipStepFused16, u16, FPRecipStepFused)
FPV_SOFT_BINARY(FPVectorRecipStepFused32, u32, FPRecipStepFused)
FPV_SOFT_BINARY(FPVectorRecipStepFused64, u64, FPRecipStepFused)
FPV_SOFT_BINARY(FPVectorRSqrtStepFused16, u16, FPRSqrtStepFused)
FPV_SOFT_BINARY(FPVectorRSqrtStepFused32, u32, FPRSqrtStepFused)
FPV_SOFT_BINARY(FPVectorRSqrtStepFused64, u64, FPRSqrtStepFused)
#undef FPV_SOFT_BINARY

#define FPV_SOFT_UNARY(name, FPT, fn)                                                  \
    case Opcode::name: {                                                               \
        FP_BEGIN(ARG(1).lo & 1)                                                        \
        Map1<FPT>(DST, ARG(0), [&env](FPT x) { return FP::fn<FPT>(x, env.fpcr, env.fpsr); }); \
        FP_END()                                                                       \
        break;                                                                         \
    }
FPV_SOFT_UNARY(FPVectorRecipEstimate16, u16, FPRecipEstimate)
FPV_SOFT_UNARY(FPVectorRecipEstimate32, u32, FPRecipEstimate)
FPV_SOFT_UNARY(FPVectorRecipEstimate64, u64, FPRecipEstimate)
FPV_SOFT_UNARY(FPVectorRSqrtEstimate16, u16, FPRSqrtEstimate)
FPV_SOFT_UNARY(FPVectorRSqrtEstimate32, u32, FPRSqrtEstimate)
FPV_SOFT_UNARY(FPVectorRSqrtEstimate64, u64, FPRSqrtEstimate)
#undef FPV_SOFT_UNARY

case Opcode::FPVectorSqrt32: {
    FP_BEGIN(ARG(1).lo & 1)
    Map1<u32>(DST, ARG(0), [&env](u32 x) { return Sqrt<u32>(x, env); });
    FP_END()
    break;
}
case Opcode::FPVectorSqrt64: {
    FP_BEGIN(ARG(1).lo & 1)
    Map1<u64>(DST, ARG(0), [&env](u64 x) { return Sqrt<u64>(x, env); });
    FP_END()
    break;
}

#define FPV_COMPARE(name, FPT, cond)                                                   \
    case Opcode::name: {                                                               \
        FP_BEGIN(ARG(2).lo & 1)                                                        \
        Map2<FPT>(DST, ARG(0), ARG(1), [&env](FPT x, FPT y) {                          \
            const u32 nzcv = Compare<FPT>(x, y, false, env);                           \
            return Ones<FPT>(cond);                                                    \
        });                                                                            \
        FP_END()                                                                       \
        break;                                                                         \
    }
FPV_COMPARE(FPVectorEqual32, u32, nzcv == 0x60000000)
FPV_COMPARE(FPVectorEqual64, u64, nzcv == 0x60000000)
FPV_COMPARE(FPVectorGreater32, u32, nzcv == 0x20000000)
FPV_COMPARE(FPVectorGreater64, u64, nzcv == 0x20000000)
FPV_COMPARE(FPVectorGreaterEqual32, u32, nzcv == 0x20000000 || nzcv == 0x60000000)
FPV_COMPARE(FPVectorGreaterEqual64, u64, nzcv == 0x20000000 || nzcv == 0x60000000)
#undef FPV_COMPARE
case Opcode::FPVectorEqual16: {
    FP_BEGIN(ARG(2).lo & 1)
    Map2<u16>(DST, ARG(0), ARG(1), [&env](u16 x, u16 y) { return Ones<u16>(FP::FPCompareEQ<u16>(x, y, env.fpcr, env.fpsr)); });
    FP_END()
    break;
}

#define FPV_MULADD(name, FPT)                                                          \
    case Opcode::name: {                                                               \
        FP_BEGIN(ARG(3).lo & 1)                                                        \
        const auto addend = Get<FPT>(ARG(0));                                          \
        const auto x = Get<FPT>(ARG(1));                                               \
        const auto y = Get<FPT>(ARG(2));                                               \
        Lanes<FPT> r;                                                                  \
        for (size_t i = 0; i < r.size(); ++i) {                                        \
            r[i] = MulAdd<FPT>(addend[i], x[i], y[i], env);                            \
        }                                                                              \
        Put(DST, r);                                                                   \
        FP_END()                                                                       \
        break;                                                                         \
    }
FPV_MULADD(FPVectorMulAdd16, u16)
FPV_MULADD(FPVectorMulAdd32, u32)
FPV_MULADD(FPVectorMulAdd64, u64)
#undef FPV_MULADD

#define FPV_PAIRED_ADD(name, FPT, lower)                                               \
    case Opcode::name: {                                                               \
        FP_BEGIN(ARG(2).lo & 1)                                                        \
        const auto x = Get<FPT>(ARG(0));                                               \
        const auto y = Get<FPT>(ARG(1));                                               \
        constexpr size_t n = 16 / sizeof(FPT);                                         \
        const size_t used = lower ? n / 2 : n;                                         \
        std::array<FPT, 2 * n> concat{};                                               \
        for (size_t i = 0; i < used; ++i) {                                            \
            concat[i] = x[i];                                                          \
            concat[used + i] = y[i];                                                   \
        }                                                                              \
        Lanes<FPT> r{};                                                                \
        for (size_t i = 0; i < used; ++i) {                                            \
            r[i] = Binary<FPT>(concat[2 * i], concat[2 * i + 1], env, HostOp::Add); \
        }                                                                              \
        Put(DST, r);                                                                   \
        FP_END()                                                                       \
        break;                                                                         \
    }
FPV_PAIRED_ADD(FPVectorPairedAdd32, u32, false)
FPV_PAIRED_ADD(FPVectorPairedAdd64, u64, false)
FPV_PAIRED_ADD(FPVectorPairedAddLower32, u32, true)
FPV_PAIRED_ADD(FPVectorPairedAddLower64, u64, true)
#undef FPV_PAIRED_ADD

#define FPV_ROUND_INT(name, FPT)                                                       \
    case Opcode::name: {                                                               \
        FP_BEGIN(ARG(3).lo & 1)                                                        \
        const auto rounding = FP::RoundingMode(ARG(1).lo);                             \
        const bool exact = ARG(2).lo & 1;                                              \
        Map1<FPT>(DST, ARG(0), [&](FPT x) { return RoundInt<FPT>(x, rounding, exact, env); }); \
        FP_END()                                                                       \
        break;                                                                         \
    }
FPV_ROUND_INT(FPVectorRoundInt16, u16)
FPV_ROUND_INT(FPVectorRoundInt32, u32)
FPV_ROUND_INT(FPVectorRoundInt64, u64)
#undef FPV_ROUND_INT

case Opcode::FPVectorFromHalf32: {
    FP_BEGIN(ARG(2).lo & 1)
    const auto rounding = FP::RoundingMode(ARG(1).lo);
    const auto x = Get<u16>(ARG(0));
    Lanes<u32> r;
    for (size_t i = 0; i < r.size(); ++i) {
        r[i] = Convert<u32, u16>(x[i], rounding, env);
    }
    Put(DST, r);
    FP_END()
    break;
}
case Opcode::FPVectorToHalf32: {
    FP_BEGIN(ARG(2).lo & 1)
    const auto rounding = FP::RoundingMode(ARG(1).lo);
    const auto x = Get<u32>(ARG(0));
    Lanes<u16> r{};
    for (size_t i = 0; i < x.size(); ++i) {
        r[i] = Convert<u16, u32>(x[i], rounding, env);
    }
    Put(DST, r);
    FP_END()
    break;
}

#define FPV_FROM_FIXED(name, FPT, Int)                                                 \
    case Opcode::name: {                                                               \
        FP_BEGIN(ARG(3).lo & 1)                                                        \
        const size_t fbits = size_t(ARG(1).lo);                                        \
        const auto rounding = FP::RoundingMode(ARG(2).lo);                             \
        const auto x = Get<Int>(ARG(0));                                               \
        Lanes<FPT> r;                                                                  \
        for (size_t i = 0; i < r.size(); ++i) {                                        \
            r[i] = FromFixed<FPT, Int>(x[i], fbits, rounding, env);                    \
        }                                                                              \
        Put(DST, r);                                                                   \
        FP_END()                                                                       \
        break;                                                                         \
    }
FPV_FROM_FIXED(FPVectorFromSignedFixed32, u32, s32)
FPV_FROM_FIXED(FPVectorFromSignedFixed64, u64, s64)
FPV_FROM_FIXED(FPVectorFromUnsignedFixed32, u32, u32)
FPV_FROM_FIXED(FPVectorFromUnsignedFixed64, u64, u64)
#undef FPV_FROM_FIXED

#define FPV_TO_FIXED(name, FPT, is_unsigned)                                           \
    case Opcode::name: {                                                               \
        FP_BEGIN(ARG(3).lo & 1)                                                        \
        const size_t fbits = size_t(ARG(1).lo);                                        \
        const auto rounding = FP::RoundingMode(ARG(2).lo);                             \
        Map1<FPT>(DST, ARG(0), [&](FPT x) {                                            \
            return FPT(ToFixed<FPT>(sizeof(FPT) * 8, x, fbits, is_unsigned, rounding, env)); \
        });                                                                            \
        FP_END()                                                                       \
        break;                                                                         \
    }
FPV_TO_FIXED(FPVectorToSignedFixed16, u16, false)
FPV_TO_FIXED(FPVectorToSignedFixed32, u32, false)
FPV_TO_FIXED(FPVectorToSignedFixed64, u64, false)
FPV_TO_FIXED(FPVectorToUnsignedFixed16, u16, true)
FPV_TO_FIXED(FPVectorToUnsignedFixed32, u32, true)
FPV_TO_FIXED(FPVectorToUnsignedFixed64, u64, true)
#undef FPV_TO_FIXED

#undef FP_BEGIN
#undef FP_END
