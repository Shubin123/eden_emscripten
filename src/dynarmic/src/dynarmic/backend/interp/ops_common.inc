// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Scalar integer, flag and crypto operations shared by all frontends.
// Included inside the interpreter dispatch switch; see interp_exec.h for the helper macros.

case Opcode::Void:
case Opcode::Breakpoint:
case Opcode::PushRSB:
    break;
case Opcode::Identity:
    DST = ARG(0);
    break;
case Opcode::CallHostFunction: {
    using Fn = void (*)(u64, u64, u64);
    reinterpret_cast<Fn>(ARG(0).lo)(ARG(1).lo, ARG(2).lo, ARG(3).lo);
    break;
}

case Opcode::GetCarryFromOp:
    DST.lo = ARG(0).carry;
    break;
case Opcode::GetOverflowFromOp:
    DST.lo = ARG(0).overflow;
    break;
case Opcode::GetNZCVFromOp: {
    const Slot& parent = ARG(0);
    DST.lo = NZFromResult(parent.lo, op.width) | (u32(parent.carry) << 29) | (u32(parent.overflow) << 28);
    break;
}
case Opcode::GetNZFromOp:
    DST.lo = NZFromResult(ARG(0).lo, op.width);
    break;
case Opcode::GetUpperFromOp:
    DST.lo = ARG(0).upper_lo;
    DST.hi = ARG(0).upper_hi;
    break;
case Opcode::GetLowerFromOp:
    DST.lo = ARG(0).lo;
    DST.hi = ARG(0).hi;
    break;
case Opcode::GetCFlagFromNZCV:
    DST.lo = (ARG(0).lo >> 29) & 1;
    break;
case Opcode::NZCVFromPackedFlags:
    DST.lo = ARG(0).lo & 0xF0000000;
    break;

case Opcode::Pack2x32To1x64:
    DST.lo = (ARG(0).lo & 0xFFFFFFFF) | (ARG(1).lo << 32);
    break;
case Opcode::Pack2x64To1x128:
    DST.lo = ARG(0).lo;
    DST.hi = ARG(1).lo;
    break;
case Opcode::LeastSignificantWord:
    DST.lo = u32(ARG(0).lo);
    break;
case Opcode::LeastSignificantHalf:
    DST.lo = u16(ARG(0).lo);
    break;
case Opcode::LeastSignificantByte:
    DST.lo = u8(ARG(0).lo);
    break;
case Opcode::MostSignificantWord:
    DST.lo = ARG(0).lo >> 32;
    DST.carry = (ARG(0).lo >> 31) & 1;
    break;
case Opcode::MostSignificantBit:
    DST.lo = (ARG(0).lo >> 31) & 1;
    break;
case Opcode::IsZero32:
    DST.lo = u32(ARG(0).lo) == 0;
    break;
case Opcode::IsZero64:
    DST.lo = ARG(0).lo == 0;
    break;
case Opcode::TestBit:
    DST.lo = (ARG(0).lo >> (ARG(1).lo & 63)) & 1;
    break;
case Opcode::ConditionalSelect32:
case Opcode::ConditionalSelect64:
case Opcode::ConditionalSelectNZCV:
    DST.lo = CondHolds(IR::Cond(ARG(0).lo), NZCV()) ? ARG(1).lo : ARG(2).lo;
    break;

case Opcode::LogicalShiftLeft32: {
    const u32 value = u32(ARG(0).lo);
    const u32 shift = u8(ARG(1).lo);
    if (shift == 0) {
        DST.lo = value;
        DST.carry = u8(ARG(2).lo);
    } else if (shift < 32) {
        DST.lo = u32(value << shift);
        DST.carry = (value >> (32 - shift)) & 1;
    } else {
        DST.lo = 0;
        DST.carry = shift == 32 ? (value & 1) : 0;
    }
    break;
}
case Opcode::LogicalShiftLeft64: {
    const u32 shift = u8(ARG(1).lo);
    DST.lo = shift < 64 ? ARG(0).lo << shift : 0;
    break;
}
case Opcode::LogicalShiftRight32: {
    const u32 value = u32(ARG(0).lo);
    const u32 shift = u8(ARG(1).lo);
    if (shift == 0) {
        DST.lo = value;
        DST.carry = u8(ARG(2).lo);
    } else if (shift < 32) {
        DST.lo = value >> shift;
        DST.carry = (value >> (shift - 1)) & 1;
    } else {
        DST.lo = 0;
        DST.carry = shift == 32 ? (value >> 31) : 0;
    }
    break;
}
case Opcode::LogicalShiftRight64: {
    const u32 shift = u8(ARG(1).lo);
    DST.lo = shift < 64 ? ARG(0).lo >> shift : 0;
    break;
}
case Opcode::ArithmeticShiftRight32: {
    const s32 value = s32(u32(ARG(0).lo));
    const u32 shift = u8(ARG(1).lo);
    if (shift == 0) {
        DST.lo = u32(value);
        DST.carry = u8(ARG(2).lo);
    } else if (shift < 32) {
        DST.lo = u32(value >> shift);
        DST.carry = (u32(value) >> (shift - 1)) & 1;
    } else {
        DST.lo = u32(value >> 31);
        DST.carry = u32(value) >> 31;
    }
    break;
}
case Opcode::ArithmeticShiftRight64: {
    const u32 shift = u8(ARG(1).lo);
    DST.lo = u64(s64(ARG(0).lo) >> (shift < 64 ? shift : 63));
    break;
}
case Opcode::BitRotateRight32: {
    const u32 value = u32(ARG(0).lo);
    const u32 shift = u8(ARG(1).lo);
    if (shift == 0) {
        DST.lo = value;
        DST.carry = u8(ARG(2).lo);
    } else {
        const u32 result = std::rotr(value, int(shift & 31));
        DST.lo = result;
        DST.carry = result >> 31;
    }
    break;
}
case Opcode::BitRotateRight64:
    DST.lo = std::rotr(ARG(0).lo, int(ARG(1).lo & 63));
    break;
case Opcode::RotateRightExtended: {
    const u32 value = u32(ARG(0).lo);
    DST.lo = (value >> 1) | (u32(ARG(1).lo & 1) << 31);
    DST.carry = value & 1;
    break;
}
case Opcode::LogicalShiftLeftMasked32:
    DST.lo = u32(u32(ARG(0).lo) << (ARG(1).lo & 31));
    break;
case Opcode::LogicalShiftLeftMasked64:
    DST.lo = ARG(0).lo << (ARG(1).lo & 63);
    break;
case Opcode::LogicalShiftRightMasked32:
    DST.lo = u32(ARG(0).lo) >> (ARG(1).lo & 31);
    break;
case Opcode::LogicalShiftRightMasked64:
    DST.lo = ARG(0).lo >> (ARG(1).lo & 63);
    break;
case Opcode::ArithmeticShiftRightMasked32:
    DST.lo = u32(s32(u32(ARG(0).lo)) >> (ARG(1).lo & 31));
    break;
case Opcode::ArithmeticShiftRightMasked64:
    DST.lo = u64(s64(ARG(0).lo) >> (ARG(1).lo & 63));
    break;
case Opcode::RotateRightMasked32:
    DST.lo = std::rotr(u32(ARG(0).lo), int(ARG(1).lo & 31));
    break;
case Opcode::RotateRightMasked64:
    DST.lo = std::rotr(ARG(0).lo, int(ARG(1).lo & 63));
    break;

case Opcode::Add32: {
    const u32 a = u32(ARG(0).lo), b = u32(ARG(1).lo);
    const u64 wide = u64(a) + u64(b) + (ARG(2).lo & 1);
    const u32 result = u32(wide);
    DST.lo = result;
    DST.carry = u8(wide >> 32);
    DST.overflow = u8((~(a ^ b) & (a ^ result)) >> 31);
    break;
}
case Opcode::Add64: {
    const u64 a = ARG(0).lo, b = ARG(1).lo;
    const u64 carry_in = ARG(2).lo & 1;
    const u64 result = a + b + carry_in;
    DST.lo = result;
    DST.carry = u8(result < a || (carry_in && result == a));
    DST.overflow = u8((~(a ^ b) & (a ^ result)) >> 63);
    break;
}
case Opcode::Sub32: {
    const u32 a = u32(ARG(0).lo), b = ~u32(ARG(1).lo);
    const u64 wide = u64(a) + u64(b) + (ARG(2).lo & 1);
    const u32 result = u32(wide);
    DST.lo = result;
    DST.carry = u8(wide >> 32);
    DST.overflow = u8((~(a ^ b) & (a ^ result)) >> 31);
    break;
}
case Opcode::Sub64: {
    const u64 a = ARG(0).lo, b = ~ARG(1).lo;
    const u64 carry_in = ARG(2).lo & 1;
    const u64 result = a + b + carry_in;
    DST.lo = result;
    DST.carry = u8(result < a || (carry_in && result == a));
    DST.overflow = u8((~(a ^ b) & (a ^ result)) >> 63);
    break;
}
case Opcode::Mul32:
    DST.lo = u32(u32(ARG(0).lo) * u32(ARG(1).lo));
    break;
case Opcode::Mul64:
    DST.lo = ARG(0).lo * ARG(1).lo;
    break;
case Opcode::SignedMultiplyHigh64:
    DST.lo = u64((__int128(s64(ARG(0).lo)) * __int128(s64(ARG(1).lo))) >> 64);
    break;
case Opcode::UnsignedMultiplyHigh64:
    DST.lo = u64((unsigned __int128)(ARG(0).lo) * (unsigned __int128)(ARG(1).lo) >> 64);
    break;
case Opcode::UnsignedDiv32: {
    const u32 d = u32(ARG(1).lo);
    DST.lo = d == 0 ? 0 : u32(ARG(0).lo) / d;
    break;
}
case Opcode::UnsignedDiv64: {
    const u64 d = ARG(1).lo;
    DST.lo = d == 0 ? 0 : ARG(0).lo / d;
    break;
}
case Opcode::SignedDiv32: {
    const s32 n = s32(u32(ARG(0).lo)), d = s32(u32(ARG(1).lo));
    if (d == 0) {
        DST.lo = 0;
    } else if (d == -1) {
        DST.lo = u32(0u - u32(n));
    } else {
        DST.lo = u32(n / d);
    }
    break;
}
case Opcode::SignedDiv64: {
    const s64 n = s64(ARG(0).lo), d = s64(ARG(1).lo);
    if (d == 0) {
        DST.lo = 0;
    } else if (d == -1) {
        DST.lo = 0ull - u64(n);
    } else {
        DST.lo = u64(n / d);
    }
    break;
}
case Opcode::And32:
    DST.lo = u32(ARG(0).lo & ARG(1).lo);
    DST.carry = DST.overflow = 0;
    break;
case Opcode::And64:
    DST.lo = ARG(0).lo & ARG(1).lo;
    DST.carry = DST.overflow = 0;
    break;
case Opcode::AndNot32:
    DST.lo = u32(ARG(0).lo & ~ARG(1).lo);
    DST.carry = DST.overflow = 0;
    break;
case Opcode::AndNot64:
    DST.lo = ARG(0).lo & ~ARG(1).lo;
    DST.carry = DST.overflow = 0;
    break;
case Opcode::Eor32:
    DST.lo = u32(ARG(0).lo ^ ARG(1).lo);
    DST.carry = DST.overflow = 0;
    break;
case Opcode::Eor64:
    DST.lo = ARG(0).lo ^ ARG(1).lo;
    DST.carry = DST.overflow = 0;
    break;
case Opcode::Or32:
    DST.lo = u32(ARG(0).lo | ARG(1).lo);
    DST.carry = DST.overflow = 0;
    break;
case Opcode::Or64:
    DST.lo = ARG(0).lo | ARG(1).lo;
    DST.carry = DST.overflow = 0;
    break;
case Opcode::Not32:
    DST.lo = u32(~ARG(0).lo);
    DST.carry = DST.overflow = 0;
    break;
case Opcode::Not64:
    DST.lo = ~ARG(0).lo;
    DST.carry = DST.overflow = 0;
    break;
case Opcode::SignExtendByteToWord:
    DST.lo = u32(s32(s8(ARG(0).lo)));
    break;
case Opcode::SignExtendHalfToWord:
    DST.lo = u32(s32(s16(ARG(0).lo)));
    break;
case Opcode::SignExtendByteToLong:
    DST.lo = u64(s64(s8(ARG(0).lo)));
    break;
case Opcode::SignExtendHalfToLong:
    DST.lo = u64(s64(s16(ARG(0).lo)));
    break;
case Opcode::SignExtendWordToLong:
    DST.lo = u64(s64(s32(ARG(0).lo)));
    break;
case Opcode::ZeroExtendByteToWord:
case Opcode::ZeroExtendByteToLong:
    DST.lo = u8(ARG(0).lo);
    break;
case Opcode::ZeroExtendHalfToWord:
case Opcode::ZeroExtendHalfToLong:
    DST.lo = u16(ARG(0).lo);
    break;
case Opcode::ZeroExtendWordToLong:
    DST.lo = u32(ARG(0).lo);
    break;
case Opcode::ZeroExtendLongToQuad:
    DST.lo = ARG(0).lo;
    DST.hi = 0;
    break;
case Opcode::ByteReverseWord:
    DST.lo = __builtin_bswap32(u32(ARG(0).lo));
    break;
case Opcode::ByteReverseHalf:
    DST.lo = __builtin_bswap16(u16(ARG(0).lo));
    break;
case Opcode::ByteReverseDual:
    DST.lo = __builtin_bswap64(ARG(0).lo);
    break;
case Opcode::CountLeadingZeros32:
    DST.lo = std::countl_zero(u32(ARG(0).lo));
    break;
case Opcode::CountLeadingZeros64:
    DST.lo = std::countl_zero(ARG(0).lo);
    break;
case Opcode::ExtractRegister32: {
    const u32 a = u32(ARG(0).lo), b = u32(ARG(1).lo);
    const u32 lsb = u32(ARG(2).lo) & 31;
    DST.lo = lsb == 0 ? a : u32((a >> lsb) | (b << (32 - lsb)));
    break;
}
case Opcode::ExtractRegister64: {
    const u64 a = ARG(0).lo, b = ARG(1).lo;
    const u32 lsb = u32(ARG(2).lo) & 63;
    DST.lo = lsb == 0 ? a : (a >> lsb) | (b << (64 - lsb));
    break;
}
case Opcode::ReplicateBit32:
    DST.lo = ((ARG(0).lo >> ARG(1).lo) & 1) ? 0xFFFFFFFFull : 0;
    break;
case Opcode::ReplicateBit64:
    DST.lo = ((ARG(0).lo >> ARG(1).lo) & 1) ? ~0ull : 0;
    break;
case Opcode::MaxSigned32:
    DST.lo = u32(std::max(s32(ARG(0).lo), s32(ARG(1).lo)));
    break;
case Opcode::MaxSigned64:
    DST.lo = u64(std::max(s64(ARG(0).lo), s64(ARG(1).lo)));
    break;
case Opcode::MaxUnsigned32:
    DST.lo = std::max(u32(ARG(0).lo), u32(ARG(1).lo));
    break;
case Opcode::MaxUnsigned64:
    DST.lo = std::max(ARG(0).lo, ARG(1).lo);
    break;
case Opcode::MinSigned32:
    DST.lo = u32(std::min(s32(ARG(0).lo), s32(ARG(1).lo)));
    break;
case Opcode::MinSigned64:
    DST.lo = u64(std::min(s64(ARG(0).lo), s64(ARG(1).lo)));
    break;
case Opcode::MinUnsigned32:
    DST.lo = std::min(u32(ARG(0).lo), u32(ARG(1).lo));
    break;
case Opcode::MinUnsigned64:
    DST.lo = std::min(ARG(0).lo, ARG(1).lo);
    break;

case Opcode::SignedSaturatedAddWithFlag32: {
    bool sat = false;
    DST.lo = u32(SatAddS<s32>(s32(ARG(0).lo), s32(ARG(1).lo), sat));
    DST.overflow = sat;
    break;
}
case Opcode::SignedSaturatedSubWithFlag32: {
    bool sat = false;
    DST.lo = u32(SatSubS<s32>(s32(ARG(0).lo), s32(ARG(1).lo), sat));
    DST.overflow = sat;
    break;
}
case Opcode::SignedSaturation: {
    const s32 value = s32(u32(ARG(0).lo));
    const u32 n = u32(ARG(1).lo);
    if (n >= 32) {
        DST.lo = u32(value);
        DST.overflow = 0;
        break;
    }
    const s32 max = s32((1u << (n - 1)) - 1);
    const s32 min = -max - 1;
    const s32 result = std::clamp(value, min, max);
    DST.lo = u32(result);
    DST.overflow = result != value;
    break;
}
case Opcode::UnsignedSaturation: {
    const s32 value = s32(u32(ARG(0).lo));
    const u32 n = u32(ARG(1).lo);
    const s64 max = n >= 32 ? s64(0xFFFFFFFF) : s64((1u << n) - 1);
    const s64 result = std::clamp<s64>(value, 0, max);
    DST.lo = u32(result);
    DST.overflow = result != value;
    break;
}
#define DYNARMIC_INTERP_SAT_SCALAR(name, fn, type, utype)                         \
    case Opcode::name: {                                                          \
        bool sat = false;                                                         \
        DST.lo = utype(fn<type>(type(utype(ARG(0).lo)), type(utype(ARG(1).lo)), sat)); \
        SetQC(sat);                                                               \
        break;                                                                    \
    }
DYNARMIC_INTERP_SAT_SCALAR(SignedSaturatedAdd8, SatAddS, s8, u8)
DYNARMIC_INTERP_SAT_SCALAR(SignedSaturatedAdd16, SatAddS, s16, u16)
DYNARMIC_INTERP_SAT_SCALAR(SignedSaturatedAdd32, SatAddS, s32, u32)
DYNARMIC_INTERP_SAT_SCALAR(SignedSaturatedAdd64, SatAddS, s64, u64)
DYNARMIC_INTERP_SAT_SCALAR(SignedSaturatedSub8, SatSubS, s8, u8)
DYNARMIC_INTERP_SAT_SCALAR(SignedSaturatedSub16, SatSubS, s16, u16)
DYNARMIC_INTERP_SAT_SCALAR(SignedSaturatedSub32, SatSubS, s32, u32)
DYNARMIC_INTERP_SAT_SCALAR(SignedSaturatedSub64, SatSubS, s64, u64)
DYNARMIC_INTERP_SAT_SCALAR(UnsignedSaturatedAdd8, SatAddU, u8, u8)
DYNARMIC_INTERP_SAT_SCALAR(UnsignedSaturatedAdd16, SatAddU, u16, u16)
DYNARMIC_INTERP_SAT_SCALAR(UnsignedSaturatedAdd32, SatAddU, u32, u32)
DYNARMIC_INTERP_SAT_SCALAR(UnsignedSaturatedAdd64, SatAddU, u64, u64)
DYNARMIC_INTERP_SAT_SCALAR(UnsignedSaturatedSub8, SatSubU, u8, u8)
DYNARMIC_INTERP_SAT_SCALAR(UnsignedSaturatedSub16, SatSubU, u16, u16)
DYNARMIC_INTERP_SAT_SCALAR(UnsignedSaturatedSub32, SatSubU, u32, u32)
DYNARMIC_INTERP_SAT_SCALAR(UnsignedSaturatedSub64, SatSubU, u64, u64)
#undef DYNARMIC_INTERP_SAT_SCALAR
case Opcode::SignedSaturatedDoublingMultiplyReturnHigh16: {
    bool sat = false;
    DST.lo = u16(SatDoublingMulHigh<s16>(s16(ARG(0).lo), s16(ARG(1).lo), false, sat));
    SetQC(sat);
    break;
}
case Opcode::SignedSaturatedDoublingMultiplyReturnHigh32: {
    bool sat = false;
    DST.lo = u32(SatDoublingMulHigh<s32>(s32(ARG(0).lo), s32(ARG(1).lo), false, sat));
    SetQC(sat);
    break;
}

case Opcode::CRC32Castagnoli8:
    DST.lo = Common::Crypto::CRC32::ComputeCRC32Castagnoli(u32(ARG(0).lo), ARG(1).lo, 1);
    break;
case Opcode::CRC32Castagnoli16:
    DST.lo = Common::Crypto::CRC32::ComputeCRC32Castagnoli(u32(ARG(0).lo), ARG(1).lo, 2);
    break;
case Opcode::CRC32Castagnoli32:
    DST.lo = Common::Crypto::CRC32::ComputeCRC32Castagnoli(u32(ARG(0).lo), ARG(1).lo, 4);
    break;
case Opcode::CRC32Castagnoli64:
    DST.lo = Common::Crypto::CRC32::ComputeCRC32Castagnoli(u32(ARG(0).lo), ARG(1).lo, 8);
    break;
case Opcode::CRC32ISO8:
    DST.lo = Common::Crypto::CRC32::ComputeCRC32ISO(u32(ARG(0).lo), ARG(1).lo, 1);
    break;
case Opcode::CRC32ISO16:
    DST.lo = Common::Crypto::CRC32::ComputeCRC32ISO(u32(ARG(0).lo), ARG(1).lo, 2);
    break;
case Opcode::CRC32ISO32:
    DST.lo = Common::Crypto::CRC32::ComputeCRC32ISO(u32(ARG(0).lo), ARG(1).lo, 4);
    break;
case Opcode::CRC32ISO64:
    DST.lo = Common::Crypto::CRC32::ComputeCRC32ISO(u32(ARG(0).lo), ARG(1).lo, 8);
    break;
case Opcode::AESDecryptSingleRound:
    AesOp(DST, ARG(0), Common::Crypto::AES::DecryptSingleRound);
    break;
case Opcode::AESEncryptSingleRound:
    AesOp(DST, ARG(0), Common::Crypto::AES::EncryptSingleRound);
    break;
case Opcode::AESInverseMixColumns:
    AesOp(DST, ARG(0), Common::Crypto::AES::InverseMixColumns);
    break;
case Opcode::AESMixColumns:
    AesOp(DST, ARG(0), Common::Crypto::AES::MixColumns);
    break;
case Opcode::SM4AccessSubstitutionBox:
    DST.lo = kSM4SBox[u8(ARG(0).lo)];
    break;
case Opcode::ZeroVector:
    DST.lo = 0;
    DST.hi = 0;
    break;
