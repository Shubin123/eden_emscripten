// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Pointer authentication (Armv8.3 PAuth) without keys: every pointer authentication code is zero,
// which is how the hint-space forms (PACIASP, AUTIASP, XPACLRI and so on, executed as NOPs) already
// behave. Signing, authenticating and stripping leave a pointer unchanged, so the authenticated
// branches and loads are their plain counterparts.

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

namespace {

/// The forms with a zero modifier (Z) encode Rn as 31.
bool PointerUnchanged(TranslatorVisitor& v, bool Z, Reg Rn) {
    if (Z && Rn != Reg::R31) {
        return v.UnallocatedEncoding();
    }
    return true;
}

}  // Anonymous namespace

bool TranslatorVisitor::PACIA_1(bool Z, Reg Rn, Reg) {
    return PointerUnchanged(*this, Z, Rn);
}

bool TranslatorVisitor::PACIB_1(bool Z, Reg Rn, Reg) {
    return PointerUnchanged(*this, Z, Rn);
}

bool TranslatorVisitor::PACDA(bool Z, Reg Rn, Reg) {
    return PointerUnchanged(*this, Z, Rn);
}

bool TranslatorVisitor::PACDB(bool Z, Reg Rn, Reg) {
    return PointerUnchanged(*this, Z, Rn);
}

bool TranslatorVisitor::AUTIA_1(bool Z, Reg Rn, Reg) {
    return PointerUnchanged(*this, Z, Rn);
}

bool TranslatorVisitor::AUTIB_1(bool Z, Reg Rn, Reg) {
    return PointerUnchanged(*this, Z, Rn);
}

bool TranslatorVisitor::AUTDA(bool Z, Reg Rn, Reg) {
    return PointerUnchanged(*this, Z, Rn);
}

bool TranslatorVisitor::AUTDB(bool Z, Reg Rn, Reg) {
    return PointerUnchanged(*this, Z, Rn);
}

bool TranslatorVisitor::XPAC_1(bool, Reg) {
    return true;
}

bool TranslatorVisitor::PACGA(Reg, Reg, Reg Rd) {
    X(64, Rd, ir.Imm64(0));
    return true;
}

bool TranslatorVisitor::BRA(bool Z, bool, Reg Rn, Reg Rm) {
    if (!Z && Rm != Reg::R31) {
        return UnallocatedEncoding();
    }
    return BR(Rn);
}

bool TranslatorVisitor::BLRA(bool Z, bool, Reg Rn, Reg Rm) {
    if (!Z && Rm != Reg::R31) {
        return UnallocatedEncoding();
    }
    return BLR(Rn);
}

bool TranslatorVisitor::RETA(bool) {
    return RET(Reg::R30);
}

bool TranslatorVisitor::LDRA(bool, bool S, Imm<9> imm9, bool W, Reg Rn, Reg Rt) {
    if (W && Rn == Rt && Rn != Reg::R31) {
        return UnpredictableInstruction();
    }
    const u64 offset = concatenate(Imm<1>{S}, imm9, Imm<3>{0}).SignExtend<u64>();
    const IR::U64 base = Rn == Reg::SP ? IR::U64(SP(64)) : IR::U64(X(64, Rn));
    const IR::U64 address = ir.Add(base, ir.Imm64(offset));
    X(64, Rt, Mem(address, 8, IR::AccType::NORMAL));
    if (W) {
        if (Rn == Reg::SP) {
            SP(64, address);
        } else {
            X(64, Rn, address);
        }
    }
    return true;
}

}  // namespace Dynarmic::A64
