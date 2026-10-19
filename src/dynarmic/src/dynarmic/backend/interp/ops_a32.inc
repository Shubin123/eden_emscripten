// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// AArch32 state, coprocessor and memory operations.
// Included inside the A32 interpreter dispatch switch.

case Opcode::A32SetCheckBit:
    state.check_bit = ARG(0).lo & 1;
    break;
case Opcode::A32GetRegister:
    DST.lo = state.reg[A32::RegNumber(A32::Reg(ARG(0).lo))];
    break;
case Opcode::A32GetExtendedRegister32:
    DST.lo = state.ext[size_t(ARG(0).lo) - size_t(A32::ExtReg::S0)];
    break;
case Opcode::A32GetExtendedRegister64: {
    const size_t index = (size_t(ARG(0).lo) - size_t(A32::ExtReg::D0)) * 2;
    DST.lo = u64(state.ext[index]) | (u64(state.ext[index + 1]) << 32);
    break;
}
case Opcode::A32GetVector: {
    const A32::ExtReg reg = A32::ExtReg(ARG(0).lo);
    if (A32::IsDoubleExtReg(reg)) {
        const size_t index = (size_t(reg) - size_t(A32::ExtReg::D0)) * 2;
        DST.lo = u64(state.ext[index]) | (u64(state.ext[index + 1]) << 32);
        DST.hi = 0;
    } else {
        const size_t index = (size_t(reg) - size_t(A32::ExtReg::Q0)) * 4;
        std::memcpy(&DST.lo, &state.ext[index], 16);
    }
    break;
}
case Opcode::A32SetRegister:
    state.reg[A32::RegNumber(A32::Reg(ARG(0).lo))] = u32(ARG(1).lo);
    break;
case Opcode::A32SetExtendedRegister32:
    state.ext[size_t(ARG(0).lo) - size_t(A32::ExtReg::S0)] = u32(ARG(1).lo);
    break;
case Opcode::A32SetExtendedRegister64: {
    const size_t index = (size_t(ARG(0).lo) - size_t(A32::ExtReg::D0)) * 2;
    state.ext[index] = u32(ARG(1).lo);
    state.ext[index + 1] = u32(ARG(1).lo >> 32);
    break;
}
case Opcode::A32SetVector: {
    const A32::ExtReg reg = A32::ExtReg(ARG(0).lo);
    if (A32::IsDoubleExtReg(reg)) {
        const size_t index = (size_t(reg) - size_t(A32::ExtReg::D0)) * 2;
        state.ext[index] = u32(ARG(1).lo);
        state.ext[index + 1] = u32(ARG(1).lo >> 32);
    } else {
        const size_t index = (size_t(reg) - size_t(A32::ExtReg::Q0)) * 4;
        std::memcpy(&state.ext[index], &ARG(1).lo, 16);
    }
    break;
}
case Opcode::A32GetCpsr:
    DST.lo = state.Cpsr();
    break;
case Opcode::A32SetCpsr:
    state.SetCpsr(u32(ARG(0).lo));
    break;
case Opcode::A32SetCpsrNZCV:
case Opcode::A32SetCpsrNZCVRaw:
    state.nzcv = u32(ARG(0).lo) & 0xF0000000;
    break;
case Opcode::A32SetCpsrNZCVQ:
    state.nzcv = u32(ARG(0).lo) & 0xF0000000;
    state.q = (ARG(0).lo >> 27) & 1;
    break;
case Opcode::A32SetCpsrNZ:
    state.nzcv = (state.nzcv & 0x30000000) | (u32(ARG(0).lo) & 0xC0000000);
    break;
case Opcode::A32SetCpsrNZC:
    state.nzcv = (state.nzcv & 0x10000000) | (u32(ARG(0).lo) & 0xC0000000) | (u32(ARG(1).lo & 1) << 29);
    break;
case Opcode::A32GetCFlag:
    DST.lo = (state.nzcv >> 29) & 1;
    break;
case Opcode::A32OrQFlag:
    state.q |= ARG(0).lo & 1;
    break;
case Opcode::A32GetGEFlags:
    DST.lo = state.ge;
    break;
case Opcode::A32SetGEFlags:
    state.ge = u32(ARG(0).lo);
    break;
case Opcode::A32SetGEFlagsCompressed:
    state.ge = A32State::ExpandGE(u32(ARG(0).lo) >> 16);
    break;
case Opcode::A32BXWritePC: {
    const u32 new_pc = u32(ARG(0).lo);
    const u32 upper_without_t = u32(A32::LocationDescriptor{block.end_location}.SetSingleStepping(false).UniqueHash() >> 32) & 0xFFFFFFFE;
    if (new_pc & 1) {
        state.reg[15] = new_pc & 0xFFFFFFFE;
        state.upper = upper_without_t | 1;
    } else {
        state.reg[15] = new_pc & 0xFFFFFFFC;
        state.upper = upper_without_t;
    }
    wrote_pc = true;
    break;
}
case Opcode::A32UpdateUpperLocationDescriptor:
    if (!wrote_pc) {
        state.upper = u32(A32::LocationDescriptor{block.end_location}.SetSingleStepping(false).UniqueHash() >> 32);
    }
    break;
case Opcode::A32CallSupervisor:
    UpdateTicks();
    conf.callbacks->CallSVC(u32(ARG(0).lo));
    ResetTicks();
    break;
case Opcode::A32ExceptionRaised:
    UpdateTicks();
    conf.callbacks->ExceptionRaised(u32(ARG(0).lo), A32::Exception(ARG(1).lo));
    ResetTicks();
    break;
case Opcode::A32DataSynchronizationBarrier:
case Opcode::A32DataMemoryBarrier:
    std::atomic_thread_fence(std::memory_order_seq_cst);
    break;
case Opcode::A32InstructionSynchronizationBarrier:
    if (conf.hook_isb) {
        conf.callbacks->InstructionSynchronizationBarrierRaised();
    }
    break;
case Opcode::A32GetFpscr:
    DST.lo = state.Fpscr();
    break;
case Opcode::A32SetFpscr:
    state.SetFpscr(u32(ARG(0).lo));
    break;
case Opcode::A32GetFpscrNZCV:
    DST.lo = state.fpsr_nzcv;
    break;
case Opcode::A32SetFpscrNZCV:
    state.fpsr_nzcv = u32(ARG(0).lo) & 0xF0000000;
    break;
case Opcode::A32ClearExclusive:
    state.exclusive_state = 0;
    break;

case Opcode::A32ReadMemory8:
    DST.lo = Read<u8>(u32(ARG(1).lo));
    break;
case Opcode::A32ReadMemory16:
    DST.lo = Read<u16>(u32(ARG(1).lo));
    break;
case Opcode::A32ReadMemory32:
    DST.lo = Read<u32>(u32(ARG(1).lo));
    break;
case Opcode::A32ReadMemory64:
    DST.lo = Read<u64>(u32(ARG(1).lo));
    break;
case Opcode::A32ExclusiveReadMemory8:
    DST.lo = ExclusiveRead<u8>(u32(ARG(1).lo));
    break;
case Opcode::A32ExclusiveReadMemory16:
    DST.lo = ExclusiveRead<u16>(u32(ARG(1).lo));
    break;
case Opcode::A32ExclusiveReadMemory32:
    DST.lo = ExclusiveRead<u32>(u32(ARG(1).lo));
    break;
case Opcode::A32ExclusiveReadMemory64:
    DST.lo = ExclusiveRead<u64>(u32(ARG(1).lo));
    break;
case Opcode::A32WriteMemory8:
    Write<u8>(u32(ARG(1).lo), u8(ARG(2).lo));
    break;
case Opcode::A32WriteMemory16:
    Write<u16>(u32(ARG(1).lo), u16(ARG(2).lo));
    break;
case Opcode::A32WriteMemory32:
    Write<u32>(u32(ARG(1).lo), u32(ARG(2).lo));
    break;
case Opcode::A32WriteMemory64:
    Write<u64>(u32(ARG(1).lo), ARG(2).lo);
    break;
case Opcode::A32ExclusiveWriteMemory8:
    DST.lo = ExclusiveWrite<u8>(u32(ARG(1).lo), u8(ARG(2).lo));
    break;
case Opcode::A32ExclusiveWriteMemory16:
    DST.lo = ExclusiveWrite<u16>(u32(ARG(1).lo), u16(ARG(2).lo));
    break;
case Opcode::A32ExclusiveWriteMemory32:
    DST.lo = ExclusiveWrite<u32>(u32(ARG(1).lo), u32(ARG(2).lo));
    break;
case Opcode::A32ExclusiveWriteMemory64:
    DST.lo = ExclusiveWrite<u64>(u32(ARG(1).lo), ARG(2).lo);
    break;

case Opcode::A32CoprocInternalOperation:
case Opcode::A32CoprocSendOneWord:
case Opcode::A32CoprocSendTwoWords:
case Opcode::A32CoprocGetOneWord:
case Opcode::A32CoprocGetTwoWords:
case Opcode::A32CoprocLoadWords:
case Opcode::A32CoprocStoreWords:
    Coprocessor(op, s);
    break;
