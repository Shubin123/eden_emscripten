// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// A64 state, system register and memory operations.
// Included inside the A64 interpreter dispatch switch.

case Opcode::A64SetCheckBit:
    state.check_bit = ARG(0).lo & 1;
    break;
case Opcode::A64GetCFlag:
    DST.lo = (state.nzcv >> 29) & 1;
    break;
case Opcode::A64GetNZCVRaw:
    DST.lo = state.nzcv;
    break;
case Opcode::A64SetNZCVRaw:
case Opcode::A64SetNZCV:
    state.nzcv = u32(ARG(0).lo) & 0xF0000000;
    break;
case Opcode::A64GetW:
    DST.lo = u32(state.reg[ARG(0).lo]);
    break;
case Opcode::A64GetX:
    DST.lo = state.reg[ARG(0).lo];
    break;
case Opcode::A64GetS:
    DST.lo = u32(state.vec[ARG(0).lo * 2]);
    DST.hi = 0;
    break;
case Opcode::A64GetD:
    DST.lo = state.vec[ARG(0).lo * 2];
    DST.hi = 0;
    break;
case Opcode::A64GetQ:
    DST.lo = state.vec[ARG(0).lo * 2];
    DST.hi = state.vec[ARG(0).lo * 2 + 1];
    break;
case Opcode::A64GetSP:
    DST.lo = state.sp;
    break;
case Opcode::A64GetFPCR:
    DST.lo = state.fpcr;
    break;
case Opcode::A64GetFPSR:
    DST.lo = state.fpsr;
    break;
case Opcode::A64SetW:
    state.reg[ARG(0).lo] = u32(ARG(1).lo);
    break;
case Opcode::A64SetX:
    state.reg[ARG(0).lo] = ARG(1).lo;
    break;
case Opcode::A64SetS:
    state.vec[ARG(0).lo * 2] = u32(ARG(1).lo);
    state.vec[ARG(0).lo * 2 + 1] = 0;
    break;
case Opcode::A64SetD:
    state.vec[ARG(0).lo * 2] = ARG(1).lo;
    state.vec[ARG(0).lo * 2 + 1] = 0;
    break;
case Opcode::A64SetQ:
    state.vec[ARG(0).lo * 2] = ARG(1).lo;
    state.vec[ARG(0).lo * 2 + 1] = ARG(1).hi;
    break;
case Opcode::A64SetSP:
    state.sp = ARG(0).lo;
    break;
case Opcode::A64SetFPCR:
    state.fpcr = u32(ARG(0).lo) & A64State::FPCR_MASK;
    break;
case Opcode::A64SetFPSR:
    state.fpsr = u32(ARG(0).lo) & A64State::FPSR_MASK;
    break;
case Opcode::A64SetPC:
    state.pc = ARG(0).lo;
    break;
case Opcode::A64CallSupervisor:
    conf.callbacks->CallSVC(u32(ARG(0).lo));
    state.exclusive_state = 0;
    break;
case Opcode::A64ExceptionRaised:
    conf.callbacks->ExceptionRaised(ARG(0).lo, A64::Exception(ARG(1).lo));
    break;
case Opcode::A64DataCacheOperationRaised:
    conf.callbacks->DataCacheOperationRaised(A64::DataCacheOperation(ARG(1).lo), ARG(2).lo);
    break;
case Opcode::A64InstructionCacheOperationRaised:
    conf.callbacks->InstructionCacheOperationRaised(A64::InstructionCacheOperation(ARG(0).lo), ARG(1).lo);
    break;
case Opcode::A64DataSynchronizationBarrier:
case Opcode::A64DataMemoryBarrier:
    std::atomic_thread_fence(std::memory_order_seq_cst);
    break;
case Opcode::A64InstructionSynchronizationBarrier:
    if (conf.hook_isb) {
        conf.callbacks->InstructionSynchronizationBarrierRaised();
        CheckHaltAfterCallback();
    }
    break;
case Opcode::A64GetCNTFRQ:
    DST.lo = conf.cntfrq_el0;
    break;
case Opcode::A64GetCNTPCT:
    if (!conf.wall_clock_cntpct) {
        UpdateTicks();
    }
    DST.lo = conf.callbacks->GetCNTPCT();
    break;
case Opcode::A64GetCTR:
    DST.lo = conf.ctr_el0;
    break;
case Opcode::A64GetDCZID:
    DST.lo = conf.dczid_el0;
    break;
case Opcode::A64GetTPIDR:
    DST.lo = conf.tpidr_el0 ? *conf.tpidr_el0 : 0;
    break;
case Opcode::A64GetTPIDRRO:
    DST.lo = conf.tpidrro_el0 ? *conf.tpidrro_el0 : 0;
    break;
case Opcode::A64SetTPIDR:
    if (conf.tpidr_el0) {
        *conf.tpidr_el0 = ARG(0).lo;
    }
    break;
case Opcode::A64ClearExclusive:
    state.exclusive_state = 0;
    break;

case Opcode::A64ReadMemory8:
    DST.lo = Read<u8>(ARG(1).lo);
    break;
case Opcode::A64ReadMemory16:
    DST.lo = Read<u16>(ARG(1).lo);
    break;
case Opcode::A64ReadMemory32:
    DST.lo = Read<u32>(ARG(1).lo);
    break;
case Opcode::A64ReadMemory64:
    DST.lo = Read<u64>(ARG(1).lo);
    break;
case Opcode::A64ReadMemory128: {
    const Vector value = Read128(ARG(1).lo);
    DST.lo = value[0];
    DST.hi = value[1];
    break;
}
case Opcode::A64ExclusiveReadMemory8:
    DST.lo = ExclusiveRead<u8>(ARG(1).lo);
    break;
case Opcode::A64ExclusiveReadMemory16:
    DST.lo = ExclusiveRead<u16>(ARG(1).lo);
    break;
case Opcode::A64ExclusiveReadMemory32:
    DST.lo = ExclusiveRead<u32>(ARG(1).lo);
    break;
case Opcode::A64ExclusiveReadMemory64:
    DST.lo = ExclusiveRead<u64>(ARG(1).lo);
    break;
case Opcode::A64ExclusiveReadMemory128: {
    const Vector value = ExclusiveRead<Vector>(ARG(1).lo);
    DST.lo = value[0];
    DST.hi = value[1];
    break;
}
case Opcode::A64WriteMemory8:
    Write<u8>(ARG(1).lo, u8(ARG(2).lo));
    break;
case Opcode::A64WriteMemory16:
    Write<u16>(ARG(1).lo, u16(ARG(2).lo));
    break;
case Opcode::A64WriteMemory32:
    Write<u32>(ARG(1).lo, u32(ARG(2).lo));
    break;
case Opcode::A64WriteMemory64:
    Write<u64>(ARG(1).lo, ARG(2).lo);
    break;
case Opcode::A64WriteMemory128:
    Write128(ARG(1).lo, Vector{ARG(2).lo, ARG(2).hi});
    break;
case Opcode::A64ExclusiveWriteMemory8:
    DST.lo = ExclusiveWrite<u8>(ARG(1).lo, u8(ARG(2).lo));
    break;
case Opcode::A64ExclusiveWriteMemory16:
    DST.lo = ExclusiveWrite<u16>(ARG(1).lo, u16(ARG(2).lo));
    break;
case Opcode::A64ExclusiveWriteMemory32:
    DST.lo = ExclusiveWrite<u32>(ARG(1).lo, u32(ARG(2).lo));
    break;
case Opcode::A64ExclusiveWriteMemory64:
    DST.lo = ExclusiveWrite<u64>(ARG(1).lo, ARG(2).lo);
    break;
case Opcode::A64ExclusiveWriteMemory128:
    DST.lo = ExclusiveWrite<Vector>(ARG(1).lo, Vector{ARG(2).lo, ARG(2).hi});
    break;
